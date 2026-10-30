
from pathlib import Path
import argparse
import json
import re

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

from ingestion import load_pdf_documents


# ---------------- CONFIGURATION ----------------

BASE_DIR = Path(__file__).resolve().parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
INDEX_PATH = VECTORSTORE_DIR / "index.faiss"
CHUNKS_PATH = VECTORSTORE_DIR / "chunks.json"

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

TOP_K = 4
RRF_CONSTANT = 60
CANDIDATE_MULTIPLIER = 4
DUPLICATE_SIMILARITY_THRESHOLD = 0.88

print(f"Loading embedding model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)


# ---------------- TEXT PROCESSING ----------------

def tokenize(text):
    """Unicode-aware tokenization for multilingual text."""
    if not isinstance(text, str):
        return []

    return re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)


def normalize_text(text):
    if not isinstance(text, str):
        return ""
    return " ".join(tokenize(text))


def is_devanagari(text):
    return bool(re.search(r"[\u0900-\u097F]", text or ""))


def query_variants(query):
    """
    Preserve the original query and add safe, common Hinglish variants.
    Does not apply ITRANS to ordinary Roman Hindi.
    """
    original = query.strip()
    if not original:
        return []

    variants = [original]

    # Common Roman Hindi -> Devanagari word mappings.
    word_map = {
        "yojana": "योजना",
        "ka": "का",
        "ki": "की",
        "ke": "के",
        "kya": "क्या",
        "hai": "है",
        "hain": "हैं",
        "objective": "उद्देश्य",
        "uddeshya": "उद्देश्य",
        "lakshya": "लक्ष्य",
        "pani": "पानी",
        "fasal": "फसल",
        "kisan": "किसान",
        "labh": "लाभ",
        "kitna": "कितना",
        "kab": "कब",
        "kaise": "कैसे",
    }

    words = original.split()
    converted_words = [
        word_map.get(word.strip(".,?!:;").casefold(), word)
        for word in words
    ]
    hindi_variant = " ".join(converted_words)

    if hindi_variant.casefold() != original.casefold():
        variants.append(hindi_variant)

    return list(dict.fromkeys(variants))

def is_near_duplicate(candidate, selected):
    candidate_text = normalize_text(candidate.get("text", ""))

    if not candidate_text:
        return True

    candidate_tokens = set(candidate_text.split())

    for existing in selected:
        existing_text = normalize_text(existing.get("text", ""))

        if candidate_text == existing_text:
            return True

        existing_tokens = set(existing_text.split())

        if not candidate_tokens or not existing_tokens:
            continue

        intersection = len(candidate_tokens & existing_tokens)
        union = len(candidate_tokens | existing_tokens)
        similarity = intersection / union if union else 0.0

        if similarity >= DUPLICATE_SIMILARITY_THRESHOLD:
            return True

    return False


# ---------------- BUILD INDEX ----------------

def build_index():
    chunks = load_pdf_documents()

    if not chunks:
        print("No document chunks found. Add PDFs to documents first.")
        return

    valid_chunks = [
        chunk for chunk in chunks
        if isinstance(chunk, dict)
        and isinstance(chunk.get("text"), str)
        and chunk["text"].strip()
    ]

    if not valid_chunks:
        print("No usable text chunks were extracted from the PDFs.")
        return

    texts = [chunk["text"] for chunk in valid_chunks]
    print(f"\nGenerating embeddings for {len(texts)} chunks...")

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2 or len(embeddings) != len(valid_chunks):
        raise ValueError("Embedding generation returned an invalid shape.")

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_PATH))

    with open(CHUNKS_PATH, "w", encoding="utf-8") as file:
        json.dump(valid_chunks, file, ensure_ascii=False, indent=2)

    print("\nIndex created successfully!")
    print(f"Indexed chunks: {index.ntotal}")
    print(f"FAISS index: {INDEX_PATH}")
    print(f"Chunk metadata: {CHUNKS_PATH}")


# ---------------- LOAD INDEX ----------------

def load_index():
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            "Index not found. Build it using: python retrieval.py --build"
        )

    index = faiss.read_index(str(INDEX_PATH))

    with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not isinstance(chunks, list):
        raise ValueError("Saved chunk metadata is not a list.")

    if index.ntotal != len(chunks):
        raise ValueError(
            "Index and chunk metadata do not match. "
            "Rebuild using: python retrieval.py --build"
        )

    if not chunks:
        raise ValueError("The saved index contains no chunks.")

    return index, chunks


# ---------------- HYBRID SEARCH ----------------

def search_documents(query, top_k=TOP_K):
    if not isinstance(query, str) or not query.strip():
        return []

    if not isinstance(top_k, int) or top_k < 1:
        return []

    index, chunks = load_index()
    variants = query_variants(query)
    total_chunks = len(chunks)

    candidate_k = min(
        total_chunks,
        max(top_k, top_k * CANDIDATE_MULTIPLIER),
    )

    # Semantic search: preserve the original query.
    query_vector = model.encode(
        [query.strip()],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    query_vector = np.asarray(query_vector, dtype=np.float32)

    semantic_scores, semantic_indices = index.search(
        query_vector, candidate_k
    )

    semantic_ranks = {}
    semantic_score_map = {}

    for rank, (score, raw_idx) in enumerate(
        zip(semantic_scores[0], semantic_indices[0]), start=1
    ):
        idx = int(raw_idx)
        if 0 <= idx < total_chunks:
            semantic_ranks[idx] = rank
            semantic_score_map[idx] = float(score)

    # BM25 search over the original and transliterated query tokens.
    tokenized_chunks = [
        tokenize(chunk.get("text", "")) for chunk in chunks
    ]

    if not any(tokenized_chunks):
        return []

    bm25 = BM25Okapi(tokenized_chunks)
    keyword_ranks = {}
    keyword_score_map = {}

    for variant in variants:
        tokens = tokenize(variant)
        if not tokens:
            continue

        scores = bm25.get_scores(tokens)
        order = np.argsort(scores)[::-1][:candidate_k]

        for rank, raw_idx in enumerate(order, start=1):
            idx = int(raw_idx)
            score = float(scores[idx])

            if score == 0:
                continue

            # Keep the best rank/score for each chunk across variants.
            if idx not in keyword_ranks or rank < keyword_ranks[idx]:
                keyword_ranks[idx] = rank
                keyword_score_map[idx] = score

    # Reciprocal Rank Fusion.
    candidate_ids = set(semantic_ranks) | set(keyword_ranks)
    results = []

    for idx in candidate_ids:
        fusion_score = 0.0

        if idx in semantic_ranks:
            fusion_score += 1.0 / (
                RRF_CONSTANT + semantic_ranks[idx]
            )

        if idx in keyword_ranks:
            fusion_score += 1.0 / (
                RRF_CONSTANT + keyword_ranks[idx]
            )

        result = chunks[idx].copy()
        result["semantic_score"] = semantic_score_map.get(idx)
        result["keyword_score"] = keyword_score_map.get(idx, 0.0)
        result["fusion_score"] = float(fusion_score)
        results.append(result)

    results.sort(
        key=lambda item: item["fusion_score"],
        reverse=True,
    )

    selected = []

    for result in results:
        if is_near_duplicate(result, selected):
            continue

        selected.append(result)

        if len(selected) >= top_k:
            break

    return selected


# ---------------- COMMAND LINE ----------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multilingual hybrid document retrieval"
    )
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--search", type=str)

    args = parser.parse_args()

    if args.build:
        build_index()

    elif args.search:
        try:
            results = search_documents(args.search)

            if not results:
                print("No matching passages found.")

            for rank, item in enumerate(results, start=1):
                print(f"\n{'=' * 50}")
                print(f"Rank: {rank}")
                print(f"Source: {item.get('source', 'Unknown document')}")
                print(f"Page: {item.get('page', 'Unknown')}")
                print(f"Chunk ID: {item.get('chunk_id', 'Unknown')}")
                print(f"Semantic score: {item.get('semantic_score')}")
                print(f"Keyword score: {item.get('keyword_score', 0):.3f}")
                print(f"Fusion score: {item.get('fusion_score', 0):.5f}")
                print(f"\nText:\n{item.get('text', '')}")

        except (FileNotFoundError, ValueError) as error:
            print(f"Search error: {error}")

    else:
        parser.print_help()
