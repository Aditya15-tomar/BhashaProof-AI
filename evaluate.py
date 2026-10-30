
import csv
import json
import re
from datetime import datetime
from pathlib import Path

from rag import answer_question
from retrieval import search_documents


# ---------------- CONFIGURATION ----------------

BASE_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = BASE_DIR / "evaluation"
DATASET_PATH = EVALUATION_DIR / "gold_questions.json"
REPORT_DIR = EVALUATION_DIR / "reports"

TOP_K = 4

# Phrases that indicate the chatbot cannot answer from the available evidence.
# Keep these specific to reduce false positives on answerable questions.
REFUSAL_PATTERNS = [
    r"couldn't find relevant information",
    r"do not provide enough information",
    r"does not provide enough information",
    r"cannot answer reliably",
    r"not enough information",
    r"unable to answer",
    r"not mentioned in the document",
    r"uploaded documents do not",
    r"do not contain any information",
    r"does not contain any information",
    r"don't contain any information",
    r"no information (?:is )?(?:provided|available|mentioned)",
    r"cannot be answered from",
    r"cannot determine",
    r"cannot be determined",
    r"cannot be reliably determined",
    r"cannot be found in",
    r"not stated in the document",
    r"not specified in the document",
    r"not available in the provided",
    r"does not mention",
    r"do not mention",
    r"no mention of",
    r"not provided in the document",
    r"cannot be answered from the provided documents",
    r"answer reliably",
    r"पर्याप्त जानकारी नहीं",
    r"दस्तावेज़ में .* नहीं",
    r"जानकारी नहीं मिली",
    r"उल्लेख नहीं है",
    r"उल्लेखित नहीं है",
    r"निर्धारित नहीं किया जा सकता",
    r"उत्तर देना संभव नहीं है",
    r"जानकारी उपलब्ध नहीं है",
    r"कोई जानकारी नहीं है",
    r"उपलब्ध दस्तावेज़ों में .* नहीं",
]


# ---------------- HELPERS ----------------

def normalize_document_name(name):
    """Compare document names without depending on directory paths."""
    if not name:
        return ""

    return Path(str(name)).name.casefold().strip()


def normalize_page(page):
    """Normalize page numbers for comparisons."""
    if page is None:
        return None

    return str(page).strip()


def source_pairs(sources):
    """Convert source metadata into comparable document/page pairs."""
    pairs = set()

    for source in sources or []:
        if not isinstance(source, dict):
            continue

        document = source.get("source") or source.get("document", "")
        page = source.get("page")

        if document and page is not None:
            pairs.add(
                (
                    normalize_document_name(document),
                    normalize_page(page),
                )
            )

    return pairs


def expected_pairs(expected_sources):
    """Read expected source document/page pairs from the dataset."""
    pairs = set()

    for item in expected_sources or []:
        document = normalize_document_name(item.get("document", ""))

        for page in item.get("pages", []):
            pairs.add((document, normalize_page(page)))

    return pairs


def is_abstention(answer):
    """
    Heuristically detect an explicit no-answer response.

    This checks wording, not whether the answer is factually supported.
    """
    text = (answer or "").casefold()

    return any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in REFUSAL_PATTERNS
    )


def extract_citations(answer):
    """
    Extract citations such as:
      [Source 1, Page 2]
      【Source 1, Page 2】
      [Source 1]
      【Source 1】

    Also accepts optional whitespace and case variations.
    """
    pattern = re.compile(
        r"[\[【]\s*Source\s+(\d+)"
        r"(?:\s*,?\s*Page\s+(\d+))?"
        r"\s*[\]】]",
        re.IGNORECASE,
    )

    citations = []

    for match in pattern.finditer(answer or ""):
        citations.append(
            {
                "source_number": int(match.group(1)),
                "page": match.group(2),
            }
        )

    return citations


def citation_validity(answer, sources):
    """
    Check whether citations point to retrieved sources and,
    when a page is included, whether the page matches metadata.

    This checks citation references, not factual entailment.
    """
    citations = extract_citations(answer)

    if not citations:
        return None, 0, 0

    valid_count = 0
    sources = sources or []

    for citation in citations:
        source_number = citation["source_number"]

        # Source numbering in the answer starts at 1.
        if source_number < 1 or source_number > len(sources):
            continue

        source = sources[source_number - 1]

        if not isinstance(source, dict):
            continue

        actual_page = source.get("page")
        cited_page = citation["page"]

        # A source-only citation is valid if its source number exists.
        if cited_page is None:
            valid_count += 1
        elif (
            actual_page is not None
            and normalize_page(actual_page) == normalize_page(cited_page)
        ):
            valid_count += 1

    return valid_count / len(citations), valid_count, len(citations)


def load_dataset():
    """Load and validate the evaluation dataset."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}\n"
            "Create evaluation/gold_questions.json first."
        )

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list) or not dataset:
        raise ValueError(
            "The dataset must be a non-empty JSON list of test questions."
        )

    seen_ids = set()

    for index, item in enumerate(dataset, start=1):
        for required in ("id", "question", "expected_answerable"):
            if required not in item:
                raise ValueError(
                    f"Dataset item {index} is missing '{required}'."
                )

        question_id = str(item["id"])

        if question_id in seen_ids:
            raise ValueError(f"Duplicate dataset ID: {question_id}")

        seen_ids.add(question_id)

        if not isinstance(item["expected_answerable"], bool):
            raise ValueError(
                f"{question_id}: expected_answerable must be true or false."
            )

        if not isinstance(item.get("expected_sources", []), list):
            raise ValueError(
                f"{question_id}: expected_sources must be a list."
            )

    return dataset


# ---------------- EVALUATION ----------------

def evaluate():
    dataset = load_dataset()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    retrieval_hits = 0
    retrieval_cases = 0

    abstention_correct = 0
    abstention_cases = 0

    total_citations = 0
    valid_citations = 0

    citation_cases = 0
    citation_case_hits = 0

    print(f"Evaluating {len(dataset)} questions...\n")

    for position, item in enumerate(dataset, start=1):
        question_id = str(item["id"])
        question = item["question"]
        language = item.get("language", "same as the question")
        expected_answerable = item["expected_answerable"]
        expected = expected_pairs(item.get("expected_sources", []))

        print(f"[{position}/{len(dataset)}] {question_id}: {question}")

        try:
            retrieved = search_documents(question, top_k=TOP_K) or []

            result = answer_question(
                question,
                language=language,
            )

            if not isinstance(result, dict):
                raise TypeError(
                    "answer_question() must return a dictionary."
                )

            answer = result.get("answer", "") or ""

            # Prefer the source list associated with the generated answer.
            # Fall back to retrieval results if it is missing or empty.
            sources = result.get("sources") or retrieved

            actual_pairs = source_pairs(retrieved)
            answer_pairs = source_pairs(sources)

            # Retrieval recall: at least one expected document/page retrieved.
            retrieval_hit = None

            if expected:
                retrieval_cases += 1
                retrieval_hit = bool(expected & actual_pairs)

                if retrieval_hit:
                    retrieval_hits += 1

            # Citation reference validity.
            citation_score, valid_count, citation_count = (
                citation_validity(answer, sources)
            )

            if citation_count > 0:
                citation_cases += 1
                total_citations += citation_count
                valid_citations += valid_count

                if valid_count == citation_count:
                    citation_case_hits += 1

            # Answerability: an unanswerable question should be refused;
            # an answerable question should not be refused.
            abstained = is_abstention(answer)
            abstention_correct_case = (
                abstained == (not expected_answerable)
            )

            abstention_cases += 1

            if abstention_correct_case:
                abstention_correct += 1

            # Checks whether any expected source/page appears in answer sources.
            # This is not the same as checking citations in answer text.
            expected_citation_hit = None

            if expected:
                expected_citation_hit = bool(expected & answer_pairs)

            row = {
                "id": question_id,
                "question": question,
                "language": language,
                "expected_answerable": expected_answerable,
                "expected_source_pages": json.dumps(
                    sorted(expected),
                    ensure_ascii=False,
                ),
                "retrieved_source_pages": json.dumps(
                    sorted(actual_pairs),
                    ensure_ascii=False,
                ),
                "retrieval_hit": retrieval_hit,
                "answer_abstained": abstained,
                "abstention_correct": abstention_correct_case,
                "citation_count": citation_count,
                "valid_citation_count": valid_count,
                "citation_precision": citation_score,
                "expected_source_cited": expected_citation_hit,
                "answer": answer,
                "error": "",
            }

            rows.append(row)

            print(
                f"  Retrieval hit: {retrieval_hit}; "
                f"Abstained: {abstained}; "
                f"Abstention correct: {abstention_correct_case}; "
                f"Citations: {valid_count}/{citation_count}\n"
            )

        except Exception as error:
            rows.append(
                {
                    "id": question_id,
                    "question": question,
                    "language": language,
                    "expected_answerable": expected_answerable,
                    "expected_source_pages": json.dumps(
                        sorted(expected),
                        ensure_ascii=False,
                    ),
                    "retrieved_source_pages": "[]",
                    "retrieval_hit": False if expected else None,
                    "answer_abstained": None,
                    "abstention_correct": None,
                    "citation_count": 0,
                    "valid_citation_count": 0,
                    "citation_precision": None,
                    "expected_source_cited": None,
                    "answer": "",
                    "error": str(error),
                }
            )

            print(f"  ERROR: {error}\n")

    # ---------------- METRICS ----------------

    def rate(numerator, denominator):
        if denominator == 0:
            return None

        return round(numerator / denominator, 4)

    summary = {
        "evaluated_at": datetime.now().astimezone().isoformat(),
        "dataset_size": len(dataset),
        "top_k": TOP_K,
        "retrieval_recall_at_k": rate(
            retrieval_hits,
            retrieval_cases,
        ),
        "retrieval_cases_with_expected_sources": retrieval_cases,
        "retrieval_hits": retrieval_hits,
        "abstention_accuracy": rate(
            abstention_correct,
            abstention_cases,
        ),
        "abstention_test_cases": abstention_cases,
        "citation_reference_precision": rate(
            valid_citations,
            total_citations,
        ),
        "citation_reference_cases_all_valid": rate(
            citation_case_hits,
            citation_cases,
        ),
        "citation_cases": citation_cases,
        "total_citations": total_citations,
        "valid_citations": valid_citations,
        "note": (
            "Citation validity checks source references and page metadata "
            "only. It does not measure semantic faithfulness. Review answers "
            "against their evidence to assess faithfulness. Abstention "
            "accuracy is based on heuristic phrase detection."
        ),
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = REPORT_DIR / f"evaluation_{timestamp}.csv"
    json_path = REPORT_DIR / f"summary_{timestamp}.json"

    fieldnames = list(rows[0].keys()) if rows else []

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("\n========== EVALUATION SUMMARY ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nDetailed report: {csv_path}")
    print(f"Summary report:  {json_path}")


if __name__ == "__main__":
    evaluate()
