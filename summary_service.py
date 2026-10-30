import json
import re
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage

from rag import llm, normalize_language, language_instructions


BASE_DIR = Path(__file__).resolve().parent
CHUNKS_PATH = BASE_DIR / "vectorstore" / "chunks.json"

MAX_EVIDENCE_CHARS = 28000
MAX_EVIDENCE_CHUNKS = 40
MAX_SUMMARY_POINTS = 6
MAX_IMPORTANT_POINTS = 8
MAX_KEY_TERMS = 6


# --------------------------------------------------
# TEXT HELPERS
# --------------------------------------------------

def _normalize_for_match(text):
    """Normalize text to tolerate Unicode, whitespace and OCR formatting."""
    text = unicodedata.normalize("NFKC", str(text or ""))

    text = "".join(
        char
        for char in text
        if unicodedata.category(char) not in {"Cf", "Cc"}
        or char in "\n\t"
    )

    text = "".join(char for char in text if char.isalnum())
    return text.casefold()


def _clean_text(text):
    """Clean ordinary whitespace without changing the source wording."""
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_json(content):
    """Extract a JSON object from an LLM response."""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            elif hasattr(item, "text"):
                parts.append(str(item.text))
            else:
                parts.append(str(item))
        content = "\n".join(parts)

    content = str(content or "").strip()

    if not content:
        raise ValueError("The AI returned an empty response.")

    content = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        content,
        flags=re.IGNORECASE,
    )
    content = re.sub(r"\s*```\s*$", "", content)

    try:
        result = json.loads(content)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()

    for index, character in enumerate(content):
        if character != "{":
            continue
        try:
            result, _ = decoder.raw_decode(content[index:])
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            continue

    raise ValueError("The AI did not return a valid JSON object.")


# --------------------------------------------------
# LOAD DOCUMENT TEXT
# --------------------------------------------------

def _load_document_chunks(filename):
    """Load indexed chunks belonging only to the requested PDF."""
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            "The search index is missing. Please reindex your PDFs."
        )

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not isinstance(chunks, list):
        raise ValueError("The indexed chunks file has an invalid format.")

    matching = []

    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue

        if str(chunk.get("source", "")).strip() != filename:
            continue

        text = _clean_text(chunk.get("text", ""))

        try:
            page = int(chunk.get("page", 0))
        except (TypeError, ValueError):
            continue

        if text and page >= 1:
            matching.append({
                "source": filename,
                "page": page,
                "text": text,
            })

    matching.sort(key=lambda item: item["page"])

    if not matching:
        raise ValueError(
            f"No indexed text was found for '{filename}'. "
            "Check the filename or reindex the PDF."
        )

    return matching


def _prepare_evidence(chunks):
    """Prepare page-labelled evidence within the configured limits."""
    if len(chunks) <= MAX_EVIDENCE_CHUNKS:
        selected = chunks
    else:
        last_index = len(chunks) - 1
        indices = {
            round(i * last_index / (MAX_EVIDENCE_CHUNKS - 1))
            for i in range(MAX_EVIDENCE_CHUNKS)
        }
        selected = [chunks[i] for i in sorted(indices)]

    evidence_parts = []
    page_texts = defaultdict(list)
    total_chars = 0
    truncated = len(selected) < len(chunks)

    for chunk in selected:
        remaining = MAX_EVIDENCE_CHARS - total_chars

        if remaining <= 0:
            truncated = True
            break

        text = chunk["text"]

        if len(text) > remaining:
            text = text[:remaining]
            truncated = True

        if not text.strip():
            continue

        page = chunk["page"]
        evidence_parts.append(f"[PAGE {page}]\n{text}")
        page_texts[page].append(text)
        total_chars += len(text)

    if not evidence_parts:
        raise ValueError("No usable text was found in the selected PDF.")

    combined_page_texts = {
        page: "\n".join(parts)
        for page, parts in page_texts.items()
    }

    return (
        "\n\n".join(evidence_parts),
        dict(combined_page_texts),
        sorted(combined_page_texts),
        truncated,
        len(evidence_parts),
    )


# --------------------------------------------------
# SOURCE VERIFICATION
# --------------------------------------------------

def _verify_quote(quote, page_texts):
    """
    Verify a contiguous source quotation.

    A summary claim may paraphrase its source, but the evidence quote
    must actually occur in the extracted text.
    """
    if not isinstance(quote, str):
        return None

    quote = quote.strip().strip("\"'“”‘’")

    if len(quote) < 12:
        return None

    normalized_quote = _normalize_for_match(quote)

    if len(normalized_quote) < 10:
        return None

    for page, text in page_texts.items():
        normalized_page = _normalize_for_match(text)

        if normalized_quote in normalized_page:
            return page

    return None


def _verified_claims(items, page_texts, limit):
    """Keep only claims supported by a matching source quotation."""
    if not isinstance(items, list):
        return []

    results = []
    seen = set()

    for item in items:
        if not isinstance(item, dict):
            continue

        claim = _clean_text(item.get("claim", ""))
        quote = _clean_text(item.get("evidence_quote", ""))

        if not claim or not quote:
            continue

        page = _verify_quote(quote, page_texts)

        if page is None:
            continue

        normalized_claim = _normalize_for_match(claim)

        if not normalized_claim or normalized_claim in seen:
            continue

        # Avoid accepting a raw fragment as a finished summary.
        if len(claim) < 15:
            continue

        seen.add(normalized_claim)
        results.append({
            "claim": claim,
            "page": page,
        })

        if len(results) >= limit:
            break

    return results


def _verified_key_terms(items, page_texts, limit=MAX_KEY_TERMS):
    """Accept key terms only when their evidence is source-supported."""
    if not isinstance(items, list):
        return []

    results = []
    seen = set()

    for item in items:
        if not isinstance(item, dict):
            continue

        term = _clean_text(item.get("term", ""))
        meaning = _clean_text(item.get("meaning", ""))
        quote = _clean_text(item.get("evidence_quote", ""))

        if not term or not meaning or not quote:
            continue

        if _verify_quote(quote, page_texts) is None:
            continue

        normalized_term = _normalize_for_match(term)
        normalized_quote = _normalize_for_match(quote)

        if not normalized_term or normalized_term not in normalized_quote:
            continue

        if normalized_term in seen:
            continue

        seen.add(normalized_term)
        results.append({
            "term": term,
            "meaning": meaning,
        })

        if len(results) >= limit:
            break

    return results


# --------------------------------------------------
# SAFE FALLBACK
# --------------------------------------------------

def _line_quality(line):
    """Return whether a line looks usable as a source excerpt."""
    line = _clean_text(line)

    if len(line) < 25:
        return False

    if not re.search(r"[A-Za-zऀ-ॿ]", line):
        return False

    # Reject lines dominated by punctuation and OCR separators.
    letters = sum(char.isalpha() for char in line)
    if letters / max(len(line), 1) < 0.35:
        return False

    # Reject suspiciously fragmented OCR such as many isolated slashes.
    if line.count("/") >= 4:
        return False

    return True


def _extract_source_lines(chunks, limit=MAX_SUMMARY_POINTS + MAX_IMPORTANT_POINTS):
    """
    Return clean source excerpts only when they look readable.

    These are explicitly source excerpts, not AI-generated interpretations.
    """
    results = []
    seen = set()

    for chunk in chunks:
        for line in re.split(r"[\r\n]+", chunk["text"]):
            line = _clean_text(line).strip(" \t•-–—")

            if not _line_quality(line):
                continue

            normalized = _normalize_for_match(line)

            if not normalized or normalized in seen:
                continue

            if re.fullmatch(
                r"[\d\s./:-]+|aboutblank|printcontent",
                line,
                flags=re.IGNORECASE,
            ):
                continue

            seen.add(normalized)
            results.append({
                "claim": line,
                "page": chunk["page"],
            })

            if len(results) >= limit:
                return results

    return results


def _fallback_key_terms(chunks, limit=MAX_KEY_TERMS):
    """Extract recognizable labels without inventing their definitions."""
    heading_patterns = (
        "योजना का नाम",
        "विभाग",
        "योजना का उद्देश्य",
        "लाभार्थी वर्ग",
        "लाभ की श्रेणी",
        "आवेदन प्रक्रिया",
        "समय सीमा",
        "आवेदन शुल्क",
        "योजना का लाभ",
        "scheme name",
        "department",
        "eligibility",
        "application process",
        "processing time",
    )

    results = []
    seen = set()

    for chunk in chunks:
        for line in re.split(r"[\r\n]+", chunk["text"]):
            line = _clean_text(line).strip(" :-–—")

            if not line or len(line) > 100:
                continue

            if not _line_quality(line):
                continue

            normalized = _normalize_for_match(line)

            if not normalized or normalized in seen:
                continue

            if any(pattern.casefold() in line.casefold()
                   for pattern in heading_patterns):
                seen.add(normalized)
                results.append({
                    "term": line,
                    "meaning": (
                        "This label appears in the source document. "
                        "Its detailed meaning could not be verified "
                        "from the extracted text."
                    ),
                })

                if len(results) >= limit:
                    return results

    return results


# --------------------------------------------------
# AI SUMMARY
# --------------------------------------------------

def _request_summary(filename, evidence, language, retry=False):
    """Ask the model for grounded claims and source quotations."""
    selected_language = normalize_language(language)
    language_rule = language_instructions(selected_language)

    retry_rule = ""
    if retry:
        retry_rule = """
IMPORTANT: Your previous answer did not provide verifiable quotations.
Try again. Copy each evidence_quote as one exact, continuous passage
from the supplied text. Do not invent, translate, shorten, or paraphrase
the quotation. If a fact cannot be supported, omit it.
"""

    system_prompt = f"""
You are BhashaProof AI, a careful multilingual document summarizer.

Summarize ONLY the supplied extracted text from ONE PDF.

LANGUAGE:
{language_rule}

RULES:
1. Write readable, concise summary claims in the requested language.
2. Do not use outside knowledge or invent facts.
3. Preserve names, dates, percentages, amounts, eligibility conditions,
   and timelines only when supported by the source.
4. Every claim must have an evidence_quote copied verbatim from the
   supplied text.
5. Each evidence_quote must be a continuous passage, not a paraphrase.
6. Copy quotations in their original language, even if the claim is
   translated.
7. Do not include page numbers inside claims; page references are added
   by the application.
8. Do not treat headings as explanations or meanings.
9. Ignore unreadable or badly corrupted OCR instead of guessing.
10. Avoid duplicate facts and repeated wording.
11. Prefer meaningful facts, procedures, eligibility, dates, amounts,
    and conditions over navigation text or URLs.
12. Return fewer items rather than unsupported claims.
{retry_rule}

Return ONLY valid JSON:
{{
  "summary_points": [
    {{
      "claim": "A concise summary of an important supported fact",
      "evidence_quote": "An exact continuous quotation from the evidence"
    }}
  ],
  "important_points": [
    {{
      "claim": "An important fact or condition",
      "evidence_quote": "An exact continuous quotation from the evidence"
    }}
  ],
  "key_terms": [
    {{
      "term": "A term found in the evidence",
      "meaning": "A short meaning supported by the evidence",
      "evidence_quote": "An exact continuous quotation containing the term"
    }}
  ]
}}

Maximums: 6 summary points, 8 important points, 6 key terms.
Do not include Markdown or any text outside the JSON.
"""

    human_prompt = f"""
Document filename: {filename}

Summarize this document only. Do not combine it with other documents.
The text may contain OCR mistakes. Never guess what unclear text means.

DOCUMENT EVIDENCE:

{evidence}
"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ])

    return _extract_json(response.content)


def _format_claims(items):
    """
    Format claims as a Markdown bullet list, one point per line.

    Each point ends with its page reference so the frontend can render
    the summary as a readable list instead of one long paragraph.
    """
    return "\n".join(
        f"- {item['claim']} *(Page {item['page']})*"
        for item in items
    )


# --------------------------------------------------
# PUBLIC SUMMARY FUNCTION
# --------------------------------------------------

def generate_document_summary(filename, language="same"):
    """
    Generate a grounded summary for one indexed PDF.

    Claims are accepted only when their source quotations can be verified.
    If generation or verification fails, do not present unreadable OCR as
    though it were a reliable AI-generated summary.
    """
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("A valid document filename is required.")

    filename = filename.strip()
    chunks = _load_document_chunks(filename)

    (
        evidence,
        page_texts,
        evidence_pages,
        truncated,
        evidence_chunks_used,
    ) = _prepare_evidence(chunks)

    summary_claims = []
    important_claims = []
    key_terms = []
    model_error = None
    retry_used = False

    # First attempt, followed by one stricter retry if verification fails.
    for attempt in range(2):
        try:
            data = _request_summary(
                filename=filename,
                evidence=evidence,
                language=language,
                retry=(attempt == 1),
            )

            candidate_summary = _verified_claims(
                data.get("summary_points", []),
                page_texts,
                MAX_SUMMARY_POINTS,
            )

            candidate_important = _verified_claims(
                data.get("important_points", []),
                page_texts,
                MAX_IMPORTANT_POINTS,
            )

            candidate_terms = _verified_key_terms(
                data.get("key_terms", []),
                page_texts,
            )

            if candidate_summary or candidate_important:
                summary_claims = candidate_summary
                important_claims = candidate_important
                key_terms = candidate_terms
                retry_used = attempt == 1
                model_error = None
                break

            model_error = (
                "The AI returned claims whose source quotations "
                "could not be verified."
            )

        except Exception as exc:
            model_error = f"{type(exc).__name__}: {exc}"

        retry_used = attempt == 1

    fallback_used = False

    # Use only readable excerpts if no verifiable AI claims were produced.
    if not summary_claims and not important_claims:
        fallback_used = True
        readable_lines = _extract_source_lines(chunks)

        summary_claims = readable_lines[:MAX_SUMMARY_POINTS]
        important_claims = readable_lines[
            MAX_SUMMARY_POINTS:
            MAX_SUMMARY_POINTS + MAX_IMPORTANT_POINTS
        ]

    if not key_terms:
        key_terms = _fallback_key_terms(chunks)

    summary = _format_claims(summary_claims)

    important_points = [
        f"{item['claim']} [Page {item['page']}]"
        for item in important_claims
    ]

    if not summary:
        summary = (
            "A reliable summary could not be generated because the "
            "extracted PDF text was unreadable or the AI-generated "
            "claims could not be verified. Reindex the PDF and check "
            "its text extraction or OCR quality."
        )

    if truncated:
        coverage_note = (
            "Only a selection of indexed text chunks was reviewed. "
            "Some document sections may not be represented."
        )
    else:
        coverage_note = (
            "All indexed chunks selected for this document were reviewed. "
            "Unreadable scanned text or extraction errors may still affect "
            "completeness."
        )

    if fallback_used:
        coverage_note += (
            " Readable source excerpts were used where possible because "
            "no AI claims could be verified. These excerpts are not a "
            "complete AI-generated summary."
        )

    if retry_used:
        coverage_note += (
            " A stricter second attempt was made to obtain verifiable "
            "source quotations."
        )

    if model_error:
        coverage_note += (
            " Summary-generation detail: " + model_error[:500]
        )

    limitations = (
        "The summary is based on extracted indexed text, not a fresh "
        "visual inspection of the PDF. OCR errors, missing tables, or "
        "unreadable scanned content may affect completeness. Verify "
        "important dates, names, amounts, and legal provisions against "
        "the original document."
    )

    return {
        "document": filename,
        "summary": summary,
        "important_points": important_points,
        "key_terms": key_terms,
        "limitations": limitations,
        "sources": [
            {"document": filename, "page": page}
            for page in evidence_pages
        ],
        "evidence_pages": evidence_pages,
        "evidence_chunks_used": evidence_chunks_used,
        "total_indexed_chunks": len(chunks),
        "coverage_note": coverage_note,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generation_id": uuid4().hex,
    }