import json
from datetime import datetime, timezone
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage

from rag import llm, normalize_language, language_instructions
from summary_service import (
    _clean_text,
    _extract_json,
    _load_document_chunks,
    _normalize_for_match,
    _prepare_evidence,
    _verify_quote,
)


MAX_DOCUMENTS = 12
MAX_FEES = 8
MAX_DEADLINES = 8

LIMITATIONS = (
    "This checklist lists only items stated in the indexed text of the PDF. "
    "Each item shows its page so you can confirm it in the original document. "
    "Scanned or unclear text may hide items, so check the original before acting. "
    "This is not official advice from the issuing authority."
)


def _request_checklist(filename, evidence, language):
    """Ask the model for checklist items, each with an exact source quote."""
    language_rule = language_instructions(normalize_language(language))

    system_prompt = f"""
You extract practical checklists from ONE government or policy PDF.

LANGUAGE:
{language_rule}

RULES:
1. Include only items that the supplied text states explicitly.
2. Never invent amounts, dates, documents or conditions.
3. Every item must have an evidence_quote copied verbatim from the text.
4. Copy evidence_quote in its original language, as one continuous passage.
5. Documents: papers or certificates the applicant must submit or bring.
6. Fees: amounts or charges the applicant must pay, with their purpose.
7. Deadlines: dates, time limits or processing periods.
8. If a category has nothing stated, return an empty list for it.
9. Ignore navigation text, URLs and unreadable OCR.

Return ONLY valid JSON in this shape:
{{
  "documents": [{{"item": "...", "evidence_quote": "..."}}],
  "fees": [{{"item": "...", "evidence_quote": "..."}}],
  "deadlines": [{{"item": "...", "evidence_quote": "..."}}]
}}

Maximums: {MAX_DOCUMENTS} documents, {MAX_FEES} fees, {MAX_DEADLINES} deadlines.
Do not include Markdown or any text outside the JSON.
"""

    human_prompt = f"""
Document filename: {filename}

DOCUMENT EVIDENCE:

{evidence}
"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ])

    return _extract_json(response.content)


def _verify_items(items, page_texts, limit):
    """Keep only items whose quoted evidence appears in the document text."""
    if not isinstance(items, list):
        return []

    results = []
    seen = set()

    for entry in items:
        if not isinstance(entry, dict):
            continue

        text = _clean_text(entry.get("item", ""))
        quote = _clean_text(entry.get("evidence_quote", ""))

        if not text or not quote:
            continue

        page = _verify_quote(quote, page_texts)

        if page is None:
            continue

        key = _normalize_for_match(text)

        if not key or key in seen:
            continue

        seen.add(key)
        results.append({"item": text, "page": page})

        if len(results) >= limit:
            break

    return results


def generate_checklist(filename, language="same"):
    """Build a verified checklist of documents, fees and deadlines for one PDF."""
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("A valid document filename is required.")

    filename = filename.strip()
    chunks = _load_document_chunks(filename)

    evidence, page_texts, _pages, _truncated, _used = _prepare_evidence(chunks)

    notes = []

    try:
        data = _request_checklist(filename, evidence, language)
    except Exception as exc:
        data = {}
        notes.append(f"The AI step failed: {type(exc).__name__}: {str(exc)[:200]}")

    documents = _verify_items(data.get("documents", []), page_texts, MAX_DOCUMENTS)
    fees = _verify_items(data.get("fees", []), page_texts, MAX_FEES)
    deadlines = _verify_items(data.get("deadlines", []), page_texts, MAX_DEADLINES)

    if not (documents or fees or deadlines):
        notes.append("No items could be verified against the document text.")

    limitations = LIMITATIONS
    if notes:
        limitations += " " + " ".join(notes)

    return {
        "document": filename,
        "documents": documents,
        "fees": fees,
        "deadlines": deadlines,
        "limitations": limitations,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generation_id": uuid4().hex,
    }