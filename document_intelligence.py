
from pathlib import Path
from datetime import datetime
import re
import json

import pymupdf
import pdfplumber


DOCUMENTS_DIR = Path("documents")

# Recognized date formats
DATE_PATTERNS = [
    r"\b\d{4}-\d{2}-\d{2}\b",
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b",
    r"\b\d{1,2}[- ](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[- ]\d{4}\b",
]

VERSION_PATTERNS = [
    r"\bversion\s*[:#-]?\s*([A-Za-z0-9._-]+)",
    r"\brev(?:ision)?\.?\s*[:#-]?\s*([A-Za-z0-9._-]+)",
    r"\b(v\d+(?:\.\d+)*)\b",
]

# Convert recognized units to a canonical unit.
# Original values are always preserved separately.
UNIT_FACTORS = {
    "m": ("length", "m", 1.0),
    "meter": ("length", "m", 1.0),
    "meters": ("length", "m", 1.0),
    "km": ("length", "m", 1000.0),
    "kilometer": ("length", "m", 1000.0),
    "kilometers": ("length", "m", 1000.0),
    "cm": ("length", "m", 0.01),
    "mm": ("length", "m", 0.001),
    "g": ("mass", "g", 1.0),
    "gram": ("mass", "g", 1.0),
    "kg": ("mass", "g", 1000.0),
    "kilogram": ("mass", "g", 1000.0),
    "mg": ("mass", "g", 0.001),
    "l": ("volume", "L", 1.0),
    "liter": ("volume", "L", 1.0),
    "litre": ("volume", "L", 1.0),
    "ml": ("volume", "L", 0.001),
}


def parse_date(value):
    """Parse common dates without guessing ambiguous dates."""
    value = value.strip()

    formats = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%d %B %Y",
        "%d %b %Y",
        "%d-%B-%Y",
        "%d-%b-%Y",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            continue

    # A numeric date such as 03/04/2025 is ambiguous:
    # do not assume whether it means March 4 or April 3.
    return None


def detect_dates(text):
    """Return detected dates with nearby evidence."""
    results = []
    seen = set()

    for pattern in DATE_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            raw = match.group(0)
            normalized = parse_date(raw)

            # Do not silently guess the meaning of ambiguous dates.
            key = (raw, match.start())
            if key in seen:
                continue
            seen.add(key)

            start = max(0, match.start() - 90)
            end = min(len(text), match.end() + 90)

            results.append({
                "raw_date": raw,
                "normalized_date": normalized,
                "evidence": text[start:end].replace("\n", " ").strip(),
                "character_position": match.start(),
            })

    return results


def detect_versions(text):
    """Return version and revision identifiers with evidence."""
    results = []

    for pattern in VERSION_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start = max(0, match.start() - 60)
            end = min(len(text), match.end() + 60)

            results.append({
                "identifier": match.group(1),
                "matched_text": match.group(0),
                "evidence": text[start:end].replace("\n", " ").strip(),
            })

    # Deduplicate overlapping patterns.
    unique = {}
    for item in results:
        unique[(item["identifier"], item["matched_text"])] = item

    return list(unique.values())


def extract_metadata(pdf_path):
    """Extract PDF metadata, dates, and version identifiers."""
    pdf_path = Path(pdf_path)

    with pymupdf.open(pdf_path) as pdf:
        metadata = pdf.metadata or {}
        pages = []

        for page_number, page in enumerate(pdf, start=1):
            pages.append({
                "page": page_number,
                "text": page.get_text("text").strip(),
            })

    full_text = "\n".join(item["text"] for item in pages)

    return {
        "source": pdf_path.name,
        "page_count": len(pages),
        "file_metadata": {
            "title": metadata.get("title"),
            "author": metadata.get("author"),
            "creation_date": metadata.get("creationDate"),
            "modification_date": metadata.get("modDate"),
        },
        "dates": detect_dates(full_text),
        "versions": detect_versions(full_text),
        "pages": pages,
    }


def extract_tables(pdf_path):
    """Extract tables while retaining page numbers and original cells."""
    pdf_path = Path(pdf_path)
    results = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            try:
                tables = page.extract_tables() or []
            except Exception:
                tables = []

            for table_index, table in enumerate(tables, start=1):
                cleaned_rows = []

                for row in table:
                    cleaned = [
                        str(cell).strip() if cell is not None else ""
                        for cell in row
                    ]

                    if any(cleaned):
                        cleaned_rows.append(cleaned)

                if not cleaned_rows:
                    continue

                results.append({
                    "source": pdf_path.name,
                    "page": page_number,
                    "table_index": table_index,
                    "headers": cleaned_rows[0],
                    "rows": cleaned_rows[1:],
                    "raw_rows": cleaned_rows,
                })

    return results


def normalize_measurement(value):
    """
    Normalize simple measurements such as 1 km and 1000 m.

    This intentionally does not convert currencies, percentages,
    area, or units that have not been explicitly defined.
    """
    if not isinstance(value, str):
        return None

    text = value.strip().replace(",", "")

    match = re.fullmatch(
        r"([-+]?\d+(?:\.\d+)?)\s*([A-Za-z]+)",
        text,
    )

    if not match:
        return None

    number = float(match.group(1))
    unit = match.group(2).lower()

    if unit not in UNIT_FACTORS:
        return None

    dimension, canonical_unit, factor = UNIT_FACTORS[unit]

    return {
        "original_value": value,
        "original_number": number,
        "original_unit": match.group(2),
        "dimension": dimension,
        "normalized_value": number * factor,
        "normalized_unit": canonical_unit,
    }


def normalize_table(table):
    """Add normalized measurement data without changing source cells."""
    normalized_rows = []

    for row in table["rows"]:
        normalized_cells = []

        for cell in row:
            normalized_cells.append(normalize_measurement(cell))

        normalized_rows.append({
            "original_cells": row,
            "normalized_cells": normalized_cells,
        })

    return {
        **table,
        "normalized_rows": normalized_rows,
    }


def analyze_document(pdf_path):
    """Run all Phase 4 analysis on one PDF."""
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    metadata = extract_metadata(pdf_path)
    tables = [
        normalize_table(table)
        for table in extract_tables(pdf_path)
    ]

    return {
        "source": metadata["source"],
        "page_count": metadata["page_count"],
        "file_metadata": metadata["file_metadata"],
        "dates": metadata["dates"],
        "versions": metadata["versions"],
        "tables": tables,
    }


def compare_documents(file_a, file_b):
    """
    Compare document metadata and extracted table rows.

    This is a first-pass structural comparison, not a semantic
    guarantee that two rows describe the same requirement.
    """
    doc_a = analyze_document(file_a)
    doc_b = analyze_document(file_b)

    def table_row_set(document):
        rows = set()

        for table in document["tables"]:
            for row in table["rows"]:
                normalized = tuple(
                    " ".join(cell.lower().split())
                    for cell in row
                )
                rows.add(normalized)

        return rows

    rows_a = table_row_set(doc_a)
    rows_b = table_row_set(doc_b)

    return {
        "document_a": doc_a["source"],
        "document_b": doc_b["source"],
        "versions": {
            "document_a": doc_a["versions"],
            "document_b": doc_b["versions"],
        },
        "dates": {
            "document_a": doc_a["dates"],
            "document_b": doc_b["dates"],
        },
        "table_comparison": {
            "rows_only_in_a": [list(row) for row in sorted(rows_a - rows_b)],
            "rows_only_in_b": [list(row) for row in sorted(rows_b - rows_a)],
            "identical_rows": len(rows_a & rows_b),
        },
        "warning": (
            "Different table rows may reflect formatting, OCR, or "
            "content changes. Review the cited source pages before "
            "concluding that a requirement has changed."
        ),
    }


if __name__ == "__main__":
    DOCUMENTS_DIR.mkdir(exist_ok=True)

    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDFs found in the documents folder.")
    else:
        for pdf_path in pdf_files:
            try:
                result = analyze_document(pdf_path)

                output_path = pdf_path.with_name(
                    f"{pdf_path.stem}_intelligence.json"
                )

                output_path.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                print(f"\nAnalyzed: {pdf_path.name}")
                print(f"Pages: {result['page_count']}")
                print(f"Dates found: {len(result['dates'])}")
                print(f"Version identifiers: {len(result['versions'])}")
                print(f"Tables extracted: {len(result['tables'])}")
                print(f"Saved report: {output_path}")

            except Exception as error:
                print(f"Error analyzing {pdf_path.name}: {error}")
