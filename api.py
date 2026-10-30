
import hashlib
import json
import threading
from pathlib import Path
from datetime import date

from typing import List
from pydantic import BaseModel, Field
from fastapi import HTTPException
from checklist_service import generate_checklist

import pymupdf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag import answer_question
from retrieval import build_index
from summary_service import generate_document_summary
from document_intelligence import (
    analyze_document,
    compare_documents,
    extract_tables,
    normalize_table,
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
INDEX_PATH = BASE_DIR / "vectorstore" / "index.faiss"
CHUNKS_PATH = BASE_DIR / "vectorstore" / "chunks.json"

DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_TOTAL_SIZE = 30 * 1024 * 1024
MAX_SUMMARY_FILES = 50

app = FastAPI(
    title="BhashaProof AI API",
    version="1.2.0",
    description=(
        "Multilingual PDF RAG chatbot with document summaries, "
        "document intelligence, table extraction, and comparison."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

index_lock = threading.Lock()


# --------------------------------------------------
# REQUEST MODELS
# --------------------------------------------------

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=5000)
    language: str = Field(
        default="same as the question",
        max_length=40,
    )


class SummaryRequest(BaseModel):
    filenames: list[str] = Field(min_length=1, max_length=50)
    language: str = Field(
        default="same as the question",
        max_length=40,
    )


class CompareDocumentsRequest(BaseModel):
    file_a: str = Field(min_length=1, max_length=255)
    file_b: str = Field(min_length=1, max_length=255)


# --------------------------------------------------
# SHARED HELPERS
# --------------------------------------------------

def index_is_ready():
    return INDEX_PATH.exists() and CHUNKS_PATH.exists()


def get_document_path(filename: str) -> Path:
    """
    Resolve a PDF inside DOCUMENTS_DIR only.
    Reject path traversal and non-PDF filenames.
    """
    if not filename or Path(filename).name != filename:
        raise HTTPException(
            status_code=400,
            detail="Provide a document filename, not a path.",
        )

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF documents are supported.",
        )

    root = DOCUMENTS_DIR.resolve()
    path = (root / filename).resolve()

    if path.parent != root:
        raise HTTPException(
            status_code=400,
            detail="Invalid document path.",
        )

    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Document '{filename}' was not found.",
        )

    return path


def list_documents():
    """
    List indexed documents using the source names stored in chunks.json.
    """
    if not CHUNKS_PATH.exists():
        return []

    try:
        with CHUNKS_PATH.open("r", encoding="utf-8") as file:
            chunks = json.load(file)
    except (OSError, ValueError):
        return []

    if not isinstance(chunks, list):
        return []

    indexed = {}

    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue

        name = chunk.get("source")
        if not isinstance(name, str) or not name:
            continue

        item = indexed.setdefault(
            name,
            {
                "name": name,
                "pages_seen": set(),
                "chunks": 0,
            },
        )

        page = chunk.get("page")
        if page is not None:
            item["pages_seen"].add(page)

        item["chunks"] += 1

    result = []

    for name, item in indexed.items():
        result.append({
            "name": name,
            "pages": len(item["pages_seen"]),
            "chunks": item["chunks"],
            "indexed": True,
        })

    return sorted(
        result,
        key=lambda item: item["name"].lower(),
    )


def rebuild_index():
    """
    Serialize index rebuilds so concurrent requests cannot rebuild
    the FAISS index simultaneously.
    """
    with index_lock:
        build_index()

    if not index_is_ready():
        raise RuntimeError(
            "Index files were not created. Check that valid PDFs "
            "exist in the documents folder."
        )


def get_indexed_document_names():
    return {
        item["name"]
        for item in list_documents()
    }


# --------------------------------------------------
# BASIC AND HEALTH ENDPOINTS
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "BhashaProof AI Backend is running",
        "status": "success",
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "index_ready": index_is_ready(),
    }


@app.get("/api/documents")
def documents():
    return {
        "documents": list_documents(),
        "index_ready": index_is_ready(),
    }


# --------------------------------------------------
# PDF UPLOAD
# --------------------------------------------------

@app.post("/api/upload")
async def upload_pdfs(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(
            status_code=400,
            detail="Select at least one PDF.",
        )

    saved = []
    total_size = 0

    try:
        for uploaded in files:
            if (
                not uploaded.filename
                or not uploaded.filename.lower().endswith(".pdf")
            ):
                raise HTTPException(
                    status_code=400,
                    detail="Only PDF files are accepted.",
                )

            content = await uploaded.read()
            total_size += len(content)

            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"{uploaded.filename} exceeds "
                        "the 10 MB limit."
                    ),
                )

            if total_size > MAX_TOTAL_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail="The total upload size exceeds 30 MB.",
                )

            if not content.startswith(b"%PDF-"):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"{uploaded.filename} is not "
                        "a valid PDF file."
                    ),
                )

            try:
                with pymupdf.open(
                    stream=content,
                    filetype="pdf",
                ) as pdf:
                    if pdf.page_count == 0:
                        raise ValueError("PDF contains no pages.")
            except Exception as error:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot read {uploaded.filename}: {error}",
                ) from error

            # Hash the content to avoid filename collisions.
            digest = hashlib.sha256(content).hexdigest()[:12]
            original_stem = Path(uploaded.filename).stem

            safe_stem = "".join(
                char
                if char.isalnum() or char in "-_"
                else "_"
                for char in original_stem
            )[:60]

            destination = (
                DOCUMENTS_DIR / f"{safe_stem}_{digest}.pdf"
            )

            if not destination.exists():
                destination.write_bytes(content)

            saved.append(destination.name)

        if saved:
            try:
                rebuild_index()
            except Exception as error:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "Files were saved, but indexing failed: "
                        f"{error}"
                    ),
                ) from error

        return {
            "message": (
                f"Uploaded and indexed {len(saved)} PDF file(s)."
            ),
            "saved": saved,
            "documents": list_documents(),
        }

    finally:
        for uploaded in files:
            await uploaded.close()


# --------------------------------------------------
# REBUILD SEARCH INDEX
# --------------------------------------------------

@app.post("/api/reindex")
def reindex():
    if not list(DOCUMENTS_DIR.glob("*.pdf")):
        raise HTTPException(
            status_code=400,
            detail="No PDFs found in the documents folder.",
        )

    try:
        rebuild_index()

        return {
            "message": "Index rebuilt successfully.",
            "documents": list_documents(),
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Index rebuild failed: {error}",
        ) from error


# --------------------------------------------------
# MULTILINGUAL RAG CHAT
# --------------------------------------------------

@app.post("/api/chat")
def chat(request: ChatRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Enter a question.",
        )

    if not index_is_ready():
        raise HTTPException(
            status_code=400,
            detail="Upload and index a PDF before asking questions.",
        )

    try:
        # Existing RAG behavior: search across all indexed PDFs.
        result = answer_question(
            question,
            language=request.language,
        )

        sources = []

        for source in result.get("sources", []):
            sources.append({
                "document": source.get(
                    "source",
                    "Unknown document",
                ),
                "page": source.get("page"),
                "text": source.get("text", ""),
                "semantic_score": source.get("semantic_score"),
            })

        return {
            "answer": result.get("answer", ""),
            "sources": sources,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not generate an answer: {error}",
        ) from error


# --------------------------------------------------
# SEPARATE SUMMARIES FOR SELECTED PDFS
# --------------------------------------------------

@app.post("/api/summaries")
def document_summaries(request: SummaryRequest):
    """
    Generate a separate summary and important points for each
    selected indexed PDF.

    The summary service filters chunks by exact source filename.
    Regular chat continues to search across all indexed PDFs.
    """
    if not index_is_ready():
        raise HTTPException(
            status_code=400,
            detail="Upload and index PDFs before generating summaries.",
        )

    # Preserve the user's selection order while removing duplicates.
    filenames = list(dict.fromkeys(request.filenames))

    if len(filenames) > MAX_SUMMARY_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Select no more than {MAX_SUMMARY_FILES} PDFs.",
        )

    indexed_names = get_indexed_document_names()

    # Validate every selected file before making any model calls.
    for filename in filenames:
        get_document_path(filename)

        if filename not in indexed_names:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"'{filename}' is not indexed. "
                    "Please rebuild the index and try again."
                ),
            )

    summaries = []

    for filename in filenames:
        try:
            summary = generate_document_summary(
                filename=filename,
                language=request.language,
            )

            summaries.append(summary)

        except Exception as error:
            # Avoid silently presenting partial or invented results.
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Could not summarize '{filename}': {error}"
                ),
            ) from error

    return {
        "status": "success",
        "count": len(summaries),
        "summaries": summaries,
    }


# --------------------------------------------------
# DOCUMENT INTELLIGENCE
# --------------------------------------------------

@app.get("/api/documents/{filename}/intelligence")
def document_intelligence(filename: str):
    """
    Return PDF metadata, page count, detected dates,
    and detected version/revision identifiers.
    """
    path = get_document_path(filename)

    try:
        result = analyze_document(path)

        return {
            **result,
            "analysis_status": "success",
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not analyze the PDF: {error}",
        ) from error


# --------------------------------------------------
# EXTRACTED TABLES AND NORMALIZED MEASUREMENTS
# --------------------------------------------------

@app.get("/api/documents/{filename}/tables")
def document_tables(filename: str):
    """
    Return extracted tables, original cell values,
    normalized measurements, and source page numbers.
    """
    path = get_document_path(filename)

    try:
        tables = [
            normalize_table(table)
            for table in extract_tables(path)
        ]

        return {
            "document": path.name,
            "table_count": len(tables),
            "tables": tables,
            "note": (
                "Table extraction can be imperfect for scanned PDFs "
                "or complex layouts. Verify important values against "
                "the original page."
            ),
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not extract tables: {error}",
        ) from error


# --------------------------------------------------
# DATE / FRESHNESS SUMMARY
# --------------------------------------------------

@app.get("/api/documents/{filename}/freshness")
def document_freshness(filename: str):
    """
    Summarize detected dates to help users review document freshness.
    This does not determine whether a policy is legally current.
    """
    path = get_document_path(filename)

    try:
        result = analyze_document(path)

        dated_items = [
            item
            for item in result.get("dates", [])
            if item.get("normalized_date")
        ]

        dated_items.sort(
            key=lambda item: item["normalized_date"]
        )

        latest_date = (
            dated_items[-1]["normalized_date"]
            if dated_items
            else None
        )

        today = date.today().isoformat()

        return {
            "document": path.name,
            "today": today,
            "page_count": result.get("page_count"),
            "detected_date_count": len(result.get("dates", [])),
            "normalized_date_count": len(dated_items),
            "latest_detected_date": latest_date,
            "versions": result.get("versions", []),
            "dates": result.get("dates", []),
            "freshness_status": (
                "no_unambiguous_date_found"
                if latest_date is None
                else "review_required"
            ),
            "warning": (
                "The latest date found in the text is not necessarily "
                "the publication or effective date. This result does "
                "not confirm that the document is current or valid."
            ),
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not check document dates: {error}",
        ) from error


# --------------------------------------------------
# COMPARE TWO PDF DOCUMENTS
# --------------------------------------------------

@app.post("/api/documents/compare")
def compare_pdf_documents(request: CompareDocumentsRequest):
    """
    Compare two PDFs by detected metadata, dates, versions,
    and extracted table rows.
    """
    if request.file_a == request.file_b:
        raise HTTPException(
            status_code=400,
            detail="Choose two different PDF files to compare.",
        )

    path_a = get_document_path(request.file_a)
    path_b = get_document_path(request.file_b)

    try:
        result = compare_documents(path_a, path_b)

        return {
            "comparison_status": "success",
            **result,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not compare the PDFs: {error}",
        ) from error

