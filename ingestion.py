
from pathlib import Path
import fitz

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Folder containing PDF files
DOCUMENTS_DIR = Path("documents")

# Split long text into smaller chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def load_pdf_documents():
    chunks = []

    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in the documents folder.")
        return chunks

    for pdf_path in pdf_files:
        print(f"\nReading: {pdf_path.name}")

        try:
            with fitz.open(pdf_path) as pdf:
                for page_index, page in enumerate(pdf):
                    text = page.get_text("text").strip()

                    if not text:
                        continue

                    page_chunks = splitter.split_text(text)

                    for chunk_index, chunk in enumerate(page_chunks):
                        chunks.append({
                            "text": chunk,
                            "source": pdf_path.name,
                            "page": page_index + 1,
                            "chunk_id": (
                                f"{pdf_path.stem}_"
                                f"{page_index + 1}_"
                                f"{chunk_index}"
                            ),
                        })

        except Exception as error:
            print(f"Error reading {pdf_path.name}: {error}")

    return chunks


if __name__ == "__main__":
    documents = load_pdf_documents()

    print("\n" + "=" * 40)
    print(f"Total chunks extracted: {len(documents)}")
    print("=" * 40)

    for item in documents[:3]:
        print(f"\nSource: {item['source']}")
        print(f"Page: {item['page']}")
        print(f"Chunk ID: {item['chunk_id']}")
        print(f"Text: {item['text'][:300]}")
