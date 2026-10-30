
import hashlib
import json
from pathlib import Path

import streamlit as st
from rag import answer_question
from retrieval import build_index

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="BhashaProof AI",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

DOCS = Path("documents")
INDEX = Path("vectorstore/index.faiss")
CHUNKS = Path("vectorstore/chunks.json")
DOCS.mkdir(parents=True, exist_ok=True)

# ---------------- STYLING ----------------

st.markdown("""
<style>
/* Main application background and default text */
.stApp {
    background: #f5f7fc;
    color: #18233f;
}

[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
    background-color: #f5f7fc;
}

[data-testid="stMain"] p,
[data-testid="stMain"] label,
[data-testid="stMain"] li,
[data-testid="stMain"] .stMarkdown,
[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3,
[data-testid="stMain"] h4 {
    color: #18233f;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111d36;
}

[data-testid="stSidebar"] * {
    color: #f4f7ff;
}

[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: #18233f;
}

/* Page heading */
.hero {
    padding: 28px 30px;
    border-radius: 20px;
    background: linear-gradient(120deg, #e0f2fe, #ede9fe);
    border: 1px solid #d6e3ff;
    margin: 8px 0 24px 0;
}

.hero h1 {
    color: #15264a !important;
    font-size: 32px;
    font-weight: 750;
    margin: 0 0 10px 0;
}

.hero p {
    color: #425579 !important;
    font-size: 16px;
    margin: 5px 0;
}

/* Cards */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff;
    border: 1px solid #e0e7f2;
    border-radius: 16px;
}

/* Buttons */
.stButton > button {
    background: #ffffff;
    color: #243657 !important;
    border: 1px solid #d7dfed;
    border-radius: 10px;
    min-height: 42px;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: #eef2ff;
    color: #3730a3 !important;
    border-color: #818cf8;
}

/* Primary button */
.stButton > button[kind="primary"] {
    background: #4f46e5;
    color: #ffffff !important;
    border: 1px solid #4f46e5;
}

.stButton > button[kind="primary"]:hover {
    background: #4338ca;
    color: #ffffff !important;
}

/* Text input, chat input, and select */
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] textarea,
.stTextInput input,
[data-baseweb="select"] > div {
    background: #ffffff !important;
    color: #18233f !important;
    border-color: #d7dfed !important;
}

[data-testid="stChatInput"] textarea {
    caret-color: #4f46e5;
}

[data-testid="stChatInput"] textarea::placeholder,
.stTextInput input::placeholder {
    color: #7a879f !important;
    opacity: 1;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: #ffffff;
    border: 1px solid #e3e9f4;
    border-radius: 15px;
    padding: 16px;
    margin-bottom: 12px;
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li {
    color: #18233f !important;
    line-height: 1.7;
}

/* Expander */
[data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid #dfe6f2;
    border-radius: 12px;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p {
    color: #243657 !important;
}

/* Upload area */
[data-testid="stFileUploader"] {
    background: #f8faff;
    border: 1px dashed #a5b4fc;
    border-radius: 12px;
    padding: 12px;
}

[data-testid="stFileUploader"] * {
    color: #243657;
}

/* Metrics and dividers */
[data-testid="stMetricValue"] {
    color: #4f46e5;
}

hr {
    border-color: #e1e7f0;
}

/* Footer-like helper text */
.helper-text {
    color: #64748b;
    font-size: 13px;
}

/* Small screens */
@media (max-width: 768px) {
    .hero {
        padding: 20px;
    }

    .hero h1 {
        font-size: 25px;
    }
}
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ---------------- HELPERS ----------------

def pdf_files():
    return sorted(DOCS.glob("*.pdf"))


def get_sources():
    if not INDEX.exists() or not CHUNKS.exists():
        return []

    try:
        with open(CHUNKS, encoding="utf-8") as file:
            chunks = json.load(file)
        return sorted({chunk["source"] for chunk in chunks})
    except (OSError, ValueError, KeyError):
        return []


def save_pdfs(files):
    saved = 0

    for uploaded in files:
        if not uploaded.name.lower().endswith(".pdf"):
            st.warning(f"Skipped {uploaded.name}: not a PDF.")
            continue

        content = uploaded.getvalue()

        if not content.startswith(b"%PDF-"):
            st.warning(f"Skipped {uploaded.name}: invalid PDF file.")
            continue

        stem = "".join(
            c if c.isalnum() or c in "-_" else "_"
            for c in Path(uploaded.name).stem
        )[:65]

        digest = hashlib.sha256(content).hexdigest()[:10]
        destination = DOCS / f"{stem}_{digest}.pdf"

        if not destination.exists():
            destination.write_bytes(content)
            saved += 1

    return saved


# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.markdown("# 🌐 BhashaProof AI")
    st.caption("Your multilingual document assistant")
    st.divider()

    page = st.radio(
        "Navigate",
        ["Chat", "Documents", "About"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("### 🌍 Answer language")

    language = st.selectbox(
        "Choose language",
        ["Same as question", "English", "Hindi", "Hinglish"],
        label_visibility="collapsed",
    )

    st.divider()
    sources = get_sources()

    st.markdown("### 📊 Knowledge base")
    st.metric("PDFs stored", len(pdf_files()))
    st.metric("Indexed documents", len(sources))

    if sources:
        st.success("Search index ready")
    else:
        st.info("Upload a PDF to begin")

    st.divider()
    st.caption("🔐 Keep your API key private.")
    st.caption("Always verify important answers against their sources.")


# ---------------- HERO ----------------

st.markdown("""
<div class="hero">
    <h1>Welcome to BhashaProof AI ✨</h1>
    <p>Ask questions from your PDFs in English, Hindi, or Hinglish.</p>
    <p>Search your documents, get grounded answers, and inspect the sources.</p>
</div>
""", unsafe_allow_html=True)


# ---------------- PDF UPLOAD ----------------

with st.container(border=True):
    st.subheader("📄 Your documents")
    st.write("Upload one or more PDF files, then index them to make their content searchable.")

    uploads = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="PDFs are stored in your local documents folder.",
    )

    left, right = st.columns(2)

    with left:
        save_clicked = st.button(
            "⬆️ Save and index PDFs",
            type="primary",
            use_container_width=True,
            disabled=not uploads,
        )

    with right:
        rebuild_clicked = st.button(
            "🔄 Rebuild index",
            use_container_width=True,
        )

    if save_clicked and uploads:
        try:
            with st.spinner("Saving files and indexing documents..."):
                saved = save_pdfs(uploads)

                if saved:
                    build_index()
                    st.session_state.messages = []
                    st.success(f"Indexed {saved} new PDF file(s).")
                    st.rerun()
                else:
                    st.info(
                        "No new PDFs were saved. These files may already "
                        "exist, or the files may be invalid."
                    )
        except Exception as error:
            st.error(f"Indexing failed: {error}")

    if rebuild_clicked:
        if not pdf_files():
            st.warning("Upload at least one PDF first.")
        else:
            try:
                with st.spinner("Rebuilding the complete search index..."):
                    build_index()
                    st.session_state.messages = []
                st.success("Index rebuilt successfully.")
                st.rerun()
            except Exception as error:
                st.error(f"Index rebuild failed: {error}")


# ---------------- DOCUMENTS PAGE ----------------

if page == "Documents":
    st.subheader("📚 Document library")

    files = pdf_files()
    indexed = get_sources()

    if not files:
        st.info("No PDF files found. Upload one above.")
    else:
        for file in files:
            with st.container(border=True):
                col1, col2 = st.columns([4, 1])

                with col1:
                    st.markdown(f"**📄 {file.name}**")
                    if file.name in indexed:
                        st.success("Included in current index")
                    else:
                        st.warning("Not included in current index")

                with col2:
                    st.caption("File size")
                    st.write(f"{file.stat().st_size / 1024:.1f} KB")

    st.caption(
        "To remove a PDF, delete it from the documents folder and rebuild "
        "the index. A document deletion control will be added separately."
    )


# ---------------- ABOUT PAGE ----------------

elif page == "About":
    st.subheader("💡 About BhashaProof AI")

    st.write(
        "BhashaProof AI is a multilingual Retrieval-Augmented Generation "
        "application that answers questions using your PDF documents."
    )

    st.markdown("### Technology stack")
    st.markdown("""
    - **Streamlit:** user interface
    - **PyMuPDF:** PDF text extraction
    - **Sentence Transformers:** multilingual embeddings
    - **FAISS:** semantic retrieval
    - **BM25 + RRF:** hybrid search
    - **Groq:** language-model responses
    """)

    st.markdown("### Current capabilities")
    st.markdown("""
    - Upload multiple PDFs
    - Ask questions in English, Hindi, and Hinglish
    - Search document passages
    - View source document names and page numbers
    """)


# ---------------- CHAT PAGE ----------------

elif page == "Chat":
    st.subheader("💬 Chat with your documents")

    sources = get_sources()

    if not sources:
        st.info("Upload a PDF and click 'Save and index PDFs' to start chatting.")
    else:
        st.caption(f"Searching across {len(sources)} indexed PDF(s).")

    st.markdown("#### ⚡ Quick questions")

    examples = [
        "What is the application fee?",
        "Who is eligible for this scheme?",
        "इस योजना के लिए कौन पात्र है?",
        "Yojana mein kitni hectare zameen ka benefit mil sakta hai?",
    ]

    col1, col2 = st.columns(2)

    for index, example in enumerate(examples):
        with (col1 if index % 2 == 0 else col2):
            if st.button(example, key=f"example_{index}", use_container_width=True):
                st.session_state.pending_question = example

    st.divider()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("📚 View source passages"):
                    for source in message["sources"]:
                        st.markdown(
                            f"**{source['source']} · Page {source['page']}**"
                        )
                        st.write(source["text"])
                        st.divider()

    typed_question = st.chat_input(
        "Ask a question about your documents..."
    )

    question = typed_question

    if not question and st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None

    if question:
        if not INDEX.exists() or not CHUNKS.exists():
            st.warning("Upload and index a PDF before asking questions.")
        else:
            st.session_state.messages.append(
                {"role": "user", "content": question}
            )

            with st.chat_message("user"):
                st.markdown(question)

            selected_language = (
                "same as the question"
                if language == "Same as question"
                else language
            )

            with st.chat_message("assistant"):
                with st.spinner("Searching evidence and preparing an answer..."):
                    try:
                        result = answer_question(
                            question,
                            language=selected_language,
                        )

                        answer = result["answer"]
                        retrieved_sources = result["sources"]

                        st.markdown(answer)

                        if retrieved_sources:
                            with st.expander("📚 View source passages"):
                                for source in retrieved_sources:
                                    st.markdown(
                                        f"**{source['source']} · "
                                        f"Page {source['page']}**"
                                    )
                                    st.write(source["text"])
                                    st.divider()

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": retrieved_sources,
                        })

                    except Exception as error:
                        st.error(f"Could not generate an answer: {error}")

    if st.session_state.messages:
        if st.button("🗑️ Clear conversation"):
            st.session_state.messages = []
            st.rerun()
