\# 🌐 BhashaProof AI – Multilingual RAG Chatbot



\*\*Ask questions. Understand documents. Get answers in your language.\*\*



BhashaProof AI is an AI-powered multilingual chatbot that allows users to upload PDF documents and ask questions about their content in English, Hindi, and Hinglish. It uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from uploaded documents and generate context-aware answers with source citations.



The main goal of this project is to make document understanding easier, faster, and accessible across different languages.



\## 🚀 Key Features



\- \*\*Multilingual Chat:\*\* Ask questions in English, Hindi, or Hinglish.

\- \*\*PDF Upload:\*\* Upload PDF documents and build a searchable knowledge base.

\- \*\*Retrieval-Augmented Generation (RAG):\*\* Generate answers using information retrieved from uploaded documents.

\- \*\*Hybrid Search:\*\* Combines semantic search and BM25 keyword search using Reciprocal Rank Fusion (RRF).

\- \*\*Source Citations:\*\* Provides document names and page references to help users verify answers.

\- \*\*Document Summarization:\*\* Generate summaries and important points from selected documents.

\- \*\*Document Intelligence:\*\* Extract useful information and structured data from documents.

\- \*\*Document Comparison:\*\* Compare information across multiple uploaded documents.

\- \*\*Context-Aware Responses:\*\* Provides answers based on available document content and handles unsupported questions honestly.



\## 🛠️ Technology Stack



| Component | Technology |

|---|---|

| Programming Language | Python |

| Backend | FastAPI |

| Frontend | React.js |

| Styling | Tailwind CSS |

| Language Model | Groq API |

| Embedding Model | paraphrase-multilingual-MiniLM-L12-v2 |

| Vector Database | FAISS |

| Keyword Search | BM25 |

| PDF Processing | PyMuPDF |

| Development Tools | VS Code, Git, GitHub |



\## ⚙️ How It Works



1\. \*\*Upload:\*\* The user uploads one or more PDF documents.

2\. \*\*Extraction:\*\* The system extracts text from the uploaded PDFs.

3\. \*\*Chunking:\*\* Extracted text is divided into smaller chunks.

4\. \*\*Embedding:\*\* Text chunks are converted into multilingual vector embeddings.

5\. \*\*Indexing:\*\* FAISS stores the embeddings for efficient similarity search.

6\. \*\*Retrieval:\*\* Hybrid retrieval combines semantic and keyword-based search.

7\. \*\*Generation:\*\* The retrieved information is passed to the language model through Groq.

8\. \*\*Response:\*\* The chatbot generates an answer in the user's language with relevant source citations.



\## 🏗️ Project Structure



```text

BhashaProof-AI/

│

├── api.py

├── app.py

├── ingestion.py

├── retrieval.py

├── rag.py

├── summary\_service.py

├── document\_intelligence.py

├── verification.py

├── versioning.py

├── checklist\_service.py

├── checklist\_route.py

├── evaluate.py

├── requirements.txt

│

├── documents/

│

├── evaluation/

│

├── vectorstore/

│

├── frontend/

│   └── frontend/

│       ├── src/

│       ├── public/

│       ├── package.json

│       └── vite.config.js

│

├── .env.example

├── .gitignore

└── README.md

```



\## 💻 Installation and Setup



\### 1. Clone the Repository



```bash

git clone https://github.com/YOUR-USERNAME/BhashaProof-AI.git

```



```bash

cd BhashaProof-AI

```



\### 2. Create a Python Virtual Environment



```bash

python -m venv venv

```



Activate it on Windows:



```bash

venv\\Scripts\\activate

```



\### 3. Install Backend Dependencies



```bash

pip install -r requirements.txt

```



\### 4. Configure Environment Variables



Create a `.env` file in the project root and add:



```env

GROQ\_API\_KEY=your\_groq\_api\_key

GROQ\_MODEL=your\_supported\_groq\_model

```



Replace the placeholder values with your actual Groq API credentials and model name.



\*\*Important:\*\* Never upload your `.env` file or API key to GitHub.



\### 5. Run the Backend



From the project root:



```bash

uvicorn api:app --reload

```



Backend URL:



```text

http://127.0.0.1:8000

```



API documentation:



```text

http://127.0.0.1:8000/docs

```



\### 6. Run the Frontend



Open another terminal:



```bash

cd frontend/frontend

```



Install frontend dependencies:



```bash

npm install

```



Start the development server:



```bash

npm run dev

```



Frontend URL:



```text

http://localhost:5173

```



\## 📌 Example Use Cases



\- Understanding university policies and student handbooks.

\- Extracting important information from government documents.

\- Summarizing lengthy PDF reports.

\- Comparing multiple policy documents.

\- Asking questions about documents in different languages.

\- Finding relevant information without manually reading every page.



\## 🔮 Future Scope



\- Support for additional Indian and international languages.

\- OCR support for scanned PDF documents.

\- Improved document version tracking and freshness detection.

\- Highlighting exact source text within PDF pages.

\- Claim-level answer verification.

\- Advanced table extraction and comparison.

\- Improved retrieval accuracy across different domains.



\## 🎯 Project Objective



The objective of BhashaProof AI is to develop an intelligent document-question-answering system that combines multilingual language understanding with document-based retrieval. It aims to provide accurate, relevant, and verifiable answers while making information accessible to users in their preferred language.



\## 👨‍💻 Author



\*\*Aditya Singh Tomar\*\*



Project: BhashaProof AI – Design and Development of Multilingual RAG Chatbot



\---



⭐ If you find this project interesting, consider giving this repository a star!

