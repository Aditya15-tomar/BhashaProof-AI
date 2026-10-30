import { useEffect, useRef, useState } from "react";
import {
  ArrowUp, BookOpen, Check, ChevronDown, FileText,
  Globe2, Languages, ListChecks, LoaderCircle, MessageSquare, Plus, RefreshCw,
  Sparkles, UploadCloud, X, PanelLeftClose, PanelLeftOpen, Trash2,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./App.css";

const API = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

const LANGUAGES = [
  { value: "same", label: "Auto-detect language" },
  { value: "English", label: "English" },
  { value: "Hindi", label: "हिन्दी · Hindi" },
  { value: "Hinglish", label: "Hinglish" },
  { value: "Bengali", label: "বাংলা · Bengali" },
  { value: "Tamil", label: "தமிழ் · Tamil" },
  { value: "Telugu", label: "తెలుగు · Telugu" },
  { value: "Marathi", label: "मराठी · Marathi" },
  { value: "Gujarati", label: "ગુજરાતી · Gujarati" },
];

// kind: "summary" uses /api/summaries; "checklist" uses /api/checklist;
// "chat" asks the question once per selected PDF.
const SUGGESTIONS = [
  {
    icon: BookOpen,
    title: "Give me a summary",
    shortLabel: "Summary",
    kind: "summary",
    prompt: "Summarize the main points of my uploaded documents.",
    tone: "lilac",
  },
  {
    icon: FileText,
    title: "Find key information",
    shortLabel: "Key information",
    kind: "chat",
    prompt: "What are the most important rules and requirements in this document?",
    tone: "peach",
  },
  {
    icon: Globe2,
    title: "Explain it simply",
    shortLabel: "Simple explanation",
    kind: "chat",
    prompt: "Explain the main ideas in this document in simple language.",
    tone: "mint",
  },
  {
    icon: ListChecks,
    title: "Checklist and deadlines",
    shortLabel: "Checklist",
    kind: "checklist",
    prompt: "List the documents required, fees and deadlines.",
    tone: "sky",
  },
];

const MAX_FILE_BYTES = 10 * 1024 * 1024;
const MAX_TOTAL_BYTES = 30 * 1024 * 1024;

function getDocumentList(data) {
  if (Array.isArray(data)) return data;
  return data?.documents || data?.files || data?.items || [];
}

function getDocumentName(doc) {
  return typeof doc === "string"
    ? doc
    : doc?.filename || doc?.name || doc?.document || doc?.source || "Untitled PDF";
}

function getDocumentMeta(doc) {
  if (typeof doc === "string") return "PDF document";
  const pages = doc?.pages ?? doc?.page_count;
  const chunks = doc?.chunks ?? doc?.chunk_count;
  if (pages != null && chunks != null) return `${pages} pages · ${chunks} chunks`;
  if (pages != null) return `${pages} pages`;
  if (chunks != null) return `${chunks} chunks`;
  return "Indexed document";
}

function formatAnswer(answer) {
  if (typeof answer === "string") return answer;
  if (answer == null) return "I couldn't find an answer in the response.";
  return JSON.stringify(answer, null, 2);
}

function languageParam(language) {
  return language === "same" ? "same as the question" : language;
}

function AssistantMarkdown({ children }) {
  return (
    <div className="bp-summary-markdown">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ children: linkChildren, ...props }) => (
            <a {...props} target="_blank" rel="noopener noreferrer">
              {linkChildren}
            </a>
          ),
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}

const CHECKLIST_SECTIONS = [
  { key: "documents", title: "Documents to bring" },
  { key: "fees", title: "Fees and payments" },
  { key: "deadlines", title: "Deadlines and timelines" },
];

function ChecklistCard({ data }) {
  const [checked, setChecked] = useState({});
  const [copied, setCopied] = useState(false);

  const sections = CHECKLIST_SECTIONS.map((section) => ({
    ...section,
    items: Array.isArray(data?.[section.key]) ? data[section.key] : [],
  }));

  const total = sections.reduce((count, section) => count + section.items.length, 0);
  const done = Object.values(checked).filter(Boolean).length;

  function toggle(id) {
    setChecked((previous) => ({ ...previous, [id]: !previous[id] }));
  }

  function buildMarkdown() {
    const lines = [`# Checklist: ${data.document}`, ""];

    for (const section of sections) {
      if (!section.items.length) continue;

      lines.push(`## ${section.title}`);
      section.items.forEach((item, index) => {
        const id = `${section.key}-${index}`;
        const mark = checked[id] ? "x" : " ";
        lines.push(`- [${mark}] ${item.item} (Page ${item.page})`);
      });
      lines.push("");
    }

    if (total === 0) {
      lines.push("No items were found in this document.", "");
    }

    lines.push("---", data.limitations || "");
    return lines.join("\n");
  }

  async function copyList() {
    try {
      await navigator.clipboard.writeText(buildMarkdown());
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {
      setCopied(false);
    }
  }

  function downloadList() {
    const blob = new Blob([buildMarkdown()], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${String(data.document || "checklist").replace(/\.pdf$/i, "")}-checklist.md`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="bp-checklist">
      <div className="bp-checklist-head">
        <div>
          <strong>{total ? `${done} of ${total} done` : "No items found"}</strong>
          <span>{data.document}</span>
        </div>
        <div className="bp-checklist-actions">
          <button type="button" onClick={copyList} disabled={total === 0}>
            {copied ? "Copied" : "Copy"}
          </button>
          <button type="button" onClick={downloadList} disabled={total === 0}>
            Download
          </button>
        </div>
      </div>

      {sections.map((section) =>
        section.items.length > 0 && (
          <div className="bp-checklist-section" key={section.key}>
            <h4>{section.title}</h4>
            {section.items.map((item, index) => {
              const id = `${section.key}-${index}`;
              return (
                <label className={`bp-checklist-item ${checked[id] ? "is-done" : ""}`} key={id}>
                  <input
                    type="checkbox"
                    checked={!!checked[id]}
                    onChange={() => toggle(id)}
                  />
                  <span>{item.item}</span>
                  <em>Page {item.page}</em>
                </label>
              );
            })}
          </div>
        )
      )}

      {data.limitations && <p className="bp-checklist-note">{data.limitations}</p>}
    </div>
  );
}

export default function App() {
  const [documents, setDocuments] = useState([]);
  const [selectedNames, setSelectedNames] = useState([]);
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [language, setLanguage] = useState("same");
  const [connected, setConnected] = useState(false);
  const [indexReady, setIndexReady] = useState(false);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [reindexing, setReindexing] = useState(false);
  const [error, setError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [dragging, setDragging] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [notice, setNotice] = useState("");
  const fileInputRef = useRef(null);
  const chatEndRef = useRef(null);
  const questionRef = useRef(null);
  const knownNamesRef = useRef([]);

  useEffect(() => {
    loadDocuments();
  }, []);

  // Keep the selection in sync with the library.
  // New PDFs are selected automatically; removed PDFs are dropped.
  useEffect(() => {
    const names = documents.map(getDocumentName);
    setSelectedNames((previous) => {
      const newOnes = names.filter((name) => !knownNamesRef.current.includes(name));
      knownNamesRef.current = names;
      return [...previous.filter((name) => names.includes(name)), ...newOnes];
    });
  }, [documents]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, loading]);

  const allNames = documents.map(getDocumentName);

  async function loadDocuments() {
    try {
      const response = await fetch(`${API}/api/documents`);
      if (!response.ok) throw new Error("Could not load your documents.");
      const data = await response.json();
      const list = getDocumentList(data);
      setDocuments(list);
      setConnected(true);
      setIndexReady(Boolean(data?.index_ready ?? data?.indexed ?? list.length));
    } catch {
      setConnected(false);
      setIndexReady(false);
    }
  }

  function toggleSelected(name) {
    setSelectedNames((previous) =>
      previous.includes(name)
        ? previous.filter((item) => item !== name)
        : [...previous, name]
    );
  }

  function selectAll() {
    setSelectedNames([...allNames]);
  }

  function selectNone() {
    setSelectedNames([]);
  }

  function startNewChat() {
    setMessages([]);
    setQuestion("");
    setError("");
    setNotice("");
    questionRef.current?.focus();
  }

  function chooseFiles(fileList) {
    const pdfs = Array.from(fileList || []).filter(
      (file) => file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf")
    );
    if (!pdfs.length) {
      setError("Please choose PDF files only.");
      return;
    }
    const oversized = pdfs.find((file) => file.size > MAX_FILE_BYTES);
    if (oversized) {
      setError(`${oversized.name} is larger than the 10 MB per-file limit.`);
      return;
    }
    setError("");
    setSelectedFiles((previous) => {
      const combined = [...previous, ...pdfs];
      const unique = new Map(combined.map((file) => [`${file.name}-${file.size}`, file]));
      const files = Array.from(unique.values());
      const totalSize = files.reduce((sum, file) => sum + file.size, 0);
      if (totalSize > MAX_TOTAL_BYTES) {
        setError("Your selected files exceed the 30 MB total limit.");
        return previous;
      }
      return files;
    });
  }

  async function uploadFiles() {
    if (!selectedFiles.length || uploading) return;
    setUploading(true);
    setError("");
    setNotice("");
    try {
      const formData = new FormData();
      selectedFiles.forEach((file) => formData.append("files", file));
      const response = await fetch(`${API}/api/upload`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data?.detail || data?.message || "PDF upload failed.");
      }
      setNotice(data?.message || "Your PDFs were uploaded successfully.");
      setSelectedFiles([]);
      if (fileInputRef.current) fileInputRef.current.value = "";
      await loadDocuments();
      setIndexReady(true);
    } catch (err) {
      setError(err.message || "Could not upload the PDFs. Check that the backend is running.");
    } finally {
      setUploading(false);
    }
  }

  async function removeDocument(name) {
    if (!window.confirm(`Remove "${name}" from your library? This cannot be undone.`)) return;

    setError("");
    setNotice("");
    try {
      const response = await fetch(`${API}/api/documents/${encodeURIComponent(name)}`, {
        method: "DELETE",
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data?.detail || data?.message || "Could not remove the PDF.");
      }
      setNotice(data?.message || `${name} was removed.`);
      await loadDocuments();
    } catch (err) {
      setError(err.message || "Could not remove the PDF.");
    }
  }

  async function rebuildIndex() {
    if (reindexing) return;
    setReindexing(true);
    setError("");
    setNotice("");
    try {
      const response = await fetch(`${API}/api/reindex`, { method: "POST" });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data?.detail || data?.message || "Could not rebuild the index.");
      }
      setNotice(data?.message || "Your document index has been rebuilt.");
      setIndexReady(true);
      await loadDocuments();
    } catch (err) {
      setError(err.message || "Indexing failed. Check the backend logs.");
    } finally {
      setReindexing(false);
    }
  }

  async function sendMessage(text = question) {
    const cleanQuestion = text.trim();
    if (!cleanQuestion || loading) return;

    if (!connected) {
      setError("The backend is offline. Start the FastAPI server and try again.");
      return;
    }

    if (selectedNames.length === 0) {
      setError("Select at least one PDF in your document library first.");
      return;
    }

    setError("");
    setNotice("");
    setQuestion("");
    setMessages((previous) => [
      ...previous,
      { id: `${Date.now()}-user`, role: "user", content: cleanQuestion },
    ]);
    setLoading(true);

    try {
      const response = await fetch(`${API}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: cleanQuestion,
          language: languageParam(language),
          filenames: [...selectedNames],
        }),
      });

      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data?.detail || data?.message || "Could not generate an answer.");
      }

      setMessages((previous) => [
        ...previous,
        {
          id: `${Date.now()}-assistant`,
          role: "assistant",
          content: formatAnswer(data?.answer ?? data?.response ?? data?.result),
          sources: Array.isArray(data?.sources) ? data.sources : [],
        },
      ]);
    } catch (err) {
      setMessages((previous) => [
        ...previous,
        {
          id: `${Date.now()}-error`,
          role: "assistant",
          content: `I couldn't complete that request. ${err.message || "Please try again."}`,
          failed: true,
          sources: [],
        },
      ]);
    } finally {
      setLoading(false);
      questionRef.current?.focus();
    }
  }

  // Runs a suggestion card against the selected PDFs.
  // Each selected PDF gets its own labelled result.
  async function runSuggestion(suggestion) {
    if (loading) return;

    if (!connected) {
      setError("The backend is offline. Start the FastAPI server and try again.");
      return;
    }

    if (selectedNames.length === 0) {
      setError("Select at least one PDF in your document library first.");
      return;
    }

    const names = [...selectedNames];

    setError("");
    setNotice("");
    setLoading(true);

    const scope = names.length === 1 ? `"${names[0]}"` : `${names.length} selected documents`;
    setMessages((previous) => [
      ...previous,
      {
        id: `${Date.now()}-request`,
        role: "user",
        label: "You",
        content: `${suggestion.title} for ${scope}`,
      },
    ]);

    const headers = { "Content-Type": "application/json" };

    try {
      if (suggestion.kind === "checklist") {
        const response = await fetch(`${API}/api/checklist`, {
          method: "POST",
          headers,
          body: JSON.stringify({
            filenames: names,
            language: languageParam(language),
          }),
        });

        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw new Error(payload?.detail || payload?.message || "Could not build the checklist.");
        }

        const list = Array.isArray(payload?.checklists) ? payload.checklists : [];
        const byName = new Map(list.map((item) => [item.document, item]));

        const replies = names.map((name, index) => {
          const item = byName.get(name);
          return {
            id: `${Date.now()}-${index}-checklist`,
            role: "assistant",
            label: `Checklist for ${name}`,
            checklist: item || null,
            content: item ? "" : "No checklist was returned for this document.",
            sources: [],
          };
        });

        setMessages((previous) => [...previous, ...replies]);
      } else if (suggestion.kind === "summary") {
        const response = await fetch(`${API}/api/summaries`, {
          method: "POST",
          headers,
          body: JSON.stringify({
            filenames: names,
            language: languageParam(language),
          }),
        });

        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw new Error(payload?.detail || payload?.message || "Could not generate summaries.");
        }

        const list = Array.isArray(payload?.summaries) ? payload.summaries : [];
        const byName = new Map(list.map((item) => [item.document, item]));

        const replies = names.map((name, index) => {
          const item = byName.get(name);
          return {
            id: `${Date.now()}-${index}-summary`,
            role: "assistant",
            label: `Summary of ${name}`,
            content: item?.summary || "No summary was returned for this document.",
            sources: Array.isArray(item?.sources) ? item.sources : [],
          };
        });

        setMessages((previous) => [...previous, ...replies]);
      } else {
        for (const [index, name] of names.entries()) {
          const label = `${suggestion.shortLabel} of ${name}`;

          try {
            const response = await fetch(`${API}/api/chat`, {
              method: "POST",
              headers,
              body: JSON.stringify({
                question: `${suggestion.prompt} Answer only from the document "${name}".`,
                language: languageParam(language),
                filenames: [name],
              }),
            });

            const data = await response.json().catch(() => ({}));
            if (!response.ok) {
              throw new Error(data?.detail || data?.message || "Could not generate an answer.");
            }

            setMessages((previous) => [
              ...previous,
              {
                id: `${Date.now()}-${index}-chat`,
                role: "assistant",
                label,
                content: formatAnswer(data?.answer ?? data?.response ?? data?.result),
                sources: Array.isArray(data?.sources) ? data.sources : [],
              },
            ]);
          } catch (err) {
            setMessages((previous) => [
              ...previous,
              {
                id: `${Date.now()}-${index}-error`,
                role: "assistant",
                label,
                content: `I couldn't complete this for ${name}. ${err.message || "Please try again."}`,
                failed: true,
                sources: [],
              },
            ]);
          }
        }
      }
    } catch (err) {
      setMessages((previous) => [
        ...previous,
        {
          id: `${Date.now()}-suggestion-error`,
          role: "assistant",
          label: suggestion.shortLabel,
          content: `I couldn't complete that request. ${err.message || "Please try again."}`,
          failed: true,
          sources: [],
        },
      ]);
    } finally {
      setLoading(false);
      questionRef.current?.focus();
    }
  }

  function handleComposerKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  }

  const canAsk = connected && selectedNames.length > 0 && !loading;

  return (
    <div className="bp-app">
      {sidebarOpen && (
        <aside className="bp-sidebar">
          <div className="bp-brand">
            <div className="bp-brand-mark"><Sparkles size={21} strokeWidth={2.1} /></div>
            <div className="bp-brand-copy">
              <span className="bp-brand-name">bhashaproof<span>®</span></span>
              <span className="bp-brand-caption">YOUR KNOWLEDGE, REVEALED</span>
            </div>
            <button className="bp-icon-button bp-sidebar-close" onClick={() => setSidebarOpen(false)} title="Collapse sidebar" aria-label="Collapse sidebar">
              <PanelLeftClose size={18} />
            </button>
          </div>

          <button className="bp-new-chat" onClick={startNewChat}>
            <Plus size={18} />
            <span>New conversation</span>
            <span className="bp-shortcut">⌘ K</span>
          </button>

          <div className="bp-sidebar-label">YOUR WORKSPACE</div>
          <button className="bp-nav-item bp-nav-active">
            <MessageSquare size={17} />
            <span>Document assistant</span>
            <span className="bp-active-dot" />
          </button>

          {/* Document library with upload directly beneath */}
          <div className="bp-docs-heading">
            <span>DOCUMENT LIBRARY</span>
            <span className="bp-count">{documents.length}</span>
          </div>

          <div className="bp-sidebar-upload">
            <div
              className={`bp-mini-drop ${dragging ? "is-dragging" : ""}`}
              onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={(event) => {
                event.preventDefault();
                setDragging(false);
                chooseFiles(event.dataTransfer.files);
              }}
            >
              <UploadCloud size={20} />
              <span>
                Drop PDFs here or{" "}
                <button type="button" className="bp-inline-link" onClick={() => fileInputRef.current?.click()}>
                  browse
                </button>
              </span>
              <small>PDF · 10 MB each · 30 MB total</small>
              <input
                ref={fileInputRef}
                className="bp-file-input"
                type="file"
                accept=".pdf,application/pdf"
                multiple
                onChange={(event) => {
                  chooseFiles(event.target.files);
                  event.target.value = "";
                }}
              />
            </div>

            {selectedFiles.length > 0 && (
              <div className="bp-pending-list">
                {selectedFiles.map((file) => (
                  <div className="bp-pending-item" key={`${file.name}-${file.size}`}>
                    <FileText size={14} />
                    <span title={file.name}>{file.name}</span>
                    <button
                      type="button"
                      aria-label={`Remove ${file.name}`}
                      onClick={() => setSelectedFiles((previous) => previous.filter((item) => item !== file))}
                    >
                      <X size={13} />
                    </button>
                  </div>
                ))}
                <div className="bp-upload-row">
                  <button className="bp-primary-button" onClick={uploadFiles} disabled={uploading}>
                    {uploading ? <LoaderCircle className="bp-spin" size={15} /> : <UploadCloud size={15} />}
                    {uploading ? "Uploading…" : `Upload ${selectedFiles.length}`}
                  </button>
                  <button className="bp-secondary-button" onClick={() => setSelectedFiles([])} disabled={uploading}>
                    Clear
                  </button>
                </div>
              </div>
            )}

            {notice && <div className="bp-message bp-message-success"><Check size={14} />{notice}</div>}
            {error && <div className="bp-message bp-message-error">{error}</div>}
          </div>

          <div className="bp-docs-tools">
            <span className="bp-docs-tools-label">
              {selectedNames.length} of {documents.length} selected
            </span>
            <div className="bp-docs-tools-actions">
              <button type="button" onClick={selectAll} disabled={!documents.length}>All</button>
              <button type="button" onClick={selectNone} disabled={!selectedNames.length}>None</button>
              <button
                type="button"
                onClick={rebuildIndex}
                disabled={reindexing || uploading || documents.length === 0}
                title="Rebuild index"
              >
                {reindexing ? <LoaderCircle className="bp-spin" size={12} /> : <RefreshCw size={12} />}
              </button>
            </div>
          </div>

          <div className="bp-document-list">
            {documents.length ? documents.map((doc, index) => {
              const name = getDocumentName(doc);
              const checked = selectedNames.includes(name);
              return (
                <div className={`bp-document ${checked ? "is-selected" : ""}`} key={`${name}-${index}`}>
                  <input
                    type="checkbox"
                    className="bp-doc-checkbox"
                    checked={checked}
                    onChange={() => toggleSelected(name)}
                    aria-label={`Use ${name}`}
                  />
                  <div className="bp-document-icon"><FileText size={16} /></div>
                  <div className="bp-document-info">
                    <span className="bp-document-name" title={name}>{name}</span>
                    <span className="bp-document-meta">{getDocumentMeta(doc)}</span>
                  </div>
                  <button
                    type="button"
                    className="bp-doc-remove"
                    onClick={() => removeDocument(name)}
                    aria-label={`Remove ${name}`}
                    title="Remove PDF"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              );
            }) : (
              <div className="bp-empty-docs">
                <div className="bp-empty-docs-icon"><BookOpen size={19} /></div>
                <p>Your library starts here</p>
                <span>Upload a PDF to build your knowledge space.</span>
              </div>
            )}
          </div>

          <div className="bp-sidebar-spacer" />

          <div className="bp-sidebar-bottom">
            <div className="bp-status-card">
              <span className={`bp-status-indicator ${connected ? "is-online" : ""}`} />
              <div>
                <span className="bp-status-title">{connected ? "System connected" : "Connecting to backend"}</span>
                <span className="bp-status-subtitle">
                  {connected
                    ? (indexReady ? "Your document index is ready" : "Upload a PDF to get started")
                    : "Check your local API server"}
                </span>
              </div>
            </div>
          </div>
        </aside>
      )}

      <main className="bp-main">
        <header className="bp-topbar">
          <div className="bp-topbar-left">
            {!sidebarOpen && (
              <button className="bp-icon-button" onClick={() => setSidebarOpen(true)} aria-label="Open sidebar" title="Open sidebar">
                <PanelLeftOpen size={19} />
              </button>
            )}
            <div className="bp-breadcrumb">
              <span>Workspace</span>
              <span className="bp-breadcrumb-slash">/</span>
              <strong>Document assistant</strong>
            </div>
          </div>
          <div className="bp-topbar-right">
            <span className={`bp-connection-pill ${connected ? "connected" : ""}`}>
              <span className="bp-status-indicator" />
              {connected ? "All systems normal" : "Backend offline"}
            </span>
            <div className="bp-avatar">B</div>
          </div>
        </header>

        <div className={`bp-content ${messages.length ? "bp-content-chatting" : ""}`}>
          {messages.length === 0 ? (
            <section className="bp-welcome">
              <div className="bp-section-heading">
                <div>
                  <span className="bp-section-kicker">A GOOD PLACE TO START</span>
                  <h2>What would you like to know?</h2>
                </div>
                <span className="bp-section-caption">
                  {selectedNames.length
                    ? `Using ${selectedNames.length} selected ${selectedNames.length === 1 ? "PDF" : "PDFs"}`
                    : "Select a PDF in your library to begin"}
                </span>
              </div>

              <div className="bp-suggestion-grid">
                {SUGGESTIONS.map((suggestion) => {
                  const Icon = suggestion.icon;
                  return (
                    <button
                      className="bp-suggestion-card"
                      key={suggestion.title}
                      onClick={() => runSuggestion(suggestion)}
                      disabled={!canAsk}
                    >
                      <span className={`bp-suggestion-icon ${suggestion.tone}`}><Icon size={19} /></span>
                      <span className="bp-suggestion-title">{suggestion.title}</span>
                      <span className="bp-suggestion-prompt">{suggestion.prompt}</span>
                      <span className="bp-suggestion-arrow"><ArrowUp size={15} /></span>
                    </button>
                  );
                })}
              </div>

              <div className="bp-welcome-intro">
                <div className="bp-eyebrow">
                  <span className="bp-eyebrow-icon"><Sparkles size={14} /></span>
                  YOUR MULTILINGUAL KNOWLEDGE SPACE
                </div>

                <h1>Knowledge has no<br /><span>language barrier.</span></h1>
                <p className="bp-welcome-subtitle">
                  Your documents hold the answers. Ask in your language,
                  explore the details, and see exactly where every answer comes from.
                </p>

                <div className="bp-feature-strip">
                  <div className="bp-feature">
                    <div className="bp-feature-icon feature-purple"><BookOpen size={18} /></div>
                    <div><strong>Document-grounded</strong><span>Answers from your PDFs</span></div>
                  </div>
                  <div className="bp-feature">
                    <div className="bp-feature-icon feature-peach"><Languages size={18} /></div>
                    <div><strong>Multilingual</strong><span>Ask in your own words</span></div>
                  </div>
                  <div className="bp-feature">
                    <div className="bp-feature-icon feature-green"><Check size={18} /></div>
                    <div><strong>Traceable answers</strong><span>Sources you can verify</span></div>
                  </div>
                </div>
              </div>
            </section>
          ) : (
            <section className="bp-chat-thread">
              <div className="bp-chat-heading">
                <div className="bp-eyebrow">
                  <span className="bp-eyebrow-icon"><Sparkles size={14} /></span>
                  YOUR DOCUMENT ASSISTANT
                </div>
                <h1>Let’s find your answer.</h1>
                <p>Answers grounded in your documents, with sources you can verify.</p>
              </div>

              {messages.map((message) => (
                <article
                  className={`bp-message-row ${message.role === "user" ? "is-user" : "is-assistant"}`}
                  key={message.id}
                >
                  <div className={`bp-message-avatar ${message.role === "user" ? "user-avatar" : "assistant-avatar"}`}>
                    {message.role === "user" ? "Y" : <Sparkles size={17} />}
                  </div>
                  <div className="bp-message-body">
                    <div className="bp-message-author">
                      {message.label || (message.role === "user" ? "You" : "BhashaProof")}
                      {message.failed && <span className="bp-error-label"> · Could not answer</span>}
                    </div>

                    {message.checklist ? (
                      <ChecklistCard data={message.checklist} />
                    ) : (
                      <div className="bp-message-content">
                        {message.role === "assistant" ? (
                          <AssistantMarkdown>{message.content}</AssistantMarkdown>
                        ) : (
                          message.content
                        )}
                      </div>
                    )}

                    {message.role === "assistant" && message.sources?.length > 0 && (
                      <details className="bp-sources">
                        <summary>
                          <BookOpen size={15} /> Supporting sources <span>{message.sources.length}</span>
                          <ChevronDown size={15} />
                        </summary>
                        <div className="bp-source-list">
                          {message.sources.map((source, index) => {
                            const sourceName = source.document || source.source || source.filename || "Source document";
                            const page = source.page ?? source.page_number;
                            const passage = source.text || source.content || source.chunk || "";
                            return (
                              <div className="bp-source-card" key={`${sourceName}-${page ?? "page"}-${index}`}>
                                <div className="bp-source-card-heading">
                                  <FileText size={15} />
                                  <strong>{sourceName}</strong>
                                  {page != null && <span>Page {page}</span>}
                                </div>
                                {passage && <p>{passage}</p>}
                              </div>
                            );
                          })}
                        </div>
                      </details>
                    )}
                  </div>
                </article>
              ))}

              {loading && (
                <div className="bp-message-row is-assistant">
                  <div className="bp-message-avatar assistant-avatar"><Sparkles size={17} /></div>
                  <div className="bp-message-body">
                    <div className="bp-message-author">BhashaProof</div>
                    <div className="bp-thinking"><span /><span /><span /> Reading your documents…</div>
                  </div>
                </div>
              )}
              <div ref={chatEndRef} />
            </section>
          )}
        </div>

        <footer className="bp-composer-area">
          <div className="bp-context-line">
            {selectedNames.length
              ? `Answering from ${selectedNames.length === 1 ? selectedNames[0] : `${selectedNames.length} selected PDFs`}`
              : "No PDF selected. Choose one in your document library."}
          </div>

          <form className="bp-composer" onSubmit={(event) => { event.preventDefault(); sendMessage(); }}>
            <textarea
              ref={questionRef}
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={handleComposerKeyDown}
              placeholder={selectedNames.length ? "Ask a question about your selected documents…" : "Select a PDF to start asking questions…"}
              rows={2}
              aria-label="Ask a question about your documents"
            />
            <div className="bp-composer-bottom">
              <label className="bp-language-select">
                <Languages size={15} />
                <select value={language} onChange={(event) => setLanguage(event.target.value)} aria-label="Answer language">
                  {LANGUAGES.map((item) => (
                    <option key={item.value} value={item.value}>{item.label}</option>
                  ))}
                </select>
                <ChevronDown size={14} />
              </label>
              <div className="bp-composer-right">
                <span className="bp-enter-hint">↵ <span>to send</span></span>
                <button
                  className="bp-send-button"
                  type="submit"
                  disabled={!question.trim() || loading || selectedNames.length === 0}
                  aria-label="Send message"
                >
                  {loading ? <LoaderCircle className="bp-spin" size={17} /> : <ArrowUp size={19} />}
                </button>
              </div>
            </div>
          </form>
          <div className="bp-disclaimer">
            <span className="bp-disclaimer-icon"><Check size={11} /></span>
            BhashaProof can make mistakes. Verify important information against the original source.
          </div>
        </footer>
      </main>
    </div>
  );
}
