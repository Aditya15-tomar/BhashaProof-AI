import { useEffect, useMemo, useState } from "react";
import {
  FileText,
  Sparkles,
  LoaderCircle,
  AlertCircle,
  CheckCircle2,
  BookOpen,
  ListChecks,
  Languages,
  ChevronDown,
  Hash,
  ShieldAlert,
  CalendarDays,
  Files,
  ArrowUpRight,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./SummaryPanel.css";

const API = (
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"
).replace(/\/$/, "");

const LANGUAGE_OPTIONS = [
  { value: "same", label: "Same as selected language" },
  { value: "English", label: "English" },
  { value: "Hindi", label: "हिन्दी (Hindi)" },
  { value: "Hinglish", label: "Hinglish" },
  { value: "Bengali", label: "বাংলা (Bengali)" },
  { value: "Tamil", label: "தமிழ் (Tamil)" },
  { value: "Telugu", label: "తెలుగు (Telugu)" },
  { value: "Marathi", label: "मराठी (Marathi)" },
  { value: "Gujarati", label: "ગુજરાતી (Gujarati)" },
];

function MarkdownContent({ children }) {
  const content =
    typeof children === "string" && children.trim()
      ? children
      : "No content was provided.";

  return (
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
      {content}
    </ReactMarkdown>
  );
}

function getFilename(document) {
  if (typeof document === "string") return document;

  return (
    document?.filename ||
    document?.name ||
    document?.document ||
    document?.source ||
    ""
  );
}

function formatDate(value) {
  if (!value) return "";

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) return "";

  return date.toLocaleString();
}

function normalizeError(payload, fallback) {
  if (typeof payload?.detail === "string") return payload.detail;
  if (typeof payload?.message === "string") return payload.message;
  return fallback;
}

function cleanText(value) {
  if (typeof value === "string") return value.trim();
  if (typeof value === "number") return String(value);
  return "";
}

function getPageReferences(text) {
  if (!text) return [];

  const matches = [
    ...text.matchAll(
      /\[\s*Pages?\s+(\d+(?:\s*[-,]\s*\d+)*)\s*\]/gi
    ),
    ...text.matchAll(/\bPages?\s+(\d+)\b/gi),
  ];

  const pages = [];

  for (const match of matches) {
    const value = match[1];
    const range = value.match(/^(\d+)\s*-\s*(\d+)$/);

    if (range) {
      const first = Number(range[1]);
      const last = Number(range[2]);

      if (last >= first && last - first <= 100) {
        for (let page = first; page <= last; page += 1) {
          pages.push(page);
        }
      }
    } else {
      for (const part of value.split(/[,-]/)) {
        const page = Number(part.trim());

        if (Number.isFinite(page)) {
          pages.push(page);
        }
      }
    }
  }

  return [...new Set(pages)].sort((a, b) => a - b);
}

function stripPageReferences(text) {
  return cleanText(text)
    .replace(
      /\s*\[\s*Pages?\s+\d+(?:\s*[-,]\s*\d+)*\s*\]/gi,
      ""
    )
    .replace(/\s+Pages?\s+\d+\b/gi, "")
    .replace(/\s{2,}/g, " ")
    .trim();
}

function getPoints(item) {
  if (!Array.isArray(item?.important_points)) return [];

  return item.important_points
    .map((point) => {
      if (typeof point === "string") {
        return {
          text: stripPageReferences(point),
          pages: getPageReferences(point),
        };
      }

      if (point && typeof point === "object") {
        const text = cleanText(
          point.text || point.point || point.description
        );

        return {
          text: stripPageReferences(text),
          pages: Array.isArray(point.pages)
            ? point.pages.map(Number).filter(Number.isFinite)
            : getPageReferences(text),
        };
      }

      return null;
    })
    .filter((point) => point?.text);
}

function getTerms(item) {
  if (!Array.isArray(item?.key_terms)) return [];

  return item.key_terms
    .map((term) => ({
      term: cleanText(term?.term || term?.name),
      meaning: cleanText(term?.meaning || term?.definition),
    }))
    .filter((term) => term.term || term.meaning);
}

function getEvidencePages(item) {
  const pages = Array.isArray(item?.evidence_pages)
    ? item.evidence_pages
    : Array.isArray(item?.sources)
      ? item.sources.map((source) => source.page)
      : [];

  return [...new Set(pages.map(Number).filter(Number.isFinite))].sort(
    (a, b) => a - b
  );
}

function getSources(item) {
  if (!Array.isArray(item?.sources)) return [];

  const seen = new Set();

  return item.sources.filter((source) => {
    if (!source || typeof source !== "object") return false;

    const key = `${source.document || item.document}-${source.page}`;

    if (seen.has(key)) return false;

    seen.add(key);

    return Boolean(source.page);
  });
}

function getCoverageMessage(item) {
  return (
    cleanText(item?.limitations) ||
    cleanText(item?.coverage_note) ||
    "No additional coverage notes were provided."
  );
}

function normalizeLanguage(language) {
  const value = String(language || "same").trim();

  if (
    ["same", "same as the question", "same as selected language"].includes(
      value.toLowerCase()
    )
  ) {
    return "same";
  }

  return value;
}

export default function SummaryPanel({
  documents = [],
  language = "same",
}) {
  const availableDocuments = useMemo(() => {
    const seen = new Set();

    return documents
      .map(getFilename)
      .filter((filename) => {
        if (!filename || seen.has(filename)) return false;

        seen.add(filename);
        return true;
      });
  }, [documents]);

  const [mode, setMode] = useState("multiple");
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [summaryLanguage, setSummaryLanguage] = useState(
    normalizeLanguage(language)
  );
  const [summaries, setSummaries] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  // Synchronize selected PDFs with the indexed documents.
  useEffect(() => {
    setSelectedFiles((current) => {
      const valid = current.filter((name) =>
        availableDocuments.includes(name)
      );

      if (mode === "all") {
        return [...availableDocuments];
      }

      if (mode === "single") {
        return [valid[0] || availableDocuments[0]].filter(Boolean);
      }

      if (valid.length > 0) {
        return valid;
      }

      return availableDocuments.slice(
        0,
        Math.min(2, availableDocuments.length)
      );
    });
  }, [availableDocuments, mode]);

  // Remove reports for documents that are no longer indexed.
  useEffect(() => {
    setSummaries((current) =>
      current.filter((item) =>
        availableDocuments.includes(item.document)
      )
    );
  }, [availableDocuments]);

  // Keep the report language synchronized with the parent component.
  useEffect(() => {
    setSummaryLanguage(normalizeLanguage(language));
  }, [language]);

  function changeMode(nextMode) {
    setMode(nextMode);
    setError("");
    setNotice("");
    setSummaries([]);

    if (nextMode === "all") {
      setSelectedFiles([...availableDocuments]);
    } else if (nextMode === "single") {
      setSelectedFiles((current) => {
        const first = current.find((name) =>
          availableDocuments.includes(name)
        );

        return [first || availableDocuments[0]].filter(Boolean);
      });
    } else {
      setSelectedFiles((current) => {
        const valid = current.filter((name) =>
          availableDocuments.includes(name)
        );

        if (valid.length > 0) return valid;

        return availableDocuments.slice(
          0,
          Math.min(2, availableDocuments.length)
        );
      });
    }
  }

  function toggleFile(filename) {
    setError("");
    setNotice("");
    setSummaries([]);

    setSelectedFiles((current) => {
      if (mode === "single") {
        return [filename];
      }

      if (current.includes(filename)) {
        return current.filter((name) => name !== filename);
      }

      return [...current, filename];
    });
  }

  async function generateSummaries() {
    if (loading) return;

    if (selectedFiles.length === 0) {
      setError("Select at least one indexed PDF first.");
      return;
    }

    setSummaries([]);
    setError("");
    setNotice("");
    setLoading(true);

    try {
      const response = await fetch(`${API}/api/summaries`, {
        method: "POST",
        cache: "no-store",
        headers: {
          "Content-Type": "application/json",
          "Cache-Control": "no-cache",
        },
        body: JSON.stringify({
          filenames: [...selectedFiles],
          language: summaryLanguage || "same",
        }),
      });

      const payload = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(
          normalizeError(
            payload,
            `Summary request failed with status ${response.status}.`
          )
        );
      }

      if (
        payload?.status !== "success" ||
        !Array.isArray(payload?.summaries)
      ) {
        throw new Error(
          "The backend did not return a valid summary list. Check the backend terminal."
        );
      }

      const resultByName = new Map(
        payload.summaries.map((item) => [item.document, item])
      );

      const orderedResults = selectedFiles
        .map((filename) => resultByName.get(filename))
        .filter(Boolean);

      if (orderedResults.length === 0) {
        throw new Error(
          "No summaries were returned for the selected PDFs. Check the filenames returned by the backend."
        );
      }

      setSummaries(orderedResults);

      const missing = selectedFiles.filter(
        (filename) => !resultByName.has(filename)
      );

      setNotice(
        missing.length
          ? `No summary was returned for: ${missing.join(", ")}`
          : `Generated ${orderedResults.length} document ${
              orderedResults.length === 1 ? "report" : "reports"
            }.`
      );
    } catch (err) {
      console.error("Summary generation failed:", err);

      setSummaries([]);

      setError(
        err?.message ||
          "Could not generate summaries. Check that the backend is running."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="bp-summary-panel">
      <header className="bp-summary-hero">
        <div className="bp-summary-hero-top">
          <div className="bp-summary-brand">
            <div className="bp-summary-brand-icon">
              <Sparkles size={21} />
            </div>

            <div>
              <span className="bp-summary-eyebrow">
                BHASHAProof AI / INTELLIGENCE
              </span>
              <h2>Document intelligence</h2>
            </div>
          </div>

          <span className="bp-summary-status">
            <span className="bp-summary-status-dot" />
            Evidence-based reports
          </span>
        </div>

        <p className="bp-summary-hero-description">
          Turn your PDFs into structured reports with clear findings,
          important terminology, and page-level source references.
        </p>

        <div className="bp-summary-hero-meta">
          <span>
            <Files size={15} />
            {availableDocuments.length} indexed{" "}
            {availableDocuments.length === 1 ? "document" : "documents"}
          </span>

          <span>
            <Languages size={15} />
            Multilingual output
          </span>
        </div>
      </header>

      <div className="bp-summary-workspace">
        <div className="bp-summary-workspace-heading">
          <div>
            <span className="bp-summary-eyebrow">REPORT SETTINGS</span>
            <h3>Build your report</h3>
            <p>
              Choose which documents to analyze and your output language.
            </p>
          </div>
        </div>

        {availableDocuments.length === 0 ? (
          <div className="bp-summary-empty">
            <div className="bp-summary-empty-icon">
              <FileText size={26} />
            </div>
            <strong>No indexed PDFs yet</strong>
            <p>
              Upload your PDF files and finish indexing to generate a report.
            </p>
          </div>
        ) : (
          <>
            <div className="bp-summary-controls">
              <div className="bp-summary-field">
                <label htmlFor="summary-mode">
                  Documents to summarize
                </label>

                <div className="bp-summary-select-wrap">
                  <select
                    id="summary-mode"
                    value={mode}
                    onChange={(event) => changeMode(event.target.value)}
                    disabled={loading}
                  >
                    <option value="single">One PDF</option>
                    <option value="multiple">Selected PDFs</option>
                    <option value="all">All PDFs</option>
                  </select>
                  <ChevronDown size={16} />
                </div>
              </div>

              <div className="bp-summary-field">
                <label htmlFor="summary-language">
                  Report language
                </label>

                <div className="bp-summary-select-wrap">
                  <select
                    id="summary-language"
                    value={summaryLanguage}
                    onChange={(event) => {
                      setSummaryLanguage(event.target.value);
                      setSummaries([]);
                      setNotice("");
                      setError("");
                    }}
                    disabled={loading}
                  >
                    {LANGUAGE_OPTIONS.map((item) => (
                      <option key={item.value} value={item.value}>
                        {item.label}
                      </option>
                    ))}
                  </select>
                  <ChevronDown size={16} />
                </div>
              </div>
            </div>

            <div className="bp-summary-file-list">
              <div className="bp-summary-file-list-heading">
                <span>Available documents</span>
                <span className="bp-summary-selection-count">
                  {selectedFiles.length} selected
                </span>
              </div>

              {availableDocuments.map((filename) => {
                const checked =
                  mode === "all" || selectedFiles.includes(filename);

                return (
                  <label
                    className={`bp-summary-file ${
                      checked ? "is-selected" : ""
                    }`}
                    key={filename}
                  >
                    <input
                      type={mode === "single" ? "radio" : "checkbox"}
                      name="summary-document"
                      checked={checked}
                      disabled={loading || mode === "all"}
                      onChange={() => toggleFile(filename)}
                    />

                    <span className="bp-summary-file-icon">
                      <FileText size={18} />
                    </span>

                    <span
                      className="bp-summary-file-name"
                      title={filename}
                    >
                      {filename}
                    </span>

                    {checked && (
                      <CheckCircle2
                        className="bp-summary-file-check"
                        size={18}
                      />
                    )}
                  </label>
                );
              })}
            </div>

            <button
              className="bp-summary-generate"
              onClick={generateSummaries}
              disabled={loading || selectedFiles.length === 0}
              type="button"
            >
              {loading ? (
                <>
                  <LoaderCircle className="bp-summary-spin" size={18} />
                  Analyzing documents...
                </>
              ) : (
                <>
                  <Sparkles size={18} />
                  Generate intelligence report
                  <ArrowUpRight size={17} />
                </>
              )}
            </button>
          </>
        )}
      </div>

      {error && (
        <div className="bp-summary-message bp-summary-error" role="alert">
          <AlertCircle size={19} />
          <span>{error}</span>
        </div>
      )}

      {notice && (
        <div className="bp-summary-message bp-summary-notice" role="status">
          <CheckCircle2 size={19} />
          <span>{notice}</span>
        </div>
      )}

      {loading && (
        <div className="bp-summary-loading">
          <div className="bp-summary-loading-icon">
            <LoaderCircle className="bp-summary-spin" size={23} />
          </div>

          <div>
            <strong>Reading your documents</strong>
            <p>
              Preparing a separate report for each selected PDF. This may
              take a little time.
            </p>
          </div>
        </div>
      )}

      {summaries.length > 0 && !loading && (
        <div className="bp-summary-results" aria-live="polite">
          <div className="bp-summary-results-heading">
            <div>
              <span className="bp-summary-eyebrow">ANALYSIS COMPLETE</span>
              <h3>Your intelligence reports</h3>
              <p>
                Each report is organized by document. Check its cited pages
                before relying on important claims.
              </p>
            </div>

            <span className="bp-summary-count">
              <Files size={15} />
              {summaries.length}{" "}
              {summaries.length === 1 ? "report" : "reports"}
            </span>
          </div>

          {summaries.map((item, index) => {
            const points = getPoints(item);
            const terms = getTerms(item);
            const pages = getEvidencePages(item);
            const sources = getSources(item);

            return (
              <article
                className="bp-summary-card"
                key={`${item.document}-${item.generation_id || index}`}
              >
                <div className="bp-summary-card-header">
                  <div className="bp-summary-card-identity">
                    <div className="bp-summary-card-icon">
                      <FileText size={22} />
                    </div>

                    <div className="bp-summary-card-title">
                      <span className="bp-summary-document-number">
                        DOCUMENT {String(index + 1).padStart(2, "0")}
                      </span>

                      <h3>{item.document || "Untitled document"}</h3>

                      {item.generated_at && (
                        <span className="bp-summary-generated-at">
                          <CalendarDays size={13} />
                          Generated {formatDate(item.generated_at)}
                        </span>
                      )}
                    </div>
                  </div>

                  <span className="bp-summary-document-badge">
                    <CheckCircle2 size={14} />
                    Analyzed
                  </span>
                </div>

                <div className="bp-summary-card-body">
                  <section className="bp-summary-section bp-summary-overview-section">
                    <div className="bp-summary-section-heading">
                      <span className="bp-summary-section-icon">
                        <BookOpen size={18} />
                      </span>

                      <div>
                        <span className="bp-summary-section-kicker">
                          AT A GLANCE
                        </span>
                        <h4>Executive summary</h4>
                      </div>
                    </div>

                    <div className="bp-summary-overview bp-summary-markdown">
                      <MarkdownContent>
                        {item.summary ||
                          "No executive summary was returned."}
                      </MarkdownContent>
                    </div>
                  </section>

                  <section className="bp-summary-section">
                    <div className="bp-summary-section-heading">
                      <span className="bp-summary-section-icon">
                        <ListChecks size={18} />
                      </span>

                      <div>
                        <span className="bp-summary-section-kicker">
                          KEY TAKEAWAYS
                        </span>
                        <h4>Important findings</h4>
                      </div>

                      {points.length > 0 && (
                        <span className="bp-summary-section-count">
                          {points.length}
                        </span>
                      )}
                    </div>

                    {points.length > 0 ? (
                      <div className="bp-summary-points">
                        {points.map((point, pointIndex) => (
                          <div
                            className="bp-summary-point"
                            key={`${pointIndex}-${point.text}`}
                          >
                            <span className="bp-summary-point-number">
                              {String(pointIndex + 1).padStart(2, "0")}
                            </span>

                            <div className="bp-summary-point-content">
                              <div className="bp-summary-point-markdown bp-summary-markdown">
                                <MarkdownContent>
                                  {point.text}
                                </MarkdownContent>
                              </div>

                              {point.pages.length > 0 && (
                                <div className="bp-summary-inline-pages">
                                  {point.pages.map((page) => (
                                    <span key={page}>Page {page}</span>
                                  ))}
                                </div>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="bp-summary-muted">
                        No important findings were returned.
                      </p>
                    )}
                  </section>

                  {terms.length > 0 && (
                    <section className="bp-summary-section">
                      <div className="bp-summary-section-heading">
                        <span className="bp-summary-section-icon">
                          <Hash size={18} />
                        </span>

                        <div>
                          <span className="bp-summary-section-kicker">
                            TERMINOLOGY
                          </span>
                          <h4>Key terms explained</h4>
                        </div>
                      </div>

                      <div className="bp-summary-terms">
                        {terms.map((term, termIndex) => (
                          <div
                            className="bp-summary-term"
                            key={`${term.term}-${termIndex}`}
                          >
                            <strong>{term.term || "Key term"}</strong>

                            <div className="bp-summary-term-markdown bp-summary-markdown">
                              <MarkdownContent>
                                {term.meaning ||
                                  "No definition was provided."}
                              </MarkdownContent>
                            </div>
                          </div>
                        ))}
                      </div>
                    </section>
                  )}

                  <section className="bp-summary-section">
                    <div className="bp-summary-section-heading">
                      <span className="bp-summary-section-icon">
                        <ShieldAlert size={18} />
                      </span>

                      <div>
                        <span className="bp-summary-section-kicker">
                          TRANSPARENCY
                        </span>
                        <h4>Coverage and limitations</h4>
                      </div>
                    </div>

                    <div className="bp-summary-coverage-box bp-summary-markdown">
                      <MarkdownContent>
                        {getCoverageMessage(item)}
                      </MarkdownContent>

                      {item.coverage_note &&
                        item.limitations &&
                        item.coverage_note !== item.limitations && (
                          <div className="bp-summary-extra-coverage">
                            <MarkdownContent>
                              {item.coverage_note}
                            </MarkdownContent>
                          </div>
                        )}
                    </div>
                  </section>

                  <section className="bp-summary-section bp-summary-sources-section">
                    <div className="bp-summary-section-heading">
                      <span className="bp-summary-section-icon">
                        <Files size={18} />
                      </span>

                      <div>
                        <span className="bp-summary-section-kicker">
                          TRACEABLE EVIDENCE
                        </span>
                        <h4>Source references</h4>
                      </div>
                    </div>

                    {pages.length > 0 ? (
                      <>
                        <div className="bp-summary-source-intro">
                          Evidence pages included in this report
                        </div>

                        <div className="bp-summary-page-list">
                          {pages.map((page) => (
                            <span key={page}>Page {page}</span>
                          ))}
                        </div>
                      </>
                    ) : (
                      <p className="bp-summary-muted">
                        No evidence page list was returned.
                      </p>
                    )}

                    {sources.length > 0 && (
                      <div className="bp-summary-source-list">
                        {sources.map((source, sourceIndex) => (
                          <div
                            className="bp-summary-source-row"
                            key={`${source.document}-${source.page}-${sourceIndex}`}
                          >
                            <FileText size={15} />
                            <span>
                              {source.document || item.document}
                            </span>
                            <span className="bp-summary-source-page">
                              Page {source.page}
                            </span>
                          </div>
                        ))}
                      </div>
                    )}

                    {item.generation_id && (
                      <details className="bp-summary-generation-details">
                        <summary>Generation details</summary>

                        <p>
                          <strong>Generation ID:</strong>{" "}
                          <code>{item.generation_id}</code>
                        </p>

                        {Number.isFinite(item.total_indexed_chunks) && (
                          <p>
                            <strong>Indexed text chunks:</strong>{" "}
                            {item.total_indexed_chunks}
                          </p>
                        )}
                      </details>
                    )}
                  </section>
                </div>
              </article>
            );
          })}
        </div>
      )}
    </section>
  );
}