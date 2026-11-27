// Minimal front-end for the RAG chatbot. No framework: fetch + DOM.
// All text from the server is inserted with textContent (never innerHTML),
// so document content cannot inject HTML or scripts into the page.
"use strict";

const API = "/api/v1";
const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------- helpers

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const child of children) {
    if (child == null) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

async function api(path, options = {}) {
  let res;
  try {
    res = await fetch(API + path, options);
  } catch {
    throw new Error("Could not reach the server.");
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.message || `Request failed (${res.status}).`);
  return body;
}

function setStatus(text, kind = "info") {
  const s = $("upload-status");
  s.textContent = text;
  s.className = "status " + (text ? kind : "");
}

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

// ---------------------------------------------------------------- health

async function loadHealth() {
  const pill = $("health");
  try {
    const res = await fetch("/health");
    const h = await res.json();
    const problems = [];
    if (!h.llm_configured) problems.push("LLM not configured");
    if (!h.ocr_available) problems.push("OCR unavailable");
    pill.textContent = problems.length ? problems.join(" · ") : "Ready";
    pill.className = "pill " + (problems.length ? "warn" : "ok");
  } catch {
    pill.textContent = "Server unreachable";
    pill.className = "pill warn";
  }
}

// ---------------------------------------------------------------- documents

// Uploads are indexed in the background. While any document is "processing" we
// re-check the list every few seconds, and say so when one of ours finishes.
const POLL_MS = 2500;
const watching = new Set(); // document_ids uploaded in this tab that are still processing
let pollTimer = null;

function reportFinished(d) {
  if (d.status === "failed") {
    setStatus(`Indexing "${d.filename}" failed: ${d.error_message}`, "err");
  } else {
    const extra = d.warnings.length ? ` Note: ${d.warnings.join(" ")}` : "";
    setStatus(`"${d.filename}" is ready: indexed into ${d.chunk_count} chunks.${extra}`, "ok");
  }
}

const FILE_LABELS = { pdf: "PDF", docx: "DOC", image: "IMG" };

// Built with DOM calls (not innerHTML), like everything else on this page.
function trashIcon() {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  for (const [k, v] of Object.entries({ viewBox: "0 0 24 24", width: "16", height: "16", fill: "none", stroke: "currentColor", "stroke-width": "2", "stroke-linecap": "round", "stroke-linejoin": "round" })) {
    svg.setAttribute(k, v);
  }
  for (const d of ["M3 6h18", "M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2", "M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"]) {
    const path = document.createElementNS(ns, "path");
    path.setAttribute("d", d);
    svg.append(path);
  }
  return svg;
}

async function loadDocuments() {
  let data;
  try {
    data = await api("/documents");
  } catch (e) {
    setStatus(e.message, "err");
    return;
  }
  const list = $("doc-list");
  list.replaceChildren();
  $("doc-empty").hidden = data.total > 0;
  $("doc-count").textContent = data.total;

  const scope = $("scope");
  const previous = scope.value;
  scope.replaceChildren(el("option", { value: "" }, "All indexed documents"));

  for (const d of data.documents) {
    const meta = [
      d.file_kind.toUpperCase(),
      formatSize(d.size_bytes),
      d.page_count ? `${d.page_count} page${d.page_count > 1 ? "s" : ""}` : null,
      d.chunk_count ? `${d.chunk_count} chunk${d.chunk_count > 1 ? "s" : ""}` : null,
      d.ocr_used ? (d.ocr_language && d.ocr_language !== "eng" ? `OCR (${d.ocr_language})` : "OCR") : null,
      d.embedding_model ? `embedding: ${shortModel(d.embedding_model)}` : null,
    ].filter(Boolean).join(" · ");

    list.append(
      el("li", { class: "doc" },
        el("span", { class: `file-icon ${d.file_kind}`, "aria-hidden": "true" }, FILE_LABELS[d.file_kind] || "FILE"),
        el("span", { class: "doc-name", title: d.filename }, d.filename),
        el("button", { class: "link", type: "button", title: `Delete "${d.filename}"`, "aria-label": `Delete ${d.filename}`, onclick: () => deleteDocument(d) }, trashIcon()),
        el("span", { class: "doc-meta" }, el("span", { class: `badge ${d.status}` }, d.status), " ", meta),
        d.error_message ? el("span", { class: "doc-error" }, d.error_message) : null,
      ),
    );
    if (d.status === "indexed") {
      scope.append(el("option", { value: d.document_id }, d.filename));
    }
  }
  if ([...scope.options].some((o) => o.value === previous)) scope.value = previous;

  for (const d of data.documents) {
    if (watching.has(d.document_id) && d.status !== "processing") {
      watching.delete(d.document_id);
      reportFinished(d);
      loadHealth();
    }
  }
  clearTimeout(pollTimer);
  if (data.documents.some((d) => d.status === "processing")) {
    pollTimer = setTimeout(loadDocuments, POLL_MS);
  }
}

async function deleteDocument(doc) {
  const ok = window.confirm(`Delete "${doc.filename}"? This removes its file and search index.`);
  if (!ok) return;
  try {
    await api(`/documents/${doc.document_id}`, { method: "DELETE" });
    setStatus(`Deleted "${doc.filename}".`, "ok");
  } catch (e) {
    setStatus(e.message, "err");
  }
  loadDocuments();
  loadHealth();
}

// ---------------------------------------------------------------- model choices

// "sentence-transformers/all-MiniLM-L6-v2" -> "all-MiniLM-L6-v2"
const shortModel = (name) => name.split("/").pop();

async function loadModels() {
  let data;
  try {
    data = await api("/models");
  } catch {
    return; // keep the "Default" options; the server then uses its defaults
  }
  const fill = (select, names, defaultName) => {
    select.replaceChildren(
      ...names.map((n) => el("option", { value: n }, n === defaultName ? `${shortModel(n)} (default)` : shortModel(n))),
    );
    select.value = defaultName;
  };
  fill($("embedding-model"), data.embedding_models, data.default_embedding_model);
  fill($("llm-model"), data.llm_models, data.default_llm_model);
}

function setupUpload() {
  const input = $("file-input");
  const zone = $("dropzone");
  const button = $("upload-btn");

  const showChosen = () => {
    const f = input.files[0];
    button.disabled = !f;
    zone.classList.toggle("has-file", !!f);
    zone.querySelector("strong").textContent = f ? f.name : "Choose a file";
  };
  input.addEventListener("change", showChosen);

  ["dragenter", "dragover"].forEach((t) =>
    zone.addEventListener(t, (e) => { e.preventDefault(); zone.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((t) =>
    zone.addEventListener(t, (e) => { e.preventDefault(); zone.classList.remove("drag"); }));
  zone.addEventListener("drop", (e) => {
    if (e.dataTransfer.files.length) {
      input.files = e.dataTransfer.files;
      showChosen();
    }
  });

  $("upload-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const file = input.files[0];
    if (!file) return;

    const form = new FormData();
    form.append("file", file);
    const embeddingModel = $("embedding-model").value;
    if (embeddingModel) form.append("embedding_model", embeddingModel);
    button.disabled = true;
    setStatus(`Uploading "${file.name}"…`, "info");
    try {
      const res = await api("/documents/upload", { method: "POST", body: form });
      const d = res.document;
      if (d.status === "processing") {
        watching.add(d.document_id);
        setStatus(
          res.already_indexed
            ? `"${d.filename}" is already being indexed. It will show "indexed" when ready.`
            : `Uploaded "${d.filename}". Indexing in the background — you can keep chatting meanwhile.`,
          "info",
        );
      } else if (res.already_indexed) {
        setStatus(`"${d.filename}" was already indexed — nothing to redo.`, "ok");
      } else {
        reportFinished(d);
      }
      input.value = "";
      showChosen();
    } catch (err) {
      setStatus(err.message, "err");
      button.disabled = false;
    }
    loadDocuments();
    loadHealth();
  });
}

// ---------------------------------------------------------------- chat

// Conversation memory. Each finished turn is kept here and the recent ones are sent
// with the next question, so follow-ups like "what is her email?" make sense.
// It is also saved in this browser (localStorage) so a page refresh keeps the chat.
const STORAGE_KEY = "rag-chat-history";
const MAX_STORED_TURNS = 50;
const SEND_TURNS = 6; // the server also caps this (CHAT_HISTORY_TURNS)
let turns = []; // [{ question, answer, sources }]

function saveTurns() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(turns.slice(-MAX_STORED_TURNS)));
  } catch {
    // Private mode or storage full: memory still works until the page is closed.
  }
}

function loadTurns() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    return Array.isArray(saved) ? saved.filter((t) => typeof t?.question === "string" && typeof t?.answer === "string") : [];
  } catch {
    return [];
  }
}

// The welcome screen (with example questions) shows only while the chat is empty.
function updateWelcome() {
  $("welcome").hidden = $("conversation").children.length > 0;
}

function addMessage(kind, text) {
  const node = el("div", { class: `msg ${kind}` }, text);
  $("conversation").append(node);
  updateWelcome();
  node.scrollIntoView({ behavior: "smooth", block: "end" });
  return node;
}

// Let the question box grow with its text (up to the CSS max-height).
function autoGrow(textarea) {
  textarea.style.height = "auto";
  textarea.style.height = `${textarea.scrollHeight}px`;
}

function renderSources(sources) {
  if (!sources.length) return null;
  const box = el("div", { class: "sources" }, el("h3", {}, "Sources"));
  sources.forEach((s) => {
    const where = s.page_number != null ? `${s.filename}, page ${s.page_number}` : s.filename;
    box.append(
      el("details", {},
        // Re-ranking happens on the server; show its score when there is one.
        el("summary", {}, `${where} (relevance ${(s.rerank_score ?? s.score).toFixed(2)})`),
        el("blockquote", {}, s.snippet),
      ),
    );
  });
  return box;
}

function showBotAnswer(node, answer, sources, meta) {
  node.className = "msg bot";
  node.textContent = answer;
  const box = renderSources(sources || []);
  if (box) node.append(box);
  if (meta?.cached) {
    // Reused an earlier answer to the same question: instant, and no AI request was used.
    const by = meta.llm_model ? ` · originally by ${meta.llm_model}` : "";
    node.append(el("div", { class: "answer-meta cached" }, `⚡ From cache: no AI request used${by}`));
  } else if (meta?.llm_model) {
    node.append(el("div", { class: "answer-meta" }, `Answered by ${meta.llm_model}`));
  }
}

function setupChat() {
  const form = $("ask-form");
  const question = $("question");
  const button = $("ask-btn");

  turns = loadTurns();
  for (const t of turns) {
    addMessage("user", t.question);
    showBotAnswer(addMessage("bot", ""), t.answer, t.sources, t.meta);
  }

  updateWelcome();

  $("new-chat-btn").addEventListener("click", () => {
    turns = [];
    saveTurns();
    $("conversation").replaceChildren();
    updateWelcome();
    question.focus();
  });

  // Example questions: clicking one asks it straight away.
  for (const chip of document.querySelectorAll("#suggestions .chip")) {
    chip.addEventListener("click", () => {
      question.value = chip.dataset.q;
      form.requestSubmit();
    });
  }

  question.addEventListener("input", () => autoGrow(question));

  question.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      form.requestSubmit();
    }
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const q = question.value.trim();
    if (!q) return;

    addMessage("user", q);
    question.value = "";
    autoGrow(question);
    button.disabled = true;
    const pending = addMessage("bot thinking", "Searching your documents");

    const payload = {
      question: q,
      history: turns.slice(-SEND_TURNS).map((t) => ({ question: t.question, answer: t.answer.slice(0, 10000) })),
    };
    if ($("scope").value) payload.document_id = $("scope").value;
    if ($("llm-model").value) payload.llm_model = $("llm-model").value;

    try {
      const res = await api("/chat/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const meta = { llm_model: res.llm_model, reranked: res.reranked, cached: res.cached };
      showBotAnswer(pending, res.answer, res.sources, meta);
      turns.push({ question: q, answer: res.answer, sources: res.sources, meta });
      saveTurns();
    } catch (err) {
      pending.className = "msg bot error";
      pending.textContent = err.message;
    } finally {
      button.disabled = false;
      question.focus();
    }
  });
}

// ---------------------------------------------------------------- theme

// Five choices. "auto" follows the computer's light/dark setting (and changes with it).
// The choice is remembered in this browser; index.html applies it before the page draws.
const THEME_KEY = "rag-theme";

function savedTheme() {
  try {
    return localStorage.getItem(THEME_KEY) || "auto";
  } catch {
    return "auto";
  }
}

function applyTheme(choice) {
  const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  document.documentElement.dataset.theme = choice === "auto" ? (dark ? "dark" : "light") : choice;
  for (const btn of document.querySelectorAll(".theme-btn")) {
    btn.setAttribute("aria-pressed", String(btn.dataset.choice === choice));
  }
}

function setupTheme() {
  for (const btn of document.querySelectorAll(".theme-btn")) {
    btn.addEventListener("click", () => {
      try {
        localStorage.setItem(THEME_KEY, btn.dataset.choice);
      } catch {
        // Private mode: the theme still changes, it just isn't remembered.
      }
      applyTheme(btn.dataset.choice);
    });
  }
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => applyTheme(savedTheme()));
  applyTheme(savedTheme());
}

// ---------------------------------------------------------------- init

setupTheme();
setupUpload();
setupChat();
loadHealth();
loadDocuments();
loadModels();
