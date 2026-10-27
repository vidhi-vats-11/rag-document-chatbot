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
    pill.textContent = problems.length ? problems.join(" · ") : `Ready · ${h.llm_provider}`;
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
    ].filter(Boolean).join(" · ");

    list.append(
      el("li", { class: "doc" },
        el("span", { class: "doc-name" }, d.filename, el("span", { class: `badge ${d.status}` }, d.status)),
        el("button", { class: "link", type: "button", title: "Delete document", onclick: () => deleteDocument(d) }, "Delete"),
        el("span", { class: "doc-meta" }, meta),
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

// Friendly names for common Tesseract codes; anything else is shown by its code.
const LANGUAGE_NAMES = {
  eng: "English", hin: "Hindi", mar: "Marathi", ben: "Bengali", guj: "Gujarati", pan: "Punjabi",
  tam: "Tamil", tel: "Telugu", kan: "Kannada", mal: "Malayalam", ori: "Odia", urd: "Urdu",
  san: "Sanskrit", nep: "Nepali", ara: "Arabic", fra: "French", deu: "German", spa: "Spanish",
  por: "Portuguese", ita: "Italian", nld: "Dutch", rus: "Russian", ukr: "Ukrainian", pol: "Polish",
  tur: "Turkish", ell: "Greek", heb: "Hebrew", fas: "Persian", chi_sim: "Chinese (Simplified)",
  chi_tra: "Chinese (Traditional)", jpn: "Japanese", kor: "Korean", tha: "Thai", vie: "Vietnamese",
  ind: "Indonesian", msa: "Malay", swe: "Swedish", nor: "Norwegian", dan: "Danish", fin: "Finnish",
};

async function loadOcrLanguages() {
  const select = $("ocr-language");
  let data;
  try {
    data = await api("/documents/ocr-languages");
  } catch {
    return; // keep the default "English" option
  }
  const named = [];
  const other = [];
  for (const code of data.languages) {
    if (code === "eng") continue;
    // Pair every language with English: most documents mix in some English words.
    const option = { value: `${code}+eng`, label: LANGUAGE_NAMES[code] ? `${LANGUAGE_NAMES[code]} + English` : `${code} + English` };
    (LANGUAGE_NAMES[code] ? named : other).push(option);
  }
  named.sort((a, b) => a.label.localeCompare(b.label));
  other.sort((a, b) => a.label.localeCompare(b.label));
  select.replaceChildren(el("option", { value: "" }, "English"));
  if (named.length) select.append(el("optgroup", { label: "Common" }, ...named.map((o) => el("option", { value: o.value }, o.label))));
  if (other.length) select.append(el("optgroup", { label: "All other languages (by code)" }, ...other.map((o) => el("option", { value: o.value }, o.label))));
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
    const language = $("ocr-language").value;
    if (language) form.append("ocr_language", language);
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

function addMessage(kind, text) {
  const node = el("div", { class: `msg ${kind}` }, text);
  $("conversation").append(node);
  node.scrollIntoView({ behavior: "smooth", block: "end" });
  return node;
}

function renderSources(sources) {
  if (!sources.length) return null;
  const box = el("div", { class: "sources" }, el("h3", {}, "Sources"));
  sources.forEach((s) => {
    const where = s.page_number != null ? `${s.filename}, page ${s.page_number}` : s.filename;
    box.append(
      el("details", {},
        el("summary", {}, `${where} (relevance ${s.score.toFixed(2)})`),
        el("blockquote", {}, s.snippet),
      ),
    );
  });
  return box;
}

function showBotAnswer(node, answer, sources) {
  node.className = "msg bot";
  node.textContent = answer;
  const box = renderSources(sources || []);
  if (box) node.append(box);
}

function setupChat() {
  const form = $("ask-form");
  const question = $("question");
  const button = $("ask-btn");

  turns = loadTurns();
  for (const t of turns) {
    addMessage("user", t.question);
    showBotAnswer(addMessage("bot", ""), t.answer, t.sources);
  }

  $("new-chat-btn").addEventListener("click", () => {
    turns = [];
    saveTurns();
    $("conversation").replaceChildren();
    question.focus();
  });

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
    button.disabled = true;
    const pending = addMessage("bot thinking", "Searching your documents…");

    const payload = {
      question: q,
      history: turns.slice(-SEND_TURNS).map((t) => ({ question: t.question, answer: t.answer.slice(0, 10000) })),
    };
    if ($("scope").value) payload.document_id = $("scope").value;

    try {
      const res = await api("/chat/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      showBotAnswer(pending, res.answer, res.sources);
      turns.push({ question: q, answer: res.answer, sources: res.sources });
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

// ---------------------------------------------------------------- init

setupUpload();
setupChat();
loadHealth();
loadDocuments();
loadOcrLanguages();
