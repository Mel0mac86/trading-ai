/* AI Trading PWA — chat diretta con Claude (Anthropic Messages API).
 *
 * Privacy: la chiave API e la cronologia restano SOLO in localStorage su questo
 * dispositivo. Le richieste vanno direttamente da browser -> api.anthropic.com
 * (header anthropic-dangerous-direct-browser-access). Nessun server intermedio,
 * nessuna chiave nel repository pubblico.
 */
"use strict";

const LS = {
  key: "ait_api_key",
  model: "ait_model",
  history: "ait_history",
};
const API_URL = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION = "2023-06-01"; // header anthropic-version (baseline stabile)

// --- Contesto della piattaforma: l'AI "sa" cos'e' trading-ai e i suoi risultati.
const SYSTEM_PROMPT = `Sei "AI Trading", l'assistente della piattaforma open-source **trading-ai**
(repo GitHub Mel0mac86/trading-ai). Rispondi in italiano, in modo chiaro e onesto.
Parli con il proprietario della piattaforma, che non e' un programmatore esperto.

COS'E' LA PIATTAFORMA
- Obiettivo: NON predire il mercato con certezza, ma SCOPRIRE automaticamente
  pattern e strategie statisticamente robuste, con difese rigorose contro
  overfitting e data-leakage.
- 9 moduli: 1) Data Engine, 2) Feature Engineering (ATR, RSI, MACD, ADX, EMA,
  VWAP, Bollinger, candlestick, struttura di mercato FVG/BOS/CHoCH, swing, S/R),
  3) Pattern Discovery (clustering KMeans non supervisionato + validazione OOS +
  Deflated Sharpe Ratio per correggere il multiple-testing),
  4) Strategy Generator (entry/exit/SL/TP/break-even/trailing in unita' di ATR +
  backtester event-driven con costi reali di spread/slippage/commissioni),
  5) Validation (walk-forward, Monte Carlo, robustezza, sensitivity),
  6) EA Generator (export Expert Advisor MQL4/MQL5 con il modello embeddato),
  7) AI Feedback (ottimizzazione robusta dei parametri + versioning),
  9) Report (equity, drawdown, Sharpe/Sortino/Calmar).
- Gira da sola: GitHub Actions ogni lunedi' scarica ~17 anni di dati XAUUSD da un
  dataset Kaggle, esegue tutta la pipeline e archivia report + EA come artifact.
  Su Kaggle (12 GB RAM + GPU) si possono fare ricerche piu' pesanti e schedulate.

RISULTATO PRINCIPALE TROVATO (sii sempre onesto su questo)
- Una sola strategia ha superato tutti i filtri su 17 anni di XAUUSD H1:
  **PAT17_SHORT** (cluster 17, direzione short), Deflated Sharpe ~0.997.
  Su 25 pattern candidati, 24 sono stati SCARTATI: e' il comportamento corretto.
- Parametri base: SL 2 ATR, TP 3 ATR, break-even a 1 ATR, trailing 1.5 ATR.
  Risultato base ~+12.8% totale, max drawdown ~-5.6%, profit factor ~1.49.
- Versione ottimizzata (Modulo 7): SL 1.5, TP 3, BE 1, trailing 0 ->
  ~+30% totale, drawdown ~-5%, profit factor ~1.77, ancora robusta.
- ONESTA': e' un edge REALE ma MODESTO (~1.8%/anno base). Lo Sharpe "15" che
  appare e' un artefatto del campionamento rado dell'equity, NON e' reale; i
  numeri veri sono Sortino ~1.74 e Calmar ~0.13. Non promettere guadagni certi.
  M15 e H4 non hanno prodotto nessuna strategia robusta.

COME RISPONDERE
- Spiega in modo semplice, con numeri concreti quando li hai.
- Se ti chiedono di "migliorare l'AI": suggerisci piu' strumenti (EURUSD, GBPUSD,
  US500...), piu' timeframe, piu' cluster, GPU su Kaggle, e il ciclo di feedback
  del Modulo 7. Ricorda sempre il rischio di overfitting.
- Non hai accesso in tempo reale ai file o al mercato: ragioni sulla conoscenza
  della piattaforma e su cio' che l'utente ti riporta. Dillo se serve.
- Mai consigli finanziari garantiti: il trading comporta rischio di perdita.`;

const SUGGESTIONS = [
  "Spiegami PAT17_SHORT in parole semplici",
  "Questa versione e' gia' ottimizzata?",
  "Come posso far migliorare l'AI su Kaggle?",
  "Quali rischi ha questa strategia?",
];

// --- Stato -----------------------------------------------------------------
let messages = loadHistory(); // [{role, content}]
let streaming = false;

// --- DOM -------------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const chatEl = $("chat");
const inputEl = $("input");
const formEl = $("composer");
const sendBtn = $("send");

// --- Init ------------------------------------------------------------------
window.addEventListener("DOMContentLoaded", () => {
  renderAll();
  registerSW();
  wireUI();
  if (!getKey()) openSettings();
});

function wireUI() {
  $("openSettings").onclick = openSettings;
  $("closeSettings").onclick = closeSettings;
  $("saveSettings").onclick = saveSettings;
  $("clearAll").onclick = clearAll;
  $("newChat").onclick = newChat;
  $("settingsModal").onclick = (e) => { if (e.target.id === "settingsModal") closeSettings(); };

  formEl.addEventListener("submit", (e) => { e.preventDefault(); send(); });
  inputEl.addEventListener("input", autoGrow);
  // Invio con Enter (Shift+Enter = a capo) — utile da tastiera, innocuo su mobile.
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  });
}

function autoGrow() {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 140) + "px";
}

// --- localStorage ----------------------------------------------------------
function getKey() { return localStorage.getItem(LS.key) || ""; }
function getModel() { return localStorage.getItem(LS.model) || "claude-opus-4-8"; }
function loadHistory() {
  try { return JSON.parse(localStorage.getItem(LS.history)) || []; }
  catch { return []; }
}
function saveHistory() { localStorage.setItem(LS.history, JSON.stringify(messages)); }

// --- Impostazioni ----------------------------------------------------------
function openSettings() {
  $("apiKey").value = getKey();
  $("model").value = getModel();
  $("settingsModal").classList.add("open");
}
function closeSettings() { $("settingsModal").classList.remove("open"); }
function saveSettings() {
  const k = $("apiKey").value.trim();
  if (k) localStorage.setItem(LS.key, k);
  localStorage.setItem(LS.model, $("model").value);
  closeSettings();
}
function clearAll() {
  if (!confirm("Cancellare la chiave API e tutta la cronologia da questo dispositivo?")) return;
  localStorage.removeItem(LS.key);
  localStorage.removeItem(LS.history);
  messages = [];
  renderAll();
  closeSettings();
}
function newChat() {
  if (streaming) return;
  if (messages.length && !confirm("Iniziare una nuova chat? La conversazione attuale verra' cancellata.")) return;
  messages = [];
  saveHistory();
  renderAll();
}

// --- Rendering -------------------------------------------------------------
function renderAll() {
  chatEl.innerHTML = "";
  if (messages.length === 0) { renderWelcome(); return; }
  for (const m of messages) addBubble(m.role === "user" ? "user" : "ai", m.content);
  scrollDown();
}

function renderWelcome() {
  const w = document.createElement("div");
  w.className = "welcome";
  w.innerHTML = `<h2>Ciao 👋</h2>
    <p>Sono l'AI della tua piattaforma <b>trading-ai</b>. Chiedimi delle strategie
    scoperte, dei risultati, dei rischi o di come migliorare la ricerca.</p>
    <div class="chips"></div>`;
  const chips = w.querySelector(".chips");
  SUGGESTIONS.forEach((s) => {
    const c = document.createElement("button");
    c.className = "chip"; c.textContent = s; c.type = "button";
    c.onclick = () => { inputEl.value = s; autoGrow(); send(); };
    chips.appendChild(c);
  });
  chatEl.appendChild(w);
}

function addBubble(kind, text) {
  const el = document.createElement("div");
  el.className = "msg " + kind;
  if (kind === "ai") el.innerHTML = renderMarkdown(text);
  else el.textContent = text;
  chatEl.appendChild(el);
  return el;
}

// Markdown minimale e sicuro (no HTML grezzo): **bold**, `code`, a capo.
function renderMarkdown(text) {
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  let html = esc(text);
  html = html.replace(/```([\s\S]*?)```/g, (_, c) => `<pre><code>${c.trim()}</code></pre>`);
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  return html.split(/\n{2,}/).map((p) => `<p>${p.replace(/\n/g, "<br>")}</p>`).join("");
}

function scrollDown() { chatEl.scrollTop = chatEl.scrollHeight; }

function setStreaming(on) {
  streaming = on;
  sendBtn.disabled = on;
  sendBtn.textContent = on ? "■" : "➤";
}

// --- Invio + streaming -----------------------------------------------------
async function send() {
  if (streaming) return;
  const text = inputEl.value.trim();
  if (!text) return;

  const key = getKey();
  if (!key) { openSettings(); return; }

  if (messages.length === 0) chatEl.innerHTML = ""; // togli il welcome
  messages.push({ role: "user", content: text });
  addBubble("user", text);
  inputEl.value = ""; autoGrow();
  saveHistory();
  scrollDown();

  setStreaming(true);
  const aiEl = addBubble("ai", "");
  aiEl.innerHTML = '<span class="cursor">▌</span>';
  scrollDown();

  let acc = "";
  try {
    await streamCompletion(key, getModel(), messages, (delta) => {
      acc += delta;
      aiEl.innerHTML = renderMarkdown(acc) + '<span class="cursor">▌</span>';
      scrollDown();
    });
    aiEl.innerHTML = renderMarkdown(acc || "(nessuna risposta)");
    messages.push({ role: "assistant", content: acc });
    saveHistory();
  } catch (err) {
    aiEl.remove();
    addBubble("err", "⚠️ " + (err && err.message ? err.message : String(err)));
    // Togliamo l'ultimo messaggio utente dalla history del modello cosi' puo' riprovare pulito.
    if (messages[messages.length - 1]?.role === "user") { /* lo lasciamo a video, ma non bloccante */ }
  } finally {
    setStreaming(false);
    scrollDown();
  }
}

// Chiama la Messages API in streaming (SSE) e invoca onDelta(text) per ogni pezzo.
async function streamCompletion(apiKey, model, msgs, onDelta) {
  const resp = await fetch(API_URL, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-api-key": apiKey,
      "anthropic-version": ANTHROPIC_VERSION,
      "anthropic-dangerous-direct-browser-access": "true",
    },
    body: JSON.stringify({
      model,
      max_tokens: 2048,
      system: SYSTEM_PROMPT,
      stream: true,
      messages: msgs.map((m) => ({ role: m.role, content: m.content })),
    }),
  });

  if (!resp.ok) {
    let detail = "";
    try { const j = await resp.json(); detail = j?.error?.message || JSON.stringify(j); }
    catch { detail = await resp.text().catch(() => ""); }
    throw new Error(friendlyError(resp.status, detail));
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop(); // ultima riga (forse incompleta) resta nel buffer
    for (const line of lines) {
      const s = line.trim();
      if (!s.startsWith("data:")) continue;
      const payload = s.slice(5).trim();
      if (!payload || payload === "[DONE]") continue;
      let evt;
      try { evt = JSON.parse(payload); } catch { continue; }
      if (evt.type === "content_block_delta" && evt.delta?.type === "text_delta") {
        onDelta(evt.delta.text);
      } else if (evt.type === "error") {
        throw new Error(evt.error?.message || "Errore dallo streaming.");
      }
    }
  }
}

function friendlyError(status, detail) {
  if (status === 401) return "Chiave API non valida. Controllala nelle Impostazioni (⚙︎).";
  if (status === 403) return "Accesso negato (403). La chiave non ha i permessi o l'account non ha crediti.";
  if (status === 429) return "Troppe richieste o credito esaurito (429). Riprova tra poco.";
  if (status === 400 && /credit|balance/i.test(detail)) return "Credito Anthropic insufficiente. Aggiungi crediti sull'account.";
  return `Errore ${status}: ${detail || "richiesta non riuscita"}.`;
}

// --- Service worker --------------------------------------------------------
function registerSW() {
  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("./service-worker.js").catch(() => {});
  }
}
