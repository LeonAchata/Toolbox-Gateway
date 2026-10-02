"use strict";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const store = {
  get(key, fallback) {
    try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage unavailable */ }
  },
};

const state = {
  transport: store.get("tg.transport", "http"),
  model: store.get("tg.model", ""),
  bases: { http: null, ws: null },
  httpSession: null,
  ws: null,
  wsReady: false,
  busy: false,
  turns: [],
  selectedTurn: null,
  liveTurn: null,
};

const $ = (sel) => document.querySelector(sel);
const els = {
  app: $("#app"),
  messages: $("#messages"),
  empty: $("#emptyState"),
  input: $("#input"),
  composer: $("#composer"),
  send: $("#sendBtn"),
  model: $("#modelSelect"),
  status: $("#status"),
  timeline: $("#timeline"),
  traceSummary: $("#traceSummary"),
  toolList: $("#toolList"),
  toolCount: $("#toolCount"),
  statGrid: $("#statGrid"),
  modelTable: $("#modelTable"),
  metricsSince: $("#metricsSince"),
  dialog: $("#settingsDialog"),
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function escapeHtml(text) {
  return String(text ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

// A deliberately small Markdown subset. Input is escaped first, so the only
// HTML that reaches the page is the markup produced here.
function renderMarkdown(source) {
  const blocks = [];
  let text = escapeHtml(source).replace(/```(\w*)\n?([\s\S]*?)```/g, (_, _lang, code) => {
    blocks.push(`<pre><code>${code.replace(/\n$/, "")}</code></pre>`);
    return `\u0000${blocks.length - 1}\u0000`;
  });
  const inline = (s) => s
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");

  return text.split(/\n{2,}/).map((para) => {
    const trimmed = para.trim();
    const block = trimmed.match(/^\u0000(\d+)\u0000$/);
    if (block) return blocks[Number(block[1])];
    const lines = trimmed.split("\n");
    if (lines.every((l) => /^\s*([-*]|\d+\.)\s+/.test(l))) {
      const tag = /^\s*\d+\./.test(lines[0]) ? "ol" : "ul";
      return `<${tag}>${lines.map((l) => `<li>${inline(l.replace(/^\s*([-*]|\d+\.)\s+/, ""))}</li>`).join("")}</${tag}>`;
    }
    return `<p>${inline(trimmed).replace(/\n/g, "<br>")}</p>`;
  }).join("").replace(/\u0000(\d+)\u0000/g, (_, i) => blocks[Number(i)]);
}

function fmtMs(ms) {
  if (ms == null) return "-";
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`;
}

function fmtCost(usd) {
  if (!usd) return "$0";
  if (usd < 0.01) return `$${Number(usd.toPrecision(2))}`;
  return `$${usd.toFixed(4)}`;
}

function fmtInt(n) {
  return new Intl.NumberFormat().format(n ?? 0);
}

function fmtJson(value) {
  if (value && typeof value === "object") {
    const entries = Object.entries(value);
    if (entries.length && entries.every(([, v]) => v === null || typeof v !== "object")) {
      return entries.map(([k, v]) => `${k}: ${typeof v === "string" ? JSON.stringify(v) : v}`).join("\n");
    }
  }
  return JSON.stringify(value, null, 2);
}

function icon(name) {
  return `<svg><use href="#i-${name}"/></svg>`;
}

async function fetchJson(url, options = {}, timeoutMs = 120000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = body.detail ?? body.error?.message ?? `HTTP ${response.status}`;
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return body;
  } catch (error) {
    if (error.name === "AbortError") throw new Error("The request timed out");
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

// ---------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------

async function resolveBases() {
  const overrides = store.get("tg.urls", {});
  const served = location.protocol.startsWith("http");
  let proxied = false;
  if (served && !(overrides.http && overrides.ws)) {
    try {
      await fetchJson("/api/http/health", {}, 2500);
      proxied = true;
    } catch { /* not behind the bundled nginx proxy */ }
  }
  state.bases.http = overrides.http || (proxied ? "/api/http" : "http://localhost:8001");
  state.bases.ws = overrides.ws || (proxied ? "/api/ws" : "http://localhost:8002");
}

const base = () => state.bases[state.transport];

function wsUrl() {
  const url = new URL(`${state.bases.ws.replace(/\/$/, "")}/ws`, location.href);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return url.toString();
}

// ---------------------------------------------------------------------------
// Status, models, tools, metrics
// ---------------------------------------------------------------------------

function setStatus(kind, text) {
  els.status.dataset.state = kind;
  els.status.title = text;
  els.status.querySelector(".status-text").textContent = text;
}

async function refreshHealth() {
  try {
    const health = await fetchJson(`${base()}/health`, {}, 5000);
    const label = state.transport === "ws" ? "WebSocket" : "HTTP";
    if (health.status === "ok") {
      setStatus("ok", state.transport === "ws" && !state.wsReady ? `${label}, connecting` : `${label} agent online`);
    } else {
      const down = Object.entries(health.dependencies || {})
        .filter(([, v]) => v !== "ok").map(([k]) => k.replace("_", " "));
      setStatus("degraded", `Degraded: ${down.join(", ") || "unknown"}`);
    }
  } catch {
    setStatus("down", "Agent unreachable");
  }
}

async function loadModels() {
  try {
    const { models } = await fetchJson(`${base()}/models`, {}, 8000);
    const options = [`<option value="">Gateway default</option>`].concat(
      models.map((m) => {
        const label = `${m.name}  (${m.provider_model})${m.available ? "" : " - not configured"}`;
        return `<option value="${escapeHtml(m.name)}" ${m.available ? "" : "disabled"}>${escapeHtml(label)}</option>`;
      }),
    );
    els.model.innerHTML = options.join("");
    const stillValid = models.some((m) => m.name === state.model && m.available);
    els.model.value = stillValid ? state.model : "";
    const fallback = models.find((m) => m.default);
    if (fallback) els.model.options[0].textContent = `Gateway default (${fallback.name})`;
  } catch {
    /* keep whatever is there; health shows the problem */
  }
}

async function loadTools() {
  try {
    const { tools } = await fetchJson(`${base()}/tools`, {}, 8000);
    els.toolCount.textContent = tools.length;
    els.toolList.innerHTML = tools.map((tool) => {
      const schema = tool.inputSchema || {};
      const required = new Set(schema.required || []);
      const params = Object.entries(schema.properties || {}).map(([name, spec]) =>
        `<span class="param ${required.has(name) ? "req" : ""}" title="${escapeHtml(spec.description || "")}">${escapeHtml(name)}: ${escapeHtml(spec.type || spec.anyOf?.map((t) => t.type).join("|") || "any")}${required.has(name) ? "" : "?"}</span>`,
      ).join("");
      return `<article class="tool-card">
        <header><code>${escapeHtml(tool.name)}</code><span class="cat">${escapeHtml(tool.category || "")}</span></header>
        <p>${escapeHtml(tool.description)}</p>
        <div class="params">${params || '<span class="param">no parameters</span>'}</div>
      </article>`;
    }).join("");
  } catch (error) {
    els.toolList.innerHTML = `<p class="muted">Could not load tools: ${escapeHtml(error.message)}</p>`;
  }
}

async function loadMetrics() {
  try {
    const metrics = await fetchJson(`${base()}/metrics`, {}, 8000);
    const total = metrics.total;
    els.metricsSince.textContent = new Date(metrics.since).toLocaleTimeString();
    const stats = [
      ["Requests", fmtInt(total.requests)],
      ["Cache hit rate", `${Math.round((metrics.cache.hit_rate || 0) * 100)}%`],
      ["Tokens", fmtInt(total.total_tokens)],
      ["Estimated cost", fmtCost(total.cost_usd)],
      ["Latency p50", fmtMs(total.latency_ms.p50)],
      ["Latency p95", fmtMs(total.latency_ms.p95)],
    ];
    els.statGrid.innerHTML = stats.map(([k, v]) =>
      `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
    const rows = Object.entries(metrics.by_model);
    els.modelTable.innerHTML = rows.length
      ? `<thead><tr><th>Model</th><th>Req</th><th>Err</th><th>Tokens</th><th>Cost</th><th>p50</th></tr></thead>
         <tbody>${rows.map(([name, m]) => `<tr><td>${escapeHtml(name)}</td><td>${m.requests}</td><td>${m.errors}</td><td>${fmtInt(m.total_tokens)}</td><td>${fmtCost(m.cost_usd)}</td><td>${fmtMs(m.latency_ms.p50)}</td></tr>`).join("")}</tbody>`
      : "";
  } catch (error) {
    els.statGrid.innerHTML = `<p class="muted">Metrics unavailable: ${escapeHtml(error.message)}</p>`;
    els.modelTable.innerHTML = "";
  }
}

// ---------------------------------------------------------------------------
// Rendering turns
// ---------------------------------------------------------------------------

function newTurn(text) {
  const turn = { id: crypto.randomUUID?.() ?? String(Date.now()), user: text, trace: [], answer: null, usage: null, model: null, error: null };
  state.turns.push(turn);
  els.empty?.remove();

  const user = document.createElement("div");
  user.className = "msg user";
  user.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;

  const assistant = document.createElement("div");
  assistant.className = "msg assistant";
  assistant.dataset.turn = turn.id;
  assistant.innerHTML = `<div class="bubble"><div class="live"><div class="live-step"><span class="spinner"></span>Thinking</div></div></div>`;

  els.messages.append(user, assistant);
  turn.el = assistant;
  selectTurn(turn);
  scrollToBottom();
  return turn;
}

function renderLive(turn) {
  const steps = [];
  for (const event of turn.trace) {
    if (event.type === "llm") {
      const calls = event.tool_calls.length;
      steps.push(`<div class="live-step llm done">${icon("check")}<span><code>${escapeHtml(event.model)}</code> replied${calls ? ` with ${calls} tool call${calls > 1 ? "s" : ""}` : ""} in ${fmtMs(event.latency_ms)}</span></div>`);
    } else if (event.type === "tool") {
      steps.push(`<div class="live-step tool">${icon("wrench")}<span><code>${escapeHtml(event.name)}</code> returned ${escapeHtml(event.result.slice(0, 60))}</span></div>`);
    }
  }
  if (turn.pending) {
    steps.push(`<div class="live-step"><span class="spinner"></span>${escapeHtml(turn.pending)}</div>`);
  }
  turn.el.querySelector(".bubble").innerHTML = `<div class="live">${steps.join("")}</div>`;
  scrollToBottom();
}

function renderAnswer(turn) {
  const bubble = turn.el.querySelector(".bubble");
  turn.el.classList.toggle("error", Boolean(turn.error));
  bubble.innerHTML = turn.error ? escapeHtml(turn.error) : renderMarkdown(turn.answer || "");

  turn.el.querySelector(".msg-meta")?.remove();
  if (turn.usage) {
    const meta = document.createElement("div");
    meta.className = "msg-meta";
    const parts = [
      turn.model ? `<span>${escapeHtml(turn.model)}</span>` : "",
      `<button type="button">${turn.usage.tool_calls} tool call${turn.usage.tool_calls === 1 ? "" : "s"}</button>`,
      `<span>${fmtMs(turn.elapsed ?? turn.usage.latency_ms)}</span>`,
      `<span>${fmtInt(turn.usage.input_tokens + turn.usage.output_tokens)} tokens</span>`,
      `<span>${fmtCost(turn.usage.cost_usd)}</span>`,
    ].filter(Boolean);
    meta.innerHTML = parts.join('<span class="sep" aria-hidden="true">&middot;</span>');
    meta.querySelector("button").addEventListener("click", () => {
      selectTurn(turn);
      showTab("trace");
      openInspector();
    });
    turn.el.append(meta);
  }
  scrollToBottom();
}

function selectTurn(turn) {
  state.selectedTurn = turn;
  document.querySelectorAll(".msg.assistant").forEach((el) =>
    el.classList.toggle("selected", el.dataset.turn === turn.id && state.turns.length > 1));
  renderTrace();
}

function renderTrace() {
  const turn = state.selectedTurn;
  if (!turn) return;
  const llm = turn.trace.filter((e) => e.type === "llm");
  const tools = turn.trace.filter((e) => e.type === "tool");
  const tokens = llm.reduce((n, e) => n + e.usage.input_tokens + e.usage.output_tokens, 0);
  const cost = llm.reduce((n, e) => n + e.cost_usd, 0);
  els.traceSummary.innerHTML = turn.trace.length ? [
    ["LLM calls", llm.length],
    ["Tools", tools.length],
    ["Tokens", fmtInt(tokens)],
    ["Cost", fmtCost(cost)],
  ].map(([k, v]) => `<div class="mini-stat"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("") : "";

  const items = turn.trace.map((event) => {
    if (event.type === "llm") {
      const calls = event.tool_calls.map((c) => `<code>${escapeHtml(c.name)}</code>`).join(", ");
      return `<li class="llm"><span class="node">${icon("cpu")}</span>
        <div class="step-head"><span class="step-title">${escapeHtml(event.model)} <span class="muted">round ${event.round}</span></span><span class="step-time">${fmtMs(event.latency_ms)}</span></div>
        <div class="chips">
          <span class="chip">${escapeHtml(event.provider_model)}</span>
          <span class="chip">${fmtInt(event.usage.input_tokens)} in / ${fmtInt(event.usage.output_tokens)} out</span>
          <span class="chip">${fmtCost(event.cost_usd)}</span>
          ${event.cached ? '<span class="chip cached">cached</span>' : ""}
          ${event.finish_reason !== "stop" && event.finish_reason !== "tool_calls" ? `<span class="chip err">${escapeHtml(event.finish_reason)}</span>` : ""}
        </div>
        <div class="step-body">${calls ? `Requested ${calls}` : "Wrote the final answer"}</div></li>`;
    }
    if (event.type === "tool") {
      return `<li class="tool ${event.is_error ? "failed" : ""}"><span class="node">${icon("wrench")}</span>
        <div class="step-head"><span class="step-title"><code>${escapeHtml(event.name)}</code></span><span class="step-time">${fmtMs(event.duration_ms)}</span></div>
        <div class="kv"><span class="label">Arguments</span>${escapeHtml(fmtJson(event.arguments))}</div>
        <div class="kv result"><span class="label">${event.is_error ? "Error" : "Result"}</span>${escapeHtml(event.result)}</div></li>`;
    }
    if (event.type === "answer") {
      return `<li class="answer"><span class="node">${icon("check")}</span>
        <div class="step-head"><span class="step-title">Answer</span></div></li>`;
    }
    return "";
  });
  if (turn.pending) {
    items.push(`<li class="llm"><span class="node"><span class="spinner"></span></span><div class="step-head"><span class="step-title">${escapeHtml(turn.pending)}</span></div></li>`);
  }
  els.timeline.innerHTML = items.join("") || `<li class="timeline-empty">${turn.error ? escapeHtml(turn.error) : "Waiting for the first step."}</li>`;
}

function scrollToBottom() {
  els.messages.scrollTop = els.messages.scrollHeight;
}

function setBusy(busy) {
  state.busy = busy;
  els.send.disabled = busy || !els.input.value.trim();
}

// ---------------------------------------------------------------------------
// Transports
// ---------------------------------------------------------------------------

async function sendHttp(turn, text) {
  try {
    const body = await fetchJson(`${state.bases.http}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, model: state.model || null, session_id: state.httpSession }),
    });
    state.httpSession = body.session_id;
    Object.assign(turn, { trace: body.trace, answer: body.answer, usage: body.usage, model: body.model });
  } catch (error) {
    turn.error = error.message;
  }
  renderAnswer(turn);
  renderTrace();
  setBusy(false);
  loadMetricsIfVisible();
}

function connectWs() {
  if (state.ws && state.ws.readyState <= 1) return;
  state.wsReady = false;
  const ws = new WebSocket(wsUrl());
  state.ws = ws;

  ws.onopen = () => { state.wsReady = true; refreshHealth(); };
  ws.onclose = () => {
    state.wsReady = false;
    if (state.liveTurn) {
      failLiveTurn("Connection closed before the answer arrived");
    }
    if (state.transport === "ws") {
      setStatus("down", "WebSocket closed, retrying");
      setTimeout(connectWs, 2000);
    }
  };
  ws.onmessage = (message) => handleWsEvent(JSON.parse(message.data));
}

function failLiveTurn(text) {
  const turn = state.liveTurn;
  turn.pending = null;
  turn.error = text;
  state.liveTurn = null;
  renderAnswer(turn);
  renderTrace();
  setBusy(false);
}

function handleWsEvent(event) {
  const turn = state.liveTurn;
  switch (event.type) {
    case "connected":
      refreshHealth();
      return;
    case "start":
      if (turn) { turn.pending = `Calling ${event.model || "the default model"}`; renderLive(turn); }
      return;
    case "llm_start":
      if (turn) { turn.pending = `Calling ${event.model === "default" ? "the default model" : event.model}`; renderLive(turn); renderTrace(); }
      return;
    case "llm":
    case "tool":
      if (!turn) return;
      turn.trace.push(event);
      turn.pending = event.type === "tool" ? null : (event.tool_calls.length ? "Running tools" : null);
      renderLive(turn);
      if (state.selectedTurn === turn) renderTrace();
      return;
    case "answer":
      if (!turn) return;
      turn.trace.push(event);
      turn.answer = event.content;
      turn.pending = null;
      return;
    case "done":
      if (!turn) return;
      Object.assign(turn, { usage: event.usage, model: event.model, elapsed: event.elapsed_ms });
      state.liveTurn = null;
      renderAnswer(turn);
      renderTrace();
      setBusy(false);
      loadMetricsIfVisible();
      return;
    case "error":
      if (turn) failLiveTurn(event.message);
      return;
    default:
  }
}

function sendWs(turn, text) {
  if (!state.ws || state.ws.readyState !== WebSocket.OPEN) {
    turn.error = "WebSocket is not connected yet. Try again in a moment.";
    renderAnswer(turn);
    renderTrace();
    setBusy(false);
    connectWs();
    return;
  }
  state.liveTurn = turn;
  state.ws.send(JSON.stringify({ type: "message", content: text, model: state.model || null }));
}

function submit(text) {
  text = text.trim();
  if (!text || state.busy) return;
  els.input.value = "";
  autosize();
  setBusy(true);
  const turn = newTurn(text);
  if (state.transport === "ws") sendWs(turn, text);
  else sendHttp(turn, text);
}

async function resetConversation() {
  if (state.httpSession) {
    fetch(`${state.bases.http}/sessions/${state.httpSession}`, { method: "DELETE" }).catch(() => {});
    state.httpSession = null;
  }
  if (state.ws?.readyState === WebSocket.OPEN) state.ws.send(JSON.stringify({ type: "reset" }));
  state.turns = [];
  state.selectedTurn = null;
  els.messages.innerHTML = "";
  els.messages.append(els.empty);
  els.timeline.innerHTML = '<li class="timeline-empty">Send a message to see the agent\'s steps.</li>';
  els.traceSummary.innerHTML = "";
}

// ---------------------------------------------------------------------------
// UI wiring
// ---------------------------------------------------------------------------

function showTab(name) {
  document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  document.querySelectorAll(".tab-panel").forEach((p) => { p.hidden = p.dataset.panel !== name; });
  if (name === "metrics") loadMetrics();
}

function loadMetricsIfVisible() {
  if (!document.querySelector('[data-panel="metrics"]').hidden) loadMetrics();
}

const isNarrow = () => window.matchMedia("(max-width: 1000px)").matches;

function openInspector() {
  if (isNarrow()) els.app.classList.add("inspector-open");
  else els.app.classList.remove("inspector-hidden");
}

function toggleInspector() {
  if (isNarrow()) els.app.classList.toggle("inspector-open");
  else els.app.classList.toggle("inspector-hidden");
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  $("#themeToggle use").setAttribute("href", theme === "dark" ? "#i-sun" : "#i-moon");
}

function autosize() {
  els.input.style.height = "auto";
  els.input.style.height = `${Math.min(els.input.scrollHeight, 200)}px`;
  els.send.disabled = state.busy || !els.input.value.trim();
}

async function setTransport(transport) {
  state.transport = transport;
  store.set("tg.transport", transport);
  document.querySelectorAll(".segmented button").forEach((b) =>
    b.setAttribute("aria-checked", String(b.dataset.transport === transport)));
  if (transport === "ws") connectWs();
  else if (state.ws) { const ws = state.ws; state.ws = null; ws.onclose = null; ws.close(); }
  await refreshHealth();
}

function wire() {
  els.composer.addEventListener("submit", (e) => { e.preventDefault(); submit(els.input.value); });
  els.input.addEventListener("input", autosize);
  els.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); submit(els.input.value); }
  });
  $("#suggestions").addEventListener("click", (e) => {
    if (e.target.closest("button")) submit(e.target.closest("button").textContent);
  });
  els.model.addEventListener("change", () => { state.model = els.model.value; store.set("tg.model", state.model); });
  document.querySelectorAll(".segmented button").forEach((b) =>
    b.addEventListener("click", () => setTransport(b.dataset.transport)));
  document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
  $("#metricsRefresh").addEventListener("click", loadMetrics);
  $("#resetBtn").addEventListener("click", resetConversation);
  $("#panelToggle").addEventListener("click", toggleInspector);
  $("#themeToggle").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    store.set("tg.theme", next);
    applyTheme(next);
  });
  els.messages.addEventListener("click", (e) => {
    const el = e.target.closest(".msg.assistant");
    const turn = el && state.turns.find((t) => t.id === el.dataset.turn);
    if (turn) selectTurn(turn);
  });
  $("#settingsBtn").addEventListener("click", () => {
    const urls = store.get("tg.urls", {});
    $("#httpUrl").value = urls.http || "";
    $("#wsUrl").value = urls.ws || "";
    els.dialog.showModal();
  });
  els.dialog.addEventListener("close", async () => {
    if (els.dialog.returnValue !== "save") return;
    store.set("tg.urls", { http: $("#httpUrl").value.trim(), ws: $("#wsUrl").value.trim() });
    await resolveBases();
    if (state.ws) { const ws = state.ws; state.ws = null; ws.onclose = null; ws.close(); }
    await setTransport(state.transport);
    loadModels();
    loadTools();
  });
}

async function init() {
  applyTheme(store.get("tg.theme", window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
  wire();
  autosize();
  await resolveBases();
  await setTransport(state.transport);
  loadModels();
  loadTools();
  setInterval(refreshHealth, 15000);
}

init();
