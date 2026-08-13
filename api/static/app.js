"use strict";

const chat = document.getElementById("chat");
const form = document.getElementById("askForm");
const input = document.getElementById("queryInput");
const micBtn = document.getElementById("micBtn");
const recordingEl = document.getElementById("recording");

let mediaRecorder = null;
let chunks = [];

function addMessage(role, html) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.innerHTML = html;
  chat.appendChild(el);
  chat.scrollTop = chat.scrollHeight;
}

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function latencyHtml(lat) {
  if (!lat) return "";
  const stages = [
    ["STT", lat.stt_ms],
    ["Guard", lat.guardrail_ms],
    ["Embed", lat.embed_ms],
    ["Retrieve", lat.retrieval_ms],
    ["Generate", lat.generation_ms],
    ["Ground", lat.grounding_ms],
  ];
  const parts = stages.map(([name, ms]) => `<span class="stage">${name} <b>${ms.toFixed(1)}ms</b></span>`).join("");
  return `<div class="meta">${parts}<span class="stage">CORE <b>${lat.total_core_ms.toFixed(1)}ms</b></span><span class="stage">E2E <b>${lat.total_end_to_end_ms.toFixed(1)}ms</b></span></div>`;
}

function badgeFor(result) {
  if (result.status === "SUCCESS") return `<span class="badge good">ANSWERED</span>`;
  const labels = {
    GUARDRAIL_REJECTED: "BLOCKED",
    OFF_TOPIC: "OFF-TOPIC",
    NO_CONTEXT: "NO CONTEXT",
    UNGROUNDED: "NOT GROUNDED",
    STT_ERROR: "STT ERROR",
    GENERATION_ERROR: "GEN ERROR",
    INTERNAL_ERROR: "ERROR",
  };
  const kind = (labels[result.status] || result.status).replace(/-/g, " ");
  return `<span class="badge bad">${kind}</span>`;
}

function renderResult(result, transcript) {
  let html = `<div>${badgeFor(result)}${transcript ? `<span class="badge warn">STT: ${escapeHtml(transcript)}</span>` : ""}</div>`;
  html += `<div>${escapeHtml(result.answer)}</div>`;
  if (!result.guardrail.allowed && result.guardrail_reason) {
    html += `<div class="meta">${escapeHtml(result.guardrail_reason)}</div>`;
  }
  if (result.grounding_score !== null && result.grounding_score !== undefined) {
    html += `<div class="meta">grounding score: ${result.grounding_score}</div>`;
  }
  html += latencyHtml(result.latency);
  if (result.contexts && result.contexts.length) {
    const items = result.contexts
      .map((c) => `<details><summary>chunk [${escapeHtml(c.language)}] score ${c.score}${c.selected ? " (gold)" : ""}</summary>${escapeHtml(c.text)}</details>`)
      .join("");
    html += `<div class="contexts">retrieved context:${items}</div>`;
  }
  addMessage("bot", html);
}

async function sendQuery(text) {
  if (!text.trim()) return;
  addMessage("user", escapeHtml(text));
  input.value = "";
  const resp = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query: text }),
  });
  if (!resp.ok) {
    addMessage("bot", `<div>Server error: ${resp.status}</div>`);
    return;
  }
  const result = await resp.json();
  renderResult(result, null);
  refreshMetrics();
}

async function sendAudio(blob, filename) {
  addMessage("user", "(voice input)");
  const fd = new FormData();
  fd.append("file", blob, filename);
  const resp = await fetch("/api/ask/voice", { method: "POST", body: fd });
  if (!resp.ok) {
    addMessage("bot", `<div>Voice error: ${resp.status} ${await resp.text()}</div>`);
    return;
  }
  const result = await resp.json();
  renderResult(result, result.transcript);
  refreshMetrics();
}

async function refreshHealth() {
  try {
    const r = await fetch("/api/health");
    const h = await r.json();
    document.getElementById("health").innerHTML =
      `<span>${h.index.children} children vectors</span><br>` +
      `<span>LLM: ${h.llm || "-"}</span> ${h.llm_available ? '<span class="ok">ready</span>' : '<span class="warn">no key</span>'}`;
  } catch {
    document.getElementById("health").innerHTML = `<span class="warn">API not reachable</span>`;
  }
}

async function refreshMetrics() {
  try {
    const r = await fetch("/api/metrics");
    const m = await r.json();
    const live = m.live;
    const bench = m.benchmark;
    const setVal = (id, v) => (document.getElementById(id).textContent = v === undefined || v === null ? "-" : v.toFixed(1));
    setVal("coreP50", live.live_core_ms.p50);
    setVal("coreP70", live.live_core_ms.p70);
    setVal("coreP100", live.live_core_ms.p100);
    setVal("e2eP50", live.live_end_to_end_ms.p50);
    setVal("e2eP70", live.live_end_to_end_ms.p70);
    setVal("e2eP100", live.live_end_to_end_ms.p100);
    if (bench && bench.num_queries) {
      document.getElementById("metricsNote").textContent =
        `Benchmark over ${bench.num_queries} queries: P50 ${bench.total_core_ms.p50}ms / P70 ${bench.total_core_ms.p70}ms / P100 ${bench.total_core_ms.p100}ms (core) ` +
        `| statuses ${JSON.stringify(bench.status_counts)} | success ${(bench.success_rate * 100).toFixed(0)}%`;
    }
  } catch {
    /* metrics not ready */
  }
}

function startRecording() {
  if (!navigator.mediaDevices || !window.MediaRecorder) {
    alert("MediaRecorder not supported in this browser");
    return;
  }
  navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
    mediaRecorder = new MediaRecorder(stream);
    chunks = [];
    mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
    mediaRecorder.onstop = () => {
      const type = chunks[0] && chunks[0].type ? chunks[0].type.split(";")[0] : "audio/webm";
      const ext = type.includes("mp4") || type.includes("m4a") ? "m4a" : "webm";
      const blob = new Blob(chunks, { type });
      stream.getTracks().forEach((t) => t.stop());
      recordingEl.classList.add("hidden");
      micBtn.classList.remove("recording");
      micBtn.textContent = "MIC";
      if (blob.size > 200) sendAudio(blob, `recording.${ext}`);
    };
    mediaRecorder.start();
    recordingEl.classList.remove("hidden");
    micBtn.classList.add("recording");
    micBtn.textContent = "STOP";
  }).catch(() => alert("Microphone access denied"));
}

micBtn.addEventListener("click", () => {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
  } else {
    startRecording();
  }
});

form.addEventListener("submit", (e) => {
  e.preventDefault();
  sendQuery(input.value);
});

refreshHealth();
refreshMetrics();
setInterval(refreshMetrics, 5000);
