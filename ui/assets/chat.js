import { postSSE } from "./sse.js";

const API = "/api/edge";
const MAX = 500;
const $ = (id) => document.getElementById(id);
const log = $("log"), q = $("q"), form = $("form"), send = $("send"), count = $("count");

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}
function scroll() { window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" }); }
function banner(msg) { const b = $("banner"); b.textContent = msg || ""; b.style.display = msg ? "block" : "none"; }

async function refreshStatus() {
  const s = $("status");
  try {
    const info = await (await fetch(`${API}/info`)).json();
    if (info.model_loaded) {
      s.className = "pill good";
      s.innerHTML = `<span class="dot"></span>Student v${info.model_version} · ${info.n_faqs} instant answers`;
    } else {
      s.className = "pill warn";
      s.innerHTML = `<span class="dot"></span>No student yet: every answer comes from the server`;
    }
    banner("");
  } catch {
    s.className = "pill bad";
    s.innerHTML = `<span class="dot"></span>Assistant unreachable`;
    banner("Can't reach the assistant right now. Check your connection and try again in a moment.");
  }
}

async function loadSuggestions() {
  try {
    const { faqs } = await (await fetch(`${API}/faqs`)).json();
    const box = $("suggest");
    box.replaceChildren();
    for (const f of faqs.slice(0, 6)) {
      const b = el("button", "", f.question);
      b.type = "button";
      b.onclick = () => ask(f.question);
      box.append(b);
    }
  } catch { /* suggestions are optional */ }
}

function feedbackRow(requestId) {
  const row = el("div", "fb");
  const label = el("span", "small muted", "Helpful?");
  const up = el("button", "", "👍 Yes"), down = el("button", "", "👎 No");
  for (const [btn, helpful] of [[up, true], [down, false]]) {
    btn.type = "button";
    btn.onclick = async () => {
      up.disabled = down.disabled = true;
      try {
        await fetch(`${API}/feedback`, { method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ request_id: requestId, helpful }) });
        label.textContent = "Thanks for the feedback.";
      } catch { label.textContent = "Couldn't send feedback."; }
    };
  }
  row.append(label, up, down);
  return row;
}

function badgeStudent(meta) {
  const b = el("div", "badge");
  const tag = el("span", "tag");
  const sw = el("span", "sw"); sw.style.background = "var(--accent)";
  tag.append(sw, document.createTextNode("Instant answer"));
  b.append(tag, el("span", "", `matched “${meta.matched_question}”`),
    el("span", "", `${Math.round(meta.confidence * 100)}% confident`),
    el("span", "", `${meta.latency_ms.toFixed(0)} ms · student v${meta.model_version}`));
  return b;
}

function badgeTeacher(stats) {
  const b = el("div", "badge");
  const tag = el("span", "tag");
  const sw = el("span", "sw"); sw.style.background = "var(--teacher)";
  tag.append(sw, document.createTextNode("Answered by the server model"));
  b.append(tag);
  if (stats.gateway_ttft_ms) b.append(el("span", "", `first words after ${(stats.gateway_ttft_ms / 1000).toFixed(1)} s`));
  if (stats.tokens) b.append(el("span", "", `${stats.tokens} tokens`));
  return b;
}

function waitCard(meta) {
  const card = el("div", "wait");
  const f = meta.filler || {};
  card.append(el("strong", "", f.title || "One moment…"), el("span", "", f.body || ""));
  if (meta.related && meta.related.length) {
    card.append(el("div", "small muted", "Meanwhile, these instant answers might help:"));
    const rel = el("div", "related");
    for (const r of meta.related) {
      const btn = el("button", "", r.question);
      btn.type = "button";
      btn.onclick = () => ask(r.question);
      rel.append(btn);
    }
    card.append(rel);
  }
  return card;
}

async function ask(text) {
  const question = text.trim();
  if (!question) { q.focus(); return; }
  if (question.length > MAX) { banner(`Please keep questions under ${MAX} characters.`); return; }
  banner("");
  const user = el("div", "msg user"); user.append(el("div", "bubble", question)); log.append(user);
  const bot = el("div", "msg bot"); log.append(bot);
  const bubble = el("div", "bubble", "…"); bot.append(bubble);
  send.disabled = true; q.value = ""; updateCount(); scroll();

  let meta = null, wait = null, started = false;
  try {
    await postSSE(`${API}/ask`, { question, stream: true }, (event, data) => {
      if (event === "meta") {
        meta = data;
        if (data.route === "teacher") { wait = waitCard(data); bot.append(wait); }
      } else if (event === "token") {
        if (!started) { bubble.textContent = ""; started = true; }
        bubble.textContent += data.t;
        scroll();
      } else if (event === "done") {
        if (wait) wait.remove();
        bot.append(meta && meta.route === "student" ? badgeStudent(meta) : badgeTeacher(data));
        bot.append(feedbackRow(meta.request_id));
      } else if (event === "error") {
        if (wait) wait.remove();
        bubble.remove();
        bot.append(el("div", "notice", data.message || "Something went wrong."));
      } else if (event === "http_error") {
        bubble.remove();
        bot.append(el("div", "notice", data.message || `The assistant could not answer (HTTP ${data.status}).`));
      }
    });
  } catch {
    bubble.remove();
    bot.append(el("div", "notice", "Can't reach the assistant right now. Please try again in a moment."));
    refreshStatus();
  } finally {
    send.disabled = false; q.focus(); scroll();
  }
}

function updateCount() {
  const n = q.value.trim().length;
  count.textContent = `${n}/${MAX}`;
  count.classList.toggle("over", n > MAX);
  send.disabled = n === 0 || n > MAX;
  q.style.height = "auto"; q.style.height = Math.min(q.scrollHeight, 160) + "px";
}

form.addEventListener("submit", (e) => { e.preventDefault(); ask(q.value); });
q.addEventListener("input", updateCount);
q.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); if (!send.disabled) ask(q.value); } });
updateCount(); refreshStatus(); loadSuggestions();
setInterval(refreshStatus, 15000);
