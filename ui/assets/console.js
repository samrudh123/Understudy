const API = "/api/console";
const $ = (id) => document.getElementById(id);
const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const pct = (v) => (v === null || v === undefined ? "–" : `${(v * 100).toFixed(1)}%`);
function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; }
function toast(msg) { const t = $("toast"); t.textContent = msg; t.classList.add("show"); setTimeout(() => t.classList.remove("show"), 3500); }
async function api(path, opts = {}) {
  const r = await fetch(API + path, { headers: { "Content-Type": "application/json" }, ...opts });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || `HTTP ${r.status}`);
  return body;
}
const reviewer = $("reviewer");
reviewer.value = localStorage.getItem("reviewer") || "";
reviewer.oninput = () => localStorage.setItem("reviewer", reviewer.value);

// ------------------------------------------------------------ overview tiles
function tile(label, value, sub) {
  const t = el("div", "tile");
  t.append(el("div", "label", label), el("div", "value", value), el("div", "sub", sub || ""));
  return t;
}
async function loadOverview() {
  try {
    const o = await api("/api/overview");
    const e = o.edge || {}, g = o.gate || {}, ch = g.challenger || {};
    const tiles = $("tiles");
    tiles.replaceChildren(
      tile("Serving student", e.model_loaded ? `v${e.model_version}` : "none",
        e.model_loaded ? `${e.n_faqs} FAQs · τ = ${fmt(e.tau)}` : "edge has no model yet"),
      tile("Fallback ratio (10 min)", pct(o.fallback_ratio_10m), "share of questions sent to the teacher"),
      tile("Pending approvals", String(o.pending_approvals), `${o.approved_faqs} FAQs approved so far`),
      tile("Last gate", g.passed === undefined ? "–" : (g.passed ? "passed ✓" : "failed ✕"),
        g.passed === undefined ? "" : `precision ${pct(ch.traffic_precision)} · coverage ${pct(ch.traffic_coverage)}`),
      tile("Last pipeline run", o.pipeline ? o.pipeline.state : "–",
        o.pipeline ? `${o.pipeline.name} · ${fmt(o.pipeline.duration_s, 0)} s` : ""),
    );
    const al = $("alerts");
    const firing = (o.alerts || []).filter((a) => a.state === "firing");
    if (o.alerts === null) { al.className = "pill"; al.textContent = "alerts: Prometheus unreachable"; }
    else if (firing.length) { al.className = "pill bad"; al.innerHTML = `<span class="dot"></span>${firing.map((a) => a.name).join(", ")}`; }
    else { al.className = "pill good"; al.innerHTML = `<span class="dot"></span>no alerts firing`; }
  } catch (err) { $("tiles").replaceChildren(tile("Console API", "down", String(err.message))); }
}

// ------------------------------------------------------------ DAG
function layers(nodes, edges) {
  const depth = Object.fromEntries(nodes.map((n) => [n.name, 0]));
  for (let i = 0; i < nodes.length; i++)
    for (const e of edges) depth[e.to] = Math.max(depth[e.to], depth[e.from] + 1);
  const cols = {};
  for (const n of nodes) (cols[depth[n.name]] ||= []).push(n);
  return cols;
}
async function loadDag(refresh = false) {
  const d = await api(`/api/dag${refresh ? "?refresh=true" : ""}`);
  const outdated = new Set(d.outdated || []);
  const cols = layers(d.nodes, d.edges);
  const W = 138, H = 64, GX = 26, GY = 14, pos = {};
  const outer = $("dag");
  const box = el("div");
  box.style.position = "relative";
  outer.replaceChildren(box);
  let maxRows = 0;
  Object.entries(cols).forEach(([c, list]) => {
    maxRows = Math.max(maxRows, list.length);
    list.forEach((n, r) => {
      pos[n.name] = { x: c * (W + GX), y: r * (H + GY) };
      const status = outdated.has(n.name) && n.status !== "failed" ? "outdated" : n.status;
      const node = el("div", `node ${status}`);
      node.style.left = `${pos[n.name].x}px`; node.style.top = `${pos[n.name].y}px`;
      node.title = `${n.cmd || ""}${n.finished_at ? `\nlast run: ${new Date(n.finished_at).toLocaleString()}` : ""}`;
      const label = { success: "✓ succeeded", failed: "✕ failed", never_run: "not run yet", outdated: "↻ out of date" }[status] || status;
      const rows = n.rows_out >= 1000 ? `${(n.rows_out / 1000).toFixed(1)}k` : (n.rows_out ?? 0);
      const meta = n.duration_s !== null && n.duration_s !== undefined ? `${fmt(n.duration_s, 1)} s · ${rows} rows` : "";
      node.append(el("div", "n", n.name), el("div", "meta", label), el("div", "meta", meta));
      box.append(node);
    });
  });
  const width = Object.keys(cols).length * (W + GX), height = maxRows * (H + GY);
  box.style.height = `${height}px`;
  box.style.width = `${width - GX}px`;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("width", width); svg.setAttribute("height", height);
  for (const e of d.edges) {
    const a = pos[e.from], b = pos[e.to];
    if (!a || !b) continue;
    const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
    const x1 = a.x + W, y1 = a.y + H / 2, x2 = b.x, y2 = b.y + H / 2, mx = (x1 + x2) / 2;
    p.setAttribute("d", `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`);
    p.setAttribute("fill", "none"); p.setAttribute("stroke", "#c9c8c2"); p.setAttribute("stroke-width", "1.5");
    svg.append(p);
  }
  box.prepend(svg);
  $("dag-legend").textContent = d.outdated === null ? "dvc status unavailable"
    : `${outdated.size} stage(s) out of date · left border: green = last run succeeded, red = failed, amber = out of date`;
}

// ------------------------------------------------------------ pipeline runs
async function loadRun() {
  const s = await api("/api/pipeline/status");
  const box = $("run");
  box.replaceChildren();
  const busy = !!s.current;
  for (const id of ["btn-discover", "btn-retrain", "btn-sim"]) $(id).disabled = busy;
  if (s.current) {
    box.append(el("div", "small", `Running “${s.current.name}” (${s.current.reason}) since ${new Date(s.current.started_at).toLocaleTimeString()}…`));
    const pre = el("div", "log", (s.current.log_tail || []).join("\n"));
    box.append(pre); pre.scrollTop = pre.scrollHeight;
  }
  const hist = $("history");
  if (!s.history.length) { hist.textContent = "No pipeline runs from the console yet."; return busy; }
  const t = el("table");
  t.innerHTML = "<tr><th>Started</th><th>Run</th><th>Reason</th><th>State</th><th class='num'>Duration</th></tr>";
  for (const r of s.history) {
    const tr = el("tr", r.state === "failed" ? "failed" : "");
    for (const v of [new Date(r.started_at).toLocaleString(), r.name, r.reason, r.state, `${fmt(r.duration_s, 0)} s`]) tr.append(el("td", "", v));
    tr.lastChild.className = "num";
    t.append(tr);
  }
  hist.replaceChildren(t);
  return busy;
}
async function startRun(target) {
  try { await api("/api/pipeline/run", { method: "POST", body: JSON.stringify({ target }) }); toast(`Started ${target}`); poll(); }
  catch (e) { toast(e.message); }
}
$("btn-discover").onclick = () => startRun("discover");
$("btn-retrain").onclick = () => startRun("retrain");
$("btn-refresh").onclick = () => loadDag(true).then(() => toast("DAG status refreshed"));
$("btn-sim").onclick = async () => {
  const weeks = $("sim-weeks").value.trim();
  if (!/^[0-9,\- ]+$/.test(weeks)) { toast("Enter weeks like 3 or 2-3"); return; }
  try { await api("/api/simulate", { method: "POST", body: JSON.stringify({ weeks }) }); toast(`Simulating week(s) ${weeks}`); poll(); }
  catch (e) { toast(e.message); }
};

// ------------------------------------------------------------ approvals
async function loadQueue() {
  const a = await api("/api/approvals");
  $("pending-count").textContent = `${a.pending.length} pending`;
  $("pending-count").className = a.pending.length ? "pill warn" : "pill";
  const box = $("queue");
  box.className = "";
  if (!a.pending.length) { box.className = "muted small"; box.textContent = "Nothing to review. New drafts appear here after discovery runs."; return; }
  box.replaceChildren();
  for (const d of a.pending) {
    const card = el("div", "draft");
    const head = el("div", "small muted",
      `${d.candidate_id} · ${d.support} questions (${pct(d.share_of_fallbacks)} of fallbacks) · cohesion ${fmt(d.cohesion)} · ${d.teacher_model} / ${d.prompt_version}`);
    const qIn = el("input"); qIn.value = d.question;
    const aIn = el("textarea"); aIn.rows = 3; aIn.value = d.answer;
    const pIn = el("textarea"); pIn.rows = 4; pIn.value = d.paraphrases.join("\n");
    const row = (label, input) => { const r = el("div", "row"); r.append(el("label", "", label), input); return r; };
    const ex = el("details"); ex.append(el("summary", "", `Real user questions (${d.examples.length})`), el("div", "", d.examples.join(" · ")));
    const actions = el("div", "actions");
    const approveB = el("button", "primary", "Approve"), rejectB = el("button", "danger", "Reject"), mergeB = el("button", "", "Merge into");
    const sel = el("select");
    for (const f of a.approved) { const o = el("option", "", f.question); o.value = f.faq_id; sel.append(o); }
    if (!a.approved.length) { mergeB.disabled = true; sel.disabled = true; }
    const send = async (decision) => {
      if (!reviewer.value.trim()) { toast("Enter your name as reviewer first"); reviewer.focus(); return; }
      const body = { decision, reviewer: reviewer.value.trim(), question: qIn.value, answer: aIn.value,
        paraphrases: pIn.value.split("\n").map((s) => s.trim()).filter(Boolean) };
      if (decision === "merge") body.merge_into = sel.value;
      try { const r = await api(`/api/approvals/${d.candidate_id}`, { method: "POST", body: JSON.stringify(body) });
        toast(`${r.faq_id}: ${r.decision}`); card.remove(); loadQueue(); loadOverview(); }
      catch (e) { toast(e.message); }
    };
    approveB.onclick = () => send("approve"); rejectB.onclick = () => send("reject"); mergeB.onclick = () => send("merge");
    actions.append(approveB, rejectB, mergeB, sel);
    card.append(head, row("Question", qIn), row("Answer", aIn), row("Paraphrases", pIn), ex, actions);
    box.append(card);
  }
}

// ------------------------------------------------------------ models + runs
async function loadModels() {
  const box = $("models");
  try {
    const m = await api("/api/models");
    if (!m.versions.length) { box.textContent = "No registered versions yet."; return; }
    const t = el("table");
    t.innerHTML = "<tr><th>Version</th><th>Aliases</th><th class='num'>Precision</th><th class='num'>Coverage</th><th class='num'>p95 ms</th><th></th></tr>";
    for (const v of m.versions) {
      const tr = el("tr");
      tr.append(el("td", "", `v${v.version}`), el("td", "", v.aliases.join(", ")),
        el("td", "num", pct(Number(v.tags.traffic_precision))), el("td", "num", pct(Number(v.tags.traffic_coverage))),
        el("td", "num", fmt(Number(v.tags.latency_p95_ms), 1)));
      const td = el("td");
      if (!v.aliases.includes("champion")) {
        const b = el("button", "", "Promote"); b.onclick = async () => {
          try { const r = await api("/api/models/promote", { method: "POST", body: JSON.stringify({ version: v.version }) });
            toast(`Champion is now v${r.champion}${r.edge.reloaded ? " (edge reloaded)" : ""}`); refreshAll(); } catch (e) { toast(e.message); } };
        td.append(b);
      }
      tr.append(td); t.append(tr);
    }
    box.replaceChildren(t);
  } catch (e) { box.textContent = `Registry unavailable: ${e.message}`; }
}
$("btn-rollback").onclick = async () => {
  if (!confirm("Move the champion alias back to the previous version and reload the edge?")) return;
  try { const r = await api("/api/models/rollback", { method: "POST", body: "{}" });
    toast(`Rolled back to v${r.champion} in ${r.seconds} s${r.edge.reloaded ? " (edge reloaded)" : " (edge NOT reloaded)"}`); refreshAll(); }
  catch (e) { toast(e.message); }
};
async function loadRuns() {
  const box = $("runs");
  try {
    const [r, links] = await Promise.all([api("/api/runs"), api("/api/links")]);
    if (!r.runs.length) { box.textContent = "No runs yet."; return; }
    const t = el("table");
    t.innerHTML = "<tr><th>Run</th><th>Status</th><th class='num'>Precision</th><th class='num'>Coverage</th><th class='num'>FAQs</th></tr>";
    for (const x of r.runs) {
      const tr = el("tr", x.status === "FAILED" ? "failed" : "");
      const a = el("a", "", x.name || x.run_id.slice(0, 8));
      a.href = `${links.mlflow}/#/experiments/${x.experiment_id}/runs/${x.run_id}`; a.target = "_blank";
      const td = el("td"); td.append(a);
      tr.append(td, el("td", "", x.status), el("td", "num", pct(x.test_traffic_precision)),
        el("td", "num", pct(x.test_traffic_coverage)), el("td", "num", x.n_faqs ?? "–"));
      t.append(tr);
    }
    box.replaceChildren(t);
  } catch (e) { box.textContent = `MLflow unavailable: ${e.message}`; }
}
async function loadLinks() {
  try {
    const l = await api("/api/links");
    const span = $("links"); span.replaceChildren();
    const names = { mlflow: "MLflow", grafana: "Grafana", prometheus: "Prometheus", alertmanager: "Alertmanager" };
    for (const [k, v] of Object.entries(l)) { const a = el("a", "", names[k] || k); a.href = v; a.target = "_blank"; span.append(a, document.createTextNode(" ")); }
  } catch { /* optional */ }
}

let timer = null;
async function poll() {
  clearTimeout(timer);
  const busy = await loadRun().catch(() => false);
  if (busy) { loadDag().catch(() => {}); timer = setTimeout(poll, 3000); }
  else { refreshAll(); timer = setTimeout(poll, 20000); }
}
function refreshAll() { loadOverview(); loadDag().catch((e) => { $("dag").textContent = e.message; }); loadQueue().catch(() => {}); loadModels(); loadRuns(); }
loadLinks(); refreshAll(); poll();
