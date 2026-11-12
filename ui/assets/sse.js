// POST + Server-Sent Events over fetch (EventSource only supports GET).
export async function postSSE(url, body, onEvent, signal) {
  const resp = await fetch(url, {
    method: "POST", signal,
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
  });
  const type = resp.headers.get("content-type") || "";
  if (!type.includes("text/event-stream")) {
    let data = {};
    try { data = await resp.json(); } catch { /* not JSON */ }
    onEvent("http_error", { status: resp.status, ...data });
    return;
  }
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const raw = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      let event = "message", data = "";
      for (const line of raw.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (data) onEvent(event, JSON.parse(data));
    }
  }
}
