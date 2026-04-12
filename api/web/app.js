const TOKEN_KEY = "ppe_api_access_token";

function getToken() {
  return sessionStorage.getItem(TOKEN_KEY) || "";
}

function setToken(value) {
  if (value) sessionStorage.setItem(TOKEN_KEY, value);
  else sessionStorage.removeItem(TOKEN_KEY);
  updateTokenBadge();
}

function updateTokenBadge() {
  const el = document.getElementById("token-status");
  if (!el) return;
  const t = getToken();
  if (t) {
    el.textContent = "Token in session";
    el.className = "badge ok";
  } else {
    el.textContent = "No token";
    el.className = "badge";
  }
}

async function parseErrorBody(response) {
  const ct = response.headers.get("content-type") || "";
  if (ct.includes("application/json")) {
    const data = await response.json();
    if (typeof data.detail === "string") return data.detail;
    if (Array.isArray(data.detail)) {
      return data.detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
    }
    return JSON.stringify(data);
  }
  const text = await response.text();
  return text || response.statusText;
}

function setOutput(id, text, isError) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = text;
  el.classList.remove("empty", "error");
  if (isError) el.classList.add("error");
  else if (!text) el.classList.add("empty");
}

document.getElementById("btn-health")?.addEventListener("click", async () => {
  const out = "out-health";
  setOutput(out, "Loading…");
  try {
    const r = await fetch("/health");
    const body = await r.json();
    setOutput(out, JSON.stringify(body, null, 2), !r.ok);
  } catch (e) {
    setOutput(out, String(e), true);
  }
});

document.getElementById("form-token")?.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const out = "out-token";
  const u = document.getElementById("auth-user").value;
  const p = document.getElementById("auth-pass").value;
  setOutput(out, "Requesting token…");
  document.getElementById("token-display").textContent = "";
  try {
    const r = await fetch("/auth/token", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: u, password: p }),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
      setToken("");
      const msg =
        typeof data.detail === "string"
          ? data.detail
          : Array.isArray(data.detail)
            ? data.detail.map((d) => d.msg || JSON.stringify(d)).join("; ")
            : JSON.stringify(data) || r.statusText;
      setOutput(out, `${r.status} ${r.statusText}\n${msg}`, true);
      return;
    }
    setToken(data.access_token || "");
    setOutput(out, JSON.stringify({ access_token: "(stored)", token_type: data.token_type }, null, 2));
    const td = document.getElementById("token-display");
    const tok = getToken();
    td.textContent = tok ? `${tok.slice(0, 24)}… (${tok.length} chars)` : "";
  } catch (e) {
    setOutput(out, String(e), true);
  }
});

document.getElementById("btn-copy-token")?.addEventListener("click", async () => {
  const t = getToken();
  if (!t) return;
  try {
    await navigator.clipboard.writeText(t);
    const out = document.getElementById("out-token");
    if (out && !out.classList.contains("error")) {
      const prev = out.textContent;
      out.textContent = prev + "\n(copied to clipboard)";
      setTimeout(() => {
        out.textContent = prev;
      }, 1500);
    }
  } catch (_) {
    /* ignore */
  }
});

document.getElementById("btn-clear-token")?.addEventListener("click", () => {
  setToken("");
  document.getElementById("token-display").textContent = "";
  setOutput("out-token", "");
});

document.getElementById("form-predict-image")?.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const out = "out-predict-image";
  const tableWrap = document.getElementById("table-predict-image");
  const token = getToken();
  if (!token) {
    setOutput(out, "Get a JWT first (POST /auth/token).", true);
    tableWrap.innerHTML = "";
    return;
  }
  const input = document.getElementById("file-image");
  const file = input.files?.[0];
  if (!file) {
    setOutput(out, "Choose an image file.", true);
    tableWrap.innerHTML = "";
    return;
  }
  setOutput(out, "Running inference…");
  tableWrap.innerHTML = "";
  const fd = new FormData();
  fd.append("file", file, file.name);
  try {
    const r = await fetch("/predict/image", {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` },
      body: fd,
    });
    if (!r.ok) {
      const msg = await parseErrorBody(r);
      setOutput(out, `${r.status} ${r.statusText}\n${msg}`, true);
      return;
    }
    const data = await r.json();
    setOutput(out, JSON.stringify(data, null, 2));
    tableWrap.innerHTML = renderDetectionsTable(data.detections || []);
  } catch (e) {
    setOutput(out, String(e), true);
    tableWrap.innerHTML = "";
  }
});

document.getElementById("form-predict-batch")?.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const out = "out-predict-batch";
  const tableWrap = document.getElementById("table-predict-batch");
  const token = getToken();
  if (!token) {
    setOutput(out, "Get a JWT first (POST /auth/token).", true);
    tableWrap.innerHTML = "";
    return;
  }
  const input = document.getElementById("files-batch");
  const files = input.files ? Array.from(input.files) : [];
  if (!files.length) {
    setOutput(out, "Choose one or more images.", true);
    tableWrap.innerHTML = "";
    return;
  }
  setOutput(out, "Running batch inference…");
  tableWrap.innerHTML = "";
  const fd = new FormData();
  for (const f of files) {
    fd.append("files", f, f.name);
  }
  try {
    const r = await fetch("/predict/batch", {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` },
      body: fd,
    });
    if (!r.ok) {
      const msg = await parseErrorBody(r);
      setOutput(out, `${r.status} ${r.statusText}\n${msg}`, true);
      return;
    }
    const data = await r.json();
    setOutput(out, JSON.stringify(data, null, 2));
    const rows = (data.results || []).map((img, i) => {
      const sub = renderDetectionsTable(img.detections || [], `Image ${i + 1} — ${img.image_id || ""}`);
      return `<div class="hint" style="margin-top:1rem">${sub}</div>`;
    });
    tableWrap.innerHTML = rows.join("") || "<p class=\"hint\">No result rows.</p>";
  } catch (e) {
    setOutput(out, String(e), true);
    tableWrap.innerHTML = "";
  }
});

document.getElementById("btn-metrics")?.addEventListener("click", async () => {
  const out = "out-metrics";
  setOutput(out, "Loading…");
  try {
    const r = await fetch("/metrics");
    const text = await r.text();
    setOutput(out, text, !r.ok);
  } catch (e) {
    setOutput(out, String(e), true);
  }
});

function renderDetectionsTable(detections, caption) {
  if (!detections.length) {
    return `<p class="hint">${caption ? caption + " — " : ""}No detections.</p>`;
  }
  const head =
    "<thead><tr><th>class</th><th>id</th><th>conf</th><th>bbox (x1,y1,x2,y2)</th></tr></thead>";
  const body = detections
    .map((d) => {
      const b = d.bbox || {};
      const bb = [b.x1, b.y1, b.x2, b.y2].map((x) => (typeof x === "number" ? x.toFixed(1) : x)).join(", ");
      return `<tr><td>${escapeHtml(d.class_name)}</td><td>${d.class_id}</td><td>${Number(d.confidence).toFixed(3)}</td><td>${escapeHtml(bb)}</td></tr>`;
    })
    .join("");
  const cap = caption ? `<p class="hint" style="margin-bottom:0.35rem">${escapeHtml(caption)}</p>` : "";
  return `${cap}<table class="detections">${head}<tbody>${body}</tbody></table>`;
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

updateTokenBadge();
