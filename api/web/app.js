const TOKEN_KEY = "ppe_api_access_token";

/**
 * API origin for fetch(). Empty = same-origin (UI served from this app under /ui/).
 * Use ?api=http://host:port when the page is opened as file:// or from another host.
 */
function apiBase() {
  const params = new URLSearchParams(window.location.search);
  const q = params.get("api");
  if (q) return q.replace(/\/$/, "");
  if (window.location.protocol === "file:") return "http://127.0.0.1:8000";
  return "";
}

function apiUrl(path) {
  const p = path.startsWith("/") ? path : `/${path}`;
  const base = apiBase();
  return base ? `${base}${p}` : p;
}

function fixLinksForFileOrigin() {
  if (window.location.protocol !== "file:") return;
  const base = apiBase();
  document.querySelectorAll('a[href^="/"]').forEach((a) => {
    a.href = `${base}${a.getAttribute("href")}`;
  });
}

fixLinksForFileOrigin();

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

/** Stable hue 0–359 from class name for box colors. */
function hueFromClassName(name) {
  const s = String(name || "");
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
  return h % 360;
}

/**
 * Decode `file`, draw it on a canvas at natural size, overlay detection boxes (pixel coords from API).
 */
function createAnnotatedCanvas(file, detections) {
  const url = URL.createObjectURL(file);
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(url);
      const w = img.naturalWidth;
      const h = img.naturalHeight;
      const canvas = document.createElement("canvas");
      canvas.width = w;
      canvas.height = h;
      canvas.className = "prediction-canvas";
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0);

      const lineW = Math.max(2, Math.round(Math.min(w, h) / 400));
      const fontSize = Math.max(12, Math.round(Math.min(w, h) / 35));
      const pad = 4;

      for (const d of detections) {
        const b = d.bbox || {};
        const x1 = Number(b.x1);
        const y1 = Number(b.y1);
        const x2 = Number(b.x2);
        const y2 = Number(b.y2);
        if (![x1, y1, x2, y2].every((n) => Number.isFinite(n))) continue;

        const bw = x2 - x1;
        const bh = y2 - y1;
        const hue = hueFromClassName(d.class_name);
        const color = `hsl(${hue} 78% 52%)`;

        ctx.strokeStyle = color;
        ctx.lineWidth = lineW;
        ctx.strokeRect(x1, y1, bw, bh);

        const label = `${d.class_name} ${Number(d.confidence).toFixed(2)}`;
        ctx.font = `600 ${fontSize}px system-ui, sans-serif`;
        const tw = ctx.measureText(label).width;
        const labelH = fontSize + pad * 2;
        let ly = y1 - labelH;
        if (ly < 0) ly = y1 + 2;

        ctx.fillStyle = "rgba(0,0,0,0.72)";
        ctx.fillRect(x1, ly, tw + pad * 2, labelH);
        ctx.fillStyle = "#fff";
        ctx.fillText(label, x1 + pad, ly + fontSize + pad - 3);
      }

      resolve(canvas);
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Failed to decode image for preview."));
    };
    img.src = url;
  });
}

function appendRawJsonDetails(parent, data) {
  const det = document.createElement("details");
  det.className = "prediction-raw-json";
  const sum = document.createElement("summary");
  sum.textContent = "Raw JSON";
  const pre = document.createElement("pre");
  pre.textContent = JSON.stringify(data, null, 2);
  det.appendChild(sum);
  det.appendChild(pre);
  parent.appendChild(det);
}

function formatSingleSummary(fileName, data) {
  const n = (data.detections || []).length;
  return [
    `file: ${fileName}`,
    `image_id: ${data.image_id}`,
    `inference_time_ms: ${data.inference_time_ms}`,
    `detections: ${n}`,
  ].join("\n");
}

function formatBatchSummary(data, fileCount, resultCount) {
  const lines = [
    `total_inference_time_ms: ${data.total_inference_time_ms}`,
    `images (uploaded): ${fileCount}`,
    `results (API): ${resultCount}`,
  ];
  if (fileCount !== resultCount) {
    lines.push("warning: upload count and result count differ — pairing by index may be wrong.");
  }
  return lines.join("\n");
}

async function renderSinglePredictionViz(vizEl, file, data) {
  vizEl.innerHTML = "";
  const wrap = document.createElement("div");
  wrap.className = "prediction-viz";

  const cap = document.createElement("p");
  cap.className = "prediction-viz-caption";
  cap.textContent = `${file.name} · ${data.image_id} · ${data.inference_time_ms} ms · ${(data.detections || []).length} detection(s)`;
  wrap.appendChild(cap);

  let decoded = false;
  try {
    const canvas = await createAnnotatedCanvas(file, data.detections || []);
    wrap.appendChild(canvas);
    decoded = true;
  } catch (e) {
    const err = document.createElement("p");
    err.className = "hint";
    err.textContent = String(e.message || e);
    wrap.appendChild(err);
  }

  if (decoded && !(data.detections || []).length) {
    const hint = document.createElement("p");
    hint.className = "hint";
    hint.textContent = "No detections.";
    wrap.appendChild(hint);
  }

  appendRawJsonDetails(wrap, data);
  vizEl.appendChild(wrap);
}

/**
 * API returns `results` in the same order as uploaded `files` (see predict_batch loop in predict.py).
 */
async function renderBatchPredictionViz(vizEl, files, data) {
  vizEl.innerHTML = "";
  const results = data.results || [];
  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    const imgResult = results[i];
    const block = document.createElement("div");
    block.className = "prediction-viz prediction-viz-batch-item";

    if (!imgResult) {
      const cap = document.createElement("p");
      cap.className = "prediction-viz-caption";
      cap.textContent = `${file.name} — (no matching result row)`;
      block.appendChild(cap);
      vizEl.appendChild(block);
      continue;
    }

    const cap = document.createElement("p");
    cap.className = "prediction-viz-caption";
    cap.textContent = `${file.name} · ${imgResult.image_id} · ${imgResult.inference_time_ms} ms · ${(imgResult.detections || []).length} detection(s)`;
    block.appendChild(cap);

    let decoded = false;
    try {
      const canvas = await createAnnotatedCanvas(file, imgResult.detections || []);
      block.appendChild(canvas);
      decoded = true;
    } catch (e) {
      const err = document.createElement("p");
      err.className = "hint";
      err.textContent = String(e.message || e);
      block.appendChild(err);
    }

    if (decoded && !(imgResult.detections || []).length) {
      const hint = document.createElement("p");
      hint.className = "hint";
      hint.textContent = "No detections.";
      block.appendChild(hint);
    }

    appendRawJsonDetails(block, imgResult);
    vizEl.appendChild(block);
  }

  const full = document.createElement("details");
  full.className = "prediction-raw-json";
  const sum = document.createElement("summary");
  sum.textContent = "Raw JSON (full batch response)";
  const pre = document.createElement("pre");
  pre.textContent = JSON.stringify(data, null, 2);
  full.appendChild(sum);
  full.appendChild(pre);
  vizEl.appendChild(full);
}

document.getElementById("btn-health")?.addEventListener("click", async () => {
  const out = "out-health";
  setOutput(out, "Loading…");
  try {
    const r = await fetch(apiUrl("/health"));
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
    const r = await fetch(apiUrl("/auth/token"), {
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
  const vizEl = document.getElementById("viz-predict-image");
  const token = getToken();
  if (!token) {
    setOutput(out, "Get a JWT first (POST /auth/token).", true);
    vizEl.innerHTML = "";
    return;
  }
  const input = document.getElementById("file-image");
  const file = input.files?.[0];
  if (!file) {
    setOutput(out, "Choose an image file.", true);
    vizEl.innerHTML = "";
    return;
  }
  setOutput(out, "Running inference…");
  vizEl.innerHTML = "";
  const fd = new FormData();
  fd.append("file", file, file.name);
  try {
    const r = await fetch(apiUrl("/predict/image"), {
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
    setOutput(out, formatSingleSummary(file.name, data));
    await renderSinglePredictionViz(vizEl, file, data);
  } catch (e) {
    setOutput(out, String(e), true);
    vizEl.innerHTML = "";
  }
});

document.getElementById("form-predict-batch")?.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const out = "out-predict-batch";
  const vizEl = document.getElementById("viz-predict-batch");
  const token = getToken();
  if (!token) {
    setOutput(out, "Get a JWT first (POST /auth/token).", true);
    vizEl.innerHTML = "";
    return;
  }
  const input = document.getElementById("files-batch");
  const files = input.files ? Array.from(input.files) : [];
  if (!files.length) {
    setOutput(out, "Choose one or more images.", true);
    vizEl.innerHTML = "";
    return;
  }
  setOutput(out, "Running batch inference…");
  vizEl.innerHTML = "";
  const fd = new FormData();
  for (const f of files) {
    fd.append("files", f, f.name);
  }
  try {
    const r = await fetch(apiUrl("/predict/batch"), {
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
    const results = data.results || [];
    setOutput(out, formatBatchSummary(data, files.length, results.length));
    await renderBatchPredictionViz(vizEl, files, data);
  } catch (e) {
    setOutput(out, String(e), true);
    vizEl.innerHTML = "";
  }
});

document.getElementById("btn-metrics")?.addEventListener("click", async () => {
  const out = "out-metrics";
  setOutput(out, "Loading…");
  try {
    const r = await fetch(apiUrl("/metrics"));
    const text = await r.text();
    setOutput(out, text, !r.ok);
  } catch (e) {
    setOutput(out, String(e), true);
  }
});

updateTokenBadge();
