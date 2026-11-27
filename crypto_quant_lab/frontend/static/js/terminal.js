/* CRYPTO QUANT LAB — Global Loading State, Progress Trackers, API Client & Plotly Helpers */

let _cqlPendingRequests = 0;
const _cqlTimers = new Map();

function toggleSidebar() {
  const sb = document.getElementById("cql-sidebar");
  if (sb) sb.classList.toggle("open");
}

function setTopLoader(active) {
  const bar = document.getElementById("cql-top-loader");
  if (!bar) return;
  if (active) {
    bar.classList.add("active");
  } else {
    bar.classList.remove("active");
  }
}

function setButtonLoading(btn, isLoading, loadingText = "Processing...") {
  if (!btn) return;
  if (isLoading) {
    if (!btn.dataset.origHtml) {
      btn.dataset.origHtml = btn.innerHTML;
    }
    btn.disabled = true;
    btn.classList.add("is-loading");
    btn.innerHTML = `<span class="cql-spinner" aria-hidden="true"></span><span>${escapeHtml(loadingText)}</span>`;
  } else {
    btn.disabled = false;
    btn.classList.remove("is-loading");
    if (btn.dataset.origHtml) {
      btn.innerHTML = btn.dataset.origHtml;
      delete btn.dataset.origHtml;
    }
  }
}

function showProcessing(panelId, title, steps = []) {
  const el = document.getElementById(panelId);
  if (!el) return;
  const started = performance.now();
  if (_cqlTimers.has(panelId)) {
    clearInterval(_cqlTimers.get(panelId));
  }
  const stepsHtml = steps.length
    ? `<div class="cql-processing-steps">${steps.map((s, i) => `<span>${i + 1}. ${escapeHtml(s)}</span>`).join("")}</div>`
    : "";
  el.innerHTML = `
    <div class="cql-processing-head">
      <div class="cql-processing-title-wrap">
        <span class="cql-spinner cql-spinner-lg" aria-hidden="true"></span>
        <span>${escapeHtml(title)}</span>
      </div>
      <span class="mono text-muted" style="font-size:12px;" id="${panelId}-elapsed">0.0s elapsed</span>
    </div>
    ${stepsHtml}
  `;
  el.classList.add("visible");
  const timerId = setInterval(() => {
    const tEl = document.getElementById(`${panelId}-elapsed`);
    if (!tEl) {
      clearInterval(timerId);
      _cqlTimers.delete(panelId);
      return;
    }
    tEl.textContent = `${((performance.now() - started) / 1000).toFixed(1)}s elapsed`;
  }, 100);
  _cqlTimers.set(panelId, timerId);
}

function hideProcessing(panelId) {
  if (_cqlTimers.has(panelId)) {
    clearInterval(_cqlTimers.get(panelId));
    _cqlTimers.delete(panelId);
  }
  const el = document.getElementById(panelId);
  if (el) {
    el.classList.remove("visible");
    el.innerHTML = "";
  }
}

function renderTableSkeleton(tbodyId, cols = 6, rows = 4) {
  const tbody = document.getElementById(tbodyId);
  if (!tbody) return;
  let html = "";
  for (let r = 0; r < rows; r++) {
    html += "<tr>";
    for (let c = 0; c < cols; c++) {
      html += `<td><div class="cql-skeleton-line" style="width:${60 + ((r * 17 + c * 23) % 35)}%;"></div></td>`;
    }
    html += "</tr>";
  }
  tbody.innerHTML = html;
}

function escapeHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

async function apiJson(url, method = "GET", body = null, options = {}) {
  const showTop = options.showTopLoader !== false;
  if (showTop) {
    _cqlPendingRequests += 1;
    setTopLoader(true);
  }
  const opts = {
    method,
    headers: { "X-Requested-With": "XMLHttpRequest" },
  };
  if (body !== null) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  try {
    const resp = await fetch(url, opts);
    if (resp.status === 401) {
      window.location.href = "/login";
      throw new Error("Session expired. Redirecting to sign in.");
    }
    const text = await resp.text();
    let data = {};
    try {
      data = text ? JSON.parse(text) : {};
    } catch {
      data = { detail: text };
    }
    if (!resp.ok) {
      throw new Error(data.detail || `Request failed (${resp.status})`);
    }
    return data;
  } finally {
    if (showTop) {
      _cqlPendingRequests = Math.max(0, _cqlPendingRequests - 1);
      if (_cqlPendingRequests === 0) {
        setTopLoader(false);
      }
    }
  }
}

function showNotice(targetId, message, mode = "success") {
  const el = document.getElementById(targetId);
  if (!el) return;
  const resolved = typeof mode === "boolean" ? (mode ? "error" : "success") : mode;
  el.style.display = "block";
  const prefixMap = {
    error: "Error:",
    warning: "Notice:",
    info: "Working:",
    success: "Completed:",
  };
  const colorMap = {
    error: { border: "#ef4444", text: "#fca5a5" },
    warning: { border: "#f59e0b", text: "#fde68a" },
    info: { border: "#38bdf8", text: "#bae6fd" },
    success: { border: "#10b981", text: "#d1fae5" },
  };
  const c = colorMap[resolved] || colorMap.success;
  el.style.borderColor = c.border;
  el.style.color = c.text;
  el.innerHTML = `<strong class="mono" style="margin-right:6px;">${prefixMap[resolved] || ""}</strong>${escapeHtml(message)}`;
}

function hideNotice(targetId) {
  const el = document.getElementById(targetId);
  if (el) el.style.display = "none";
}

function plotlyDarkLayout(title, yTitle = "") {
  return {
    title: { text: title, font: { color: "#f1f5f9", size: 14, family: "Plus Jakarta Sans" } },
    paper_bgcolor: "#111827",
    plot_bgcolor: "#0b111b",
    font: { color: "#94a3b8", family: "JetBrains Mono", size: 11 },
    margin: { l: 55, r: 24, t: 42, b: 40 },
    xaxis: { gridcolor: "#1f2937", zerolinecolor: "#334155" },
    yaxis: { title: yTitle, gridcolor: "#1f2937", zerolinecolor: "#334155" },
    legend: { orientation: "h", y: -0.18, font: { color: "#cbd5e1" } },
  };
}

/* Global Active Background Job Indicator */
let _globalJobsTimer = null;
async function refreshGlobalJobsIndicator() {
  const host = document.getElementById("cql-global-jobs");
  if (!host) return;
  try {
    const jobs = await apiJson("/api/jobs", "GET", null, { showTopLoader: false });
    const active = jobs.filter(j => j.status === "running" || j.status === "queued");
    if (active.length > 0) {
      const top = active[0];
      host.style.display = "inline-flex";
      host.innerHTML = `<span class="cql-spinner" aria-hidden="true"></span><span>${active.length} active job (${escapeHtml(top.job_type)} · ${top.progress_pct}%)</span>`;
      if (!_globalJobsTimer) {
        _globalJobsTimer = setInterval(refreshGlobalJobsIndicator, 3500);
      }
    } else {
      host.style.display = "none";
      host.innerHTML = "";
      if (_globalJobsTimer) {
        clearInterval(_globalJobsTimer);
        _globalJobsTimer = null;
      }
    }
  } catch {
    // Silent ignore if page navigating or logged out
  }
}

document.addEventListener("DOMContentLoaded", () => {
  // Show top progress bar on page navigation clicks
  document.querySelectorAll("a[href]").forEach(a => {
    const href = a.getAttribute("href") || "";
    if (href.startsWith("/") && !href.includes("/export.") && a.target !== "_blank") {
      a.addEventListener("click", () => setTopLoader(true));
    }
  });
  refreshGlobalJobsIndicator();
});
