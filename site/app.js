"use strict";

const DEMO_FALLBACK = {
  schema_version: "site.v1",
  generated_at: "2026-04-26T00:00:00Z",
  source: "embedded-demo",
  runs: [
    {
      id: "fallback-bge-scifact",
      benchmark: "beir",
      dataset: "scifact",
      dataset_label: "SciFact",
      model_id: "BAAI/bge-large-en-v1.5",
      model_label: "BGE Large v1.5",
      family: "dense",
      params_m: 335,
      system: "dense-hnsw",
      system_label: "HNSW",
      backend: "FAISS HNSWFlat",
      compression: "fp16+graph",
      status: "completed",
      quality: { ndcg_at_10: 0.727, recall_at_100: 0.872, mrr_at_10: 0.689, map_at_100: 0.646 },
      latency: { e2e_query_ms_p50: 20.4, e2e_query_ms_p99: 37.9, query_encode_ms_p50: 16.7, retrieval_ms_p50_topk100: 2.8, latency_sample_size: 200 },
      storage: { index_bytes: 10400000, index_bytes_per_doc: 2080 },
      cost: { cost_per_million_queries_usd: 11.28 },
      protocol: { hardware: "A100 40GB", precision: "fp16" },
    },
    {
      id: "fallback-colbert-limit",
      benchmark: "limit",
      dataset: "limit",
      dataset_label: "LIMIT",
      model_id: "colbert-ir/colbertv2.0",
      model_label: "ColBERTv2",
      family: "late_interaction",
      params_m: 110,
      system: "li-fastplaid",
      system_label: "FastPlaid",
      backend: "PyLate FastPlaid",
      compression: "plaid-pq",
      status: "completed",
      quality: { ndcg_at_10: 0.684, recall_at_100: 0.792, mrr_at_10: 0.613, map_at_100: 0.554 },
      latency: { e2e_query_ms_p50: 38.9, e2e_query_ms_p99: 62.5, query_encode_ms_p50: 12.1, retrieval_ms_p50_topk100: 25.5, latency_sample_size: 200 },
      storage: { index_bytes: 185000000, index_bytes_per_doc: 3700 },
      cost: { cost_per_million_queries_usd: 21.51 },
      protocol: { hardware: "A100 40GB", precision: "fp16" },
    },
  ],
};

const METRICS = {
  ndcg_at_10: "nDCG@10",
  recall_at_100: "Recall@100",
  mrr_at_10: "MRR@10",
  map_at_100: "MAP@100",
};

const AXES = {
  latency: { label: "Query latency", short: "query latency", field: "latency_ms", title: "p50 query latency (ms)", units: "ms" },
  storage: { label: "Index size", short: "index", field: "storage_gb", title: "index size (GB)", units: "GB" },
};

const FAMILY = {
  dense: { label: "Dense", color: "#234b3d" },
  late_interaction: { label: "Late interaction", color: "#a23b17" },
  hybrid: { label: "Hybrid (RRF)", color: "#5b3f6e" },
  sparse: { label: "Sparse (BM25)", color: "#3f5d75" },
  unknown: { label: "Other", color: "#8e887b" },
};

const STATUS_LABEL = {
  completed: "Measured",
  quality_only: "Quality only",
  latency_skipped: "Latency skipped",
  not_applicable_small_corpus: "N/A · small corpus",
  failed: "Failed",
  pending: "Pending",
  unknown: "Unknown",
};

const BENCHMARK_COPY = {
  beir: "Standard retrieval across scientific, biomedical, financial, and argumentative search.",
  bright: "Reasoning-heavy retrieval where matching requires more than topical similarity.",
  limit: "Synthetic separation benchmark designed to expose failures of single-vector dense retrieval.",
  lotte: "Long-tail forum retrieval across lifestyle, technology, writing, science, and recreation.",
};

const BENCHMARK_TITLE = {
  beir: "BEIR",
  bright: "BRIGHT",
  limit: "LIMIT",
  lotte: "LoTTE",
};

const BENCHMARK_FULL = {
  beir: "BEIR · standard retrieval",
  bright: "BRIGHT · reasoning retrieval",
  limit: "LIMIT · separation",
  lotte: "LoTTE · forum retrieval",
};

const FRONTIER_LABELS = {
  latency: ["Fastest on frontier", "Best balance", "Highest quality"],
  storage: ["Smallest on frontier", "Best balance", "Highest quality"],
};

// Frontend-side label tidy-up. The export script's model_label values are
// mostly OK but a few are noisy ("Bm25 Py", "Mxbai Edge Colbert V0 32m") and
// hybrid rows just show the last component instead of the full composition.
const MODEL_LABEL_OVERRIDES = {
  "Alibaba-NLP/gte-large-en-v1.5": "GTE Large v1.5",
  "BAAI/bge-large-en-v1.5": "BGE Large v1.5",
  "BAAI/bge-small-en-v1.5": "BGE Small v1.5",
  "Qwen/Qwen3-Embedding-8B": "Qwen3 Embedding 8B",
  "intfloat/e5-large-v2": "E5 Large v2",
  "nvidia/NV-Embed-v2": "NV-Embed v2",
  "lightonai/GTE-ModernColBERT-v1": "GTE-ModernColBERT v1",
  "lightonai/Reason-ModernColBERT": "Reason-ModernColBERT",
  "lightonai/colbertv2.0": "ColBERTv2",
  "mixedbread-ai/mxbai-edge-colbert-v0-32m": "mxbai Edge ColBERT 32M",
  "bm25-py": "BM25",
};

// Shorter labels for hybrid composition strings ("rrf:a+b+c"). Each component
// is mapped to a concise tag and they're joined with " + ".
const HYBRID_COMPONENT_LABELS = {
  "bm25-py": "BM25",
  "BAAI/bge-large-en-v1.5": "BGE Large",
  "BAAI/bge-small-en-v1.5": "BGE Small",
  "Alibaba-NLP/gte-large-en-v1.5": "GTE Large",
  "intfloat/e5-large-v2": "E5 Large",
  "Qwen/Qwen3-Embedding-8B": "Qwen3 8B",
  "nvidia/NV-Embed-v2": "NV-Embed v2",
  "lightonai/GTE-ModernColBERT-v1": "GTE-ModernColBERT",
  "lightonai/Reason-ModernColBERT": "Reason-ModernColBERT",
  "lightonai/colbertv2.0": "ColBERTv2",
  "mixedbread-ai/mxbai-edge-colbert-v0-32m": "mxbai Edge ColBERT",
};

function prettyModelLabel(modelId, fallback) {
  if (!modelId) return fallback || "";
  if (modelId.startsWith("rrf:")) {
    const components = modelId.slice(4).split("+");
    return components.map((c) => HYBRID_COMPONENT_LABELS[c] || shortenModel(c)).join(" + ");
  }
  return MODEL_LABEL_OVERRIDES[modelId] || fallback || shortenModel(modelId);
}

const state = {
  rows: [],
  sourceType: "demo",
  metric: "ndcg_at_10",
  axis: "latency",
  family: "all",
  benchmark: null,
  datasetByBenchmark: {},
  sortKey: "quality_selected",
  sortDirection: "desc",
  chartContextKey: "",
  chartFadeInFlight: false,
  chartRenderVersion: 0,
  chartLastAnimatedVersion: -1,
};

function prefersReducedMotion() {
  return window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

document.addEventListener("DOMContentLoaded", init);

async function init() {
  const payload = await loadPayload();
  state.sourceType = payload.source_type || "demo";
  state.rows = normalizePayload(payload).filter(hasUsableQuality).map(withSelectedMetric);
  state.benchmark = pickInitialBenchmark(state.rows);
  bindControls();
  renderAll();
  enhanceFilters();
  setupPaperGrain();
}

// Interactive paper-grain particle field. Particles render as warm dark dots
// across the viewport. Each has a rest position and is gently pushed away from
// the cursor on approach, then springs back when the cursor leaves. Idle drift
// keeps the grain looking organic even when the cursor is still.
function setupPaperGrain() {
  const canvas = document.getElementById("paperGrain");
  if (!canvas) return;
  const ctx = canvas.getContext("2d", { alpha: true });
  if (!ctx) return;

  const reduced = prefersReducedMotion();
  const isTouch = window.matchMedia && window.matchMedia("(hover: none)").matches;

  // Tunables — feel free to tweak these to taste.
  const REPULSION_RADIUS = 135;
  const REPULSION_STRENGTH = 0.42;
  const SPRING = 0.045;
  const DAMPING = 0.86;
  const DRIFT = 0.025;

  let dpr = Math.min(window.devicePixelRatio || 1, 2);
  let viewportWidth = 0;
  let viewportHeight = 0;
  let particles = [];
  let mouseX = -10000;
  let mouseY = -10000;
  let cursorActive = false;
  let lastFrame = 0;

  function targetParticleCount() {
    const area = viewportWidth * viewportHeight;
    // ~1 particle per 180 px² of viewport (denser than before by ~22%),
    // capped for huge displays / floored for phones.
    return Math.max(1800, Math.min(8500, Math.round(area / 180)));
  }

  function seedParticles() {
    const count = targetParticleCount();
    particles = new Array(count);
    for (let i = 0; i < count; i++) {
      const x = Math.random() * viewportWidth;
      const y = Math.random() * viewportHeight;
      particles[i] = {
        x,
        y,
        baseX: x,
        baseY: y,
        vx: 0,
        vy: 0,
        size: 0.55 + Math.random() * 0.95,
        alpha: 0.22 + Math.random() * 0.38,
      };
    }
  }

  function resize() {
    viewportWidth = window.innerWidth;
    viewportHeight = window.innerHeight;
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(viewportWidth * dpr);
    canvas.height = Math.round(viewportHeight * dpr);
    canvas.style.width = `${viewportWidth}px`;
    canvas.style.height = `${viewportHeight}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    seedParticles();
    if (reduced) renderStatic();
  }

  function renderStatic() {
    ctx.clearRect(0, 0, viewportWidth, viewportHeight);
    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      ctx.fillStyle = `rgba(28, 22, 14, ${p.alpha.toFixed(3)})`;
      ctx.fillRect(p.x, p.y, p.size, p.size);
    }
  }

  function tick(now) {
    const elapsed = lastFrame ? Math.min(40, now - lastFrame) : 16;
    lastFrame = now;
    const dt = elapsed / 16; // normalize to ~60fps

    ctx.clearRect(0, 0, viewportWidth, viewportHeight);

    const radiusSq = REPULSION_RADIUS * REPULSION_RADIUS;
    const cursorOn = cursorActive;
    const cx = mouseX;
    const cy = mouseY;

    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];

      // Idle brownian drift so the field is alive even at rest.
      p.vx += (Math.random() - 0.5) * DRIFT * dt;
      p.vy += (Math.random() - 0.5) * DRIFT * dt;

      // Cursor repulsion within radius, force ∝ (radius - dist) / radius.
      if (cursorOn) {
        const dx = p.x - cx;
        const dy = p.y - cy;
        const distSq = dx * dx + dy * dy;
        if (distSq < radiusSq && distSq > 0.01) {
          const dist = Math.sqrt(distSq);
          const force = ((REPULSION_RADIUS - dist) / REPULSION_RADIUS) * REPULSION_STRENGTH;
          p.vx += (dx / dist) * force * dt;
          p.vy += (dy / dist) * force * dt;
        }
      }

      // Spring-back to rest position + damping.
      p.vx += (p.baseX - p.x) * SPRING * dt;
      p.vy += (p.baseY - p.y) * SPRING * dt;
      const damp = Math.pow(DAMPING, dt);
      p.vx *= damp;
      p.vy *= damp;

      p.x += p.vx * dt;
      p.y += p.vy * dt;

      ctx.fillStyle = `rgba(28, 22, 14, ${p.alpha.toFixed(3)})`;
      ctx.fillRect(p.x, p.y, p.size, p.size);
    }

    requestAnimationFrame(tick);
  }

  const root = document.documentElement;
  let lightTargetX = window.innerWidth / 2;
  let lightTargetY = window.innerHeight / 2;
  let lightCurrentX = lightTargetX;
  let lightCurrentY = lightTargetY;
  let lightRaf = null;

  function lightTick() {
    lightCurrentX += (lightTargetX - lightCurrentX) * 0.16;
    lightCurrentY += (lightTargetY - lightCurrentY) * 0.16;
    root.style.setProperty("--mouse-x", `${lightCurrentX}px`);
    root.style.setProperty("--mouse-y", `${lightCurrentY}px`);
    if (Math.abs(lightTargetX - lightCurrentX) > 0.4 || Math.abs(lightTargetY - lightCurrentY) > 0.4) {
      lightRaf = requestAnimationFrame(lightTick);
    } else {
      lightRaf = null;
    }
  }

  if (!isTouch) {
    document.addEventListener("mousemove", (event) => {
      mouseX = event.clientX;
      mouseY = event.clientY;
      cursorActive = true;
      lightTargetX = event.clientX;
      lightTargetY = event.clientY;
      if (!lightRaf && !reduced) lightRaf = requestAnimationFrame(lightTick);
    });
    document.addEventListener("mouseleave", () => {
      cursorActive = false;
    });
    window.addEventListener("blur", () => {
      cursorActive = false;
    });
  }

  let resizeTimer = null;
  window.addEventListener("resize", () => {
    if (resizeTimer) clearTimeout(resizeTimer);
    resizeTimer = setTimeout(resize, 120);
  });

  resize();
  if (!reduced) requestAnimationFrame(tick);
}

function hasUsableQuality(row) {
  if (!row.quality) return false;
  return Object.values(row.quality).some(isFiniteNumber);
}

async function loadPayload() {
  for (const candidate of [
    { url: "data/results.json", type: "real" },
    { url: "data/demo-results.json", type: "demo" },
  ]) {
    try {
      const response = await fetch(candidate.url, { cache: "no-store" });
      if (!response.ok) continue;
      const payload = await response.json();
      if (Array.isArray(payload.runs) && payload.runs.length) {
        return { ...payload, source_type: candidate.type };
      }
    } catch (_error) {
      // Embedded fallback covers direct file opens or temporary fetch failures.
    }
  }
  return { ...DEMO_FALLBACK, source_type: "demo" };
}

function pickInitialBenchmark(rows) {
  const order = ["beir", "bright", "limit", "lotte"];
  const found = unique(rows.map((row) => row.benchmark));
  for (const candidate of order) if (found.includes(candidate)) return candidate;
  return found[0] || null;
}

function normalizePayload(payload) {
  return (Array.isArray(payload) ? payload : payload.runs || [])
    .map((run, index) => normalizeRun(run, index))
    .filter(Boolean);
}

function normalizeRun(run, index) {
  if (run.model && run.dataset && run.quality) {
    return normalizeBenchmarkResult(run, index);
  }

  const storageBytes = toNumber(run.storage?.index_bytes ?? run.storage?.index_bytes_backend);
  return {
    id: run.id || `row-${index}`,
    benchmark: clean(run.benchmark, "unknown"),
    dataset: clean(run.dataset, "unknown"),
    dataset_label: clean(run.dataset_label, titleCase(run.dataset || "unknown")),
    model_id: clean(run.model_id, "unknown-model"),
    model_label: prettyModelLabel(run.model_id, run.model_label),
    family: normalizeFamily(run.family),
    params_m: toNumber(run.params_m),
    system: clean(run.system, "unknown"),
    system_label: systemLabel(run.system || "unknown"),
    backend: clean(run.backend, run.system || "unknown"),
    compression: clean(run.compression, "unknown"),
    exact: Boolean(run.exact),
    status: normalizeStatus(run.status, run.latency, run.quality),
    quality: normalizeQuality(run.quality),
    latency: run.latency || {},
    storage: run.storage || {},
    cost: run.cost || {},
    protocol: run.protocol || {},
    result_path: run.result_path || "",
    latency_ms: toNumber(run.latency?.e2e_query_ms_p50),
    latency_p99_ms: toNumber(run.latency?.e2e_query_ms_p99),
    storage_gb: storageBytes == null ? null : storageBytes / 1_000_000_000,
    cost_per_million_queries_usd: toNumber(run.cost?.cost_per_million_queries_usd),
  };
}

function normalizeBenchmarkResult(result, index) {
  const model = result.model || {};
  const dataset = result.dataset || {};
  const system = result.retrieval_system || {};
  const systemName = system.name || inferSystemName(model);
  const storageBytes = pickIndexBytes(result.storage || {});
  return {
    id: `${dataset.benchmark}:${dataset.name}:${model.id}:${systemName}:${index}`,
    benchmark: clean(dataset.benchmark, "unknown"),
    dataset: clean(dataset.name, "unknown"),
    dataset_label: datasetLabel(dataset.benchmark, dataset.name),
    model_id: clean(model.id, "unknown-model"),
    model_label: modelLabel(model.id || "unknown-model", result),
    family: normalizeFamily(system.family || model.family),
    params_m: toNumber(model.params_m),
    system: systemName,
    system_label: systemLabel(systemName),
    backend: clean(system.backend, systemName),
    compression: clean(system.compression, storageCompression(result.storage || {})),
    exact: Boolean(system.exact),
    status: normalizeStatus(result.status, result.latency, result.quality),
    quality: normalizeQuality(result.quality),
    latency: result.latency || {},
    storage: result.storage || {},
    cost: result.cost || {},
    protocol: {
      hardware: result.run?.hardware_label || result.system?.platform || "unknown",
      precision: result.run?.precision || result.precision || "unknown",
      created_at: result.run?.created_at,
    },
    result_path: result.result_path || "",
    latency_ms: toNumber(result.latency?.e2e_query_ms_p50),
    latency_p99_ms: toNumber(result.latency?.e2e_query_ms_p99),
    storage_gb: storageBytes == null ? null : storageBytes / 1_000_000_000,
    cost_per_million_queries_usd: toNumber(result.cost?.cost_per_million_queries_usd),
  };
}

function bindControls() {
  byId("metricSelect").addEventListener("change", (event) => {
    state.metric = event.target.value;
    state.rows = state.rows.map(withSelectedMetric);
    renderAll();
  });
  byId("axisSelect").addEventListener("change", (event) => {
    state.axis = event.target.value;
    renderAll();
  });
  byId("familySelect").addEventListener("change", (event) => {
    state.family = event.target.value;
    renderAll();
  });
  byId("drawerClose").addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeDrawer();
  });

  document.querySelectorAll("#systemsTable th[data-sort]").forEach((th) => {
    th.addEventListener("click", () => {
      const key = th.dataset.sort;
      if (state.sortKey === key) {
        state.sortDirection = state.sortDirection === "asc" ? "desc" : "asc";
      } else {
        state.sortKey = key;
        state.sortDirection = ["model_label", "dataset_label", "system_label", "status"].includes(key)
          ? "asc"
          : "desc";
      }
      renderTableOnly();
    });
  });
}

function renderAll() {
  renderFamilyOptions();
  renderBenchmarkTabs();
  if (!state.benchmark) {
    renderStatLine([]);
    renderEmptyLeaderboard();
    return;
  }
  renderBenchmarkIntro();
  renderChartTitle();
  const rows = currentBenchmarkRows();
  // Leader stat reflects whatever the chart and table are showing (current
  // benchmark + dataset + family filter), not the global best.
  renderStatLine(rows);
  refreshRowMap(rows);
  renderChart(rows);
  renderFrontierStrip(rows);
  renderTable(rows);
}

function refreshRowMap(displayed) {
  state.rowMap = new Map();
  for (const row of state.rows) state.rowMap.set(row.id, row);
  for (const row of displayed) state.rowMap.set(row.id, row);
}

function findRow(id) {
  return state.rowMap?.get(id) || state.rows.find((row) => row.id === id);
}

function renderTableOnly() {
  const rows = currentBenchmarkRows();
  renderTable(rows);
}

function renderFamilyOptions() {
  const families = ["all", ...unique(state.rows.map((row) => row.family)).sort()];
  const select = byId("familySelect");
  const current = select.value || state.family;
  select.innerHTML = families
    .map((family) => {
      const label = family === "all" ? "All families" : FAMILY[family]?.label || titleCase(family);
      return `<option value="${escapeAttr(family)}">${escapeHtml(label)}</option>`;
    })
    .join("");
  select.value = families.includes(current) ? current : "all";
  syncCustomDropdown(select);
}

function renderStatLine(rows) {
  const best = maxBy(rows.filter((row) => isFiniteNumber(row.quality_selected)), (row) => row.quality_selected);
  const leader = byId("statLeader");
  if (best) {
    leader.innerHTML = `<span class="leader-tag">Top</span><span class="stat">${escapeHtml(best.model_label)} <span class="leader-sep">·</span> <strong>${formatNumber(best.quality_selected, 3)}</strong> ${escapeHtml(METRICS[state.metric])}</span>`;
  } else {
    leader.innerHTML = "";
  }
}

function renderBenchmarkTabs() {
  const benchmarks = unique(state.rows.map((row) => row.benchmark)).sort(benchmarkSort);
  const container = byId("benchmarkTabs");
  if (!benchmarks.length) {
    container.innerHTML = "";
    return;
  }
  if (!benchmarks.includes(state.benchmark)) state.benchmark = benchmarks[0];

  // Capture pre-render indicator position so the FLIP slides from the old tab.
  const existingIndicator = container.querySelector(".tab-indicator");
  const fromRect = existingIndicator ? existingIndicator.getBoundingClientRect() : null;
  const containerRect = container.getBoundingClientRect();

  container.innerHTML = benchmarks
    .map((benchmark) => {
      const isSelected = benchmark === state.benchmark;
      return `<button type="button" role="tab" data-benchmark="${escapeAttr(benchmark)}" aria-selected="${isSelected ? "true" : "false"}">
        <span class="tab-name">${escapeHtml(BENCHMARK_TITLE[benchmark] || benchmark.toUpperCase())}</span>
      </button>`;
    })
    .join("");

  const indicator = document.createElement("span");
  indicator.className = "tab-indicator";
  container.appendChild(indicator);

  positionTabIndicator(container, indicator, fromRect, containerRect);

  container.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      state.benchmark = button.dataset.benchmark;
      renderAll();
    });
  });
}

function positionTabIndicator(container, indicator, fromRect, prevContainerRect) {
  const activeButton = container.querySelector('button[aria-selected="true"]');
  if (!activeButton) {
    indicator.style.width = "0";
    return;
  }
  const containerRect = container.getBoundingClientRect();
  const targetRect = activeButton.getBoundingClientRect();
  const targetX = targetRect.left - containerRect.left + container.scrollLeft;
  const targetWidth = targetRect.width;

  if (fromRect && prevContainerRect) {
    // Start at the previous indicator's geometry (relative to the new container),
    // then transition to the target position on the next frame.
    const fromX = fromRect.left - prevContainerRect.left;
    indicator.style.transition = "none";
    indicator.style.transform = `translateX(${fromX}px)`;
    indicator.style.width = `${fromRect.width}px`;
    // Force reflow so the starting state is committed before transition kicks in.
    indicator.getBoundingClientRect();
    requestAnimationFrame(() => {
      indicator.style.transition = "";
      indicator.classList.add("animated");
      indicator.style.transform = `translateX(${targetX}px)`;
      indicator.style.width = `${targetWidth}px`;
    });
  } else {
    // First paint: snap into place without animating.
    indicator.style.transform = `translateX(${targetX}px)`;
    indicator.style.width = `${targetWidth}px`;
    requestAnimationFrame(() => {
      indicator.classList.add("animated");
    });
  }
}

function renderBenchmarkIntro() {
  const container = byId("benchmarkIntro");
  const benchmark = state.benchmark;
  const benchmarkRows = state.rows.filter((row) => row.benchmark === benchmark);
  const datasets = unique(benchmarkRows.map((row) => row.dataset)).sort();

  ensureSelectedDataset(benchmark, datasets);
  const selectedDataset = state.datasetByBenchmark[benchmark] || "all";

  const chipKeys = datasets.length > 1 ? ["all", ...datasets] : datasets;
  const onlyOne = chipKeys.length === 1;
  const datasetChips = `<div class="dataset-chips" role="tablist" aria-label="Datasets">
      <span class="dataset-chips-label">Dataset</span>
      ${chipKeys
        .map((dataset) => {
          const pressed = onlyOne ? "true" : (dataset === selectedDataset ? "true" : "false");
          const label = dataset === "all" ? "Complete configs" : datasetLabelFor(dataset, benchmarkRows);
          const disabled = onlyOne ? "disabled" : "";
          return `<button type="button" data-dataset="${escapeAttr(dataset)}" aria-pressed="${pressed}" ${disabled}>${escapeHtml(label)}</button>`;
        })
        .join("")}
    </div>`;

  container.innerHTML = `
    <div>
      <p>${escapeHtml(BENCHMARK_COPY[benchmark] || "Retrieval benchmark.")}</p>
      ${datasetChips}
    </div>
  `;

  container.querySelectorAll(".dataset-chips button").forEach((button) => {
    button.addEventListener("click", () => {
      state.datasetByBenchmark[benchmark] = button.dataset.dataset;
      renderAll();
    });
  });
}

function ensureSelectedDataset(benchmark, datasets) {
  const selected = state.datasetByBenchmark[benchmark];
  if (selected && selected !== "all" && !datasets.includes(selected)) {
    state.datasetByBenchmark[benchmark] = "all";
  }
}

function renderChartTitle() {
  const axis = AXES[state.axis];
  byId("chartTitle").textContent = `${METRICS[state.metric]} vs. ${axis.short}`;
}

function renderEmptyLeaderboard() {
  byId("benchmarkIntro").innerHTML = "";
  byId("chartTitle").textContent = "—";
  const plot = byId("plot");
  if (window.Plotly) Plotly.purge("plot");
  plot.innerHTML = '<div class="empty">No data loaded</div>';
  byId("frontierStrip").innerHTML = "";
  document.querySelector("#systemsTable tbody").innerHTML = "";
  byId("tableCount").textContent = "0 rows";
  byId("chartLegend").innerHTML = "";
}

function currentBenchmarkRows() {
  const benchmark = state.benchmark;
  const selectedDataset = state.datasetByBenchmark[benchmark] || "all";
  const benchmarkRows = filteredRows().filter((row) => row.benchmark === benchmark);
  if (selectedDataset !== "all") {
    return benchmarkRows.filter((row) => row.dataset === selectedDataset);
  }
  const datasets = unique(benchmarkRows.map((row) => row.dataset));
  if (datasets.length <= 1) return benchmarkRows;
  return aggregateAcrossDatasets(benchmarkRows);
}

function aggregateAcrossDatasets(rows) {
  const expectedDatasetCount = unique(rows.map((row) => row.dataset)).length;
  const groups = groupBy(rows, (row) => `${row.family}::${row.model_id}::${row.system}`);
  return Object.values(groups).filter((groupRows) => {
    return unique(groupRows.map((row) => row.dataset)).length === expectedDatasetCount;
  }).map((groupRows) => {
    const first = groupRows[0];
    const count = unique(groupRows.map((row) => row.dataset)).length;
    const aggregatedQuality = aggregateQuality(groupRows);
    const allCompleted = groupRows.every((r) => r.status === "completed");
    return {
      ...first,
      id: `agg:${first.benchmark}:${first.family}:${first.model_id}:${first.system}`,
      dataset: "_aggregated",
      dataset_label: `${count} ${count === 1 ? "dataset" : "datasets"} · avg`,
      quality: aggregatedQuality,
      quality_selected: aggregatedQuality[state.metric] ?? null,
      latency_ms: meanFinite(groupRows.map((r) => r.latency_ms)),
      latency_p99_ms: meanFinite(groupRows.map((r) => r.latency_p99_ms)),
      storage_gb: meanFinite(groupRows.map((r) => r.storage_gb)),
      cost_per_million_queries_usd: meanFinite(groupRows.map((r) => r.cost_per_million_queries_usd)),
      status: allCompleted ? "completed" : "quality_only",
      __aggregated: true,
      __underlying: groupRows,
      __dataset_count: count,
    };
  });
}

function aggregateQuality(rows) {
  return {
    ndcg_at_10: meanFinite(rows.map((r) => r.quality?.ndcg_at_10)),
    recall_at_100: meanFinite(rows.map((r) => r.quality?.recall_at_100)),
    mrr_at_10: meanFinite(rows.map((r) => r.quality?.mrr_at_10)),
    map_at_100: meanFinite(rows.map((r) => r.quality?.map_at_100)),
  };
}

function meanFinite(values) {
  const finite = values.filter((v) => isFiniteNumber(v));
  if (!finite.length) return null;
  return finite.reduce((sum, value) => sum + value, 0) / finite.length;
}

function renderChart(rows) {
  const axis = AXES[state.axis];
  const plot = byId("plot");
  const plotRows = rows
    .filter((row) => isFiniteNumber(row.quality_selected))
    .filter((row) => isFiniteNumber(row[axis.field]));

  if (!window.Plotly) {
    plot.innerHTML = '<div class="empty">Plot library unavailable</div>';
    byId("chartLegend").innerHTML = "";
    return;
  }
  if (!plotRows.length) {
    Plotly.purge("plot");
    plot.innerHTML = '<div class="empty">No measurements on this axis</div>';
    byId("chartLegend").innerHTML = "";
    return;
  }

  const frontier = paretoFrontier(plotRows, axis.field);
  const frontierIds = new Set(frontier.map((row) => row.id));

  const traces = Object.entries(groupBy(plotRows, (row) => `${row.family}:${row.model_id}`)).map(([, groupRows]) => {
    groupRows.sort((a, b) => a[axis.field] - b[axis.field]);
    const family = FAMILY[groupRows[0].family] || FAMILY.unknown;
    return {
      type: "scatter",
      mode: "markers",
      name: groupRows[0].model_label,
      x: groupRows.map((row) => row[axis.field]),
      y: groupRows.map((row) => row.quality_selected),
      customdata: groupRows.map((row) => row.id),
      text: groupRows.map((row) => `${row.model_label} · ${row.system_label}<br>${row.dataset_label}`),
      hovertemplate: `<b>%{text}</b><br>${axis.title}: %{x}<br>${METRICS[state.metric]}: %{y:.3f}<extra></extra>`,
      marker: {
        color: family.color,
        symbol: "circle",
        size: groupRows.map((row) => (frontierIds.has(row.id) ? 16 : 12)),
        opacity: groupRows.map((row) => {
          if (frontierIds.has(row.id)) return 1;
          return row.status === "completed" ? 0.7 : 0.45;
        }),
        line: {
          color: groupRows.map((row) => (frontierIds.has(row.id) ? "#14110b" : "#f5f1e8")),
          width: groupRows.map((row) => (frontierIds.has(row.id) ? 1.6 : 1.2)),
        },
      },
      showlegend: false,
    };
  });

  if (frontier.length >= 2) {
    traces.push({
      type: "scatter",
      mode: "lines",
      name: "Pareto frontier",
      x: frontier.map((row) => row[axis.field]),
      y: frontier.map((row) => row.quality_selected),
      hoverinfo: "skip",
      line: { color: "#14110b", width: 1.6, shape: "spline", smoothing: 0.6 },
      showlegend: false,
    });
  }

  const xValues = plotRows.map((row) => row[axis.field]);
  const yValues = plotRows.map((row) => row.quality_selected);
  const yMax = Math.max(...yValues.filter(isFiniteNumber));
  const xMin = Math.min(...xValues.filter(isFiniteNumber));

  // Detect a structural change by *filter context*, not trace identity.
  // BEIR and BRIGHT share model labels, so trace-name detection misses
  // Any filter change — benchmark, dataset, family, status, metric, axis —
  // triggers the JS crossfade. Plotly's built-in tween produced big vertical
  // jumps when the metric scale changed (nDCG → Recall) and big horizontal
  // jumps when the axis changed (latency → storage), which read as "points
  // flying" rather than a smooth transition. The crossfade is uniform and
  // clean across every kind of switch.
  const contextKey = [
    state.benchmark || "",
    state.datasetByBenchmark[state.benchmark] || "all",
    state.family,
    state.metric,
    state.axis,
  ].join("::");
  const isStructural = state.chartContextKey !== contextKey;
  state.chartContextKey = contextKey;
  if (isStructural) state.chartRenderVersion += 1;

  const layout = {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      margin: { t: 28, r: 24, b: 56, l: 60 },
      font: { family: "Inter Tight, sans-serif", color: "#14110b", size: 12 },
      transition: {
        // On structural swaps we provide motion via CSS crossfade and the
        // frontier draw-in, so Plotly should snap content instantly.
        duration: prefersReducedMotion() || isStructural ? 0 : 450,
        easing: "cubic-in-out",
      },
      hoverlabel: {
        bgcolor: "#14110b",
        bordercolor: "#14110b",
        font: { family: "Inter Tight, sans-serif", color: "#f5f1e8", size: 12 },
      },
      hovermode: "closest",
      showlegend: false,
      annotations: [
        {
          xref: "paper",
          yref: "paper",
          x: state.axis === "latency" || state.axis === "cost" || state.axis === "storage" ? 0.02 : 0.98,
          y: 0.98,
          text: "↖ better",
          showarrow: false,
          font: { family: "JetBrains Mono, monospace", size: 10, color: "#a23b17" },
          xanchor: "left",
          yanchor: "top",
        },
      ],
      xaxis: {
        title: { text: axis.title, font: { family: "JetBrains Mono, monospace", size: 10, color: "#6c655a" }, standoff: 16 },
        type: shouldUseLog(xValues) ? "log" : "linear",
        gridcolor: "rgba(108, 101, 90, 0.14)",
        gridwidth: 1,
        zeroline: false,
        linecolor: "rgba(108, 101, 90, 0.55)",
        tickcolor: "rgba(108, 101, 90, 0.55)",
        ticks: "outside",
        ticklen: 5,
        tickfont: { family: "JetBrains Mono, monospace", size: 10, color: "#6c655a" },
        nticks: 6,
        showspikes: false,
        ...(shouldUseLog(xValues) ? logTickConfig(xValues) : {}),
      },
      yaxis: {
        title: { text: METRICS[state.metric], font: { family: "JetBrains Mono, monospace", size: 10, color: "#6c655a" }, standoff: 16 },
        range: yRange(yValues),
        gridcolor: "rgba(108, 101, 90, 0.14)",
        gridwidth: 1,
        zeroline: false,
        linecolor: "rgba(108, 101, 90, 0.55)",
        tickcolor: "rgba(108, 101, 90, 0.55)",
        ticks: "outside",
        ticklen: 5,
        tickfont: { family: "JetBrains Mono, monospace", size: 10, color: "#6c655a" },
        nticks: 6,
        tickformat: ".2f",
      },
    };
  const config = { displayModeBar: false, responsive: true };

  updateChart(plot, traces, layout, config, isStructural);

  renderChartLegend(plotRows);
}

// Plotly attaches its .on event-emitter API to the plot element only after
// Plotly.react / newPlot has run. We defer hook installation so that callers
// (including the structural-change path that defers react inside a setTimeout)
// don't try to call plot.on before it exists.
function attachPlotHooks(plot) {
  if (typeof plot.on !== "function") return;
  if (!plot.__clickHooked) {
    plot.__clickHooked = true;
    plot.on("plotly_click", (event) => {
      const id = event.points?.[0]?.customdata;
      const row = findRow(id);
      if (row) openDrawer(row);
    });
  }
  if (!plot.__frontierDrawHooked) {
    plot.__frontierDrawHooked = true;
    plot.on("plotly_afterplot", () => animateFrontierDrawIfNeeded(plot));
  }
}

// Draws the Pareto frontier line in left-to-right via SVG stroke-dashoffset.
// Plotly fires plotly_afterplot on every redraw (including hover), so a
// version guard ensures the animation only runs once per structural render.
function animateFrontierDrawIfNeeded(plot) {
  if (state.chartLastAnimatedVersion === state.chartRenderVersion) return;
  state.chartLastAnimatedVersion = state.chartRenderVersion;
  if (prefersReducedMotion()) return;

  const paths = plot.querySelectorAll(".scatterlayer .trace path.js-line");
  let frontierPath = null;
  paths.forEach((path) => {
    const stroke = (path.getAttribute("style") || "") + " " + (path.getAttribute("stroke") || "");
    if (stroke.includes("rgb(20, 17, 11)") || stroke.includes("#14110b")) {
      frontierPath = path;
    }
  });
  if (!frontierPath || typeof frontierPath.getTotalLength !== "function") return;

  const length = frontierPath.getTotalLength();
  if (!length) return;
  frontierPath.style.transition = "none";
  frontierPath.style.strokeDasharray = `${length}`;
  frontierPath.style.strokeDashoffset = `${length}`;
  // Force reflow so the offset is committed before the transition begins.
  frontierPath.getBoundingClientRect();
  frontierPath.style.transition = "stroke-dashoffset 700ms cubic-bezier(0.2, 0.7, 0.2, 1)";
  frontierPath.style.strokeDashoffset = "0";
}

// Wrapper around Plotly.react that crossfades `#plot` for structural changes
// (different trace count or different trace identity). For non-structural
// changes (metric/axis swap with same trace structure) we delegate straight
// to Plotly so its built-in transition can tween values in place.
function updateChart(plot, traces, layout, config, isStructural) {
  const reduced = prefersReducedMotion();

  if (!isStructural || reduced) {
    Plotly.react("plot", traces, layout, config);
    attachPlotHooks(plot);
    return;
  }

  // If a fade is already mid-flight (rapid tab switching), cancel it and
  // run the new render synchronously so we don't stack timeouts.
  if (state.chartFadeInFlight) {
    state.chartFadeInFlight = false;
    plot.classList.remove("is-fading", "is-restoring");
    Plotly.react("plot", traces, layout, config);
    attachPlotHooks(plot);
    return;
  }

  state.chartFadeInFlight = true;
  plot.classList.remove("is-restoring");
  plot.classList.add("is-fading");

  // After the fade-out (180ms), swap the trace set with transition 0 so
  // Plotly snaps the new state in instantly — the JS provides the motion.
  window.setTimeout(() => {
    Plotly.react("plot", traces, layout, config);
    attachPlotHooks(plot);
    // Next frame: clear the fade-out class and add restoring so opacity/
    // transform tween back to rest. The 240ms transition is defined in CSS.
    requestAnimationFrame(() => {
      plot.classList.remove("is-fading");
      plot.classList.add("is-restoring");
      // Drop the restoring class once it has finished so subsequent
      // metric/axis swaps don't carry stale transition CSS.
      window.setTimeout(() => {
        plot.classList.remove("is-restoring");
        state.chartFadeInFlight = false;
      }, 240);
    });
  }, 180);
}

function renderChartLegend(rows) {
  const families = unique(rows.map((row) => row.family));
  const swatches = families
    .map((family) => `<span class="swatch ${escapeAttr(family)}">${escapeHtml(FAMILY[family]?.label || titleCase(family))}</span>`)
    .join("");
  const frontier = '<span class="swatch frontier">Pareto frontier</span>';
  byId("chartLegend").innerHTML = `${swatches}${frontier}`;
}

function renderFrontierStrip(rows) {
  const axis = AXES[state.axis];
  const candidates = rows
    .filter((row) => isFiniteNumber(row.quality_selected))
    .filter((row) => isFiniteNumber(row[axis.field]));
  const frontier = paretoFrontier(candidates, axis.field);
  const labels = FRONTIER_LABELS[state.axis] || FRONTIER_LABELS.latency;
  const picks = pickThreeFrontier(frontier, axis.field);

  const container = byId("frontierStrip");
  if (!picks.length) {
    container.innerHTML = "";
    return;
  }

  container.innerHTML = picks
    .map((row, index) => {
      const eyebrowText = picks.length === 1 ? "Frontier" : labels[index];
      // The middle pick uses a non-obvious knee-point heuristic; surface a
      // footnote-style link to the methodology so the formula is one click away.
      const isBalance = picks.length === 3 && index === 1;
      const eyebrowMarkup = isBalance
        ? `${escapeHtml(eyebrowText)}<a class="frontier-note" href="methodology.html#best-balance" aria-label="How best balance is computed">*</a>`
        : escapeHtml(eyebrowText);
      const qualityBlock = `<span class="frontier-stat"><span>${escapeHtml(METRICS[state.metric])}</span>${formatMaybe(row.quality_selected, 3)}</span>`;
      const axisBlock = `<span class="frontier-stat"><span>${escapeHtml(axis.label)}</span>${formatAxisValue(row, axis)}</span>`;
      return `<button class="frontier-pick" type="button" data-id="${escapeAttr(row.id)}">
        <span class="frontier-eyebrow">${eyebrowMarkup}</span>
        <span class="frontier-name">${escapeHtml(row.model_label)}</span>
        <span class="frontier-system">${escapeHtml(row.system_label)} · ${escapeHtml(row.dataset_label)}</span>
        <span class="frontier-stats">${qualityBlock}${axisBlock}</span>
      </button>`;
    })
    .join("");

  container.querySelectorAll("button.frontier-pick").forEach((button) => {
    button.addEventListener("click", () => {
      const row = findRow(button.dataset.id);
      if (row) openDrawer(row);
    });
  });

  // The asterisk link nested inside the Best-balance card should navigate
  // to the methodology section, not open the drawer.
  container.querySelectorAll("a.frontier-note").forEach((note) => {
    note.addEventListener("click", (event) => event.stopPropagation());
  });
}

function pickThreeFrontier(frontier, axisField) {
  if (!frontier.length) return [];
  if (frontier.length <= 3) return frontier;
  const first = frontier[0];
  const last = frontier[frontier.length - 1];
  const knee = findKneePoint(frontier, axisField) || frontier[Math.floor(frontier.length / 2)];
  return [first, knee, last];
}

// Knee-point of the Pareto frontier — the system that gives up the most
// quality per unit of axis cost relative to its neighbours. Both axes are
// normalised to [0, 1], we draw a chord between the cheapest and the
// highest-quality frontier members, then pick the frontier point with the
// largest perpendicular distance above that chord.
function findKneePoint(frontier, axisField) {
  if (frontier.length < 3) return null;
  const xs = frontier.map((p) => p[axisField]);
  const ys = frontier.map((p) => p.quality_selected);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
  const xRange = xMax - xMin || 1;
  const yRange = yMax - yMin || 1;
  const ax = 0;
  const ay = (frontier[0].quality_selected - yMin) / yRange;
  const bx = 1;
  const by = (frontier[frontier.length - 1].quality_selected - yMin) / yRange;
  const dx = bx - ax;
  const dy = by - ay;
  let best = null;
  let bestDist = -Infinity;
  for (let i = 1; i < frontier.length - 1; i++) {
    const px = (frontier[i][axisField] - xMin) / xRange;
    const py = (frontier[i].quality_selected - yMin) / yRange;
    // Signed perpendicular distance, positive when point sits above the chord.
    const dist = (dx * (py - ay)) - (dy * (px - ax));
    if (dist > bestDist) {
      bestDist = dist;
      best = frontier[i];
    }
  }
  return best;
}

function renderTable(rows) {
  const tbody = document.querySelector("#systemsTable tbody");
  const sorted = [...rows].sort((a, b) => compareRows(a, b, state.sortKey, state.sortDirection));
  renderTableHeading();
  byId("tableCount").textContent = `${sorted.length} ${sorted.length === 1 ? "row" : "rows"}`;

  if (!sorted.length) {
    tbody.innerHTML = `<tr><td colspan="6" class="empty" style="min-height:120px">No rows match the current filters</td></tr>`;
  } else {
    tbody.innerHTML = sorted.map(tableRowHtml).join("");
    tbody.querySelectorAll("tr[data-id]").forEach((tr) => {
      tr.addEventListener("click", () => {
        const row = findRow(tr.dataset.id);
        if (row) openDrawer(row);
      });
    });
  }

  document.querySelectorAll("#systemsTable th[data-sort]").forEach((th) => {
    th.removeAttribute("aria-sort");
    th.removeAttribute("data-arrow");
    if (th.dataset.sort === state.sortKey) {
      th.setAttribute("aria-sort", state.sortDirection === "asc" ? "ascending" : "descending");
      th.setAttribute("data-arrow", state.sortDirection === "asc" ? "↑" : "↓");
    }
  });
}

function renderTableHeading() {
  const title = byId("systemsTableTitle");
  if (!title) return;
  const benchmark = state.benchmark;
  const selectedDataset = state.datasetByBenchmark[benchmark] || "all";
  const benchmarkRows = filteredRows().filter((row) => row.benchmark === benchmark);
  const hasMultipleDatasets = unique(benchmarkRows.map((row) => row.dataset)).length > 1;
  title.textContent = selectedDataset === "all" && hasMultipleDatasets
    ? "Complete configurations"
    : "All measured systems";
}

function tableRowHtml(row) {
  return `
    <tr data-id="${escapeAttr(row.id)}">
      <td class="cell-model">
        <span class="family-glyph ${escapeAttr(row.family)}">
          <span><strong>${escapeHtml(row.model_label)}</strong><small>${escapeHtml(FAMILY[row.family]?.label || row.family)}</small></span>
        </span>
      </td>
      <td>${escapeHtml(row.dataset_label)}</td>
      <td class="cell-system"><strong>${escapeHtml(row.system_label)}</strong><small>${escapeHtml(row.compression)}</small></td>
      <td class="num">${formatMaybe(row.quality_selected, 3)}</td>
      <td class="num">${formatMaybe(row.latency_ms, 1)}</td>
      <td class="num">${formatMaybe(row.storage_gb, 3)}</td>
    </tr>
  `;
}

function openDrawer(row) {
  const drawer = byId("detailDrawer");
  byId("detailBody").innerHTML = `
    <div class="drawer-content">
      <p class="eyebrow">${escapeHtml(BENCHMARK_TITLE[row.benchmark] || row.benchmark.toUpperCase())} · ${escapeHtml(row.dataset_label)}</p>
      <h2>${escapeHtml(row.model_label)}</h2>
      <p class="drawer-system"><strong>${escapeHtml(row.system_label)}</strong> · ${escapeHtml(row.backend)}<br>${escapeHtml(row.compression)} · ${escapeHtml(FAMILY[row.family]?.label || row.family)}</p>

      <section class="drawer-section">
        <h3 class="drawer-section-title">Quality</h3>
        <ul class="drawer-rows">
          <li><span class="label">nDCG@10</span><span class="value">${formatMaybe(row.quality?.ndcg_at_10, 3)}</span></li>
          <li><span class="label">Recall@100</span><span class="value">${formatMaybe(row.quality?.recall_at_100, 3)}</span></li>
          <li><span class="label">MRR@10</span><span class="value">${formatMaybe(row.quality?.mrr_at_10, 3)}</span></li>
          <li><span class="label">MAP@100</span><span class="value">${formatMaybe(row.quality?.map_at_100, 3)}</span></li>
        </ul>
      </section>

      <section class="drawer-section">
        <h3 class="drawer-section-title">Performance</h3>
        <ul class="drawer-rows">
          <li><span class="label">p50 query latency</span><span class="value">${formatMaybe(row.latency_ms, 1)} ms</span></li>
          <li><span class="label">p99 query latency</span><span class="value">${formatMaybe(row.latency_p99_ms, 1)} ms</span></li>
          ${row.latency?.e2e_query_ms_p50_serial != null ? `<li><span class="label">p50 query latency, serial</span><span class="value">${formatMaybe(row.latency.e2e_query_ms_p50_serial, 1)} ms</span></li>` : ""}
          <li><span class="label">Encode p50</span><span class="value">${formatMaybe(row.latency?.query_encode_ms_p50, 1)} ms</span></li>
          <li><span class="label">Retrieve p50</span><span class="value">${formatMaybe(row.latency?.retrieval_ms_p50_topk100, 1)} ms</span></li>
          <li><span class="label">Index size</span><span class="value">${formatMaybe(row.storage_gb, 3)} GB</span></li>
        </ul>
      </section>

      ${componentsSectionHtml(row)}
      ${aggregatedBreakdownHtml(row)}

      <section class="drawer-section">
        <h3 class="drawer-section-title">Setup</h3>
        <ul class="drawer-rows">
          <li><span class="label">Family</span><span class="value">${escapeHtml(FAMILY[row.family]?.label || row.family)}</span></li>
          <li><span class="label">Hardware</span><span class="value">${escapeHtml(row.protocol?.hardware || "unknown")}</span></li>
          <li><span class="label">Precision</span><span class="value">${escapeHtml(row.protocol?.precision || "unknown")}</span></li>
          <li><span class="label">Latency sample</span><span class="value">${formatMaybe(row.latency?.latency_sample_size, 0)} queries</span></li>
        </ul>
      </section>

      <section class="drawer-section">
        <h3 class="drawer-section-title">Provenance</h3>
        <ul class="drawer-rows">
          <li><span class="label">Model ID</span><span class="value">${escapeHtml(row.model_id)}</span></li>
          <li><span class="label">Result file</span><span class="value">${escapeHtml(row.result_path || "pending")}</span></li>
        </ul>
      </section>
    </div>
  `;
  drawer.classList.add("open");
  drawer.setAttribute("aria-hidden", "false");
}

function aggregatedBreakdownHtml(row) {
  if (!row.__aggregated || !Array.isArray(row.__underlying)) return "";
  const items = row.__underlying
    .slice()
    .sort((a, b) => (b.quality_selected ?? -Infinity) - (a.quality_selected ?? -Infinity))
    .map((u) => {
      const lat = u.latency_ms != null ? `${formatNumber(u.latency_ms, 1)} ms` : "--";
      const q = formatMaybe(u.quality_selected, 3);
      return `<li><span class="label">${escapeHtml(u.dataset_label)}</span><span class="value">${q} · ${lat}</span></li>`;
    })
    .join("");
  return `
    <section class="drawer-section">
      <h3 class="drawer-section-title">Per-dataset breakdown</h3>
      <ul class="drawer-rows">${items}</ul>
    </section>
  `;
}

function componentsSectionHtml(row) {
  if (row.family !== "hybrid" && row.family !== "sparse") return "";
  const latency = row.latency || {};
  const storage = row.storage || {};
  const components = [];
  if (latency.bm25_e2e_ms_p50 != null || storage.bm25_index_bytes != null) {
    components.push({
      label: "BM25",
      e2e: latency.bm25_e2e_ms_p50,
      bytes: storage.bm25_index_bytes,
    });
  }
  if (latency.dense_e2e_ms_p50 != null || storage.dense_index_bytes != null) {
    components.push({
      label: "Dense",
      e2e: latency.dense_e2e_ms_p50,
      bytes: storage.dense_index_bytes,
    });
  }
  if (latency.li_e2e_ms_p50 != null || storage.li_index_bytes != null) {
    components.push({
      label: "Late interaction",
      e2e: latency.li_e2e_ms_p50,
      bytes: storage.li_index_bytes,
    });
  }
  if (latency.fusion_ms_p50 != null) {
    components.push({
      label: "Fusion (RRF k=60)",
      e2e: latency.fusion_ms_p50,
      bytes: null,
    });
  }
  if (components.length === 0) return "";

  const rows = components
    .map((component) => {
      const e2e = component.e2e != null ? `${formatNumber(component.e2e, 1)} ms` : "--";
      const bytes = component.bytes != null ? `${formatNumber(component.bytes / 1e9, 3)} GB` : "";
      const valueParts = [e2e, bytes].filter(Boolean).join(" · ");
      return `<li><span class="label">${escapeHtml(component.label)}</span><span class="value">${escapeHtml(valueParts)}</span></li>`;
    })
    .join("");

  return `
    <section class="drawer-section">
      <h3 class="drawer-section-title">Components${row.family === "hybrid" ? " · parallel fanout" : ""}</h3>
      <ul class="drawer-rows">${rows}</ul>
    </section>
  `;
}

function closeDrawer() {
  const drawer = byId("detailDrawer");
  drawer.classList.remove("open");
  drawer.setAttribute("aria-hidden", "true");
}

function paretoFrontier(rows, xField) {
  const candidates = rows
    .filter((row) => isFiniteNumber(row[xField]) && isFiniteNumber(row.quality_selected))
    .sort((a, b) => a[xField] - b[xField]);
  const frontier = [];
  let bestQuality = -Infinity;
  for (const row of candidates) {
    if (row.quality_selected > bestQuality) {
      frontier.push(row);
      bestQuality = row.quality_selected;
    }
  }
  return frontier;
}

function filteredRows() {
  return state.rows.filter((row) => state.family === "all" || row.family === state.family);
}

function withSelectedMetric(row) {
  return { ...row, quality_selected: row.quality?.[state.metric] ?? null };
}

function formatAxisValue(row, axis) {
  if (axis.field === "storage_gb") return `${formatMaybe(row[axis.field], 3)} GB`;
  return `${formatMaybe(row[axis.field], 1)} ms`;
}

function compareRows(a, b, key, direction) {
  const av = key === "quality_selected" ? a.quality_selected : a[key];
  const bv = key === "quality_selected" ? b.quality_selected : b[key];
  const multiplier = direction === "asc" ? 1 : -1;
  if (av == null && bv == null) return 0;
  if (av == null) return 1;
  if (bv == null) return -1;
  if (typeof av === "number" && typeof bv === "number") return (av - bv) * multiplier;
  return String(av).localeCompare(String(bv)) * multiplier;
}

function shouldUseLog(values) {
  const finite = values.filter(isFiniteNumber).filter((v) => v > 0);
  if (finite.length < 2) return false;
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  return max / min >= 30;
}

// Pick clean tick values (1, 2, 5, 10, 20, 50, ...) that fall within the data
// range, so Plotly's log axes don't fall back to abbreviated minor labels
// like "2" and "5" meaning 20 and 50.
function logTickConfig(values) {
  const finite = values.filter(isFiniteNumber).filter((v) => v > 0);
  if (!finite.length) return {};
  const lo = Math.min(...finite);
  const hi = Math.max(...finite);
  const candidates = [
    0.01, 0.02, 0.05, 0.1, 0.2, 0.5,
    1, 2, 5, 10, 20, 50,
    100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000,
  ];
  const tickvals = candidates.filter((v) => v >= lo / 1.5 && v <= hi * 1.5);
  if (tickvals.length < 2) return {};
  const ticktext = tickvals.map((v) => {
    if (v >= 1000) return `${v / 1000}k`;
    if (v >= 1) return String(v);
    return String(v);
  });
  return { tickmode: "array", tickvals, ticktext };
}

function normalizeStatus(status, latency = {}, quality = {}) {
  if (["not_applicable_small_corpus", "failed", "pending", "quality_only", "latency_skipped"].includes(status)) return status;
  if (latency?.status === "skipped") return "quality_only";
  const hasQuality = Object.values(quality || {}).some((value) => isFiniteNumber(toNumber(value)));
  const hasLatency = isFiniteNumber(toNumber(latency?.e2e_query_ms_p50));
  if (status && status !== "completed") return status;
  if (hasQuality && !hasLatency) return "quality_only";
  return "completed";
}

function normalizeQuality(quality = {}) {
  return {
    ndcg_at_10: toNumber(quality.ndcg_at_10),
    recall_at_100: toNumber(quality.recall_at_100),
    mrr_at_10: toNumber(quality.mrr_at_10),
    map_at_100: toNumber(quality.map_at_100),
  };
}

function pickIndexBytes(storage) {
  for (const key of ["index_bytes_backend", "index_bytes_binary_rerank", "index_bytes_muvera", "index_bytes_pq", "index_bytes_hnsw", "index_bytes_fp16", "index_bytes_int8"]) {
    const value = toNumber(storage[key]);
    if (value != null) return value;
  }
  return null;
}

function storageCompression(storage = {}) {
  if (storage.index_bytes_binary_rerank != null) return "1-bit+rerank";
  if (storage.index_bytes_muvera != null) return "muvera";
  if (storage.index_bytes_pq != null) return "pq";
  if (storage.index_bytes_hnsw != null) return "hnsw";
  if (storage.index_bytes_int8 != null) return "int8";
  return "fp16";
}

function inferSystemName(model) {
  if (model?.family === "late_interaction") return model.search_mode === "plaid" ? "li-fastplaid" : "li-exact";
  return "dense-flat";
}

function modelLabel(modelId, result) {
  const truncateDim = result.retrieval_system?.params?.truncate_dim || result.model?.truncate_dim;
  const label = prettyModelLabel(modelId);
  return truncateDim ? `${label} @${truncateDim}d` : label;
}

function shortenModel(modelId = "") {
  return modelId
    .split("/")
    .pop()
    .replaceAll("-", " ")
    .replace(/\bbge\b/i, "BGE")
    .replace(/\bgte\b/i, "GTE")
    .replace(/\bnv\b/i, "NV")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function datasetLabel(benchmark, dataset) {
  if (benchmark === "bright") return `BRIGHT ${titleCase(dataset)}`;
  if (benchmark === "limit") return "LIMIT";
  return titleCase(dataset || "unknown");
}

function datasetLabelFor(dataset, rows) {
  return rows.find((row) => row.dataset === dataset)?.dataset_label || titleCase(dataset);
}

function systemLabel(system) {
  return {
    "dense-flat": "Flat exact",
    "dense-hnsw": "HNSW (high recall)",
    "dense-opq-ivfpq": "OPQ-IVF-PQ",
    "dense-rabitq": "RaBitQ",
    "dense-scann": "ScaNN",
    "dense-binary-rerank": "Binary + rerank",
    "li-exact": "Exact MaxSim",
    "li-fastplaid": "FastPlaid",
    "li-muvera": "MUVERA",
    "sparse-bm25": "BM25 (sparse)",
    "hybrid-rrf-bm25-dense": "RRF · BM25 + dense",
    "hybrid-rrf-bm25-li": "RRF · BM25 + LI",
    "hybrid-rrf-bm25-dense-li": "RRF · BM25 + dense + LI",
    "hybrid-rrf-dense-li": "RRF · dense + LI",
    hybrid: "Hybrid",
  }[system] || titleCase(system);
}

function normalizeFamily(family) {
  return ["dense", "late_interaction", "hybrid", "sparse"].includes(family) ? family : "unknown";
}

function benchmarkSort(a, b) {
  const order = ["beir", "bright", "limit", "lotte"];
  return (order.indexOf(a) === -1 ? 99 : order.indexOf(a)) - (order.indexOf(b) === -1 ? 99 : order.indexOf(b)) || a.localeCompare(b);
}

function yRange(values) {
  const finite = values.filter(isFiniteNumber);
  if (!finite.length) return undefined;
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  const span = max - min;
  const pad = Math.max(0.015, span * 0.1);
  return [Math.max(0, min - pad), Math.min(1, max + pad)];
}

function groupBy(items, keyFn) {
  return items.reduce((acc, item) => {
    const key = keyFn(item);
    acc[key] ||= [];
    acc[key].push(item);
    return acc;
  }, {});
}

function unique(values) {
  return [...new Set(values.filter((value) => value != null && value !== ""))];
}

function minBy(items, fn) {
  return items.reduce((best, item) => (best == null || fn(item) < fn(best) ? item : best), null);
}

function maxBy(items, fn) {
  return items.reduce((best, item) => (best == null || fn(item) > fn(best) ? item : best), null);
}

function toNumber(value) {
  if (value == null || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function isFiniteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function formatNumber(value, digits) {
  const number = toNumber(value);
  if (number == null) return "--";
  return number.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

function formatMaybe(value, digits) {
  return toNumber(value) == null ? "--" : formatNumber(value, digits);
}

function clean(value, fallback) {
  return value == null || value === "" ? fallback : String(value);
}

function titleCase(value = "") {
  return String(value).replaceAll("_", " ").replaceAll("-", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function byId(id) {
  return document.getElementById(id);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function escapeAttr(value) {
  // Quoted attribute values are safe once HTML special chars are escaped.
  // The aggressive regex previously corrupted ids like "...v1.5..." → "...v1_5..."
  // which broke rowMap lookups for table-row and frontier-pick clicks.
  return escapeHtml(value);
}

// === Custom dropdown ======================================================

const CHEVRON_SVG = '<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><path d="M2.5 4.5l3.5 3.5 3.5-3.5" stroke="currentColor" stroke-width="1.4" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>';

function enhanceFilters() {
  document.querySelectorAll(".filter select").forEach(enhanceSelect);
  document.addEventListener("click", handleOutsideClick);
}

function handleOutsideClick(event) {
  document.querySelectorAll(".cs-menu.open").forEach((menu) => {
    const wrapper = menu.parentElement;
    if (wrapper && !wrapper.contains(event.target)) {
      closeDropdown(wrapper);
    }
  });
}

function enhanceSelect(select) {
  if (select.dataset.enhanced === "true") return;
  select.dataset.enhanced = "true";

  const wrapper = document.createElement("div");
  wrapper.className = "cs";

  const trigger = document.createElement("button");
  trigger.type = "button";
  trigger.className = "cs-trigger";
  trigger.setAttribute("aria-haspopup", "listbox");
  trigger.setAttribute("aria-expanded", "false");
  if (select.id) trigger.setAttribute("aria-labelledby", `${select.id}-label`);

  const valueSpan = document.createElement("span");
  valueSpan.className = "cs-value";

  const chevron = document.createElement("span");
  chevron.className = "cs-chevron";
  chevron.innerHTML = CHEVRON_SVG;

  trigger.append(valueSpan, chevron);

  const menu = document.createElement("ul");
  menu.className = "cs-menu";
  menu.setAttribute("role", "listbox");

  wrapper.append(trigger, menu);

  // Connect the label that already lives in the .filter to our new trigger.
  const label = select.parentElement?.querySelector("label");
  if (label && select.id) label.id = `${select.id}-label`;
  if (label) {
    label.htmlFor = "";
    label.addEventListener("click", () => trigger.focus());
    label.style.cursor = "pointer";
  }

  select.style.display = "none";
  select.setAttribute("aria-hidden", "true");
  select.parentElement.insertBefore(wrapper, select);

  let activeIndex = -1;

  function syncTrigger() {
    valueSpan.textContent = currentLabel(select);
  }

  function rebuildMenu() {
    menu.innerHTML = "";
    Array.from(select.options).forEach((opt) => {
      const li = document.createElement("li");
      li.className = "cs-option";
      li.setAttribute("role", "option");
      li.dataset.value = opt.value;
      li.textContent = opt.textContent;
      if (opt.value === select.value) li.setAttribute("aria-selected", "true");
      li.addEventListener("click", () => choose(opt.value));
      li.addEventListener("mouseenter", () => setActive(Array.from(menu.children).indexOf(li)));
      menu.appendChild(li);
    });
  }

  function open() {
    rebuildMenu();
    menu.classList.add("open");
    trigger.setAttribute("aria-expanded", "true");
    const items = menu.children;
    activeIndex = Array.from(items).findIndex((li) => li.getAttribute("aria-selected") === "true");
    if (activeIndex < 0) activeIndex = 0;
    setActive(activeIndex);
  }

  function close() {
    menu.classList.remove("open");
    trigger.setAttribute("aria-expanded", "false");
  }

  function setActive(i) {
    Array.from(menu.children).forEach((li, idx) => {
      li.classList.toggle("cs-active", idx === i);
    });
    activeIndex = i;
    if (menu.children[i]) menu.children[i].scrollIntoView({ block: "nearest" });
  }

  function choose(value) {
    select.value = value;
    select.dispatchEvent(new Event("change", { bubbles: true }));
    syncTrigger();
    close();
    trigger.focus();
  }

  trigger.addEventListener("click", () => {
    if (menu.classList.contains("open")) close();
    else open();
  });

  trigger.addEventListener("keydown", (event) => {
    const isOpen = menu.classList.contains("open");
    if (!isOpen) {
      if (event.key === "ArrowDown" || event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        open();
      }
      return;
    }
    const items = menu.children;
    if (!items.length) return;
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setActive((activeIndex + 1) % items.length);
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setActive((activeIndex - 1 + items.length) % items.length);
    } else if (event.key === "Home") {
      event.preventDefault();
      setActive(0);
    } else if (event.key === "End") {
      event.preventDefault();
      setActive(items.length - 1);
    } else if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      const li = items[activeIndex];
      if (li) choose(li.dataset.value);
    } else if (event.key === "Escape") {
      event.preventDefault();
      close();
    } else if (event.key === "Tab") {
      close();
    }
  });

  syncTrigger();
  wrapper.__sync = syncTrigger;
}

function currentLabel(select) {
  const option = Array.from(select.options).find((opt) => opt.value === select.value);
  return option ? option.textContent : "";
}

function closeDropdown(wrapper) {
  const trigger = wrapper.querySelector(".cs-trigger");
  const menu = wrapper.querySelector(".cs-menu");
  if (menu) menu.classList.remove("open");
  if (trigger) trigger.setAttribute("aria-expanded", "false");
}

function syncCustomDropdown(select) {
  const wrapper = select?.previousElementSibling;
  if (wrapper && wrapper.__sync) wrapper.__sync();
}
