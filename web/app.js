const state = {
  rows: [],
  mode: "generate",
  previewTimer: null,
  previewController: null,
  previewImageUrl: null,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {
      ...(options.body ? { "Content-Type": "application/json" } : {}),
      ...(options.headers || {}),
    },
  });
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch (_) {
      // Keep the HTTP fallback when the proxy returns a non-JSON error.
    }
    throw new Error(message);
  }
  return response;
}

function setBusy(isBusy, title = "Loading") {
  const status = $("#run-status");
  status.hidden = !isBusy;
  status.classList.remove("error");
  status.classList.toggle("fullscreen-loading", isBusy);
  $("#status-title").textContent = title;
  // $("#status-message").textContent = message;
  $$('button[type="submit"]').forEach((button) => { button.disabled = isBusy; });
  if (isBusy) {
    window.requestAnimationFrame(() => {
      status.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }
}

function showError(error) {
  const status = $("#run-status");
  status.hidden = false;
  status.classList.add("error");
  $("#status-title").textContent = "The request could not be completed";
  $("#status-message").textContent = error.message || String(error);
  $$('button[type="submit"]').forEach((button) => { button.disabled = false; });
}

function formatNumber(value, digits = 2) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(digits) : "—";
}

function probability(row, key) {
  const value = row[key] ?? row[key.replace("predicted_", "")];
  return Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(1)}%` : "—";
}

const previewMetricIds = [
  "input-eraactivity",
  "input-toxicity",
  "input-weight",
  "input-logp",
  "input-hbdhba",
  "input-sa",
];

function setPreviewMetrics(value) {
  previewMetricIds.forEach((id) => {
    document.getElementById(id).textContent = value;
  });
}

function resetMoleculePreview(message = "Enter a valid SMILES to preview") {
  const image = $("#input-image");
  image.hidden = true;
  image.removeAttribute("src");
  image.alt = "";
  $("#input-image-loader").hidden = true;
  const placeholder = $("#input-image-placeholder");
  placeholder.textContent = message;
  placeholder.hidden = false;
  setPreviewMetrics("N/A");
  if (state.previewImageUrl) {
    URL.revokeObjectURL(state.previewImageUrl);
    state.previewImageUrl = null;
  }
}

function setMoleculePreviewLoading() {
  $("#input-image").hidden = true;
  $("#input-image-placeholder").hidden = true;
  $("#input-image-loader").hidden = false;
  setPreviewMetrics("…");
}

async function loadMoleculePreview(smiles) {
  state.previewController?.abort();
  const controller = new AbortController();
  state.previewController = controller;
  setMoleculePreviewLoading();

  try {
    const [scoreResponse, imageResponse] = await Promise.all([
      api("/api/score", {
        method: "POST",
        body: JSON.stringify({ smiles }),
        signal: controller.signal,
      }),
      api(`/api/molecule-image?smiles=${encodeURIComponent(smiles)}`, {
        signal: controller.signal,
      }),
    ]);
    const [scores, imageBlob] = await Promise.all([scoreResponse.json(), imageResponse.blob()]);
    if (controller.signal.aborted) return;

    $("#input-eraactivity").textContent = probability(scores, "predicted_eralpha_activity_probability");
    $("#input-toxicity").textContent = probability(scores, "predicted_clintox_toxicity_probability");
    $("#input-weight").textContent = formatNumber(scores.molecular_weight, 1);
    $("#input-logp").textContent = formatNumber(scores.logp);
    $("#input-hbdhba").textContent = `${scores.hbd} / ${scores.hba}`;
    $("#input-sa").textContent = formatNumber(scores.sa_score);

    if (state.previewImageUrl) URL.revokeObjectURL(state.previewImageUrl);
    state.previewImageUrl = URL.createObjectURL(imageBlob);
    const image = $("#input-image");
    image.onload = () => {
      if (controller.signal.aborted) return;
      $("#input-image-loader").hidden = true;
      image.hidden = false;
    };
    image.alt = `2D structure for ${scores.smiles}`;
    image.src = state.previewImageUrl;
  } catch (error) {
    if (error.name === "AbortError") return;
    resetMoleculePreview(error.message || "Enter a complete, valid SMILES string.");
  }
}

function scheduleMoleculePreview() {
  window.clearTimeout(state.previewTimer);
  state.previewController?.abort();
  const smiles = $("#smiles").value.trim();
  if (!smiles) {
    resetMoleculePreview();
    return;
  }
  state.previewTimer = window.setTimeout(() => loadMoleculePreview(smiles), 650);
}

function scoreDelta(value, beneficialWhenPositive, multiplier = 1) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return null;
  const displayed = numeric * multiplier;
  const delta = document.createElement("small");
  delta.className = "score-delta";
  if (displayed > 0) {
    delta.classList.add(beneficialWhenPositive ? "favorable" : "unfavorable");
  } else if (displayed < 0) {
    delta.classList.add(beneficialWhenPositive ? "unfavorable" : "favorable");
  } else {
    delta.classList.add("neutral");
  }
  delta.textContent = `(${displayed >= 0 ? "+" : ""}${displayed.toFixed(1)})`;
  return delta;
}

function scoreCell(label, value, delta = null) {
  const cell = document.createElement("div");
  cell.className = "score";
  const strong = document.createElement("strong");
  const span = document.createElement("span");
  strong.textContent = value;
  if (delta) strong.append(" ", delta);
  span.textContent = label;
  cell.append(strong, span);
  return cell;
}

async function copyText(text, button) {
  try {
    await navigator.clipboard.writeText(text);
  } catch (_) {
    const field = document.createElement("textarea");
    field.value = text;
    field.style.position = "fixed";
    field.style.opacity = "0";
    document.body.append(field);
    field.select();
    document.execCommand("copy");
    field.remove();
  }
  const original = button.textContent;
  button.textContent = "Copied";
  window.setTimeout(() => { button.textContent = original; }, 1200);
}

function activateMode(mode) {
  $$('.mode-tab').forEach((tab) => tab.classList.toggle("active", tab.dataset.panel === mode));
  $$('.panel').forEach((panel) => {
    const active = panel.id === `${mode}-panel`;
    panel.hidden = !active;
    panel.classList.toggle("active", active);
  });
}

function optimizeCandidate(smiles) {
  activateMode("optimize");
  const input = $("#smiles");
  input.value = smiles;
  input.dispatchEvent(new Event("input", { bubbles: true }));
  input.focus();
  $("#optimize-panel").scrollIntoView({ behavior: "smooth", block: "start" });
}

function candidateImageActions(smiles) {
  const actions = document.createElement("div");
  actions.className = "molecule-image-actions";

  const copy = document.createElement("button");
  copy.type = "button";
  copy.textContent = "Copy SMILES";
  copy.addEventListener("click", () => copyText(smiles, copy));

  const optimize = document.createElement("button");
  optimize.type = "button";
  optimize.textContent = "Optimize";
  optimize.addEventListener("click", () => optimizeCandidate(smiles));

  actions.append(copy, optimize);
  return actions;
}

function renderCandidates(rows, mode) {
  const grid = $("#candidate-grid");
  grid.replaceChildren();
  if (!rows.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = mode === "optimize"
      ? "No supported structural analogs were generated for this molecule."
      : "No generated molecules passed the current RDKit filters.";
    grid.append(empty);
    return;
  }

  rows.forEach((row, index) => {
    const card = document.createElement("article");
    card.className = "molecule-card";
    if (mode === "optimize" && !row.improves_objective) {
      card.classList.add("not-improved");
    }
    const imageWrap = document.createElement("div");
    imageWrap.className = "molecule-image";
    const image = document.createElement("img");
    image.loading = "lazy";
    image.alt = `2D structure for candidate ${index + 1}`;
    image.src = `/api/molecule-image?smiles=${encodeURIComponent(row.smiles)}`;
    imageWrap.append(image, candidateImageActions(row.smiles));

    const body = document.createElement("div");
    body.className = "molecule-body";
    const scores = document.createElement("div");
    scores.className = "score-grid";
    scores.append(
      scoreCell(
        "ERα ACTIVITY",
        probability(row, "predicted_eralpha_activity_probability"),
        scoreDelta(row.delta_eralpha_activity, true, 100),
      ),
      scoreCell(
        "TOXICITY",
        probability(row, "predicted_clintox_toxicity_probability"),
        scoreDelta(row.delta_clintox_toxicity, false, 100),
      ),
      scoreCell("WEIGHT", formatNumber(row.molecular_weight, 1)),
      scoreCell("LOG P", formatNumber(row.logp), scoreDelta(row.delta_logp, false)),
      scoreCell("HBD / HBA", `${row.hbd ?? "—"} / ${row.hba ?? "—"}`),
      scoreCell("SA SCORE", formatNumber(row.sa_score)),
    );
    body.append(scores);

    if (row.passes_filters === false) {
      const warning = document.createElement("p");
      warning.className = "filter-warning";
      warning.textContent = "Outside one or more current RDKit property limits.";
      body.append(warning);
    }
    if (row.transformations) {
      const note = document.createElement("p");
      note.className = "transform-note";
      note.textContent = `Transformation: ${row.transformations}`;
      body.append(note);
    }
    card.append(imageWrap, body);
    grid.append(card);
  });
}

function renderResultsHeading({ successful, total, label, emptyLabel }) {
  const heading = $("#results-title");
  heading.replaceChildren();
  heading.classList.toggle("result-empty-callout", successful === 0);
  if (successful === 0) {
    heading.textContent = emptyLabel;
    return;
  }
  const count = document.createElement("span");
  count.className = "result-success-count";
  count.textContent = successful;
  heading.append(count, `/${total} ${label}`);
}

function showResults({ heading, rows, mode }) {
  state.rows = rows;
  state.mode = mode;
  $("#results").hidden = false;
  renderResultsHeading(heading);
  $("#export-actions").hidden = rows.length === 0;
  renderCandidates(rows, mode);
  $("#results").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function submitGenerate(event) {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  const payload = {
    prefix: form.get("prefix"),
    batch_size: Number(form.get("batch_size")),
    temperature: Number(form.get("temperature")),
    top_k: Number(form.get("top_k")),
    max_length: Number(form.get("max_length")),
  };
  setBusy(true, "Generating and screening molecules");
  try {
    const response = await api("/api/generate", { method: "POST", body: JSON.stringify(payload) });
    const data = await response.json();
    setBusy(false);
    const passed = Number(data.counts.passed_rdkit_filters ?? data.candidates.length);
    const attempted = Number(data.input?.batch_size ?? payload.batch_size);
    showResults({
      heading: {
        successful: passed,
        total: attempted,
        label: "generations passed",
        emptyLabel: `No generations passed (${passed}/${attempted})`,
      },
      rows: data.candidates,
      mode: "generate",
    });
  } catch (error) {
    showError(error);
  }
}

async function submitOptimize(event) {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  const payload = {
    smiles: form.get("smiles"),
    objective: form.get("objective"),
    limit: Number(form.get("limit")),
  };
  setBusy(true, "Creating and ranking analogs");
  try {
    const response = await api("/api/optimize", { method: "POST", body: JSON.stringify(payload) });
    const data = await response.json();
    setBusy(false);
    const improved = Number(data.counts.improved_analogs ?? 0);
    const displayed = Number(data.counts.displayed_analogs ?? data.analogs.length);
    const objectiveLabel = {
      "Increase Predicted ERα Activity": "increased ERα activity",
      "Lower Predicted ClinTox Toxicity": "lowered toxicity",
      "Decrease LogP": "decreased LogP",
    }[payload.objective] || "improved the objective";
    showResults({
      heading: {
        successful: improved,
        total: displayed,
        label: objectiveLabel,
        emptyLabel: displayed
          ? `0/${displayed} ${objectiveLabel}`
          : "No optimization candidates found",
      },
      rows: data.analogs,
      mode: "optimize",
    });
  } catch (error) {
    showError(error);
  }
}

async function exportRows(format) {
  if (!state.rows.length) return;
  try {
    const response = await api(`/api/export/${format}`, {
      method: "POST",
      body: JSON.stringify({ rows: state.rows, filename: `${state.mode}_candidates` }),
    });
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${state.mode}_candidates.${format}`;
    document.body.append(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  } catch (error) {
    showError(error);
  }
}

function metricLabel(key) {
  return key.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function appendMetricObject(root, title, values) {
  const group = document.createElement("section");
  group.className = "metrics-group";
  const heading = document.createElement("h3");
  heading.textContent = title;
  group.append(heading);
  Object.entries(values).forEach(([key, value]) => {
    if (value && typeof value === "object" && !Array.isArray(value)) {
      const subheading = document.createElement("h4");
      subheading.textContent = metricLabel(key);
      group.append(subheading);
      Object.entries(value).forEach(([nestedKey, nestedValue]) => {
        const row = document.createElement("div");
        row.className = "metric-row";
        const label = document.createElement("span");
        const result = document.createElement("strong");
        label.textContent = metricLabel(nestedKey);
        result.textContent = Array.isArray(nestedValue)
          ? JSON.stringify(nestedValue)
          : typeof nestedValue === "number"
            ? formatNumber(nestedValue, 3)
            : String(nestedValue);
        row.append(label, result);
        group.append(row);
      });
      return;
    }
    const row = document.createElement("div");
    row.className = "metric-row";
    const label = document.createElement("span");
    const result = document.createElement("strong");
    label.textContent = metricLabel(key);
    result.textContent = Array.isArray(value)
      ? JSON.stringify(value)
      : typeof value === "number"
        ? formatNumber(value, 3)
        : String(value);
    row.append(label, result);
    group.append(row);
  });
  root.append(group);
}

let evaluationLoaded = false;

async function loadEvaluation() {
  if (evaluationLoaded) return;
  const content = $("#metrics-content");
  try {
    const [modelsResponse, metricsResponse] = await Promise.all([api("/api/models"), api("/api/metrics")]);
    const models = await modelsResponse.json();
    const metrics = await metricsResponse.json();
    renderEvaluation(metrics, models);
    evaluationLoaded = true;
  } catch (error) {
    content.replaceChildren();
    const message = document.createElement("p");
    message.textContent = `Evaluation data could not be loaded: ${error.message}`;
    content.append(message);
  }
}

function renderEvaluation(metrics, models) {
  const content = $("#metrics-content");
  content.replaceChildren();
  appendMetricObject(content, "Artifact status", {
    activity_model: models.activity.available ? "Available" : "Missing",
    toxicity_model: models.toxicity.available ? "Available" : "Missing",
    evaluation_report: metrics.available ? "Available" : "Missing",
  });
  if (!metrics.available) {
    const note = document.createElement("p");
    note.textContent = metrics.message;
    content.append(note);
    return;
  }
  appendMetricObject(content, "Trained at", {
    generated_at_utc: metrics.generated_at_utc
  });
  Object.entries(metrics).forEach(([key, value]) => {
    if (["available", "generated_at_utc", "evaluation_scope"].includes(key)) return;
    if (value && typeof value === "object" && !Array.isArray(value)) {
      appendMetricObject(content, metricLabel(key), value);
    }
  });
}

$$('.mode-tab').forEach((button) => {
  button.addEventListener("click", () => activateMode(button.dataset.panel));
});

const evaluationDialog = $("#evaluation-dialog");

$$('[data-open-evaluation]').forEach((button) => {
  button.addEventListener("click", () => {
    evaluationDialog.showModal();
    loadEvaluation();
  });
});

$("[data-close-evaluation]").addEventListener("click", () => evaluationDialog.close());

evaluationDialog.addEventListener("click", (event) => {
  if (event.target === evaluationDialog) evaluationDialog.close();
});

const helpBubble = $("#help-bubble");
let activeHelpButton = null;

function closeHelpBubble() {
  if (activeHelpButton) activeHelpButton.setAttribute("aria-expanded", "false");
  activeHelpButton = null;
  helpBubble.hidden = true;
}

function positionHelpBubble(button) {
  const buttonRect = button.getBoundingClientRect();
  const bubbleRect = helpBubble.getBoundingClientRect();
  const gutter = 12;
  const centeredLeft = buttonRect.left + buttonRect.width / 2 - bubbleRect.width / 2;
  const left = Math.min(
    Math.max(gutter, centeredLeft),
    window.innerWidth - bubbleRect.width - gutter,
  );
  let top = buttonRect.bottom + 9;
  if (top + bubbleRect.height > window.innerHeight - gutter) {
    top = buttonRect.top - bubbleRect.height - 9;
  }
  helpBubble.style.left = `${left}px`;
  helpBubble.style.top = `${Math.max(gutter, top)}px`;
}

$$('.help-button').forEach((button) => {
  button.addEventListener("click", (event) => {
    event.stopPropagation();
    if (activeHelpButton === button && !helpBubble.hidden) {
      closeHelpBubble();
      return;
    }
    closeHelpBubble();
    activeHelpButton = button;
    button.setAttribute("aria-expanded", "true");
    helpBubble.textContent = button.dataset.help;
    helpBubble.hidden = false;
    positionHelpBubble(button);
  });
});

document.addEventListener("click", (event) => {
  if (!helpBubble.contains(event.target)) closeHelpBubble();
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && activeHelpButton) {
    const button = activeHelpButton;
    closeHelpBubble();
    button.focus();
  }
});

window.addEventListener("resize", closeHelpBubble);
document.addEventListener("scroll", closeHelpBubble, true);

$$('.example').forEach((button) => {
  button.addEventListener("click", () => {
    const target = document.getElementById(button.dataset.target);
    target.value = button.dataset.value;
    target.focus();
    target.dispatchEvent(new Event("input", { bubbles: true }));
  });
});

$("#smiles").addEventListener("input", scheduleMoleculePreview);
$("#generate-form").addEventListener("submit", submitGenerate);
$("#optimize-form").addEventListener("submit", submitOptimize);
$$('[data-export]').forEach((button) => button.addEventListener("click", () => exportRows(button.dataset.export)));
