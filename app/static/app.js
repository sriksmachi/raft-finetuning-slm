import { api as request } from "./api.js";

const $ = (id) => document.getElementById(id);
let mode = localStorage.getItem("studio-mode") === "production" ? "production" : "demo";
const stageLabels = {
  data_preparation: "Data",
  training: "Fine-tuning",
  evaluation: "Evaluation",
  decision: "Candidate release",
  deployment: "Deployment gate",
};
const recipeLabels = {
  "context-grounded": "Context-grounded (RAFT)",
  "context-assisted": "Context-assisted (RAG)",
};
let config;
let evaluationShowcase;
let runs = [];
let selected;
let tab = "data_preparation";
let busy = false;
let pollTimer;
let refreshing = false;
let selectionVersion = 0;
let preparationLog;
let preparationLogError;
const PRODUCTION_POLL_INTERVAL_MS = 10000;

function preparationStatusText(value) {
  let message = String(value).replace(/\u001b\[[0-?]*[ -/]*[@-~]/g, "").trim();
  const entries = [...message.matchAll(
    /\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:\d{2})?\s+\[(DEBUG|INFO|WARNING|ERROR|CRITICAL|STATUS)\]\s*/g,
  )];
  if (entries.length) {
    const latest = entries.at(-1);
    message = message.slice(latest.index + latest[0].length).trim();
    if (latest[1] !== "STATUS") message = message.replace(/^[\w.]+:\s*/, "");
  }
  const progress = message.match(/^(Pages|Processing chunks):\s*\d+%.*?\|\s*(\d+)\/(\d+)/);
  if (progress) {
    return progress[1] === "Pages"
      ? `Extracted ${progress[2]} of ${progress[3]} pages`
      : `Processed ${progress[2]} of ${progress[3]} chunks`;
  }
  message = message.replace(/^(Extracting page \d+\/\d+):\s*page_\d+\.jpg$/, "$1")
    .replace(/^Generated samples:\s*(\d+)\/(\d+)$/, "Generated $1 of $2 samples");
  return message.replace(/\b(page|pages|PDF|answer|chunk)\s+(\d+)\/(\d+)\b/gi, "$1 $2 of $3");
}

function renderPreparationLog() {
  const visible = mode === "production" && Boolean(selected);
  $("preparation-log-panel").hidden = !visible;
  if (!visible) return;
  const current = preparationLog?.run_id === selected.run_id ? preparationLog : undefined;
  $("preparation-status").textContent = preparationLogError
    || (current ? preparationStatusText(current.current_status) : "Loading preparation log...");
  $("preparation-log").textContent = current?.log || "";
  $("preparation-log-updated").textContent = current
    ? `Last log update: ${new Date(current.updated_at).toLocaleString()}. ${running() ? "Checks every 10 seconds." : "Job is no longer running."}${current.truncated ? " Showing recent output only." : ""}`
    : "Checks every 10 seconds while preparation is running.";
}

async function refreshPreparationLog() {
  if (mode !== "production" || !selected) return;
  const runId = selected.run_id;
  const version = selectionVersion;
  try {
    const result = await api(`/api/runs/${encodeURIComponent(runId)}/log`);
    if (version !== selectionVersion || selected?.run_id !== runId) return;
    preparationLog = result;
    preparationLogError = undefined;
  } catch (error) {
    if (version !== selectionVersion || selected?.run_id !== runId) return;
    preparationLogError = error.message;
  }
}

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function referenceLabel(value) {
  return String(value ?? "Pending").replace(/^demo:\/\//, "").replace(/\bdemo-/g, "");
}

function modelLabel(id) {
  const model = config.models.find((entry) => entry.id === id);
  return model ? model.label.replace(/\s*\(simulated\)$/i, "") : referenceLabel(id);
}

function workflowText(value) {
  return String(value).replace(/\b(?:[Ss]imulated|[Ii]llustrative|[Dd]emo)\s+/g, "")
    .replace(/\s+fixture\b/g, "").replace(/\s{2,}/g, " ");
}

function showError(error) {
  $("error").textContent = workflowText(error.message);
  $("error").hidden = false;
}

async function api(path, body) {
  return request(path, body, mode);
}

function field(form, name) {
  return form.elements.namedItem(name);
}

function stageSucceeded(name) {
  return selected?.stages[name]?.status === "succeeded";
}

function running() {
  return Object.values(selected?.stages ?? {}).some((stage) => stage.status === "running");
}

function unlockedStageIndex(run = selected) {
  if (!run || !config) return 0;
  if (mode === "production") return 0;
  let unlocked = 0;
  for (let index = 0; index < config.stage_order.length - 1; index += 1) {
    const stage = config.stage_order[index];
    if (run.stages[stage].status !== "succeeded") break;
    unlocked = index + 1;
  }
  return unlocked;
}

function canOpenStage(name) {
  return config.stage_order.indexOf(name) <= unlockedStageIndex();
}

function updateWizardNavigation() {
  if (!selected || !config) {
    $("wizard-back").disabled = true;
    $("wizard-next").disabled = true;
    return;
  }
  const index = config.stage_order.indexOf(tab);
  const unlocked = unlockedStageIndex();
  $("wizard-back").disabled = index <= 0 || busy;
  $("wizard-next").disabled = index < 0 || index >= unlocked
    || index >= config.stage_order.length - 1 || busy;
  $("wizard-position").textContent = `Step ${index + 1} of ${config.stage_order.length}`;
  const nextRequired = config.stage_order[unlocked];
  $("wizard-status").textContent = mode === "production"
    ? "Only data preparation is enabled in production. Later stages are not connected."
    : selected.status === "completed"
    ? "Workflow complete. Review any completed step."
    : `Current required step: ${stageLabels[nextRequired]}. Future steps unlock in order.`;
}

function summary(target, heading, entries) {
  const card = element("div", undefined, "summary-card");
  card.append(element("h3", heading));
  const list = element("dl");
  for (const [label, value] of entries) {
    list.append(element("dt", label), element("dd", String(value ?? "Pending")));
  }
  card.append(list);
  $(target).append(card);
  return card;
}

function badge(text) {
  return element("span", text.toUpperCase(), `badge ${text}`);
}

function setTab(name) {
  if (!config) return;
  if (selected && !canOpenStage(name)) return;
  tab = name;
  for (const stage of config.stage_order) {
    $(`panel-${stage}`).hidden = stage !== tab;
  }
  for (const button of $("stage-nav").children) {
    const active = button.dataset.stage === name;
    button.classList.toggle("active", active);
    if (active) button.setAttribute("aria-current", "step");
    else button.removeAttribute("aria-current");
  }
  updateWizardNavigation();
}

function renderRuns() {
  $("run-select").replaceChildren();
  $("run-list").replaceChildren();
  for (const run of runs) {
    const name = run.training ? modelLabel(run.training.config.base_model) : "Data preparation";
    const option = element("option", `${name} - ${referenceLabel(run.run_id)}`);
    option.value = run.run_id;
    $("run-select").append(option);
    const button = element("button", undefined, "run-item");
    button.type = "button";
    button.classList.toggle("active", run.run_id === selected?.run_id);
    button.append(element("strong", name), element("small", referenceLabel(run.run_id)), badge(run.status));
    button.addEventListener("click", () => selectRun(run.run_id));
    $("run-list").append(button);
  }
  $("run-select").value = selected?.run_id ?? "";
}

function scoreText(value) {
  return value == null ? "Not scored" : `${(Number(value) * 100).toFixed(1)}%`;
}

function answerCell(answer) {
  const cell = element("td", undefined, "evidence-answer");
  cell.append(element("p", answer.text || "No response"));
  cell.append(element(
    "small",
    `Correctness ${scoreText(answer.scores.correctness)} · Faithfulness ${scoreText(answer.scores.faithfulness)} · Relevancy ${scoreText(answer.scores.relevancy)} · ${Number(answer.latency_ms).toFixed(0)} ms`,
  ));
  return cell;
}

function renderEvidenceRows() {
  if (!evaluationShowcase) return;
  const search = $("evidence-search").value.trim().toLowerCase();
  const type = $("evidence-type").value;
  const comparison = $("evidence-comparison").value;
  const records = evaluationShowcase.records.filter((record) => {
    if (type && record.type !== type) return false;
    const fineTuned = record.answers.fine_tuned.scores.correctness;
    const original = record.answers.oob.scores.correctness;
    const gpt = record.answers.gpt.scores.correctness;
    if (comparison === "fine_tuned_better" && !(fineTuned > original)) return false;
    if (comparison === "gpt_better" && !(gpt > fineTuned)) return false;
    if (comparison === "ties" && !(fineTuned === original && original === gpt)) return false;
    if (!search) return true;
    return [
      record.query, record.cot_answer, record.answers.gpt.text,
      record.answers.oob.text, record.answers.fine_tuned.text,
    ].some((value) => String(value).toLowerCase().includes(search));
  });

  $("evidence-rows").replaceChildren();
  for (const record of records) {
    const row = element("tr");
    const query = element("td", undefined, "evidence-query");
    query.append(element("strong", record.query), element("small", `${record.id} · ${record.type}`));
    row.append(query, element("td", record.cot_answer, "evidence-answer"));
    row.append(
      answerCell(record.answers.gpt),
      answerCell(record.answers.oob),
      answerCell(record.answers.fine_tuned),
    );
    $("evidence-rows").append(row);
  }
  $("evidence-count").textContent =
    `Showing ${records.length} of ${evaluationShowcase.summary.sample_count} sealed-test records.`;
}

function renderEvaluationShowcase() {
  $("showcase-metrics").replaceChildren();
  const summaryData = evaluationShowcase.summary;
  for (const key of ["fine_tuned", "oob", "gpt"]) {
    const model = summaryData.models[key];
    const card = element("article", undefined, "metric-card");
    card.append(element("h4", model.label));
    const list = element("dl");
    for (const [label, value] of [
      ["Faithfulness", scoreText(model.scores.faithfulness)],
      ["Relevancy", scoreText(model.scores.relevancy)],
      ["Correctness", scoreText(model.scores.correctness)],
      ["Mean latency", `${Number(model.latency_avg_ms).toFixed(0)} ms`],
      ["P95 latency", `${Number(model.latency_p95_ms).toFixed(0)} ms`],
    ]) {
      list.append(element("dt", label), element("dd", value));
    }
    card.append(list);
    $("showcase-metrics").append(card);
  }
  $("evaluation-showcase-title").textContent =
    `Response-level model comparison · ${summaryData.sample_count} records`;
  $("showcase-provenance").textContent =
    `Evaluation ${summaryData.evaluation_run_id} · Training ${summaryData.training_job_name} · ${new Date(summaryData.created_at).toLocaleString()}`;
  renderEvidenceRows();
}

function renderSelected() {
  if (!selected || !config) return;
  const run = selected;
  $("run-title").textContent = run.training ? `${modelLabel(run.training.config.base_model)} fine-tuning` : "New fine-tuning run";
  $("run-meta").textContent = `Run ${referenceLabel(run.run_id)} | Updated ${new Date(run.updated_at).toLocaleString()}`;
  $("run-status").textContent = run.status.toUpperCase();
  $("run-status").className = `badge ${run.status}`;
  $("run-record").textContent = JSON.stringify(run, null, 2);
  $("stage-nav").replaceChildren();
  config.stage_order.forEach((name, index) => {
    const state = run.stages[name];
    const unlocked = index <= unlockedStageIndex(run);
    const button = element("button", undefined, "stage-step");
    button.type = "button";
    button.dataset.stage = name;
    button.disabled = !unlocked;
    button.classList.toggle("locked", !unlocked);
    button.classList.toggle("succeeded", state.status === "succeeded");
    button.append(
      element("strong", `${index + 1}. ${stageLabels[name]}`),
      element("span", unlocked ? state.status.replaceAll("_", " ") : "locked"),
    );
    if (unlocked) button.addEventListener("click", () => setTab(name));
    $("stage-nav").append(button);
  });
  const active = config.stage_order.find((name) => run.stages[name].status === "running");
  $("progress-panel").hidden = !active;
  if (active) {
    const value = run.stages[active].progress;
    $("progress-label").textContent = `${stageLabels[active]} in progress`;
    $("progress-value").textContent = mode === "production" ? "Running" : `${value}%`;
    if (mode === "production") {
      $("stage-progress").removeAttribute("value");
      $("stage-progress").textContent = "Running";
    } else {
      $("stage-progress").value = value;
      $("stage-progress").textContent = `${value}%`;
    }
  }
  renderPreparationLog();
  for (const target of ["data-summary", "training-summary", "evaluation-summary", "decision-summary", "deployment-summary"]) {
    $(target).replaceChildren();
  }
  for (const name of config.stage_order) {
    if (run.stages[name].error) {
      const targets = { data_preparation: "data-summary", training: "training-summary", evaluation: "evaluation-summary", decision: "decision-summary", deployment: "deployment-summary" };
      $(targets[name]).append(element("div", `${workflowText(run.stages[name].error)} Create a new run to try again.`, "error"));
    }
  }
  if (run.data_card) {
    const card = run.data_card;
    summary("data-summary", "Data card", [
      ["Recipe", recipeLabels[card.recipe] ?? card.recipe],
      ["Generated samples", card.record_count],
      ["Source", referenceLabel(card.source_folder ?? card.source_references?.join(", "))],
      ["Split counts", JSON.stringify(card.split_counts)],
      ["Validation", typeof card.validation === "object" ? `${card.validation.status}: ${card.validation.checked_records} records checked, ${card.validation.invalid_records} invalid` : card.validation],
      ["Fingerprint", referenceLabel(card.fingerprint)], ["Data asset", referenceLabel(card.asset_reference ?? card.azure_ml_asset)],
    ]);
    for (const sample of card.samples ?? []) {
      const preview = element("details");
      preview.append(element("summary", `${sample.split ?? "train"}: ${sample.question}`),
        element("p", sample.cot_answer));
      $("data-summary").append(preview);
    }
  } else {
    $("data-summary").append(element("p", "Data preparation has not completed yet.", "hint"));
  }
  if (run.training) {
    summary("training-summary", "Candidate lineage", [
      ["Base model", modelLabel(run.training.config.base_model)],
      [stageSucceeded("training") ? "Candidate" : "Planned candidate", referenceLabel(run.training.candidate_id)],
      ["Training job", referenceLabel(run.training.job_name)],
      [stageSucceeded("training") ? "Saved artifact" : "Expected artifact (not produced)", referenceLabel(run.training.model_artifact)],
    ]);
  }
  if (run.evaluation) {
    summary("evaluation-summary", "Bound evaluation evidence", [
      ["Candidate", referenceLabel(run.evaluation.candidate_id)], ["Test split", referenceLabel(run.evaluation.test_split)],
      ["Dataset fingerprint", referenceLabel(run.evaluation.data_fingerprint)],
    ]);
    $("evaluation-summary").append(element(
      "p",
      "Quality scores and measured inference latencies are shown once in the Azure evaluation evidence above.",
      "hint",
    ));
    const gate = run.evaluation.gate;
    const card = summary("decision-summary", "Registration gate", [
      ["Status", gate.status.toUpperCase()], ["Evidence", workflowText(gate.message)],
      ["Policy", "Candidate JSON validity must meet or exceed GPT, with latency no higher than the baseline."],
    ]);
    card.append(element("pre", JSON.stringify(gate.values, null, 2)));
    const recommendation = run.evaluation.recommendation;
    if (recommendation) {
      summary("decision-summary", "Deployment recommendation", [
        ["Readiness", recommendation.ready ? "Recommended for review" : "Not ready"],
        ["Reasoning", recommendation.ready ? "Ready for deployment after human confirmation." : workflowText(recommendation.summary)],
      ]);
    }
  } else {
    $("evaluation-summary").append(element("p", "No evaluation evidence yet. Complete training first.", "hint"));
    $("decision-summary").append(element("p", "Pending evidence. Registration is unavailable until evaluation completes.", "hint"));
  }
  const registration = run.decisions.registration;
  if (registration) {
    summary("decision-summary", "Candidate release approval", [
      ["Model", `${referenceLabel(registration.model_name)}:${registration.version}`],
      ["Approver", registration.actor === "local-demo-user" ? "local-user" : registration.actor],
      ["Override", registration.override ? "Yes" : "No"],
      ["Reason", registration.reason === "Simulated gate passed." ? "Gate passed." : registration.reason || "Gate passed"],
      ["Decision time", registration.decided_at],
    ]);
  }
  if (run.release) {
    summary("decision-summary", "Zero-traffic candidate", [
      ["Release", referenceLabel(run.release.release_id)],
      ["Status", workflowText(run.release.status).replaceAll("_", " ")],
      ["Endpoint", referenceLabel(run.release.endpoint?.uri)],
      ["Deployment", run.release.endpoint?.deployment],
      ["Traffic", `${run.release.endpoint?.traffic_percent}%`],
      ["Drift reference", referenceLabel(run.release.drift_reference?.uri)],
    ]);
  }
  if (run.deployment) {
    const gate = run.deployment.deployment_gate;
    if (gate) {
      summary("deployment-summary", "Live deployment evidence", [
        ["Gate", gate.status.toUpperCase()],
        ["Sealed examples", gate.sample_count],
        ["Mean token-F1", `${(gate.values.mean_token_f1 * 100).toFixed(1)}%`],
        ["Mean latency", `${gate.values.mean_latency_ms} ms`],
        ["P95 latency", `${gate.values.p95_latency_ms} ms`],
      ]);
    }
    summary("deployment-summary", "Traffic and monitoring outcome", [
      ["Endpoint", referenceLabel(run.deployment.endpoint_name)],
      ["Candidate traffic", run.deployment.traffic
        ? `${run.deployment.traffic.candidate_percent}%`
        : "Completed"],
      ["Health", run.deployment.monitoring.status],
      ["Drift schedule", run.deployment.monitoring.schedule ?? "Configured"],
      ["PSI threshold", run.deployment.monitoring.psi_threshold ?? "0.2"],
      ["Monitoring reference", referenceLabel(run.deployment.monitoring.reference_uri)],
    ]);
  } else {
    $("deployment-summary").append(element("p",
      run.release
        ? "The candidate has zero traffic. Run the deployment gate before routing requests."
        : "Complete the approved candidate release before the deployment gate.", "hint"));
  }
  const locked = busy || running() || run.status === "failed" || mode === "production";
  $("train").disabled = locked || !stageSucceeded("data_preparation") || stageSucceeded("training");
  $("suggest").disabled = locked || !stageSucceeded("data_preparation");
  $("evaluate").disabled = locked || !stageSucceeded("training") || stageSucceeded("evaluation");
  $("register").disabled = locked || !stageSucceeded("evaluation") || Boolean(registration);
  $("deploy").disabled = locked || !registration || stageSucceeded("deployment");
  $("override-fields").hidden = run.evaluation?.gate.status !== "blocked" || Boolean(registration);
  $("create-form").querySelector("button").disabled = busy || running();
  $("run-select").disabled = busy;
  $("new-run").disabled = busy;
  setTab(tab);
  renderRuns();
  schedulePoll();
}

function schedulePoll() {
  clearTimeout(pollTimer);
  if (running()) {
    pollTimer = setTimeout(async () => {
      try {
        await refresh();
      } catch (error) {
        showError(error);
        schedulePoll();
      }
    }, mode === "production" ? PRODUCTION_POLL_INTERVAL_MS : 1200);
  }
}

async function refresh() {
  if (refreshing) return;
  refreshing = true;
  const version = selectionVersion;
  try {
    const result = await api("/api/runs");
    if (version !== selectionVersion) return;
    const previous = selected;
    runs = result.runs;
    if (selected) selected = runs.find((run) => run.run_id === selected.run_id);
    if (previous && selected) {
      const completedRunningStage = config.stage_order.find((name) =>
        previous.stages[name].status === "running"
        && selected.stages[name].status === "succeeded");
      if (completedRunningStage) tab = mode === "production" ? "data_preparation" : selected.current_stage;
    }
    if (!selected && runs.length) {
      selected = runs[0];
      tab = selected.current_stage;
    }
    if (selected) {
      await refreshPreparationLog();
      if (version !== selectionVersion) return;
      renderSelected();
    }
    else {
      renderRuns();
      renderPreparationLog();
      $("run-title").textContent = "New data preparation";
      $("run-meta").textContent = "Choose a source folder to create your first run.";
      $("run-status").textContent = "READY";
      $("wizard-status").textContent = mode === "production"
        ? "Only data preparation is enabled in production. Later stages are not connected." : "";
      updateWizardNavigation();
    }
  } finally {
    refreshing = false;
  }
}

async function selectRun(id) {
  if (busy) return;
  const version = ++selectionVersion;
  try {
    const result = await api(`/api/runs/${encodeURIComponent(id)}`);
    if (version !== selectionVersion) return;
    selected = result;
    tab = selected.current_stage;
    $("error").hidden = true;
    if (selected.training?.config) prefill(selected.training.config);
    preparationLog = undefined;
    preparationLogError = undefined;
    renderSelected();
    await refreshPreparationLog();
    if (version === selectionVersion) renderPreparationLog();
  } catch (error) {
    showError(error);
  }
}

function prefill(parameters) {
  const form = $("training-form");
  for (const [name, value] of Object.entries(parameters)) {
    const control = field(form, name);
    if (control && control.type !== "checkbox") control.value = value;
  }
}

async function action(callback) {
  if (busy) return;
  busy = true;
  selectionVersion += 1;
  clearTimeout(pollTimer);
  $("error").hidden = true;
  renderSelected();
  try {
    await callback();
    await refresh();
  } catch (error) {
    showError(error);
  } finally {
    busy = false;
    renderSelected();
  }
}

function bindForm(id, endpoint, buildBody, nextTab) {
  const form = $(id);
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    if (!selected) return;
    action(async () => {
      selected = await api(`/api/runs/${encodeURIComponent(selected.run_id)}/${endpoint}`, buildBody(form));
      tab = nextTab;
      form.reset();
      if (selected?.training?.config) prefill(selected.training.config);
    });
  });
}

$("create-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  action(async () => {
    selected = await api("/api/runs", {
      source_folder: field(form, "source_folder").value.trim(),
      recipe: field(form, "recipe").value,
      ...(mode === "demo" ? { scenario: field(form, "scenario").value } : {
        num_questions: Number(field(form, "num_questions").value),
        oracle_probability: Number(field(form, "oracle_probability").value),
        data_asset_name: field(form, "data_asset_name").value.trim(),
      }),
      num_samples: Number(field(form, "num_samples").value),
      confirmed: field(form, "confirmed").checked,
    });
    tab = "data_preparation";
    field(form, "confirmed").checked = false;
  });
});
bindForm("training-form", "training", (form) => {
  const body = { confirmed: field(form, "confirmed").checked, base_model: field(form, "base_model").value };
  for (const name of ["max_seq_length", "batch_size", "gradient_accumulation", "learning_rate", "epochs", "lora_rank"]) {
    body[name] = Number(field(form, name).value);
  }
  return body;
}, "training");
bindForm("evaluation-form", "evaluation", (form) => ({ confirmed: field(form, "confirmed").checked }), "evaluation");
bindForm("register-form", "register", (form) => ({
  confirmed: field(form, "confirmed").checked,
  actor: field(form, "actor").value.trim(),
  override: field(form, "override").checked,
  confirmation: field(form, "confirmation").value,
  reason: field(form, "reason").value.trim(),
}), "decision");
bindForm("deployment-form", "deployment", (form) => ({
  confirmed: field(form, "confirmed").checked,
  actor: field(form, "actor").value.trim(),
  traffic_percent: Number(field(form, "traffic_percent").value),
}), "deployment");
$("suggest").addEventListener("click", () => action(async () => {
  const result = await api(`/api/runs/${encodeURIComponent(selected.run_id)}/training/suggestions`, {});
  prefill(result.parameters);
  $("suggestion-note").textContent = `Suggested for ${selected.data_card.split_counts.train} training records. Review sequence length, batch size, and memory requirements before starting.`;
}));
$("new-run").addEventListener("click", () => {
  setTab("data_preparation");
  field($("create-form"), "source_folder").focus();
});
$("refresh").addEventListener("click", () => action(refresh));
$("run-select").addEventListener("change", (event) => selectRun(event.target.value));
$("wizard-back").addEventListener("click", () => {
  const index = config.stage_order.indexOf(tab);
  if (index > 0) setTab(config.stage_order[index - 1]);
});
$("wizard-next").addEventListener("click", () => {
  const index = config.stage_order.indexOf(tab);
  if (index >= 0 && index < unlockedStageIndex()) setTab(config.stage_order[index + 1]);
});
$("download").addEventListener("click", () => {
  if (!selected) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(selected, null, 2)], { type: "application/json" }));
  const anchor = element("a");
  anchor.href = url;
  anchor.download = `${selected.run_id}.json`;
  anchor.click();
  URL.revokeObjectURL(url);
});

async function initialize() {
  config = await api("/api/config");
  evaluationShowcase = await api("/api/evaluation-showcase");
  if (config.mode !== mode) {
    throw new Error("The API is configured for a different execution mode.");
  }
  $("mode").value = mode;
  $("mode-note").textContent = mode === "demo" ? "Demo only. No Azure services are called."
    : "Production: real, billable GPT generation and Azure ML publication. Only data preparation is available.";
  $("production-options").hidden = mode !== "production";
  $("demo-options").hidden = mode !== "demo";
  field($("create-form"), "recipe").querySelector('[value="context-assisted"]').disabled = mode === "production";
  $("confirmation-label").textContent = mode === "production"
    ? "I confirm billable GPT generation, PDF extraction, and Azure ML data publication."
    : "I confirm the data source and generation recipe.";
  const models = field($("training-form"), "base_model");
  for (const model of config.models) {
    const option = element("option", modelLabel(model.id));
    option.value = model.id;
    models.append(option);
  }
  $("mode").addEventListener("change", (event) => {
    localStorage.setItem("studio-mode", event.target.value);
    window.location.reload();
  });
  await refresh();
  renderEvaluationShowcase();
  if (selected?.training?.config) prefill(selected.training.config);
}
$("evidence-search").addEventListener("input", renderEvidenceRows);
$("evidence-type").addEventListener("change", renderEvidenceRows);
$("evidence-comparison").addEventListener("change", renderEvidenceRows);
initialize().catch(showError);
