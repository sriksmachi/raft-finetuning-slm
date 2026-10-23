import { api as request } from "./api.js";

const $ = (id) => document.getElementById(id);
const stageLabels = {
  data_preparation: "Data",
  training: "Fine-tuning",
  evaluation: "Evaluation",
  decision: "Candidate release",
  deployment: "Endpoint and monitoring",
};
const recipeLabels = {
  "context-grounded": "Context-grounded (RAFT)",
  "context-assisted": "Context-assisted (RAG)",
};
let config;
let evaluationShowcase;
let evaluationJobs = [];
let releaseEvaluations = [];
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
  const visible = Boolean(selected);
  $("preparation-log-panel").hidden = !visible;
  if (!visible) return;
  const jobType = selected.run_type === "release" ? "release" : selected.run_type === "evaluation" ? "evaluation"
    : selected.run_type === "training" ? "training" : "preparation";
  $("job-log-title").textContent = `${{ release: "Candidate release", evaluation: "Evaluation", training: "Fine-tuning", preparation: "Preparation" }[jobType]} job status`;
  const current = preparationLog?.run_id === selected.run_id ? preparationLog : undefined;
  $("preparation-status").textContent = preparationLogError
    || (current ? preparationStatusText(current.current_status) : `Loading ${jobType} log...`);
  $("preparation-log").textContent = current?.log || "";
  $("preparation-log-updated").textContent = current
    ? `Last log update: ${new Date(current.updated_at).toLocaleString()}. ${running() ? "Checks every 10 seconds." : "Job is no longer running."}${current.truncated ? " Showing recent output only." : ""}`
    : "Checks every 10 seconds while the job is running.";
}

async function refreshPreparationLog() {
  if (!selected) return;
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
  return String(value ?? "Pending");
}

function modelLabel(id) {
  const model = config.models.find((entry) => entry.id === id);
  return model ? model.label : referenceLabel(id);
}

function workflowText(value) {
  return String(value);
}

function showError(error) {
  $("error").textContent = workflowText(error.message);
  $("error").hidden = false;
}

async function api(path, body) {
  return request(path, body);
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

function activeRun() {
  return (running() ? selected : undefined) ?? runs.find((run) => run.status === "running");
}

function trainingNeedsRefresh(run) {
  return run?.run_type === "training" && ["running", "failed"].includes(run.status)
    && Boolean(run.training?.job_name)
    && !["Completed", "Failed", "Canceled", "Cancelled"].includes(run.training.job_status);
}

function renderTaskControls() {
  const active = Boolean(activeRun());
  for (const id of ["create-form", "training-form", "evaluation-form", "release-plan-form"]) {
    $(id).hidden = active;
  }
  $("release-approval-form").hidden = active || selected?.release?.status !== "awaiting_approval";
  $("task-controls").hidden = !active;
  $("cancel-task").disabled = busy;
}

function visibleStages() {
  return config.stage_order.filter((stage) => stage !== "deployment");
}

function unlockedStageIndex() {
  if (config) return Math.min(
    config.enabled_stages?.includes("decision") ? 3 : 2, visibleStages().length - 1);
  return 0;
}

function canOpenStage(name) {
  const index = visibleStages().indexOf(name);
  return index >= 0 && index <= unlockedStageIndex();
}

function updateWizardNavigation() {
  if (!selected || !config) {
    $("wizard-back").disabled = true;
    $("wizard-next").disabled = true;
    return;
  }
  const stages = visibleStages();
  const index = stages.indexOf(tab);
  const unlocked = unlockedStageIndex();
  $("wizard-back").disabled = index <= 0 || busy;
  $("wizard-next").disabled = index < 0 || index >= unlocked
    || index >= stages.length - 1 || busy;
  $("wizard-position").textContent = `Step ${index + 1} of ${stages.length}`;
  $("wizard-status").textContent = "Review candidate policy and lineage; deployment requires approval of a sourced GPU hourly estimate.";
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
  if (name === "deployment") name = "decision";
  if (selected && !canOpenStage(name)) return;
  tab = name;
  for (const stage of visibleStages()) {
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

function renderEvaluationJob() {
  const control = $("evaluation-job");
  const job = evaluationJobs.find((entry) => entry.job_name === control.value);
  $("evaluation-job-summary").replaceChildren();
  if (job) {
    summary("evaluation-job-summary", "Selected training job", [
      ["Job ID", job.job_name], ["Training date", job.training_date ?? "Not recorded"],
      ["Model", job.base_model ?? "Not recorded"], ["Dataset version", job.dataset_version ?? "Not recorded"],
      ["Number of samples", job.sample_count ?? "Not recorded"], ["Epochs", job.epochs ?? "Not recorded"],
      ["Adapter name", job.adapter_name ?? "Not recorded"],
      ["GPT baseline", job.baseline_id ? `Reuse ${job.baseline_id}` : "Generate a fresh baseline"],
    ]);
  }
  $("evaluation-job-note").textContent = job?.unavailable_reason
    || (job ? "Review the selected job and its sealed test split before confirming evaluation."
      : "No completed training jobs are recorded. Complete training, then refresh.");
  const active = busy || runs.some((run) => run.status === "running");
  control.disabled = active;
  $("evaluate").disabled = active || !job?.eligible;
}

function renderEvaluationJobs() {
  const control = $("evaluation-job");
  const previous = control.value;
  control.replaceChildren();
  for (const job of evaluationJobs) {
    const option = element("option", `${job.job_name} - ${job.base_model ?? "Model not recorded"}`);
    option.value = job.job_name;
    option.disabled = !job.eligible;
    control.append(option);
  }
  if (evaluationJobs.some((job) => job.job_name === previous)) control.value = previous;
  else if (evaluationJobs.some((job) => job.job_name === selected?.training?.job_name)) control.value = selected.training.job_name;
  renderEvaluationJob();
}

function renderEvaluationMetrics(evaluation) {
  $("evaluation-metrics").replaceChildren();
  if (!evaluation?.metrics?.length) return;
  $("evaluation-metrics").append(element("h3", "Evaluation metrics"));
  const table = element("table");
  table.append(element("caption", "Metrics on the same sealed test split", "sr-only"));
  const head = element("thead");
  const header = element("tr");
  for (const text of ["Metric", "Fine-tuned SLM", "Original SLM", "GPT baseline"]) {
    const cell = element("th", text);
    cell.setAttribute("scope", "col");
    header.append(cell);
  }
  head.append(header);
  const body = element("tbody");
  for (const metric of evaluation.metrics) {
    const row = element("tr");
    const heading = element("th", metric.label);
    heading.setAttribute("scope", "row");
    row.append(heading);
    for (const model of ["fine_tuned", "oob", "gpt"]) {
      const value = metric[model];
      row.append(element("td", metric.key === "token_cost" && model !== "gpt" ? "Not applicable (GPU billed)"
        : value == null ? "Not recorded"
        : metric.unit === "score" ? scoreText(value)
        : metric.unit === "ms" ? `${Number(value).toFixed(1)} ms`
        : metric.unit === "tokens" ? Number(value).toLocaleString() : `$${Number(value).toFixed(6)}`));
    }
    body.append(row);
  }
  table.append(head, body);
  $("evaluation-metrics").append(table);
  if (evaluation.token_usage) {
    const usage = evaluation.token_usage;
    $("evaluation-metrics").append(element("p",
      `Only named frontier API models have token prices in token_prices.json (for example gpt-4.1, gpt-4.1-mini, gpt-5.2). GPT identity: ${usage.models.gpt.model_ids?.join(", ") || "Not recorded"}. Unset prices or unknown model identities are not zero. Self-hosted SLM token prices are not applicable; GPU hourly costs are separate. ${usage.note}`, "hint"));
    const judge = usage.models.judge;
    $("evaluation-metrics").append(element("p",
      `Judge (${judge.model_ids?.join(", ") || "model not recorded"}): ${judge.input_tokens ?? "Not recorded"} input / ${judge.output_tokens ?? "Not recorded"} output tokens; ${judge.estimated_cost_usd == null ? "cost unavailable" : `$${judge.estimated_cost_usd.toFixed(6)} estimated`}. Price snapshot: ${usage.price_snapshot.sha256}`, "hint"));
  }
}

function renderProductionEvaluation() {
  const evaluation = selected?.evaluation;
  const completed = Boolean(evaluation?.summary);
  $("evaluation-showcase").hidden = !completed;
  evaluationShowcase = completed ? { summary: evaluation.summary, records: evaluation.records } : undefined;
  renderEvaluationMetrics(evaluation);
  if (completed) {
    renderEvaluationShowcase();
    summary("evaluation-summary", "Bound evaluation evidence", [
      ["Training job", evaluation.candidate_id], ["Dataset", evaluation.dataset_id],
      ["Sealed test records", evaluation.summary.sample_count], ["Judge model", evaluation.judge_deployment],
      ["GPT baseline", evaluation.baseline_id], ["Test fingerprint", evaluation.data_fingerprint],
    ]);
  } else {
    $("evaluation-summary").append(element("p",
      evaluation ? `Evaluation ${evaluation.job_status}. Current activity appears in the run log.`
        : "Choose a completed training job above to evaluate its sealed test split.", "hint"));
  }
  if (evaluation?.job_uri?.startsWith("https://")) {
    const link = element("a", "Open evaluation job in Azure ML", "action-link");
    link.href = evaluation.job_uri;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    $("evaluation-summary").append(link);
  }
}

function releaseDetailsTable(caption, headings, rows) {
  const table = element("table", undefined, "release-details-table");
  table.append(element("caption", caption));
  const head = element("thead");
  const header = element("tr");
  for (const text of headings) {
    const cell = element("th", text);
    cell.setAttribute("scope", "col");
    header.append(cell);
  }
  head.append(header);
  const body = element("tbody");
  for (const [label, value] of rows) {
    const row = element("tr");
    const heading = element("th", label);
    heading.setAttribute("scope", "row");
    row.append(heading, element("td", value ?? "Not recorded"));
    body.append(row);
  }
  table.append(head, body);
  const wrap = element("div", undefined, "table-wrap release-details");
  wrap.append(table);
  return wrap;
}

function gateBadge(passed, label) {
  const state = passed === true ? "pass" : passed === false ? "fail" : "unknown";
  return element("span", label ?? (passed === true ? "PASS" : passed === false ? "FAIL" : "NOT RECORDED"),
    `gate-badge ${state}`);
}

function renderReleaseReadiness() {
  const target = $("release-readiness");
  target.replaceChildren();
  const entry = releaseEvaluations.find((item) => item.run_id === $("release-evaluation").value);
  if (!entry) {
    target.append(element("p", "Complete an evaluation to review candidate readiness.", "hint"));
    $("plan-release").disabled = true;
    return;
  }
  const gate = entry.readiness;
  $("release-approval-form").hidden = selected?.release?.status !== "awaiting_approval"
    || selected.release.source_run_id !== entry.run_id;
  const heading = element("h3", "Release readiness: ");
  heading.append(gateBadge(gate.passed, gate.status.toUpperCase()));
  target.append(heading);
  const policy = gate.policy;
  if (policy) {
    target.append(releaseDetailsTable("Release policy", ["Check", "Release requirement"], [
      ["Policy version", policy.version],
      ["Quality metrics", `All must pass: ${policy.metrics.map((name) => name[0].toUpperCase() + name.slice(1)).join(", ")}.`],
      ["Sealed-test examples", `At least ${policy.minimum_examples} examples.`],
      ["Improvement over original SLM", `At least ${Number((policy.minimum_absolute_uplift * 100).toFixed(6))} percentage points.`],
      ["Gap to GPT baseline", `Less than ${Number((policy.maximum_relative_gpt_shortfall_exclusive * 100).toFixed(6))}% below GPT (relative gap).`],
      ["Statistical confidence", `The ${Number((policy.confidence * 100).toFixed(6))}% confidence interval for paired uplift must stay entirely above zero.`],
      ...(policy.perfect_baseline_exception ? [
        ["Perfect baseline exception", "For a metric with original SLM scores exactly 100%, the fine-tuned model must also score exactly 100% on every sealed-test example. Only then are the uplift and positive-confidence requirements waived. Sample count and GPT-gap checks still apply."],
      ] : []),
      ["Confidence calculation", `Paired percentile bootstrap; ${policy.bootstrap_replicates.toLocaleString("en-US")} resamples; seed ${policy.bootstrap_seed}.`],
      ["Informational only", "Latency, token usage and token costs do not affect readiness."],
      ["Policy controls", "Fixed policy. No edits or overrides."],
    ]));
  } else {
    target.append(element("p", "Policy details are not recorded. Refresh evaluation evidence before planning a release.", "hint"));
  }
  const sampleStatus = element("div", undefined, "summary-card gate-summary");
  const countRecorded = Number.isInteger(gate.sample_count);
  sampleStatus.append(
    gateBadge(policy && countRecorded ? gate.sample_count >= policy.minimum_examples : undefined),
    element("span", `Sealed-test examples: ${countRecorded ? gate.sample_count : "Not recorded"} observed; ${policy ? `at least ${policy.minimum_examples} required` : "requirement not recorded"}.`),
  );
  target.append(sampleStatus);
  const table = element("table");
  table.append(element("caption", "Quality metric results"));
  const header = element("tr");
  for (const label of ["Metric", "Fine-tuned", "Original", "GPT", "Uplift (points)", "GPT gap", "95% uplift CI (points)", "Uplift gate", "GPT gap gate", "Confidence gate", "Result"]) {
    const cell = element("th", label);
    cell.setAttribute("scope", "col");
    header.append(cell);
  }
  const head = element("thead");
  head.append(header);
  const body = element("tbody");
  for (const metric of gate.metrics) {
    const atCeiling = metric.baseline_at_ceiling === true;
    const row = element("tr");
    for (const text of [metric.metric, scoreText(metric.fine_tuned), scoreText(metric.oob), scoreText(metric.gpt),
      (metric.absolute_uplift * 100).toFixed(2), `${(metric.relative_gpt_shortfall * 100).toFixed(2)}%`,
      metric.uplift_ci95.map((n) => (n * 100).toFixed(2)).join(" to ")]) row.append(element("td", text));
    for (const [check, requirement] of [
      ["uplift", atCeiling ? "Required: remain exactly 100% on every example; no regression. Two-point uplift is waived only for a perfect tie."
        : policy ? `Required: at least ${Number((policy.minimum_absolute_uplift * 100).toFixed(6))} points uplift.` : "Requirement not recorded."],
      ["gpt_gap", policy ? `Required: strictly less than ${Number((policy.maximum_relative_gpt_shortfall_exclusive * 100).toFixed(6))}% gap.` : "Requirement not recorded."],
      ["positive_confidence_interval", atCeiling
        ? "Positive uplift confidence is waived only for a perfect 100% tie; this is not evidence of improvement."
        : "Required: lower confidence bound strictly above 0."],
    ]) {
      const cell = element("td");
      const label = metric.ceiling_exception_applied && check !== "gpt_gap" ? "PASS (PERFECT TIE)" : undefined;
      cell.append(gateBadge(metric.checks?.[check], label), element("small", requirement, "gate-detail"));
      row.append(cell);
    }
    const result = element("td");
    result.append(gateBadge(metric.passed));
    row.append(result);
    body.append(row);
  }
  table.append(head, body);
  const wrap = element("div", undefined, "table-wrap");
  wrap.append(table);
  target.append(wrap);
  if (gate.reasons.length) {
    const reasons = element("ul", undefined, "gate-reasons");
    const checkLabels = {
      uplift: "minimum uplift", gpt_gap: "GPT baseline gap",
      positive_confidence_interval: "positive 95% uplift confidence interval",
    };
    for (const reason of gate.reasons) {
      const item = element("li");
      item.append(gateBadge(false), element("span",
        reason.replace(/\b(uplift|gpt_gap|positive_confidence_interval)\b/g, (name) => checkLabels[name])));
      reasons.append(item);
    }
    target.append(reasons);
  }
  target.append(element("h3", "Activity lineage"));
  const lineage = entry.lineage;
  for (const [activity, fields] of [
    ["1. Data preparation", [
      ["Preparation run", "preparation_run"], ["Published dataset", "training_dataset"],
    ]],
    ["2. Training", [
      ["Studio run", "training_run"], ["Azure ML job", "training_job"],
      ["Training date", "training_date"], ["Model", "model"],
      ["Dataset", "training_dataset"], ["Dataset samples", "training_samples"],
      ["Epochs", "epochs"], ["Adapter name", "adapter_name"],
    ]],
    ["3. Evaluation", [
      ["Studio run", "evaluation_run"], ["Azure ML job", "evaluation_job"],
      ["Dataset", "evaluation_dataset"], ["Split", "evaluation_split"],
      ["Evaluated samples", "evaluation_samples"], ["Judge model", "judge"],
      ["GPT baseline", "gpt_baseline"], ["Sealed-test SHA-256", "sealed_test_sha256"],
    ]],
  ]) {
    target.append(releaseDetailsTable(activity, ["Parameter", "Recorded value"],
      fields.map(([label, key]) => [label, lineage[key]])));
  }
  $("plan-release").disabled = busy || running() || !gate.passed || !policy;
}

function renderProductionRelease() {
  const select = $("release-evaluation");
  const previous = select.value;
  select.replaceChildren();
  for (const entry of releaseEvaluations) {
    const option = element("option", `${entry.evaluation_job_name} — ${entry.readiness.status}`);
    option.value = entry.run_id;
    select.append(option);
  }
  const preferred = selected?.release?.source_run_id || previous || selected?.run_id;
  if (releaseEvaluations.some((entry) => entry.run_id === preferred)) select.value = preferred;
  else if (releaseEvaluations.length) select.value = releaseEvaluations[0].run_id;
  select.disabled = busy || running();
  renderReleaseReadiness();
  const release = selected?.run_type === "release" ? selected.release : undefined;
  const plan = release?.plan;
  $("release-approval-form").hidden = release?.status !== "awaiting_approval";
  $("approve-release").disabled = busy || running();
  $("gpu-cost-confirmation").hidden = plan?.action !== "register_and_deploy";
  field($("release-approval-form"), "accept_gpu_cost").required = plan?.action === "register_and_deploy";
  if (plan) {
    summary("decision-summary", "Reviewed release plan", [
      ["Action", plan.action === "register" ? "Register only — no GPU allocated" : "Register and deploy"],
      ["Model", `${plan.model_name}:${plan.model_version}`], ["Plan expires", plan.expires_at],
      ["Immutable base model", plan.base_model_reference], ["Plan fingerprint", release.plan_hash],
    ]);
    if (plan.gpu) summary("decision-summary", "GPU sizing and ongoing hourly cost", [
      ["SKU", plan.gpu.sku], ["Region", plan.gpu.region],
      ["Estimated hourly USD", `$${plan.gpu.price.hourly_usd}`], ["Price basis", plan.gpu.price.basis],
      ["GPT rationale", plan.gpu.rationale], ["Serving", `${plan.gpu.serving_dtype}; estimated ${plan.gpu.estimated_required_gpu_memory_gb} GB required`],
      ["Availability", plan.gpu.quota_note], ["Price source", plan.gpu.price.source],
    ]);
  }
  if (release?.registered_model) summary("decision-summary", "Registered candidate", [
    ["Model", `${release.registered_model.name}:${release.registered_model.version}`],
    ["Identity", release.registered_model.id], ["Status", release.status],
  ]);
  if (release?.endpoint) {
    summary("deployment-summary", "Dedicated authenticated scoring endpoint", [
      ["Endpoint", release.endpoint.name], ["Scoring URI", release.endpoint.uri ?? "Provisioning"],
      ["Validation", release.endpoint.validated ? `Passed ${release.endpoint.validated_at}` : "Not yet validated"],
      ["Authentication", "Azure ML token; credentials are never shown"],
      ["Monitoring schedule", release.monitoring?.schedule ?? "Not yet created"],
      ["Monitoring state", release.monitoring?.status ?? "Pending"],
      ["Monitoring scope", release.monitoring?.note ?? "Request-feature PSI"],
    ]);
  }
}

function renderDataSamples(samples) {
  const section = element("section", undefined, "sample-data");
  const heading = element("h3", "Sample data");
  heading.id = "sample-data-heading";
  section.setAttribute("aria-labelledby", heading.id);
  section.append(heading);
  if (!samples.length) {
    section.append(element("p", "No sample data is available for this run.", "hint"));
  } else {
    const wrapper = element("div", undefined, "table-wrap");
    wrapper.tabIndex = 0;
    wrapper.setAttribute("role", "region");
    wrapper.setAttribute("aria-label", "Sample data table");
    const table = element("table", undefined, "sample-data-table");
    table.append(element("caption", "Prepared dataset samples with all generated fields", "sr-only"));
    const header = element("thead");
    const headerRow = element("tr");
    const columns = [
      ["split", "Split"], ["id", "ID"], ["type", "Type"], ["question", "Question"],
      ["context", "Context"], ["oracle_context", "Oracle context"],
      ["cot_answer", "CoT answer"], ["instruction", "Instruction"],
    ];
    const known = new Set(columns.map(([key]) => key));
    for (const sample of samples) {
      for (const key of Object.keys(sample)) {
        if (!known.has(key)) {
          columns.push([key, key.replaceAll("_", " ")]);
          known.add(key);
        }
      }
    }
    for (const [key, label] of columns) {
      const cell = element("th", label);
      cell.setAttribute("scope", "col");
      cell.dataset.field = key;
      headerRow.append(cell);
    }
    header.append(headerRow);
    const body = element("tbody");
    for (const sample of samples) {
      const row = element("tr");
      for (const [key] of columns) {
        const value = key === "split" ? sample.split ?? "train" : sample[key];
        const text = value === undefined ? "Not in saved preview"
          : typeof value === "object" ? JSON.stringify(value, null, 2) : String(value);
        const cell = element("td", text);
        cell.dataset.field = key;
        row.append(cell);
      }
      body.append(row);
    }
    table.append(header, body);
    wrapper.append(table);
    section.append(element("p", "Up to two samples per split. Scroll horizontally to review every field.", "hint"), wrapper);
  }
  $("data-summary").append(section);
}

function availableDatasets() {
  return runs.filter((run) => run.mode === "production" && run.simulated === false
    && !["training", "evaluation", "release"].includes(run.run_type) && run.recipe === "context-grounded"
    && run.stages.data_preparation.status === "succeeded" && run.data_card?.asset_reference);
}

function renderTrainingDatasets() {
  const control = field($("training-form"), "dataset_run_id");
  const previous = control.value;
  const datasets = availableDatasets();
  control.replaceChildren();
  for (const run of datasets) {
    const option = element("option", `${run.data_card.asset_reference} (${run.data_card.record_count} samples)`);
    option.value = run.run_id;
    control.append(option);
  }
  if (datasets.some((run) => run.run_id === previous)) control.value = previous;
  else if (datasets.some((run) => run.run_id === selected?.run_id)) control.value = selected.run_id;
  updateDatasetControls();
}

function updateDatasetControls() {
  const form = $("training-form");
  const manual = field(form, "dataset_source").value === "manual";
  $("dataset-run-field").hidden = manual;
  $("dataset-id-field").hidden = !manual;
  field(form, "dataset_id").disabled = !manual;
  field(form, "dataset_id").required = manual;
  field(form, "dataset_run_id").disabled = manual;
  field(form, "dataset_run_id").required = !manual;
  const rag = field(form, "dataset_recipe").value === "context-assisted";
  $("dataset-selection-note").textContent = rag
    ? "RAG retrieves context at inference time; its separate training schema is not defined. Choose RAFT to train."
    : manual ? "Use a pinned URI-folder asset: azureml:dataset-name:version. The job validates train/validation JSONL records."
      : "Choose a successfully published production dataset. Preparing a new dataset is not required.";
  $("train").disabled = busy || runs.some((run) => run.status === "running") || rag
    || (!manual && !field(form, "dataset_run_id").value);
}

function showDatasetFormat() {
  const rag = field($("training-form"), "dataset_recipe").value === "context-assisted";
  $("dataset-format-title").textContent = rag ? "RAG dataset guidance" : "Expected RAFT dataset format";
  $("dataset-format-description").textContent = rag
    ? "RAG supplies retrieved context when answering; this is not a fine-tuning recipe by itself. A separate RAG training schema is not defined, so RAG training is disabled. The example below is the supported RAFT format."
    : "Enter azureml:dataset-name:version for a URI-folder asset containing train.jsonl and validation.jsonl. Keep test.jsonl sealed for evaluation. Each JSONL line must contain all fields below; type must be oracle or distractor. Instructions contain the context and question, and cot_answer contains the grounded answer. Source contexts must not overlap across splits.";
  $("dataset-format-example").textContent = JSON.stringify({
    id: "sample-1", type: "oracle", question: "What is HBM?",
    context: { sentences: [["HBM is high-bandwidth memory."]] },
    oracle_context: "HBM is high-bandwidth memory.",
    instruction: "<DOCUMENT>HBM is high-bandwidth memory.</DOCUMENT>\n### Question:\nWhat is HBM?",
    cot_answer: "<ANSWER>High-bandwidth memory.</ANSWER>",
  }, null, 2);
  if (!$("dataset-format-dialog").open) $("dataset-format-dialog").showModal();
}

function renderStageNavigation(run = selected) {
  $("stage-nav").replaceChildren();
  visibleStages().forEach((name, index) => {
    const deployment = run?.stages.deployment;
    const state = (name === "decision" && deployment && deployment.status !== "not_started"
      ? deployment : run?.stages[name]) ?? { status: "not_started" };
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
}

function renderSelected() {
  if (!selected || !config) return;
  const run = selected;
  $("run-title").textContent = run.training ? `${modelLabel(run.training.config.base_model)} fine-tuning` : "New fine-tuning run";
  $("run-meta").textContent = `Run ${referenceLabel(run.run_id)} | Updated ${new Date(run.updated_at).toLocaleString()}`;
  $("run-status").textContent = run.status.toUpperCase();
  $("run-status").className = `badge ${run.status}`;
  $("run-record").textContent = JSON.stringify(run, null, 2);
  renderStageNavigation(run);
  const active = config.stage_order.find((name) => run.stages[name].status === "running");
  $("progress-panel").hidden = !active;
  if (active) {
    $("progress-label").textContent = `${stageLabels[active]} in progress`;
    $("progress-value").textContent = "Running";
    $("stage-progress").removeAttribute("value");
    $("stage-progress").textContent = "Running";
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
    if (card.release_sample_check) {
      const check = card.release_sample_check;
      const note = element("p", undefined, "gate-summary");
      note.append(gateBadge(check.passed), element("span",
        `Release sample-count check: ${check.observed_test_examples} sealed-test examples; at least ${check.minimum_test_examples} required. ${check.passed ? "Sample requirement met; quality gates still apply." : "Generate more samples before release evaluation."}`));
      $("data-summary").append(note);
    }
    renderDataSamples(card.samples ?? []);
  } else {
    $("data-summary").append(element("p", "Data preparation has not completed yet.", "hint"));
  }
  if (run.training) {
    summary("training-summary", "Candidate lineage", [
      ["Base model", modelLabel(run.training.config.base_model)],
      [stageSucceeded("training") ? "Candidate" : "Planned candidate", referenceLabel(run.training.candidate_id)],
      ["Training job", referenceLabel(run.training.job_name)],
      ["Azure job status", run.training.job_status], ["Dataset", run.training.dataset_id],
      [stageSucceeded("training") ? "Saved artifact" : "Expected artifact (not produced)", referenceLabel(run.training.model_artifact)],
    ]);
    if (run.training.job_uri?.startsWith("https://")) {
      const link = element("a", "Open training job in Azure ML", "action-link");
      link.href = run.training.job_uri;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      $("training-summary").append(link);
    }
  }
  renderProductionEvaluation();
  renderProductionRelease();
  $("train").disabled = busy || running();
  $("evaluate").disabled = busy || running() || !evaluationJobs.find((job) => job.job_name === $("evaluation-job").value)?.eligible;
  $("create-form").querySelector("button").disabled = busy || running();
  $("run-select").disabled = busy;
  $("new-run").disabled = busy;
  renderTrainingDatasets();
  renderEvaluationJobs();
  renderTaskControls();
  setTab(tab);
  renderRuns();
  schedulePoll();
}

function schedulePoll() {
  clearTimeout(pollTimer);
  if (activeRun()) {
    pollTimer = setTimeout(async () => {
      try {
        await refresh();
      } catch (error) {
        showError(error);
        schedulePoll();
      }
    }, PRODUCTION_POLL_INTERVAL_MS);
  }
}

async function refresh() {
  if (refreshing) {
    schedulePoll();
    return;
  }
  refreshing = true;
  const version = selectionVersion;
  try {
    const result = await api("/api/runs");
    if (version !== selectionVersion) return;
    const previous = selected;
    runs = result.runs;
    if (selected) selected = runs.find((run) => run.run_id === selected.run_id);
    if (!selected && runs.length) {
      selected = runs[0];
      tab = selected.current_stage;
    }
    if (trainingNeedsRefresh(selected)) {
      const latest = await api(`/api/runs/${encodeURIComponent(selected.run_id)}`);
      if (version !== selectionVersion) return;
      selected = latest;
      runs = runs.map((run) => run.run_id === latest.run_id ? latest : run);
    }
    if (config.enabled_stages?.includes("evaluation")) {
      const jobs = await api("/api/training-jobs");
      if (version !== selectionVersion) return;
      evaluationJobs = jobs.jobs;
    }
    if (config.enabled_stages?.includes("decision")) {
      const catalogue = await api("/api/release-evaluations");
      if (version !== selectionVersion) return;
      releaseEvaluations = catalogue.evaluations;
    }
    if (previous && selected) {
      const completedRunningStage = config.stage_order.find((name) =>
        previous.stages[name].status === "running"
        && selected.stages[name].status === "succeeded");
      if (completedRunningStage) tab = completedRunningStage;
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
      $("wizard-status").textContent = "Prepare, train, evaluate, then review candidate policy and an approved deployment plan.";
      updateWizardNavigation();
      renderTrainingDatasets();
      renderEvaluationJobs();
      renderStageNavigation();
      renderProductionEvaluation();
      renderProductionRelease();
      renderTaskControls();
      schedulePoll();
      setTab(tab);
    }
  } catch (error) {
    if (version === selectionVersion && selected) renderSelected();
    throw error;
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
    runs = runs.map((run) => run.run_id === result.run_id ? result : run);
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
    if (control?.type === "checkbox") control.checked = value;
    else if (control) control.value = value;
    if (name === "base_model") field(form, "production_base_model").value = value;
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
    renderSelected();
    await refresh();
  } catch (error) {
    showError(error);
  } finally {
    busy = false;
    renderSelected();
  }
}

$("create-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  action(async () => {
    selected = await api("/api/runs", {
      source_folder: field(form, "source_folder").value.trim(),
      recipe: field(form, "recipe").value,
      num_questions: Number(field(form, "num_questions").value),
      oracle_probability: Number(field(form, "oracle_probability").value),
      data_asset_name: field(form, "data_asset_name").value.trim(),
      num_samples: Number(field(form, "num_samples").value),
      confirmed: field(form, "confirmed").checked,
    });
    tab = "data_preparation";
    field(form, "confirmed").checked = false;
  });
});
function trainingBody(form) {
  const body = {
    confirmed: field(form, "confirmed").checked,
    base_model: field(form, "production_base_model").value.trim(),
  };
  for (const name of ["max_seq_length", "batch_size", "gradient_accumulation", "learning_rate", "epochs", "lora_rank"]) {
    body[name] = Number(field(form, name).value);
  }
  body.dataset_recipe = field(form, "dataset_recipe").value;
  if (field(form, "dataset_source").value === "manual") body.dataset_id = field(form, "dataset_id").value.trim();
  else body.dataset_run_id = field(form, "dataset_run_id").value;
  body.seed = Number(field(form, "seed").value);
  body.load_in_4bit = field(form, "load_in_4bit").checked;
  body.chat_template = field(form, "chat_template").value.trim();
  body.lora_adapter_name = field(form, "lora_adapter_name").value.trim();
  return body;
}
$("training-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  action(async () => {
    selected = await api("/api/training", trainingBody(form));
    tab = "training";
    field(form, "confirmed").checked = false;
  });
});
$("evaluation-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  action(async () => {
    selected = await api("/api/evaluation", {
      confirmed: field(form, "confirmed").checked,
      job_name: $("evaluation-job").value,
    });
    tab = "evaluation";
    field(form, "confirmed").checked = false;
  });
});
$("release-evaluation").addEventListener("change", renderReleaseReadiness);
field($("release-plan-form"), "action").addEventListener("change", () => {
  const deploying = field($("release-plan-form"), "action").value === "register_and_deploy";
  $("release-deployment-options").hidden = !deploying;
  for (const name of ["endpoint_name", "production_data_uri", "monitor_compute"]) field($("release-plan-form"), name).required = deploying;
});
$("release-plan-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  action(async () => {
    const deploying = field(form, "action").value === "register_and_deploy";
    selected = await api("/api/release-plans", {
      evaluation_run_id: $("release-evaluation").value,
      action: field(form, "action").value, model_name: field(form, "model_name").value.trim(),
      confirmed: field(form, "confirmed").checked,
      ...(deploying ? { endpoint_name: field(form, "endpoint_name").value.trim(),
        production_data_uri: field(form, "production_data_uri").value.trim(),
        monitor_compute: field(form, "monitor_compute").value.trim() } : {}),
    });
    tab = "decision";
    field(form, "confirmed").checked = false;
  });
});
$("release-approval-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  action(async () => {
    selected = await api(`/api/release-plans/${encodeURIComponent(selected.run_id)}/approve`, {
      plan_hash: selected.release.plan_hash, actor: field(form, "actor").value.trim(),
      reason: field(form, "reason").value.trim(), confirmed: field(form, "confirmed").checked,
      accept_gpu_cost: field(form, "accept_gpu_cost").checked,
    });
    field(form, "confirmed").checked = false;
    field(form, "accept_gpu_cost").checked = false;
  });
});
$("new-run").addEventListener("click", () => {
  setTab("data_preparation");
  field($("create-form"), "source_folder").focus();
});
$("evaluation-job").addEventListener("change", () => {
  field($("evaluation-form"), "confirmed").checked = false;
  renderEvaluationJob();
});
$("dataset-format-help").addEventListener("click", showDatasetFormat);
field($("training-form"), "dataset_source").addEventListener("change", () => {
  updateDatasetControls();
  if (field($("training-form"), "dataset_source").value === "manual") showDatasetFormat();
});
field($("training-form"), "dataset_recipe").addEventListener("change", () => {
  updateDatasetControls();
  if (field($("training-form"), "dataset_source").value === "manual") showDatasetFormat();
});
$("refresh").addEventListener("click", () => action(refresh));
$("run-select").addEventListener("change", (event) => selectRun(event.target.value));
$("wizard-back").addEventListener("click", () => {
  const stages = visibleStages();
  const index = stages.indexOf(tab);
  if (index > 0) setTab(stages[index - 1]);
});
$("wizard-next").addEventListener("click", () => {
  const stages = visibleStages();
  const index = stages.indexOf(tab);
  if (index >= 0 && index < unlockedStageIndex()) setTab(stages[index + 1]);
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

$("cancel-task").addEventListener("click", () => {
  const run = activeRun();
  if (!run) return;
  action(async () => {
    selected = await api(`/api/runs/${encodeURIComponent(run.run_id)}/cancel`, {});
    runs = runs.map((entry) => entry.run_id === selected.run_id ? selected : entry);
    tab = selected.current_stage;
  });
});

async function initialize() {
  config = await api("/api/config");
  if (config.mode !== "production" || config.simulated !== false) {
    throw new Error("A production API with non-simulated execution is required.");
  }
  $("evaluation-showcase").hidden = true;
  prefill(config.training_defaults ?? {});
  for (const model of config.models) {
    const suggestion = element("option");
    suggestion.value = model.id;
    $("production-models").append(suggestion);
  }
  await refresh();
  if (selected?.training?.config) prefill(selected.training.config);
}
$("evidence-search").addEventListener("input", renderEvidenceRows);
$("evidence-type").addEventListener("change", renderEvidenceRows);
$("evidence-comparison").addEventListener("change", renderEvidenceRows);
initialize().catch(showError);
