const $ = (id) => document.getElementById(id);
const stageLabels = {
  data_preparation: "Data",
  training: "Fine-tuning",
  evaluation: "Evaluation",
  decision: "Decision",
  deployment: "Deployment",
};
const recipeLabels = {
  "context-grounded": "Context-grounded (RAFT)",
  "context-assisted": "Context-assisted (RAG)",
};
let config;
let runs = [];
let selected;
let tab = "data_preparation";
let busy = false;
let pollTimer;
let refreshing = false;
let selectionVersion = 0;

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
  const response = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: { Accept: "application/json", ...(body === undefined ? {} : { "Content-Type": "application/json" }) },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
  const result = await response.json();
  if (!response.ok) {
    const detail = result.detail;
    const message = typeof detail === "string" ? detail :
      Array.isArray(detail) ? detail.map((issue) => `${issue.loc.join(".")}: ${issue.msg}`).join("; ") :
      `API returned ${response.status}.`;
    throw new Error(message);
  }
  return result;
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

function renderMetrics(evaluation) {
  const wrap = element("div", undefined, "table-wrap");
  const table = element("table");
  table.append(element("caption", "Model comparison on the same sealed test split"));
  const head = element("tr");
  head.append(element("th", "Model"));
  for (const metric of config.metrics) {
    const unit = metric.unit === "score" ? "%" : metric.unit;
    head.append(element("th", `${metric.label} (${unit})`));
  }
  const thead = element("thead");
  thead.append(head);
  table.append(thead);
  const tbody = element("tbody");
  const labels = { candidate: "Fine-tuned SLM", base_slm: "Original SLM", gpt: "GPT baseline" };
  for (const model of evaluation.models) {
    const row = element("tr");
    row.append(element("td", labels[model.role] ?? model.role));
    for (const metric of config.metrics) {
      const value = model.metrics[metric.key];
      const formatted = value == null ? "Not available" :
        metric.key === "token_cost" ? `$${Number(value).toFixed(5)}` :
        metric.key === "latency" ? Number(value).toFixed(0) :
        `${(Number(value) * 100).toFixed(1)}%`;
      row.append(element("td", formatted));
    }
    tbody.append(row);
  }
  table.append(tbody);
  wrap.append(table);
  $("evaluation-summary").append(wrap);
  $("evaluation-summary").append(element("p",
    "SLM costs use a per-response compute allocation; GPT costs use token pricing.", "hint"));
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
    const button = element("button", undefined, "stage-step");
    button.type = "button";
    button.dataset.stage = name;
    button.append(element("strong", `${index + 1}. ${stageLabels[name]}`), element("span", state.status.replaceAll("_", " ")));
    button.addEventListener("click", () => setTab(name));
    $("stage-nav").append(button);
  });
  const active = config.stage_order.find((name) => run.stages[name].status === "running");
  $("progress-panel").hidden = !active;
  if (active) {
    const value = run.stages[active].progress;
    $("progress-label").textContent = `${stageLabels[active]} in progress`;
    $("progress-value").textContent = `${value}%`;
    $("stage-progress").value = value;
    $("stage-progress").textContent = `${value}%`;
  }
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
      ["Source", referenceLabel(card.source_folder ?? card.source_references?.join(", "))],
      ["Split counts", JSON.stringify(card.split_counts)],
      ["Validation", typeof card.validation === "object" ? `${card.validation.status}: ${card.validation.checked_records} records checked, ${card.validation.invalid_records} invalid` : card.validation],
      ["Fingerprint", referenceLabel(card.fingerprint)], ["Data asset", referenceLabel(card.asset_reference ?? card.azure_ml_asset)],
    ]);
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
    renderMetrics(run.evaluation);
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
    summary("decision-summary", "Registration decision", [
      ["Model", `${referenceLabel(registration.model_name)}:${registration.version}`],
      ["Approver", registration.actor === "local-demo-user" ? "local-user" : registration.actor],
      ["Override", registration.override ? "Yes" : "No"],
      ["Reason", registration.reason === "Simulated gate passed." ? "Gate passed." : registration.reason || "Gate passed"],
      ["Decision time", registration.decided_at],
    ]);
  }
  if (run.deployment) {
    summary("deployment-summary", "Deployment outcome", [
      ["Endpoint", referenceLabel(run.deployment.endpoint_name)],
      ["Endpoint reference", referenceLabel(run.deployment.endpoint_uri)],
      ["Health", run.deployment.monitoring.status],
      ["Requests", run.deployment.monitoring.request_count],
      ["Error rate", `${(run.deployment.monitoring.error_rate * 100).toFixed(2)}%`],
      ["P95 latency", `${run.deployment.monitoring.p95_latency_ms} ms`],
      ["Drift score", run.deployment.monitoring.drift_score],
    ]);
  } else {
    $("deployment-summary").append(element("p",
      registration ? "Candidate registered. Deployment still needs your confirmation." : "Register a candidate before deployment.", "hint"));
  }
  const locked = busy || running() || run.status === "failed";
  $("train").disabled = locked || !stageSucceeded("data_preparation") || stageSucceeded("training");
  $("suggest").disabled = locked || !stageSucceeded("data_preparation");
  $("evaluate").disabled = locked || !stageSucceeded("training") || stageSucceeded("evaluation");
  $("register").disabled = locked || !stageSucceeded("evaluation") || Boolean(registration);
  $("deploy").disabled = locked || !registration || stageSucceeded("deployment");
  $("override-fields").hidden = run.evaluation?.gate.status !== "blocked" || Boolean(registration);
  $("create-form").querySelector("button").disabled = busy;
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
        pollTimer = setTimeout(schedulePoll, 3000);
      }
    }, 1200);
  }
}

async function refresh() {
  if (refreshing) return;
  refreshing = true;
  const version = selectionVersion;
  try {
    const result = await api("/api/runs");
    if (version !== selectionVersion) return;
    runs = result.runs;
    if (selected) selected = runs.find((run) => run.run_id === selected.run_id);
    if (!selected && runs.length) {
      selected = runs[0];
      tab = selected.current_stage;
    }
    if (selected) renderSelected();
    else renderRuns();
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
    renderSelected();
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
      scenario: field(form, "scenario").value,
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
bindForm("deployment-form", "deployment", (form) => ({ confirmed: field(form, "confirmed").checked }), "deployment");
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
  if (config.mode !== "demo" || config.production_enabled) {
    throw new Error("The API is configured for a different execution mode.");
  }
  const models = field($("training-form"), "base_model");
  for (const model of config.models) {
    const option = element("option", modelLabel(model.id));
    option.value = model.id;
    models.append(option);
  }
  await refresh();
  if (selected?.training?.config) prefill(selected.training.config);
}
initialize().catch(showError);
