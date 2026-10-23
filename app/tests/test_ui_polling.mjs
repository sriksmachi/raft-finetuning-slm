import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const source = readFileSync(new URL("../static/app.js", import.meta.url), "utf8")
  .replace(/^import[^\n]+\n/, "")
  .replace("initialize().catch(showError);", "");

function uiContext(request) {
  const controls = new Map();
  const timers = [];
  const createElement = (tagName) => {
    const fields = new Map();
    return {
      tagName, children: [], attributes: {}, dataset: {}, value: "", type: "text",
      classList: { toggle() {} },
      listeners: {},
      addEventListener(name, callback) { this.listeners[name] = callback; },
      showModal() { this.open = true; },
      append(...nodes) { this.children.push(...nodes); },
      replaceChildren(...nodes) { this.children = nodes; },
      setAttribute(name, value) { this.attributes[name] = value; },
      removeAttribute(name) { delete this.attributes[name]; },
      querySelector() { return createElement("button"); },
      elements: { namedItem(name) {
        if (!fields.has(name)) fields.set(name, createElement("input"));
        return fields.get(name);
      } },
    };
  };
  const context = vm.createContext({
    document: {
      createElement,
      getElementById: (id) => {
        if (!controls.has(id)) controls.set(id, createElement("div"));
        return controls.get(id);
      },
    },
    request,
    setTimeout: (callback, delay) => { timers.push({ callback, delay }); return timers.length; },
    clearTimeout() {},
  });
  vm.runInContext(source, context);
  return { context, timers, controls };
}

for (const stage of ["data_preparation", "training", "evaluation", "decision", "deployment"]) {
  test(`${stage} hides forms on every tab and restores them after cancellation`, () => {
    const { context, controls } = uiContext();
    context.stageName = stage;
    vm.runInContext(`
      config = {stage_order: ["data_preparation", "training", "evaluation", "decision", "deployment"],
                enabled_stages: ["decision"]};
      selected = {run_id: "prod-active", status: "running",
                  stages: {[stageName]: {status: "running"}}};
      renderTaskControls();
    `, context);
    const forms = ["create-form", "training-form", "evaluation-form", "release-plan-form", "release-approval-form"];
    for (const tab of ["data_preparation", "training", "evaluation", "decision"]) {
      context.tabName = tab;
      vm.runInContext("setTab(tabName)", context);
      for (const id of forms) assert.equal(controls.get(id).hidden, true, id);
    }
    assert.equal(controls.get("task-controls").hidden, false);
    vm.runInContext(`
      selected.status = "cancelled";
      selected.stages[stageName].status = "cancelled";
      renderTaskControls();
      schedulePoll();
    `, context);
    for (const id of forms.slice(0, 4)) assert.equal(controls.get(id).hidden, false, id);
    assert.equal(controls.get("release-approval-form").hidden, true);
    assert.equal(controls.get("task-controls").hidden, true);
  });
}

test("an active background run keeps forms hidden while reviewing a different run", () => {
  const { context, controls, timers } = uiContext();
  vm.runInContext(`
    selected = {run_id: "prod-old", stages: {}};
    runs = [{run_id: "prod-active", status: "running"}];
    renderTaskControls();
    schedulePoll();
    busy = true;
    renderTaskControls();
  `, context);
  assert.equal(controls.get("create-form").hidden, true);
  assert.equal(controls.get("task-controls").hidden, false);
  assert.equal(controls.get("cancel-task").disabled, true);
  assert.equal(timers.at(-1).delay, 10000);
});

test("cancel action targets the active run and displays cancellation errors", async () => {
  const calls = [];
  const { context, controls } = uiContext(async (path, body) => {
    calls.push([path, body]);
    throw new Error("Unable to stop task");
  });
  vm.runInContext(`
    selected = {run_id: "prod-old", stages: {}};
    runs = [{run_id: "prod-active", status: "running"}];
    renderSelected = renderTaskControls;
  `, context);
  await controls.get("cancel-task").listeners.click();
  // The click handler starts the asynchronous action without blocking the event.
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(calls[0][0], "/api/runs/prod-active/cancel");
  assert.equal(controls.get("error").textContent, "Unable to stop task");
  assert.equal(controls.get("create-form").hidden, true);
  assert.equal(controls.get("cancel-task").disabled, false);
});

test("successful cancellation restores forms even when the following refresh fails", async () => {
  const { context, controls, timers } = uiContext(async () => ({
    run_id: "prod-active", status: "cancelled", current_stage: "training",
    stages: {training: {status: "cancelled"}},
  }));
  vm.runInContext(`
    selected = {run_id: "prod-active", status: "running", stages: {training: {status: "running"}}};
    runs = [selected];
    renderSelected = () => { renderTaskControls(); schedulePoll(); };
    refresh = async () => { throw new Error("Refresh failed"); };
    $("training-form").elements.namedItem("epochs").value = "3";
  `, context);
  controls.get("cancel-task").listeners.click();
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(controls.get("training-form").hidden, false);
  assert.equal(controls.get("task-controls").hidden, true);
  assert.equal(controls.get("training-form").elements.namedItem("epochs").value, "3");
  assert.equal(vm.runInContext("runs[0].status", context), "cancelled");
  assert.equal(timers.length, 1);
  assert.equal(controls.get("error").textContent, "Refresh failed");
});

test("production always polls after exactly 10 seconds and stops after completion", async () => {
  const { context, timers } = uiContext();
  vm.runInContext(`
    selected = {stages: {data_preparation: {status: "running"}}};
    schedulePoll();
  `, context);
  assert.equal(timers.at(-1).delay, 10000);
  vm.runInContext(`selected.stages.data_preparation.status = "succeeded"; schedulePoll();`, context);
  assert.equal(timers.length, 1);
  vm.runInContext(`
    selected.stages.data_preparation.status = "running";
    refresh = async () => { throw new Error("Temporary connection error"); };
    schedulePoll();
  `, context);
  await timers.at(-1).callback();
  assert.equal(timers.at(-1).delay, 10000);
});

for (const stage of ["data_preparation", "training", "evaluation"]) {
test(`each production ${stage} refresh reads the run log, including final status`, async () => {
  const calls = [];
  let status = "running";
  const { context, timers } = uiContext(async (path) => {
    calls.push(path);
    if (path.endsWith("/log")) return { run_id: "prod-one", current_status: "Generating samples" };
    return { runs: [{
      run_id: "prod-one", current_stage: stage,
      stages: { [stage]: { status } },
    }] };
  });
  context.stageName = stage;
  vm.runInContext(`
    config = {stage_order: [stageName]};
    selected = {run_id: "prod-one", stages: {[stageName]: {status: "running"}}};
    renderSelected = () => schedulePoll();
  `, context);
  await vm.runInContext("refresh()", context);
  assert.deepEqual(calls, ["/api/runs", "/api/runs/prod-one/log"]);
  assert.equal(timers.at(-1).delay, 10000);
  status = "succeeded";
  await timers.at(-1).callback();
  assert.equal(calls.filter((path) => path.endsWith("/log")).length, 2);
  assert.equal(timers.length, 1);
});
}

test("late log responses cannot overwrite a newly selected run", async () => {
  let resolve;
  const { context } = uiContext(() => new Promise((done) => { resolve = done; }));
  vm.runInContext(`selected = {run_id: "prod-old"};`, context);
  const pending = vm.runInContext("refreshPreparationLog()", context);
  vm.runInContext(`selectionVersion += 1; selected = {run_id: "prod-new"};`, context);
  resolve({ run_id: "prod-old", current_status: "Old output" });
  await pending;
  assert.equal(vm.runInContext("preparationLog", context), undefined);
});

for (const status of ["running", "succeeded", "failed", "cancelled"]) {
  test(`page reload reconciles running training with Azure and handles ${status}`, async () => {
    const calls = [];
    const run = {
      run_id: "prod-training", run_type: "training", status: "running",
      current_stage: "training", stages: {training: {status: "running"}},
    };
    const latest = {
      ...run, status: status === "succeeded" ? "ready" : status,
      stages: {training: {status}},
    };
    const { context, timers } = uiContext(async (path) => {
      calls.push(path);
      if (path.endsWith("/log")) return {run_id: run.run_id, current_status: status};
      return path === "/api/runs" ? {runs: [run]} : latest;
    });
    vm.runInContext(`
      config = {stage_order: ["training"]};
      renderSelected = () => schedulePoll();
    `, context);
    await vm.runInContext("refresh()", context);
    assert.deepEqual(calls, ["/api/runs", "/api/runs/prod-training", "/api/runs/prod-training/log"]);
    assert.equal(vm.runInContext("selected.stages.training.status", context), status);
    assert.equal(vm.runInContext("runs[0].stages.training.status", context), status);
    assert.equal(timers.length, status === "running" ? 1 : 0);
  });
}

test("selecting a running job replaces stale history and resumes polling", async () => {
  const { context, timers } = uiContext(async (path) => path.endsWith("/log")
    ? {run_id: "prod-training", current_status: "Running"}
    : {run_id: "prod-training", run_type: "training", status: "running",
      current_stage: "training", stages: {training: {status: "running"}}});
  vm.runInContext(`
    runs = [{run_id: "prod-training", status: "running"}];
    renderSelected = () => schedulePoll();
  `, context);
  await vm.runInContext(`selectRun("prod-training")`, context);
  assert.equal(timers.at(-1).delay, 10000);
  assert.equal(vm.runInContext("runs[0] === selected", context), true);
});

test("an overlapping refresh does not stop polling a running job", async () => {
  const { context, timers } = uiContext();
  vm.runInContext(`
    selected = {status: "running", stages: {training: {status: "running"}}};
    refreshing = true;
  `, context);
  await vm.runInContext("refresh()", context);
  assert.equal(timers.at(-1).delay, 10000);
});

test("Azure errors on reload keep the running job rendered and polling retryable", async () => {
  const { context, timers } = uiContext(async (path) => {
    if (path === "/api/runs") return {runs: [{
      run_id: "prod-training", run_type: "training", status: "running",
      current_stage: "training", stages: {training: {status: "running"}},
    }]};
    throw new Error("Unable to fetch Azure ML status");
  });
  vm.runInContext(`
    config = {stage_order: ["training"]};
    renderSelected = () => schedulePoll();
  `, context);
  await assert.rejects(vm.runInContext("refresh()", context), /Unable to fetch Azure ML status/);
  assert.equal(vm.runInContext("selected.run_id", context), "prod-training");
  assert.equal(timers.at(-1).delay, 10000);
  assert.equal(vm.runInContext("refreshing", context), false);
});

test("cancel controls have space before the next stage card", () => {
  const styles = readFileSync(new URL("../static/styles.css", import.meta.url), "utf8");
  assert.match(styles, /#task-controls\s*\{[^}]*margin-bottom:\s*24px/);
});

test("log failures are displayed and can be retried", async () => {
  let fail = true;
  const { context, controls } = uiContext(async () => {
    if (fail) throw new Error("Cannot read preparation log");
    return { run_id: "prod-one", current_status: "Recovered" };
  });
  vm.runInContext(`selected = {run_id: "prod-one"};`, context);
  await vm.runInContext("refreshPreparationLog()", context);
  vm.runInContext("renderPreparationLog()", context);
  assert.equal(controls.get("preparation-status").textContent, "Cannot read preparation log");
  assert.equal(vm.runInContext("preparationLogError", context), "Cannot read preparation log");
  fail = false;
  await vm.runInContext("refreshPreparationLog()", context);
  vm.runInContext("renderPreparationLog()", context);
  assert.equal(controls.get("preparation-status").textContent, "Recovered");
  assert.equal(vm.runInContext("preparationLogError", context), undefined);
  assert.equal(vm.runInContext("preparationLog.current_status", context), "Recovered");
});

test("preparation status formats mixed terminal output into readable activity", () => {
  const { context } = uiContext();
  const cases = [
    [
      "Pages: 38%|\u2588\u2588\u2588\u258a | 8/21 [00:42<00:53, 4.10s/it]2026-10-21 20:01:02 [INFO] lib.pdf_to_chunks: Extracting page 9/21: page_0009.jpg",
      "Extracting page 9 of 21",
    ],
    ["Pages: 38%|\u2588\u2588\u2588\u258a | 8/21 [00:42<00:53, 4.10s/it]", "Extracted 8 of 21 pages"],
    ["Processing chunks: 20%|## | 2/10 [00:03<00:10, 1s/it]", "Processed 2 of 10 chunks"],
    ["2026-10-21 20:01:02 [INFO] lib.dataset_generation: Generated samples: 12/240", "Generated 12 of 240 samples"],
    ["2026-10-21 20:01:02 [INFO] lib.dataset_generation: Generating answer 2/5 for chunk 3", "Generating answer 2 of 5 for chunk 3"],
    ["2026-10-21T12:45:49.886089Z [STATUS] Data preparation succeeded: dataset validated and published.", "Data preparation succeeded: dataset validated and published."],
    ["2026-10-21 20:01:02 [WARNING] azure.identity: Token request failed", "Token request failed"],
    ["\u001b[32m2026-10-21 20:01:02 [INFO] lib.pdf_to_chunks: Saved extracted page 9/21\u001b[0m", "Saved extracted page 9 of 21"],
    ["Azure publication failed", "Azure publication failed"],
    ["2026-10-21T12:45:49.886089Z [STATUS] Interrupted: server stopped during preparation.", "Interrupted: server stopped during preparation."],
    ["Could not read /datasets/2026/10/file.jsonl", "Could not read /datasets/2026/10/file.jsonl"],
  ];
  for (const [input, expected] of cases) {
    context.inputStatus = input;
    assert.equal(vm.runInContext("preparationStatusText(inputStatus)", context), expected);
  }
});

test("status headline is formatted while recent job logs remain unchanged", () => {
  const { context, controls } = uiContext();
  const raw = "Pages: 38%|### | 8/21 [00:42<00:53, 4.10s/it]2026-10-21 20:01:02 [INFO] lib.pdf_to_chunks: Extracting page 9/21: page_0009.jpg";
  context.rawOutput = raw;
  vm.runInContext(`
    selected = {run_id: "prod-one", stages: {data_preparation: {status: "running"}}};
    preparationLog = {run_id: "prod-one", current_status: rawOutput, log: rawOutput,
                      updated_at: "2026-10-21T14:31:02Z"};
    renderPreparationLog();
  `, context);
  assert.equal(controls.get("preparation-status").textContent, "Extracting page 9 of 21");
  assert.equal(controls.get("preparation-log").textContent, raw);
  assert.match(controls.get("preparation-log-updated").textContent, /Checks every 10 seconds/);
});

test("sample table shows every generated field without truncating CoT or JSON context", () => {
  const { context, controls } = uiContext();
  const samples = [
    { split: "train", id: "sample-1", type: "oracle", question: "How much memory?",
      context: { sentences: [["H200 has 141 GB of HBM3e."]] },
      oracle_context: "H200 has 141 GB of HBM3e.",
      instruction: "<DOCUMENT>H200 has 141 GB of HBM3e.</DOCUMENT>\n### Question:\nHow much memory?",
      cot_answer: "The source states 141 GB.\n<ANSWER>141 GB</ANSWER>", source_page: 1 },
    { split: "validation", question: "Which precision?", cot_answer: "FP4" },
    { split: "test", question: "Why bandwidth?", cot_answer: "Repeated weight reads." },
    { question: "<script>alert('test')</script>", cot_answer: "<ANSWER>Literal text</ANSWER>" },
  ];
  context.samples = samples;
  vm.runInContext("renderDataSamples(samples)", context);
  const section = controls.get("data-summary").children[0];
  assert.equal(section.attributes["aria-labelledby"], "sample-data-heading");
  assert.equal(section.children[0].textContent, "Sample data");
  const wrapper = section.children[2];
  assert.equal(wrapper.tabIndex, 0);
  const table = wrapper.children[0];
  assert.equal(table.tagName, "table");
  const [, header, body] = table.children;
  assert.deepEqual(Array.from(header.children[0].children, (cell) => cell.textContent),
    ["Split", "ID", "Type", "Question", "Context", "Oracle context", "CoT answer", "Instruction", "source page"]);
  assert.ok(header.children[0].children.every((cell) => cell.attributes.scope === "col"));
  assert.equal(body.children.length, samples.length);
  for (const [index, row] of body.children.entries()) {
    for (const cell of row.children) {
      const key = cell.dataset.field;
      const value = key === "split" ? samples[index].split ?? "train" : samples[index][key];
      assert.equal(cell.textContent, value === undefined ? "Not in saved preview"
        : typeof value === "object" ? JSON.stringify(value, null, 2) : String(value));
    }
  }
});

test("runs without samples show a labelled empty state", () => {
  const { context, controls } = uiContext();
  vm.runInContext("renderDataSamples([])", context);
  const section = controls.get("data-summary").children[0];
  assert.equal(section.children[0].textContent, "Sample data");
  assert.equal(section.children[1].textContent, "No sample data is available for this run.");
});

test("published dataset picker excludes simulated, failed, RAG, and training runs", () => {
  const { context } = uiContext();
  vm.runInContext(`
    const published = {
      run_id: "prod-data", mode: "production", simulated: false, recipe: "context-grounded",
      stages: {data_preparation: {status: "succeeded"}},
      data_card: {asset_reference: "azureml:data:1"},
    };
    runs = [published, {...published, simulated: true}, {...published, run_type: "training"},
      {...published, run_type: "evaluation"},
      {...published, run_type: "release"},
      {...published, recipe: "context-assisted"},
      {...published, stages: {data_preparation: {status: "failed"}}},
      {...published, data_card: null}];
  `, context);
  assert.equal(vm.runInContext("availableDatasets().length", context), 1);
  assert.equal(vm.runInContext("availableDatasets()[0].run_id", context), "prod-data");
});

test("manual RAFT selection shows the full contract; RAG training stays disabled", () => {
  const { context, controls } = uiContext();
  vm.runInContext(`
    const form = $("training-form");
    field(form, "dataset_source").value = "manual";
    field(form, "dataset_recipe").value = "context-grounded";
    updateDatasetControls();
    showDatasetFormat();
  `, context);
  assert.equal(controls.get("dataset-format-dialog").open, true);
  assert.match(controls.get("dataset-format-description").textContent, /azureml:dataset-name:version/);
  assert.deepEqual(Object.keys(JSON.parse(controls.get("dataset-format-example").textContent)).sort(),
    ["id", "type", "question", "context", "oracle_context", "instruction", "cot_answer"].sort());
  assert.equal(controls.get("train").disabled, false);
  vm.runInContext(`
    field($("training-form"), "dataset_recipe").value = "context-assisted";
    updateDatasetControls();
    showDatasetFormat();
  `, context);
  assert.equal(controls.get("train").disabled, true);
  assert.match(controls.get("dataset-format-description").textContent, /RAG training is disabled/);
});

test("training payload forwards every trainer control and exactly one dataset selector", () => {
  const { context } = uiContext();
  vm.runInContext(`
    const form = $("training-form");
    for (const [key, value] of Object.entries({
      production_base_model: "Llama-3.2-3B-Instruct",
      max_seq_length: "2048", batch_size: "2", gradient_accumulation: "16",
      learning_rate: "0.00002", epochs: "0.5", lora_rank: "16",
      seed: "42", chat_template: "llama-3.2", lora_adapter_name: "adapter",
      dataset_source: "manual", dataset_id: " azureml:my-data:1 ",
      dataset_recipe: "context-grounded", dataset_run_id: "prod-data",
    })) field(form, key).value = value;
    field(form, "confirmed").checked = true;
    field(form, "load_in_4bit").checked = true;
  `, context);
  const body = JSON.parse(vm.runInContext("JSON.stringify(trainingBody(form))", context));
  assert.deepEqual(body, {
    confirmed: true, base_model: "Llama-3.2-3B-Instruct", max_seq_length: 2048,
    batch_size: 2, gradient_accumulation: 16, learning_rate: .00002, epochs: .5,
    lora_rank: 16, seed: 42, chat_template: "llama-3.2", lora_adapter_name: "adapter",
    load_in_4bit: true, dataset_id: "azureml:my-data:1", dataset_recipe: "context-grounded",
  });
  vm.runInContext(`field(form, "dataset_source").value = "published";`, context);
  const published = JSON.parse(vm.runInContext("JSON.stringify(trainingBody(form))", context));
  assert.equal(published.dataset_run_id, "prod-data");
  assert.equal("dataset_id" in published, false);
});

test("production fine-tuning is accessible without preparation and uses training log headings", () => {
  const { context, controls } = uiContext();
  vm.runInContext(`
    config = {stage_order: ["data_preparation", "training", "evaluation", "decision", "deployment"]};
  `, context);
  assert.equal(vm.runInContext(`canOpenStage("training")`, context), true);
  assert.equal(vm.runInContext(`canOpenStage("evaluation")`, context), true);
  assert.equal(vm.runInContext(`canOpenStage("decision")`, context), false);
  vm.runInContext(`
    selected = {run_id: "prod-training", run_type: "training", stages: {training: {status: "running"}}};
    renderPreparationLog();
    schedulePoll();
  `, context);
  assert.equal(controls.get("job-log-title").textContent, "Fine-tuning job status");
});

test("stage navigation works before the first run without a redundant fine-tuning button", () => {
  const { context, controls } = uiContext();
  vm.runInContext(`
    config = {stage_order: ["data_preparation", "training", "evaluation", "decision", "deployment"]};
    renderStageNavigation();
  `, context);
  assert.deepEqual(Array.from(controls.get("stage-nav").children, (button) => button.disabled),
    [false, false, false, true]);
  const html = readFileSync(new URL("../static/index.html", import.meta.url), "utf8");
  assert.equal(html.includes('id="configure-training"'), false);
  assert.equal(source.includes('$("configure-training")'), false);
});

test("production has four tabs and keeps deployment within candidate release", () => {
  const {context, controls} = uiContext();
  vm.runInContext(`
    config = {
      stage_order: ["data_preparation", "training", "evaluation", "decision", "deployment"],
      enabled_stages: ["data_preparation", "training", "evaluation", "decision", "deployment"],
    };
    selected = {
      status: "running", current_stage: "deployment",
      stages: Object.fromEntries(config.stage_order.map(name => [name, {
        status: name === "deployment" ? "running" : "succeeded",
      }])),
    };
    renderStageNavigation();
    setTab(selected.current_stage);
  `, context);
  const tabs = controls.get("stage-nav").children;
  assert.deepEqual(Array.from(tabs, (button) => button.dataset.stage),
    ["data_preparation", "training", "evaluation", "decision"]);
  assert.equal(vm.runInContext("tab", context), "decision");
  assert.equal(controls.get("panel-decision").hidden, false);
  assert.equal(controls.get("wizard-position").textContent, "Step 4 of 4");
  assert.equal(controls.get("wizard-next").disabled, true);
  assert.equal(tabs[3].children[1].textContent, "running");
  assert.equal(vm.runInContext("canOpenStage('deployment')", context), false);
  vm.runInContext(`
    selected.stages.deployment.status = "failed";
    renderStageNavigation();
  `, context);
  assert.equal(controls.get("stage-nav").children[3].children[1].textContent, "failed");
});

test("deployment controls and endpoint results are contained in the fourth panel", () => {
  const html = readFileSync(new URL("../static/index.html", import.meta.url), "utf8");
  assert.equal(html.includes('id="panel-deployment"'), false);
  assert.equal(html.includes("STAGE 05"), false);
  const releasePanel = html.slice(html.indexOf('id="panel-decision"')).split("</section>")[0];
  assert.match(releasePanel, /id="deployment-summary"/);
  assert.match(releasePanel, /id="release-approval-form"/);
  assert.equal(releasePanel.includes('id="deployment-form"'), false);
  assert.equal(releasePanel.includes("<h3>Endpoint and monitoring</h3>"), false);
  assert.equal(source.includes("Choose Register and deploy in Candidate release to review SKU"), false);
  const styles = readFileSync(new URL("../static/styles.css", import.meta.url), "utf8");
  assert.match(styles, /\.stage-nav\s*\{[^}]*grid-template-columns: repeat\(4, minmax\(0, 1fr\)\)/);
  assert.match(styles, /table\s*\{[^}]*width: 100%/);
});

test("evaluation job selection displays every requested training parameter", () => {
  const { context, controls } = uiContext();
  vm.runInContext(`
    evaluationJobs = [{
      job_name: "trained-job", eligible: true, training_date: "2026-10-21T12:00:00Z",
      base_model: "Llama-3.2-3B-Instruct", dataset_version: "data-211026150828",
      sample_count: 40, epochs: 2, adapter_name: "test-adapter", baseline_id: "baseline:1",
    }];
    $("evaluation-job").value = "trained-job";
    renderEvaluationJobs();
  `, context);
  const card = controls.get("evaluation-job-summary").children[0];
  const values = Array.from(card.children[1].children, (node) => node.textContent);
  for (const expected of [
    "trained-job", "2026-10-21T12:00:00Z", "Llama-3.2-3B-Instruct",
    "data-211026150828", "40", "2", "test-adapter", "Reuse baseline:1",
  ]) assert.ok(values.includes(expected), expected);
  assert.equal(controls.get("evaluate").disabled, false);
  vm.runInContext(`runs = [{status: "running"}]; renderEvaluationJob();`, context);
  assert.equal(controls.get("evaluate").disabled, true);
});

test("metrics table shows all models, units, and unmeasured token cost honestly", () => {
  const { context, controls } = uiContext();
  vm.runInContext(`
    renderEvaluationMetrics({metrics: [
      {label: "Faithfulness", unit: "score", fine_tuned: .9, oob: .7, gpt: .95},
      {label: "Mean latency", unit: "ms", fine_tuned: 123, oob: 150, gpt: 200},
      {label: "Token cost (not recorded)", unit: "USD", fine_tuned: null, oob: null, gpt: null},
    ]});
  `, context);
  const table = controls.get("evaluation-metrics").children[1];
  assert.deepEqual(Array.from(table.children[1].children[0].children, (cell) => cell.textContent),
    ["Metric", "Fine-tuned SLM", "Original SLM", "GPT baseline"]);
  const rows = table.children[2].children;
  assert.equal(rows[0].children[1].textContent, "90.0%");
  assert.equal(rows[1].children[1].textContent, "123.0 ms");
  assert.equal(rows[2].children[1].textContent, "Not recorded");
});

test("evaluation log title and HTTP adapter use production without a mode argument", async () => {
  const { context, controls, timers } = uiContext();
  vm.runInContext(`
    selected = {run_id: "prod-eval", run_type: "evaluation", stages: {evaluation: {status: "running"}}};
    renderPreparationLog(); schedulePoll();
  `, context);
  assert.equal(controls.get("job-log-title").textContent, "Evaluation job status");
  assert.equal(timers.at(-1).delay, 10000);
  const urls = [];
  const apiSource = readFileSync(new URL("../static/api.js", import.meta.url), "utf8")
    .replace("export async function", "async function");
  const apiContext = vm.createContext({
    fetch: async (url) => { urls.push(url); return {ok: true, json: async () => ({})}; },
  });
  vm.runInContext(apiSource, apiContext);
  await vm.runInContext(`api("/api/evaluation", {confirmed: true})`, apiContext);
  await vm.runInContext(`api("/api/training-jobs")`, apiContext);
  await vm.runInContext(`api("/api/production/runs/prod-one")`, apiContext);
  assert.deepEqual(urls, ["/api/production/evaluation", "/api/production/training-jobs", "/api/production/runs/prod-one"]);
});

test("Azure portal links use the theme and retain visible keyboard focus styling", () => {
  assert.match(source, /element\("a", "Open training job in Azure ML", "action-link"\)/);
  assert.match(source, /element\("a", "Open evaluation job in Azure ML", "action-link"\)/);
  const styles = readFileSync(new URL("../static/styles.css", import.meta.url), "utf8");
  assert.match(styles, /\.action-link:hover/);
  assert.match(styles, /a:focus-visible/);
  assert.match(styles, /\.action-link\s*\{[^}]*text-decoration: none/);
});

test("production evaluation without a release gate renders and continues polling", () => {
  const { context, timers, controls } = uiContext();
  vm.runInContext(`
    config = {models: [], stage_order: ["data_preparation", "training", "evaluation", "decision", "deployment"]};
    selected = {
      run_id: "prod-eval", run_type: "evaluation", status: "running", current_stage: "evaluation",
      stages: Object.fromEntries(config.stage_order.map(name => [name, {
        status: name === "evaluation" ? "running" : "not_started", progress: 0, error: null,
      }])),
      training: {config: {base_model: "Llama-3.2-3B-Instruct"}},
      evaluation: {candidate_id: "trained-job", job_status: "Running"},
      decisions: {registration: null},
    };
    runs = [selected];
    renderSelected();
  `, context);
  assert.equal(timers.at(-1).delay, 10000);
  assert.equal(controls.has("override-fields"), false);
  assert.equal(controls.get("evaluation-showcase").hidden, true);
});

test("all production release routes are isolated and readiness errors keep their reasons", async () => {
  const calls = [];
  const context = vm.createContext({
    fetch: async (path) => {
      calls.push(path);
      return {ok: true, json: async () => ({})};
    },
  });
  const source = readFileSync(new URL("../static/api.js", import.meta.url), "utf8")
    .replace("export async function api", "async function api");
  vm.runInContext(source, context);
  for (const path of ["/api/release-evaluations", "/api/release-plans", "/api/release-plans/prod-release/approve"]) {
    context.path = path;
    await vm.runInContext("api(path)", context);
  }
  assert.deepEqual(calls, ["/api/production/release-evaluations", "/api/production/release-plans", "/api/production/release-plans/prod-release/approve"]);
  context.fetch = async () => ({ok: false, status: 409, json: async () => ({
    detail: {message: "Scientific policy blocked.", readiness: {reasons: ["At least 30 examples."]}},
  })});
  await assert.rejects(vm.runInContext("api('/api/release-plans', {})", context), /At least 30 examples/);
});

test("release readiness shows quality policy, confidence intervals and complete lineage", () => {
  const {context, controls} = uiContext();
  vm.runInContext(`
    selected = {stages: {}, release: {status: "awaiting_approval", source_run_id: "prod-eval"}};
    releaseEvaluations = [{
      run_id: "prod-eval", lineage: {preparation_run: "prod-preparation", evaluation_job: "eval-job", training_job: "train-job", training_dataset: "azureml:data:1", evaluation_dataset: "azureml:data:1"},
      readiness: {status: "ready", passed: true, sample_count: 30, reasons: [], policy: {
        version: "paired-quality-v1", metrics: ["faithfulness", "relevancy", "correctness"],
        minimum_examples: 30, minimum_absolute_uplift: .02,
        maximum_relative_gpt_shortfall_exclusive: .05, confidence: .95,
        bootstrap_replicates: 5000, bootstrap_seed: 42,
      }, metrics: [{
        metric: "correctness", fine_tuned: .8, oob: .7, gpt: .82, absolute_uplift: .1,
        relative_gpt_shortfall: .02439, uplift_ci95: [.05, .15], passed: true,
        checks: {uplift: true, gpt_gap: true, positive_confidence_interval: true},
      }]},
    }];
    $("release-evaluation").value = "prod-eval";
    renderReleaseReadiness();
  `, context);
  const text = (node) => [node.textContent ?? "", ...node.children.map(text)].join(" ");
  const readiness = text(controls.get("release-readiness"));
  assert.match(readiness, /5,000 resamples; seed 42/);
  assert.match(readiness, /No edits or overrides/);
  assert.match(readiness, /At least 2 percentage points/);
  assert.match(readiness, /Less than 5% below GPT/);
  assert.match(readiness, /5.00 to 15.00/);
  assert.match(readiness, /azureml:data:1/);
  const descendants = (node) => [node, ...node.children.flatMap(descendants)];
  const nodes = descendants(controls.get("release-readiness"));
  assert.deepEqual(nodes.filter((node) => node.tagName === "caption").map((node) => node.textContent),
    ["Release policy", "Quality metric results", "1. Data preparation", "2. Training", "3. Evaluation"]);
  assert.equal(nodes.filter((node) => node.tagName === "dl" || node.tagName === "details").length, 0);
  const tables = nodes.filter((node) => node.tagName === "table");
  assert.match(text(tables[2]), /prod-preparation/);
  assert.match(text(tables[3]), /train-job/);
  assert.match(text(tables[3]), /Not recorded/);
  assert.match(text(tables[4]), /eval-job/);
  assert.equal(controls.get("plan-release").disabled, false);
  vm.runInContext(`
    releaseEvaluations[0].readiness.policy.minimum_examples = 40;
    renderReleaseReadiness();
  `, context);
  assert.match(text(controls.get("release-readiness")), /At least 40 examples/);
  vm.runInContext(`
    delete releaseEvaluations[0].readiness.policy;
    renderReleaseReadiness();
  `, context);
  assert.equal(controls.get("plan-release").disabled, true);
  assert.match(text(controls.get("release-readiness")), /Policy details are not recorded/);
});

test("readiness uses labeled green/red badges for sample size and each metric gate", () => {
  const {context, controls} = uiContext();
  vm.runInContext(`
    selected = {stages: {}};
    releaseEvaluations = [{
      run_id: "prod-two", lineage: {}, readiness: {
        status: "blocked", passed: false, sample_count: 2,
        policy: {version: "paired-quality-v1",
          metrics: ["faithfulness", "relevancy", "correctness"],
          minimum_examples: 30, minimum_absolute_uplift: .02,
          maximum_relative_gpt_shortfall_exclusive: .05, confidence: .95,
          bootstrap_replicates: 5000, bootstrap_seed: 42},
        reasons: ["At least 30 sealed-test examples required; observed 2.",
          "faithfulness: failed uplift, positive_confidence_interval.",
          "relevancy: failed positive_confidence_interval.",
          "correctness: failed uplift, positive_confidence_interval."],
        metrics: ["faithfulness", "relevancy", "correctness"].map(metric => ({
          metric, fine_tuned: .8, oob: .79, gpt: .82, absolute_uplift: .01,
          relative_gpt_shortfall: .02439, uplift_ci95: [-.01, .03], passed: false,
          checks: {uplift: metric === "relevancy", gpt_gap: true,
            positive_confidence_interval: false},
        })),
      },
    }];
    $("release-evaluation").value = "prod-two";
    renderReleaseReadiness();
  `, context);
  const descendants = (node) => [node, ...node.children.flatMap(descendants)];
  const text = (node) => [node.textContent ?? "", ...node.children.map(text)].join(" ");
  const nodes = descendants(controls.get("release-readiness"));
  const badges = nodes.filter(node => node.className?.startsWith("gate-badge"));
  assert.equal(badges.filter(node => node.className === "gate-badge pass").length, 4);
  assert.equal(badges.filter(node => node.className === "gate-badge fail").length, 14);
  assert.equal(badges.filter(node => node.className === "gate-badge unknown").length, 0);
  assert.match(text(controls.get("release-readiness")), /2 observed; at least 30 required/);
  assert.match(text(controls.get("release-readiness")), /positive 95% uplift confidence interval/);
  assert.doesNotMatch(text(controls.get("release-readiness")), /positive_confidence_interval/);
  assert.equal(controls.get("plan-release").disabled, true);
  const styles = readFileSync(new URL("../static/styles.css", import.meta.url), "utf8");
  assert.match(styles, /\.gate-badge\.pass\s*\{[^}]*color: #14532d; background: #dcfce7/);
  assert.match(styles, /\.gate-badge\.fail\s*\{[^}]*color: #991b1b; background: #fee2e2/);
  vm.runInContext(`
    delete releaseEvaluations[0].readiness.metrics[0].checks.uplift;
    renderReleaseReadiness();
  `, context);
  assert.equal(descendants(controls.get("release-readiness"))
    .filter(node => node.className === "gate-badge unknown").length, 1);
});

test("UI contains no mode selector, simulated forms, or saved demo evidence loader", () => {
  const html = readFileSync(new URL("../static/index.html", import.meta.url), "utf8");
  assert.doesNotMatch(html, /id="(?:mode|demo-options|register-form|deployment-form|suggest)"/);
  assert.doesNotMatch(source, /studio-mode|localStorage|demo:\/\/|\/api\/evaluation-showcase|traffic_percent/);
  for (const id of ["production-options", "production-dataset-options", "production-trainer-options",
    "production-evaluation-options", "production-release-options"]) {
    assert.doesNotMatch(html, new RegExp(`id="${id}"[^>]*hidden`));
  }
});

test("data card distinguishes release sample sufficiency from successful preparation", () => {
  const {context, controls} = uiContext();
  vm.runInContext(`
    config = {models: [], stage_order: ["data_preparation", "training", "evaluation", "decision", "deployment"]};
    selected = {
      run_id: "prod-data", status: "completed", current_stage: "data_preparation",
      stages: Object.fromEntries(config.stage_order.map(name => [name, {
        status: name === "data_preparation" ? "succeeded" : "not_started", progress: 0, error: null,
      }])),
      data_card: {recipe: "context-grounded", record_count: 300, samples: [],
        validation: {status: "passed", checked_records: 300, invalid_records: 0},
        release_sample_check: {minimum_test_examples: 30, observed_test_examples: 29, passed: false}},
    };
    renderSelected();
  `, context);
  const text = node => [node.textContent ?? "", ...node.children.map(text)].join(" ");
  assert.match(text(controls.get("data-summary")), /FAIL.*29 sealed-test examples; at least 30 required/);
  assert.match(text(controls.get("data-summary")), /passed: 300 records checked/);
  vm.runInContext(`
    selected.data_card.release_sample_check = {
      minimum_test_examples: 30, observed_test_examples: 30, passed: true,
    };
    renderSelected();
  `, context);
  assert.match(text(controls.get("data-summary")), /PASS.*30 sealed-test examples/);
  assert.match(text(controls.get("data-summary")), /quality gates still apply/);
});

test("release readiness is separated from the evaluation dropdown", () => {
  const styles = readFileSync(new URL("../static/styles.css", import.meta.url), "utf8");
  assert.match(styles, /#release-readiness\s*\{[^}]*margin-top: 20px/);
});

test("perfect-baseline metric explains waived gates without claiming positive uplift", () => {
  const {context, controls} = uiContext();
  vm.runInContext(`
    selected = {stages: {}};
    releaseEvaluations = [{run_id: "prod-perfect", lineage: {}, readiness: {
      status: "ready", passed: true, sample_count: 30, reasons: [],
      policy: {version: "paired-quality-v2", metrics: ["faithfulness"],
        minimum_examples: 30, minimum_absolute_uplift: .02,
        maximum_relative_gpt_shortfall_exclusive: .05, confidence: .95,
        bootstrap_replicates: 5000, bootstrap_seed: 42,
        perfect_baseline_exception: {baseline_score: 1, candidate_score: 1,
          waived_checks: ["uplift", "positive_confidence_interval"]}},
      metrics: [{metric: "faithfulness", fine_tuned: 1, oob: 1, gpt: 1,
        absolute_uplift: 0, relative_gpt_shortfall: 0, uplift_ci95: [0, 0],
        baseline_at_ceiling: true, ceiling_exception_applied: true, passed: true,
        checks: {uplift: true, gpt_gap: true, positive_confidence_interval: true}}],
    }}];
    $("release-evaluation").value = "prod-perfect";
    renderReleaseReadiness();
  `, context);
  const descendants = node => [node, ...node.children.flatMap(descendants)];
  const text = node => [node.textContent ?? "", ...node.children.map(text)].join(" ");
  const nodes = descendants(controls.get("release-readiness"));
  const badges = nodes.filter(node => node.textContent === "PASS (PERFECT TIE)");
  assert.equal(badges.length, 2);
  assert.ok(badges.every(node => node.className === "gate-badge pass"));
  assert.match(text(controls.get("release-readiness")), /Sample count and GPT-gap checks still apply/);
  const quality = nodes.find(node => node.tagName === "table"
    && node.children.some(child => child.tagName === "caption" && child.textContent === "Quality metric results"));
  const row = quality.children.find(node => node.tagName === "tbody").children[0];
  assert.match(text(row.children[7]), /Required: remain exactly 100% on every example/);
  assert.doesNotMatch(text(row.children[7]), /Required: at least 2 points/);
  assert.match(text(row.children[9]), /not evidence of improvement/);
  assert.equal(controls.get("plan-release").disabled, false);
  vm.runInContext(`
    const metric = releaseEvaluations[0].readiness.metrics[0];
    metric.fine_tuned = .99;
    metric.ceiling_exception_applied = false;
    metric.checks.uplift = metric.checks.positive_confidence_interval = metric.passed = false;
    releaseEvaluations[0].readiness.passed = false;
    releaseEvaluations[0].readiness.status = "blocked";
    renderReleaseReadiness();
  `, context);
  assert.equal(descendants(controls.get("release-readiness"))
    .filter(node => node.textContent === "PASS (PERFECT TIE)").length, 0);
  assert.equal(controls.get("plan-release").disabled, true);
});
