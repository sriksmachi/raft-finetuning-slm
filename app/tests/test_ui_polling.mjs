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
      addEventListener() {},
      showModal() { this.open = true; },
      append(...nodes) { this.children.push(...nodes); },
      replaceChildren(...nodes) { this.children = nodes; },
      setAttribute(name, value) { this.attributes[name] = value; },
      elements: { namedItem(name) {
        if (!fields.has(name)) fields.set(name, createElement("input"));
        return fields.get(name);
      } },
    };
  };
  const context = vm.createContext({
    localStorage: { getItem: () => "production" },
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

test("production polls after exactly 10 seconds; demo keeps its existing interval", async () => {
  const { context, timers } = uiContext();
  vm.runInContext(`
    selected = {stages: {data_preparation: {status: "running"}}};
    schedulePoll();
  `, context);
  assert.equal(timers.at(-1).delay, 10000);
  vm.runInContext(`mode = "demo"; schedulePoll();`, context);
  assert.equal(timers.at(-1).delay, 1200);
  vm.runInContext(`selected.stages.data_preparation.status = "succeeded"; schedulePoll();`, context);
  assert.equal(timers.length, 2);
  vm.runInContext(`
    mode = "production";
    selected.stages.data_preparation.status = "running";
    refresh = async () => { throw new Error("Temporary connection error"); };
    schedulePoll();
  `, context);
  await timers.at(-1).callback();
  assert.equal(timers.at(-1).delay, 10000);
});

for (const stage of ["data_preparation", "training"]) {
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
      production_base_model: "Llama-3.2-3B-Instruct", base_model: "demo-phi-3-mini",
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
  vm.runInContext(`mode = "demo";`, context);
  const demo = JSON.parse(vm.runInContext("JSON.stringify(trainingBody(form))", context));
  assert.equal(demo.base_model, "demo-phi-3-mini");
  assert.equal("dataset_id" in demo, false);
  assert.equal("seed" in demo, false);
});

test("production fine-tuning is accessible without preparation and uses training log headings", () => {
  const { context, controls } = uiContext();
  vm.runInContext(`
    config = {stage_order: ["data_preparation", "training", "evaluation", "decision", "deployment"]};
  `, context);
  assert.equal(vm.runInContext(`canOpenStage("training")`, context), true);
  assert.equal(vm.runInContext(`canOpenStage("evaluation")`, context), false);
  vm.runInContext(`
    selected = {run_id: "prod-training", run_type: "training", stages: {training: {status: "running"}}};
    renderPreparationLog();
    schedulePoll();
  `, context);
  assert.equal(controls.get("job-log-title").textContent, "Fine-tuning job status");
});
