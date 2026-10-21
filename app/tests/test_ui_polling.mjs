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
  const context = vm.createContext({
    localStorage: { getItem: () => "production" },
    document: {
      getElementById: (id) => {
        if (!controls.has(id)) controls.set(id, { addEventListener() {} });
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

test("each production refresh reads the selected run log, including final status", async () => {
  const calls = [];
  let status = "running";
  const { context, timers } = uiContext(async (path) => {
    calls.push(path);
    if (path.endsWith("/log")) return { run_id: "prod-one", current_status: "Generating samples" };
    return { runs: [{
      run_id: "prod-one", current_stage: "data_preparation",
      stages: { data_preparation: { status } },
    }] };
  });
  vm.runInContext(`
    config = {stage_order: ["data_preparation"]};
    selected = {run_id: "prod-one", stages: {data_preparation: {status: "running"}}};
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
