# SLM Studio - GitHub Copilot Product Spec

**Status:** Production-only preparation, training, evaluation, scientific candidate release, and approved endpoint deployment implemented in four tabs. Demo UI, simulator APIs/code, seeded runs and saved demo evidence removed. Cloud paths are mock-tested; live billable deployment validation remains pending.
**Purpose:** A living product spec for incremental implementation with GitHub Copilot (GHCP).
**Scope:** Product requirements and acceptance criteria only; this document does not authorize implementation.
**Inspiration:** [ideaweaver-ai/ideaweaver-slm-builder](https://github.com/ideaweaver-ai/ideaweaver-slm-builder). SLM Studio deliberately avoids rebuilding Azure dashboards.

IdeaWeaver's from-scratch TinyStories model is a UX reference, not a compatible
training recipe for this pretrained Llama RAFT pipeline. SLM Studio preserves
its 4-bit rsLoRA architecture, pretrained tokenizer and chat template,
response-only grounded supervision, validation, and reproducible seed. It
adopts only compatible presentation concepts such as hardware-aware precision
guidance and explicit microbatch/effective-token accounting.

## 1. Goals and boundaries

SLM Studio is a small, locally hosted UI over this repository's SLM fine-tuning workflow. It helps users prepare data, configure training, compare models, and make an evidence-based registration and deployment decision without memorizing CLI arguments.

Core objectives:
- Support four tabs: data preparation, fine-tuning, evaluation, and candidate release with integrated deployment/monitoring.
- Execute real Azure-connected operations only; do not offer simulation or seed fabricated evidence.
- Reuse the existing pipeline rather than create a second training or evaluation implementation.
- Show concise decision evidence, including the GPT-baseline gate and actual metric values.
- Require explicit approval of the fixed release policy, lineage, and planned deployment cost; production gates cannot be overridden.
- Use Azure ML Studio and Azure AI Foundry for detailed logs, loss curves, charts, and monitoring.

Non-goals:
- Training dashboards, live loss charts, or a model-comparison playground. Lightweight job-status logs are supported.
- A hosted, public, or multi-user service in the initial release.
- Autonomous deployment based only on an LLM recommendation.
- Replacing Azure ML or implementing a new fine-tuning engine.

The five internal stage states are retained for persisted workflow compatibility; deployment is part of the fourth UI tab. "Production" means real pipeline execution, not public hosting.

## 2. Functional features

The state below describes the Studio UI/API, not the availability of standalone
pipeline scripts. Production means real pipeline execution, not public hosting.

| ID | Feature | State | Required behavior / acceptance |
|---|---|---|---|
| F-01 | Production-only workflow | Implemented and integration-tested | No mode selector, demo API routes, simulator code, seeded demo runs or saved demo evidence. Production APIs remain segregated in dedicated modules under `/api/production/`. Preserve existing `runs/production/` records and Azure job ledgers. An empty workspace stays empty. Four equal-width tabs fill the navigation row; deployment is integrated into Candidate release, without duplicate shortcuts. |
| F-02 | Data preparation | Context-grounded production flow implemented and integration-tested; context-assisted pending definition | Select a server-accessible PDF folder, recipe, and exact total number of samples across all splits. Preserve questions-per-call and oracle-probability controls. Confirm billable execution, then run PDF extraction, dataset generation/validation, and versioned Azure ML data-asset publication asynchronously. Persist a data card under a unique Studio run ID and show split counts, fingerprint, validation, and asset reference. Show all generated sample fields in a labelled, scrollable table: ID, type, question, context, oracle context, CoT answer, and instruction, plus split. Recover older limited previews from local JSONL files when available. Short generation and pipeline errors fail visibly; interrupted stages retain failed state. The context-assisted recipe remains disabled in production until defined. Tests isolate cloud execution; a live billable Azure smoke test is not yet performed. |
| F-03 | Training configuration | Production trainer controls implemented and integration-tested; GPT suggestions planned | Select a Foundry base model and edit all supported trainer flags: sequence-length fallback, learning rate, epochs, batch size, gradient accumulation, LoRA rank/adapter name, chat template, seed, and 4-bit loading. Choose a successful published preparation run or enter a pinned `azureml:name:version` asset. Show the RAFT JSONL contract in a popup; RAG guidance is available but RAG training remains disabled. Infrastructure remains backend-configured. GPT-assisted parameter suggestions are not implemented; no nonfunctional suggestion button is shown. |
| F-04 | Training submission | Production implemented and integration-tested without billable execution | Confirm dataset, trainer settings, and GPU charges before asynchronous Stage 02 submission. Preserve preparation lineage, trainer configuration, dataset binding, Azure job identity/link and artifact reference. Write/poll status logs every 10 seconds. Failures remain visible; interrupted monitoring warns remote jobs may continue. Azure ML links use the existing theme and visible keyboard focus. |
| F-05 | Model evaluation | Production quality, token usage and model-specific frontier pricing implemented; cloud calls mocked | Choose a saved completed training job and review date, model, dataset version, sample count, epochs and adapter. Reuse a recorded GPT baseline or generate one. Evaluate all three models on the same sealed test with the backend judge. Validate identities, hashes, judge and finite complete scores. Display quality, latency and measured input/output/total tokens. Root `token_prices.json` contains only explicit frontier API model IDs (for example GPT-4.1, GPT-4.1-mini, GPT-5.2), never local SLM tariffs. Match actual API model identities; persist the price snapshot before evaluation. Unknown identities, rates or usage remain unavailable, not zero. Report judge and reused historical baseline usage separately. GPU costs remain hourly and distinct. Search/filter responses and poll logs every 10 seconds. |
| F-06 | Scientific readiness | Fixed policy v2, perfect-baseline exception and colored gate results implemented and tested | Select recent completed evaluations. Independently require faithfulness, relevancy and correctness uplift >=0.02 over OOB, GPT relative shortfall strictly <5%, paired-bootstrap 95% uplift lower bound >0 and at least 30 sealed examples. For an exactly 100% original-SLM metric, waive only uplift and positive-confidence checks if the candidate remains exactly 100% on every example; reject any regression and retain other gates. Use 5,000 paired percentile resamples, seed 42. Show a readable policy table and separate preparation, training and evaluation lineage tables. Green PASS/red FAIL badges identify each gate; label ceiling exceptions PASS (PERFECT TIE), and missing results gray NOT RECORDED. Show observed values, requirements and readable failures. Labels supplement color for accessibility. Latency and token costs are informational. No edits or overrides; GPT cannot alter readiness. |
| F-07 | Candidate registration | Register-only production path implemented and mock-tested | Revalidate cloud training/evaluation identity, pinned dataset and sealed artifact hashes at planning and approval. Record plan hash, actor, reason and time. Register the adapter/tokenizer only, with no environment, endpoint or GPU allocation. Reuse only an existing model version with matching source/evidence tags. Persist registered identity before any deployment; failures and interruption warnings remain visible. Stage 04 CLI now registers only. |
| F-08 | Endpoint deployment and monitoring | Approved production workflow implemented and mock-tested; live billable validation pending | Choose register-and-deploy. GPT selects from supported regional, memory-safe GPU SKUs using full fp16 base size, LoRA technique and bounded request tokens; Azure Retail Prices supplies the regional Linux PAYG hourly estimate. Display SKU, rationale, price/source, expiry and quota caveat before explicit cost approval. Deploy a new dedicated authenticated AML endpoint with scoring collection; preserve existing endpoints/routing. Require an existing CPU cluster and governed normalized production request stream; reject training/test data and malformed monitoring input. Enable weekly request-feature PSI monitoring, validate registered deployment identity and default endpoint scoring, and share the URI without credentials. No canary/rollout controls; Stage 05 CLI uses a hash-approved plan. Poll logs every 10 seconds; retain model/resource identity on partial failure and warn of continuing costs. |

Implementation: [production API](../app/production_api.py),
[pipeline adapter](../app/preparation.py), and
[fast API integration tests](../app/tests/test_production_api.py).
Run `python -m pytest app\tests\test_production_api.py -q` without cloud calls.

## 3. Technical features

| ID | Requirement | Required behavior / acceptance |
|---|---|---|
| T-01 | Reuse pipeline logic | Backend adapters reuse [Azure ML helpers](../lib/azureml_ops.py), [data preparation](../scripts/01_prepare_data.py), [training orchestration](../scripts/02_azureml_fine_tuning.py), and [evaluation orchestration](../scripts/03_azureml_slm_vs_gpt_evaluation.py). Shared logic may be extracted when implemented; do not duplicate it or assume every helper already lives in `lib/`. |
| T-02 | Local credentials | Run locally using the operator's Azure CLI credentials on the backend. The browser never receives Azure credentials, Hugging Face tokens, or Key Vault secrets. Bind locally by default; public exposure and multi-user authentication are out of scope. |
| T-03 | Long-running operations | Submit work and return identifiers without holding an HTTP request open for a full Azure job. Support manual refresh or lightweight stage-status refresh; no live telemetry streaming. Surface errors explicitly and retain stage outcomes across restart. |
| T-04 | Production integrity | Only production routes and non-simulated records are accepted. No demo adapters, route fallback, automatic seeds or synthesized release evidence. Preserve atomic writes, safe IDs, explicit corruption errors and compatibility with historical production records. |
| T-05 | Safe actions | Confirm billable operations and exact deployment plan/cost. Block expired or changed plans, policy overrides and automatic retries after partial failures. Persist approvals and resource identities without secrets. |
| T-06 | Testable behavior | Validate removed routes return 404, empty stores remain empty, run selection, stage transitions, candidate-bound evaluation, missing/failed evidence, colored gate outcomes, confirmations, atomic-write failure, and restart recovery. Mock billable execution in tests, not in the product. |
| T-07 | Cohesive library boundaries | Implemented: [release](../lib/release.py) owns scientific policy, bound evidence and receipts; [Azure operations](../lib/azureml_ops.py) owns jobs, status polling, model registration, dedicated deployment and endpoint invocation; [monitoring](../lib/monitoring.py) owns drift profiles, URI validation and schedules; [model artifacts](../lib/lora_artifacts.py) shares base-checkpoint and adapter resolution. Keep cloud-invoked training, inference, scoring and drift-job entry points separate. Remove obsolete latency/token-F1 gates, traffic promotion and duplicated evidence helpers; update all callers rather than retain unused compatibility wrappers. |

### API responsibilities

FastAPI is the thin backend. All workflow APIs use `/api/production/`;
former unnamespaced simulator endpoints are unavailable. Keep these responsibilities stable:

| Surface | Responsibility |
|---|---|
| Workspace and runs | Identify production execution; list/read Studio runs and their stage state. |
| Data preparation | Submit preparation and return a run ID; expose the resulting data card. |
| Training | Expose defaults and submit training using accepted configuration; GPT suggestions remain planned. |
| Evaluation | Submit comparison for an explicit or resolved latest candidate; expose evidence. |
| Gate and registration | Read fixed scientific policy, comparisons and lineage; register only passing evidence after approval. |
| Deployment and links | Confirm and submit deployment; return Azure resource/job/MLflow links and coarse status. |

## 4. UI screens

These are logical screens or sections, not a requirement for separate pages. A small frontend is sufficient.

| Screen | Main content and actions |
|---|---|
| Studio / run selection | Real-service cost notice, current run, recent runs, stage summary, and themed Azure links. No mode selector. |
| Data preparation | Folder and generation-recipe selection, billable-action confirmation, validation outcome, and data card. |
| Fine-tuning | Dataset selector/manual asset ID, format popup, Foundry model selection, editable defaults, Train confirmation, and job link. |
| Evaluation | Candidate and baseline identities, Evaluate action, five-metric comparison table, and evidence status. |
| Candidate release (tab 4) | Recent evaluations, release-policy and activity-lineage tables, colored gate badges, register-only/deploy choices, SKU/hourly quote and exact-plan approval. Includes actual deployment status, authenticated scoring URI, scoring validation and weekly drift schedule. No fifth tab, rollout controls or redundant endpoint/monitoring placeholder text. |

UI rules: clearly distinguish pending, failed, passed, blocked and unrecorded states using explicit labels and color. Tables use their available width; long fields wrap or scroll. Never show a recommendation or successful submission as a completed deployment. Do not add dashboard infrastructure to display these summaries.

## 5. Data and run state

### Run records

The `runs/` folder is the local Studio workflow ledger and state store. Azure ML and MLflow remain authoritative for remote jobs, artifacts, and metrics.

- Studio state lives under `runs/production/`. The simulator store and seeded demo data have been removed.
- Existing training records use `runs/<Azure ML job name>.yaml` through [the run registry](../lib/run_registry.py). Preserve compatibility rather than silently change the meaning of existing IDs.
- A Studio run has a unique workflow ID created during data preparation. Store linked training job names, evaluation job names, MLflow IDs, and model versions as distinct identifiers.
- Persist schema version, mode, timestamps, current stage, per-stage outcome/error, and lineage references.
- Persist the data card, training configuration, evidence, price snapshot, fixed policy, deployment plan/hash, approval and resource identities as sections or referenced artifacts.
- Never store credentials or secret values in run records.

### Data card and lineage

Minimum data-card content: generation recipe, source references, split counts, validation outcome, dataset fingerprint, and Azure ML asset name/version. Link the dataset to training configuration, the saved model artifact, the exact evaluation test split, and any registered/deployed model.

### Stage state

Workflow: `data preparation -> training -> evaluation -> decision/registration -> deployment/monitoring`.

Each stage records `not_started`, `running`, `succeeded`, or `failed`. Gate outcomes and approvals are separate fields. Production registration requires passing evidence; deployment additionally requires exact-plan and hourly-cost approval. Later stages cannot consume missing or failed prerequisites.

### Latest-run selection

Default evaluation to the latest submitted training record in the active production workspace and experiment. Resolve and persist the exact candidate before submission; allow explicit selection. If that candidate is unfinished or failed, show its state and block evaluation rather than silently switch to an older model. Do not use registration time or file modification time to determine latest.

### Initial evaluation metrics

Compare all three models using the same records and scoring policy:

| Metric | Intended meaning | Better direction |
|---|---|---|
| Faithfulness | Support for the response in supplied context. | Higher |
| Accuracy | Correctness against reference answers or an agreed rubric. | Higher |
| Relevancy | How well the response addresses the task/question. | Higher |
| Latency | Response-generation duration under a documented measurement policy. | Lower |
| Token cost | Per-response cost using a documented pricing/accounting policy. | Lower |

Production uses faithfulness, relevancy and correctness for immutable `paired-quality-v2` readiness. When the original SLM scores exactly 100% for a metric, require the fine-tuned model to score exactly 100% on every sealed-test example; waive only that metric's uplift and positive-confidence checks. Do not treat a perfect tie as evidence of improvement. Near-perfect baselines and any regression do not qualify. Sample-count, GPT-gap and other metrics' requirements remain unchanged. Latency, token usage and frontier API costs are informational. Price only named API models; self-hosted SLM token cost is not applicable. Preserve unknown values, exact API model identities, reused-baseline provenance and immutable price snapshots.

## 6. Architecture decisions

```text
Local browser UI -> Local FastAPI backend -> Existing pipeline adapters -> Azure ML / MLflow
                              |                         |               Foundry / Key Vault
                              +-> Local YAML ledger     |
```

| ID | Decision | Rationale / boundary |
|---|---|---|
| AD-01 | Local hosting and backend Azure CLI credentials initially. | Matches the current operator workflow; hosted identity design is deferred. |
| AD-02 | Thin FastAPI backend; vanilla JavaScript frontend. | Keep production orchestration separate from UI without rebuilding the pipeline or retaining a simulator. |
| AD-03 | Reuse shared Python functions, not CLI subprocess wrappers. | Keep validation, gate logic, and model handling consistent with scripts. |
| AD-04 | YAML for local workflow state; Azure for execution truth. | Preserve the existing run-ledger direction while reconciling remote status on refresh. |
| AD-05 | Deep-link to Azure for detailed observability. | No loss charts or monitoring dashboard. Saved job-status logs refresh every 10 seconds. |
| AD-06 | Evaluate raw training output before registration. | Preserve the existing non-circular gate: evaluate the model saved by [the training entry point](../lib/train.py), then register that same artifact. |
| AD-07 | Candidate-bound evidence and human deployment confirmation. | A gate uses the selected model's sealed evidence. GPT advises hardware only; fixed scientific policy has no override. Deployment requires reviewed GPU cost approval. |

## 7. Incremental delivery

Production preparation, training, evaluation and approved candidate release are implemented with mocked cloud tests. Live billable deployment validation and production training suggestions remain pending.

| Increment | Scope | Exit criteria |
|---|---|---|
| I-01: Production-only Studio | Completed: remove demo UI, simulator API/code, seed records and saved demo data. | Four full-width tabs, production-only routing, empty unseeded workspace and preserved production records are tested. |
| I-02: Gate and registration | Implemented: select a completed evaluation, review fixed scientific policy/lineage and register passing evidence without overrides. | Candidate-bound hashes, cloud lineage, human approval and audit records are mock-tested. |
| I-03: Data and training | Implemented: preparation, complete CoT sample/data-card tables, dataset selection, trainer controls and training submission. GPT suggestions remain planned. | Dataset-to-model lineage persists; accepted configuration is used by the existing training pipeline. |
| I-04: Comparison and recommendation | Implemented: evaluation job selection, three-model quality/latency comparison, frontier pricing and transparent scientific readiness. | Fixed scoring policy, colored per-check outcomes and immutable cost snapshots are tested; missing values are not fabricated. |
| I-05: Deployment and monitoring | Implemented: review GPT SKU/Azure hourly quote, approve exact plan, deploy dedicated scoring endpoint and weekly PSI schedule. | Mock tests verify identity, cost approval, default-endpoint scoring and partial failures; live cloud validation remains pending. No rollout controls. |

Credential protection, production evidence integrity, error handling, and decision auditing belong to the first applicable increment, not a later hardening phase.

## 8. Open decisions and maintenance

Resolve these before implementing the affected feature:

| Decision | Blocks |
|---|---|
| What exactly is context-assisted dataset generation, and how do its behavior and schema differ from the existing context-grounded recipe? | F-02 |
| Which Foundry models and GPT deployment should future training-parameter suggestions support? | Planned F-03 suggestions |
| Which additional scoring rubrics or compute-cost allocation are needed beyond the fixed quality policy and informational latency/frontier pricing? | Future evaluation enhancements |
| Who approves a billable Azure smoke test and its resource cleanup? | Live F-08 validation |
| What cleanup/rollback automation is appropriate after partially completed deployment? | Future F-08 enhancement; current errors preserve resource identities and warn about charges |

Maintenance rules:
- Keep this file as the product source of truth; avoid a second competing plan.
- Keep feature, technical, and decision IDs stable. Add or revise requirements instead of rewriting unrelated sections.
- For each GHCP implementation task, name the increment and requirement IDs, bounded scope, and acceptance checks. Explicitly exclude other increments.
- Mark delivery status only after acceptance checks pass; retain unresolved decisions rather than hide assumptions in code.
- Record significant product changes below in one line. Do not store detailed engineering task logs here.

### Change log

| Date | Change |
|---|---|
| 2026-10-02 | Reformatted as a living GHCP product spec; reconciled the five-stage Studio goal with a gate-first production increment and retained the no-dashboard boundary. |
| Current release | Completed production preparation, fine-tuning, evaluation and scientific release integration; model-specific frontier token pricing, themed Azure links, readable policy/activity lineage, full-width four-tab navigation and colored gate results. Removed demo functionality/code/APIs/data, redundant fine-tuning button and endpoint placeholder copy; retained production records and documented live-cloud validation as pending. |
| 2026-10-22 | Revised readiness to paired-quality-v2: per-metric perfect-100% baseline ties waive only impossible positive-uplift/confidence requirements. Require exact per-example non-regression; retain sample-count and GPT-gap gates and display PASS (PERFECT TIE) with explanatory requirements. |
| 2026-10-22 | Aligned UI/API preparation defaults to 300 total samples; explain the source-grouped split and 30 sealed-test-example release minimum. Data cards check actual test counts independently from dataset validation and quality readiness; smaller experimental datasets remain supported. |
| 2026-10-22 | Scanned all library modules and consolidated four helper modules into release, Azure operations and monitoring boundaries. Removed unused legacy release/deployment policies and traffic promotion, and shared checkpoint resolution, metric definitions, evidence hashes and JSON writing without removing Azure runtime entry points. |
