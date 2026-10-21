# SLM Studio - GitHub Copilot Product Spec

**Status:** Five-stage demo implemented; production context-grounded data preparation connected to the UI and API. Later production stages remain planned.
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
- Support the five-stage workflow: data preparation, fine-tuning, evaluation, deployment decision, and deployment/monitoring.
- Offer an isolated demo mode and a real Azure-connected production mode.
- Reuse the existing pipeline rather than create a second training or evaluation implementation.
- Show concise decision evidence, including the GPT-baseline gate and actual metric values.
- Require user confirmation for deployment and deliberate confirmation for a gate override.
- Use Azure ML Studio and Azure AI Foundry for detailed logs, loss curves, charts, and monitoring.

Non-goals:
- Training dashboards, live loss charts, embedded log viewers, or a model-comparison playground.
- A hosted, public, or multi-user service in the initial release.
- Autonomous deployment based only on an LLM recommendation.
- Replacing Azure ML or implementing a new fine-tuning engine.

Planning assumption: the five-stage workflow is the target product; the earlier gate-only UI is its first production increment, not a competing scope. "Production mode" means real pipeline execution, not production hosting.

## 2. Functional features

The state below describes the Studio UI/API, not the availability of standalone
pipeline scripts. Production means real pipeline execution, not public hosting.

| ID | Feature | State | Required behavior / acceptance |
|---|---|---|---|
| F-01 | Demo and production modes | Implemented for data preparation; later production actions disabled | A visible mode selector distinguishes simulated and real actions. Demo uses dummy records in `runs/demo/` and never invokes pipeline modules, Azure, or billable GPT calls. Production preparation uses the existing pipeline through separate `/api/production` endpoints and persists isolated records in `runs/production/`. |
| F-02 | Data preparation | Context-grounded production flow implemented and integration-tested; context-assisted pending definition | Select a server-accessible PDF folder, recipe, and exact total number of samples across all splits. Preserve questions-per-call and oracle-probability controls. Confirm billable execution, then run PDF extraction, dataset generation/validation, and versioned Azure ML data-asset publication asynchronously. Persist a data card under a unique Studio run ID and show split counts, fingerprint, validation, asset reference, and sample question/answer previews. Short generation and pipeline errors fail visibly; interrupted stages retain failed state. The context-assisted recipe remains disabled in production until defined. Tests isolate cloud execution; a live billable Azure smoke test is not yet performed. |
| F-03 | Training configuration | Demo implemented; production planned | Select an available Azure AI Foundry base model and edit training parameters, including batch size and gradient accumulation. Suggest and prefill parameters using dataset characteristics and GPT-assisted interpretation of the [Unsloth LoRA guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide). Suggestions remain editable and are not treated as validated facts. |
| F-04 | Training submission | Demo implemented; production planned | Confirm the selected data, model, parameters, and GPU cost implications before training. Persist the submitted Azure ML job identity and configuration. Demo shows simulated progress; production shows coarse stage status and Azure links, not training telemetry. |
| F-05 | Model evaluation | Demo and read-only Azure snapshot implemented; production submission planned | Default to the latest applicable training run, with explicit run selection available. Evaluate the saved fine-tuned artifact, the original base SLM, and the GPT baseline on the same sealed test records. Show the five metrics defined in Section 5 and persist the evidence. |
| F-06 | Gate and recommendation | Demo implemented; production and LLM recommendation planned | Show passed, blocked, or pending evidence status with actual comparison values. Use scores as context for a meta-prompt that recommends deployment readiness and explains its reasoning. An LLM recommendation cannot change the deterministic registration gate. |
| F-07 | Register and override | Demo implemented; production planned | Allow candidate release when the gate passes. A blocked gate requires a deliberate override with typed confirmation and a recorded reason. Record actor, time, candidate, and evidence. Stage 04 registers the candidate, deploys it at zero traffic, and creates the drift reference. |
| F-08 | Deployment and monitoring | Demo implemented; production planned | After separate user confirmation, run the deployment gate, configure drift monitoring, and route the selected canary or full traffic percentage. Production will orchestrate `scripts/04_release_candidate.py` and `scripts/05_deployment_gate.py`; the demo performs no live calls. |

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
| T-04 | Mode isolation | Use a shared API contract with separate demo and production execution adapters and state paths. Demo records must never qualify as production candidates or evidence. |
| T-05 | Safe actions | Show confirmations for billable generation, training, evaluation, and deployment. Repeated actions must not silently create duplicate resources. Persist override and deployment decisions without secrets. |
| T-06 | Testable behavior | Validate demo isolation, run selection, stage transitions, candidate-bound evaluation, missing/failed evidence, gate outcomes, confirmations, and restart recovery. Demo metrics must be clearly labeled simulated. |

### API responsibilities

FastAPI is the thin backend. Demo endpoints remain under `/api`; production
preparation uses `/api/production/config` and `/api/production/runs`. Exact routes
for later production stages remain implementation details; keep these responsibilities stable:

| Surface | Responsibility |
|---|---|
| Mode and runs | Identify active mode; list/read Studio runs and their stage state. |
| Data preparation | Submit preparation and return a run ID; expose the resulting data card. |
| Training | Suggest parameters and submit training using accepted configuration. |
| Evaluation | Submit comparison for an explicit or resolved latest candidate; expose evidence. |
| Gate and registration | Read gate outcome; register or explicitly override a blocked gate. |
| Deployment and links | Confirm and submit deployment; return Azure resource/job/MLflow links and coarse status. |

## 4. UI screens

These are logical screens or sections, not a requirement for separate pages. A small frontend is sufficient.

| Screen | Main content and actions |
|---|---|
| Studio / run selection | Visible demo/production mode, current run, recent runs, stage summary, and Azure links. |
| Data preparation | Folder and generation-recipe selection, billable-action confirmation, validation outcome, and data card. |
| Fine-tuning | Data summary, Foundry model selection, suggested editable parameters, Train confirmation, and job link. |
| Evaluation | Candidate and baseline identities, Evaluate action, five-metric comparison table, and evidence status. |
| Candidate release | Offline gate status, actual values, readiness recommendation, approval/override flow, registration receipt, zero-traffic endpoint, and drift reference. |
| Deployment gate | Endpoint gate evidence, operator confirmation, selected traffic percentage, resulting traffic map, and drift-monitoring schedule. |omes, and links to Azure-native views. |

UI rules: clearly distinguish pending, failed, simulated, passed, and blocked states. Never show a recommendation or successful submission as a completed deployment. Do not add dashboard infrastructure to display these summaries.

## 5. Data and run state

### Run records

The `runs/` folder is the local Studio workflow ledger and state store. Azure ML and MLflow remain authoritative for remote jobs, artifacts, and metrics.

- Production state lives under `runs/`; demo state lives under `runs/demo/` and is excluded from production selection.
- Existing training records use `runs/<Azure ML job name>.yaml` through [the run registry](../lib/run_registry.py). Preserve compatibility rather than silently change the meaning of existing IDs.
- A Studio run has a unique workflow ID created during data preparation. Store linked training job names, evaluation job names, MLflow IDs, and model versions as distinct identifiers.
- Persist schema version, mode, timestamps, current stage, per-stage outcome/error, and lineage references.
- Persist the data card, chosen training configuration, evaluation evidence, gate decision, recommendation, override reason, and deployment result as sections or referenced artifacts. Exact file layout is deferred.
- Never store credentials or secret values in run records.

### Data card and lineage

Minimum data-card content: generation recipe, source references, split counts, validation outcome, dataset fingerprint, and Azure ML asset name/version. Link the dataset to training configuration, the saved model artifact, the exact evaluation test split, and any registered/deployed model.

### Stage state

Workflow: `data preparation -> training -> evaluation -> decision/registration -> deployment/monitoring`.

Each stage records `not_started`, `running`, `succeeded`, or `failed`. Gate outcomes and user decisions are separate fields, not inferred from stage completion. Registration requires passing evidence or an explicit override; deployment additionally requires user confirmation. Later stages must not consume missing or failed prerequisites.

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

These are target product metrics, not all currently implemented. Existing JSON-validity and latency gate evidence remains visible until an explicit gate-policy revision. Do not silently replace the current gate with new metrics. Record scorer/policy versions, model identities, and missing values; missing metrics are not zero scores. Local SLM token cost requires an agreed compute-cost allocation, not GPT token pricing.

## 6. Architecture decisions

```text
Local browser UI -> Local FastAPI backend -> Existing pipeline adapters -> Azure ML / MLflow
                              |                         |               Foundry / Key Vault
                              +-> Local YAML ledger     |
                              +-> Isolated demo adapter (no Azure calls)
```

| ID | Decision | Rationale / boundary |
|---|---|---|
| AD-01 | Local hosting and backend Azure CLI credentials initially. | Matches the current operator workflow; hosted identity design is deferred. |
| AD-02 | Thin FastAPI backend; minimal frontend. | Keep orchestration separate from UI without rebuilding the pipeline. Frontend framework remains undecided. |
| AD-03 | Reuse shared Python functions, not CLI subprocess wrappers. | Keep validation, gate logic, and model handling consistent with scripts. |
| AD-04 | YAML for local workflow state; Azure for execution truth. | Preserve the existing run-ledger direction while reconciling remote status on refresh. |
| AD-05 | Deep-link to Azure for detailed observability. | No loss charts, embedded log tailing, or monitoring dashboard. Coarse stage refresh is allowed. |
| AD-06 | Evaluate raw training output before registration. | Preserve the existing non-circular gate: evaluate the model saved by [the training entry point](../lib/train.py), then register that same artifact. |
| AD-07 | Candidate-bound evidence and human deployment confirmation. | A gate must use the selected model's evidence. An LLM recommendation is advisory; overrides are explicit and auditable. |

## 7. Incremental delivery

Each increment delivers a usable slice. All are **not started**; timing estimates are deferred until the open decisions are resolved.

| Increment | Scope | Exit criteria |
|---|---|---|
| I-01: Spec and demo | Resolve blocking definitions; minimal mode-aware UI/API with dummy five-stage records. | The workflow can be demonstrated without pipeline imports, Azure calls, or secrets. |
| I-02: Gate and registration | First production slice: select an existing training job, view its evidence/gate, register or deliberately override, and open Azure links. | Candidate-bound evidence, CLI credential isolation, confirmations, and audit records work. |
| I-03: Data and training | Add real preparation, data cards, model selection, parameter suggestions, and training submission. | Dataset-to-model lineage persists; accepted configuration is used by the existing training pipeline. |
| I-04: Comparison and recommendation | Add evaluation submission and the three-model/five-metric comparison with advisory readiness judgment. | Scoring and cost policies are explicit; existing gate behavior changes only through an approved policy revision. |
| I-05: Deployment and monitoring | Wire the Stage 04 candidate release and Stage 05 deployment gate after confirmation. | Deployment and monitoring outcomes persist; resource links work; no dashboards are added. |

Credential protection, demo isolation, error handling, and decision auditing belong to the first applicable increment, not a later hardening phase.

## 8. Open decisions and maintenance

Resolve these before implementing the affected feature:

| Decision | Blocks |
|---|---|
| What exactly is context-assisted dataset generation, and how do its behavior and schema differ from the existing context-grounded recipe? | F-02 |
| Which Foundry models and GPT deployment are supported for training suggestions and evaluation? | F-03, F-05 |
| What scoring rubrics, latency protocol, compute-cost allocation, and eventual gate thresholds apply to the five metrics? | F-05, any gate-policy revision |
| Which Stage 04/05 operations are mandatory, optional, or separately confirmed, including cleanup/rollback? | F-08 |
| How will Studio workflow records coexist with existing per-job YAML records, including safe writes and workspace scoping? | Production state integration |
| Which minimal frontend approach best fits the team: plain React or server-rendered HTML/htmx? | UI implementation |

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
