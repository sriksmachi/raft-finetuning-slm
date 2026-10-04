# SLM Studio

SLM Studio is a script-and-notebook, production-oriented workflow for supervised fine-tuning (SFT) of small language models (SLMs) on Azure Machine Learning. The pipeline — data preparation, versioned assets, managed-identity training, sealed-test evaluation, gated deployment, and drift monitoring — is generic to any SFT objective (instruction tuning, domain adaptation, structured extraction, tool use). It ships with a worked example for **context-grounded question answering**.

The active workflow uses Azure Machine Learning data, environment, job, model, endpoint, and schedule assets. Reusable implementation code lives in `lib/`; scripts orchestrate every state-changing stage, while notebooks are limited to readiness and exploration. To adapt the playbook to a different SFT task, swap the dataset builder and record schema (`lib/dataset_generation.py`, `lib/data.py`) and the prompt contract (`lib/prompts.py`); the training, evaluation, deployment, and monitoring stages are unchanged.

## Why this repository

**Impact:** it turns SLM fine-tuning from an ad-hoc notebook experiment into a repeatable, governed path to production. A specialized small model produced this way delivers domain accuracy comparable to a much larger general model at a fraction of the serving cost and latency, with the data lineage, evaluation evidence, and rollback controls that a real deployment review requires.

Top three reasons to learn from it:

1. **End-to-end and production-oriented, not a toy.** Every stage a real project needs — leakage-free data splits, immutable versioned assets, managed-identity training, sealed-test evaluation, gated deployment, lineage from git-commit to deployed model and drift monitoring — is wired together and explained, so you see how the pieces fit rather than a single isolated training script.
2. **Reusable across supervised fine-tuning tasks.** The training, evaluation, deployment, and monitoring stages are task-agnostic; only the dataset builder and prompt contract change. Context-grounded question answering is the worked example, but the same scaffold applies to instruction tuning, domain adaptation, structured extraction, or tool use.
3. **Governance and evaluation are first-class.** Explicit quality gates, an LLM-judge evaluation harness, PSI drift detection, and secret-free credential handling teach the operational discipline that separates a demo from a deployable system.

## Architecture

![SLM Studio end-to-end design: data preparation and versioned data asset, data exploration, Azure ML LoRA training, three-model inference with DeepEval, gated candidate registration, and deployment with drift monitoring, alongside the supporting Azure services.](docs/architecture-png.png)

The pipeline flows through scripts and supporting notebooks; each stage emits an artifact that the next stage consumes. Test-set evaluation happens inside the inference job, so separate training and offline evaluation notebooks are not required. Supporting Azure services (Azure OpenAI, GPU compute with managed identity, Key Vault, AI Foundry base weights, MLflow, and drift monitoring) attach to the stages that use them. Source: [docs/architecture.drawio](docs/architecture.drawio).

## Workflow

Run in order from the repository root. Each stage produces the inputs the next stage consumes.

| # | Entry point | Description | Outputs | Dependencies |
|---|----------|-------------|---------|--------------|
| 00 | [Workspace Readiness](notebooks/00_workspace_readiness.ipynb) | Health-checks the active kernel and compute before any fine-tuning work: Python and package versions, GPU/CUDA/VRAM/bf16, Azure ML workspace and GPU compute visibility, Hugging Face Hub reachability, a two-step QLoRA SFT loop, and a sample generation. Emits a PASS/FAIL summary; raises `AssertionError` if any required check fails. Run this first after provisioning or restarting compute to confirm the environment is fine-tuning-ready. | Structured readiness report in the notebook output. | The venv from [scripts/setup_env.sh](scripts/setup_env.sh); GPU compute; Azure ML workspace access. |
| 01 | [Data Preparation](scripts/01_prepare_data.py) | Extracts PDF chunks, generates and validates local context-grounded JSONL splits with grouped (leakage-free) train/validation/test partitioning, fingerprints the dataset, and optionally publishes an immutable Azure ML data asset. Synthetic generation is optional and billable. | Local `train`/`validation`/`test` JSONL splits, `manifest.json` (split counts, class balance, SHA-256 fingerprint), versioned Azure ML data asset. | Local PDF/text chunks; Azure OpenAI (only for synthetic generation); Azure ML workspace/datastore (only to publish). |
| 01 | [Data Exploration](notebooks/01_data_exploration.ipynb) | Explores the generated dataset: validates oracle/distractor balance and answer formatting, and analyzes CoT answer and instruction length distributions to choose `max_new_tokens` and `max_seq_length` for training. | Quality/balance summaries, length-distribution plots, computed columns persisted for training configuration. | Local context-grounded JSONL splits from 01. |
| 02 | [Azure ML Fine-Tuning](scripts/02_azureml_fine_tuning.py) | Registers a versioned CUDA environment and submits a managed-identity SLM fine-tuning command job (`lib/train.py`, Unsloth-based LoRA adaptation) with prompt-token loss masking. It records lineage but does not register or deploy a model. | Registered training environment, completed MLflow job, named LoRA adapter and tokenizer in the job output, run-ledger entry. | Versioned data asset from 01; GPU compute cluster with managed identity; versioned base weights. |
| 03 | [Offline Evaluation](scripts/03_azureml_slm_vs_gpt_evaluation.py) | Mounts the completed training job's raw adapter output before registration, compares OOB, fine-tuned, and GPT responses on the complete sealed test set, and applies DeepEval faithfulness, relevancy, and correctness metrics. | Candidate-bound CSV, hash-bound evidence manifest, RAG means and latency in MLflow. | Completed Stage 02 job; sealed test split; GPU host; independent Azure OpenAI judge. |
| 04 | [Candidate Release](scripts/04_release_candidate.py) | Verifies Stage 02/03 cloud lineage and immutable evidence, applies aggregate and slice release thresholds, captures explicit approver identity/reason, registers the model, deploys it at zero traffic, and materializes a PSI drift baseline. | Approval and release receipt, registered model version, zero-traffic endpoint deployment, drift reference profile. | Passing Stage 03 evidence; explicit human approval; deployable base-model reference; endpoint infrastructure. |
| 05 | [Deployment Gate](scripts/05_deployment_gate.py) | Invokes the named zero-traffic candidate on the complete sealed test set, gates on token-F1 plus mean/p95 latency, configures scheduled PSI monitoring, and only then routes canary or full traffic. | Deployment evidence, immutable promotion receipt, explicit traffic map, scheduled drift monitor. | Stage 04 receipt; governed production request URI; monitoring compute; one incumbent deployment for partial rollout. |

Prior exploratory and Kaggle notebooks are retained under `notebooks/legacy/` for historical reference. They are not part of the production workflow.

**Governed release sequence** (run in this order):
```powershell
python scripts/02_azureml_fine_tuning.py --wait
python scripts/03_azureml_slm_vs_gpt_evaluation.py --storage-account-name <account> --gpt-endpoint <judge-endpoint> --wait
python scripts/04_release_candidate.py --endpoint-name <endpoint> --base-model-reference <deployable-model> --approve --approved-by <identity> --approval-reason <change-record>
python scripts/05_deployment_gate.py --release-receipt output/releases/<release>/candidate-release.yaml --production-data-uri <governed-uri> --traffic-percent 10
```
Every job submitted by 02 is recorded as a YAML file under `runs/` (see `lib/run_registry.py`). Stages 03 and 04 default to the latest matching record, while release and promotion receipts under `output/releases/` bind every state transition to immutable evidence. Use `--traffic-percent 100` for the first deployment because a canary split requires one incumbent deployment.

## Repository Layout

```text
.
├── notebooks/
│   ├── 00_workspace_readiness.ipynb
│   ├── 01_data_exploration.ipynb
│   └── legacy/
├── scripts/
│   ├── 01_prepare_data.py
│   ├── 02_azureml_fine_tuning.py
│   ├── 03_azureml_slm_vs_gpt_evaluation.py
│   ├── 04_release_candidate.py
│   ├── 05_deployment_gate.py
│   └── setup_env.sh         # Provision /mnt venv + Jupyter kernel on AML compute
├── lib/
│   ├── azureml_ops.py       # Jobs, assets, registration, deployment, promotion
│   ├── config.py            # Environment-driven workspace connection
│   ├── data.py              # Validation, grouped splits, fingerprint, publication
│   ├── evaluation.py        # Exact match and answer token-F1
│   ├── deepeval_metrics.py  # Azure-authenticated DeepEval RAG judge
│   ├── endpoint_inference.py # Endpoint invocation and held-out evaluation
│   ├── inference.py         # Remote fine-tuned-SLM-vs-GPT inference/evaluation entry point
│   ├── lora_artifacts.py    # Named adapter export, validation, and loading
│   ├── release.py           # Evidence validation, release policy, receipts
│   ├── drift_schedule.py    # Governed scheduled PSI monitoring
│   ├── monitoring.py        # Reference profiles and PSI drift
│   ├── monitor_drift.py     # Scheduled monitoring job entry point
│   ├── run_registry.py      # YAML run ledger under runs/ (handoff across stages 02-05)
│   ├── train.py             # Remote SLM finetuning command-job entry point (Unsloth LoRA)
│   ├── score.py             # Managed endpoint scoring entry point
│   └── prompts.py           # Shared training/inference prompt contract
├── environments/            # Azure ML conda definitions
├── data/                    # Local source and context-grounded training data
├── output/                  # Historical local predictions
├── runs/                    # YAML record per training run (gitignored; see lib/run_registry.py)
└── tests/                   # Fast governance and metric tests
```

## Prerequisites

- Python 3.12.10 (pinned by `environments/train-conda.yml` for training; the local notebook kernel should match)
- Azure CLI authenticated with `az login` for local work
- Azure ML workspace, GPU/CPU compute clusters, and at least one datastore
- Permissions to create Azure ML assets, jobs, endpoints, deployments, and schedules
- Managed identity on compute with least-privilege datastore access
- Azure Key Vault access for a gated Hugging Face model token
- Accepted license/access for `meta-llama/Llama-3.2-3B-Instruct`
- Azure OpenAI only when regenerating synthetic data from local documents
- Azure OpenAI / Azure AI Foundry GPT deployment for Stage 03 comparison and judging

### Azure ML datastore

Stage 01 uploads the generated dataset through an Azure ML datastore. Before
running `scripts/01_prepare_data.py`, create a datastore backed by an existing
Blob Storage container if the workspace does not already have one. Save the
following as `datastore.yml`, replacing the placeholder values:

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/azureBlob.schema.json
name: <datastore-name>
type: azure_blob
account_name: <storage-account-name>
container_name: <container-name>
```

Create the datastore in the target workspace:

```powershell
az login
az extension add --name ml
az ml datastore create `
  --file .\datastore.yml `
  --resource-group <resource-group> `
  --workspace-name <workspace-name>
```

This credentialless definition uses identity-based access. Grant the user
running Stage 01 and the relevant compute managed identities the least
privilege required on the container, such as `Storage Blob Data Contributor`.
Then set `AZUREML_DATASTORE_NAME=<datastore-name>` in `.env` or pass
`--datastore <datastore-name>` to Stage 01. The storage account and container
must exist before creating the datastore.

### Local tokenizer-loading compatibility

Transformers 4.57.2 has a local-checkpoint tokenizer-loading bug that raises
`AttributeError: 'dict' object has no attribute 'model_type'` inside
`tokenization_utils_base.py`. This can affect downloaded Foundry checkpoints
passed to Unsloth; it is not a dataset or response-masking error. The
training and inference configurations pin `transformers==4.57.1` to avoid
the 4.57.2 regression. Both use `torch>=2.9,<2.10` with the CUDA 12.8 wheel
index and pin `torchao==0.17.0`; TorchAO 0.18 requires a newer Torch API and
fails while importing Unsloth with Torch 2.9. Unsloth, TRL, and xFormers are
not pinned to exact versions; verify the resolved versions in the job's startup
logs rather than assuming a fixed dependency set.

Submit training with environment `slm-studio-finetuning`, with its version read
from `AZURE_TRAIN_ENVIRONMENT_VERSION` in `.env`, using
the digest-pinned CUDA 12.8 / OpenMPI 5 / Ubuntu 24 base image configured by
the orchestration.
Use a new unused environment version when changing its dependencies or image.
Environment registration first tries the requested version. If Azure rejects a
changed definition because that version is immutable, the shared registration
helper logs the conflict and registers the same local definition under the next
unused numeric version (including archived versions when determining the next
number). The returned version is used for the job and run record; existing
environments and local conda files are not overwritten. Concurrent version
conflicts are retried a bounded number of times; unrelated errors propagate.
Use `--skip-environment` to deliberately reuse an existing training environment.
Training reads `AZURE_TRAIN_ENVIRONMENT_VERSION` from the repository `.env`
(default `1` only when the setting is absent), unless the CLI explicitly
overrides it with `--train-environment-version`. After registration succeeds,
the actual version is saved back to `.env` and refreshed in the current process.
Numeric versions never downgrade an already higher saved version. Failed
registrations do not update `.env`; unrelated keys and `train-conda.yml` are
preserved. CLI training uses the same version resolver and persistence.
Inference defaults to `slm-studio-inference`, version `2`, including DeepEval, and is registered
separately from training. Changing a dependency file
does not repair an existing Azure ML job environment. Full GPU compatibility
still requires a workspace readiness check and a real job run; this
configuration alone is not evidence that training has succeeded.

### Active Azure workflow defaults

| Setting | Default |
|---------|---------|
| Workspace / GPU compute | `sriks-aml-ws` / `gpu-cluster2` |
| Data asset alias | `slm-studio-instance-security:20260805.042350` |
| Training environment / experiment | `slm-studio-finetuning:<AZURE_TRAIN_ENVIRONMENT_VERSION>` / `slm-studio-finetuning` |
| Evaluation environment / experiment | `slm-studio-inference:2` / `slm-studio-slm-vs-gpt-eval` |
| Candidate model name | `slm-studio-llama32-3b` |

The data alias points to the same stored files as the existing data asset;
it does not regenerate or modify the dataset. Model versions remain explicit
or run-specific; an environment or asset name alone does not establish a
successful training run or a passing evaluation gate.

### Local environment on Azure ML compute

Use [scripts/setup_env.sh](scripts/setup_env.sh) to provision a Python 3.12.10 venv on the compute VM's local `/mnt` scratch disk. `/mnt` is dramatically faster than `~/cloudfiles` (an Azure Files share) for `import`-heavy notebooks, but it is wiped on compute stop/restart, so re-run the script after every restart.

```bash
bash scripts/setup_env.sh
source /mnt/tmp/venvs/slm-studio-py312-mnt/bin/activate
```

The script installs `uv`, pins its Python install and wheel cache to `/mnt`, installs `requirements.txt`, and registers a Jupyter kernelspec named `slm-studio-py312-mnt` (display name `Python 3.12.10 (SLM Studio /mnt)`) so the notebooks can pick it directly. Override `VENV_DIR`, `KERNEL_NAME`, or `KERNEL_DISPLAY_NAME` to run multiple parallel environments.

After setup, open [notebooks/00_workspace_readiness.ipynb](notebooks/00_workspace_readiness.ipynb) with the `slm-studio-py312-mnt` kernel and run it top-to-bottom to confirm the environment is fine-tuning-ready before submitting any Azure ML jobs.

### Local environment on a workstation

For local (non-Azure ML compute) work, a plain venv is enough:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
az login
```

Set workspace identifiers in `.env` or the process environment:

```dotenv
AZURE_SUBSCRIPTION_ID=<subscription-id>
AZURE_RESOURCE_GROUP=<resource-group>
AZUREML_WORKSPACE_NAME=<workspace-name>
AZUREML_DATASTORE_NAME=<datastore-name>

# Required only for synthetic data generation
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-08-01-preview
AZURE_OPENAI_GPT_DEPLOYMENT=gpt vision
```

`AZUREML_DATASTORE_NAME` is optional when the workspace has a default
datastore. If the workspace has no `workspaceblobstore` default, set it to an
existing datastore or pass `--datastore <datastore-name>` to
`scripts/01_prepare_data.py`.

>> Do not store credentials or tokens in `.env`, notebooks, job parameters, or MLflow. The training job can retrieve a Hugging Face token from Key Vault using managed identity.

## LoRA Training Outputs and Inference

Training exports `outputs/model/finetuned_adapter/` containing
`adapter_model.safetensors`, `adapter_config.json`, and the saved tokenizer.
`run_id.txt` remains at the model output root. Base weights are not merged or
copied into this output; intermediate checkpoints and optimizer states remain
in the job-local `training_checkpoints/` directory.

Use `--lora-adapter-name <name>` in the fine-tuning script to choose another
directory/adapter name (letters, digits, underscores, and hyphens only).
The inference job still mounts the training job's `outputs/model` artifact
through `--fine-tuned-model`; it resolves the single exported adapter, loads the
explicit `--model` base weights, and attaches/activates the adapter with PEFT.
Use the exact base model version used for training. Adapter inference loads a
second copy of that same base, leaving the returned OOB model unmodified.
The inference entry point expects adapter exports,
not historical merged checkpoints.

Managed endpoint scoring supports adapter exports and legacy merged models.
For adapters, pass `base_model` to `deploy_managed_endpoint` (the deployment
notebook reads `SLM_BASE_MODEL`). This must reference the original base weights
available in the deployment: a provisioned checkpoint path or accessible
Hugging Face ID. Configure required model access separately; the training
job's temporary mount path cannot be reused and base weights are not bundled
in the registered adapter.

### Three-model DeepEval comparison

The evaluation submission script selects `--mode both --compare-with-gpt`.
Only `test.jsonl` supplies evaluation records. Every model receives the same
question and flattened retrieved context using the training prompt contract;
`cot_answer` is retained as ground truth and never included in generation prompts.
The models generate reasoning/`<ANSWER>` text, not forced JSON responses.

DeepEval runs `FaithfulnessMetric`, `AnswerRelevancyMetric`, and a `GEval`
correctness metric. Scores range from 0 to 1; higher is better. Faithfulness
uses retrieved context, answer relevancy uses the question, and correctness
compares the response semantically against `cot_answer`. Metrics are measured
locally using an Azure-authenticated judge; DeepEval telemetry is disabled and
no Confident AI upload is performed. Judging makes additional billable Azure
OpenAI calls.

Use `--judge-deployment <deployment>` to select an independent judge on the same
`--gpt-endpoint`; otherwise the GPT response deployment also acts as judge,
which can introduce evaluator bias.

The final `<run_id>_predictions.csv` is saved in the Azure ML predictions output,
logged as an MLflow artifact, and uploaded through the existing blob-storage
path. Each row contains `id`, `question`, `context`, `cot_answer`, all three
responses, and `{oob|fine_tuned|gpt}_{faithfulness|relevancy|correctness}_{score|reason}`
columns, plus per-model inference/evaluation errors and judge provenance.
Missing responses or invalid judge results are not assigned zero or successful
default scores: the CSV is preserved and the job fails explicitly. RAG averages
are logged only for complete evaluations.

Stage 04 evaluates faithfulness, relevancy, correctness, slice quality,
latency, and the GPT correctness comparison from the complete hash-bound CSV.
Thresholds are explicit CLI policy and cannot be bypassed by Stage 04.
JSON validity remains a diagnostic, not a release gate.
Register the updated inference dependencies under a new unused environment
version; existing images do not gain DeepEval when local files change.

## Data Preparation

Run the complete preparation pipeline from the repository root:

```powershell
python scripts/01_prepare_data.py
```

Reuse existing chunks or keep the run local without publishing to Azure ML:

```powershell
python scripts/01_prepare_data.py --skip-extract
python scripts/01_prepare_data.py --skip-publish
```

All questions generated from one oracle source chunk remain in the same split. This grouped split is mandatory: row-wise random splitting leaks source evidence across train, validation, and test sets and inflates evaluation.

## Dataset Contract

Each JSONL record contains:

- `id`: stable sample identifier
- `type`: `oracle` or `distractor`
- `question`: synthetic user question
- `context`: structured retrieved documents
- `oracle_context`: source evidence used to generate the answer
- `cot_answer`: supervised grounded response and final `<ANSWER>` span
- `instruction`: retrieved documents plus question supplied to the model

Publication fails on missing/empty fields, unsupported sample types, duplicate question/evidence pairs, or source overlap across splits. `manifest.json` records split counts, class balance, creation time, and the SHA-256 fingerprint.

## Production Gates

The notebooks make these controls explicit, but owners must set thresholds from business and risk requirements:

- Data: ownership, consent/license, PII classification, grouped splits, drift baseline, immutable versions
- Training: managed identity, Key Vault, pinned environment, MLflow lineage, checkpoint policy, cost limits
- Evaluation: sealed test data, quality by sample type, groundedness, abstention, safety, fairness, latency, load, cost
- Deployment: zero-traffic candidate, private networking, token auth, quotas, staged rollout, rollback deployment
- Monitoring: request/response governance, error and latency alerts, outcome quality, PSI drift, safety, budget alerts
- Governance: model/data cards, approver, incident owner, retention, deletion, license and red-team evidence

## Validation

Run fast local checks before submitting cloud jobs:

```powershell
python -m pytest tests -q
python -m compileall -q lib
```

Cloud resources are intentionally not created by tests. Execute notebook cloud cells only after replacing placeholder asset versions, endpoint names, compute names, and governed monitoring URIs.
