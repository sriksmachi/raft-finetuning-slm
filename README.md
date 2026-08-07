# SLM Supervised Fine-Tuning Playbook on Azure Machine Learning

A notebook-led, production-oriented playbook for supervised fine-tuning (SFT) of small language models (SLMs) on Azure Machine Learning. The pipeline — data preparation, versioned assets, managed-identity training, sealed-test evaluation, gated deployment, and drift monitoring — is generic to any SFT objective (instruction tuning, domain adaptation, structured extraction, tool use). It ships with a fully worked example for **Retrieval-Augmented Fine-Tuning (RAFT)**, a context-grounded question-answering recipe.

The active workflow uses Azure Machine Learning data, environment, job, model, endpoint, and schedule assets. Reusable implementation code lives in `lib/`; notebooks explain the data-science decisions and orchestrate those components. To adapt the playbook to a different SFT task, swap the dataset builder and record schema (`lib/raft_datagen.py`, `lib/data.py`) and the prompt contract (`lib/prompts.py`); the training, evaluation, deployment, and monitoring stages are unchanged.

## Why this repository

**Impact:** it turns SLM fine-tuning from an ad-hoc notebook experiment into a repeatable, governed path to production. A specialized small model produced this way delivers domain accuracy comparable to a much larger general model at a fraction of the serving cost and latency, with the data lineage, evaluation evidence, and rollback controls that a real deployment review requires.

Top three reasons to learn from it:

1. **End-to-end and production-oriented, not a toy.** Every stage a real project needs — leakage-free data splits, immutable versioned assets, managed-identity training, sealed-test evaluation, gated deployment, lineage from git-commit to deployed model and drift monitoring — is wired together and explained, so you see how the pieces fit rather than a single isolated training script.
2. **Reusable beyond RAFT.** The training, evaluation, deployment, and monitoring stages are task-agnostic; only the dataset builder and prompt contract change. RAFT is the worked example, but the same scaffold applies to instruction tuning, domain adaptation, structured extraction, or tool use.
3. **Governance and evaluation are first-class.** Explicit quality gates, an LLM-judge evaluation harness, SHAP diagnostics, PSI drift detection, and secret-free credential handling teach the operational discipline that separates a demo from a deployable system.

## RAFT vs. standard RAG

RAFT ([Zhang et al., 2024, arXiv:2403.10131](https://arxiv.org/abs/2403.10131)) trains the model on prompts that mix a relevant *oracle* document with several irrelevant *distractor* documents, and supervises a chain-of-thought answer that cites the oracle evidence. Standard RAG only *retrieves* context at inference time and relies on a general-purpose model to ignore noise. RAFT additionally *teaches* the model, during fine-tuning, to identify the correct evidence, ignore distractors, and produce grounded, cited answers — yielding a domain-specialized SLM that still composes with a retriever at serving time.

## Architecture

![End-to-end design of the SLM fine-tuning playbook: data preparation and versioned data asset, data exploration, Azure ML fine-tuning, registered candidate model, offline RAG evaluation, and gated deployment with drift monitoring, alongside the supporting Azure services.](docs/architecture-png.png)

The pipeline flows top to bottom through the five notebooks; each stage emits a versioned artifact that the next stage consumes. Supporting Azure services (Azure OpenAI, GPU compute with managed identity, Key Vault, AI Foundry base weights, MLflow, and drift monitoring) attach to the stages that use them. Source: [docs/architecture.drawio](docs/architecture.drawio).

## Notebooks

Run in order from the repository root. Each stage produces the inputs the next stage consumes.

| # | Notebook | Description | Outputs | Dependencies |
|---|----------|-------------|---------|--------------|
| 01 | [Data Preparation](notebooks/01_prepare_data.py) | Extracts PDF chunks, generates and validates local RAFT JSONL splits with grouped (leakage-free) train/validation/test partitioning, fingerprints the dataset, and optionally publishes an immutable Azure ML data asset. Synthetic generation is optional and billable. | Local `train`/`validation`/`test` JSONL splits, `manifest.json` (split counts, class balance, SHA-256 fingerprint), versioned Azure ML data asset. | Local PDF/text chunks; Azure OpenAI (only for synthetic generation); Azure ML workspace/datastore (only to publish). |
| 02 | [Data Exploration](notebooks/02_data_exploration.ipynb) | Explores the generated dataset: validates oracle/distractor balance and answer formatting, and analyzes CoT answer and instruction length distributions to choose `max_new_tokens` and `max_seq_length` for training. | Quality/balance summaries, length-distribution plots, computed columns persisted for training configuration. | Local RAFT JSONL splits from 01. |
| 03 | [Azure ML Fine-Tuning](notebooks/03_azureml_fine_tuning.ipynb) | Registers a pinned CUDA environment, submits a managed-identity SLM fine-tuning command job (`lib/train.py`) with PEFT/LoRA-style adaptation and prompt-token loss masking, tracks it with MLflow, writes a merged model to the datastore, and registers a candidate model. | Registered training environment, completed MLflow job run, merged model in the workspace datastore, registered candidate model version. | Versioned data asset from 01; GPU compute cluster with managed identity; base weights via Azure AI Foundry catalog or Hugging Face (Key Vault token). |
| 04 | [Offline Inference & RAG Evaluation](notebooks/04_azureml_offline_inference_evaluation.ipynb) | Runs local generation comparing the base Hugging Face model against the registered fine-tuned model on the same sealed test records, then scores relevancy, groundedness, and coherence with an Azure OpenAI judge via Azure AI Evaluation. | Row-level and aggregate comparison metrics, judge scores, qualitative failure inspection. | Registered candidate model from 03; sealed test split from 01; GPU host; Azure OpenAI judge deployment (billable). |
| 05 | [Deployment, Evaluation & Monitoring](notebooks/05_inference_evaluation_monitoring.ipynb) | Deploys the registered candidate to a managed online endpoint with zero traffic, runs smoke and held-out tests, applies a token-F1 + latency promotion gate, logs evidence, demonstrates SHAP on a declared scalar target, and schedules PSI request-drift detection. | Zero-traffic online deployment, evaluation evidence, promotion decision, SHAP diagnostic artifact, scheduled drift-monitoring job. | Registered candidate model from 03; sealed test split from 01; managed online endpoint. |

Prior exploratory and Kaggle notebooks are retained under `notebooks/legacy/` for historical reference. They are not part of the production workflow.

## Repository Layout

```text
.
├── notebooks/
│   ├── 01_prepare_data.py
│   ├── 02_data_exploration.ipynb
│   ├── 03_azureml_fine_tuning.ipynb
│   ├── 04_azureml_offline_inference_evaluation.ipynb
│   ├── 05_inference_evaluation_monitoring.ipynb
│   └── legacy/
├── lib/
│   ├── azureml_ops.py       # Jobs, assets, registration, deployment, promotion
│   ├── config.py            # Environment-driven workspace connection
│   ├── data.py              # Validation, grouped splits, fingerprint, publication
│   ├── evaluation.py        # Exact match and answer token-F1
│   ├── explainability.py    # SHAP adapter for declared scalar text targets
│   ├── inference.py         # Endpoint invocation and held-out evaluation
│   ├── monitoring.py        # Reference profiles and PSI drift
│   ├── monitor_drift.py     # Scheduled monitoring job entry point
│   ├── train.py             # Remote SLM finetuning command-job entry point
│   ├── score.py             # Managed endpoint scoring entry point
│   └── prompts.py           # Shared training/inference prompt contract
├── environments/            # Pinned Azure ML conda definitions
├── data/                    # Local source and RAFT data
├── output/                  # Historical local predictions
└── tests/                   # Fast governance and metric tests
```

## Prerequisites

- Python 3.11
- Azure CLI authenticated with `az login` for local work
- Azure ML workspace and GPU/CPU compute clusters
- Permissions to create Azure ML assets, jobs, endpoints, deployments, and schedules
- Managed identity on compute with least-privilege datastore access
- Azure Key Vault access for a gated Hugging Face model token
- Accepted license/access for `meta-llama/Llama-3.2-1B-Instruct`
- Azure OpenAI only when regenerating synthetic data from local documents

Install the notebook/control-plane environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-azureml.txt
az login
```

Set workspace identifiers in `.env` or the process environment:

```dotenv
AZURE_SUBSCRIPTION_ID=<subscription-id>
AZURE_RESOURCE_GROUP=<resource-group>
AZUREML_WORKSPACE_NAME=<workspace-name>

# Required only for synthetic data generation
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-08-01-preview
AZURE_OPENAI_GPT_DEPLOYMENT=gpt vision
```

>> Do not store credentials or tokens in `.env`, notebooks, job parameters, or MLflow. The training job can retrieve a Hugging Face token from Key Vault using managed identity.

## Data Preparation

Run the complete preparation pipeline from the repository root:

```powershell
python notebooks/01_prepare_data.py
```

Reuse existing chunks or keep the run local without publishing to Azure ML:

```powershell
python notebooks/01_prepare_data.py --skip-extract
python notebooks/01_prepare_data.py --skip-publish
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
- Evaluation: sealed test data, quality by RAFT type, groundedness, abstention, safety, fairness, latency, load, cost
- Deployment: zero-traffic candidate, private networking, token auth, quotas, staged rollout, rollback deployment
- Monitoring: request/response governance, error and latency alerts, outcome quality, PSI drift, safety, budget alerts
- Governance: model/data cards, approver, incident owner, retention, deletion, license and red-team evidence

SHAP is applied only to a declared scalar quality target. It does not provide a general causal explanation of free-form generation.

## Validation

Run fast local checks before submitting cloud jobs:

```powershell
python -m pytest tests -q
python -m compileall -q lib
```

Cloud resources are intentionally not created by tests. Execute notebook cloud cells only after replacing placeholder asset versions, endpoint names, compute names, and governed monitoring URIs.
