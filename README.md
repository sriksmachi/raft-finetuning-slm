# RAFT Fine-Tuning on Azure Machine Learning

A notebook-led, production-oriented workflow for adapting a small language model to domain-specific, context-grounded question answering with Retrieval-Augmented Fine-Tuning (RAFT).

The active workflow uses Azure Machine Learning data, environment, job, model, endpoint, and schedule assets. Reusable implementation code lives in `lib/`; notebooks explain the data-science decisions and orchestrate those components.

## Architecture

```text
Local PDF/text chunks
        |
        v
01 Data Preparation
  schema + leakage checks + fingerprint
        |
        v
Azure ML versioned data asset (workspace datastore)
        |
        v
02 Azure ML Fine-Tuning
  Finetuning command job + MLflow + named model output
        |
        v
Azure ML registered candidate model
        |
        v
03 Inference, Evaluation, and Monitoring
  zero-traffic deployment -> held-out gate -> promotion
  SHAP diagnostic + request drift schedule
```

## Notebooks

Run in order from the repository root:

1. [Data Preparation](notebooks/01_prepare_data.py) extracts PDF chunks, generates and validates local RAFT JSONL splits, and publishes an immutable Azure ML data asset. Synthetic generation is optional and billable.
2. [Azure ML Fine-Tuning](notebooks/02_azureml_fine_tuning.ipynb) registers a pinned CUDA environment, submits a managed-identity SLM Finetuning job, tracks it with MLflow, stores the merged model in Azure Storage, and registers a candidate model.
3. [Inference, Evaluation, and Monitoring](notebooks/03_inference_evaluation_monitoring.ipynb) deploys the registered version with zero traffic, runs smoke and held-out tests, applies a promotion gate, logs evaluation evidence, demonstrates target-specific SHAP, and defines scheduled PSI drift detection.

Prior exploratory and Kaggle notebooks are retained under `notebooks/legacy/` for historical reference. They are not part of the production workflow.

## Repository Layout

```text
.
├── notebooks/
│   ├── 01_prepare_data.py
│   ├── 02_azureml_fine_tuning.ipynb
│   ├── 03_inference_evaluation_monitoring.ipynb
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

Do not store credentials or tokens in `.env`, notebooks, job parameters, or MLflow. The training job can retrieve a Hugging Face token from Key Vault using managed identity.

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
