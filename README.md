# 🧪 SLM Studio

**Turn domain documents into evaluated, release-ready small language model candidates.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](environments/train-conda.yml)
[![Azure Machine Learning](https://img.shields.io/badge/Azure-Machine%20Learning-0078D4)](https://azure.microsoft.com/products/machine-learning)
[![Built with FastAPI](https://img.shields.io/badge/Built%20with-FastAPI-009688?logo=fastapi&logoColor=white)](app/main.py)

SLM Studio is a local web application and Python workflow for preparing
document-grounded training data, fine-tuning small language models (SLMs),
comparing them with their original model and a GPT baseline, and registering
or deploying candidates on Azure Machine Learning after explicit approval.

[Getting started](#installation-and-quick-start) ·
[Step-by-step usage guide](docs/how_to_use.md) ·
[Architecture](docs/architecture-png.png)

## 🎯 The problem

General-purpose models are not automatically reliable on an organization's
specialized documents. Frontier-model APIs can also introduce latency and
usage costs that make a smaller, domain-adapted model worth investigating.
But fine-tuning alone does not demonstrate that a smaller model is better,
close enough to a GPT baseline, or ready to serve.

SLM Studio connects that experiment into one traceable workflow: create
grounded examples from PDFs, train a candidate, measure quality and latency
on a sealed test split, and require evidence and human approval before release.
It helps determine **whether** an SLM is suitable for a workload rather than
assuming that fine-tuning guarantees quality or savings.

## 🧭 Key objectives

- **Ground adaptation in your documents.** Generate context-grounded RAFT
  (Retrieval-Augmented Fine-Tuning) examples from PDFs.
- **Make experiments traceable.** Preserve versioned data, training settings,
  Azure job references, and evaluation lineage.
- **Measure before releasing.** Compare the fine-tuned SLM, original SLM, and
  GPT using faithfulness, relevancy, correctness, latency, and token usage.
- **Enforce evidence-based release.** Apply fixed quality gates and explicit
  human approval rather than automatically deploying every trained model.
- **Make costs visible.** Surface billable operations, configured frontier-token
  estimates, and deployment GPU estimates before approval.

## 🔄 From documents to deployment

**PDFs → Grounded dataset → Fine-tuned SLM → Comparative evaluation → Approved candidate → Optional endpoint**

| Stage | What you do | Outputs you can produce |
| --- | --- | --- |
| 📚 Prepare data | Extract PDF content and generate grounded examples | Validated train/validation/test JSONL splits, dataset fingerprint, and a versioned Azure ML data asset |
| 🛠️ Fine-tune | Select a base model and configure LoRA training | Azure ML training jobs, adapter artifacts, logs, and training lineage |
| 📊 Evaluate | Test the candidate against the original SLM and GPT | Quality and latency metrics, response-level comparisons, token usage, and configured API cost estimates |
| ✅ Register | Review release-policy evidence and approve a candidate | A registered Azure ML model linked to its evaluation evidence |
| 🚀 Deploy | Review and approve a dedicated deployment plan | An authenticated Azure ML endpoint, scoring validation, and monitoring configuration |

The Studio exposes four tabs: **Data**, **Fine-tuning**, **Evaluation**, and
**Candidate release**. Registration and optional deployment share the last tab;
the command-line workflow provides five separate stage scripts.

### Release is gated, not automatic

Candidates need at least **30 sealed-test examples**. Each quality metric must
show at least **2 percentage points** of mean uplift over the original SLM,
strictly **less than 5%** relative shortfall from GPT, and a paired-bootstrap
**95% uplift interval strictly above zero**. A narrowly defined perfect-baseline
tie exception applies; see the [complete release rules](docs/how_to_use.md#5-release-a-candidate).
Latency and token costs are informational, not release gates.

## Installation and quick start

### Prerequisites

- **Git and Python 3.12**; Azure ML environment definitions use Python 3.12.10.
- **Azure CLI** for signing in.
- For production operations: an **Azure ML workspace**, an **Azure OpenAI GPT
  deployment**, and permissions to use the required resources.
- For training/evaluation: dedicated **Azure ML GPU compute**, managed identity,
  and suitable quota. Review backend compute and identity settings before
  submission; local installation does not provision these prerequisites.
- For deployment: regional GPU capacity, an existing CPU monitoring cluster,
  and a governed production-request folder.

### 1. Clone and install

Run these commands in **Windows PowerShell**:

```powershell
git clone https://github.com/sriksmachi/raft-finetuning-slm.git
Set-Location raft-finetuning-slm
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r app\requirements.txt
```

Only opening the interface? Install [app/requirements.txt](app/requirements.txt)
alone instead. The production stages also need [requirements.txt](requirements.txt).
GPU training and inference run in Azure ML environments, not in the local web app.

### 2. Configure Azure

```powershell
az login
```

Create a local `.env` file in the repository root with your resource values:

```dotenv
AZURE_SUBSCRIPTION_ID=<subscription-id>
AZURE_RESOURCE_GROUP=<resource-group>
AZUREML_WORKSPACE_NAME=<workspace-name>
AZURE_OPENAI_ENDPOINT=https://<resource-name>.openai.azure.com/
AZURE_OPENAI_API_VERSION=<supported-api-version>
AZURE_OPENAI_GPT_DEPLOYMENT=<gpt-deployment-name>
```

Keep credentials out of source control. Follow the
[Azure access and compute guidance](docs/how_to_use.md#2-prepare-data) before
submitting work. An optional `AZURE_OPENAI_JUDGE_DEPLOYMENT` selects a separate
evaluation judge.

### 3. Launch Studio

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open **<http://127.0.0.1:8000>**.

### 4. Run your first workflow

1. **Data:** select a local PDF folder and prepare a context-grounded RAFT dataset.
2. **Fine-tuning:** select the published dataset and base model, review trainer
   settings, and submit training.
3. **Evaluation:** select the completed training job and run the sealed-test
   comparison against the original SLM and GPT.
4. **Candidate release:** inspect the policy results and lineage, then explicitly
   approve registration only or registration with deployment.

Use the [stage-by-stage guide](docs/how_to_use.md) for detailed instructions,
monitoring, retries, token-price configuration, and command-line alternatives.

> ⚠️ **Real services, real costs.** Data generation and judging call Azure OpenAI;
> training, evaluation, and deployment use billable Azure resources. Allocated
> deployment GPUs keep accruing charges. Cancelling local work does not guarantee
> remote jobs stop or resources are deleted.
>
> 🔒 **Local control plane.** Studio has no multi-user authentication. Keep it
> bound to localhost, run one production operation at a time, and do not expose
> the app as a public service.

## 📁 Repository map

| Location | Purpose |
| --- | --- |
| [app/](app/) | FastAPI web app, workflow APIs, frontend, and app tests |
| [lib/](lib/) | Shared data, training, evaluation, Azure ML, and release logic |
| [scripts/](scripts/) | Five command-line workflow stages and supporting utilities |
| [environments/](environments/) | Azure ML training and inference environment definitions |
| [notebooks/](notebooks/) | Notebook workflows |
| [docs/](docs/) | [Usage guide](docs/how_to_use.md) and architecture assets |

Local run records and evidence are saved under `runs/production/`.

## 🧪 Development checks

Run the fast, credential-free production integration checks from the repository root:

```powershell
python -m pytest app\tests\test_production_api.py app\tests\test_training_api.py app\tests\test_evaluation_api.py app\tests\test_candidate_api.py -q
```

See [installation checks](docs/how_to_use.md#7-check-the-installation) for the full
app test suite and optional Node.js UI polling tests.

## 📄 License

SLM Studio is distributed under the [MIT License](LICENSE).
