# RAFT Fine-Tuning for Small Language Models

End-to-end pipeline for adapting a small language model (Llama-3.2-1B-Instruct) to a domain using **RAFT (Retrieval-Augmented Fine-Tuning)**, then comparing it head-to-head against a frontier model (GPT-4.1) on the same evaluation set.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Environment Setup](#environment-setup)
5. [Pipeline Walkthrough](#pipeline-walkthrough)
   - [Stage 1 — Generate the RAFT dataset (GPT-4o)](#stage-1--generate-the-raft-dataset-gpt-4o)
   - [Stage 2 — Fine-tune the SLM on the RAFT dataset](#stage-2--fine-tune-the-slm-on-the-raft-dataset)
   - [Stage 3 — Evaluate the fine-tuned SLM (Kaggle)](#stage-3--evaluate-the-fine-tuned-slm-kaggle)
   - [Stage 4 — Generate and evaluate GPT-4.1 on the same set](#stage-4--generate-and-evaluate-gpt-41-on-the-same-set)
   - [Stage 5 — Final comparison summary](#stage-5--final-comparison-summary)
6. [File Reference](#file-reference)
7. [Dataset Schema](#dataset-schema)
8. [Troubleshooting](#troubleshooting)

---

## Overview

**RAFT (Retrieval-Augmented Fine-Tuning)** trains a language model to answer from a noisy retrieved context that contains both the relevant *oracle* passage and several *distractor* documents. This makes the model robust to imperfect retrieval at inference time — a 1B SLM fine-tuned this way can rival much larger frontier models on a narrow domain at a fraction of the cost.

This repository contains five stages, end to end:

| Stage | Artifact | Purpose |
|-------|----------|---------|
| 1 | [raft_datagen.py](raft_datagen.py) | Generate Q / CoT-Answer / Distractor-Context triplets from PDFs using GPT-4o |
| 2 | [raft-finetuning-slm.ipynb](raft-finetuning-slm.ipynb) | Fine-tune Llama-3.2-1B-Instruct on the RAFT dataset (Unsloth + LoRA, on Kaggle) |
| 3 | [raft-evaluate-finetuned.ipynb](raft-evaluate-finetuned.ipynb) | Run the fine-tuned SLM on the held-out test split and score it (Kaggle) |
| 4 | [RAFT_Inference.py](RAFT_Inference.py) + [RAFT_LLAMA_Evaluate.py](RAFT_LLAMA_Evaluate.py) | Run GPT-4.1 on the same held-out test split, save predictions, then score them |
| 5 | Notebook / CSVs | Compare metrics from stages 3 and 4 side-by-side |

**Reference:** *RAFT: Adapting Language Model to Domain Specific RAG* (Zhang et al., 2024).

---

## Architecture

```
PDF documents (data/*.pdf)
        │
        ▼
┌─────────────────────────┐
│  pdf_to_chunks.py       │   GPT-4o vision → per-page .txt chunks
└────────┬────────────────┘
         ▼
┌─────────────────────────┐
│  Stage 1                │   GPT-4o (JSON)
│  raft_datagen.py        │   → train.jsonl / validation.jsonl / test.jsonl
└────────┬────────────────┘
         ▼
┌─────────────────────────┐
│  Stage 2                │   Unsloth + LoRA on Kaggle GPU
│  raft-finetuning-slm    │   → fine-tuned Llama-3.2-1B on HF Hub
│  .ipynb                 │
└────────┬────────────────┘
         ▼
┌─────────────────────────┐         ┌─────────────────────────┐
│  Stage 3                │         │  Stage 4                │
│  raft-evaluate-         │         │  RAFT_Inference.py      │
│  finetuned.ipynb        │         │  RAFT_LLAMA_Evaluate.py │
│  (fine-tuned SLM)       │         │  GPT-4.1 predictions    │
│  scored by GPT-4o judge │         │  scored by GPT-4o judge │
└────────┬────────────────┘         └────────┬────────────────┘
         │                                   │
         └────────────┬──────────────────────┘
                      ▼
            ┌─────────────────────┐
            │  Stage 5            │
            │  Comparison summary │
            │  (metrics CSVs)     │
            └─────────────────────┘
```

All evaluations use the **same** test split (`data/training_data_raft/test.jsonl`) and the **same** judge LLM (GPT-4o via LlamaIndex `Faithfulness`, `Relevancy`, `Correctness`) — so the SLM and GPT-4.1 scores are directly comparable.

---

## Prerequisites

- Python 3.11+
- Azure CLI (`az login`) — auth for all Azure OpenAI calls
- Azure OpenAI resource with:
  - A **GPT-4o** deployment (used for chunk extraction, RAFT generation, and as the judge LLM)
  - A **GPT-4.1** deployment (used as the response model in stage 4)
- A Kaggle account with GPU enabled (T4 / P100 / A10) — for stages 2 and 3
- A Hugging Face account with a write token — to push and pull the fine-tuned adapter

---

## Environment Setup

```powershell
git clone <your-repo>
cd raft-finetuning-slm
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```dotenv
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-08-01-preview
AZURE_OPENAI_GPT4O_DEPLOYMENT=gpt-4o
AZURE_OPENAI_GPT41_DEPLOYMENT=gpt-4.1
```

Authentication uses `AzureCliCredential` — run `az login` once before any script.

---

## Pipeline Walkthrough

### Stage 1 — Generate the RAFT dataset (GPT-4o)

Extracts text chunks from PDFs, then asks GPT-4o to generate, per chunk:

- `num_questions` synthetic questions
- A chain-of-thought answer with `##begin_quote## … ##end_quote##` citations and a final `<ANSWER>:` tag
- Context = oracle chunk + 3 randomly sampled distractor chunks (shuffled)

With probability `p=0.8` the oracle chunk is included; with `p=0.2` it is replaced with a distractor — teaching the model to refuse when retrieval fails.

```powershell
# Full pipeline (PDF extraction + Q/A/D generation, 5 questions per chunk)
python raft_datagen.py

# Skip extraction, generate 3 questions per chunk
python raft_datagen.py --skip-extract --num-questions 3
```

**Output:** `data/training_data_raft/{train,validation,test}.jsonl` (80 / 10 / 10 split).

---

### Stage 2 — Fine-tune the SLM on the RAFT dataset

Open [raft-finetuning-slm.ipynb](raft-finetuning-slm.ipynb) on Kaggle (GPU enabled).

The notebook:

1. Loads the RAFT JSONL splits
2. Loads `unsloth/Llama-3.2-1B-Instruct` in 4-bit NF4
3. Attaches LoRA adapters (`r=8`) to all attention + MLP projections
4. Formats examples with the Llama-3.2 chat template (system + `<Retrieved Documents>` + CoT answer)
5. Trains for 1 epoch with `trl.SFTTrainer` using `train_on_responses_only`
6. Merges adapters into the base model in 16-bit
7. Pushes to Hugging Face Hub (default: `sriksmachi/llama32_1bn_instruct_raft`)

| Hyperparameter | Value |
|----------------|-------|
| `max_seq_length` | 2048 |
| `load_in_4bit` | True |
| LoRA rank `r` | 8 |
| `learning_rate` | 2e-5 |
| `num_train_epochs` | 1 |
| Effective batch size | 32 (2 × 16 accumulation) |

---

### Stage 3 — Evaluate the fine-tuned SLM (Kaggle)

Open [raft-evaluate-finetuned.ipynb](raft-evaluate-finetuned.ipynb) on Kaggle (GPU enabled).

The notebook:

1. Loads the fine-tuned model from Hugging Face Hub via Unsloth (4-bit)
2. Loads `data/training_data_raft/test.jsonl` (upload as a Kaggle dataset, or pull from your repo)
3. Generates an answer for each test record using the same chat template / system prompt used at training time
4. Saves predictions to `slm_predictions.csv`
5. (Optional) Scores predictions with LlamaIndex `Faithfulness`, `Relevancy`, `Correctness` — judge LLM = Azure OpenAI GPT-4o (credentials supplied via Kaggle Secrets)
6. Writes per-sample scores to `slm_eval_results.csv` plus a printed summary

Download `slm_eval_results.csv` from Kaggle outputs for stage 5.

---

### Stage 4 — Generate and evaluate GPT-4.1 on the same set

Run locally. First generate GPT-4.1 responses and store them in `output/`, then evaluate those saved responses with LlamaIndex evaluators and the GPT-4o judge:

```powershell
# Generate GPT-4.1 predictions on the held-out RAFT test split
python raft_inference.py `
  --dataset ./data/training_data_raft/test.jsonl `
  --response-model gpt-4.1 `
  --output ./output/raft_predictions.csv

# Evaluate saved predictions with LlamaIndex RAG evaluators
python raft_llama_evaluate.py `
  --predictions ./output/raft_predictions.csv `
  --eval-model gpt-4o `
  --output ./output/raft_llama_eval_results.csv
```

**Outputs:** `./output/raft_predictions.csv` for generated responses, including `model_name` and `type` columns, and `./output/raft_llama_eval_results.csv` with the same evaluation columns as the SLM CSV.

---

### Stage 5 — Final comparison summary

Once both CSVs are available, compare the means side-by-side:

```powershell
python -c "import pandas as pd; slm = pd.read_csv('data/slm_eval_results.csv'); gpt = pd.read_csv('output/raft_llama_eval_results.csv'); cols = ['faithfulness','relevancy','correctness']; print(pd.DataFrame({'fine_tuned_slm': slm[cols].mean(), 'gpt_4_1': gpt[cols].mean()}).round(3))"
```

You'll get something like:

```
              fine_tuned_slm  gpt_4_1
faithfulness           0.93     0.96
relevancy              0.95     0.97
correctness            4.10     4.55
```

**Interpretation:** the 1B fine-tuned SLM closes most of the gap with GPT-4.1 on a domain-specific RAG task — at orders-of-magnitude lower inference cost. Use this table as the headline result of the project.

---

## File Reference

```
raft-finetuning-slm/
├── pdf_to_chunks.py                  # PDF → JPEG → .txt chunks (GPT-4o vision)
├── raft_datagen.py                   # Stage 1 — RAFT Q/A/D dataset generator
├── raft-finetuning-slm.ipynb         # Stage 2 — Unsloth + LoRA fine-tuning (Kaggle)
├── raft-evaluate-finetuned.ipynb     # Stage 3 — Fine-tuned SLM evaluation (Kaggle)
├── RAFT_Inference.py                 # Stage 4 — GPT-4.1 prediction generation
├── RAFT_LLAMA_Evaluate.py            # Stage 4 — saved prediction evaluation (LlamaIndex)
├── data_exploration.ipynb            # Dataset inspection helper
├── requirements.txt
├── .env                              # not committed
└── data/
    ├── *.pdf                         # Source domain documents
    ├── images/                       # Rendered page images
    ├── chunks/                       # Extracted text chunks
    └── training_data_raft/
        ├── train.jsonl
        ├── validation.jsonl
        └── test.jsonl                # held-out split — used by stages 3 + 4
```

---

## Dataset Schema

`data/training_data_raft/*.jsonl`

```json
{
  "id":             "seed_task_12",
  "type":           "general",
  "question":       "What is the recommended patch cycle for OS packages?",
  "context":        { "title": [...], "sentences": [[oracle, distractor_1, distractor_2, distractor_3]] },
  "oracle_context": "OS packages should be patched on a monthly cycle ...",
  "cot_answer":     "##begin_quote## OS packages should be patched ... ##end_quote##\n<ANSWER>: Monthly ...",
  "instruction":    "<DOCUMENT>...</DOCUMENT>\n<DOCUMENT>...</DOCUMENT>\n...\n### Question:\n..."
}
```

The `instruction` field is what the model sees at training and inference time (context + question). The `cot_answer` is the supervised target.

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `AzureCliCredential.get_token failed` | `az login` session expired | Re-run `az login` in the same shell |
| `az.cmd timed out after 10 seconds` | Multiple threads each spawning `az.cmd` | Already mitigated — token is fetched once on the main thread |
| `PDFInfoNotInstalledError` | Old `pdf2image` / poppler dependency | Replaced with `pymupdf` — no system binaries required |
| `httpx proxies` error | `httpx>=0.28` removed `proxies` kwarg | Pin `httpx>=0.23,<0.28` (already in `requirements.txt`) |
| Kaggle OOM loading model | Trying to load in fp16 instead of 4-bit | Keep `load_in_4bit=True` in stage 3 notebook |
