# RAFT / RAG Fine-Tuning for Small Language Models

End-to-end pipeline for generating fine-tuning datasets from domain-specific PDF documents and training a small language model with RAFT (Retrieval-Augmented Fine-Tuning) methodology using Azure OpenAI and Hugging Face.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Environment Setup](#environment-setup)
5. [Pipeline Walkthrough](#pipeline-walkthrough)
   - [Step 1 — Extract text chunks from PDFs](#step-1--extract-text-chunks-from-pdfs)
   - [Step 2 — Generate RAG Q&A dataset](#step-2--generate-rag-qa-dataset)
   - [Step 3 — Generate RAFT Q&A/D dataset](#step-3--generate-raft-qad-dataset)
   - [Step 4 — Fine-tune the model](#step-4--fine-tune-the-model)
   - [Step 5 — Evaluate with RAGAS](#step-5--evaluate-with-ragas)
6. [File Reference](#file-reference)
7. [CLI Reference](#cli-reference)
8. [Dataset Schemas](#dataset-schemas)
9. [Troubleshooting](#troubleshooting)

---

## Overview

**RAFT (Retrieval-Augmented Fine-Tuning)** trains a language model to identify the correct answer from a noisy retrieved context that contains both a relevant *oracle* passage and several *distractor* documents. This makes the model more robust in production RAG pipelines compared to standard SFT on clean context.

This project provides:

| Script | Purpose |
|--------|---------|
| `pdf_to_chunks.py` | Convert PDFs → per-page JPEG images → `.txt` chunk files via GPT-4o vision |
| `rag_datagen.py` | Generate a RAG Q&A dataset (question / answer / context) from chunks |
| `raft_datagen.py` | Generate a RAFT Q&A/D triplet dataset (question / cot_answer / distractor context) |
| `rag_evaluate.py` | Evaluate answer quality across models (GPT-4o vs fine-tuned HF model) using RAGAS |
| `raft-finetuning-slm.ipynb` | Fine-tune Llama-3.2-1B-Instruct on the RAFT dataset using Unsloth + LoRA |

---

## Architecture

```
PDF documents (data/*.pdf)
        │
        ▼
┌─────────────────────┐
│   pdf_to_chunks.py  │  GPT-4o vision → per-page .txt files
│   (Step 1)          │  Output: data/chunks/<pdf-stem>/page_NNNN.txt
└────────┬────────────┘
         │
         ├──────────────────────────────────┐
         ▼                                  ▼
┌─────────────────────┐          ┌──────────────────────┐
│   rag_datagen.py    │          │   raft_datagen.py    │
│   (Step 2)          │          │   (Step 3)           │
│                     │          │                      │
│  GPT-4o JSON mode   │          │  GPT-4o generates    │
│  5 Q&A pairs/chunk  │          │  questions + CoT     │
│  parallel workers   │          │  answers + distractors│
│                     │          │  parallel workers    │
│  Output:            │          │                      │
│  data/training_data/│          │  Output:             │
│  train.jsonl        │          │  data/training_data_ │
│  test.jsonl         │          │  raft/{train,         │
└─────────────────────┘          │  validation,test}.   │
                                 │  jsonl               │
         ▼                       └──────────┬───────────┘
┌─────────────────────┐                     │
│   rag_evaluate.py   │                     ▼
│   (Step 5)          │          ┌──────────────────────┐
│                     │          │raft-finetuning-slm   │
│  RAGAS metrics:     │          │.ipynb (Step 4)       │
│  - faithfulness     │          │                      │
│  - answer_relevancy │          │  Unsloth + LoRA      │
│  - context_precision│          │  Llama-3.2-1B-Instruct│
│  - context_recall   │◄─────────│  → HF Hub            │
│                     │          └──────────────────────┘
│  Output:            │
│  data/eval_results  │
│  .csv               │
└─────────────────────┘
```

---

## Prerequisites

- Python 3.11+
- [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli) installed and logged in (`az login`)
- Azure OpenAI resource with:
  - A **GPT-4o** deployment (used for text extraction, Q&A generation, and RAGAS judging)
  - An **embedding** deployment (optional, for `answer_relevancy` metric; falls back to GPT-4o)
- GPU with ≥16 GB VRAM for fine-tuning (Kaggle T4 / A10 recommended)
- `pymupdf` requires no system binaries — PDF rendering is fully self-contained

---

## Environment Setup

### 1. Clone and create virtual environment

```bash
git clone <your-repo>
cd raft-finetuning-slm
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```dotenv
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_GPT4O_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large   # optional
```

> **Note:** API key authentication is not used. All scripts authenticate via `AzureCliCredential` — run `az login` once before executing any script.

### 3. Verify Azure login

```bash
az account show
```

---

## Pipeline Walkthrough

### Step 1 — Extract text chunks from PDFs

Converts each PDF page to a JPEG image and calls GPT-4o vision to extract the visible text. Outputs one `.txt` file per page.

```bash
# Process all PDFs in ./data/
python pdf_to_chunks.py

# Process a specific PDF
python pdf_to_chunks.py path/to/document.pdf
```

**Output structure:**

```
data/
  images/<pdf-stem>/page_0001.jpg  ...
  chunks/<pdf-stem>/page_0001.txt  ...
```

Existing files are skipped automatically — safe to re-run after adding new PDFs.

---

### Step 2 — Generate RAG Q&A dataset

Reads `.txt` chunk files and calls GPT-4o in JSON mode to produce question/answer/context triples. Workers run in parallel (8 threads by default).

```bash
# Full pipeline (extract + generate)
python rag_datagen.py

# Skip extraction if chunks already exist
python rag_datagen.py --skip-extract
```

**Output:** `data/training_data/train.jsonl` and `test.jsonl` (80/20 split).

---

### Step 3 — Generate RAFT Q&A/D dataset

Extends the RAG dataset with RAFT-specific distractor documents. For each chunk, GPT-4o generates:
- `num_questions` questions
- A chain-of-thought answer with `##begin_quote##` citations and a final `<ANSWER>:` tag
- Context containing the oracle chunk + 3 randomly sampled distractor chunks (shuffled)

With probability `p=0.8` the oracle is included; with `p=0.2` it is replaced — training robustness for cases where retrieval fails.

```bash
# Full pipeline (extract + generate, 5 questions per chunk)
python raft_datagen.py

# Skip extraction, generate 3 questions per chunk
python raft_datagen.py --skip-extract --num-questions 3

# All options
python raft_datagen.py --help
```

**Output:** `data/training_data_raft/train.jsonl`, `validation.jsonl`, `test.jsonl` (80/10/10 split).

---

### Step 4 — Fine-tune the model

Open `raft-finetuning-slm.ipynb` (designed for Kaggle GPU environment).

The notebook:
1. Loads the RAFT JSONL dataset
2. Loads `unsloth/Llama-3.2-1B-Instruct` in 4-bit NF4 quantisation
3. Attaches LoRA adapters (`r=16`) to all attention and MLP projections
4. Formats examples with the Llama-3.2 chat template (including oracle + distractor context)
5. Trains for 1 epoch with `trl.SFTTrainer`
6. Merges adapters into the base model and saves in 16-bit
7. Pushes to Hugging Face Hub

**Key hyperparameters:**

| Parameter | Value |
|-----------|-------|
| `max_seq_length` | 2048 |
| `load_in_4bit` | True |
| LoRA rank `r` | 16 |
| `learning_rate` | 2e-5 |
| `num_train_epochs` | 1 |
| Effective batch size | 16 (2 × 8 accumulation steps) |

---

### Step 5 — Evaluate with RAGAS

Evaluates answer quality across models using four [RAGAS](https://docs.ragas.io) metrics. GPT-4o acts as the judge LLM for all evaluations.

```bash
# Evaluate both GPT-4o and the fine-tuned HF model
python rag_evaluate.py

# GPT-4o only, limit to 50 records for a quick check
python rag_evaluate.py --models gpt4o --limit 50

# Fine-tuned Llama model only
python rag_evaluate.py --models hf --hf-model sriksmachi/llama32_1bn_instruct_raft

# Evaluate on RAFT test split instead of RAG
python rag_evaluate.py --dataset data/training_data_raft/test.jsonl

# All options
python rag_evaluate.py --help
```

**RAGAS Metrics:**

| Metric | What it measures |
|--------|-----------------|
| `faithfulness` | Answer is grounded in the retrieved context (no hallucination) |
| `answer_relevancy` | Answer directly addresses the question |
| `context_precision` | Retrieved context is relevant to the question |
| `context_recall` | Context contains sufficient information to answer |

**Output:** Console table + `data/eval_results.csv`

---

## File Reference

```
raft-finetuning-slm/
├── pdf_to_chunks.py          # PDF → images → .txt chunks (GPT-4o vision)
├── rag_datagen.py            # RAG Q&A dataset generator
├── raft_datagen.py           # RAFT Q&A/D triplet dataset generator
├── rag_evaluate.py           # RAGAS evaluation across models
├── raft-finetuning-slm.ipynb # Unsloth LoRA fine-tuning notebook (Kaggle)
├── data_exploration.ipynb    # Dataset exploration notebook
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (not committed)
└── data/
    ├── *.pdf                 # Source domain documents
    ├── images/               # Rendered page images (pdf_to_chunks output)
    ├── chunks/               # Extracted text chunks (pdf_to_chunks output)
    ├── training_data/        # RAG Q&A dataset (rag_datagen output)
    └── training_data_raft/   # RAFT Q&A/D dataset (raft_datagen output)
```

---

## CLI Reference

### `pdf_to_chunks.py`

```
python pdf_to_chunks.py [pdf_path]

Arguments:
  pdf_path    (optional) Path to a specific PDF. Defaults to all PDFs in ./data/
```

### `rag_datagen.py`

```
python rag_datagen.py [--skip-extract]

Options:
  --skip-extract    Skip PDF extraction, use existing chunk files
```

### `raft_datagen.py`

```
python raft_datagen.py [--skip-extract] [--num-questions N]

Options:
  --skip-extract        Skip PDF extraction, use existing chunk files
  --num-questions N     Questions to generate per chunk (default: 5)
```

### `rag_evaluate.py`

```
python rag_evaluate.py [options]

Options:
  --models {gpt4o,hf} [{gpt4o,hf} ...]
                        Models to evaluate (default: gpt4o hf)
  --hf-model HF_MODEL   HF Hub model ID or local path (default: sriksmachi/llama32_1bn_instruct_raft)
  --dataset PATH        JSONL file to evaluate on (default: data/training_data/test.jsonl)
  --limit N             Cap evaluation at N records
  --output PATH         Results CSV output path (default: data/eval_results.csv)
```

---

## Dataset Schemas

### RAG dataset (`data/training_data/*.jsonl`)

```json
{
  "question": "What is the recommended patch cycle for OS packages?",
  "answer":   "Monthly patching is recommended, with critical patches applied within 48 hours.",
  "context":  "OS packages should be patched on a monthly cycle..."
}
```

### RAFT dataset (`data/training_data_raft/*.jsonl`)

```json
{
  "id":             "seed_task_12",
  "type":           "general",
  "question":       "What is the recommended patch cycle for OS packages?",
  "context":        { "title": [...], "sentences": [[oracle_chunk, distractor_1, distractor_2, distractor_3]] },
  "oracle_context": "OS packages should be patched on a monthly cycle...",
  "cot_answer":     "##begin_quote## OS packages should be patched... ##end_quote##\n<ANSWER>: Monthly patching...",
  "instruction":    "<DOCUMENT>...</DOCUMENT>\n<DOCUMENT>...</DOCUMENT>\n...\n### Question:\n..."
}
```

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `AzureCliCredential.get_token failed` | `az login` session expired or `az` CLI not on PATH | Run `az login` in the same terminal |
| `az.cmd timed out after 10 seconds` | Multiple parallel threads each spawning `az.cmd` | Already mitigated — token is fetched once on the main thread before workers start |
| `KeyError: 'messages'` | `save_datasets` called before `messages` column built | Fixed in current code — column is constructed from `instruction`/`cot_answer` before saving |
| `PDFInfoNotInstalledError` | Old `pdf2image` dependency requiring poppler | Replaced with `pymupdf` — no system binaries required |
| `httpx proxies` error | `httpx>=0.28` removed the `proxies` kwarg used by older openai SDK | Pin `httpx>=0.23.0,<0.28.0` (already in `requirements.txt`) |
| HF model OOM on CPU | Large model loaded without quantisation | Pass `--hf-model` pointing to a 4-bit GPTQ/GGUF version, or run on GPU |
