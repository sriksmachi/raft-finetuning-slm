# RAFT Fine-Tuning for Small Language Models

End-to-end pipeline: PDFs → RAFT dataset → fine-tuned SLM → RAG evaluation

Powered by **Azure OpenAI (GPT-4o)**, **Unsloth + LoRA**, and **RAGAS**

---

## Why RAFT?

- **Standard SFT** trains on clean oracle context — brittle when retrieval is noisy
- **RAFT** (Retrieval-Augmented Fine-Tuning) mixes the oracle chunk with **distractor** chunks
- With probability `p = 0.2` the oracle is *removed entirely*, teaching the model to recognize when context is insufficient
- Result: a **small** model (Llama-3.2-1B) that is robust inside a real RAG pipeline

---

## Pipeline Architecture

| Step | Script | Output |
|------|--------|--------|
| 1. Extract | `pdf_to_chunks.py` | `data/chunks/<pdf>/page_NNNN.txt` (GPT-4o vision OCR) |
| 2. Generate | `raft_datagen.py` | `train.jsonl` / `validation.jsonl` / `test.jsonl` (80/10/10) |
| 3. Fine-tune | `raft-finetuning-slm.ipynb` | LoRA adapter on Llama-3.2-1B-Instruct → HF Hub |
| 4. Evaluate | `rag_evaluate.py` | RAGAS scores: GPT-4o vs fine-tuned SLM |

Each Q&A example contains: `question`, chain-of-thought `cot_answer` with `##begin_quote##` citations, and a shuffled context of **1 oracle + 3 distractors**.

---

## Tech Stack

- **Azure OpenAI GPT-4o** — vision OCR, Q&A synthesis, RAGAS judge (auth via `AzureCliCredential`, no API keys)
- **PyMuPDF** — self-contained PDF → image rendering
- **Unsloth + LoRA** — 4-bit quantized fine-tuning of Llama-3.2-1B on a single T4/A10
- **RAGAS** — `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`
- **HuggingFace `datasets`** — JSONL train/val/test splits
- Parallel chunk processing with `concurrent.futures` for fast dataset generation

---

## Results & Takeaways

- A **1B-parameter** model fine-tuned with RAFT can approach GPT-4o quality on **domain-specific** RAG tasks
- Distractor-aware training meaningfully improves `faithfulness` and `context_precision`
- Fully reproducible: `python pdf_to_chunks.py` → `python raft_datagen.py` → notebook → `python rag_evaluate.py`
- Evaluation results land in `data/llama_rag_eval_results.csv` for side-by-side comparison

**Next:** swap in your own PDFs, tune `--num-questions`, and publish your adapter to the Hub.
