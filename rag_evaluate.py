"""Evaluate RAG answer quality across multiple models using RAGAS metrics.

RAGAS Metrics
-------------
- faithfulness        : Is the answer supported by the retrieved context?
- answer_relevancy    : Is the answer actually addressing the question?
- context_precision   : Is the retrieved context ranked/relevant to the question?
- context_recall      : Does the context contain enough info to answer the question?

Models supported
----------------
  gpt4o   — Azure OpenAI GPT-4o (AzureCliCredential auth)
  hf      — Any HuggingFace causal LM loaded locally (e.g. Unsloth fine-tuned Llama-3.2)

Usage
-----
  python rag_evaluate.py                                       # evaluate both models
  python rag_evaluate.py --models gpt4o                       # GPT-4o only
  python rag_evaluate.py --models hf                          # HF model only
  python rag_evaluate.py --hf-model sriksmachi/llama32_1bn_instruct_raft
  python rag_evaluate.py --dataset data/training_data/test.jsonl --limit 50
  python rag_evaluate.py --output results/eval.csv

Environment variables (from .env)
----------------------------------
  AZURE_OPENAI_ENDPOINT              Azure OpenAI resource endpoint
  AZURE_OPENAI_API_VERSION           API version (e.g. 2024-02-01)
  AZURE_OPENAI_GPT4O_DEPLOYMENT      Deployment name for GPT-4o
  AZURE_OPENAI_EMBEDDING_DEPLOYMENT  Deployment name for text-embedding model
                                     (used by answer_relevancy metric;
                                      falls back to GPT4O_DEPLOYMENT if unset)
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from azure.identity import AzureCliCredential, get_bearer_token_provider
from datasets import Dataset as HFDataset
from dotenv import load_dotenv
from openai import AzureOpenAI
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from tqdm import tqdm

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------

AZURE_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
GPT4O_DEPLOYMENT  = os.getenv("AZURE_OPENAI_GPT4O_DEPLOYMENT")
EMBED_DEPLOYMENT  = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") or GPT4O_DEPLOYMENT

RAGAS_METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]

# System prompt used consistently across all model inference calls
_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using only the information "
    "provided in the context. Be concise and accurate."
)

# ---------------------------------------------------------------------------
# Azure credential (created once; shared by all clients)
# ---------------------------------------------------------------------------

_credential      = AzureCliCredential()
_token_provider  = get_bearer_token_provider(
    _credential,
    "https://cognitiveservices.azure.com/.default",
)


def _make_openai_client() -> AzureOpenAI:
    """AzureOpenAI client with a freshly fetched token (avoids concurrent az-cli spawns)."""
    token = _credential.get_token("https://cognitiveservices.azure.com/.default").token
    return AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_version=AZURE_API_VERSION,
        azure_ad_token=token,
    )


def _make_langchain_llm():
    """LangChain AzureChatOpenAI wrapper used as the RAGAS judge LLM."""
    from langchain_openai import AzureChatOpenAI

    return AzureChatOpenAI(
        azure_deployment=GPT4O_DEPLOYMENT,
        azure_endpoint=AZURE_ENDPOINT,
        api_version=AZURE_API_VERSION,
        azure_ad_token_provider=_token_provider,
        temperature=0,
    )


def _make_langchain_embeddings():
    """LangChain AzureOpenAIEmbeddings used by the answer_relevancy metric."""
    from langchain_openai import AzureOpenAIEmbeddings

    return AzureOpenAIEmbeddings(
        azure_deployment=EMBED_DEPLOYMENT,
        azure_endpoint=AZURE_ENDPOINT,
        api_version=AZURE_API_VERSION,
        azure_ad_token_provider=_token_provider,
    )


# ---------------------------------------------------------------------------
# Model inference helpers
# ---------------------------------------------------------------------------

def _answer_with_gpt4o(client: AzureOpenAI, question: str, context: str) -> str:
    """Generate an answer using GPT-4o from a question + context pair."""
    response = client.chat.completions.create(
        model=GPT4O_DEPLOYMENT,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def _answer_with_hf_model(pipeline, question: str, context: str) -> str:
    """Generate an answer using a HuggingFace text-generation pipeline."""
    messages = [
        {"role": "system",    "content": _SYSTEM_PROMPT},
        {"role": "user",      "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    # Use the tokenizer's chat template if available, else fall back to manual formatting
    try:
        prompt = pipeline.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt = (
            f"<|system|>\n{_SYSTEM_PROMPT}\n"
            f"<|user|>\nContext:\n{context}\n\nQuestion: {question}\n"
            "<|assistant|>\n"
        )

    out = pipeline(prompt, max_new_tokens=256, do_sample=False, return_full_text=False)
    return out[0]["generated_text"].strip()


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_eval_records(path: str, limit: int | None = None) -> list[dict]:
    """Load RAG evaluation records from a JSONL file.

    Each record must contain 'question', 'answer', and 'context' keys
    (the schema produced by rag_datagen.py).
    """
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if limit:
        records = records[:limit]
    log.info("Loaded %d evaluation record(s) from '%s'", len(records), path)
    return records


# ---------------------------------------------------------------------------
# RAGAS evaluation
# ---------------------------------------------------------------------------

def run_ragas_evaluation(
    records: list[dict],
    model_answers: list[str],
    judge_llm,
    embeddings,
    model_label: str,
) -> dict[str, float]:
    """Build a RAGAS HF Dataset, configure metrics, and return per-metric scores.

    Parameters
    ----------
    records       : Raw evaluation records (question / answer / context)
    model_answers : Answers generated by the candidate model
    judge_llm     : LangChain LLM wrapper used as RAGAS judge
    embeddings    : LangChain embeddings used by answer_relevancy
    model_label   : Display name for logging
    """
    log.info("Running RAGAS evaluation for: %s", model_label)

    ragas_data = {
        "question":      [r["question"] for r in records],
        "answer":        model_answers,
        # RAGAS expects a list-of-lists for contexts (multiple retrieved docs supported)
        "contexts":      [[r["context"]] for r in records],
        # ground_truths drives context_recall; wrap in list per ragas convention
        "ground_truths": [[r["answer"]] for r in records],
    }
    dataset = HFDataset.from_dict(ragas_data)

    wrapped_llm = LangchainLLMWrapper(judge_llm)
    for metric in RAGAS_METRICS:
        metric.llm = wrapped_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = embeddings

    result = evaluate(dataset, metrics=RAGAS_METRICS)
    scores = {k: round(float(v), 4) for k, v in result.items()}
    log.info("Scores for %-40s %s", model_label, scores)
    return scores


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate RAG answer quality across models using RAGAS"
    )
    p.add_argument(
        "--models",
        nargs="+",
        choices=["gpt4o", "hf"],
        default=["gpt4o", "hf"],
        help="Which models to evaluate (default: both)",
    )
    p.add_argument(
        "--hf-model",
        default="sriksmachi/llama32_1bn_instruct_raft",
        help="HuggingFace Hub model ID or local path for the 'hf' model",
    )
    p.add_argument(
        "--dataset",
        default="./data/training_data/test.jsonl",
        help="Path to the JSONL evaluation file (default: RAG test split)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap evaluation at N records (useful for quick sanity checks)",
    )
    p.add_argument(
        "--output",
        default="./data/eval_results.csv",
        help="Path to write the results CSV (default: data/eval_results.csv)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    log.info("=" * 60)
    log.info("RAG evaluation started")
    log.info("Models : %s", args.models)
    log.info("Dataset: %s  (limit=%s)", args.dataset, args.limit)
    log.info("=" * 60)

    try:
        # ── 1. Load evaluation records ──────────────────────────────────────
        records = load_eval_records(args.dataset, limit=args.limit)
        if not records:
            log.error("No records loaded from '%s'", args.dataset)
            sys.exit(1)

        # ── 2. Initialise shared RAGAS judge (Azure GPT-4o) ─────────────────
        log.info("Initialising RAGAS judge LLM and embeddings ...")
        judge_llm  = _make_langchain_llm()
        embeddings = _make_langchain_embeddings()

        all_results: dict[str, dict] = {}

        # ── 3. GPT-4o inference + evaluation ────────────────────────────────
        if "gpt4o" in args.models:
            log.info("GPT-4o inference ...")
            oai_client = _make_openai_client()
            gpt4o_answers = [
                _answer_with_gpt4o(oai_client, r["question"], r["context"])
                for r in tqdm(records, desc="GPT-4o", unit="q")
            ]
            all_results["gpt4o"] = run_ragas_evaluation(
                records, gpt4o_answers, judge_llm, embeddings, model_label="gpt4o"
            )

        # ── 4. HuggingFace model inference + evaluation ──────────────────────
        if "hf" in args.models:
            log.info("Loading HF model '%s' ...", args.hf_model)
            import torch
            from transformers import pipeline as hf_pipeline

            pipe = hf_pipeline(
                "text-generation",
                model=args.hf_model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )
            hf_answers = [
                _answer_with_hf_model(pipe, r["question"], r["context"])
                for r in tqdm(records, desc=f"HF ({args.hf_model})", unit="q")
            ]
            all_results[args.hf_model] = run_ragas_evaluation(
                records, hf_answers, judge_llm, embeddings, model_label=args.hf_model
            )

        # ── 5. Display and save comparison table ────────────────────────────
        if all_results:
            results_df = pd.DataFrame(all_results).T
            results_df.index.name = "model"

            print("\n" + "=" * 60)
            print("RAGAS Evaluation Results")
            print("=" * 60)
            print(results_df.to_string())
            print()

            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(out_path)
            log.info("Results saved to '%s'", out_path)

        log.info("=" * 60)
        log.info("RAG evaluation complete")
        log.info("=" * 60)
        sys.exit(0)

    except Exception:
        log.exception("Fatal error — aborting")
        sys.exit(1)
