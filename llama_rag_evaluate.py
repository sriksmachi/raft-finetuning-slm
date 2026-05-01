"""Evaluate RAG responses using LlamaIndex evaluators.

This script:
1. Loads RAFT-style JSONL records.
2. Generates candidate answers using Azure OpenAI through LlamaIndex.
3. Scores answers using LlamaIndex evaluators:
   - Faithfulness
   - Relevancy
   - Correctness

Default dataset points to the filtered validation split:
  data/training_data_raft/filtered/validation.jsonl
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
from azure.identity import AzureCliCredential
from dotenv import load_dotenv
from tqdm import tqdm

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.evaluation import (
    CorrectnessEvaluator,
    FaithfulnessEvaluator,
    RelevancyEvaluator,
)
from llama_index.llms.azure_openai import AzureOpenAI

# Keep this in sync with the system prompt used to fine-tune the SLM.
_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using the provided context. "
    "You should Answer ### Question STRICTLY in this FORMAT: "
    "### Step-by-step reasoning: Use several quotes from <Retrieved Documents>: "
    "##begin_quote## [Relevant text 1] ##end_quote## "
    "##begin_quote## [Relevant text 2] ##end_quote## "
    "Then think step-by-step. <ANSWER>A/B/C/D</ANSWER>"
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Quiet noisy per-request HTTP logs from httpx (used by openai/azure clients).
logging.getLogger("httpx").setLevel(logging.WARNING)

# Fetch token once and reuse, matching raft_datagen.py's CLI auth pattern.
_ad_token = AzureCliCredential().get_token(
    "https://cognitiveservices.azure.com/.default"
).token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RAG quality with LlamaIndex evaluators"
    )
    parser.add_argument(
        "--dataset",
        default="./data/training_data_raft/filtered/validation.jsonl",
        help="Path to input JSONL file",
    )
    parser.add_argument(
        "--response-model",
        default=os.getenv("AZURE_OPENAI_GPT41_DEPLOYMENT", "gpt-4.1"),
        help="Azure OpenAI deployment name for generating responses",
    )
    parser.add_argument(
        "--eval-model",
        default=os.getenv("AZURE_OPENAI_GPT40_DEPLOYMENT", "gpt-4o"),
        help="Azure OpenAI deployment name for evaluation",
    )
    parser.add_argument(
        "--response-model-name",
        default=os.getenv("AZURE_OPENAI_RESPONSE_MODEL_NAME", "gpt-4o"),
        help="OpenAI model identifier for tokenization (must be recognized by llama-index, e.g. gpt-4o)",
    )
    parser.add_argument(
        "--eval-model-name",
        default=os.getenv("AZURE_OPENAI_EVAL_MODEL_NAME", "gpt-4o"),
        help="OpenAI model identifier for tokenization (must be recognized by llama-index, e.g. gpt-4o)",
    )
    parser.add_argument(
        "--output",
        default="./data/llama_rag_eval_results.csv",
        help="Path to output CSV",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of records",
    )
    return parser.parse_args()


def build_llm(deployment: str, model_name: str) -> AzureOpenAI:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    if not endpoint or not api_version:
        raise ValueError(
            "Missing AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_VERSION in environment."
        )

    return AzureOpenAI(
        model=model_name,
        deployment_name=deployment,
        api_key=_ad_token,
        azure_endpoint=endpoint,
        api_version=api_version,
        use_azure_ad=True,
        azure_ad_token_provider=lambda: _ad_token,
        temperature=0,
    )


def load_jsonl(path: str, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if limit is not None:
        records = records[:limit]

    log.info("Loaded %d record(s) from %s", len(records), path)
    return records


def extract_context_text(record: dict[str, Any]) -> str:
    instruction = record.get("instruction")
    if isinstance(instruction, str):
        return instruction
    return ""

def reference_answer(record: dict[str, Any]) -> str:
    val = record.get("cot_answer")
    if isinstance(val, str) and val.strip():
        return val
    return ""

def generate_answer(llm: AzureOpenAI, question: str, context: str) -> str:
    # Mirror the chat-style format used during SLM fine-tuning:
    #   system  -> _SYSTEM_PROMPT
    #   user    -> "<Retrieved Documents>: \n{instruction}"  (context + question)
    #   assistant -> CoT answer ending with <ANSWER>...</ANSWER>
    user_content = (
        f"<Retrieved Documents>: \n{context}\n\n" # this already has the question included
    )
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=_SYSTEM_PROMPT),
        ChatMessage(role=MessageRole.USER, content=user_content),
    ]
    return llm.chat(messages).message.content.strip()


def evaluate_one(
    faithfulness: FaithfulnessEvaluator,
    relevancy: RelevancyEvaluator,
    correctness: CorrectnessEvaluator,
    question: str,
    answer: str,
    context: str,
    reference: str,
) -> dict[str, Any]:
    faith = faithfulness.evaluate(
        query=question,
        response=answer,
        contexts=[context],
    )
    rel = relevancy.evaluate(
        query=question,
        response=answer,
        contexts=[context],
    )
    corr = correctness.evaluate(
        query=question,
        response=answer,
        reference=reference,
    )

    return {
        "faithfulness": faith.score,
        "relevancy": rel.score,
        "correctness": corr.score,
        "faithfulness_passing": faith.passing,
        "relevancy_passing": rel.passing,
        "correctness_passing": corr.passing,
        "faithfulness_feedback": faith.feedback,
        "relevancy_feedback": rel.feedback,
        "correctness_feedback": corr.feedback,
    }


def _process_record(
    r: dict[str, Any],
    response_llm: AzureOpenAI,
    faithfulness: FaithfulnessEvaluator,
    relevancy: RelevancyEvaluator,
    correctness: CorrectnessEvaluator,
) -> dict[str, Any] | None:
    question = str(r.get("question", "")).strip()
    if not question:
        return None

    context = extract_context_text(r)
    ref = reference_answer(r)
    model_answer = generate_answer(response_llm, question, context)

    eval_result = evaluate_one(
        faithfulness=faithfulness,
        relevancy=relevancy,
        correctness=correctness,
        question=question,
        answer=model_answer,
        context=context,
        reference=ref,
    )

    return {
        "question": question,
        "model_answer": model_answer,
        "reference_answer": ref,
        **eval_result,
    }


def run(args: argparse.Namespace) -> None:
    records = load_jsonl(args.dataset, args.limit)
    if not records:
        raise ValueError("No records found in dataset.")
    log.info("Loaded %d record(s) from %s", len(records), args.dataset)
    log.info(
        "Building response LLM (deployment=%s, model=%s)",
        args.response_model,
        args.response_model_name,
    )
    response_llm = build_llm(args.response_model, args.response_model_name)
    log.info(
        "Building eval LLM (deployment=%s, model=%s)",
        args.eval_model,
        args.eval_model_name,
    )
    eval_llm = build_llm(args.eval_model, args.eval_model_name)

    faithfulness = FaithfulnessEvaluator(llm=eval_llm)
    relevancy = RelevancyEvaluator(llm=eval_llm)
    correctness = CorrectnessEvaluator(llm=eval_llm)

    rows: list[dict[str, Any]] = []

    log.info("Evaluating %d record(s) sequentially", len(records))
    for r in tqdm(records, desc="Evaluating", unit="sample"):
        try:
            row = _process_record(
                r,
                response_llm,
                faithfulness,
                relevancy,
                correctness,
            )
        except Exception as e:
            log.error("Record failed: %s", e, exc_info=True)
            continue
        if row is not None:
            rows.append(row)

    if not rows:
        raise ValueError("No rows evaluated. Check dataset format.")

    df = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    summary = {
        "faithfulness_avg": float(df["faithfulness"].dropna().mean()),
        "relevancy_avg": float(df["relevancy"].dropna().mean()),
        "correctness_avg": float(df["correctness"].dropna().mean()),
        "faithfulness_pass_rate": float(df["faithfulness_passing"].mean()),
        "relevancy_pass_rate": float(df["relevancy_passing"].mean()),
        "correctness_pass_rate": float(df["correctness_passing"].mean()),
    }

    print("\nLlamaIndex RAG Evaluation Summary")
    for k, v in summary.items():
        print(f"- {k}: {v:.4f}")
    print(f"\nSaved detailed results to: {out_path}")


if __name__ == "__main__":
    cli_args = parse_args()
    run(cli_args)
