"""Generate RAFT predictions with an Azure OpenAI GPT deployment.

This script loads RAFT-style JSONL records, generates answers with the same
chat prompt used for GPT evaluation today, and stores predictions under the
output folder for a separate LlamaIndex evaluation step.
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
from llama_index.llms.azure_openai import AzureOpenAI

_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using the provided context. "
    "DO NOT use any information that is not included in the <Retrieved Documents>. "
    "You should Answer ### Question STRICTLY in this FORMAT: "
    "### Step-by-step reasoning: Use several quotes from <Retrieved Documents>: "
    "##begin_quote## [Relevant text 1] ##end_quote## "
    "##begin_quote## [Relevant text 2] ##end_quote## "
    "Then think step-by-step. <ANSWER></ANSWER>"
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

_ad_token = AzureCliCredential().get_token(
    "https://cognitiveservices.azure.com/.default"
).token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate RAFT predictions with an Azure OpenAI GPT deployment"
    )
    parser.add_argument(
        "--dataset",
        default="./data/training_data_raft/test.jsonl",
        help="Path to input RAFT JSONL file",
    )
    parser.add_argument(
        "--response-model",
        default=os.getenv("AZURE_OPENAI_GPT_DEPLOYMENT", "gpt-4.1"),
        help="Azure OpenAI deployment name for generating responses",
    )
    parser.add_argument(
        "--response-model-name",
        default=os.getenv("AZURE_OPENAI_RESPONSE_MODEL_NAME", "gpt-4.1"),
        help="OpenAI model identifier for tokenization, e.g. gpt-4.1",
    )
    parser.add_argument(
        "--output",
        default="./output/llm_predictions.csv",
        help="Path to predictions CSV",
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
    with open(path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
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
    value = record.get("cot_answer")
    if isinstance(value, str) and value.strip():
        return value
    return ""


def record_type(record: dict[str, Any]) -> str:
    value = record.get("type")
    if isinstance(value, str) and value.strip():
        return value
    return ""


def record_id(record: dict[str, Any]) -> str:
    value = record.get("id")
    if value is None:
        return ""
    return str(value)


def generate_answer(llm: AzureOpenAI, context: str) -> str:
    user_content = f"<Retrieved Documents>: \n{context}\n\n"
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=_SYSTEM_PROMPT),
        ChatMessage(role=MessageRole.USER, content=user_content),
    ]
    response = llm.chat(messages).message.content
    return response.strip() if response else ""


def process_record(
    record: dict[str, Any],
    response_llm: AzureOpenAI,
    model_name: str,
) -> dict[str, Any] | None:
    question = str(record.get("question", "")).strip()
    if not question:
        return None

    context = extract_context_text(record)
    model_answer = generate_answer(response_llm, context)
    return {
        "model_name": model_name,
        "id": record_id(record),
        "type": record_type(record),
        "question": question,
        "context": context,
        "model_answer": model_answer,
        "reference_answer": reference_answer(record),
    }


def run(args: argparse.Namespace) -> None:
    records = load_jsonl(args.dataset, args.limit)
    if not records:
        raise ValueError("No records found in dataset.")

    log.info(
        "Building response LLM (deployment=%s, model=%s)",
        args.response_model,
        args.response_model_name,
    )
    response_llm = build_llm(args.response_model, args.response_model_name)

    rows: list[dict[str, Any]] = []
    log.info("Generating responses for %d record(s)", len(records))
    for record in tqdm(records, desc="Generating", unit="sample"):
        try:
            row = process_record(record, response_llm, args.response_model)
        except Exception as exc:
            log.error("Record failed: %s", exc, exc_info=True)
            continue
        if row is not None:
            rows.append(row)

    if not rows:
        raise ValueError("No predictions generated. Check dataset format.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved {len(rows)} prediction(s) to: {output_path}")


if __name__ == "__main__":
    cli_args = parse_args()
    run(cli_args)
