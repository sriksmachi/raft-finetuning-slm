# =============================================================================
# RAFT & RAG Training Data Generation using GPT-4o
# =============================================================================
# Generates Question-Document-Answer triplets from a PDF for RAFT fine-tuning.
# Outputs train/validate/test JSONL files under ./data/training_data/
#
# -----------------------------------------------------------------------------
# How to run
# -----------------------------------------------------------------------------
# Prerequisites:
#   1. Activate the virtual environment:
#        .\.venv\Scripts\Activate.ps1   (Windows PowerShell)
#   2. Install dependencies:
#        pip install -r requirements.txt
#   3. Authenticate with Azure (used for Azure OpenAI access):
#        az login
#   4. Set the following environment variables (e.g. in a .env file):
#        AZURE_OPENAI_ENDPOINT
#        AZURE_OPENAI_API_VERSION
#        AZURE_OPENAI_GPT4O_DEPLOYMENT
#   5. Place the source PDF at:
#        ./data/instance-security-best-practice.pdf
#
# Usage:
#   python raft_datagen.py                       # full run (extract + generate)
#   python raft_datagen.py --skip-extract        # reuse existing chunks
#   python raft_datagen.py --num-questions 10    # 10 questions per chunk
#
# Output:
#   ./data/training_data_raft/raw/{train,validation,test}.jsonl
# -----------------------------------------------------------------------------

import logging
import logging.config
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

import concurrent.futures
import numpy as np
from datasets import Dataset
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import AzureOpenAI
from tqdm import tqdm
from azure.identity import AzureCliCredential

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Azure OpenAI client
# ---------------------------------------------------------------------------

# Fetch the token once on the main thread so parallel worker threads share a
# single cached credential — avoids concurrent az-cli subprocess timeouts.
_ad_token = AzureCliCredential().get_token(
    "https://cognitiveservices.azure.com/.default"
).token

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_ad_token=_ad_token,
)

gpt4o_deployment = os.getenv("AZURE_OPENAI_GPT4O_DEPLOYMENT")

log.info("GPT-4o deployment: %s", gpt4o_deployment)

# ---------------------------------------------------------------------------
# 1. Load and chunk domain-specific documents
# ---------------------------------------------------------------------------

def remove_special_characters(string: str) -> str:
    return re.sub(r'[^a-zA-Z0-9\s]', '', string)


def load_chunks_from_dir(
    chunks_dir: str | Path,
    chunk_size: int = 1024,
    chunk_overlap: int = 50,
) -> list[str]:
    """Read .txt files produced by pdf_to_chunks, apply RecursiveCharacterTextSplitter,
    and return filtered text chunks ready for RAFT dataset generation."""
    paths = sorted(Path(chunks_dir).rglob("*.txt"))
    page_texts = [p.read_text(encoding="utf-8") for p in paths if p.stat().st_size > 0]
    log.info("Loaded %d page file(s) from '%s'", len(page_texts), chunks_dir)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks: list[str] = []
    for text in page_texts:
        for split in splitter.split_text(text):
            if len(remove_special_characters(split)) > 100:
                chunks.append(split)
    log.info("%d chunk(s) after splitting and filtering", len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# 2. Generate training data
# ---------------------------------------------------------------------------

def strip_str(s: str) -> str:
    l, r = 0, len(s) - 1
    beg_found = False
    for i in range(len(s)):
        if s[i].isalpha():
            if not beg_found:
                l = i
                beg_found = True
            else:
                r = i
    r += 2
    return s[l:min(r, len(s))]


def generate_instructions_gen(chunk: Any, num_questions: int = 5) -> list[str]:
    log.debug("Generating %d questions for chunk (len=%d)", num_questions, len(str(chunk)))
    response = client.chat.completions.create(
        model=gpt4o_deployment,
        messages=[
            {"role": "system", "content": "You are a synthetic question-answer pair generator. Given a chunk of context about some topic(s), generate exactly %s example questions a user could ask and would be answered using information from the chunk. For example, if the given context was a Wikipedia paragraph about the United States, an example question could be 'How many states are in the United States?'" % (num_questions)},
            {"role": "system", "content": "The questions should be able to be answered in a few words or less. Include only the questions in your response."},
            {"role": "user", "content": str(chunk)}
        ]
    )
    queries = response.choices[0].message.content.split('\n')
    queries = [strip_str(q) for q in queries]
    queries = [q for q in queries if any(c.isalpha() for c in q)]
    queries = queries[:num_questions]
    log.debug("Generated %d question(s)", len(queries))
    return queries


def encode_question_gen(question: str, chunk: Any) -> list[str]:
    prompt = """
        Question: {question}\n Context: {context}\n
        Answer this question using the information given in the context above and no prior knowledge. Here is things to pay attention to: 
        - First provide step-by-step reasoning on how to answer the question. 
        - In the reasoning, if you need to copy paste some sentences from the context, include them in ##begin_quote## and ##end_quote##. This would mean that things outside of ##begin_quote## and ##end_quote## are not directly copy paste from the context. 
        - End your response with final answer in the form <ANSWER>: $answer, the answer should be given in a joyful and friendly tone.
        - If the answer cannot be found in the context, say "I'm sorry, I cannot answer this question as I'm missing the required information"
        You MUST begin your final answer with the tag "<ANSWER>:".
    """.format(question=question, context=str(chunk))
    return [
        {"role": "system", "content": "You are a helpful question answerer who can provide an answer given a question and relevant context."},
        {"role": "user", "content": prompt},
    ]


def generate_label(question: str, context: Any) -> str | None:
    log.debug("Generating answer for question: %.80s", question)
    messages = encode_question_gen(question, context)
    response = client.chat.completions.create(
        model=gpt4o_deployment,
        messages=messages,
        n=1,
        temperature=0
    )
    answer = response.choices[0].message.content
    log.debug("Answer generated (%d chars)", len(answer) if answer else 0)
    return answer


ds: Dataset = Dataset.from_dict({})
errors: list = []


def add_chunk_to_dataset(
    chunks: list[str],
    chunk: str,
    num_questions: int = 5,
    num_distract: int = 3,
    p: float = 0.8,
) -> None:
    global ds, errors
    i = chunks.index(chunk)
    log.debug("Processing chunk %d/%d", i + 1, len(chunks))
    try:
        qs = generate_instructions_gen(chunk, num_questions)
    except Exception as e:
        log.error("Failed to generate questions for chunk %d: %s", i, e, exc_info=True)
        errors.append(e)
        return None

    for q in qs:
        datapt = {
            "id": None,
            "type": None,
            "question": None,
            "context": None,
            "oracle_context": None,
            "cot_answer": None,
        }
        datapt["id"] = f"seed_task_{i}"
        datapt["type"] = "general"
        datapt["question"] = q
        docs = [chunk]
        indices = list(range(0, len(chunks)))
        indices.remove(i)
        # Add num_distract random chunks as distractors for this question.
        # These will be shuffled with the oracle chunk in the final context fed to the model, 
        # so the model can't just learn to pick the first one as the answer.
        for j in random.sample(indices, num_distract):
            docs.append(chunks[j])

        # With probability p, keep the oracle chunk as the first doc; otherwise replace it with a random distractor. 
        # This creates a mix of "easy" and "hard" examples for the model.
        oracle = random.uniform(0, 1) < p
        if not oracle:
            # Replace the oracle chunk with a random distractor.
            docs[0] = chunks[random.sample(indices, 1)[0]]
        random.shuffle(docs)

        d = {"title": [], "sentences": []}
        d["title"].append(["placeholder_title"] * (num_distract + 1))
        d["sentences"].append(docs)
        datapt["context"] = d
        datapt["oracle_context"] = chunk
        datapt["type"] = "oracle" if oracle else "distractor"

        try:
            # In the original RAFT methodology, when the "golden" (relevant) document is intentionally omitted from the training instance (leaving only distractors), 
            # the CoT answer should not say there is no context. Instead, it should provide the correct answer based on the model's internal knowledge
            # Here I'm using refutal approach. modify the training behavior for queries that are "out-of-bounds" or for which the retriever fails to find a high-confidence match
            # In these cases, instead of saying "I cannot answer this question as I'm missing the required information", 
            # the model will be trained to provide a best-effort answer based on its internal knowledge, while also acknowledging the lack of relevant context.
            datapt["cot_answer"] = generate_label(q, chunk)
        except Exception as e:
            log.error("Failed to generate answer for question '%s': %s", q, e, exc_info=True)
            errors.append(e)
            continue

        context = ""
        # Docs contain both oracle and distractors, but we want to wrap them in <DOCUMENT> tags to preserve chunk boundaries for the model.
        for doc in docs:
            context += "<DOCUMENT>" + str(doc) + "</DOCUMENT>\n"
        
        context += "\n### Question:\n" + q 
        
        datapt["instruction"] = context

        if not ds:
            datapt["id"] = [datapt["id"]]
            datapt["type"] = [datapt["type"]]
            datapt["question"] = [datapt["question"]]
            datapt["context"] = [datapt["context"]]
            datapt["oracle_context"] = [datapt["oracle_context"]]
            datapt["cot_answer"] = [datapt["cot_answer"]]
            datapt["instruction"] = [datapt["instruction"]]
            ds = Dataset.from_dict(datapt)
        else:
            ds = ds.add_item(datapt)


def generate_dataset(chunks: list[str], num_questions: int = 5) -> None:
    global ds, errors
    errors = []
    ds = Dataset.from_dict({})
    log.info("Starting dataset generation for %d chunks (num_questions=%d)", len(chunks), num_questions)

    def process_chunk(chunk):
        add_chunk_to_dataset(chunks, chunk, num_questions, 3)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
        with tqdm(total=len(chunks), desc="Processing chunks") as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    log.error("Unhandled exception in worker thread: %s", e, exc_info=True)
                pbar.update(1)

    log.info("Dataset generation complete. Errors: %d/%d. Examples: %d",
             len(errors), len(chunks), len(ds) if ds else 0)

# ---------------------------------------------------------------------------
# 3. Split and save
# ---------------------------------------------------------------------------

def save_datasets(training_df, output_dir: str = "./data/training_data_raft/raw") -> None:
    log.info("Splitting %d rows into train/validate/test sets", len(training_df))
    os.makedirs(output_dir, exist_ok=True)

    train_df, validate_df, test_df = np.split(
        training_df.sample(frac=1, random_state=42),
        [int(.8 * len(training_df)), int(.9 * len(training_df))]
    )
    log.info("Split sizes — Train: %d, Validate: %d, Test: %d",
             train_df.shape[0], validate_df.shape[0], test_df.shape[0])

    train_df.to_json(f"{output_dir}/train.jsonl", orient="records", lines=True)
    validate_df.to_json(f"{output_dir}/validation.jsonl", orient="records", lines=True)
    test_df.to_json(f"{output_dir}/test.jsonl", orient="records", lines=True)
    log.info("Saved train/validation/test datasets to '%s/'", output_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        import argparse
        parser = argparse.ArgumentParser(description="RAFT dataset generation")
        parser.add_argument("--skip-extract", action="store_true",
                            help="Skip PDF extraction, use existing chunks")
        parser.add_argument("--num-questions", type=int, default=5,
                            help="Number of questions to generate per chunk (default: 5)")
        args = parser.parse_args()

        skip_extract  = args.skip_extract
        num_questions = args.num_questions
        pdf_path      = "./data/instance-security-best-practice.pdf"
        pdf_stem      = Path(pdf_path).stem
        chunks_dir    = f"./data/chunks/{pdf_stem}"
        output_dir    = "./data/training_data_raft/raw"

        log.info("=" * 60)
        log.info("RAFT data generation started")
        log.info("=" * 60)
        log.info("Source PDF: %s", pdf_path)

        # Step 1 (optional): extract text chunks from PDF via pdf_to_chunks
        if not skip_extract:
            log.info("Extracting text chunks from PDF ...")
            import pdf_to_chunks
            pdf_to_chunks.extract_chunks(pdf_path)
        else:
            log.info("Skipping extraction (--skip-extract)")

        # Step 2: load and sub-chunk
        chunks = load_chunks_from_dir(chunks_dir)
        if not chunks:
            log.error("No chunks found in '%s'. Run without --skip-extract first.", chunks_dir)
            sys.exit(1)
        log.info("Loaded %d chunk(s)", len(chunks))

        # Step 3: Generate Q/A/D triplets
        log.info("Generating %d question(s) per chunk", num_questions)
        generate_dataset(chunks, num_questions=num_questions)

        # Step 4: Format and save
        training_df = ds.to_pandas()
        log.info("%d rows after dropping nulls", len(training_df))

        save_datasets(training_df, output_dir)

        log.info("=" * 60)
        log.info("RAFT data generation complete")
        log.info("=" * 60)
        sys.exit(0)
    except Exception:
        log.exception("Fatal error — aborting")
        sys.exit(1)
