"""Generate a RAG Q&A dataset from text chunks produced by pdf_to_chunks.py.

Steps:
    1. Read .txt chunk files from data/chunks/
    2. Call GPT-4o to produce Q&A pairs for each chunk
    3. 80/20 train/test split → data/training_data/train.jsonl + test.jsonl

Usage:
    python rag_datagen.py                 # end-to-end (runs pdf_to_chunks if needed)
    python rag_datagen.py --skip-extract  # skip extraction, use existing chunks
"""

import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from azure.identity import AzureCliCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import AzureOpenAI
from sklearn.model_selection import train_test_split
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
# Azure OpenAI client
# ---------------------------------------------------------------------------

_token_provider = get_bearer_token_provider(
    AzureCliCredential(),
    "https://cognitiveservices.azure.com/.default",
)

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_ad_token_provider=_token_provider,
)

GPT4O_DEPLOYMENT = os.getenv("AZURE_OPENAI_GPT4O_DEPLOYMENT")

# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def load_chunks(chunks_dir: str | Path) -> list[tuple[str, str]]:
    """Return (filename, text) for every non-empty .txt file under *chunks_dir*."""
    paths = sorted(Path(chunks_dir).rglob("*.txt"))
    chunks = [(p.name, p.read_text(encoding="utf-8")) for p in paths if p.stat().st_size > 0]
    log.info("Loaded %d chunk(s) from '%s'", len(chunks), chunks_dir)
    return chunks


def generate_qa_for_chunk(text: str, num_questions: int = 5) -> list[tuple[str, str]]:
    """Call GPT-4o to produce *num_questions* Q&A pairs from *text*.

    Returns a list of (question, answer) tuples.
    """
    prompt = (
        f"You are an expert teacher creating exam questions.\n"
        f"Given the context below, generate exactly {num_questions} question-answer pairs.\n"
        f"Return ONLY valid JSON with a top-level key 'pairs' containing an array of objects "
        f"with keys 'question' and 'answer'. No other text.\n\n"
        f"Context:\n{text}"
    )
    try:
        response = client.chat.completions.create(
            model=GPT4O_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        # Accept {"pairs": [...]} or any single-key wrapper
        pairs = data.get("pairs") or next(iter(data.values()), [])
        return [
            (p["question"].strip(), p["answer"].strip(), text.strip())
            for p in pairs
            if "question" in p and "answer" in p
        ]
    except Exception:
        log.exception("Failed to generate Q&A (chunk preview: %.80s)", text)
        return []


def build_dataset(chunks: list[tuple[str, str]], num_questions: int = 5, max_workers: int = 8) -> list[dict]:
    """Generate Q&A pairs for every chunk in parallel. Returns chat-format records."""
    records: list[dict] = []

    def _process(name: str, text: str) -> list[dict]:
        log.info("Generating Q&A for '%s' ...", name)
        pairs = generate_qa_for_chunk(text, num_questions=num_questions)
        log.debug("  → %d pair(s) for '%s'", len(pairs), name)
        return [
            {"question": question, "answer": answer, "context": context}
            for question, answer, context in pairs
        ]

    futures = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for name, text in chunks:
            futures[executor.submit(_process, name, text)] = name

        with tqdm(total=len(futures), desc="Generating Q&A", unit="chunk") as pbar:
            for future in as_completed(futures):
                records.extend(future.result())
                pbar.update(1)

    log.info("Built %d Q&A record(s) from %d chunk(s)", len(records), len(chunks))
    return records


def save_datasets(
    records: list[dict],
    output_dir: str | Path = "./data/training_data_rag",
    test_size: float = 0.2,
) -> None:
    """Split *records* 80/20 and write train.jsonl + test.jsonl to *output_dir*."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train, test = train_test_split(records, test_size=test_size, random_state=42, shuffle=True)
    log.info("Split: train=%d, test=%d", len(train), len(test))

    for split_name, split_data in (("train", train), ("test", test)):
        out_path = output_dir / f"{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in split_data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log.info("Saved %d record(s) to '%s'", len(split_data), out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        skip_extract  = "--skip-extract" in sys.argv
        chunks_dir    = "./data/chunks"
        output_dir    = "./data/training_data_rag"
        num_questions = 5

        log.info("=" * 60)
        log.info("RAG dataset generation started")
        log.info("=" * 60)

        # Step 1 (optional): run pdf_to_chunks when no chunks exist yet
        if not skip_extract and not list(Path(chunks_dir).rglob("*.txt")):
            log.info("No chunks found — running pdf_to_chunks first ...")
            import pdf_to_chunks
            for pdf in sorted(Path("./data").glob("*.pdf")):
                pdf_to_chunks.extract_chunks(pdf)

        # Step 2: load chunks
        chunks = load_chunks(chunks_dir)
        if not chunks:
            log.error("No .txt chunks found in '%s'. Run pdf_to_chunks.py first.", chunks_dir)
            sys.exit(1)

        # Step 3: generate Q&A records
        records = build_dataset(chunks, num_questions=num_questions)
        if not records:
            log.error("No Q&A pairs generated.")
            sys.exit(1)

        # Step 4: save
        save_datasets(records, output_dir=output_dir)

        log.info("=" * 60)
        log.info("RAG dataset generation complete")
        log.info("=" * 60)
        sys.exit(0)
    except Exception:
        log.exception("Fatal error — aborting")
        sys.exit(1)