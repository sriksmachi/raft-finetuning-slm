"""Convert a PDF to per-page text chunks using GPT-4o vision.

Steps:
    1. Render each PDF page to a JPEG in data/images/<pdf-stem>/.
    2. Call GPT-4o vision on each image and save data/chunks/<pdf-stem>/.

Usage:
    python -m lib.pdf_to_chunks
    python -m lib.pdf_to_chunks path/to/document.pdf
"""

import base64
import logging
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

from azure.identity import AzureCliCredential, get_bearer_token_provider
from dotenv import load_dotenv
import fitz
from openai import AzureOpenAI
from tqdm import tqdm

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
endpoint_host = urlparse(azure_openai_endpoint).hostname or ""
token_scope = os.getenv("AZURE_OPENAI_TOKEN_SCOPE") or (
    "https://ai.azure.com/.default"
    if endpoint_host.endswith(".services.ai.azure.com")
    else "https://cognitiveservices.azure.com/.default"
)

_token_provider = get_bearer_token_provider(
    AzureCliCredential(),
    token_scope,
)

client = AzureOpenAI(
    azure_endpoint=azure_openai_endpoint,
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_ad_token_provider=_token_provider,
)

GPT4O_DEPLOYMENT = os.getenv("AZURE_OPENAI_GPT_DEPLOYMENT")


def pdf_to_images(
    pdf_path: str | Path,
    images_dir: str | Path = "./data/images",
) -> list[Path]:
    """Render each PDF page to a JPEG under images_dir/<stem>/."""
    pdf_path = Path(pdf_path)
    out_dir = Path(images_dir) / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Converting '%s' to page images in '%s'", pdf_path.name, out_dir)
    pdf = fitz.open(str(pdf_path))
    paths: list[Path] = []
    for page_number, page in enumerate(pdf, start=1):
        image_path = out_dir / f"page_{page_number:04d}.jpg"
        if not image_path.exists():
            pixmap = page.get_pixmap(dpi=150)
            pixmap.save(str(image_path))
        paths.append(image_path)
    pdf.close()

    log.info("Saved %d page image(s) to '%s'", len(paths), out_dir)
    return paths


def image_to_text(image_path: Path) -> str | None:
    """Extract Markdown text from an image using GPT-4o vision."""
    with image_path.open("rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode()

    try:
        response = client.chat.completions.create(
            model=GPT4O_DEPLOYMENT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Extract all text from this page image exactly as it appears. "
                                "Preserve headings, bullet points, and paragraph structure "
                                "using Markdown. Output only the extracted text, no commentary."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                }
            ],
        )
        return response.choices[0].message.content
    except Exception:
        log.exception("Failed to extract text from '%s'", image_path.name)
        return None


def extract_chunks(
    pdf_path: str | Path,
    images_dir: str | Path = "./data/images",
    chunks_dir: str | Path = "./data/chunks",
) -> list[Path]:
    """Render and extract one PDF into page-level text files."""
    image_paths = pdf_to_images(pdf_path, images_dir)
    out_dir = Path(chunks_dir) / Path(pdf_path).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Extracting text from %d page(s) via GPT-4o", len(image_paths))
    text_paths: list[Path] = []
    for image_path in tqdm(image_paths, desc="Pages"):
        text_path = out_dir / image_path.with_suffix(".txt").name
        if text_path.exists():
            log.debug("Skipping '%s' (already exists)", text_path.name)
            text_paths.append(text_path)
            continue
        text = image_to_text(image_path)
        if text:
            text_path.write_text(text, encoding="utf-8")
            text_paths.append(text_path)
            log.debug("Saved '%s'", text_path.name)
        else:
            log.warning("No text extracted from '%s'", image_path.name)

    log.info("Wrote %d chunk file(s) to '%s'", len(text_paths), out_dir)
    return text_paths


def main() -> int:
    """Extract one command-line PDF or every PDF in the data directory."""
    pdf_files = (
        [Path(sys.argv[1])]
        if len(sys.argv) > 1
        else sorted(Path("./data").glob("*.pdf"))
    )
    if not pdf_files:
        log.error("No PDF files found. Pass a path or place PDFs in ./data/")
        return 1

    log.info("Processing %d PDF file(s)", len(pdf_files))
    for pdf_path in pdf_files:
        extract_chunks(pdf_path)
    log.info("Done")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        log.exception("Fatal error; aborting")
        sys.exit(1)