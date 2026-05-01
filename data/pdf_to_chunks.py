"""Convert a PDF to per-page text chunks using GPT-4o vision.

Steps:
    1. Render each PDF page to a JPEG  →  data/images/<pdf-stem>/page_0001.jpg ...
    2. Call GPT-4o vision on each image →  data/chunks/<pdf-stem>/page_0001.txt ...

Usage:
    python pdf_to_chunks.py                       # process all PDFs in ./data/
    python pdf_to_chunks.py path/to/document.pdf  # single PDF
"""

import base64
import logging
import os
import sys
from pathlib import Path

from azure.identity import AzureCliCredential, get_bearer_token_provider
from dotenv import load_dotenv
import fitz  # pymupdf
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
# Steps
# ---------------------------------------------------------------------------

def pdf_to_images(
    pdf_path: str | Path,
    images_dir: str | Path = "./data/images",
) -> list[Path]:
    """Render each page of *pdf_path* to a JPEG under *images_dir/<stem>/*.

    Returns the list of saved image paths. Already-existing files are skipped.
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(images_dir) / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Converting '%s' → page images in '%s'", pdf_path.name, out_dir)
    pdf = fitz.open(str(pdf_path))
    paths: list[Path] = []
    for i, page in enumerate(pdf, start=1):
        img_path = out_dir / f"page_{i:04d}.jpg"
        if not img_path.exists():
            pix = page.get_pixmap(dpi=150)
            pix.save(str(img_path))
        paths.append(img_path)
    pdf.close()

    log.info("Saved %d page image(s) to '%s'", len(paths), out_dir)
    return paths


def image_to_text(image_path: Path) -> str | None:
    """Extract markdown text from *image_path* using GPT-4o vision.

    Returns the extracted text string, or None on failure.
    """
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

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
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
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
    """Full pipeline for one PDF: images → text chunks.

    Skips pages whose .txt output already exists.
    Returns the list of written .txt file paths.
    """
    image_paths = pdf_to_images(pdf_path, images_dir)
    out_dir = Path(chunks_dir) / Path(pdf_path).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Extracting text from %d page(s) via GPT-4o ...", len(image_paths))
    txt_paths: list[Path] = []
    for img_path in tqdm(image_paths, desc="Pages"):
        txt_path = out_dir / img_path.with_suffix(".txt").name
        if txt_path.exists():
            log.debug("Skipping '%s' (already exists)", txt_path.name)
            txt_paths.append(txt_path)
            continue
        text = image_to_text(img_path)
        if text:
            txt_path.write_text(text, encoding="utf-8")
            txt_paths.append(txt_path)
            log.debug("Saved '%s'", txt_path.name)
        else:
            log.warning("No text extracted from '%s'", img_path.name)

    log.info("Wrote %d chunk file(s) to '%s'", len(txt_paths), out_dir)
    return txt_paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        pdf_files = [Path(sys.argv[1])] if len(sys.argv) > 1 else sorted(Path("./data").glob("*.pdf"))
        if not pdf_files:
            log.error("No PDF files found. Pass a path or place PDFs in ./data/")
            sys.exit(1)

        log.info("Processing %d PDF file(s) ...", len(pdf_files))
        for pdf in pdf_files:
            extract_chunks(pdf)

        log.info("Done.")
        sys.exit(0)
    except Exception:
        log.exception("Fatal error — aborting")
        sys.exit(1)
