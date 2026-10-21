"""Prepare and publish the SLM Studio context-grounded dataset.

Extract PDF chunks, generate the context-grounded dataset, validate it, and optionally
publish a versioned Azure ML data asset. Run these commands from the repository
root.

Examples:
    python scripts/01_prepare_data.py                     # full pipeline
    python scripts/01_prepare_data.py --skip-extract      # reuse chunks
    python scripts/01_prepare_data.py --skip-generation   # validate + publish
    python scripts/01_prepare_data.py --skip-publish      # local-only run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lib.config import AzureMLConfig
from lib.data import (
    dataset_fingerprint,
    publish_data_asset,
    validate_data_asset_version,
    validate_dataset,
)

log = logging.getLogger("prepare_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path,
                        default=PROJECT_ROOT / "data" / "instance-security-best-practice.pdf")
    parser.add_argument("--images-dir", type=Path,
                        default=PROJECT_ROOT / "data" / "images")
    parser.add_argument("--chunks-dir", type=Path,
                        default=PROJECT_ROOT / "data" / "chunks")
    parser.add_argument("--dataset-dir", type=Path,
                        default=PROJECT_ROOT / "data" / "training_data_raft")
    parser.add_argument("--num-questions", type=int, default=5)
    parser.add_argument("--num-samples", type=int, help="Exact total samples across all splits (at least 3)")
    parser.add_argument("--source-folder", type=Path, help="Extract every PDF recursively in this folder instead of --pdf")
    parser.add_argument("--oracle-probability", type=float, default=0.8)
    parser.add_argument("--skip-extract", action="store_true",
                        help="Reuse existing chunk .txt files")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip PDF extraction and dataset generation")
    parser.add_argument("--skip-publish", action="store_true",
                        help="Skip Azure ML data asset publication")
    parser.add_argument("--data-asset-name", default="slm-studio-instance-security")
    parser.add_argument("--data-asset-version",
                        default=datetime.now(timezone.utc).strftime("data-%d%m%y%H%M%S"),
                        help="Azure ML data asset version (1 to 30 characters)")
    parser.add_argument(
        "--datastore",
        help=(
            "Azure ML datastore used to upload the data asset "
            "(defaults to AZUREML_DATASTORE_NAME, then the workspace default)"
        ),
    )
    parser.add_argument("--description",
                        default="SLM Studio train/validation/test splits for instance-security grounded QA")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    try:
        validate_data_asset_version(args.data_asset_version)
    except ValueError as exc:
        parser.error(str(exc))
    if args.num_questions < 1 or (args.num_samples is not None and args.num_samples < 3):
        parser.error("--num-questions must be positive and --num-samples must be at least 3")
    if not 0 <= args.oracle_probability <= 1:
        parser.error("--oracle-probability must be in [0, 1]")
    return args


def generate(args: argparse.Namespace) -> None:
    from lib import pdf_to_chunks, dataset_generation

    if not args.skip_extract:
        log.info("Extracting chunks from %s", args.source_folder or args.pdf)
        pdfs = sorted(args.source_folder.rglob("*.pdf")) if args.source_folder else [args.pdf]
        if not pdfs:
            raise ValueError("No PDFs found in source folder")
        for index, pdf in enumerate(pdfs):
            log.info("Extracting PDF %d/%d: %s", index + 1, len(pdfs), pdf.name)
            pdf_to_chunks.extract_chunks(
                pdf,
                images_dir=args.images_dir / str(index),
                chunks_dir=args.chunks_dir / str(index),
            )
            log.info("Finished PDF %d/%d: %s", index + 1, len(pdfs), pdf.name)
    else:
        log.info("Skipping PDF extraction, reusing %s", args.chunks_dir)

    chunks = dataset_generation.load_chunks_from_dir(args.chunks_dir)

    # chunks = chunks[:10]  # Limit to first 10 chunks for testing purposes

    if not chunks:
        raise RuntimeError(f"No chunks found in {args.chunks_dir}")
    
    log.info("Loaded %d chunk(s)", len(chunks))

    dataset = dataset_generation.generate_dataset(
        chunks,
        num_questions=args.num_questions,
        p=args.oracle_probability,
        num_samples=args.num_samples,
    )
    
    dataset_generation.save_datasets(dataset.to_pandas(), str(args.dataset_dir))

    log.info("Wrote splits to %s", args.dataset_dir)


def validate(dataset_dir: Path) -> None:
    log.info("Validating dataset splits")
    summary = validate_dataset(dataset_dir)
    log.info("Quality summary:\n%s", pd.DataFrame(summary).T)
    log.info("Dataset fingerprint: %s", dataset_fingerprint(dataset_dir))


def publish(args: argparse.Namespace) -> None:
    log.info("Publishing Azure ML data asset %s:%s", args.data_asset_name, args.data_asset_version)
    ml_client = AzureMLConfig.from_env().create_ml_client()
    datastore = args.datastore or os.getenv("AZUREML_DATASTORE_NAME")
    asset = publish_data_asset(
        ml_client=ml_client,
        dataset_dir=args.dataset_dir,
        name=args.data_asset_name,
        version=args.data_asset_version,
        description=args.description,
        datastore=datastore,
    )
    log.info("Published azureml:%s:%s", asset.name, asset.version)
    log.info("Storage URI: %s", asset.path)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("azure").setLevel(logging.WARNING)
    load_dotenv(PROJECT_ROOT / ".env")
    log.info("Preparation job started; requested total samples: %s", args.num_samples or "all generated samples")

    if not args.skip_generation:
        generate(args)
    else:
        log.info("Skipping data generation, using %s", args.dataset_dir)

    validate(args.dataset_dir)

    if not args.skip_publish:
        publish(args)
    else:
        log.info("Skipping Azure ML publication")
    log.info("Preparation job completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
