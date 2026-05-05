"""Merge saved RAFT prediction CSVs into one output file."""

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge RAFT prediction CSVs into a single dataframe."
    )
    parser.add_argument(
        "--predictions",
        nargs="+",
        default=[
            "./output/llm_predictions.csv",
            "./output/slm_predictions.csv",
            "./output/slm_baseline_predictions.csv",
        ],
        help="Prediction CSV files to merge.",
    )
    parser.add_argument(
        "--output",
        default="./output/merged_predictions.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def load_prediction_csv(path: str) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    dataframe["prediction_file"] = Path(path).name

    if "model_name" not in dataframe.columns:
        dataframe["model_name"] = Path(path).stem.replace("_predictions", "")

    return dataframe


def run(args: argparse.Namespace) -> None:
    merged = pd.concat(
        [load_prediction_csv(path) for path in args.predictions],
        ignore_index=True,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    print(f"Saved {len(merged)} rows to {output_path}")


if __name__ == "__main__":
    run(parse_args())
