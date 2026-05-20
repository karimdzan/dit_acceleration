"""Build side-by-side baseline/token-merge comparison images from generated folders"""

import argparse
import json
from pathlib import Path

from fae.evaluation.ditxl_eval_utils import build_side_by_side_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--token-merge-dir", required=True)
    parser.add_argument("--labels-json", required=True, help="JSON written by generate_ditxl_token_merge_eval.py.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--columns", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    with Path(args.labels_json).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    labels = [int(x) for x in payload["labels"]]
    summary = build_side_by_side_outputs(
        baseline_dir=args.baseline_dir,
        token_merge_dir=args.token_merge_dir,
        output_dir=args.output_dir,
        labels=labels,
        n=args.n,
        grid_columns=args.columns,
    )
    print(summary)


if __name__ == "__main__":
    main()
