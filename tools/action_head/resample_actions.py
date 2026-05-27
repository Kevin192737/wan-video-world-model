#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .csv_action import load_action_array, resample_trajectory, write_action_txt


def infer_start_index(indices) -> int:
    if len(indices) == 0:
        return 0
    return int(indices[0])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch resample action.txt-style CSV files to target steps."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing action txt files (e.g. *_action.txt).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to write resampled action txt files.",
    )
    parser.add_argument(
        "--target_steps",
        type=int,
        required=True,
        help="Target number of trajectory rows (excluding header).",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*_action.txt",
        help="Filename glob under input_dir.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=None,
        help="Override output start index. Default: keep each file's original first index.",
    )
    args = parser.parse_args()

    if args.target_steps <= 0:
        raise SystemExit("--target_steps must be > 0")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.input_dir.glob(args.glob))
    if not files:
        raise SystemExit(f"No files matched {args.glob} in {args.input_dir}")

    ok = 0
    for path in files:
        indices, values, header = load_action_array(path)
        values_rs = resample_trajectory(values, args.target_steps)
        out_start = args.start_index if args.start_index is not None else infer_start_index(indices)
        out_path = args.output_dir / path.name
        write_action_txt(out_path, values_rs, header, start_index=out_start)
        ok += 1
        print(
            f"[ok] {path.name}: {values.shape[0]} -> {values_rs.shape[0]} rows "
            f"(start_index={out_start})"
        )

    print(f"done: {ok} files -> {args.output_dir}")


if __name__ == "__main__":
    main()
