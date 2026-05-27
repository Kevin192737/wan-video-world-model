#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DiffSynth metadata.csv from manifest jsonl.")
    parser.add_argument("--input_manifest", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all")
    args = parser.parse_args()

    rows = []
    with args.input_manifest.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            rows.append(
                {
                    "video": item["video_path"],
                    "prompt": item["instruction"],
                }
            )

    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "prompt"])
        writer.writeheader()
        writer.writerows(rows)

    print(
        json.dumps(
            {
                "input": str(args.input_manifest.resolve()),
                "output": str(args.output_csv.resolve()),
                "count": len(rows),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
