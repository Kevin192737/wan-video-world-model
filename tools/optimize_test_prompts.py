#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def normalize_instruction(text: str) -> str:
    t = " ".join(text.strip().split())
    t = t.replace("Cocacola", "Coca-Cola")
    return t.rstrip(".")


def build_prompt(instruction: str) -> str:
    core = normalize_instruction(instruction)
    return (
        f"A robotic arm performs only one action: {core}. "
        "Keep all other objects and the other robotic arm completely still. "
        "Fixed camera view, no extra movements, no additional interactions."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize prompts for batch TI2V inference.")
    parser.add_argument("--input", type=Path, required=True, help="Input manifest jsonl path.")
    parser.add_argument("--output", type=Path, required=True, help="Output manifest jsonl path.")
    args = parser.parse_args()

    rows = []
    with args.input.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            item["instruction_raw"] = item.get("instruction", "")
            item["instruction"] = build_prompt(item.get("instruction", ""))
            rows.append(item)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for item in rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "input": str(args.input.resolve()),
                "output": str(args.output.resolve()),
                "count": len(rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
