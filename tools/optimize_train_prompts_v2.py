#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from PIL import Image


def normalize_instruction(text: str) -> str:
    t = " ".join(text.strip().split())
    t = t.replace("Cocacola", "Coca-Cola")
    return t


def infer_target_object(instruction: str) -> str:
    s = instruction.lower()
    if "coca-cola" in s or "coke" in s:
        return "Coca-Cola bottle"
    if "sprite" in s:
        return "Sprite can"
    if "green tea" in s:
        return "green tea bottle"
    if "water" in s:
        return "water bottle"
    return "target object"


def build_prompt(instruction: str) -> str:
    normalized = normalize_instruction(instruction).rstrip(".")
    target_obj = infer_target_object(normalized)
    return (
        "A robotic arm performs only one action in this exact tabletop scene: "
        f"{normalized}. "
        f"The target is the {target_obj}, and it must be placed into the empty box. "
        "Keep all other objects (including non-target drinks and the other robotic arm) completely still. "
        "Use a fixed camera view, no extra movements, no additional interactions, no object drift."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate stricter train prompts from manifest while reading every image."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    with args.input.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            image_path = Path(item["last_frame_path"])
            with Image.open(image_path) as img:
                width, height = img.size

            raw = item.get("instruction", "")
            item["instruction_raw"] = raw
            item["instruction"] = build_prompt(raw)
            item["image_width"] = width
            item["image_height"] = height
            rows.append(item)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

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
