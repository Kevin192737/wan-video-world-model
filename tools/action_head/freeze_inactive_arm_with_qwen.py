#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from .csv_action import load_action_array, resample_trajectory, write_action_txt


SYSTEM_PROMPT = """You are a robot-motion classifier.
Given one tabletop robot image and one instruction, decide which arm should move.

Output JSON only:
{
  "active_arm": "left" | "right" | "both",
  "reason": "<short reason>"
}

Rules:
1) If instruction explicitly says left/right arm, follow it.
2) If only one arm is near target and likely to execute the action, pick that arm.
3) If unclear, output "both".
"""


def extract_json_obj(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object in model output: {text[:200]}")
    return json.loads(text[start : end + 1])


def infer_active_arm_with_qwen(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    image_path: Path,
    instruction: str,
    temperature: float,
) -> str:
    image = Image.open(image_path).convert("RGB")
    user_prompt = f"Instruction: {instruction}\nReturn active_arm."
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            temperature=temperature,
        )
    generated_text = processor.batch_decode(
        generated_ids[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
    )[0]
    try:
        out = extract_json_obj(generated_text)
        arm = str(out.get("active_arm", "both")).strip().lower()
        if arm not in {"left", "right", "both"}:
            arm = "both"
        return arm
    except Exception:
        return "both"


def infer_active_arm_by_instruction(instruction: str) -> str:
    s = instruction.lower()
    left_hit = ("left arm" in s) or ("left robotic arm" in s) or ("左臂" in instruction)
    right_hit = ("right arm" in s) or ("right robotic arm" in s) or ("右臂" in instruction)
    if left_hit and not right_hit:
        return "left"
    if right_hit and not left_hit:
        return "right"
    return "both"


def arm_column_indices(header: list[str]) -> tuple[list[int], list[int]]:
    # header includes index column at position 0
    names = header[1:]
    left_idx = [i for i, n in enumerate(names) if n.startswith("idx13_left_arm") or n.startswith("idx14_left_arm")
                or n.startswith("idx15_left_arm") or n.startswith("idx16_left_arm") or n.startswith("idx17_left_arm")
                or n.startswith("idx18_left_arm") or n.startswith("idx19_left_arm") or n.startswith("left_")]
    right_idx = [i for i, n in enumerate(names) if n.startswith("idx20_right_arm") or n.startswith("idx21_right_arm")
                 or n.startswith("idx22_right_arm") or n.startswith("idx23_right_arm") or n.startswith("idx24_right_arm")
                 or n.startswith("idx25_right_arm") or n.startswith("idx26_right_arm") or n.startswith("right_")]
    return left_idx, right_idx


def lr_column_indices(header: list[str]) -> list[int]:
    names = header[1:]
    return [i for i, n in enumerate(names) if ("left" in n.lower() or "right" in n.lower())]


def freeze_inactive_arm(values, header: list[str], active_arm: str):
    out = values.copy()
    left_idx, right_idx = arm_column_indices(header)

    if active_arm == "left":
        out[:, right_idx] = out[0:1, right_idx]
    elif active_arm == "right":
        out[:, left_idx] = out[0:1, left_idx]
    return out


def freeze_inactive_arm_with_reference(values, ref_values, header: list[str], active_arm: str):
    out = values.copy()
    left_idx, right_idx = arm_column_indices(header)

    if ref_values.shape[0] != values.shape[0]:
        ref_values = resample_trajectory(ref_values, values.shape[0])

    if active_arm == "left":
        out[:, right_idx] = ref_values[:, right_idx]
    elif active_arm == "right":
        out[:, left_idx] = ref_values[:, left_idx]
    return out


def freeze_all_lr_with_reference(values, ref_values, header: list[str]):
    out = values.copy()
    lr_idx = lr_column_indices(header)
    if ref_values.shape[0] != values.shape[0]:
        ref_values = resample_trajectory(ref_values, values.shape[0])
    out[:, lr_idx] = ref_values[:, lr_idx]
    return out


def freeze_predicted_arm_with_reference(values, ref_values, header: list[str], active_arm: str):
    # NOTE: "predicted arm" means Qwen predicted active arm; we freeze the opposite arm.
    out = values.copy()
    left_idx, right_idx = arm_column_indices(header)
    if ref_values.shape[0] != values.shape[0]:
        ref_values = resample_trajectory(ref_values, values.shape[0])

    if active_arm == "left":
        out[:, right_idx] = ref_values[:, right_idx]
    elif active_arm == "right":
        out[:, left_idx] = ref_values[:, left_idx]
    return out


def load_active_arm_cache(cache_path: Path) -> dict[str, str]:
    cache: dict[str, str] = {}
    if not cache_path.exists():
        return cache
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sid = str(row.get("id", "")).strip()
            arm = str(row.get("active_arm", "")).strip().lower()
            if sid and arm in {"left", "right", "both"}:
                cache[sid] = arm
    return cache


def append_active_arm_cache(cache_path: Path, sample_id: str, active_arm: str, source: str) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {"id": sample_id, "active_arm": active_arm, "source": source},
                ensure_ascii=False,
            )
            + "\n"
        )


def write_aligned_preview(path: Path, header: list[str], values, start_index: int) -> None:
    names = ["idx" if not header[0] else header[0]] + header[1:]
    rows: list[list[str]] = [names]
    for i in range(values.shape[0]):
        rows.append([str(start_index + i)] + [f"{float(v):.6f}" for v in values[i]])
    widths = [max(len(r[c]) for r in rows) for c in range(len(rows[0]))]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        for r in rows:
            w.writerow([cell.ljust(widths[c]) for c, cell in enumerate(r)])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze inactive robot arm in action trajectories using Qwen or rule inference."
    )
    parser.add_argument("--manifest", type=Path, required=True, help="jsonl containing id,last_frame_path,instruction")
    parser.add_argument("--input_action_dir", type=Path, required=True, help="directory with *_action.txt")
    parser.add_argument("--output_action_dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["qwen", "instruction", "manual"], default="qwen")
    parser.add_argument("--manual_active_arm", choices=["left", "right", "both"], default="both")
    parser.add_argument(
        "--ref_action_dir",
        type=Path,
        default=None,
        help="Optional reference dir with <id>/action.txt. If provided, inactive arm will copy from reference trajectory.",
    )
    parser.add_argument(
        "--freeze_all_lr_from_ref",
        action="store_true",
        help="If set with --ref_action_dir, all columns containing left/right in header are copied from reference action.",
    )
    parser.add_argument(
        "--freeze_predicted_arm_from_ref",
        action="store_true",
        help="If set with --ref_action_dir, freeze opposite arm according to predicted active arm (active=left -> freeze right; active=right -> freeze left).",
    )
    parser.add_argument(
        "--active_arm_cache_jsonl",
        type=Path,
        default=None,
        help="Optional JSONL cache for active_arm per id. Existing cache will be reused to skip repeated Qwen inference.",
    )
    parser.add_argument(
        "--aligned_preview_dir",
        type=Path,
        default=None,
        help="Optional directory to write aligned, human-readable TSV previews for each output action file.",
    )
    parser.add_argument("--model_id", type=str, default="/home/models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all")
    args = parser.parse_args()

    rows = []
    with args.manifest.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    args.output_action_dir.mkdir(parents=True, exist_ok=True)

    model = None
    processor = None
    active_arm_cache: dict[str, str] = {}
    if args.active_arm_cache_jsonl is not None:
        active_arm_cache = load_active_arm_cache(args.active_arm_cache_jsonl)
        print(f"loaded active_arm cache: {len(active_arm_cache)}")
    if args.mode == "qwen":
        # Load Qwen only when needed (skip if all ids already cached).
        pending_ids = [r["id"] for r in rows if r["id"] not in active_arm_cache]
        if pending_ids:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            processor = AutoProcessor.from_pretrained(args.model_id)
        else:
            print("all active_arm labels found in cache, skip Qwen model loading")

    for i, row in enumerate(rows, start=1):
        sample_id = row["id"]
        in_path = args.input_action_dir / f"{sample_id}_action.txt"
        out_path = args.output_action_dir / f"{sample_id}_action.txt"
        if not in_path.exists():
            print(f"[{i}/{len(rows)}] skip {sample_id}: no action file")
            continue

        indices, values, header = load_action_array(in_path)
        source = args.mode
        if sample_id in active_arm_cache:
            active_arm = active_arm_cache[sample_id]
            source = "cache"
        elif args.mode == "manual":
            active_arm = args.manual_active_arm
        elif args.mode == "instruction":
            active_arm = infer_active_arm_by_instruction(row.get("instruction", ""))
        else:
            active_arm = infer_active_arm_with_qwen(
                model=model,
                processor=processor,
                image_path=Path(row["last_frame_path"]),
                instruction=row.get("instruction", ""),
                temperature=args.temperature,
            )
        if sample_id not in active_arm_cache:
            active_arm_cache[sample_id] = active_arm
            if args.active_arm_cache_jsonl is not None:
                append_active_arm_cache(args.active_arm_cache_jsonl, sample_id, active_arm, source)

        if args.ref_action_dir is not None:
            ref_path = args.ref_action_dir / sample_id / "action.txt"
            if ref_path.exists():
                _, ref_values, ref_header = load_action_array(ref_path)
                if len(ref_header) != len(header):
                    raise ValueError(
                        f"Header mismatch for {sample_id}: pred={len(header)} cols, ref={len(ref_header)} cols"
                    )
                if args.freeze_all_lr_from_ref:
                    values_out = freeze_all_lr_with_reference(values, ref_values, header)
                elif args.freeze_predicted_arm_from_ref:
                    values_out = freeze_predicted_arm_with_reference(values, ref_values, header, active_arm)
                else:
                    values_out = freeze_inactive_arm_with_reference(values, ref_values, header, active_arm)
            else:
                print(f"[{i}/{len(rows)}] warn {sample_id}: no ref action, fallback to first-frame freeze")
                values_out = freeze_inactive_arm(values, header, active_arm)
        else:
            values_out = freeze_inactive_arm(values, header, active_arm)
        start_index = int(indices[0]) if len(indices) > 0 else 0
        write_action_txt(out_path, values_out, header, start_index=start_index)
        if args.aligned_preview_dir is not None:
            preview_path = args.aligned_preview_dir / f"{sample_id}_action_aligned.txt"
            write_aligned_preview(preview_path, header, values_out, start_index=start_index)
        print(f"[{i}/{len(rows)}] {sample_id}: active_arm={active_arm} (source={source})")


if __name__ == "__main__":
    main()
