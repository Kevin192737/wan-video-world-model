#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


SYSTEM_PROMPT = """You are a prompt rewriter for robot-manipulation video generation.
Given one image (the last frame of a tabletop scene) and one raw instruction, produce a strict single-action prompt for Wan2.2 TI2V.

Rules:
1) Keep only ONE action for ONE arm.
2) Infer active_arm as one of: left, right, both. If uncertain, use both.
3) Explicitly state which arm moves and which arm must stay frozen.
4) Target object must be explicitly named from the visible scene.
5) If a box is visible, use wording equivalent to "empty box".
6) Identify visible non-target objects and list them explicitly.
7) Keep all non-target objects and the non-active arm completely still.
8) Motion constraints must be explicit: smooth and continuous trajectory, no sudden jump, no excessive elbow lift.
9) Preserve target-object appearance exactly: shape/size/color/label/packaging text must remain unchanged; do not switch to foreign packaging variant.
10) Fixed camera, no extra movement, no additional interaction.
11) Do not invent objects not visible in the image.
12) Output JSON only, no markdown.
13) final_prompt must be Chinese.

Output schema:
{
  "target_object": "<string>",
  "task": "<string>",
  "active_arm": "left|right|both",
  "non_target_objects": ["<string>", "..."],
  "constraints": ["<string>", "..."],
  "final_prompt": "<single Chinese paragraph>"
}
"""


def extract_json_obj(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Model output has no JSON object: {text[:300]}")
    return json.loads(text[start : end + 1])


def build_user_prompt(instruction: str) -> str:
    return (
        f"Raw instruction:\n{instruction}\n\n"
        "Please analyze the image and rewrite the instruction into the required strict template and output schema. "
        "The final prompt must be Chinese and must explicitly include arm-freeze and appearance-preservation constraints."
    )


def load_existing_ids(path: Path) -> set[str]:
    ids = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "id" in obj:
                ids.add(obj["id"])
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite TI2V prompts with Qwen2.5-VL.")
    parser.add_argument("--input_manifest", type=Path, required=True)
    parser.add_argument("--output_manifest", type=Path, required=True)
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all")
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_id)

    with args.input_manifest.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    done_ids = load_existing_ids(args.output_manifest)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)

    for idx, row in enumerate(rows, start=1):
        sample_id = row["id"]
        if sample_id in done_ids:
            print(f"[{idx}/{len(rows)}] skip {sample_id}")
            continue

        image = Image.open(row["last_frame_path"]).convert("RGB")
        instruction = row.get("instruction_raw", row.get("instruction", ""))
        user_prompt = build_user_prompt(instruction)

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
                max_new_tokens=300,
                do_sample=True,
                temperature=args.temperature,
            )
        generated_text = processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)[0]

        try:
            out = extract_json_obj(generated_text)
            row["instruction_raw"] = instruction
            row["instruction"] = out.get("final_prompt", instruction)
            row["vlm_target_object"] = out.get("target_object", "")
            row["vlm_task"] = out.get("task", "")
            row["vlm_active_arm"] = out.get("active_arm", "both")
            row["vlm_constraints"] = out.get("constraints", [])
            row["vlm_model"] = args.model_id
            row["vlm_ok"] = True
        except Exception as e:
            row["vlm_model"] = args.model_id
            row["vlm_ok"] = False
            row["vlm_error"] = str(e)
            row["vlm_raw_output"] = generated_text

        with args.output_manifest.open("a", encoding="utf-8") as wf:
            wf.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"[{idx}/{len(rows)}] done {sample_id}")


if __name__ == "__main__":
    main()
