#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import save_video


DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，"
    "畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def build_pipe() -> WanVideoPipeline:
    return WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth"),
        ],
        tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch TI2V inference with DiffSynth LoRA.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--lora_path", type=Path, default=None)
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all")
    parser.add_argument("--tiled", action="store_true", default=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "batch_log.jsonl"

    with args.manifest.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    pipe = build_pipe()
    if args.lora_path is not None:
        pipe.load_lora(pipe.dit, str(args.lora_path), alpha=args.lora_alpha)
        print(f"Loaded LoRA: {args.lora_path} (alpha={args.lora_alpha})")
    else:
        print("No LoRA loaded. Running base model.")

    total = len(rows)
    for i, item in enumerate(rows, start=1):
        sample_id = item["id"]
        output_path = args.output_dir / f"{sample_id}.mp4"
        if output_path.exists():
            status = {"id": sample_id, "status": "skipped_exists", "index": i, "total": total, "output": str(output_path)}
            with log_path.open("a", encoding="utf-8") as lf:
                lf.write(json.dumps(status, ensure_ascii=False) + "\n")
            print(f"[{i}/{total}] skip {sample_id} (exists)")
            continue

        prompt = item["instruction"]
        image = Image.open(item["last_frame_path"]).convert("RGB").resize((args.width, args.height))
        print(f"[{i}/{total}] run {sample_id}")
        try:
            video = pipe(
                prompt=prompt,
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                input_image=image,
                num_frames=args.num_frames,
                height=args.height,
                width=args.width,
                seed=args.seed,
                tiled=args.tiled,
            )
            save_video(video, str(output_path), fps=args.fps, quality=args.quality)
            status = {"id": sample_id, "status": "ok", "index": i, "total": total, "output": str(output_path)}
        except Exception as e:
            status = {
                "id": sample_id,
                "status": "failed",
                "error": str(e),
                "index": i,
                "total": total,
                "output": str(output_path),
            }
        with log_path.open("a", encoding="utf-8") as lf:
            lf.write(json.dumps(status, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
