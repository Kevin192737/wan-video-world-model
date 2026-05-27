#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path


def run_one(
    python_bin: str,
    repo_dir: Path,
    ckpt_dir: Path,
    image_path: str,
    prompt: str,
    output_path: Path,
    frame_num: int,
    guide_scale: float,
) -> int:
    cmd = [
        python_bin,
        str(repo_dir / "generate.py"),
        "--task",
        "ti2v-5B",
        "--size",
        "1280*704",
        "--frame_num",
        str(frame_num),
        "--ckpt_dir",
        str(ckpt_dir),
        "--offload_model",
        "True",
        "--convert_model_dtype",
        "--t5_cpu",
        "--sample_guide_scale",
        str(guide_scale),
        "--image",
        image_path,
        "--prompt",
        prompt,
        "--save_file",
        str(output_path),
    ]
    return subprocess.run(cmd, cwd=repo_dir).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch TI2V inference for optimized manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo_dir", type=Path, default=Path("/home/Wan2.2"))
    parser.add_argument("--ckpt_dir", type=Path, default=Path("/home/Wan2.2/Wan2.2-TI2V-5B"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--python_bin", type=str, default="/home/Wan2.2/.venv/bin/python")
    parser.add_argument("--frame_num", type=int, default=49)
    parser.add_argument("--guide_scale", type=float, default=6.0)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "batch_log.jsonl"

    with args.manifest.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    total = len(rows)
    for i, item in enumerate(rows, start=1):
        sample_id = item["id"]
        prompt = item["instruction"]
        image_path = item["last_frame_path"]
        output_path = args.output_dir / f"{sample_id}.mp4"

        if output_path.exists():
            status = {
                "id": sample_id,
                "status": "skipped_exists",
                "index": i,
                "total": total,
                "output": str(output_path),
            }
            with log_path.open("a", encoding="utf-8") as lf:
                lf.write(json.dumps(status, ensure_ascii=False) + "\n")
            print(f"[{i}/{total}] skip {sample_id} (exists)")
            continue

        print(f"[{i}/{total}] run {sample_id}")
        code = run_one(
            python_bin=args.python_bin,
            repo_dir=args.repo_dir,
            ckpt_dir=args.ckpt_dir,
            image_path=image_path,
            prompt=prompt,
            output_path=output_path,
            frame_num=args.frame_num,
            guide_scale=args.guide_scale,
        )
        status = {
            "id": sample_id,
            "status": "ok" if code == 0 else "failed",
            "return_code": code,
            "index": i,
            "total": total,
            "output": str(output_path),
        }
        with log_path.open("a", encoding="utf-8") as lf:
            lf.write(json.dumps(status, ensure_ascii=False) + "\n")

        if code != 0:
            print(f"[{i}/{total}] failed {sample_id}, continue")


if __name__ == "__main__":
    main()
