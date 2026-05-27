#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def extract_last_frame(video_path: Path, output_image_path: Path) -> None:
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-sseof",
        "-0.1",
        "-i",
        str(video_path),
        "-vframes",
        "1",
        str(output_image_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def build_split(split_dir: Path, split_name: str, out_dir: Path) -> list[dict]:
    frame_root = out_dir / "frames" / split_name
    items = []
    for sample_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        video_path = sample_dir / "video.mp4"
        instruction_path = sample_dir / "instruction.txt"
        action_path = sample_dir / "action.txt"
        joint_path = sample_dir / "joint.txt"
        if not video_path.exists() or not instruction_path.exists():
            continue

        frame_path = frame_root / f"{sample_dir.name}.jpg"
        extract_last_frame(video_path, frame_path)

        item = {
            "id": sample_dir.name,
            "split": split_name,
            "video_path": str(video_path.resolve()),
            "last_frame_path": str(frame_path.resolve()),
            "instruction": read_text(instruction_path),
        }
        if action_path.exists():
            item["action_path"] = str(action_path.resolve())
        if joint_path.exists():
            item["joint_path"] = str(joint_path.resolve())
        items.append(item)
    return items


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare train/test dataset manifests for Wan2.2 pipeline.")
    parser.add_argument("--train_dir", type=Path, required=True)
    parser.add_argument("--test_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    train_items = build_split(args.train_dir, "train", args.out_dir)
    test_items = build_split(args.test_dir, "test", args.out_dir)

    write_jsonl(args.out_dir / "train_manifest.jsonl", train_items)
    write_jsonl(args.out_dir / "test_manifest.jsonl", test_items)

    summary = {
        "train_count": len(train_items),
        "test_count": len(test_items),
        "train_manifest": str((args.out_dir / "train_manifest.jsonl").resolve()),
        "test_manifest": str((args.out_dir / "test_manifest.jsonl").resolve()),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
