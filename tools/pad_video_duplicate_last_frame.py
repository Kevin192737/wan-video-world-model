#!/usr/bin/env python3
"""Append one duplicated last video frame (49 -> 50 at same fps)."""
from __future__ import annotations

import argparse
import subprocess
from fractions import Fraction
from pathlib import Path


def video_fps(video: Path) -> float:
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "default=nw=1:nk=1",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    rate = r.stdout.strip()
    if "/" in rate:
        return float(Fraction(rate))
    return float(rate)


def frame_count(video: Path) -> int:
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "csv=p=0",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(r.stdout.strip())


def pad_one_frame(in_path: Path, out_path: Path, crf: int = 18) -> None:
    fps = video_fps(in_path)
    if fps <= 0:
        raise RuntimeError(f"Bad fps for {in_path}")
    stop_d = 1.0 / fps
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vf = f"tpad=stop_mode=clone:stop_duration={stop_d}"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(in_path),
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
        "-an",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> None:
    parser = argparse.ArgumentParser(description="Duplicate last frame once to +1 frame.")
    parser.add_argument("--input", type=Path, help="single mp4")
    parser.add_argument("--output", type=Path, help="output mp4 (single mode)")
    parser.add_argument("--input_dir", type=Path, help="batch: all *.mp4")
    parser.add_argument("--output_dir", type=Path, help="batch output dir")
    parser.add_argument("--skip_if_frames", type=int, default=0, help="skip if already >= this many frames (e.g. 50)")
    parser.add_argument("--crf", type=int, default=18)
    args = parser.parse_args()

    if args.input and args.output:
        n = frame_count(args.input)
        if args.skip_if_frames and n >= args.skip_if_frames:
            print(f"skip {args.input.name}: already {n} frames")
            return
        pad_one_frame(args.input, args.output, crf=args.crf)
        print(f"{args.input.name}: {n} -> {frame_count(args.output)} frames -> {args.output}")
        return

    if args.input_dir and args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for p in sorted(args.input_dir.glob("*.mp4")):
            n = frame_count(p)
            if args.skip_if_frames and n >= args.skip_if_frames:
                print(f"skip {p.name}: {n} frames")
                continue
            out = args.output_dir / p.name
            pad_one_frame(p, out, crf=args.crf)
            print(f"{p.name}: {n} -> {frame_count(out)}")
        return

    raise SystemExit("Need --input/--output or --input_dir/--output_dir")


if __name__ == "__main__":
    main()
