#!/usr/bin/env python3
"""Convert mp4 videos to fixed 30fps and 1280x720 resolution."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def probe_video(video: Path) -> dict[str, str]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_packets",
        "-show_entries",
        "stream=width,height,r_frame_rate,nb_read_packets",
        "-of",
        "json",
        str(video),
    ]
    out = subprocess.check_output(cmd, text=True)
    data = json.loads(out)
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found: {video}")
    stream = streams[0]
    return {
        "width": str(stream.get("width")),
        "height": str(stream.get("height")),
        "r_frame_rate": str(stream.get("r_frame_rate")),
        "nb_read_packets": str(stream.get("nb_read_packets")),
    }


def already_target(video: Path, target_fps: int, target_w: int, target_h: int) -> bool:
    meta = probe_video(video)
    return (
        meta["r_frame_rate"] == f"{target_fps}/1"
        and meta["width"] == str(target_w)
        and meta["height"] == str(target_h)
    )


def convert_one(
    in_path: Path,
    out_path: Path,
    target_fps: int,
    target_w: int,
    target_h: int,
    crf: int,
    keep_audio: bool,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vf = (
        f"scale={target_w}:{target_h}:flags=lanczos:force_original_aspect_ratio=decrease,"
        f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black,fps={target_fps}"
    )
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
    ]
    if keep_audio:
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-an"]
    cmd.append(str(out_path))
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def process_single(args: argparse.Namespace) -> None:
    if args.skip_if_target and already_target(args.input, args.fps, args.width, args.height):
        print(f"skip {args.input.name}: already {args.fps}fps {args.width}x{args.height}")
        return
    before = probe_video(args.input)
    convert_one(
        args.input,
        args.output,
        target_fps=args.fps,
        target_w=args.width,
        target_h=args.height,
        crf=args.crf,
        keep_audio=args.keep_audio,
    )
    after = probe_video(args.output)
    print(
        f"{args.input.name}: {before['r_frame_rate']} {before['width']}x{before['height']} "
        f"{before['nb_read_packets']}f -> {after['r_frame_rate']} {after['width']}x{after['height']} "
        f"{after['nb_read_packets']}f"
    )


def process_batch(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.input_dir.glob("*.mp4"))
    if not files:
        raise SystemExit(f"No *.mp4 files found in {args.input_dir}")
    ok = 0
    skip = 0
    for p in files:
        out = args.output_dir / p.name
        if args.skip_if_target and already_target(p, args.fps, args.width, args.height):
            print(f"skip {p.name}: already {args.fps}fps {args.width}x{args.height}")
            skip += 1
            continue
        before = probe_video(p)
        convert_one(
            p,
            out,
            target_fps=args.fps,
            target_w=args.width,
            target_h=args.height,
            crf=args.crf,
            keep_audio=args.keep_audio,
        )
        after = probe_video(out)
        ok += 1
        print(
            f"{p.name}: {before['r_frame_rate']} {before['width']}x{before['height']} "
            f"{before['nb_read_packets']}f -> {after['r_frame_rate']} {after['width']}x{after['height']} "
            f"{after['nb_read_packets']}f"
        )
    print(f"done: converted={ok}, skipped={skip}, output={args.output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert mp4 to fixed fps and resolution.")
    parser.add_argument("--input", type=Path, help="Single input mp4")
    parser.add_argument("--output", type=Path, help="Single output mp4")
    parser.add_argument("--input_dir", type=Path, help="Batch input dir (all *.mp4)")
    parser.add_argument("--output_dir", type=Path, help="Batch output dir")
    parser.add_argument("--fps", type=int, default=30, help="Target fps (default: 30)")
    parser.add_argument("--width", type=int, default=1280, help="Target width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Target height (default: 720)")
    parser.add_argument("--crf", type=int, default=18, help="x264 CRF (default: 18)")
    parser.add_argument("--keep_audio", action="store_true", help="Keep/re-encode audio track")
    parser.add_argument(
        "--skip_if_target",
        action="store_true",
        help="Skip videos already at target fps and resolution",
    )
    args = parser.parse_args()

    if args.input and args.output:
        process_single(args)
        return
    if args.input_dir and args.output_dir:
        process_batch(args)
        return
    raise SystemExit("Need --input/--output or --input_dir/--output_dir")


if __name__ == "__main__":
    main()
