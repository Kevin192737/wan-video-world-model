#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .csv_action import write_action_txt
from .kalman_smooth import smooth_action_trajectory_cv_rts
from .model import video_action_head_from_payload


def load_video_for_model(
    video_path: Path,
    num_frames: int,
    image_size: int,
) -> torch.Tensor:
    from .video_io import load_video_frames, sample_frames_uniform

    v = load_video_frames(video_path)
    v = sample_frames_uniform(v, num_frames)
    frames = v.float() / 255.0
    frames = frames.permute(0, 3, 1, 2)
    frames = torch.nn.functional.interpolate(
        frames,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer action.txt from video using trained action head.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--meta", type=Path, default=None, help="dataset_meta.json next to checkpoint")
    parser.add_argument("--stats", type=Path, default=None, help="action_norm_stats.json")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--kalman",
        action="store_true",
        help="对反标准化后的轨迹做 CV 模型 RTS 平滑（每关节独立）",
    )
    parser.add_argument("--kalman-dt", type=float, default=1.0)
    parser.add_argument("--kalman-q-pos", type=float, default=1e-8, help="过程噪声（位置）")
    parser.add_argument("--kalman-q-vel", type=float, default=1e-6, help="过程噪声（速度）")
    parser.add_argument("--kalman-r", type=float, default=1e-4, help="观测噪声（越大平滑越强）")
    args = parser.parse_args()

    ckpt_dir = args.checkpoint.parent
    meta_path = args.meta or (ckpt_dir / "dataset_meta.json")
    stats_path = args.stats
    if stats_path is None and meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        stats_path = Path(meta["stats"])
    if stats_path is None or not Path(stats_path).exists():
        raise SystemExit("Provide --stats or dataset_meta.json with stats path")

    with open(stats_path, encoding="utf-8") as f:
        st = json.load(f)
    mean = np.array(st["mean"], dtype=np.float32)
    std = np.array(st["std"], dtype=np.float32)

    payload = torch.load(args.checkpoint, map_location="cpu")
    num_action_steps = int(payload["num_action_steps"])
    num_video_frames = int(payload["num_video_frames"])
    image_size = int(payload["image_size"])
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    header = meta["header"]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = video_action_head_from_payload(payload)
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()

    vid = load_video_for_model(args.video, num_video_frames, image_size)
    with torch.no_grad():
        pred = model(vid.unsqueeze(0).to(device)).cpu().numpy()[0]
    pred = pred * std + mean

    if args.kalman:
        pred = smooth_action_trajectory_cv_rts(
            pred.astype(np.float32),
            dt=args.kalman_dt,
            q_pos=args.kalman_q_pos,
            q_vel=args.kalman_q_vel,
            r_meas=args.kalman_r,
        )

    write_action_txt(args.output, pred.astype(np.float32), header, start_index=args.start_index)
    print(f"Wrote {args.output} shape [{num_action_steps}, {pred.shape[1]}]")


if __name__ == "__main__":
    main()
