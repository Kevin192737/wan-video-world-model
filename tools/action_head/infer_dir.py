#!/usr/bin/env python3
"""对目录内所有 .mp4 批量推理 action.txt（模型只加载一次）。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .csv_action import write_action_txt
from .infer import load_video_for_model
from .kalman_smooth import smooth_action_trajectory_cv_rts
from .model import video_action_head_from_payload


def main() -> None:
    p = argparse.ArgumentParser(description="Batch infer action.txt from videos in a directory.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--meta", type=Path, default=None)
    p.add_argument("--stats", type=Path, default=None)
    p.add_argument("--video-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--kalman", action="store_true", help="CV + RTS 平滑（与 infer.py 一致）")
    p.add_argument("--kalman-dt", type=float, default=1.0)
    p.add_argument("--kalman-q-pos", type=float, default=1e-8)
    p.add_argument("--kalman-q-vel", type=float, default=1e-6)
    p.add_argument("--kalman-r", type=float, default=1e-4)
    args = p.parse_args()

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

    video_dir = args.video_dir.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(video_dir.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"No .mp4 under {video_dir}")

    for i, vp in enumerate(videos, 1):
        vid = load_video_for_model(vp, num_video_frames, image_size)
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
        out_path = out_dir / f"{vp.stem}_action.txt"
        write_action_txt(out_path, pred.astype(np.float32), header, start_index=args.start_index)
        print(f"[{i}/{len(videos)}] {vp.name} -> {out_path}")

    print(f"Done: {len(videos)} files -> {out_dir}")


if __name__ == "__main__":
    main()
