#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import WorldModelConfig
from .wan_world_model import WanVideoWorldModel


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wan2.2 视频世界模型：单步预测或多步 rollout"
    )
    parser.add_argument("--image", type=Path, required=True, help="当前状态观测图（首帧/末帧）")
    parser.add_argument(
        "--instruction",
        type=str,
        action="append",
        required=True,
        help="文本动作/指令，可多次传入以做多步 rollout",
    )
    parser.add_argument("--ckpt_dir", type=Path, default=None, help="Wan2.2-TI2V-5B 权重目录")
    parser.add_argument(
        "--action_head",
        type=Path,
        default=None,
        help="动作头 checkpoint（action_head.pt），不提供则只生成视频",
    )
    parser.add_argument("--work_dir", type=Path, default=None)
    parser.add_argument("--repo_dir", type=Path, default=None)
    parser.add_argument("--no_action", action="store_true", help="禁用动作头预测")
    parser.add_argument("--frame_num", type=int, default=49)
    parser.add_argument("--guide_scale", type=float, default=6.0)
    args = parser.parse_args()

    cfg = WorldModelConfig(
        frame_num=args.frame_num,
        guide_scale=args.guide_scale,
        action_head_checkpoint=args.action_head,
    )
    if args.ckpt_dir:
        cfg.ckpt_dir = args.ckpt_dir
    if args.work_dir:
        cfg.work_dir = args.work_dir
    if args.repo_dir:
        cfg.repo_dir = args.repo_dir
    cfg.resolve_paths()

    model = WanVideoWorldModel(cfg)
    final_state, results = model.rollout(
        args.image,
        args.instruction,
        predict_action=not args.no_action and args.action_head is not None,
    )

    summary = {
        "final_image": str(final_state.image_path),
        "num_steps": len(results),
        "steps": [
            {
                "step": r.step_index,
                "instruction": r.instruction,
                "video": str(r.video_path),
                "next_image": str(r.next_image_path),
            }
            for r in results
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
