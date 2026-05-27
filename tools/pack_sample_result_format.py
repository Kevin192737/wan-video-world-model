#!/usr/bin/env python3
"""Pack predicted action/joint/video files into sample_result-like folder layout."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_sample_id_from_action(path: Path) -> str:
    name = path.name
    suffix = "_action.txt"
    if not name.endswith(suffix):
        raise ValueError(f"Unexpected action filename: {name}")
    return name[: -len(suffix)]


def parse_sample_id_from_joint(path: Path) -> str:
    name = path.name
    suffix = "_joint.txt"
    if not name.endswith(suffix):
        raise ValueError(f"Unexpected joint filename: {name}")
    return name[: -len(suffix)]


def collect_sets(action_dir: Path, joint_dir: Path, video_dir: Path) -> tuple[set[str], set[str], set[str]]:
    action_ids = {parse_sample_id_from_action(p) for p in action_dir.glob("*_action.txt")}
    joint_ids = {parse_sample_id_from_joint(p) for p in joint_dir.glob("*_joint.txt")}
    video_ids = {p.stem for p in video_dir.glob("*.mp4")}
    return action_ids, joint_ids, video_ids


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack predictions into /sample_result-style folders: <id>/{action.txt,joint.txt,video.mp4,instruction.txt}"
    )
    parser.add_argument(
        "--action_dir",
        type=Path,
        default=Path("/home/release/pred_actions_batch_51"),
        help="Directory containing *_action.txt files.",
    )
    parser.add_argument(
        "--joint_dir",
        type=Path,
        default=Path("/home/release/pred_joints_batch_51"),
        help="Directory containing *_joint.txt files.",
    )
    parser.add_argument(
        "--video_dir",
        type=Path,
        default=Path("/home/release/sample_result_pred_vlm_stylematch_50f_30fps_720p"),
        help="Directory containing <id>.mp4 files.",
    )
    parser.add_argument(
        "--instruction_dir",
        type=Path,
        default=Path("/home/release/test"),
        help="Directory containing <id>/instruction.txt to copy.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output root directory in sample_result format.",
    )
    parser.add_argument(
        "--instruction_fallback_file",
        type=Path,
        default=None,
        help="Optional fallback instruction file if <instruction_dir>/<id>/instruction.txt does not exist.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when any id is missing action/joint/video/instruction.",
    )
    parser.add_argument(
        "--clear_output",
        action="store_true",
        help="Delete output_dir before packing.",
    )
    args = parser.parse_args()

    if args.clear_output and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    action_ids, joint_ids, video_ids = collect_sets(args.action_dir, args.joint_dir, args.video_dir)
    common_ids = sorted(action_ids & joint_ids & video_ids)

    missing_action = sorted((joint_ids | video_ids) - action_ids)
    missing_joint = sorted((action_ids | video_ids) - joint_ids)
    missing_video = sorted((action_ids | joint_ids) - video_ids)

    print(f"action files: {len(action_ids)}")
    print(f"joint files : {len(joint_ids)}")
    print(f"video files : {len(video_ids)}")
    print(f"common ids  : {len(common_ids)}")

    if missing_action:
        print(f"[warn] missing action for {len(missing_action)} ids, e.g. {missing_action[:5]}")
    if missing_joint:
        print(f"[warn] missing joint for {len(missing_joint)} ids, e.g. {missing_joint[:5]}")
    if missing_video:
        print(f"[warn] missing video for {len(missing_video)} ids, e.g. {missing_video[:5]}")

    if args.strict and (missing_action or missing_joint or missing_video):
        raise SystemExit("Strict mode failed: id sets are not aligned.")

    packed = 0
    missing_instruction = []
    for sample_id in common_ids:
        out_dir = args.output_dir / sample_id
        copy_file(args.action_dir / f"{sample_id}_action.txt", out_dir / "action.txt")
        copy_file(args.joint_dir / f"{sample_id}_joint.txt", out_dir / "joint.txt")
        copy_file(args.video_dir / f"{sample_id}.mp4", out_dir / "video.mp4")

        instruction_src = args.instruction_dir / sample_id / "instruction.txt"
        if instruction_src.exists():
            copy_file(instruction_src, out_dir / "instruction.txt")
        elif args.instruction_fallback_file and args.instruction_fallback_file.exists():
            copy_file(args.instruction_fallback_file, out_dir / "instruction.txt")
            missing_instruction.append(sample_id)
        else:
            missing_instruction.append(sample_id)
            if args.strict:
                raise SystemExit(f"Strict mode failed: missing instruction for {sample_id}")

        packed += 1

    print(f"packed ids   : {packed} -> {args.output_dir}")
    if missing_instruction:
        print(
            f"[warn] instruction missing for {len(missing_instruction)} ids, "
            f"e.g. {missing_instruction[:5]}"
        )


if __name__ == "__main__":
    main()
