#!/usr/bin/env python3
"""推理：16 步过去 action + 50 帧未来视频 -> 未来 action；可选 manifest 冻结侧；可选重采样。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .csv_action import load_action_array, resample_trajectory, write_action_txt
from .future_model import future_action_head_from_payload
from .kalman_smooth import smooth_action_trajectory_cv_rts
from .side_mask import frozen_value_column_indices
from .video_io import load_video_frames, sample_frames_uniform


def _past_tensor_from_action_file(
    path: Path,
    past_len: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> torch.Tensor:
    _, vals, _ = load_action_array(path)
    a = vals.shape[0]
    if a >= past_len:
        past = vals[a - past_len :].astype(np.float32)
    else:
        pad = np.repeat(vals[:1], past_len - a, axis=0)
        past = np.concatenate([pad, vals], axis=0).astype(np.float32)
    norm = (past - mean) / std
    return torch.from_numpy(norm).float()


def _future_video_tensor(path: Path, num_frames: int, image_size: int) -> torch.Tensor:
    v = load_video_frames(path)
    v = sample_frames_uniform(v, num_frames)
    x = v.float() / 255.0
    x = x.permute(0, 3, 1, 2)
    return F.interpolate(
        x,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )


def _case_from_paths(past_action: Path, case: str | None) -> str:
    if case and case.strip():
        return case.strip()
    return past_action.resolve().parent.name


def _manifest_row(manifest: Path, case: str) -> dict[str, str] | None:
    with manifest.expanduser().open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("case") or "").strip() == case:
                return {k: (v or "") for k, v in row.items()}
    return None


def _last_row_value_dict(action_path: Path) -> dict[str, float]:
    with action_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    if not rows:
        raise ValueError(f"空 action 文件: {action_path}")
    last = rows[-1]
    return {header[i]: float(last[i]) for i in range(1, len(header))}


def _apply_side_freeze(
    values: np.ndarray,
    header: list[str],
    frozen_side: str,
    ref_last: dict[str, float],
) -> np.ndarray:
    out = values.copy()
    names = header[1:]
    for k in frozen_value_column_indices(header, frozen_side):
        col = names[k]
        if col in ref_last:
            out[:, k] = ref_last[col]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Infer future action from past action + 50-frame video.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--meta", type=Path, default=None, help="future_dataset_meta.json")
    p.add_argument("--stats", type=Path, default=None)
    p.add_argument("--past-action", type=Path, required=True, help="含至少 1 行 action；不足 past_len 左填充，超过则取最后 past_len 行")
    p.add_argument("--video", type=Path, required=True, help="通常为 50 帧 Wan 视频；将均匀采样到训练时的 future_video_frames")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resample-to", type=int, default=None, help="若设置，在反标准化后再线性重采样到该步数（如 51）")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="frozen_side_manifest.csv：按 frozen_side 将冻结侧整段设为参考 action 末行（默认用 manifest 内该 case 的 action_path）",
    )
    p.add_argument(
        "--case",
        type=str,
        default=None,
        help="manifest 中的 case id；默认取 --past-action 父目录名（如 .../test/2_12/action.txt -> 2_12）",
    )
    p.add_argument(
        "--reference-action",
        type=Path,
        default=None,
        help="冻结用参考 action.txt；不传则用 manifest 该行的 action_path",
    )
    p.add_argument(
        "--kalman",
        action="store_true",
        help="对轨迹做 CV + RTS 平滑；若同时提供 --manifest 且冻结生效，仅对非冻结侧关节维平滑",
    )
    p.add_argument("--kalman-dt", type=float, default=1.0)
    p.add_argument("--kalman-q-pos", type=float, default=1e-8)
    p.add_argument("--kalman-q-vel", type=float, default=1e-6)
    p.add_argument("--kalman-r", type=float, default=1e-4)
    args = p.parse_args()

    ckpt_dir = args.checkpoint.parent
    meta_path = args.meta or (ckpt_dir / "future_dataset_meta.json")
    stats_path = args.stats
    if stats_path is None and meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        stats_path = Path(meta["stats"])
    if stats_path is None or not Path(stats_path).exists():
        raise SystemExit("需要 --stats 或 meta 内 stats 路径")

    with open(stats_path, encoding="utf-8") as f:
        st = json.load(f)
    mean = np.array(st["mean"], dtype=np.float32)
    std = np.array(st["std"], dtype=np.float32)

    payload = torch.load(args.checkpoint, map_location="cpu")
    past_len = int(payload["past_len"])
    future_video_frames = int(payload["future_video_frames"])
    num_future_action_steps = int(payload["num_future_action_steps"])
    image_size = int(payload["image_size"])

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    header = meta["header"]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = future_action_head_from_payload(payload)
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()

    pa = _past_tensor_from_action_file(args.past_action, past_len, mean, std)
    fv = _future_video_tensor(args.video, future_video_frames, image_size)
    with torch.no_grad():
        pred = model(pa.unsqueeze(0).to(device), fv.unsqueeze(0).to(device)).cpu().numpy()[0]
    pred = (pred * std + mean).astype(np.float32)

    # manifest：先解析（供卡尔曼列掩码 + 后续冻结）
    manifest_freeze: tuple[str, dict[str, float]] | None = None
    frozen_col_ix: set[int] | None = None
    if args.manifest is not None:
        case = _case_from_paths(args.past_action, args.case)
        row = _manifest_row(args.manifest, case)
        if not row or (row.get("error") or "").strip():
            print(f"[freeze] skip: manifest 无有效行 case={case!r}", flush=True)
        else:
            fs = (row.get("frozen_side") or "").strip().lower()
            ref_path = Path((args.reference_action or row.get("action_path") or "").strip()).expanduser()
            if fs not in ("left", "right", "both"):
                print(f"[freeze] skip: frozen_side={fs!r}", flush=True)
            elif not ref_path.is_file():
                print(f"[freeze] skip: 参考文件不存在 {ref_path}", flush=True)
            else:
                try:
                    ref_last = _last_row_value_dict(ref_path)
                    manifest_freeze = (fs, ref_last)
                    frozen_col_ix = set(frozen_value_column_indices(header, fs))
                    print(f"[freeze] case={case} frozen_side={fs} ref={ref_path}", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[freeze] skip: {e}", flush=True)

    if args.kalman:
        d = pred.shape[1]
        if frozen_col_ix is not None and len(frozen_col_ix) > 0:
            cols = [i for i in range(d) if i not in frozen_col_ix]
            if cols:
                pred = smooth_action_trajectory_cv_rts(
                    pred,
                    dt=args.kalman_dt,
                    q_pos=args.kalman_q_pos,
                    q_vel=args.kalman_q_vel,
                    r_meas=args.kalman_r,
                    columns=cols,
                )
                print(f"[kalman] RTS 仅非冻结侧 {len(cols)}/{d} 维", flush=True)
            else:
                print("[kalman] 全部为冻结维，跳过平滑", flush=True)
        else:
            pred = smooth_action_trajectory_cv_rts(
                pred,
                dt=args.kalman_dt,
                q_pos=args.kalman_q_pos,
                q_vel=args.kalman_q_vel,
                r_meas=args.kalman_r,
            )

    if manifest_freeze is not None:
        fs, ref_last = manifest_freeze
        pred = _apply_side_freeze(pred.astype(np.float32), header, fs, ref_last)

    if args.resample_to is not None and args.resample_to != pred.shape[0]:
        pred = resample_trajectory(pred.astype(np.float32), args.resample_to)

    write_action_txt(
        args.output,
        pred.astype(np.float32),
        header,
        start_index=args.start_index,
    )
    print(f"Wrote {args.output} shape [{pred.shape[0]}, {pred.shape[1]}]")


if __name__ == "__main__":
    main()
