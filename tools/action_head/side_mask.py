"""与 frozen_arm_from_action / freeze_action_by_manifest 一致的左右列划分，用于训练损失掩码。"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def frozen_value_column_indices(header: list[str], frozen_side: str) -> list[int]:
    """值列下标 0..D-1（对应 header[1:]）中属于 frozen_side 的列。"""
    names = header[1:]
    out: list[int] = []
    fs = frozen_side.lower()
    for k, name in enumerate(names):
        cl = name.lower()
        if fs == "left" and "left" in cl and "right" not in cl:
            out.append(k)
        elif fs == "right" and "right" in cl and "left" not in cl:
            out.append(k)
        elif fs == "both":
            if ("left" in cl and "right" not in cl) or ("right" in cl and "left" not in cl):
                out.append(k)
    return out


def loss_weight_vector(header: list[str], frozen_side: str | None) -> np.ndarray:
    """
    训练用损失权重，形状 [D]：运动侧=1，冻结侧=0（不在这些维度上算 MSE）。
    frozen_side 为 None / 空 / neither 时全 1。
    """
    d = len(header) - 1
    w = np.ones(d, dtype=np.float32)
    if not frozen_side:
        return w
    fs = frozen_side.strip().lower()
    if fs in ("", "neither", "none", "unknown"):
        return w
    if fs not in ("left", "right", "both"):
        return w
    frozen_ixs = frozen_value_column_indices(header, fs)
    for k in frozen_ixs:
        w[k] = 0.0
    if float(w.sum()) < 1e-6:
        # 双侧都冻等退化情况，避免除零，退回全监督
        return np.ones(d, dtype=np.float32)
    return w


def load_case_frozen_side_csv(path: Path) -> dict[str, str]:
    """读 manifest：case -> frozen_side；跳过带 error 的行。"""
    out: dict[str, str] = {}
    with Path(path).expanduser().open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            case = (row.get("case") or "").strip()
            if not case:
                continue
            if (row.get("error") or "").strip():
                continue
            fs = (row.get("frozen_side") or "").strip().lower()
            if fs:
                out[case] = fs
    return out
