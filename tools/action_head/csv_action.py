"""Parse and write action.txt-style CSV (first column index, 26 value columns)."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def load_action_array(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Returns:
        indices: [L] int
        values: [L, D] float (D = len(numeric columns))
        header_cells: full header row as list (for writing back)
    """
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty action file: {path}")

    # First column may be '' or 'Unnamed: 0' in header; data first col is index
    indices = np.array([int(float(r[0])) for r in rows], dtype=np.int64)
    values = np.array([[float(x) for x in r[1:]] for r in rows], dtype=np.float32)
    return indices, values, header


def resample_trajectory(values: np.ndarray, target_len: int) -> np.ndarray:
    """Linear resample [L, D] -> [target_len, D] along time."""
    if values.shape[0] == target_len:
        return values.astype(np.float32)
    if values.shape[0] == 1:
        return np.repeat(values, target_len, axis=0).astype(np.float32)
    t_old = np.linspace(0.0, 1.0, values.shape[0], dtype=np.float32)
    t_new = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    out = np.stack([np.interp(t_new, t_old, values[:, d]) for d in range(values.shape[1])], axis=1)
    return out.astype(np.float32)


def write_action_txt(
    path: Path,
    values: np.ndarray,
    header_template: list[str],
    start_index: int = 0,
) -> None:
    """
    values: [T, D]
    header_template: e.g. from load_action_array; first cell can be '' or 'Unnamed: 0'
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    t, d = values.shape
    if len(header_template) != d + 1:
        raise ValueError(f"Header has {len(header_template)} cols, expected {d + 1} for D={d}")

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header_template)
        for i in range(t):
            row = [start_index + i] + [float(values[i, j]) for j in range(d)]
            w.writerow(row)


def default_header_from_train() -> list[str]:
    """Fallback header matching /home/release/train style."""
    return [
        "",
        "idx13_left_arm_joint1_position",
        "idx14_left_arm_joint2_position",
        "idx15_left_arm_joint3_position",
        "idx16_left_arm_joint4_position",
        "idx17_left_arm_joint5_position",
        "idx18_left_arm_joint6_position",
        "idx19_left_arm_joint7_position",
        "idx20_right_arm_joint1_position",
        "idx21_right_arm_joint2_position",
        "idx22_right_arm_joint3_position",
        "idx23_right_arm_joint4_position",
        "idx24_right_arm_joint5_position",
        "idx25_right_arm_joint6_position",
        "idx26_right_arm_joint7_position",
        "left_thumb_0_position",
        "left_thumb_1_position",
        "left_index_position",
        "left_middle_position",
        "left_ring_position",
        "left_pinky_position",
        "right_thumb_0_position",
        "right_thumb_1_position",
        "right_index_position",
        "right_middle_position",
        "right_ring_position",
        "right_pinky_position",
    ]
