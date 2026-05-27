"""对关节轨迹 [T, D] 按维做恒定速度模型 + RTS 平滑（离线卡尔曼平滑）。"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def _cv_rts_smooth_1d(
    z: np.ndarray,
    *,
    dt: float,
    q_pos: float,
    q_vel: float,
    r_meas: float,
) -> np.ndarray:
    """
    z: [T] 观测（该关节时间序列）
    返回平滑后的位置 [T]
    状态 x = [position, velocity]^T；F 为 CV；仅观测位置。
    """
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    t_len = z.size
    if t_len < 2:
        return z.astype(np.float64)

    f = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
    h = np.array([[1.0, 0.0]], dtype=np.float64)
    q = np.diag([max(q_pos, 1e-18), max(q_vel, 1e-18)]).astype(np.float64)
    r = np.array([[max(r_meas, 1e-18)]], dtype=np.float64)
    i2 = np.eye(2, dtype=np.float64)

    x_filt = np.zeros((t_len, 2), dtype=np.float64)
    p_filt = np.zeros((t_len, 2, 2), dtype=np.float64)
    x_pred = np.zeros((t_len, 2), dtype=np.float64)
    p_pred = np.zeros((t_len, 2, 2), dtype=np.float64)

    # t=0：用首观测初始化位置，速度 0，较大速度不确定
    x_filt[0] = np.array([z[0], 0.0], dtype=np.float64)
    p_filt[0] = np.diag([r_meas, max(1.0, q_vel * 1e6)])

    for t in range(1, t_len):
        x_pred[t] = f @ x_filt[t - 1]
        p_pred[t] = f @ p_filt[t - 1] @ f.T + q
        s = h @ p_pred[t] @ h.T + r
        s_inv = 1.0 / float(s[0, 0])
        k = (p_pred[t] @ h.T) * s_inv
        inn = z[t] - float((h @ x_pred[t])[0])
        x_filt[t] = (x_pred[t].reshape(2) + (k.flatten() * inn)).astype(np.float64)
        p_filt[t] = (i2 - k @ h) @ p_pred[t]

    x_smooth = np.zeros((t_len, 2), dtype=np.float64)
    x_smooth[-1] = x_filt[-1]
    for t in range(t_len - 2, -1, -1):
        try:
            p_pred_inv = np.linalg.inv(p_pred[t + 1])
        except np.linalg.LinAlgError:
            p_pred_inv = np.linalg.pinv(p_pred[t + 1])
        c_gain = p_filt[t] @ f.T @ p_pred_inv
        x_smooth[t] = x_filt[t] + c_gain @ (x_smooth[t + 1] - x_pred[t + 1])

    return (x_smooth @ h.T).ravel()


def smooth_action_trajectory_cv_rts(
    values: np.ndarray,
    *,
    dt: float = 1.0,
    q_pos: float = 1e-8,
    q_vel: float = 1e-6,
    r_meas: float = 1e-4,
    columns: Iterable[int] | None = None,
) -> np.ndarray:
    """
    对 [T, D] 动作序列在指定列上独立做 CV-RTS 平滑。

    columns: 要平滑的列下标 0..D-1；为 None 时平滑全部列。
    q_pos, q_vel: 过程噪声（越小轨迹越「僵」、越贴近线性）
    r_meas: 观测噪声（越大越不信模型预测、平滑越强）
    """
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 2 or v.shape[0] < 2:
        return values.astype(np.float32)
    _, dim = v.shape
    out = v.copy()
    idxs = list(range(dim)) if columns is None else [int(c) for c in columns]
    for d in idxs:
        if 0 <= d < dim:
            out[:, d] = _cv_rts_smooth_1d(v[:, d], dt=dt, q_pos=q_pos, q_vel=q_vel, r_meas=r_meas)
    return out.astype(np.float32)
