"""过去 16 步 action + 切分点至结尾的视频(均匀 50 帧) -> 未来 action(至结尾再重采样到固定步数)。"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .csv_action import load_action_array, resample_trajectory
from .side_mask import load_case_frozen_side_csv, loss_weight_vector
from .video_io import load_video_frames, sample_frames_uniform_from_index


class FutureActionDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        target_file: str = "action.txt",
        past_len: int = 16,
        future_video_frames: int = 50,
        num_future_action_steps: int = 51,
        image_size: int = 224,
        sample_dirs: list[str] | None = None,
        stats_path: Path | None = None,
        fit_stats: bool = True,
        manifest_path: Path | None = None,
        min_future_action_rows: int = 2,
        split_t0_min_frac: float = 0.15,
        split_t0_max_frac: float = 0.92,
        seed: int | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.target_file = target_file
        self.past_len = past_len
        self.future_video_frames = future_video_frames
        self.num_future_action_steps = num_future_action_steps
        self.image_size = image_size
        self.min_future_action_rows = min_future_action_rows
        self.split_t0_min_frac = split_t0_min_frac
        self.split_t0_max_frac = split_t0_max_frac
        self._rng = random.Random(seed)

        if sample_dirs is None:
            sample_dirs = sorted(p.name for p in self.data_root.iterdir() if p.is_dir())
        self.sample_dirs = sample_dirs
        if not self.sample_dirs:
            raise ValueError(f"No samples under {self.data_root}")

        _, _, header = load_action_array(self.data_root / self.sample_dirs[0] / self.target_file)
        self._header: list[str] = header
        self._output_dim = len(header) - 1
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

        self._case_frozen: dict[str, str] = {}
        if manifest_path:
            self._case_frozen = load_case_frozen_side_csv(Path(manifest_path).expanduser().resolve())

        if stats_path and Path(stats_path).exists():
            with open(stats_path, encoding="utf-8") as f:
                s = json.load(f)
            self.mean = np.array(s["mean"], dtype=np.float32)
            self.std = np.array(s["std"], dtype=np.float32)
        elif fit_stats:
            self._compute_stats_future_only()
            if stats_path:
                self.save_stats(Path(stats_path))

    def _compute_stats_future_only(self) -> None:
        """仅用各样本「随机一切分」后的未来 action 段（重采样后）估计 mean/std。"""
        sums = None
        sumsq = None
        count = 0
        for name in self.sample_dirs:
            _, vals, _ = load_action_array(self.data_root / name / self.target_file)
            a = vals.shape[0]
            if a <= self.past_len + self.min_future_action_rows:
                continue
            t0 = self._sample_t0(a)
            fut = resample_trajectory(vals[t0:], self.num_future_action_steps)
            if sums is None:
                sums = fut.sum(axis=0)
                sumsq = (fut ** 2).sum(axis=0)
            else:
                sums += fut.sum(axis=0)
                sumsq += (fut ** 2).sum(axis=0)
            count += fut.shape[0]
        if count == 0:
            raise ValueError("无法估计统计量：检查训练集 action 行数是否足够长")
        mean = sums / count
        var = sumsq / count - mean ** 2
        std = np.sqrt(np.maximum(var, 1e-8))
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def _sample_t0(self, num_action_rows: int) -> int:
        lo = max(self.past_len, int(num_action_rows * self.split_t0_min_frac))
        hi = min(num_action_rows - self.min_future_action_rows, int(num_action_rows * self.split_t0_max_frac))
        if hi <= lo:
            lo = self.past_len
            hi = num_action_rows - self.min_future_action_rows
        if hi <= lo:
            raise ValueError(f"action 行数={num_action_rows} 不足以切分 past={self.past_len}")
        return self._rng.randint(lo, hi + 1)

    @staticmethod
    def _action_index_to_frame_index(t0_a: int, num_action_rows: int, num_video_frames: int) -> int:
        if num_action_rows <= 1:
            return 0
        return int(round(t0_a / (num_action_rows - 1) * max(num_video_frames - 1, 0)))

    def save_stats(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"mean": self.mean.tolist(), "std": self.std.tolist()}, f, indent=2)

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def _tensor_video(self, frames_hwc_u8: torch.Tensor) -> torch.Tensor:
        x = frames_hwc_u8.float() / 255.0
        x = x.permute(0, 3, 1, 2)
        return F.interpolate(
            x,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

    def __getitem__(self, i: int) -> dict:
        name = self.sample_dirs[i]
        base = self.data_root / name
        _, vals, _ = load_action_array(base / self.target_file)
        a = vals.shape[0]
        t0 = self._sample_t0(a)

        past = vals[t0 - self.past_len : t0].astype(np.float32)  # [past_len, D]
        fut_raw = vals[t0:]
        fut = resample_trajectory(fut_raw, self.num_future_action_steps).astype(np.float32)

        assert self.mean is not None and self.std is not None
        past_n = (past - self.mean) / self.std
        fut_n = (fut - self.mean) / self.std

        v = load_video_frames(base / "video.mp4")
        vlen = int(v.shape[0])
        t0_v = self._action_index_to_frame_index(t0, a, vlen)
        v_sub = sample_frames_uniform_from_index(v, t0_v, self.future_video_frames)
        vid = self._tensor_video(v_sub)

        fs = self._case_frozen.get(name)
        mask = loss_weight_vector(self._header, fs)

        return {
            "past_action": torch.from_numpy(past_n),
            "future_video": vid,
            "future_action": torch.from_numpy(fut_n),
            "loss_mask": torch.from_numpy(mask),
            "sample_id": name,
        }

    @property
    def header(self) -> list[str]:
        return self._header

    @property
    def action_dim(self) -> int:
        return self._output_dim
