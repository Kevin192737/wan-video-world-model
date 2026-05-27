from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .csv_action import load_action_array, resample_trajectory
from .side_mask import load_case_frozen_side_csv, loss_weight_vector
from .video_io import load_video_frames, sample_frames_uniform


class VideoActionDataset(Dataset):
    """
    One sample = one subdirectory with video.mp4 + action.txt.
    Video is uniformly sampled to num_video_frames; action is resampled to num_action_steps.

    若提供 manifest_path（如 frozen_side_manifest.csv），可返回 loss_mask：
    冻结侧维度权重为 0，仅运动侧参与训练损失（与数据「只有一侧在动」一致）。
    """

    def __init__(
        self,
        data_root: Path,
        target_file: str = "action.txt",
        num_video_frames: int = 16,
        num_action_steps: int = 96,
        image_size: int = 224,
        sample_dirs: list[str] | None = None,
        stats_path: Path | None = None,
        fit_stats: bool = True,
        manifest_path: Path | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.target_file = target_file
        self.num_video_frames = num_video_frames
        self.num_action_steps = num_action_steps
        self.image_size = image_size
        self._case_frozen: dict[str, str] = {}
        if manifest_path:
            self._case_frozen = load_case_frozen_side_csv(Path(manifest_path).expanduser().resolve())

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

        if stats_path and Path(stats_path).exists():
            with open(stats_path, encoding="utf-8") as f:
                s = json.load(f)
            self.mean = np.array(s["mean"], dtype=np.float32)
            self.std = np.array(s["std"], dtype=np.float32)
        elif fit_stats:
            self._compute_stats()
            if stats_path:
                self.save_stats(Path(stats_path))

    def _compute_stats(self) -> None:
        sums = None
        sumsq = None
        count = 0
        for name in self.sample_dirs:
            _, vals, _ = load_action_array(self.data_root / name / self.target_file)
            if sums is None:
                sums = vals.sum(axis=0)
                sumsq = (vals ** 2).sum(axis=0)
            else:
                sums += vals.sum(axis=0)
                sumsq += (vals ** 2).sum(axis=0)
            count += vals.shape[0]
        mean = sums / count
        var = sumsq / count - mean ** 2
        std = np.sqrt(np.maximum(var, 1e-8))
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def save_stats(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"mean": self.mean.tolist(), "std": self.std.tolist()}, f, indent=2)

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def _load_video_tensor(self, video_path: Path) -> torch.Tensor:
        # returns [T, C, H, W] float 0..1
        v = load_video_frames(video_path)
        v = sample_frames_uniform(v, self.num_video_frames)
        frames = v.float() / 255.0
        frames = frames.permute(0, 3, 1, 2)  # TCHW
        frames = F.interpolate(
            frames,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return frames

    def __getitem__(self, i: int) -> dict:
        name = self.sample_dirs[i]
        base = self.data_root / name
        video_path = base / "video.mp4"
        action_path = base / self.target_file

        _, vals, _ = load_action_array(action_path)
        vals_r = resample_trajectory(vals, self.num_action_steps)
        assert self.mean is not None and self.std is not None
        norm = (vals_r - self.mean) / self.std

        vid = self._load_video_tensor(video_path)
        fs = self._case_frozen.get(name)
        mask = loss_weight_vector(self._header, fs)
        out: dict = {
            "video": vid,
            "action": torch.from_numpy(norm),
            "loss_mask": torch.from_numpy(mask),
            "sample_id": name,
        }
        return out

    @property
    def header(self) -> list[str]:
        return self._header

    @property
    def action_dim(self) -> int:
        return self._output_dim
