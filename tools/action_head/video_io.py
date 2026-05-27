"""Load video frames as [T, H, W, C] uint8. Uses decord."""
from __future__ import annotations

from pathlib import Path

import torch


def load_video_frames(video_path: Path) -> torch.Tensor:
    """Returns [T, H, W, C] uint8."""
    try:
        from decord import VideoReader, cpu
    except ImportError as e:
        raise ImportError("Install decord: pip install decord") from e

    vr = VideoReader(str(video_path), ctx=cpu(0))
    n = len(vr)
    if n == 0:
        raise RuntimeError(f"Empty video: {video_path}")
    frames = vr.get_batch(range(n)).asnumpy()  # [T, H, W, C]
    return torch.from_numpy(frames)


def sample_frames_uniform(frames: torch.Tensor, num_frames: int) -> torch.Tensor:
    """frames [T, H, W, C] -> [num_frames, H, W, C]"""
    t = frames.shape[0]
    idx = torch.linspace(0, t - 1, num_frames).long()
    return frames[idx]


def sample_frames_uniform_from_index(frames: torch.Tensor, start: int, num_frames: int) -> torch.Tensor:
    """从 start 帧开始到视频结尾的子序列上均匀采样 num_frames 帧（含端点）。"""
    if start < 0:
        start = 0
    sub = frames[start:]
    if sub.shape[0] == 0:
        raise RuntimeError(f"视频从 start={start} 起无帧")
    return sample_frames_uniform(sub, num_frames)
