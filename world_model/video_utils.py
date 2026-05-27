from __future__ import annotations

from pathlib import Path

from PIL import Image


def extract_last_frame(video_path: Path, output_image: Path) -> Path:
    """从视频中抽取最后一帧并保存为图像。"""
    try:
        from decord import VideoReader, cpu
    except ImportError as e:
        raise ImportError("需要 decord：pip install decord") from e

    vr = VideoReader(str(video_path), ctx=cpu(0))
    if len(vr) == 0:
        raise RuntimeError(f"空视频: {video_path}")
    frame = vr[len(vr) - 1].asnumpy()
    output_image.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(output_image)
    return output_image
