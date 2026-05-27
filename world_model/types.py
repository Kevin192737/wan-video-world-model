from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class WorldModelState:
    """世界模型当前状态：以观测图像（末帧）为主。"""

    image_path: Path
    step_index: int = 0
    last_video_path: Path | None = None
    last_instruction: str | None = None

    def to_dict(self) -> dict:
        return {
            "image_path": str(self.image_path),
            "step_index": self.step_index,
            "last_video_path": str(self.last_video_path) if self.last_video_path else None,
            "last_instruction": self.last_instruction,
        }


@dataclass
class WorldModelStepResult:
    """单步 rollout 输出。"""

    step_index: int
    instruction: str
    video_path: Path
    next_image_path: Path
    predicted_action: np.ndarray | None = None
    predicted_action_path: Path | None = None
    metadata: dict = field(default_factory=dict)
