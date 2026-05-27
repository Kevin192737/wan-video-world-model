from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class WorldModelConfig:
    """世界模型运行配置。"""

    repo_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    ckpt_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "Wan2.2-TI2V-5B"
    )
    python_bin: str = "python"
    task: str = "ti2v-5B"
    size: str = "1280*704"
    frame_num: int = 49
    guide_scale: float = 6.0
    offload_model: bool = True
    convert_model_dtype: bool = True
    t5_cpu: bool = True
    action_head_checkpoint: Path | None = None
    work_dir: Path | None = None
    seed: int | None = None

    def resolve_paths(self) -> WorldModelConfig:
        self.repo_dir = self.repo_dir.resolve()
        self.ckpt_dir = self.ckpt_dir.resolve()
        if self.action_head_checkpoint is not None:
            self.action_head_checkpoint = self.action_head_checkpoint.resolve()
        if self.work_dir is None:
            self.work_dir = self.repo_dir / "world_model_runs"
        else:
            self.work_dir = self.work_dir.resolve()
        return self
