from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from .config import WorldModelConfig
from .types import WorldModelState, WorldModelStepResult
from .video_utils import extract_last_frame


class WanVideoWorldModel:
    """
    基于 Wan2.2 TI2V 的世界模型。

    将「文本指令」视为高层动作，以当前观测图像为状态，通过视频扩散模型预测
    下一时刻的视觉轨迹，并可选地用动作头将视频映射到低维动作空间。
    """

    def __init__(self, config: WorldModelConfig | None = None) -> None:
        self.config = (config or WorldModelConfig()).resolve_paths()
        self.config.work_dir.mkdir(parents=True, exist_ok=True)
        self._action_model = None
        self._action_meta: dict | None = None
        self._action_stats: dict | None = None

    def reset(self, initial_image: Path | str) -> WorldModelState:
        image_path = Path(initial_image).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"初始图像不存在: {image_path}")
        return WorldModelState(image_path=image_path, step_index=0)

    def step(
        self,
        state: WorldModelState,
        instruction: str,
        *,
        predict_action: bool = True,
        output_video: Path | None = None,
    ) -> WorldModelStepResult:
        step_idx = state.step_index + 1
        video_path = output_video or (
            self.config.work_dir / f"step_{step_idx:04d}.mp4"
        )
        video_path = video_path.resolve()
        video_path.parent.mkdir(parents=True, exist_ok=True)

        self._run_ti2v(
            image_path=state.image_path,
            prompt=instruction,
            output_path=video_path,
        )

        next_image = self.config.work_dir / f"step_{step_idx:04d}_last.jpg"
        extract_last_frame(video_path, next_image)

        predicted_action = None
        action_path = None
        if predict_action and self.config.action_head_checkpoint is not None:
            predicted_action, action_path = self._predict_action(video_path, step_idx)

        return WorldModelStepResult(
            step_index=step_idx,
            instruction=instruction,
            video_path=video_path,
            next_image_path=next_image,
            predicted_action=predicted_action,
            predicted_action_path=action_path,
            metadata={"previous_image": str(state.image_path)},
        )

    def rollout(
        self,
        initial_image: Path | str,
        instructions: list[str],
        *,
        predict_action: bool = True,
    ) -> tuple[WorldModelState, list[WorldModelStepResult]]:
        state = self.reset(initial_image)
        results: list[WorldModelStepResult] = []
        for instruction in instructions:
            result = self.step(
                state,
                instruction,
                predict_action=predict_action,
            )
            results.append(result)
            state = WorldModelState(
                image_path=result.next_image_path,
                step_index=result.step_index,
                last_video_path=result.video_path,
                last_instruction=instruction,
            )
        self._save_rollout_log(results)
        return state, results

    def _run_ti2v(self, image_path: Path, prompt: str, output_path: Path) -> None:
        cfg = self.config
        if not cfg.ckpt_dir.is_dir():
            raise FileNotFoundError(
                f"Wan 权重目录不存在: {cfg.ckpt_dir}\n"
                "请从 Hugging Face 下载 Wan2.2-TI2V-5B，见 weights/README.md"
            )

        cmd = [
            cfg.python_bin,
            str(cfg.repo_dir / "generate.py"),
            "--task",
            cfg.task,
            "--size",
            cfg.size,
            "--frame_num",
            str(cfg.frame_num),
            "--ckpt_dir",
            str(cfg.ckpt_dir),
            "--offload_model",
            str(cfg.offload_model),
            "--sample_guide_scale",
            str(cfg.guide_scale),
            "--image",
            str(image_path),
            "--prompt",
            prompt,
            "--save_file",
            str(output_path),
        ]
        if cfg.convert_model_dtype:
            cmd.append("--convert_model_dtype")
        if cfg.t5_cpu:
            cmd.append("--t5_cpu")
        if cfg.seed is not None:
            cmd.extend(["--base_seed", str(cfg.seed)])

        proc = subprocess.run(cmd, cwd=cfg.repo_dir, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"TI2V 推理失败 (code={proc.returncode})\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
        if not output_path.is_file():
            raise RuntimeError(f"未生成视频: {output_path}")

    def _load_action_head(self) -> None:
        if self._action_model is not None:
            return
        ckpt = self.config.action_head_checkpoint
        if ckpt is None:
            return
        if not ckpt.is_file():
            raise FileNotFoundError(f"动作头权重不存在: {ckpt}")

        from tools.action_head.model import video_action_head_from_payload

        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        self._action_model = video_action_head_from_payload(payload)
        self._action_model.eval()

        ckpt_dir = ckpt.parent
        meta_path = ckpt_dir / "dataset_meta.json"
        if meta_path.is_file():
            with meta_path.open(encoding="utf-8") as f:
                self._action_meta = json.load(f)
        stats_name = (
            self._action_meta.get("stats_file", "action_norm_stats.json")
            if self._action_meta
            else "action_norm_stats.json"
        )
        stats_path = ckpt_dir / stats_name
        if stats_path.is_file():
            with stats_path.open(encoding="utf-8") as f:
                self._action_stats = json.load(f)

    def _predict_action(
        self,
        video_path: Path,
        step_idx: int,
    ) -> tuple[np.ndarray, Path]:
        from tools.action_head.infer import load_video_for_model

        self._load_action_head()
        assert self._action_model is not None and self._action_meta is not None

        num_frames = int(self._action_meta.get("num_video_frames", 16))
        image_size = int(self._action_meta.get("image_size", 224))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._action_model.to(device)

        frames = load_video_for_model(video_path, num_frames, image_size)
        with torch.no_grad():
            pred = self._action_model(frames.unsqueeze(0).to(device))
        pred_np = pred.squeeze(0).cpu().numpy()

        if self._action_stats:
            mean = np.asarray(self._action_stats["mean"], dtype=np.float32)
            std = np.asarray(self._action_stats["std"], dtype=np.float32)
            pred_np = pred_np * std + mean

        out_path = self.config.work_dir / f"step_{step_idx:04d}_action.npy"
        np.save(out_path, pred_np)
        return pred_np, out_path

    def _save_rollout_log(self, results: list[WorldModelStepResult]) -> None:
        log_path = self.config.work_dir / "rollout_log.jsonl"
        with log_path.open("w", encoding="utf-8") as f:
            for r in results:
                row = {
                    "step_index": r.step_index,
                    "instruction": r.instruction,
                    "video_path": str(r.video_path),
                    "next_image_path": str(r.next_image_path),
                    "predicted_action_path": (
                        str(r.predicted_action_path) if r.predicted_action_path else None
                    ),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
