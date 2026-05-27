# 模型权重（不包含在 Git 仓库中）

请在本目录或仓库根目录下载 Wan 官方权重，推理时通过 `--ckpt_dir` 指定路径。

## Wan2.2-TI2V-5B（世界模型默认）

推荐从 Hugging Face 下载：

```bash
# 需安装 huggingface-cli: pip install huggingface_hub
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./Wan2.2-TI2V-5B
```

或访问：<https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B>

也可使用 ModelScope，见根目录官方 `README.md` 的 Model Download 章节。

## 动作头（可选）

动作头需自行训练，见 `tools/action_head/train.py`，或将 `action_head.pt` 与 `dataset_meta.json`、`action_norm_stats.json` 放在同一目录，通过世界模型 `--action_head` 传入。
