# 动作头训练产出（不包含在 Git 仓库中）

本目录用于存放 `tools/action_head` 训练得到的检查点。仓库中仅保留部分 **JSON 元数据**（如 `dataset_meta.json`）作为格式参考；**`.pt` 权重文件已被 `.gitignore` 排除**。

训练示例：

```bash
python -m tools.action_head.train \
  --train_dir /path/to/train \
  --out_dir ./action_head_runs/my_run \
  --epochs 20
```

推理与世界模型联用：

```bash
python -m world_model.cli \
  --image /path/to/frame.jpg \
  --instruction "pick up the cup" \
  --ckpt_dir ./Wan2.2-TI2V-5B \
  --action_head ./action_head_runs/my_run/action_head.pt
```
