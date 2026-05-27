# 运行说明

本目录为提交用代码结构：`model/` 为模型相关代码，`data_generation/` 为数据与批处理脚本入口（通过符号链接指向仓库内实际路径）。

## 环境

```bash
cd code
pip install -r requirements.txt
```

可选：根目录另有 `requirements_animate.txt`、`requirements_s2v.txt`（仅在使用对应任务时需要）。

权重需按官方 `README.md` / `INSTALL.md` 下载，通过 `--ckpt_dir` 指定。

## 视频生成（Wan TI2V 等）

在**仓库根目录**（`Wan2.2/`，与 `code/` 同级）执行：

```bash
python generate.py --task ti2v-5B --ckpt_dir ./Wan2.2-TI2V-5B ... 
```

完整参数见根目录 `README.md`。

## 批量数据 / 推理脚本

- 数据准备、manifest、批量 TI2V 等：见 `data_generation/tools/`（与根目录 `tools/` 为同一目录）。
- 动作头训练与推理：`model/action_head/`（与 `tools/action_head/` 为同一目录）。

更完整的流水线说明见根目录 `PROJECT_README.md`。

## 目录映射

| 本结构 | 实际路径（仓库根下） |
|--------|----------------------|
| `model/wan` | `wan/` |
| `model/action_head` | `tools/action_head` |
| `data_generation/tools` | `tools/` |
