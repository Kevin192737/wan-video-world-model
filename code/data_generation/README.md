# 数据生成代码

- **`tools/`**：指向仓库根目录 `tools/` 的符号链接，内含：
  - `batch_infer_ti2v.py`：按 manifest 批量调用 `generate.py`
  - `prepare_release_dataset.py`：从 train/test 目录生成 jsonl / 抽帧
  - `build_diffsynth_metadata.py`、`optimize_*_prompts*.py` 等辅助脚本
  - 以及 `action_head/`（模型侧代码在 `code/model/action_head` 亦有链接）

具体用法见各脚本 `--help` 与根目录 `PROJECT_README.md`。
