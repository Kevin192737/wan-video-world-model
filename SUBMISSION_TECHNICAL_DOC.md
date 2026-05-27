# Wan2.2 扩展工程 — 技术文档（赛题提交用）

本文档对应赛方要求的「技术报告」核心内容，可与仓库内 `PROJECT_README.md`、根目录官方 `README.md` / `INSTALL.md` 一并作为复现依据。导出 PDF 时可直接以本文为正文骨架，补充图表与实验数值。

---

## 一、开源与自研声明（须在正式报告中显式写明）

| 组件 | 说明 |
|------|------|
| **Wan2.2 视频生成主干** | 采用官方开源 [Wan2.2](https://github.com/Wan-Video/Wan2.2)，本工程默认批量推理使用 **`ti2v-5B`** 任务（图生视频）。若**未对 Wan 扩散权重做微调/LoRA 训练**，须在技术报告中写明：**视频由官方预训练模型零样本生成**。 |
| **动作头（Video → Action / Future Action）** | 位于 `tools/action_head/`，为**自研轻量监督学习模块**（ResNet18 + GRU + 可选 Query-Attention），在赛方或自建轨迹数据上训练；检查点为 `action_head.pt` / `future_action_head.pt` 等，**需单独网盘提供链接与提取码**。 |
| **提示词改写（可选）** | `tools/rewrite_prompts_with_qwen_vl.py` 等依赖 **Qwen2.5-VL** 等开源多模态模型，属调用开源 API 或本地权重，非赛题核心生成器本体。 |

---

## 二、方法概述

整体为**两阶段（或三阶段）流水线**：

1. **数据准备**：从 `train`/`test` 目录抽取末帧、指令与轨迹路径，生成 jsonl **manifest**（`tools/prepare_release_dataset.py`）。
2. **视频生成**：以末帧图像 + 文本指令为条件，调用官方 `generate.py`（`ti2v-5B`）或 DiffSynth 批量脚本产出 `mp4`（`tools/batch_infer_ti2v.py`、`tools/batch_infer_ti2v_lora_diffsynth.py`）。
3. **轨迹估计（自研动作头）**：从生成视频均匀采样帧，回归与赛方格式一致的 `action.txt`（及可选 `joint.txt`）；可选 **未来段动作头**（过去 action + 未来视频 → 未来 action，`train_future.py` / `infer_future.py`）。
4. **后处理与打包**：补帧、统一 30 fps / 1280×720、重采样步数、Qwen 冻结非活动臂等（`tools/pad_video_duplicate_last_frame.py`、`convert_video_force_50f_30fps_1280x720.py`、`resample_actions.py`、`freeze_inactive_arm_with_qwen.py`、`pack_sample_result_format.py`）。

**设计动机**：在固定官方视频生成能力的前提下，用**轻量、可快速迭代**的时序头将像素域生成结果对齐到**离散关节/动作空间**，满足评测对轨迹格式与步数的要求，并控制 48 小时生成窗口内的总算力。

---

## 三、模型架构

### 3.1 视频生成（Wan2.2-TI2V-5B）

- **输入**：参考图（通常为真实轨迹视频的**最后一帧**）、文本提示（可由规则模板或 Qwen-VL 增强）。
- **输出**：短时视频序列（脚本侧可设 `frame_num`、`guide_scale` 等，默认批量脚本示例见 `tools/batch_infer_ti2v.py`）。
- **实现入口**：仓库根目录 `generate.py`，任务 `--task ti2v-5B`，权重目录由 `--ckpt_dir` 指定。

*正式报告中建议配 1 张 Wan 官方架构示意图或引用其论文/GitHub 说明；若未微调，明确写「未训练，仅推理」。*

### 3.2 动作头 `VideoActionHead`（`tools/action_head/model.py`）

- **输入**：均匀采样的 `T` 帧 RGB（默认可配置，如 16 或 50），分辨率经 resize（如 224×224）。
- **骨干**：ImageNet 预训练 **ResNet18** 逐帧提特征 → 线性投影到 `hidden` 维。
- **时序**：**GRU**（层数可配，`gru_layers`；多层时层间 dropout=0.1）。
- **读出**（二选一）：
  - **基线**：将 GRU 隐状态在时间维做**线性插值**到 `num_action_steps`，再全连接得到每步 `action_dim` 维输出。
  - **增强**：`use_query_decoder=True` 时，使用**可学习 query** + **Multihead Cross-Attention** + FFN + LayerNorm，使每一步动作显式 attend 整段视频时间轴（适合 `T_vid` 与 `num_action_steps` 接近、需细对齐的场景）。
- **损失**：对**标准化后**的动作序列做 **MSE**；若提供 `frozen_side_manifest.csv`，可对冻结侧关节掩码，使损失仅作用在运动侧（见 `train.py` 的 `--manifest`）。

### 3.3 未来动作头 `FutureActionHead`（`tools/action_head/future_model.py`）

- **输入**：过去 `past_len` 步动作（已标准化）+ **未来段**视频均匀 `future_video_frames` 帧。
- **结构**：过去动作经 Linear + GRU 得到向量；视频分支同 ResNet18 + GRU；将 `[past_vec; vid_hidden]` 拼成 memory，用 **future queries + Cross-Attention + FFN** 读出 `num_future_action_steps` 步未来动作。
- **用途**：在「已知历史动作 + 已生成未来视频」设定下，直接预测未来轨迹，可与全序列动作头对比或级联使用。

---

## 四、训练策略

### 4.1 数据格式

- 每个训练样本目录包含 **`video.mp4`** 与监督文件 **`action.txt`** 或 **`joint.txt`**（列数与赛方 CSV 一致，不含索引列由 `csv_action` 模块约定）。
- 首次训练会在 `out_dir`（或 `--stats`）写入 **`action_norm_stats.json`** / **`future_action_norm_stats.json`**，并保存 **`dataset_meta.json`** 供推理一致使用。

### 4.2 全视频 → 全序列动作（`tools/action_head/train.py`）

- **优化器**：Adam 类默认学习率 `1e-4`（可按验证集调整）。
- **批量**：`batch_size`；支持 **`torchrun` 多卡 DDP**，全局 batch = 每卡 batch × GPU 数。
- **epoch**：默认 20（可增）。
- **推荐配置提示**（脚本 `epilog`）：与「约 50 帧 Wan 视频 → 51 步 action」对齐时，可设 `--num_video_frames 50 --num_action_steps 51 --use-query-decoder --gru-layers 2`。

### 4.3 过去动作 + 未来视频 → 未来动作（`tools/action_head/train_future.py`）

- 将完整轨迹按切分点分为「过去 / 未来」，视频仅取未来段对齐的均匀帧；监督为未来段 action 重采样到 `num_future_action_steps`（默认 51）。
- 默认 `past_len=16`，`future_video_frames=50`，epoch 默认 30；同样支持 DDP。

### 4.4 推理与导出

- `python -m tools.action_head.infer`：单视频 → `pred_action.txt`。
- `infer_future.py`（若使用未来头）：需 past action 与未来视频路径等（见脚本参数）。
- 后处理：`resample_actions.py` 线性重采样到评测步数（如 51）。

---

## 五、数据生成方法

1. **Manifest 生成**：扫描 `train`/`test`，用 ffmpeg 抽每条 `video.mp4` 的**最后一帧**为 jpg，写入 `last_frame_path`、`instruction`、`action_path` 等字段（`prepare_release_dataset.py`）。
2. **（可选）提示词优化**：`optimize_test_prompts*.py`、`rewrite_prompts_with_qwen_vl.py` 生成或改写 `instruction`。
3. **批量 TI2V**：`batch_infer_ti2v.py` 逐行读取 jsonl，调用根目录 `generate.py`，输出 `<id>.mp4`；可限 `--max_samples` 做冒烟测试。
4. **（可选）DiffSynth + LoRA**：`batch_infer_ti2v_lora_diffsynth.py`。
5. **规格统一**：`convert_video_to_30fps_1280x720.py` 或 **`convert_video_force_50f_30fps_1280x720.py`**（强制 50 帧 + 30fps + 1280×720）；`pad_video_duplicate_last_frame.py` 补 1 帧等。
6. **动作推理与打包**：`action_head.infer` → 可选 `freeze_inactive_arm_with_qwen.py` → `pack_sample_result_format.py` 合并为 `<id>/action.txt|joint.txt|video.mp4|instruction.txt`。

**48 小时约束建议**：优先调小分辨率/帧数做网格试验；使用官方推荐 offload（`--offload_model True`、`--t5_cpu` 等）；批量任务并行时注意显存与 IO，避免重复失败重跑占满时间窗口。

---

## 六、消融实验分析（报告建议章节）

以下可在正式报告中用表格呈现（数值由你在验证集上填写）：

| 实验 ID | 变量 | 设置 | 验证 MSE / 下游指标 | 说明 |
|--------|------|------|---------------------|------|
| A0 | 基线 | 线性插值读出，`gru_layers=1`，`T=16` | （填） | 最小实现 |
| A1 | Query 解码器 | `use_query_decoder=True` | （填） | 细粒度时间对齐 |
| A2 | GRU 深度 | `gru_layers=2` | （填） | 更强时序建模 |
| A3 | 视频帧数 | `num_video_frames` 16 vs 50 | （填） | 与 Wan 输出长度对齐 |
| A4 | 冻结侧掩码 | 无 manifest vs `frozen_side_manifest` | （填） | 是否利用赛方冻结标注 |
| A5 | 未来头 | 仅用 `VideoActionHead` vs 增加 `FutureActionHead` | （填） | 分段预测是否更稳 |

*若无独立验证集，可报告在 hold-out 任务 ID 上的轨迹 L2 或可视化对比。*

---

## 七、生成数据可视化示例（报告建议）

建议在 `report.pdf` 中每类任务选 **2～3 个** 样本页：

- **首末帧拼图**或**短片关键帧条带**（展示物体与手臂运动是否合理）。
- **轨迹曲线**：GT vs 预测各维度随时间对比（从 `action.txt` 用 matplotlib 绘制）。
- **（可选）注意力**：若需可导出 query-attention 权重（需改模型 `need_weights=True`，属加分项非必需）。

`generated_samples/` 目录可按任务子文件夹组织，与赛方「每个任务提供样本」一致。

---

## 八、模型检查点与元信息

### 8.1 检查点（不打包进 zip）

- **Wan2.2-TI2V-5B**：官方权重，选手自备；网盘链接 + 提取码写在 **模型说明文档**（可与 `info.md` 合并一节）。
- **动作头**：`action_head.pt` / `future_action_head.pt` 及同目录 **`dataset_meta.json`**、**`*_norm_stats.json`** 必须一并提供，否则推理统计量不一致。

### 8.2 生成数据元信息（建议在 `info.md` 或单独 `generated_meta.json` 提供）

- **规模**：总样本数、train/test 划分、每任务样本数。
- **分布**：指令长度分布、动作各维度均值方差（在标准化前后分别可列）、视频时长/帧数分布。
- **流水线版本**：`generate.py` 关键参数（`frame_num`、`guide_scale`、`size`）、是否使用 Qwen 改写、是否强制 50f/720p。

---

## 九、与赛方提交目录的对应关系

赛方要求：

```text
名称_学校.zip
├── code/
│   ├── model/
│   ├── data_generation/
│   ├── README.md
│   └── requirements.txt
├── report.pdf
├── generated_samples/
└── info.md
```

**本仓库映射建议**（提交前复制或软链，勿改赛方目录名）：

| 赛方路径 | 本仓库内容 |
|----------|------------|
| `code/model/` | `tools/action_head/`（`model.py`、`future_model.py`、`train*.py`、`infer*.py`、`dataset*.py` 等） |
| `code/data_generation/` | `tools/prepare_release_dataset.py`、`batch_infer_ti2v.py`、`batch_infer_ti2v_lora_diffsynth.py`、`optimize_*`、`rewrite_prompts_with_qwen_vl.py`、视频后处理脚本等；**另需包含或引用根目录** `generate.py` 及官方 Wan 源码树（可与 `code/wan_official/` 子目录说明） |
| `code/README.md` | 综合 `PROJECT_README.md` + 根目录官方安装说明中的**本赛题实际用到的命令** |
| `code/requirements.txt` | 根目录 `requirements.txt`；动作头读视频建议补充 **`decord`**（见 `video_io.py` 注释） |

完整算法包也可用仓库内 `bash scripts/package_algorithm.sh` 生成 tar（**排除**权重与大体积 mp4），再按上表重组为 zip。

---

## 十、复现最小命令清单（可放入 `code/README.md`）

```bash
# 环境（示例）
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install decord  # 若 video_io 使用 decord

# 准备 manifest
python tools/prepare_release_dataset.py --train_dir ... --test_dir ... --out_dir ...

# 批量生成视频（需先按官方说明下载 TI2V-5B 至 --ckpt_dir）
python tools/batch_infer_ti2v.py --manifest ... --repo_dir ... --ckpt_dir ... --output_dir ...

# 训练动作头
python -m tools.action_head.train --train_dir ... --out_dir ... --target_file action.txt ...

# 推理
python -m tools.action_head.infer --checkpoint .../action_head.pt --video ... --output ...
```

分布式训练示例见 `train.py` / `train_future.py` 文件内 `torchrun` 说明。

---

## 十一、注意事项摘要

- **权重不上传 zip**：Wan 与自训动作头均走网盘链接 + 提取码。
- **未训练 Wan 须声明**：报告与 `info.md` 中写清视频模型为官方预训练、仅推理。
- **时间与效率**：在 48 小时内完成全量生成需提前做小规模压测（单条耗时 × 样本数），必要时降低帧数或分批多机。

---

*文档版本：与仓库 `PROJECT_README.md` 描述一致；若你本地路径与 `/home/release` 等不同，以实际 manifest 与脚本参数为准。*
