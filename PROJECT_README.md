# Wan2.2 工程扩展说明（视频 + 动作流水线）

本仓库在官方 [Wan2.2](https://github.com/Wan-Video/Wan2.2) 基础上增加了比赛与批处理相关脚本。官方安装、模型下载与 `generate.py` 全量参数见根目录 **README.md** 与 **INSTALL.md**。本文档只描述**本工程实际用到的能力与目录约定**。

> **说明**：`/home/release` 等数据与生成结果在仓库外，**算法打包脚本只打包本目录 `Wan2.2` 源码**；权重与 release 产出需单独拷贝或挂载。

---

## 环境要求

- Python 虚拟环境：建议在仓库根目录 `python -m venv .venv && source .venv/bin/activate`
- 依赖：`pip install -r requirements.txt`（动作头读视频需 `decord`，见 `tools/action_head/video_io.py`）
- 权重：将 **Wan2.2-TI2V-5B** 等模型放在本目录下（如 `./Wan2.2-TI2V-5B`），由 `--ckpt_dir` 指定；**不随本算法包分发**
- 数据：训练/测试样本目录通常放在 `/home/release/train`、`/home/release/test` 等路径（与脚本默认或 manifest 内路径一致）

---

## 一、视频生成（Wan TI2V-5B）

### 1.1 单条推理（官方入口）

在仓库根目录执行：

```bash
python generate.py \
  --task ti2v-5B \
  --size 1280*704 \
  --ckpt_dir ./Wan2.2-TI2V-5B \
  --offload_model True \
  --convert_model_dtype \
  --t5_cpu \
  --image <末帧或参考图路径> \
  --prompt "<文本提示>" \
  --save_file <输出.mp4>
```

- **图生视频**：提供 `--image`；**文生视频**：不传 `--image`（见官方 README）。
- **提示词增强（可选）**：`--use_prompt_extend --prompt_extend_method local_qwen|dashscope`，本地需可加载 Qwen-VL；云端需 `DASH_API_KEY` 等（见官方 README「Using Prompt Extension」）。

### 1.2 批量 TI2V（本仓库脚本）

| 脚本 | 作用 |
|------|------|
| `tools/batch_infer_ti2v.py` | 按 **jsonl manifest** 批量调用根目录 `generate.py`，每行需含 `id`、`last_frame_path`、`instruction`，输出 `<id>.mp4` 到 `--output_dir`。支持 `--max_samples`、`--frame_num`、`--guide_scale`。 |
| `tools/batch_infer_ti2v_lora_diffsynth.py` | 基于 **DiffSynth** 管线 + 可选 LoRA 的批量推理（分辨率/帧数等与脚本参数一致）。 |

Manifest 可由 `tools/prepare_release_dataset.py` 从 `train`/`test` 目录生成，也可经 `tools/rewrite_prompts_with_qwen_vl.py` 等改写 `instruction` 后使用。

### 1.3 提示词优化与 Qwen-VL 改写

| 脚本 | 作用 |
|------|------|
| `tools/optimize_test_prompts.py` | 规则化短模板（英文），从 manifest 的 `instruction` 生成较固定句式。 |
| `tools/optimize_test_prompts_v2.py` | 读图尺寸 + 略强约束的英文 prompt。 |
| `tools/optimize_train_prompts_v2.py` | 针对 train manifest 的类似优化。 |
| `tools/rewrite_prompts_with_qwen_vl.py` | 用 **Qwen2.5-VL** 看图 + 读 `instruction_raw`，输出 JSON（含 `final_prompt` 等）并写新 manifest；可配置 `--model_id`、`--max_samples`。 |

### 1.4 数据准备

| 脚本 | 作用 |
|------|------|
| `tools/prepare_release_dataset.py` | 扫描 `train`/`test` 子目录，抽视频 **最后一帧** 为 jpg，生成含 `video_path`、`last_frame_path`、`instruction`、`action_path`、`joint_path` 的 jsonl 等。 |

### 1.5 DiffSynth 元数据

| 脚本 | 作用 |
|------|------|
| `tools/build_diffsynth_metadata.py` | 将 manifest jsonl 转为 DiffSynth 用的 `metadata.csv`（`video`, `prompt` 列）。 |

---

## 二、动作头（视频 → action / joint）

实现目录：`tools/action_head/`。

### 2.1 模型结构（概念）

- 输入：均匀采样的 **T 帧** RGB（默认 16 帧，224×224）。
- 骨干：**ResNet18** 逐帧特征 → **GRU** → 时间维 **线性插值** 到 **num_action_steps**（默认 96）→ 全连接输出 **action_dim**（与 CSV 列数一致，不含索引列）。
- 训练损失：**MSE**（在标准化后的动作空间）。

### 2.2 训练

```bash
cd /home/Wan2.2 && source .venv/bin/activate
python -m tools.action_head.train \
  --train_dir /path/to/train_root \
  --out_dir /path/to/run_dir \
  --target_file action.txt \
  --num_video_frames 16 \
  --num_action_steps 96 \
  --epochs 20 \
  --batch_size 4
```

- 每个样本子目录需含 **`video.mp4`** 与 **`action.txt`**（或 `--target_file joint.txt`）。
- 首次会在 `out_dir`（或 `--stats`）写入 **`action_norm_stats.json`**，并保存 **`dataset_meta.json`**、**`action_head.pt`**。

### 2.3 推理

```bash
python -m tools.action_head.infer \
  --checkpoint /path/to/run_dir/action_head.pt \
  --video /path/to/video.mp4 \
  --output /path/to/pred_action.txt \
  --start_index 0
```

- 依赖同目录 **`dataset_meta.json`** 与统计文件路径；输出为与训练一致的 **CSV 动作格式**（`csv_action.write_action_txt`）。

### 2.4 后处理：重采样与 Qwen 冻结

| 脚本 | 作用 |
|------|------|
| `tools/action_head/resample_actions.py` | 将 `*_action.txt` 沿时间 **线性重采样** 到指定行数（如 51）。 |
| `tools/action_head/resample_joints.py` | 同上，针对 `*_joint.txt`，默认输入目录可指向 `pred_joints_batch`。 |
| `tools/action_head/freeze_inactive_arm_with_qwen.py` | 用 Qwen-VL（或规则/manual）判 **active_arm**，对非活动臂做冻结；可选 **`--ref_action_dir`** 从 test 轨迹覆盖；可选 **`--active_arm_cache_jsonl`**、**`--aligned_preview_dir`**；`--freeze_predicted_arm_from_ref` 等开关见脚本内说明。 |

---

## 三、视频后处理与提交目录打包

| 脚本 | 作用 |
|------|------|
| `tools/pad_video_duplicate_last_frame.py` | 末尾 **克隆最后一帧** 补 1 帧（如 49→50），可调 `--skip_if_frames`。 |
| `tools/convert_video_to_30fps_1280x720.py` | 缩放 + pad + **fps=30**（不强制帧数，时长不变可能变帧数）。 |
| `tools/convert_video_force_50f_30fps_1280x720.py` | **强制 50 帧 + 30fps + 1280×720**（tpad/trim）。 |
| `tools/pack_sample_result_format.py` | 合并 `*_action.txt`、`*_joint.txt`、`<id>.mp4` 与 `test/<id>/instruction.txt` 为 **`sample_result` 风格**：`<id>/action.txt|joint.txt|video.mp4|instruction.txt`。 |

---

## 四、推荐流水线（与比赛提交对齐时）

1. **准备 manifest**：`prepare_release_dataset.py` 或手写 jsonl。  
2. **（可选）提示词**：`rewrite_prompts_with_qwen_vl.py` / `optimize_test_prompts*.py`。  
3. **批量出视频**：`batch_infer_ti2v.py` 或 DiffSynth 脚本。  
4. **（可选）视频规格**：补帧 → 强制 50f/30fps/720p（见第三节）。  
5. **动作头推理**：`action_head.infer` 得 `*_action.txt`；**（可选）** `freeze_inactive_arm_with_qwen.py`。  
6. **重采样**：`resample_actions.py` / `resample_joints.py` 对齐评测步数。  
7. **打包提交目录**：`pack_sample_result_format.py`。  
8. **压缩上传**：对输出目录 `zip`/`tar`（与算法包分离）。

---

## 五、路径与约定

- **必须在仓库根目录**执行 `python -m tools.xxx`，否则易出现 `No module named 'tools'`。  
- 权重路径、数据路径以你机器为准；manifest 内常为 **绝对路径**，换机需批量替换或重新生成。  
- 官方 Wan 能力（T2V、I2V、Animate、S2V 等）仍以 **README.md** 为准，本文件不重复罗列。

---

## 六、算法源码打包

使用仓库内脚本（**不包含** `.venv`、`Wan2.2-TI2V-5B` 权重目录、仓库内临时生成的大体积 `.mp4` 等）：

```bash
bash scripts/package_algorithm.sh
# 或指定输出路径：
bash scripts/package_algorithm.sh /tmp/Wan2.2_algorithm.tar.gz
```

生成物为 **gzip tar**，仅便于归档与传输；**模型权重与 `/home/release` 下生成结果需自行管理**，不在包内。
