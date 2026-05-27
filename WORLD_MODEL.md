# Wan Video World Model

将 [Wan2.2](https://github.com/Wan-Video/Wan2.2) **TI2V-5B** 封装为可迭代的**视频世界模型**：以当前观测图像为状态、以自然语言为高层动作，预测下一段视觉轨迹，并可选地回归机器人动作空间。

## 世界模型形式化

| 符号 | 含义 |
|------|------|
| \(s_t\) | 当前观测（RGB 图像，通常为上一段视频的末帧） |
| \(a_t\) | 高层动作（文本指令 / prompt） |
| \(o_{t+1}\) | 模型输出（生成的短视频 `mp4`） |
| \(\hat{u}_{t+1}\) | 可选：动作头从 \(o_{t+1}\) 预测的低维轨迹 |

转移函数：

\[
o_{t+1} = f_{\text{Wan}}(s_t, a_t), \quad s_{t+1} = \text{LastFrame}(o_{t+1})
\]

可选：

\[
\hat{u}_{t+1} = g_{\text{action}}(o_{t+1})
\]

其中 \(f_{\text{Wan}}\) 为 Wan2.2 TI2V 扩散生成，\(g_{\text{action}}\) 为自研 `VideoActionHead`（ResNet18 + GRU，见 `tools/action_head/`）。

## 架构示意

```
                    ┌─────────────────────┐
  s_t (image) ─────►│  Wan2.2 TI2V-5B     │────► o_{t+1} (video)
  a_t (text)  ─────►│  (generate.py)      │
                    └──────────┬──────────┘
                               │
                    LastFrame  ▼
                    s_{t+1} (image)
                               │
                    ┌──────────▼──────────┐
                    │  VideoActionHead    │────► û_{t+1} (optional)
                    │  (tools/action_head)│
                    └─────────────────────┘
```

## 快速开始

### 1. 安装依赖

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install decord  # 动作头与世界模型读视频
```

### 2. 下载 Wan 权重

见 [weights/README.md](weights/README.md)。

### 3. Python API

```python
from pathlib import Path
from world_model import WanVideoWorldModel, WorldModelConfig

cfg = WorldModelConfig(
    ckpt_dir=Path("./Wan2.2-TI2V-5B"),
    action_head_checkpoint=Path("./action_head_runs/my_run/action_head.pt"),  # 可选
)
model = WanVideoWorldModel(cfg)

state = model.reset("examples/i2v_input.JPG")
result = model.step(state, "The robot arm picks up the red block.")
print(result.video_path, result.next_image_path)

# 多步 rollout
final_state, history = model.rollout(
    "examples/i2v_input.JPG",
    ["reach toward the cup", "grasp the cup handle"],
)
```

### 4. 命令行

在仓库根目录执行：

```bash
python -m world_model.cli \
  --image examples/i2v_input.JPG \
  --instruction "The robotic arm moves forward slowly." \
  --ckpt_dir ./Wan2.2-TI2V-5B \
  --work_dir ./world_model_runs/demo
```

多步：

```bash
python -m world_model.cli \
  --image frame0.jpg \
  --instruction "open the drawer" \
  --instruction "reach inside the drawer" \
  --ckpt_dir ./Wan2.2-TI2V-5B \
  --action_head ./action_head_runs/my_run/action_head.pt
```

## 与完整流水线的关系

本仓库在官方 Wan2.2 之上扩展了比赛/批处理工具链，详见 [PROJECT_README.md](PROJECT_README.md)：

- 批量 TI2V：`tools/batch_infer_ti2v.py`
- 动作头训练/推理：`tools/action_head/`
- 视频规格与提交打包：`tools/pack_sample_result_format.py` 等

技术报告骨架：[SUBMISSION_TECHNICAL_DOC.md](SUBMISSION_TECHNICAL_DOC.md)

## 引用

若使用 Wan2.2 主干，请引用官方论文与仓库。动作头为本仓库扩展模块，训练数据与权重需自行准备。
