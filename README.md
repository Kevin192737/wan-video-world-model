# Wan Video World Model

以**图像状态 + 文本动作**驱动视觉动力学的世界模型：给定当前观测与指令，预测下一段视频，并将末帧作为下一时刻状态；可选地将视频映射为机器人动作轨迹。

> 视觉动力学基于 [Wan2.2 TI2V-5B](https://github.com/Wan-Video/Wan2.2)。本仓库**不包含**权重与数据集，见 [weights/README.md](weights/README.md)。

## 模型定义

| 符号 | 含义 |
|------|------|
| \(s_t\) | 当前状态（RGB 图像，多为上一段视频末帧） |
| \(a_t\) | 高层动作（自然语言指令） |
| \(o_{t+1}\) | 预测的下一段视频 |
| \(\hat{u}_{t+1}\) | 可选：动作头输出的低维轨迹 |

\[
o_{t+1} = f(s_t, a_t), \quad s_{t+1} = \text{LastFrame}(o_{t+1})
\]

可选：\(\hat{u}_{t+1} = g(o_{t+1})\)，其中 \(g\) 为 `VideoActionHead`（ResNet18 + GRU）。

```
  s_t (image) ──►  TI2V 视频生成  ──► o_{t+1} (video)
  a_t (text)  ──►       │              │
                        └── LastFrame ─┴──► s_{t+1}
                               │
                        [可选] 动作头 ──► û_{t+1}
```

## 安装

```bash
git clone https://github.com/Kevin192737/wan-video-world-model.git
cd wan-video-world-model
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install decord
```

下载 TI2V-5B 权重（世界模型默认后端）：

```bash
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./Wan2.2-TI2V-5B
```

## 快速开始

**命令行**（在仓库根目录）：

```bash
python -m world_model.cli \
  --image examples/i2v_input.JPG \
  --instruction "The robot arm reaches toward the cup on the table." \
  --ckpt_dir ./Wan2.2-TI2V-5B
```

多步 rollout：

```bash
python -m world_model.cli \
  --image frame0.jpg \
  --instruction "open the drawer" \
  --instruction "reach inside" \
  --ckpt_dir ./Wan2.2-TI2V-5B \
  --action_head ./action_head_runs/my_run/action_head.pt
```

**Python API**：

```python
from pathlib import Path
from world_model import WanVideoWorldModel, WorldModelConfig

model = WanVideoWorldModel(WorldModelConfig(ckpt_dir=Path("./Wan2.2-TI2V-5B")))

state = model.reset("examples/i2v_input.JPG")
result = model.step(state, "Pick up the red block.")

final_state, history = model.rollout(
    "examples/i2v_input.JPG",
    ["move the arm left", "grasp the handle"],
)
```

## 配置说明

| 参数 | 说明 |
|------|------|
| `ckpt_dir` | TI2V-5B 权重目录 |
| `action_head_checkpoint` | 可选，`action_head.pt` 路径 |
| `frame_num` | 每步生成帧数（默认 49） |
| `guide_scale` | CFG 引导强度（默认 6.0） |
| `work_dir` | 输出视频与 rollout 日志目录 |

更多接口与形式化说明见 [WORLD_MODEL.md](WORLD_MODEL.md)。

## 仓库结构

| 路径 | 说明 |
|------|------|
| `world_model/` | 世界模型 API（`reset` / `step` / `rollout`） |
| `tools/action_head/` | 视频 → 动作轨迹（可选模块） |
| `tools/` | 批量推理、数据 manifest、后处理 |
| `weights/` | 权重下载说明 |
| `release/` | 本地数据集说明（不上传 Git） |

扩展流水线（批量 TI2V、动作头训练、提交打包）见 [PROJECT_README.md](PROJECT_README.md)。

## 引用与许可

- 世界模型封装与动作头扩展：本仓库。
- 视频生成主干：请引用 [Wan2.2 论文](https://arxiv.org/abs/2503.20314) 与 [官方仓库](https://github.com/Wan-Video/Wan2.2)。
- 许可：[LICENSE.txt](LICENSE.txt)（Apache 2.0）。

官方 Wan 完整文档（T2V / I2V / Animate / S2V 等）见 [WAN_OFFICIAL_README.md](WAN_OFFICIAL_README.md) 与 [INSTALL.md](INSTALL.md)。
