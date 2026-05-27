from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class VideoActionHead(nn.Module):
    """
    Uniformly sampled video frames -> per-frame ResNet18 features ->
    GRU 时间编码 ->（可选）可学习 query 做跨时间注意力读出 num_action_steps 步，
    否则对隐状态做线性时间上采样后接全连接。

    针对「~50 帧生成视频 -> 51 步 action」建议训练时设 num_video_frames=50、num_action_steps=51，
    并开启 use_query_decoder，使每一步 action 显式 attend 整段视频时间轴，而非仅依赖插值。
    """

    def __init__(
        self,
        num_action_steps: int,
        action_dim: int = 26,
        hidden: int = 256,
        backbone_dim: int = 512,
        gru_layers: int = 1,
        use_query_decoder: bool = False,
        attn_heads: int = 8,
        attn_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_action_steps = num_action_steps
        self.action_dim = action_dim
        self.hidden = hidden
        self.use_query_decoder = use_query_decoder
        self.gru_layers = gru_layers

        if hidden % attn_heads != 0:
            raise ValueError(f"hidden={hidden} 必须能被 attn_heads={attn_heads} 整除")

        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])
        self.proj = nn.Linear(backbone_dim, hidden)
        drop = 0.1 if gru_layers > 1 else 0.0
        self.gru = nn.GRU(
            hidden, hidden, batch_first=True, num_layers=gru_layers, dropout=drop
        )

        if use_query_decoder:
            self.action_queries = nn.Parameter(torch.zeros(1, num_action_steps, hidden))
            nn.init.trunc_normal_(self.action_queries, std=0.02)
            self.cross_attn = nn.MultiheadAttention(
                hidden, attn_heads, dropout=attn_dropout, batch_first=True
            )
            self.ln1 = nn.LayerNorm(hidden)
            self.ff = nn.Sequential(
                nn.Linear(hidden, hidden * 2),
                nn.GELU(),
                nn.Dropout(attn_dropout),
                nn.Linear(hidden * 2, hidden),
            )
            self.ln2 = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, action_dim)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        video: [B, T_vid, C, H, W]
        returns: [B, num_action_steps, action_dim]
        """
        b, t, c, h, w = video.shape
        x = video.reshape(b * t, c, h, w)
        feat = self.cnn(x).flatten(1)  # [B*T, 512]
        feat = feat.reshape(b, t, -1)
        feat = F.relu(self.proj(feat))
        memory, _ = self.gru(feat)  # [B, T_vid, hidden]

        if self.use_query_decoder:
            q = self.action_queries.expand(b, -1, -1)  # [B, T_act, hidden]
            attn_out, _ = self.cross_attn(q, memory, memory, need_weights=False)
            x = self.ln1(q + attn_out)
            x = self.ln2(x + self.ff(x))
            return self.head(x)

        out = memory.transpose(1, 2)  # [B, hidden, T_vid]
        out = F.interpolate(
            out, size=self.num_action_steps, mode="linear", align_corners=False
        )
        out = out.transpose(1, 2)  # [B, T_act, hidden]
        return self.head(out)


def video_action_head_from_payload(payload: dict) -> VideoActionHead:
    """从 train 保存的 checkpoint dict 构建与训练时一致的模型（含可选 query 解码器）。"""
    return VideoActionHead(
        num_action_steps=int(payload["num_action_steps"]),
        action_dim=int(payload.get("output_dim", payload.get("action_dim", 26))),
        gru_layers=int(payload.get("gru_layers", 1)),
        use_query_decoder=bool(payload.get("use_query_decoder", False)),
        attn_heads=int(payload.get("attn_heads", 8)),
        attn_dropout=float(payload.get("attn_dropout", 0.1)),
    )
