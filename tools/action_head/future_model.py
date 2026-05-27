from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class FutureActionHead(nn.Module):
    """
    过去 action [B, past_len, D] + 未来段视频 [B, T_vid, 3, H, W]
    -> 未来 action [B, num_future_action_steps, D]

    视频用 ResNet18 逐帧 + GRU；过去 action 经 Linear+GRU 得向量；
    与视频隐状态拼接为 memory，经 query cross-attention + FFN 读出各未来步。
    """

    def __init__(
        self,
        past_len: int,
        future_video_frames: int,
        num_future_action_steps: int,
        action_dim: int,
        hidden: int = 256,
        backbone_dim: int = 512,
        gru_layers: int = 1,
        attn_heads: int = 8,
        attn_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.past_len = past_len
        self.future_video_frames = future_video_frames
        self.num_future_action_steps = num_future_action_steps
        self.action_dim = action_dim
        self.hidden = hidden

        if hidden % attn_heads != 0:
            raise ValueError(f"hidden={hidden} 必须能被 attn_heads={attn_heads} 整除")

        self.past_in = nn.Linear(action_dim, hidden)
        drop = 0.1 if gru_layers > 1 else 0.0
        self.past_gru = nn.GRU(
            hidden, hidden, batch_first=True, num_layers=gru_layers, dropout=drop
        )

        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])
        self.vid_proj = nn.Linear(backbone_dim, hidden)
        self.vid_gru = nn.GRU(
            hidden, hidden, batch_first=True, num_layers=gru_layers, dropout=drop
        )

        self.future_queries = nn.Parameter(torch.zeros(1, num_future_action_steps, hidden))
        nn.init.trunc_normal_(self.future_queries, std=0.02)
        mem_len = 1 + future_video_frames
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

    def forward(self, past_action: torch.Tensor, future_video: torch.Tensor) -> torch.Tensor:
        """
        past_action: [B, past_len, D]
        future_video: [B, T_vid, 3, H, W]
        returns [B, num_future_action_steps, D]
        """
        b = past_action.shape[0]
        xp = F.relu(self.past_in(past_action))
        _, hp = self.past_gru(xp)
        past_vec = hp[-1]  # [B, hidden]

        t = future_video.shape[1]
        x = future_video.reshape(b * t, future_video.shape[2], future_video.shape[3], future_video.shape[4])
        feat = self.cnn(x).flatten(1)
        feat = feat.reshape(b, t, -1)
        feat = F.relu(self.vid_proj(feat))
        vid_h, _ = self.vid_gru(feat)  # [B, T_vid, hidden]

        mem = torch.cat([past_vec.unsqueeze(1), vid_h], dim=1)  # [B, 1+T_vid, hidden]

        q = self.future_queries.expand(b, -1, -1)
        attn_out, _ = self.cross_attn(q, mem, mem, need_weights=False)
        z = self.ln1(q + attn_out)
        z = self.ln2(z + self.ff(z))
        return self.head(z)


def future_action_head_from_payload(payload: dict) -> FutureActionHead:
    return FutureActionHead(
        past_len=int(payload["past_len"]),
        future_video_frames=int(payload["future_video_frames"]),
        num_future_action_steps=int(payload["num_future_action_steps"]),
        action_dim=int(payload.get("output_dim", payload.get("action_dim", 26))),
        gru_layers=int(payload.get("gru_layers", 1)),
        attn_heads=int(payload.get("attn_heads", 8)),
        attn_dropout=float(payload.get("attn_dropout", 0.1)),
    )
