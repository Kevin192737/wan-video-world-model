#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .dataset import VideoActionDataset
from .model import VideoActionHead


def _dist_info() -> tuple[bool, int, int, int]:
    """是否分布式、rank、world_size、local_rank（单卡时 world=1, rank=0）。"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", 0))
        return True, rank, world, local
    return False, 0, 1, 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train video -> action head（支持 torchrun 多卡 DDP）。",
        epilog=(
            "单机 4 卡示例：\n"
            "  torchrun --standalone --nproc_per_node=4 -m tools.action_head.train \\\n"
            "    --train_dir ... --out_dir ... （其余参数同单卡；--batch_size 为每卡 batch）\n"
            "与「50 帧 Wan 视频 -> 51 步 action」：\n"
            "  --num_video_frames 50 --num_action_steps 51 --use-query-decoder --gru-layers 2"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--train_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--target_file",
        type=str,
        default="action.txt",
        help="Supervision file name inside each sample dir: action.txt or joint.txt",
    )
    parser.add_argument("--stats", type=Path, default=None, help="norm stats json; created if missing")
    parser.add_argument("--num_video_frames", type=int, default=16)
    parser.add_argument("--num_action_steps", type=int, default=96)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument(
        "--gru-layers",
        type=int,
        default=1,
        help="GRU 层数；2 层时层间 dropout=0.1，略增时序建模能力",
    )
    parser.add_argument(
        "--use-query-decoder",
        action="store_true",
        help="用可学习 query + 跨时间注意力读出各 action 步（替代仅线性插值），"
        "适合 T_vid 与 num_action_steps 接近且需细对齐的场景",
    )
    parser.add_argument("--attn-heads", type=int, default=8)
    parser.add_argument("--attn-dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="每进程 batch；DDP 时全局 batch = batch_size * GPU 数",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_progress", action="store_true", help="disable tqdm bars (for logs)")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="frozen_side_manifest.csv：按 case 的 frozen_side 对损失做掩码，"
        "仅运动侧关节+手指参与 MSE（冻结侧权重为 0）",
    )
    args = parser.parse_args()

    distributed, rank, world_size, local_rank = _dist_info()
    if distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    is_main = rank == 0
    show_bar = is_main and (not args.no_progress)

    if is_main:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    stats_path = args.stats or (args.out_dir / "action_norm_stats.json")

    if not stats_path.exists():
        if is_main:
            ds_fit = VideoActionDataset(
                args.train_dir,
                target_file=args.target_file,
                num_video_frames=args.num_video_frames,
                num_action_steps=args.num_action_steps,
                image_size=args.image_size,
                stats_path=None,
                fit_stats=True,
                manifest_path=args.manifest,
            )
            ds_fit.save_stats(stats_path)
    if distributed:
        dist.barrier()

    ds = VideoActionDataset(
        args.train_dir,
        target_file=args.target_file,
        num_video_frames=args.num_video_frames,
        num_action_steps=args.num_action_steps,
        image_size=args.image_size,
        stats_path=stats_path,
        fit_stats=False,
        manifest_path=args.manifest,
    )

    if is_main:
        with open(args.out_dir / "dataset_meta.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "header": ds.header,
                    "target_file": args.target_file,
                    "num_action_steps": args.num_action_steps,
                    "num_video_frames": args.num_video_frames,
                    "image_size": args.image_size,
                    "stats": str(stats_path.resolve()),
                    "gru_layers": args.gru_layers,
                    "use_query_decoder": args.use_query_decoder,
                    "attn_heads": args.attn_heads,
                    "attn_dropout": args.attn_dropout,
                    "manifest": str(args.manifest.resolve()) if args.manifest else None,
                    "distributed_world_size": world_size,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
    if distributed:
        dist.barrier()

    sampler = (
        DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
        if distributed
        else None
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    if distributed and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = VideoActionHead(
        num_action_steps=args.num_action_steps,
        action_dim=ds.action_dim,
        gru_layers=args.gru_layers,
        use_query_decoder=args.use_query_decoder,
        attn_heads=args.attn_heads,
        attn_dropout=args.attn_dropout,
    ).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    raw = model.module if distributed else model
    opt = torch.optim.AdamW(raw.parameters(), lr=args.lr)

    epoch_pbar = tqdm(
        range(args.epochs),
        desc="epoch",
        disable=not show_bar,
    )
    for epoch in epoch_pbar:
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        total = 0.0
        n = 0
        batch_pbar = tqdm(
            loader,
            desc=f"train {epoch + 1}/{args.epochs}",
            leave=False,
            disable=not show_bar,
        )
        for batch in batch_pbar:
            vid = batch["video"].to(device, non_blocking=True)
            act = batch["action"].to(device, non_blocking=True)
            pred = model(vid)
            m = batch["loss_mask"].to(device, non_blocking=True).unsqueeze(1)  # [B, 1, D]
            err2 = (pred - act) ** 2
            m_exp = m.expand_as(err2)
            denom = m_exp.sum().clamp_min(1e-8)
            loss = (err2 * m_exp).sum() / denom
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * vid.size(0)
            n += vid.size(0)
            batch_pbar.set_postfix(loss=f"{loss.item():.4f}")

        if distributed:
            t = torch.tensor([total, float(n)], device=device, dtype=torch.float64)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            total, n = t[0].item(), int(t[1].item())
        avg = total / max(n, 1)
        if show_bar:
            epoch_pbar.set_postfix(avg_loss=f"{avg:.6f}")
        if args.no_progress and is_main:
            print(f"epoch {epoch + 1}/{args.epochs} loss {avg:.6f}")

    if is_main:
        ckpt = args.out_dir / "action_head.pt"
        state = raw.state_dict()
        torch.save(
            {
                "model": state,
                "target_file": args.target_file,
                "num_action_steps": args.num_action_steps,
                "num_video_frames": args.num_video_frames,
                "image_size": args.image_size,
                "action_dim": ds.action_dim,
                "output_dim": ds.action_dim,
                "gru_layers": args.gru_layers,
                "use_query_decoder": args.use_query_decoder,
                "attn_heads": args.attn_heads,
                "attn_dropout": args.attn_dropout,
                "manifest": str(args.manifest.resolve()) if args.manifest else None,
            },
            ckpt,
        )
        print(f"saved {ckpt}")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
