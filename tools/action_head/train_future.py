#!/usr/bin/env python3
"""训练：过去 16 步 action + 切分后视频均匀 50 帧 -> 未来 action（重采样到 num_future_action_steps）。支持 torchrun 多卡 DDP。"""
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

from .future_dataset import FutureActionDataset
from .future_model import FutureActionHead


def _dist_info() -> tuple[bool, int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return (
            True,
            int(os.environ["RANK"]),
            int(os.environ["WORLD_SIZE"]),
            int(os.environ.get("LOCAL_RANK", 0)),
        )
    return False, 0, 1, 0


def main() -> None:
    p = argparse.ArgumentParser(
        description="Train future action head (past action + future video).",
        epilog="4 卡: torchrun --standalone --nproc_per_node=4 -m tools.action_head.train_future ...",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--train_dir", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--target_file", type=str, default="action.txt")
    p.add_argument("--stats", type=Path, default=None)
    p.add_argument("--past-len", type=int, default=16)
    p.add_argument("--future-video-frames", type=int, default=50, help="与推理 Wan 视频帧数对齐")
    p.add_argument(
        "--num-future-action-steps",
        type=int,
        default=51,
        help="将「切分点到 action 末尾」重采样为该步数（训练目标维度）",
    )
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--gru-layers", type=int, default=1)
    p.add_argument("--attn-heads", type=int, default=8)
    p.add_argument("--attn-dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4, help="每进程 batch；4 卡全局约为 batch_size*4")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    distributed, rank, world_size, local_rank = _dist_info()
    if distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    is_main = rank == 0
    show = is_main and (not args.no_progress)

    if is_main:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    stats_path = args.stats or (args.out_dir / "future_action_norm_stats.json")
    ds_seed = args.seed + rank

    if not stats_path.exists():
        if is_main:
            ds_fit = FutureActionDataset(
                args.train_dir,
                target_file=args.target_file,
                past_len=args.past_len,
                future_video_frames=args.future_video_frames,
                num_future_action_steps=args.num_future_action_steps,
                image_size=args.image_size,
                stats_path=None,
                fit_stats=True,
                manifest_path=args.manifest,
                seed=ds_seed,
            )
            ds_fit.save_stats(stats_path)
    if distributed:
        dist.barrier()

    ds = FutureActionDataset(
        args.train_dir,
        target_file=args.target_file,
        past_len=args.past_len,
        future_video_frames=args.future_video_frames,
        num_future_action_steps=args.num_future_action_steps,
        image_size=args.image_size,
        stats_path=stats_path,
        fit_stats=False,
        manifest_path=args.manifest,
        seed=ds_seed,
    )

    if is_main:
        with open(args.out_dir / "future_dataset_meta.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "header": ds.header,
                    "target_file": args.target_file,
                    "past_len": args.past_len,
                    "future_video_frames": args.future_video_frames,
                    "num_future_action_steps": args.num_future_action_steps,
                    "image_size": args.image_size,
                    "stats": str(stats_path.resolve()),
                    "gru_layers": args.gru_layers,
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

    model = FutureActionHead(
        past_len=args.past_len,
        future_video_frames=args.future_video_frames,
        num_future_action_steps=args.num_future_action_steps,
        action_dim=ds.action_dim,
        gru_layers=args.gru_layers,
        attn_heads=args.attn_heads,
        attn_dropout=args.attn_dropout,
    ).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
    raw = model.module if distributed else model
    opt = torch.optim.AdamW(raw.parameters(), lr=args.lr)

    epbar = tqdm(range(args.epochs), desc="epoch", disable=not show)
    for ep in epbar:
        if sampler is not None:
            sampler.set_epoch(ep)
        model.train()
        tot, n = 0.0, 0
        inner = tqdm(loader, desc=f"train {ep + 1}/{args.epochs}", leave=False, disable=not show)
        for batch in inner:
            pa = batch["past_action"].to(device, non_blocking=True)
            fv = batch["future_video"].to(device, non_blocking=True)
            y = batch["future_action"].to(device, non_blocking=True)
            m = batch["loss_mask"].to(device, non_blocking=True).unsqueeze(1)
            pred = model(pa, fv)
            e2 = (pred - y) ** 2
            me = m.expand_as(e2)
            loss = (e2 * me).sum() / me.sum().clamp_min(1e-8)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * pa.size(0)
            n += pa.size(0)
            inner.set_postfix(loss=f"{loss.item():.4f}")
        if distributed:
            t = torch.tensor([tot, float(n)], device=device, dtype=torch.float64)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            tot, n = t[0].item(), int(t[1].item())
        avg = tot / max(n, 1)
        if show:
            epbar.set_postfix(avg_loss=f"{avg:.6f}")
        if args.no_progress and is_main:
            print(f"epoch {ep + 1}/{args.epochs} loss {avg:.6f}")

    if is_main:
        ckpt = args.out_dir / "future_action_head.pt"
        torch.save(
            {
                "model": raw.state_dict(),
                "target_file": args.target_file,
                "past_len": args.past_len,
                "future_video_frames": args.future_video_frames,
                "num_future_action_steps": args.num_future_action_steps,
                "image_size": args.image_size,
                "action_dim": ds.action_dim,
                "output_dim": ds.action_dim,
                "gru_layers": args.gru_layers,
                "attn_heads": args.attn_heads,
                "attn_dropout": args.attn_dropout,
            },
            ckpt,
        )
        print(f"saved {ckpt}")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
