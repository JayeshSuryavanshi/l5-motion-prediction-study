"""Train E3/E4, the raster CNN, and score it on the frozen holdout.

Rasterization runs on the fly in DataLoader workers (no raster cache fits
this disk); the CNN trains on MPS with bf16 autocast. Checkpoints are written
periodically so an interrupted run loses at most one checkpoint interval.

Usage:
  uv run python scripts/train_e3.py --throughput          # ~3 min: measure samples/s, no training artifacts
  uv run python scripts/train_e3.py --samples 3000000 --max-hours 8
  uv run python scripts/train_e3.py --raster 224 ...      # E4 geometry
"""

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from l5study.config import raster_config
from l5study.evaluation import run_predictors
from l5study.metrics import multi_mode_nll_loss
from l5study.models import RasterCNN
from l5study.raster_data import RasterPoolDataset, train_pool_indices

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
IN_CHANNELS = 2 * (10 + 1) + 3


def run_name(args) -> str:
    return (
        f"e3_{args.backbone}_r{args.raster}_s{args.samples}"
        f"_b{args.batch}_lr{args.lr:g}_seed{args.seed}"
    )


def make_loader(args) -> DataLoader:
    pool = train_pool_indices(DATA)
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(pool, size=min(args.samples, len(pool)), replace=False)
    dataset = RasterPoolDataset(
        {"raster_size": args.raster, "history_num_frames": 10, "future_num_frames": 50},
        chosen,
        DATA,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
        pin_memory=False,
    )


def lr_schedule(step: int, total_steps: int, warmup_fraction: float = 0.03) -> float:
    warmup = max(1, int(total_steps * warmup_fraction))
    if step < warmup:
        return step / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def measure_throughput(args) -> None:
    loader = make_loader(args)
    device = torch.device(args.device)
    model = RasterCNN(IN_CHANNELS, backbone=args.backbone, pretrained=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    seen = 0
    loader_only_deadline = time.time() + 60
    iterator = iter(loader)
    for batch in iterator:
        seen += len(batch["image"])
        if time.time() > loader_only_deadline:
            break
    print(f"loader-only: {seen / 60:.1f} samples/s with {args.workers} workers")
    seen, started = 0, time.time()
    deadline = started + 120
    for batch in iterator:
        images = batch["image"].to(device).float() / 255.0
        targets = batch["target_positions"].to(device)
        avails = batch["target_availabilities"].to(device)
        with torch.autocast(device_type=args.device, dtype=torch.bfloat16):
            predictions, log_confidences = model(images)
        loss = multi_mode_nll_loss(
            targets, predictions.float(), log_confidences.float(), avails
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        seen += len(images)
        if time.time() > deadline:
            break
    rate = seen / (time.time() - started)
    print(f"end-to-end training: {rate:.1f} samples/s")
    print(f"8h overnight at this rate: {rate * 8 * 3600 / 1e6:.1f}M samples")


def train(args) -> None:
    name = run_name(args)
    device = torch.device(args.device)
    loader = make_loader(args)
    total_steps = math.ceil(min(args.samples, len(loader.dataset)) / args.batch)
    model = RasterCNN(IN_CHANNELS, backbone=args.backbone, pretrained=True).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(
        f"{name}: {params / 1e6:.2f}M params, {total_steps} steps planned", flush=True
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_schedule(step, total_steps)
    )
    checkpoint_dir = ROOT / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    started = time.time()
    deadline = started + args.max_hours * 3600
    next_checkpoint = started + 900
    running, seen, step = [], 0, 0
    stopped_early = False
    for batch in loader:
        images = batch["image"].to(device).float() / 255.0
        targets = batch["target_positions"].to(device)
        avails = batch["target_availabilities"].to(device)
        with torch.autocast(device_type=args.device, dtype=torch.bfloat16):
            predictions, log_confidences = model(images)
        loss = multi_mode_nll_loss(
            targets, predictions.float(), log_confidences.float(), avails
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        running.append(loss.detach())
        seen += len(images)
        step += 1
        now = time.time()
        if step % 100 == 0:
            mean_loss = torch.stack(running).mean().item()
            running = []
            print(
                f"step {step}/{total_steps}  seen {seen}  train NLL {mean_loss:8.2f}  "
                f"{seen / (now - started):.1f} samples/s  "
                f"({(now - started) / 3600:.2f}h)",
                flush=True,
            )
        if now > next_checkpoint:
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "args": vars(args),
                    "step": step,
                    "seen": seen,
                },
                checkpoint_dir / f"{name}.pt",
            )
            next_checkpoint = now + 900
        if now > deadline:
            stopped_early = True
            print(f"time budget reached at {seen} samples", flush=True)
            break
    model = model.cpu().eval()
    torch.save(
        {
            "state_dict": model.state_dict(),
            "args": vars(args),
            "step": step,
            "seen": seen,
        },
        checkpoint_dir / f"{name}.pt",
    )
    print(f"saved {checkpoint_dir / name}.pt  (early_stop={stopped_early})", flush=True)
    evaluate(args, model, name, seen)


def evaluate(args, model: RasterCNN, name: str, seen: int) -> None:
    device = torch.device(args.device)
    model = model.to(device)

    def predictor(sample, num_future, dt):
        image = torch.from_numpy(sample["image"]).unsqueeze(0).to(device)
        with torch.no_grad():
            predictions, log_confidences = model(image)
        return (
            predictions[0].float().cpu().numpy().astype(np.float64),
            np.exp(log_confidences[0].float().cpu().numpy().astype(np.float64)),
        )

    eval_cfg = raster_config(raster_size=args.raster)
    out_dir = ROOT / "results" / "e3_holdout"
    results = run_predictors(
        DATA / "scenes/holdout_chopped_100", {name: predictor}, out_dir, cfg=eval_cfg
    )
    metrics = results[name]
    for key, value in metrics.items():
        print(f"{key}: {value:.3f}")
    csv_path = out_dir / "metrics.csv"
    exists = csv_path.exists()
    with csv_path.open("a", newline="") as fh:
        writer = csv.writer(fh)
        if not exists:
            writer.writerow(
                [
                    "model",
                    "raster",
                    "samples_seen",
                    "batch",
                    "lr",
                    "seed",
                    *metrics.keys(),
                ]
            )
        writer.writerow(
            [name, args.raster, seen, args.batch, args.lr, args.seed]
            + [f"{v:.6f}" for v in metrics.values()]
        )
    print(f"written: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raster", type=int, default=128)
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--samples", type=int, default=3_000_000)
    parser.add_argument("--max-hours", type=float, default=8.0)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="mps" if torch.backends.mps.is_available() else "cpu"
    )
    parser.add_argument("--throughput", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.throughput:
        args.samples = min(args.samples, 100_000)
        measure_throughput(args)
        return
    train(args)


if __name__ == "__main__":
    main()
