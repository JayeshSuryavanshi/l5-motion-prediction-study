"""Train E2, the history-only MLP (no map), and score it on the frozen holdout.

Training agents are sampled uniformly from the train pool (scenes 0-15,264 of
train.zarr; the holdout scenes are excluded by agent-row boundary) using the
competition agents_mask bundled with the dataset. Features are cached to
data/cache/ so re-runs skip extraction.

Usage:
  uv run python scripts/train_e2.py --dry-run     # 5k samples, 2 epochs
  uv run python scripts/train_e2.py               # 200k samples, 20 epochs
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from l5kit.data import ChunkedDataset
from l5kit.dataset import AgentDataset
from l5kit.rasterization import RenderContext, StubRasterizer

from l5study.config import stub_config
from l5study.evaluation import run_predictors
from l5study.features import FEATURE_DIM, extract_features, sample_features
from l5study.metrics import multi_mode_nll_loss
from l5study.models import HistoryMLP

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
HOLDOUT_SCENE_START = 15265


def train_pool_dataset(cfg: dict) -> tuple[AgentDataset, np.ndarray]:
    zarr_dataset = ChunkedDataset(str(DATA / "scenes/train.zarr")).open()
    render_context = RenderContext(
        np.asarray(cfg["raster_params"]["raster_size"]),
        np.asarray(cfg["raster_params"]["pixel_size"]),
        np.asarray(cfg["raster_params"]["ego_center"]),
        cfg["raster_params"]["set_origin_to_bottom"],
    )
    dataset = AgentDataset(cfg, zarr_dataset, StubRasterizer(render_context))
    boundary_frame = zarr_dataset.scenes[HOLDOUT_SCENE_START]["frame_index_interval"][0]
    boundary_agent = zarr_dataset.frames[boundary_frame]["agent_index_interval"][0]
    pool = np.nonzero(dataset.agents_indices < boundary_agent)[0]
    return dataset, pool


def run_name(args) -> str:
    return (
        f"e2_mlp_s{args.samples}_h{args.hidden}_d{args.depth}"
        f"_lr{args.lr:g}_e{args.epochs}_seed{args.seed}"
    )


def load_features(args, cfg: dict):
    history = cfg["model_params"]["history_num_frames"]
    cache = (
        DATA
        / "cache"
        / f"e2_{args.samples}_{args.seed}_hf{history}_b{HOLDOUT_SCENE_START}.npz"
    )
    if cache.exists():
        stored = np.load(cache)
        return stored["x"], stored["y"], stored["a"]
    dataset, pool = train_pool_dataset(cfg)
    print(f"train pool: {len(pool)} eligible agents; sampling {args.samples}")
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(pool, size=args.samples, replace=False)
    x, y, a = extract_features(dataset, indices)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, x=x, y=y, a=a)
    return x, y, a


def train(args, x, y, a) -> HistoryMLP:
    device = torch.device(args.device)
    split = int(len(x) * 0.95)
    train_tensors = [torch.from_numpy(t[:split]).to(device) for t in (x, y, a)]
    val_tensors = [torch.from_numpy(t[split:]).to(device) for t in (x, y, a)]
    model = HistoryMLP(FEATURE_DIM, hidden_dim=args.hidden, depth=args.depth).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"HistoryMLP: {params / 1e6:.2f}M params on {device}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    generator = torch.Generator().manual_seed(args.seed)
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(split, generator=generator)
        epoch_loss = torch.zeros((), device=device)
        for lo in range(0, split, args.batch):
            batch = order[lo : lo + args.batch]
            features, targets, avails = (t[batch] for t in train_tensors)
            predictions, log_confidences = model(features)
            loss = multi_mode_nll_loss(targets, predictions, log_confidences, avails)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach() * len(batch)
        scheduler.step()
        model.eval()
        with torch.no_grad():
            predictions, log_confidences = model(val_tensors[0])
            val_loss = multi_mode_nll_loss(
                val_tensors[1], predictions, log_confidences, val_tensors[2]
            ).item()
        print(
            f"epoch {epoch + 1:2d}/{args.epochs}  train NLL {epoch_loss.item() / split:8.2f}  "
            f"val NLL {val_loss:8.2f}  ({time.time() - started:.0f}s)",
            flush=True,
        )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=200_000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="mps" if torch.backends.mps.is_available() else "cpu"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        args.samples, args.epochs = 5_000, 2

    torch.manual_seed(args.seed)
    cfg = stub_config()
    x, y, a = load_features(args, cfg)
    model = train(args, x, y, a)

    name = run_name(args)
    model = model.cpu().eval()
    checkpoint = ROOT / "checkpoints" / f"{name}.pt"
    checkpoint.parent.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "args": vars(args)}, checkpoint)
    print(f"saved {checkpoint}")

    def predictor(sample, num_future, dt):
        features = torch.from_numpy(sample_features(sample))[None]
        with torch.no_grad():
            predictions, log_confidences = model(features)
        return (
            predictions[0].numpy().astype(np.float64),
            np.exp(log_confidences[0].numpy().astype(np.float64)),
        )

    out_dir = ROOT / "results" / "e2_holdout"
    results = run_predictors(
        DATA / "scenes/holdout_chopped_100", {name: predictor}, out_dir
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
                    "samples",
                    "epochs",
                    "hidden",
                    "depth",
                    "lr",
                    "seed",
                    *metrics.keys(),
                ]
            )
        writer.writerow(
            [
                name,
                args.samples,
                args.epochs,
                args.hidden,
                args.depth,
                args.lr,
                args.seed,
            ]
            + [f"{v:.6f}" for v in metrics.values()]
        )
    print(f"written: {csv_path}")


if __name__ == "__main__":
    main()
