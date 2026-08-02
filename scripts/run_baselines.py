"""Run the no-learning baselines (E0/E1) on a chopped evaluation set.

Usage:
  uv run python scripts/run_baselines.py --zarr data/scenes/sample.zarr
  uv run python scripts/run_baselines.py --zarr data/scenes/holdout.zarr

The zarr is chopped on first use (l5kit create_chopped_dataset, frame 100,
50-frame ground truth, min 10 future frames, agent filter 0.5); predictions
and a metrics CSV land in results/.
"""

import argparse
import csv
from pathlib import Path

from l5study.evaluation import STANDARD_BASELINES, chop_for_eval, run_predictors

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", type=Path, default=Path("data/scenes/sample.zarr"))
    parser.add_argument("--chop-frame", type=int, default=100)
    args = parser.parse_args()

    chopped = chop_for_eval(args.zarr, chop_frame=args.chop_frame)
    out_dir = RESULTS_DIR / f"baselines_{args.zarr.stem}"
    results = run_predictors(chopped, STANDARD_BASELINES, out_dir)

    metric_names = list(next(iter(results.values())).keys())
    header = f"{'model':20}" + "".join(f"{m[:28]:>30}" for m in metric_names)
    print("\n" + header)
    for name, metrics in sorted(
        results.items(), key=lambda kv: next(iter(kv[1].values()))
    ):
        print(f"{name:20}" + "".join(f"{metrics[m]:>30.3f}" for m in metric_names))

    csv_path = out_dir / "metrics.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", *metric_names])
        for name, metrics in results.items():
            writer.writerow([name, *[f"{metrics[m]:.6f}" for m in metric_names]])
    print(f"\nwritten: {csv_path}")


if __name__ == "__main__":
    main()
