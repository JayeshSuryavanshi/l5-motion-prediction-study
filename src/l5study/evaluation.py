"""Frozen evaluation protocol.

Scenes are chopped at a fixed frame with l5kit's create_chopped_dataset (the
same machinery that produced the competition test set): each qualifying agent
gets one prediction unit with a 50-frame future, ground truth goes to gt.csv,
and predictions written with write_pred_csv are scored by compute_metrics_csv
with the competition metric. Coordinates in both CSVs are world-frame offsets
from the agent's centroid at the prediction frame.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
from l5kit.data import ChunkedDataset
from l5kit.dataset import AgentDataset
from l5kit.evaluation import compute_metrics_csv, create_chopped_dataset, write_pred_csv
from l5kit.evaluation.metrics import (
    average_displacement_error_oracle,
    final_displacement_error_oracle,
    neg_multi_log_likelihood,
)
from l5kit.geometry import transform_points
from l5kit.rasterization import RenderContext, StubRasterizer
from tqdm import tqdm

from l5study import baselines
from l5study.config import stub_config

Predictor = Callable[
    [np.ndarray, np.ndarray, int, float], tuple[np.ndarray, np.ndarray]
]

METRICS = [
    neg_multi_log_likelihood,
    average_displacement_error_oracle,
    final_displacement_error_oracle,
]


def _single(fn) -> Predictor:
    def predictor(history, availabilities, num_future, dt):
        pred = fn(history, availabilities, num_future, dt=dt)
        return pred[None], np.array([1.0])

    return predictor


def _mixture(history, availabilities, num_future, dt):
    return baselines.kinematic_mixture(history, availabilities, num_future, dt=dt)


STANDARD_BASELINES: dict[str, Predictor] = {
    "stationary": lambda h, a, n, dt: (
        baselines.stationary(h, a, n)[None],
        np.array([1.0]),
    ),
    "constant_velocity": _single(baselines.constant_velocity),
    "constant_turn": _single(baselines.constant_turn),
    "kinematic_mixture": _mixture,
}


def history_oldest_first(sample: dict) -> tuple[np.ndarray, np.ndarray]:
    """l5kit orders history newest-first (index 0 = current frame); the
    baselines expect oldest-to-newest with the current frame last."""
    positions = np.asarray(sample["history_positions"], dtype=np.float64)[::-1]
    availabilities = np.asarray(sample["history_availabilities"], dtype=np.float64)[
        ::-1
    ]
    return positions, availabilities


def chop_for_eval(
    zarr_path: Path,
    chop_frame: int = 100,
    num_frames_gt: int = 50,
    min_frame_future: int = 10,
    filter_agents_threshold: float = 0.5,
) -> Path:
    """Idempotent wrapper around l5kit's create_chopped_dataset."""
    dest = zarr_path.parent / f"{zarr_path.stem}_chopped_{chop_frame}"
    if not (dest / "gt.csv").exists():
        create_chopped_dataset(
            str(zarr_path),
            filter_agents_threshold,
            chop_frame,
            num_frames_gt,
            min_frame_future,
        )
    return dest


def eval_dataset(
    chopped_dir: Path,
    history_num_frames: int = 10,
    future_num_frames: int = 50,
    step_time: float = 0.1,
) -> AgentDataset:
    zarr_paths = sorted(chopped_dir.glob("*.zarr"))
    if len(zarr_paths) != 1:
        raise FileNotFoundError(
            f"expected exactly one zarr in {chopped_dir}, found {zarr_paths}"
        )
    mask = np.load(chopped_dir / "mask.npz")["arr_0"]
    cfg = stub_config(history_num_frames, future_num_frames, step_time)
    render_context = RenderContext(
        np.asarray(cfg["raster_params"]["raster_size"]),
        np.asarray(cfg["raster_params"]["pixel_size"]),
        np.asarray(cfg["raster_params"]["ego_center"]),
        cfg["raster_params"]["set_origin_to_bottom"],
    )
    zarr_dataset = ChunkedDataset(str(zarr_paths[0])).open()
    return AgentDataset(
        cfg, zarr_dataset, StubRasterizer(render_context), agents_mask=mask
    )


def run_predictors(
    chopped_dir: Path,
    predictors: dict[str, Predictor],
    out_dir: Path,
    history_num_frames: int = 10,
    future_num_frames: int = 50,
    step_time: float = 0.1,
) -> dict[str, dict[str, float]]:
    """Run map-free predictors over a chopped eval set and score them.

    Returns {predictor_name: {metric_name: value}}; per-predictor prediction
    CSVs are left in out_dir for inspection.
    """
    dataset = eval_dataset(
        chopped_dir, history_num_frames, future_num_frames, step_time
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamps: list[int] = []
    track_ids: list[int] = []
    coords: dict[str, list[np.ndarray]] = {name: [] for name in predictors}
    confs: dict[str, list[np.ndarray]] = {name: [] for name in predictors}
    for sample in tqdm(dataset, desc=f"predicting {len(dataset)} agents"):
        history, availabilities = history_oldest_first(sample)
        timestamps.append(sample["timestamp"])
        track_ids.append(sample["track_id"])
        for name, predictor in predictors.items():
            preds, conf = predictor(
                history, availabilities, future_num_frames, step_time
            )
            world = np.stack(
                [
                    transform_points(p, sample["world_from_agent"])
                    - sample["centroid"][:2]
                    for p in preds
                ]
            )
            coords[name].append(world)
            confs[name].append(conf)
    results: dict[str, dict[str, float]] = {}
    for name in predictors:
        pred_csv = out_dir / f"pred_{name}.csv"
        stacked = np.asarray(coords[name])
        stacked_confs = np.asarray(confs[name])
        if stacked.shape[1] == 1:
            write_pred_csv(
                str(pred_csv),
                np.asarray(timestamps),
                np.asarray(track_ids),
                stacked[:, 0],
            )
        else:
            write_pred_csv(
                str(pred_csv),
                np.asarray(timestamps),
                np.asarray(track_ids),
                stacked,
                confs=stacked_confs,
            )
        results[name] = compute_metrics_csv(
            str(chopped_dir / "gt.csv"), str(pred_csv), METRICS
        )
    return results
