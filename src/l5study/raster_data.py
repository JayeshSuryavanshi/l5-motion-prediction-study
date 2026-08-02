"""Streaming raster data for CNN training (E3/E4).

Rasterization happens on the fly inside DataLoader workers. The rasterizing
AgentDataset is built lazily on first access in each worker process, so the
heavy l5kit objects (zarr handles, protobuf map, rasterizer) are constructed
per worker instead of being pickled across the process boundary. Images cross
the worker queue as uint8 to quarter the IPC volume; scale back to [0, 1]
floats on the training device.
"""

from pathlib import Path

import numpy as np
import torch
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import AgentDataset
from l5kit.rasterization import RenderContext, StubRasterizer, build_rasterizer
from torch.utils.data import Dataset

from l5study.config import HOLDOUT_SCENE_START, raster_config, stub_config


def train_pool_indices(data_root: Path) -> np.ndarray:
    """Dataset indices (into an AgentDataset over train.zarr with default
    masking) whose agents live strictly before the holdout scene boundary.

    Index identity note: the mapping from dataset index to agent row depends
    only on the agents_mask and the min-frame defaults, not on the rasterizer,
    so indices computed here with a stub rasterizer address the same agents in
    a rasterizing AgentDataset built from a config with identical
    history/future settings.
    """
    cfg = stub_config()
    zarr_dataset = ChunkedDataset(str(data_root / "scenes/train.zarr")).open()
    render_context = RenderContext(
        np.asarray(cfg["raster_params"]["raster_size"]),
        np.asarray(cfg["raster_params"]["pixel_size"]),
        np.asarray(cfg["raster_params"]["ego_center"]),
        cfg["raster_params"]["set_origin_to_bottom"],
    )
    dataset = AgentDataset(cfg, zarr_dataset, StubRasterizer(render_context))
    boundary_frame = zarr_dataset.scenes[HOLDOUT_SCENE_START]["frame_index_interval"][0]
    boundary_agent = zarr_dataset.frames[boundary_frame]["agent_index_interval"][0]
    return np.nonzero(dataset.agents_indices < boundary_agent)[0]


class RasterPoolDataset(Dataset):
    """Map-style dataset over a fixed list of agent indices, rasterizing lazily."""

    def __init__(self, raster_kwargs: dict, indices: np.ndarray, data_root: Path):
        self._raster_kwargs = raster_kwargs
        self._indices = indices
        self._data_root = str(data_root)
        self._inner: AgentDataset | None = None

    def __len__(self) -> int:
        return len(self._indices)

    def _dataset(self) -> AgentDataset:
        if self._inner is None:
            cfg = raster_config(**self._raster_kwargs)
            zarr_dataset = ChunkedDataset(
                str(Path(self._data_root) / "scenes/train.zarr")
            ).open()
            rasterizer = build_rasterizer(cfg, LocalDataManager(self._data_root))
            self._inner = AgentDataset(cfg, zarr_dataset, rasterizer)
        return self._inner

    def __getitem__(self, position: int) -> dict[str, torch.Tensor]:
        sample = self._dataset()[int(self._indices[position])]
        image = np.clip(sample["image"] * 255.0, 0.0, 255.0).astype(np.uint8)
        return {
            "image": torch.from_numpy(image),
            "target_positions": torch.from_numpy(
                sample["target_positions"].astype(np.float32)
            ),
            "target_availabilities": torch.from_numpy(
                sample["target_availabilities"].astype(np.float32)
            ),
        }
