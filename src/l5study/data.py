"""Zarr dataset utilities: scene-range extraction for building held-out splits."""

from pathlib import Path

import numpy as np
from l5kit.data import ChunkedDataset

_COPY_CHUNK = 1_000_000


def extract_scene_range(src_path: Path, dst_path: Path, start: int, stop: int) -> None:
    """Copy scenes [start, stop) of a zarr dataset into a new standalone zarr,
    reindexing the frame/agent/traffic-light interval pointers to the new origin."""
    src = ChunkedDataset(str(src_path)).open()
    num_scenes = len(src.scenes)
    if not 0 <= start < stop <= num_scenes:
        raise ValueError(
            f"invalid scene range [{start}, {stop}) for {num_scenes} scenes"
        )

    scenes = np.asarray(src.scenes[start:stop]).copy()
    frame_lo = int(scenes[0]["frame_index_interval"][0])
    frame_hi = int(scenes[-1]["frame_index_interval"][1])
    frames = np.asarray(src.frames[frame_lo:frame_hi]).copy()
    agent_lo = int(frames[0]["agent_index_interval"][0])
    agent_hi = int(frames[-1]["agent_index_interval"][1])
    tl_lo = int(frames[0]["traffic_light_faces_index_interval"][0])
    tl_hi = int(frames[-1]["traffic_light_faces_index_interval"][1])

    scenes["frame_index_interval"] -= frame_lo
    frames["agent_index_interval"] -= agent_lo
    frames["traffic_light_faces_index_interval"] -= tl_lo

    dst = ChunkedDataset(str(dst_path))
    dst.initialize()
    dst.scenes.append(scenes)
    dst.frames.append(frames)
    for lo in range(agent_lo, agent_hi, _COPY_CHUNK):
        dst.agents.append(np.asarray(src.agents[lo : min(lo + _COPY_CHUNK, agent_hi)]))
    for lo in range(tl_lo, tl_hi, _COPY_CHUNK):
        dst.tl_faces.append(np.asarray(src.tl_faces[lo : min(lo + _COPY_CHUNK, tl_hi)]))
