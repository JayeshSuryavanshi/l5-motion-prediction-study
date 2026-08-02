import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from l5kit.data import (
    AGENT_DTYPE,
    FRAME_DTYPE,
    SCENE_DTYPE,
    TL_FACE_DTYPE,
    ChunkedDataset,
)

from l5study.data import extract_scene_range


def build_synthetic_zarr(
    path, num_scenes: int = 4, frames_per_scene: int = 5, agents_per_frame: int = 3
):
    ds = ChunkedDataset(str(path))
    ds.initialize()
    scenes = np.zeros(num_scenes, dtype=SCENE_DTYPE)
    frames = np.zeros(num_scenes * frames_per_scene, dtype=FRAME_DTYPE)
    agents = np.zeros(len(frames) * agents_per_frame, dtype=AGENT_DTYPE)
    tl_faces = np.zeros(0, dtype=TL_FACE_DTYPE)
    for s in range(num_scenes):
        scenes[s]["frame_index_interval"] = (
            s * frames_per_scene,
            (s + 1) * frames_per_scene,
        )
    for f in range(len(frames)):
        frames[f]["agent_index_interval"] = (
            f * agents_per_frame,
            (f + 1) * agents_per_frame,
        )
        frames[f]["traffic_light_faces_index_interval"] = (0, 0)
        frames[f]["timestamp"] = f
    agents["track_id"] = np.arange(len(agents))
    agents["centroid"] = np.arange(len(agents) * 2).reshape(-1, 2)
    ds.scenes.append(scenes)
    ds.frames.append(frames)
    ds.agents.append(agents)
    ds.tl_faces.append(tl_faces)
    return ds


def test_extract_scene_range_reindexes_intervals(tmp_path):
    src_path = tmp_path / "src.zarr"
    dst_path = tmp_path / "dst.zarr"
    build_synthetic_zarr(src_path)

    extract_scene_range(src_path, dst_path, 2, 4)
    dst = ChunkedDataset(str(dst_path)).open()

    assert len(dst.scenes) == 2
    assert len(dst.frames) == 10
    assert len(dst.agents) == 30
    np.testing.assert_array_equal(dst.scenes[0]["frame_index_interval"], (0, 5))
    np.testing.assert_array_equal(dst.scenes[1]["frame_index_interval"], (5, 10))
    np.testing.assert_array_equal(dst.frames[0]["agent_index_interval"], (0, 3))
    np.testing.assert_array_equal(dst.frames[9]["agent_index_interval"], (27, 30))
    assert dst.frames[0]["timestamp"] == 10
    np.testing.assert_array_equal(np.asarray(dst.agents["track_id"]), np.arange(30, 60))


def test_extract_scene_range_rejects_bad_ranges(tmp_path):
    src_path = tmp_path / "src.zarr"
    build_synthetic_zarr(src_path)
    with pytest.raises(ValueError):
        extract_scene_range(src_path, tmp_path / "out.zarr", 3, 3)
    with pytest.raises(ValueError):
        extract_scene_range(src_path, tmp_path / "out.zarr", 0, 99)
