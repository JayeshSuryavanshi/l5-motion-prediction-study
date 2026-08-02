"""l5kit configuration builders shared by training and evaluation."""

from typing import Any

HOLDOUT_SCENE_START = 15265


def raster_config(
    history_num_frames: int = 10,
    future_num_frames: int = 50,
    raster_size: int = 224,
    pixel_size: float = 0.5,
    map_type: str = "py_semantic",
    step_time: float = 0.1,
) -> dict[str, Any]:
    return {
        "raster_params": {
            "raster_size": [raster_size, raster_size],
            "pixel_size": [pixel_size, pixel_size],
            "ego_center": [0.25, 0.5],
            "map_type": map_type,
            "satellite_map_key": "aerial_map/aerial_map.png",
            "semantic_map_key": "semantic_map/semantic_map.pb",
            "dataset_meta_key": "meta.json",
            "filter_agents_threshold": 0.5,
            "disable_traffic_light_faces": False,
            "set_origin_to_bottom": True,
        },
        "model_params": {
            "history_num_frames": history_num_frames,
            "future_num_frames": future_num_frames,
            "step_time": step_time,
            "render_ego_history": True,
        },
    }


def stub_config(
    history_num_frames: int = 10,
    future_num_frames: int = 50,
    step_time: float = 0.1,
) -> dict[str, Any]:
    """Config for map-free iteration with a StubRasterizer (no image is built)."""
    return {
        "raster_params": {
            "raster_size": [100, 100],
            "pixel_size": [0.25, 0.25],
            "ego_center": [0.5, 0.5],
            "map_type": "stub",
            "filter_agents_threshold": 0.5,
            "disable_traffic_light_faces": True,
            "set_origin_to_bottom": True,
        },
        "model_params": {
            "history_num_frames": history_num_frames,
            "future_num_frames": future_num_frames,
            "step_time": step_time,
            "render_ego_history": False,
        },
    }
