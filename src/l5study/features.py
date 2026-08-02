"""Feature extraction for map-free learned models (E2).

Features come straight from the l5kit agent sample in its native newest-first
ordering: agent-frame history positions, their availability mask, availability
-masked history yaws, and the agent extent. Unavailable frames are zeroed by
l5kit already; yaws are re-masked here for safety.
"""

import numpy as np
from tqdm import tqdm

HISTORY_FRAMES = 10
FEATURE_DIM = 4 * (HISTORY_FRAMES + 1) + 3


def sample_features(sample: dict) -> np.ndarray:
    positions = np.asarray(sample["history_positions"], dtype=np.float64)
    availabilities = np.asarray(sample["history_availabilities"], dtype=np.float64)
    yaws = np.asarray(sample["history_yaws"], dtype=np.float64).reshape(-1)
    extent = np.asarray(sample["extent"], dtype=np.float64)[:3]
    features = np.concatenate(
        [
            (positions * availabilities[:, None]).reshape(-1),
            availabilities,
            yaws * availabilities,
            extent,
        ]
    )
    return features.astype(np.float32)


def extract_features(
    dataset, indices: np.ndarray, num_future: int = 50
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Materialize (features, targets, target_availabilities) for the given
    dataset indices. Targets stay in the agent frame."""
    features = np.empty((len(indices), FEATURE_DIM), dtype=np.float32)
    targets = np.empty((len(indices), num_future, 2), dtype=np.float32)
    availabilities = np.empty((len(indices), num_future), dtype=np.float32)
    for row, index in enumerate(
        tqdm(indices, desc=f"extracting {len(indices)} samples")
    ):
        sample = dataset[int(index)]
        features[row] = sample_features(sample)
        targets[row] = sample["target_positions"]
        availabilities[row] = sample["target_availabilities"]
    return features, targets, availabilities
