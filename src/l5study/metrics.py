"""Multi-modal trajectory prediction metrics.

The scoring rule is the one used for the 2020 Lyft Motion Prediction
competition: negative log-likelihood of the ground-truth trajectory under a
mixture of unit-variance isotropic Gaussians centered on the K predicted
trajectories, weighted by predicted confidences. Gaussian normalization
constants are dropped, so a perfect prediction scores exactly 0 and values
are only meaningful relative to each other.

Validation and availability semantics deliberately mirror the reference
implementation in l5kit.evaluation.metrics (availability weights are applied
inside the square, so fractional availabilities behave identically).
"""

import numpy as np
import torch


def _validate(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    confidences: np.ndarray,
    availabilities: np.ndarray,
) -> None:
    if predictions.ndim != 3 or predictions.shape[-1] != 2:
        raise ValueError(
            f"predictions must have shape (K, T, 2), got {predictions.shape}"
        )
    num_modes, num_future, _ = predictions.shape
    if ground_truth.shape != (num_future, 2):
        raise ValueError(
            f"ground_truth must have shape ({num_future}, 2), got {ground_truth.shape}"
        )
    if confidences.shape != (num_modes,):
        raise ValueError(
            f"confidences must have shape ({num_modes},), got {confidences.shape}"
        )
    if availabilities.shape != (num_future,):
        raise ValueError(
            f"availabilities must have shape ({num_future},), got {availabilities.shape}"
        )
    for name, array in (
        ("ground_truth", ground_truth),
        ("predictions", predictions),
        ("confidences", confidences),
        ("availabilities", availabilities),
    ):
        if not np.isfinite(array).all():
            raise ValueError(f"non-finite values in {name}")
    if not np.isclose(confidences.sum(), 1.0, rtol=1e-5, atol=1e-8):
        raise ValueError(f"confidences must sum to 1, got {confidences.sum()}")


def neg_multi_log_likelihood(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    confidences: np.ndarray,
    availabilities: np.ndarray,
) -> float:
    """Score one agent.

    ground_truth: (T, 2) future positions.
    predictions: (K, T, 2) predicted futures for K modes.
    confidences: (K,) mixture weights, must sum to 1.
    availabilities: (T,) 1.0 where the ground-truth frame is valid, else 0.0.
    """
    _validate(ground_truth, predictions, confidences, availabilities)
    diff = (ground_truth[None] - predictions) * availabilities[None, :, None]
    per_mode = (diff**2).sum(axis=-1).sum(axis=-1)
    with np.errstate(divide="ignore"):
        log_terms = np.log(confidences) - 0.5 * per_mode
    peak = log_terms.max()
    return float(-(peak + np.log(np.exp(log_terms - peak).sum())))


def multi_mode_nll_loss(
    ground_truth: torch.Tensor,
    predictions: torch.Tensor,
    log_confidences: torch.Tensor,
    availabilities: torch.Tensor,
) -> torch.Tensor:
    """Batched training loss, same quantity as `neg_multi_log_likelihood`.

    ground_truth: (B, T, 2), predictions: (B, K, T, 2),
    log_confidences: (B, K) log of mixture weights (e.g. log_softmax output),
    availabilities: (B, T). Returns the scalar mean over the batch.
    """
    diff = (ground_truth.unsqueeze(1) - predictions) * availabilities.unsqueeze(
        1
    ).unsqueeze(-1)
    per_mode = (diff**2).sum(dim=-1).sum(dim=-1)
    return -torch.logsumexp(log_confidences - 0.5 * per_mode, dim=1).mean()
