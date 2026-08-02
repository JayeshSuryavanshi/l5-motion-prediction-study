import numpy as np
import pytest
import torch

from l5study.features import FEATURE_DIM, sample_features
from l5study.metrics import multi_mode_nll_loss
from l5study.models import HistoryMLP


def fake_sample(num_history: int = 11) -> dict:
    rng = np.random.default_rng(0)
    return {
        "history_positions": rng.normal(size=(num_history, 2)),
        "history_availabilities": np.ones(num_history),
        "history_yaws": rng.normal(size=(num_history, 1)),
        "extent": np.array([4.4, 1.8, 1.5]),
    }


def test_sample_features_shape_and_dtype():
    features = sample_features(fake_sample())
    assert features.shape == (FEATURE_DIM,)
    assert features.dtype == np.float32


def test_sample_features_masks_unavailable_frames():
    sample = fake_sample()
    sample["history_availabilities"][3:] = 0.0
    features = sample_features(sample)
    yaw_block = features[3 * 11 : 4 * 11]
    assert np.all(yaw_block[3:] == 0.0)
    position_block = features[: 2 * 11].reshape(11, 2)
    assert np.all(position_block[3:] == 0.0)


def test_history_mlp_shapes_and_confidences():
    model = HistoryMLP(FEATURE_DIM, num_modes=3, num_future=50)
    features = torch.randn(4, FEATURE_DIM)
    predictions, log_confidences = model(features)
    assert predictions.shape == (4, 3, 50, 2)
    assert log_confidences.shape == (4, 3)
    torch.testing.assert_close(
        log_confidences.exp().sum(dim=1), torch.ones(4), rtol=1e-5, atol=1e-6
    )


def test_history_mlp_learns_on_synthetic_batch():
    torch.manual_seed(0)
    model = HistoryMLP(FEATURE_DIM, hidden_dim=64, depth=2, num_future=10)
    features = torch.randn(32, FEATURE_DIM)
    targets = torch.randn(32, 10, 2)
    avails = torch.ones(32, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for _ in range(60):
        predictions, log_confidences = model(features)
        loss = multi_mode_nll_loss(targets, predictions, log_confidences, avails)
        if first is None:
            first = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < first * 0.5


def test_history_mlp_gradients_finite():
    model = HistoryMLP(FEATURE_DIM)
    predictions, log_confidences = model(torch.randn(2, FEATURE_DIM))
    loss = multi_mode_nll_loss(
        torch.randn(2, 50, 2) * 100, predictions, log_confidences, torch.ones(2, 50)
    )
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_predictor_contract_matches_evaluation():
    pytest.importorskip("l5kit")
    from l5study.evaluation import STANDARD_BASELINES

    sample = fake_sample()
    for name, predictor in STANDARD_BASELINES.items():
        preds, confs = predictor(sample, 50, 0.1)
        assert preds.shape[1:] == (50, 2), name
        assert confs.sum() == pytest.approx(1.0), name
