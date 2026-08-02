import importlib.util
import warnings

import numpy as np
import pytest
import torch

from l5study.metrics import multi_mode_nll_loss, neg_multi_log_likelihood


def l5kit_reference():
    if importlib.util.find_spec("l5kit") is None:
        pytest.fail(
            "l5kit is not installed; run `make setup` (a plain `uv sync` removes it)"
        )
    from l5kit.evaluation import metrics

    return metrics


def random_case(rng: np.random.Generator, num_modes: int = 3, num_future: int = 50):
    gt = rng.normal(size=(num_future, 2)) * 5
    pred = rng.normal(size=(num_modes, num_future, 2)) * 5
    conf = rng.dirichlet(np.ones(num_modes))
    avail = (rng.random(num_future) > 0.2).astype(np.float64)
    return gt, pred, conf, avail


def test_perfect_single_mode_scores_zero():
    gt = np.linspace(0, 10, 100).reshape(50, 2)
    pred = gt[None]
    score = neg_multi_log_likelihood(gt, pred, np.array([1.0]), np.ones(50))
    assert score == pytest.approx(0.0, abs=1e-9)


def test_masked_frames_do_not_count():
    rng = np.random.default_rng(0)
    gt, pred, conf, _ = random_case(rng)
    avail = np.ones(50)
    avail[10:] = 0.0
    corrupted = pred.copy()
    corrupted[:, 10:] += 100.0
    a = neg_multi_log_likelihood(gt, pred, conf, avail)
    b = neg_multi_log_likelihood(gt, corrupted, conf, avail)
    assert a == pytest.approx(b)


def test_confidence_on_better_mode_lowers_score():
    gt = np.zeros((50, 2))
    good = np.zeros((50, 2))
    bad = np.full((50, 2), 3.0)
    pred = np.stack([good, bad])
    sharp = neg_multi_log_likelihood(gt, pred, np.array([0.9, 0.1]), np.ones(50))
    diffuse = neg_multi_log_likelihood(gt, pred, np.array([0.5, 0.5]), np.ones(50))
    assert sharp < diffuse


def test_confidences_must_sum_to_one():
    gt = np.zeros((50, 2))
    pred = np.zeros((2, 50, 2))
    with pytest.raises(ValueError):
        neg_multi_log_likelihood(gt, pred, np.array([0.9, 0.3]), np.ones(50))
    with pytest.raises(ValueError):
        neg_multi_log_likelihood(gt, pred, np.array([0.5, 0.50008]), np.ones(50))


def test_malformed_inputs_are_rejected():
    gt = np.zeros((50, 2))
    pred = np.zeros((3, 50, 2))
    conf = np.array([0.5, 0.3, 0.2])
    avail = np.ones(50)
    with pytest.raises(ValueError):
        neg_multi_log_likelihood(gt, pred, np.array([1.0]), avail)
    with pytest.raises(ValueError):
        neg_multi_log_likelihood(gt[:49], pred, conf, avail)
    with pytest.raises(ValueError):
        neg_multi_log_likelihood(gt, pred, conf, avail[:49])
    with pytest.raises(ValueError):
        neg_multi_log_likelihood(gt, pred[0], conf, avail)
    bad = pred.copy()
    bad[1, 3, 0] = np.nan
    with pytest.raises(ValueError):
        neg_multi_log_likelihood(gt, bad, conf, avail)


def test_zero_confidence_is_exact_and_silent():
    gt = np.zeros((50, 2))
    pred = np.stack([np.zeros((50, 2)), np.full((50, 2), 3.0)])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        score = neg_multi_log_likelihood(gt, pred, np.array([1.0, 0.0]), np.ones(50))
    assert score == pytest.approx(0.0, abs=1e-9)


def test_matches_l5kit_reference():
    reference = l5kit_reference()
    rng = np.random.default_rng(42)
    for _ in range(100):
        gt, pred, conf, avail = random_case(rng)
        ours = neg_multi_log_likelihood(gt, pred, conf, avail)
        theirs = float(reference.neg_multi_log_likelihood(gt, pred, conf, avail))
        assert ours == pytest.approx(theirs, rel=1e-6, abs=1e-9)


def test_fractional_availabilities_match_l5kit():
    reference = l5kit_reference()
    rng = np.random.default_rng(3)
    gt, pred, conf, _ = random_case(rng)
    avail = rng.random(50)
    ours = neg_multi_log_likelihood(gt, pred, conf, avail)
    theirs = float(reference.neg_multi_log_likelihood(gt, pred, conf, avail))
    assert ours == pytest.approx(theirs, rel=1e-6, abs=1e-9)


def test_torch_loss_matches_numpy_metric():
    rng = np.random.default_rng(7)
    cases = [random_case(rng) for _ in range(8)]
    gt = torch.tensor(np.stack([c[0] for c in cases]))
    pred = torch.tensor(np.stack([c[1] for c in cases]))
    log_conf = torch.tensor(np.log(np.stack([c[2] for c in cases])))
    avail = torch.tensor(np.stack([c[3] for c in cases]))
    loss = multi_mode_nll_loss(gt, pred, log_conf, avail).item()
    expected = float(np.mean([neg_multi_log_likelihood(*c) for c in cases]))
    assert loss == pytest.approx(expected, rel=1e-6)
