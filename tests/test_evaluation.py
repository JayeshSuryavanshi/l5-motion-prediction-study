import numpy as np
import pytest

pytest.importorskip("l5kit")

from l5study.evaluation import STANDARD_BASELINES, history_oldest_first


def test_history_oldest_first_flips_l5kit_ordering():
    sample = {
        "history_positions": np.array([[0.0, 0.0], [-1.0, 0.0], [-2.0, 0.0]]),
        "history_availabilities": np.array([1.0, 1.0, 0.0]),
    }
    positions, availabilities = history_oldest_first(sample)
    np.testing.assert_allclose(positions, [[-2.0, 0.0], [-1.0, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(availabilities, [0.0, 1.0, 1.0])


@pytest.mark.parametrize("name", sorted(STANDARD_BASELINES))
def test_standard_baselines_contract(name):
    predictor = STANDARD_BASELINES[name]
    sample = {
        "history_positions": np.column_stack(
            [np.linspace(0.0, -2.0, 11), np.zeros(11)]
        ),
        "history_availabilities": np.ones(11),
    }
    preds, confs = predictor(sample, 50, 0.1)
    assert preds.ndim == 3 and preds.shape[1:] == (50, 2)
    assert confs.shape == (preds.shape[0],)
    assert confs.sum() == pytest.approx(1.0)
    assert np.isfinite(preds).all()
