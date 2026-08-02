import numpy as np
import pytest

from l5study.baselines import (
    constant_turn,
    constant_velocity,
    kinematic_mixture,
    stationary,
)
from l5study.metrics import neg_multi_log_likelihood

DT = 0.1


def line(
    num: int,
    velocity: tuple[float, float],
    start: tuple[float, float] = (0.0, 0.0),
    dt: float = DT,
):
    t = np.arange(num)[:, None] * dt
    return np.asarray(start) + np.asarray(velocity) * t


def circle(
    num: int, radius: float, angular_rate: float, phase: float = 0.0, dt: float = DT
):
    t = np.arange(num) * dt
    angles = phase + angular_rate * t
    return radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)


def step_angles(prediction: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    steps = np.diff(np.vstack([anchor, prediction]), axis=0)
    return np.unwrap(np.arctan2(steps[:, 1], steps[:, 0]))


def test_constant_velocity_extends_straight_motion():
    track = line(61, velocity=(3.0, -1.5))
    history, future = track[:11], track[11:]
    pred = constant_velocity(history, np.ones(11), 50, dt=DT)
    np.testing.assert_allclose(pred, future, atol=1e-9)
    score = neg_multi_log_likelihood(future, pred[None], np.array([1.0]), np.ones(50))
    assert score == pytest.approx(0.0, abs=1e-9)


def test_constant_turn_tracks_circular_motion():
    track = circle(61, radius=20.0, angular_rate=0.3)
    history, future = track[:11], track[11:]
    pred = constant_turn(history, np.ones(11), 50, dt=DT)
    straight = constant_velocity(history, np.ones(11), 50, dt=DT)
    turn_err = np.linalg.norm(pred - future, axis=1).mean()
    straight_err = np.linalg.norm(straight - future, axis=1).mean()
    assert turn_err < straight_err
    assert turn_err < 0.5


def test_interior_gap_shrinks_fit_window_instead_of_corrupting_it():
    track = line(61, velocity=(10.0, 0.0))
    history, future = track[:11], track[11:]
    avail = np.ones(11)
    avail[5] = 0.0
    pred = constant_velocity(history, avail, 50, dt=DT)
    np.testing.assert_allclose(pred, future, atol=1e-9)


def test_gap_at_penultimate_frame_falls_back_to_stationary():
    track = line(11, velocity=(10.0, 0.0))
    avail = np.ones(11)
    avail[-2] = 0.0
    pred = constant_velocity(track, avail, 50, dt=DT)
    np.testing.assert_allclose(pred, np.tile(track[-1], (50, 1)))


def test_braking_agent_keeps_last_valid_heading():
    northbound = line(9, velocity=(0.0, 1.0))
    stopped = np.tile(northbound[-1], (2, 1))
    history = np.vstack([northbound, stopped])
    preds, _ = kinematic_mixture(history, np.ones(11), 50, dt=DT)
    straight = preds[0]
    assert np.abs(straight[:, 0]).max() < 1e-9
    assert straight[-1, 1] > history[-1, 1]


def test_constant_turn_ignores_zero_length_last_chord():
    track = circle(12, radius=15.0, angular_rate=0.4)
    history = track[:11].copy()
    history[10] = history[9]
    pred = constant_turn(history, np.ones(11), 50, dt=DT)
    last_directed = history[9] - history[8]
    first_step = pred[0] - history[10]
    cosine = (
        first_step
        @ last_directed
        / (np.linalg.norm(first_step) * np.linalg.norm(last_directed))
    )
    assert cosine > 0.9


def test_short_history_falls_back_to_stationary():
    history = np.zeros((11, 2))
    avail = np.zeros(11)
    avail[-1] = 1.0
    history[-1] = (4.0, 2.0)
    pred = constant_velocity(history, avail, 50, dt=DT)
    np.testing.assert_allclose(pred, np.tile([4.0, 2.0], (50, 1)))


def test_stationary_with_empty_history_predicts_origin():
    pred = stationary(np.zeros((11, 2)), np.zeros(11), 50)
    np.testing.assert_allclose(pred, np.zeros((50, 2)))


def test_mixture_shapes_and_confidences():
    track = line(11, velocity=(2.0, 0.0))
    preds, conf = kinematic_mixture(track, np.ones(11), 50, dt=DT)
    assert preds.shape == (3, 50, 2)
    assert conf.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(preds[0], line(61, velocity=(2.0, 0.0))[11:], atol=1e-9)
    assert not np.allclose(preds[1], preds[2])


def test_mixture_curved_mode_sweep_scales_with_dt():
    yaw_rate = 0.5
    for dt in (0.1, 0.2):
        track = line(11, velocity=(2.0, 0.0), dt=dt)
        preds, _ = kinematic_mixture(track, np.ones(11), 50, dt=dt, yaw_rate=yaw_rate)
        angles = step_angles(preds[1], track[-1])
        sweep = angles[-1] - angles[0]
        assert sweep == pytest.approx(yaw_rate * dt * 49, rel=1e-6)


def test_mixture_never_beats_metric_floor_on_turns():
    track = circle(61, radius=15.0, angular_rate=0.4)
    history, future = track[:11], track[11:]
    preds, conf = kinematic_mixture(history, np.ones(11), 50, dt=DT, yaw_rate=0.4)
    mixture_score = neg_multi_log_likelihood(future, preds, conf, np.ones(50))
    straight = constant_velocity(history, np.ones(11), 50, dt=DT)
    straight_score = neg_multi_log_likelihood(
        future, straight[None], np.array([1.0]), np.ones(50)
    )
    assert mixture_score < straight_score
