"""Kinematic baselines: no learning, no map.

All functions operate in a single 2D frame. History is ordered oldest to
newest with the current position last; the returned future starts one step
after the current position and uses the same frame and step length.

Motion is fit only on the run of consecutively available frames ending at the
newest frame, so interior availability gaps shrink the fit window instead of
corrupting the finite differences; when too little contiguous history
remains, each function degrades explicitly (constant_turn -> constant_velocity
-> stationary). If no frame is observed at all, the fallback anchor is the
frame origin, which is only meaningful in an agent-centric frame where the
current pose is the origin (the frame this study evaluates in).

Note: constant_velocity and constant_turn are dt-invariant by construction
(dt cancels between fit and roll-out); dt matters only where an absolute
rate is imposed, i.e. kinematic_mixture's yaw_rate.
"""

import numpy as np

MIN_SPEED = 0.05


def _observed(history: np.ndarray, availabilities: np.ndarray) -> np.ndarray:
    return history[availabilities > 0]


def _contiguous_tail(history: np.ndarray, availabilities: np.ndarray) -> np.ndarray:
    """The longest run of consecutively available frames ending at the newest frame."""
    start = len(availabilities)
    while start > 0 and availabilities[start - 1] > 0:
        start -= 1
    return history[start:]


def stationary(
    history: np.ndarray, availabilities: np.ndarray, num_future: int
) -> np.ndarray:
    obs = _observed(history, availabilities)
    anchor = obs[-1] if len(obs) else np.zeros(2)
    return np.tile(anchor, (num_future, 1))


def constant_velocity(
    history: np.ndarray,
    availabilities: np.ndarray,
    num_future: int,
    dt: float = 0.1,
    fit_steps: int = 3,
) -> np.ndarray:
    tail = _contiguous_tail(history, availabilities)
    if len(tail) < 2:
        return stationary(history, availabilities, num_future)
    steps = min(fit_steps, len(tail) - 1)
    velocity = (tail[-1] - tail[-1 - steps]) / (steps * dt)
    horizon = np.arange(1, num_future + 1)[:, None] * dt
    return tail[-1] + velocity * horizon


def constant_turn(
    history: np.ndarray,
    availabilities: np.ndarray,
    num_future: int,
    dt: float = 0.1,
    fit_steps: int = 3,
) -> np.ndarray:
    """Constant speed and constant yaw rate (CTRV), fit from recent displacements.

    Headings are taken only from chords long enough to define a direction
    (length > MIN_SPEED * dt); zero-length chords from a stopped or stopping
    agent contribute no heading and no turn.
    """
    tail = _contiguous_tail(history, availabilities)
    if len(tail) < 3:
        return constant_velocity(history, availabilities, num_future, dt, fit_steps)
    diffs = np.diff(tail[-(fit_steps + 1) :], axis=0)
    lengths = np.linalg.norm(diffs, axis=1)
    speed = float(lengths.mean()) / dt
    if speed < MIN_SPEED:
        return stationary(history, availabilities, num_future)
    directed = diffs[lengths > MIN_SPEED * dt]
    if not len(directed):
        return stationary(history, availabilities, num_future)
    headings = np.arctan2(directed[:, 1], directed[:, 0])
    turns = np.arctan2(np.sin(np.diff(headings)), np.cos(np.diff(headings)))
    yaw_rate = float(turns.mean()) / dt if len(turns) else 0.0
    return _roll_out(tail[-1], speed, float(headings[-1]), yaw_rate, num_future, dt)


def kinematic_mixture(
    history: np.ndarray,
    availabilities: np.ndarray,
    num_future: int,
    dt: float = 0.1,
    yaw_rate: float = 0.1,
    confidences: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> tuple[np.ndarray, np.ndarray]:
    """Three modes: straight, curving left, curving right, at the fitted speed.

    Returns predictions of shape (3, num_future, 2) and the confidences array.
    """
    conf = np.asarray(confidences)
    tail = _contiguous_tail(history, availabilities)
    if len(tail) < 2:
        pred = stationary(history, availabilities, num_future)
        return np.stack([pred, pred, pred]), conf
    diffs = np.diff(tail[-4:], axis=0)
    lengths = np.linalg.norm(diffs, axis=1)
    speed = float(lengths.mean()) / dt
    directed = diffs[lengths > MIN_SPEED * dt]
    if speed < MIN_SPEED or not len(directed):
        pred = stationary(history, availabilities, num_future)
        return np.stack([pred, pred, pred]), conf
    heading = float(np.arctan2(directed[-1, 1], directed[-1, 0]))
    modes = [
        _roll_out(tail[-1], speed, heading, rate, num_future, dt)
        for rate in (0.0, yaw_rate, -yaw_rate)
    ]
    return np.stack(modes), conf


def _roll_out(
    start: np.ndarray,
    speed: float,
    heading: float,
    yaw_rate: float,
    num_future: int,
    dt: float,
) -> np.ndarray:
    headings = heading + yaw_rate * dt * np.arange(1, num_future + 1)
    steps = speed * dt * np.stack([np.cos(headings), np.sin(headings)], axis=1)
    return start + np.cumsum(steps, axis=0)
