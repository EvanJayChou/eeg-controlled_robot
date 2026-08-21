"""The idle gate and engagement hysteresis.

Background: a 2-class steering decoder has no way to say "neither", so it emits
confident output even at rest and the dead zone alone cannot produce a usable
idle state. These tests pin down the behaviour of the fix.
"""

from __future__ import annotations

import numpy as np
import pytest

from eegbot.config import ControlConfig
from eegbot.control.controller import ContinuousController

pytest.importorskip("pyriemann")

from eegbot.decoding.idle import (  # noqa: E402
    ENGAGED,
    RESTING,
    IdleGate,
    build_gate_dataset,
    windows_from_mask,
)
from eegbot.sigproc.apply import preprocess  # noqa: E402
from eegbot.sigproc.spec import PreprocessSpec  # noqa: E402
from eegbot.stream.synthetic import synthetic_mi_recording  # noqa: E402

SPEC = PreprocessSpec()


# === Hysteresis ===


def test_gate_starts_disengaged():
    """Must earn the first command; a robot should not move on connect."""
    controller = ContinuousController(ControlConfig(), 0.1)
    assert controller.update(0.99, engagement=0.0).is_idle


def test_engagement_below_threshold_suppresses_a_confident_command():
    controller = ContinuousController(ControlConfig(), 0.1, threshold=0.1)
    commands = [controller.update(0.99, engagement=0.0) for _ in range(50)]
    assert all(c.is_idle for c in commands)
    assert all(c.gated for c in commands)


def test_engagement_above_threshold_allows_commands():
    controller = ContinuousController(ControlConfig(), 0.1, threshold=0.1)
    commands = [controller.update(0.99, engagement=1.0) for _ in range(50)]
    assert not commands[-1].is_idle
    assert not commands[-1].gated


def test_hysteresis_prevents_chatter_at_the_boundary():
    """The failure this exists to fix.

    Engagement oscillating around a single threshold would toggle the gate every
    other tick, producing a burst of brief false activations -- which is the
    binding constraint on calibration in practice.
    """
    config = ControlConfig(engagement_threshold=0.60, engagement_release=0.40)
    controller = ContinuousController(config, 0.1, threshold=0.1)

    rng = np.random.default_rng(0)
    commands = [
        controller.update(0.99, engagement=float(np.clip(rng.normal(0.5, 0.08), 0, 1)))
        for _ in range(300)
    ]
    active = np.array([not c.is_idle for c in commands])
    onsets = int(np.sum(active[1:] & ~active[:-1]) + (1 if active[0] else 0))
    assert onsets <= 3, f"gate chattered {onsets} times around the threshold"


def test_gate_release_requires_clear_disengagement():
    config = ControlConfig(engagement_threshold=0.60, engagement_release=0.40)
    controller = ContinuousController(config, 0.1, threshold=0.1)

    for _ in range(40):
        controller.update(0.99, engagement=0.95)
    # Halfway between the two thresholds: should hold, not release.
    held = [controller.update(0.99, engagement=0.50) for _ in range(10)]
    assert not held[-1].gated

    released = [controller.update(0.99, engagement=0.05) for _ in range(40)]
    assert released[-1].gated
    assert released[-1].is_idle


def test_disengaging_decelerates_rather_than_cutting_out():
    """An abrupt stop reads as a fault; the slew limiter must still apply.

    Note the gate does not trip on the first low-engagement tick: engagement is
    smoothed before the thresholds are applied, so one bad window cannot stop
    the robot. That is deliberate -- it is the same protection that stops the
    gate chattering.
    """
    controller = ContinuousController(ControlConfig(), 0.1, threshold=0.1)
    for _ in range(60):
        controller.update(0.99, engagement=1.0)
    running = abs(controller._command)
    assert running > 0.5

    trajectory = [abs(controller.update(0.99, engagement=0.0).command) for _ in range(40)]

    assert trajectory[0] == pytest.approx(running), "one bad window must not stop the robot"
    assert trajectory[-1] == 0.0, "sustained disengagement must reach a full stop"

    # Non-increasing once it starts slowing, and it takes several ticks.
    decelerating = trajectory[trajectory.index(max(trajectory)) :]
    assert all(b <= a + 1e-12 for a, b in zip(decelerating, decelerating[1:], strict=False))
    assert sum(1 for v in trajectory if v > 0) > 3, "should coast down, not snap to zero"


def test_inverted_engagement_thresholds_are_rejected():
    with pytest.raises(ValueError, match="must be <="):
        ControlConfig(engagement_threshold=0.3, engagement_release=0.7)


def test_dead_zone_is_bounded_by_config():
    """Calibration cannot see responsiveness, so the bound protects it."""
    from eegbot.decoding.calibrate import calibrate_from_idle

    config = ControlConfig(max_dead_zone=0.4)
    rng = np.random.default_rng(1)
    chaotic = rng.uniform(0.0, 1.0, 600)
    calibration = calibrate_from_idle(chaotic, config, 0.1)
    assert calibration.threshold <= 0.4


# === Windowing ===


def test_windows_from_mask_drops_straddling_windows():
    """A window spanning a rest/imagery boundary belongs to neither class."""
    data = np.zeros((SPEC.n_channels, 3000))
    mask = np.zeros(3000, dtype=bool)
    mask[1000:2000] = True

    windows = windows_from_mask(SPEC, data, mask)

    # Windows are cut on a fixed grid from sample 0, not realigned to the mask,
    # so count grid positions that fall entirely inside [1000, 2000).
    starts = np.arange(0, data.shape[-1] - SPEC.window_samples + 1, SPEC.hop_samples)
    expected = int(np.sum((starts >= 1000) & (starts + SPEC.window_samples <= 2000)))
    assert len(windows) == expected == 23


def test_windows_from_mask_rejects_length_mismatch():
    with pytest.raises(ValueError, match="mask length"):
        windows_from_mask(SPEC, np.zeros((SPEC.n_channels, 100)), np.zeros(50, dtype=bool))


# === Gate training ===


def build_recording(n_trials=40, seed=0):
    data, events, _ = synthetic_mi_recording(SPEC, n_trials=n_trials, seed=seed)
    return preprocess(SPEC, data, causal=False), events


def test_gate_dataset_has_both_classes_and_a_guard_band():
    filtered, events = build_recording()
    X, y = build_gate_dataset(SPEC, filtered, events)

    assert set(np.unique(y)) == {RESTING, ENGAGED}
    assert X.shape[1:] == (SPEC.n_channels, SPEC.window_samples)

    # The guard band means the two classes cannot account for every window.
    total_possible = 1 + (filtered.shape[-1] - SPEC.window_samples) // SPEC.hop_samples
    assert len(X) < total_possible


def test_gate_separates_rest_from_imagery():
    """The gate's core claim, on data where we know the ground truth."""
    from sklearn.model_selection import cross_val_score

    filtered, events = build_recording(n_trials=40)
    X, y = build_gate_dataset(SPEC, filtered, events)

    gate = IdleGate(SPEC)
    scores = cross_val_score(gate.model, X, y, cv=3, scoring="roc_auc")
    assert scores.mean() > 0.75, f"gate AUC {scores.mean():.3f} is too low to be useful"


def test_gate_engagement_is_higher_during_imagery():
    filtered, events = build_recording(n_trials=40)
    X, y = build_gate_dataset(SPEC, filtered, events)
    gate = IdleGate(SPEC).fit(X, y)

    engagement = gate.engagement(X)
    assert engagement[y == ENGAGED].mean() > engagement[y == RESTING].mean()


def test_gate_engagement_one_matches_batch():
    filtered, events = build_recording(n_trials=20)
    X, y = build_gate_dataset(SPEC, filtered, events)
    gate = IdleGate(SPEC).fit(X, y)

    assert gate.engagement_one(X[0]) == pytest.approx(gate.engagement(X[:1])[0])


def test_gate_needs_both_classes():
    filtered, _ = build_recording(n_trials=5)
    with pytest.raises(ValueError, match="need both classes"):
        build_gate_dataset(SPEC, filtered, np.zeros((0, 3), dtype=int))
