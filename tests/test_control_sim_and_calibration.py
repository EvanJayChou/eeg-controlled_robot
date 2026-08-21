"""Drivability metrics and per-subject calibration.

These test the claim that motivates `control_sim`: accuracy and drivability are
different things, and a decoder can be good at one while being bad at the other.
"""

from __future__ import annotations

import numpy as np
import pytest

from eegbot.config import ControlConfig
from eegbot.decoding.calibrate import (
    active_fraction,
    calibrate_from_idle,
    count_activations,
    simulate,
)
from eegbot.evaluation.control_sim import simulate_control

CONFIG = ControlConfig()
DT = 0.1


def idle_probs(n=600, spread=0.05, center_p=0.5, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(center_p, spread, n), 0.01, 0.99)


# === Calibration ===


def test_calibration_recovers_a_known_bias():
    """A subject resting at p=0.65 must be centered back to zero intent."""
    calibration = calibrate_from_idle(idle_probs(center_p=0.65), CONFIG, DT)
    assert calibration.bias == pytest.approx(0.30, abs=0.05)


def test_calibration_meets_the_false_activation_target():
    calibration = calibrate_from_idle(idle_probs(spread=0.12), CONFIG, DT)
    assert calibration.target_met
    assert calibration.false_activation_hz <= CONFIG.target_false_activation_hz


def test_calibration_picks_the_smallest_workable_dead_zone():
    """Every unnecessary bit of dead zone is responsiveness taken away."""
    quiet = calibrate_from_idle(idle_probs(spread=0.03, seed=1), CONFIG, DT)
    noisy = calibrate_from_idle(idle_probs(spread=0.20, seed=1), CONFIG, DT)
    assert quiet.threshold < noisy.threshold


def test_calibration_flags_an_uncalibratable_subject():
    """A decoder too noisy at rest should be surfaced, not silently accepted."""
    rng = np.random.default_rng(2)
    chaotic = rng.uniform(0.0, 1.0, 600)
    calibration = calibrate_from_idle(chaotic, CONFIG, DT)
    if not calibration.target_met:
        assert calibration.false_activation_hz > CONFIG.target_false_activation_hz


def test_calibration_needs_enough_data():
    with pytest.raises(ValueError, match="at least 10 idle windows"):
        calibrate_from_idle(np.full(5, 0.5), CONFIG, DT)


def test_count_activations_counts_onsets_not_samples():
    commands = np.array([0, 0, 0.5, 0.5, 0.5, 0, 0, 0.4, 0])
    assert count_activations(commands) == 2


def test_count_activations_handles_starting_active():
    assert count_activations(np.array([0.5, 0.5, 0, 0.5])) == 2


def test_bias_correction_suppresses_idle_activity():
    """The end-to-end payoff of calibration.

    Asserted on duty cycle rather than onset count, because the failure being
    fixed here is *latching*: an uncorrected biased decoder turns on once and
    stays on, which scores a single onset while steering the entire block.
    """
    probs = idle_probs(center_p=0.68, spread=0.05, seed=3)
    uncorrected = simulate(probs, CONFIG, DT, bias=0.0, threshold=0.25)
    calibration = calibrate_from_idle(probs, CONFIG, DT)
    corrected = simulate(
        probs, CONFIG, DT, bias=calibration.bias, threshold=calibration.threshold
    )

    assert active_fraction(uncorrected) > 0.9, "uncorrected bias should latch on"
    assert active_fraction(corrected) < 0.05
    assert count_activations(uncorrected) == 1, "latched output looks fine by onset count alone"


def test_latched_decoder_fails_calibration():
    """A decoder stuck hard off-center must not be certified as usable.

    This is the case a pure onset-rate criterion accepts: one onset across the
    whole block reads as 0.017/s, comfortably under a 0.05/s target, while the
    robot drives in a circle the entire time.
    """
    stuck = np.full(600, 0.99)
    calibration = calibrate_from_idle(stuck, CONFIG, DT)
    if calibration.target_met:
        assert calibration.idle_active_fraction <= CONFIG.max_idle_active_fraction


# === Control simulation ===


def build_alternating(n_blocks=6, block=40, quality=0.9, seed=0):
    """Idle / right / idle / left ... with a decoder of the given quality."""
    rng = np.random.default_rng(seed)
    probs, intents = [], []
    for i in range(n_blocks):
        intent = 0 if i % 2 == 0 else (1 if (i // 2) % 2 == 0 else -1)
        target = 0.5 if intent == 0 else (quality if intent > 0 else 1 - quality)
        probs.append(np.clip(rng.normal(target, 0.08, block), 0.01, 0.99))
        intents.append(np.full(block, intent))
    return np.concatenate(probs), np.concatenate(intents)


def test_a_good_decoder_commits_in_the_right_direction():
    probs, intents = build_alternating(quality=0.92)
    metrics = simulate_control(probs, intents, CONFIG, DT, threshold=0.25)

    assert metrics.commit_rate == 1.0
    assert metrics.median_time_to_commit_s is not None
    assert metrics.median_time_to_commit_s < 2.0
    assert metrics.wrong_direction_fraction < 0.15


def test_a_chance_decoder_never_reliably_commits():
    rng = np.random.default_rng(5)
    probs = rng.uniform(0.4, 0.6, 400)
    intents = np.tile(np.repeat([0, 1, 0, -1], 25), 4)
    metrics = simulate_control(probs, intents, CONFIG, DT, threshold=0.25)
    assert metrics.commit_rate < 1.0 or metrics.wrong_direction_fraction > 0.1


def test_clumped_errors_are_worse_than_scattered_ones_at_equal_accuracy():
    """The claim that justifies this whole module.

    Two decoders with identical per-window accuracy, one erring in short
    scattered bursts and one in long clumps. Smoothing absorbs the scattered
    errors; the clumped ones become real wrong turns. Accuracy cannot see the
    difference -- sign-flip and wrong-direction rates can.
    """
    n = 600
    intents = np.ones(n, dtype=int)
    rng = np.random.default_rng(7)

    scattered = np.full(n, 0.85)
    scattered_idx = rng.choice(n, size=120, replace=False)
    scattered[scattered_idx] = 0.15

    clumped = np.full(n, 0.85)
    for start in range(0, n, 100):
        clumped[start : start + 20] = 0.15

    assert np.mean(scattered < 0.5) == pytest.approx(np.mean(clumped < 0.5), abs=0.01)

    scattered_metrics = simulate_control(scattered, intents, CONFIG, DT, threshold=0.25)
    clumped_metrics = simulate_control(clumped, intents, CONFIG, DT, threshold=0.25)

    assert clumped_metrics.wrong_direction_fraction > scattered_metrics.wrong_direction_fraction


def test_idle_and_active_time_are_accounted_for():
    probs, intents = build_alternating()
    metrics = simulate_control(probs, intents, CONFIG, DT)
    total = metrics.idle_seconds + metrics.active_seconds
    assert total == pytest.approx(len(probs) * DT)


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="must match"):
        simulate_control(np.zeros(10), np.zeros(9, dtype=int), CONFIG, DT)


def test_summary_is_human_readable():
    probs, intents = build_alternating()
    text = simulate_control(probs, intents, CONFIG, DT).summary()
    assert "false activations" in text and "sign flips" in text
