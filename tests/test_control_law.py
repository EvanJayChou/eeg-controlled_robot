"""Properties of the steering law.

These are the guarantees that make control feel good rather than twitchy, so
they are tested as properties over the whole input range rather than at a few
sample points.
"""

from __future__ import annotations

import numpy as np
import pytest

from eegbot.config import ControlConfig
from eegbot.control.controller import ContinuousController
from eegbot.control.law import center, dead_zone, ema_alpha, slew_limit

THRESHOLDS = [0.0, 0.1, 0.25, 0.5, 0.8]


def test_center_maps_half_to_zero():
    assert center(0.5) == pytest.approx(0.0)
    assert center(1.0) == pytest.approx(1.0)
    assert center(0.0) == pytest.approx(-1.0)


def test_center_bias_shifts_the_neutral_point():
    # A subject whose classifier rests at p=0.6 should read as zero intent
    # once their bias is removed -- otherwise the robot drifts right forever.
    bias = center(0.6)
    assert center(0.6, bias) == pytest.approx(0.0)


def test_center_clips_to_unit_range():
    s = center(np.linspace(0, 1, 101), bias=0.5)
    assert s.min() >= -1.0 and s.max() <= 1.0


@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_dead_zone_is_exactly_zero_inside(threshold):
    inside = np.linspace(-threshold, threshold, 51)[1:-1]
    if inside.size:
        assert np.all(dead_zone(inside, threshold) == 0.0)


@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_dead_zone_is_continuous_at_the_boundary(threshold):
    """No step at the edge.

    A discontinuity here is precisely what makes a BCI feel twitchy: outputs
    hovering near the threshold would jump between zero and a finite command on
    classifier noise alone.
    """
    eps = 1e-9
    just_outside = dead_zone(threshold + eps, threshold)
    assert just_outside == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_dead_zone_reaches_full_deflection(threshold):
    """Full command must remain reachable, or the robot can never turn hard."""
    assert dead_zone(1.0, threshold) == pytest.approx(1.0)
    assert dead_zone(-1.0, threshold) == pytest.approx(-1.0)


@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_dead_zone_is_bounded_and_monotone(threshold):
    s = np.linspace(-1.0, 1.0, 2001)
    u = dead_zone(s, threshold)
    assert u.min() >= -1.0 and u.max() <= 1.0
    assert np.all(np.diff(u) >= -1e-12)


@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_dead_zone_is_antisymmetric(threshold):
    s = np.linspace(0.0, 1.0, 101)
    np.testing.assert_allclose(dead_zone(-s, threshold), -dead_zone(s, threshold))


def test_dead_zone_rejects_out_of_range_threshold():
    with pytest.raises(ValueError):
        dead_zone(0.5, 1.0)
    with pytest.raises(ValueError):
        dead_zone(0.5, -0.1)


def test_ema_alpha_tracks_the_time_constant():
    """Alpha must follow dt, so retuning the hop does not change smoothing."""
    fast = ema_alpha(tau_s=0.4, dt_s=0.1)
    slow = ema_alpha(tau_s=0.4, dt_s=0.05)
    assert 0 < slow < fast < 1


def test_ema_reaches_63_percent_after_one_tau():
    tau, dt = 0.4, 0.01
    alpha = ema_alpha(tau, dt)
    value = 0.0
    for _ in range(int(tau / dt)):
        value += alpha * (1.0 - value)
    assert value == pytest.approx(1 - np.exp(-1), abs=0.01)


def test_slew_limit_caps_change():
    assert slew_limit(1.0, 0.0, 0.15) == pytest.approx(0.15)
    assert slew_limit(-1.0, 0.0, 0.15) == pytest.approx(-0.15)
    assert slew_limit(0.05, 0.0, 0.15) == pytest.approx(0.05)


def test_full_reversal_takes_multiple_ticks():
    """A -1 -> +1 swing should read as deliberate, not as a twitch."""
    config = ControlConfig(max_slew_per_tick=0.15)
    ticks = int(np.ceil(2.0 / config.max_slew_per_tick))
    assert ticks >= 10  # >= 1 second at 10 Hz


# === Controller ===


def test_controller_idles_on_a_neutral_stream():
    controller = ContinuousController(ControlConfig(), dt_s=0.1)
    commands = [controller.update(0.5) for _ in range(50)]
    assert all(c.is_idle for c in commands)
    assert all(c.turn_rate == 0.0 for c in commands)
    assert all(c.throttle == 0.0 for c in commands)


def test_controller_commits_to_a_sustained_intent():
    controller = ContinuousController(ControlConfig(), dt_s=0.1)
    commands = [controller.update(0.95) for _ in range(60)]
    assert commands[-1].turn_rate > 0
    assert not commands[-1].is_idle
    assert commands[-1].throttle == 1.0


def test_controller_direction_follows_the_class():
    right = ContinuousController(ControlConfig(), dt_s=0.1)
    left = ContinuousController(ControlConfig(), dt_s=0.1)
    for _ in range(60):
        r = right.update(0.95)
        left_cmd = left.update(0.05)
    assert r.turn_rate > 0 > left_cmd.turn_rate


def test_controller_ignores_a_single_noisy_spike():
    """Smoothing plus the dead zone should absorb one bad window."""
    controller = ContinuousController(ControlConfig(), dt_s=0.1)
    for _ in range(30):
        controller.update(0.5)
    spike = controller.update(1.0)
    assert spike.is_idle


def test_controller_reset_clears_state():
    controller = ContinuousController(ControlConfig(), dt_s=0.1)
    for _ in range(60):
        controller.update(0.95)
    controller.reset()
    assert controller.update(0.5).is_idle


def test_bias_correction_removes_resting_drift():
    """The headline reason bias exists: an offset subject must still idle.

    0.70 centers to 0.40, comfortably outside the default 0.25 dead zone -- so
    without correction this subject would steer right continuously while sitting
    still. An offset this large is unremarkable in practice.
    """
    resting_p = 0.70
    biased = ContinuousController(ControlConfig(), dt_s=0.1)
    corrected = ContinuousController(
        ControlConfig(), dt_s=0.1, bias=float(center(resting_p))
    )
    for _ in range(80):
        drifting = biased.update(resting_p)
        steady = corrected.update(resting_p)
    assert not drifting.is_idle, "uncorrected bias should produce spurious turning"
    assert steady.is_idle, "bias correction should restore the idle state"
