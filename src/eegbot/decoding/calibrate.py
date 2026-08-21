"""Per-subject calibration of the steering law.

Two numbers cannot be hardcoded and must be fitted from each subject's idle
block:

**bias** -- where that subject's classifier sits when they are doing nothing.
It is essentially never 0.5. Left uncorrected, the robot drifts one way
continuously and the rider spends the session fighting it.

**dead zone** -- how far from center the intent must travel before it counts.
Too tight and the robot twitches at rest; too loose and the subject cannot
issue a command at all. There is no universal value: it depends on how separable
that person's classes are, which varies enormously between subjects.

Both come from 60 seconds of relaxed no-imagery recording, which the session
runner collects for exactly this purpose.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eegbot.config import ControlConfig
from eegbot.control.controller import ContinuousController


@dataclass(frozen=True)
class Calibration:
    """Fitted per-subject control parameters."""

    bias: float
    threshold: float

    #: Achieved false-activation *onset* rate at `threshold`, per second.
    false_activation_hz: float

    #: Fraction of idle time spent issuing any command at `threshold`.
    idle_active_fraction: float

    #: True when no candidate threshold met both targets -- the subject's
    #: decoder is too noisy or too biased at rest. Worth surfacing rather than
    #: silently accepting: it usually means bad electrode contact or a subject
    #: near chance.
    target_met: bool

    def summary(self) -> str:
        status = "ok" if self.target_met else "TARGET NOT MET -- needs an idle gate"
        return (
            f"bias={self.bias:+.3f} dead_zone={self.threshold:.3f} "
            f"idle_onsets={self.false_activation_hz:.3f}/s "
            f"idle_active={self.idle_active_fraction:.1%} [{status}]"
        )


def count_activations(commands: np.ndarray) -> int:
    """Number of transitions from idle into an active command.

    Counts *onsets*, so a command that latches on and never releases scores 1.
    That is why `active_fraction` is measured alongside it -- see
    `ControlConfig.max_idle_active_fraction`.
    """
    active = np.asarray(commands) != 0.0
    if active.size == 0:
        return 0
    return int(np.sum(active[1:] & ~active[:-1]) + (1 if active[0] else 0))


def active_fraction(commands: np.ndarray) -> float:
    """Fraction of samples with a non-zero command."""
    commands = np.asarray(commands)
    if commands.size == 0:
        return 0.0
    return float(np.mean(commands != 0.0))


def simulate(
    probs: np.ndarray,
    config: ControlConfig,
    dt_s: float,
    *,
    bias: float,
    threshold: float,
    engagement: np.ndarray | None = None,
) -> np.ndarray:
    """Run the controller over a probability sequence, returning commands."""
    controller = ContinuousController(config, dt_s, bias=bias, threshold=threshold)
    if engagement is None:
        return np.array([controller.update(float(p)).command for p in probs])
    return np.array(
        [
            controller.update(float(p), float(e)).command
            for p, e in zip(probs, engagement, strict=True)
        ]
    )


def calibrate_from_idle(
    idle_probs: np.ndarray,
    config: ControlConfig,
    dt_s: float,
    *,
    candidates: np.ndarray | None = None,
    idle_engagement: np.ndarray | None = None,
) -> Calibration:
    """Fit bias and dead zone from idle-block probabilities.

    Picks the **smallest** dead zone that keeps idle behaviour under *both*
    targets: the onset rate `config.target_false_activation_hz` and the duty
    cycle `config.max_idle_active_fraction`. Smallest, because every bit of
    dead zone the subject does not need is responsiveness taken away from them.

    Both criteria are required because they fail differently. A twitchy decoder
    trips the onset rate; a decoder stuck off-center produces a single onset for
    the entire block -- a perfect onset rate -- while steering continuously.
    Only the duty cycle catches the second, and the second is the more common
    failure on a dry headset.

    Parameters
    ----------
    idle_probs
        p(right_hand) for each decoding window of the idle block, at the same
        hop the online loop will use.
    dt_s
        Seconds per window, i.e. `spec.hop_s`.
    idle_engagement
        `IdleGate` output for the same idle windows. **Pass this whenever a gate
        will be used at run time.** Calibrating without the gate and then
        deploying with it drives the dead zone far wider than necessary -- the
        search has to compensate for idle noise the gate would have removed --
        which costs the subject most of their ability to issue a command at all.
    """
    idle_probs = np.asarray(idle_probs, dtype=float)
    if idle_probs.size < 10:
        raise ValueError(
            f"need at least 10 idle windows to calibrate, got {idle_probs.size}"
        )

    bias = float(np.median(2.0 * idle_probs - 1.0))
    duration_s = idle_probs.size * dt_s

    if candidates is None:
        candidates = np.round(
            np.arange(0.05, config.max_dead_zone + 1e-9, 0.025), 4
        )

    best: Calibration | None = None
    for threshold in candidates:
        commands = simulate(
            idle_probs,
            config,
            dt_s,
            bias=bias,
            threshold=float(threshold),
            engagement=idle_engagement,
        )
        rate = count_activations(commands) / duration_s
        duty = active_fraction(commands)

        candidate = Calibration(
            bias=bias,
            threshold=float(threshold),
            false_activation_hz=rate,
            idle_active_fraction=duty,
            target_met=(
                rate <= config.target_false_activation_hz
                and duty <= config.max_idle_active_fraction
            ),
        )
        if candidate.target_met:
            return candidate
        # Rank fallbacks by duty cycle: continuous wrong steering is worse for
        # a rider than an occasional twitch.
        if best is None or duty < best.idle_active_fraction:
            best = candidate

    assert best is not None
    return best
