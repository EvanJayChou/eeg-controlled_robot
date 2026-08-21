"""Does the decoder actually steer?

Classification accuracy is a poor proxy for drivability, and optimising it can
even hurt. A decoder at 72% that is wrong in short scattered bursts is pleasant
to drive, because smoothing absorbs the errors. A decoder at 78% whose mistakes
arrive in half-second clumps is not, because those clumps survive the filter and
become real, wrong turns. Accuracy cannot tell the two apart.

So this module replays a decoder through the actual control law and reports what
a rider would notice:

**false-activation rate** -- commands issued while the subject is at rest. The
single most important number: a robot that wanders off unbidden is not usable no
matter how well it responds when asked.

**time to commit** -- how long from intent onset until the command reflects it.
Sets the pace at which someone can steer.

**sign-flip rate** -- direction reversals during a sustained intent. This is the
wobble a rider feels, and it is invisible in an accuracy score.

**wrong-direction fraction** -- share of active time spent steering the wrong way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eegbot.config import ControlConfig
from eegbot.control.controller import ContinuousController


@dataclass(frozen=True)
class ControlMetrics:
    false_activation_hz: float

    #: Fraction of idle time spent issuing any command. Reported alongside the
    #: onset rate because a decoder latched permanently off-center scores one
    #: onset for the whole block while steering the entire time.
    idle_active_fraction: float

    commit_rate: float
    median_time_to_commit_s: float | None
    sign_flip_hz: float
    wrong_direction_fraction: float
    idle_seconds: float
    active_seconds: float

    def summary(self) -> str:
        ttc = (
            f"{self.median_time_to_commit_s:.2f}s"
            if self.median_time_to_commit_s is not None
            else "never"
        )
        return (
            f"false activations {self.false_activation_hz:.3f}/s "
            f"({self.idle_active_fraction:.1%} of idle time) | "
            f"commit {self.commit_rate:.0%} @ {ttc} | "
            f"sign flips {self.sign_flip_hz:.3f}/s | "
            f"wrong direction {self.wrong_direction_fraction:.1%}"
        )


def _segments(intents: np.ndarray, value_test) -> list[tuple[int, int]]:
    """Contiguous ``[start, stop)`` runs where `value_test` holds."""
    mask = value_test(intents)
    if not mask.any():
        return []
    edges = np.diff(mask.astype(int))
    starts = list(np.flatnonzero(edges == 1) + 1)
    stops = list(np.flatnonzero(edges == -1) + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        stops.append(len(mask))
    return list(zip(starts, stops, strict=True))


def simulate_control(
    probs: np.ndarray,
    intents: np.ndarray,
    config: ControlConfig,
    dt_s: float,
    *,
    bias: float = 0.0,
    threshold: float | None = None,
    engagement: np.ndarray | None = None,
) -> ControlMetrics:
    """Replay `probs` through the control law and measure drivability.

    Parameters
    ----------
    probs
        p(right_hand) per decoding window, in time order.
    intents
        Ground-truth intent per window: ``+1`` right, ``-1`` left, ``0`` idle.
    dt_s
        Seconds per window (`spec.hop_s`).
    bias, threshold
        Per-subject calibration from `eegbot.decoding.calibrate`.
    engagement
        Optional p(engaged) per window from an `IdleGate`. Pass it to measure
        drivability *with* the gate in the loop -- on a 2-class decoder the idle
        numbers are usually unusable without one.
    """
    probs = np.asarray(probs, dtype=float)
    intents = np.asarray(intents, dtype=int)
    if probs.shape != intents.shape:
        raise ValueError(f"probs {probs.shape} and intents {intents.shape} must match")

    if engagement is None:
        engagement = np.ones_like(probs)
    else:
        engagement = np.asarray(engagement, dtype=float)
        if engagement.shape != probs.shape:
            raise ValueError(
                f"engagement {engagement.shape} and probs {probs.shape} must match"
            )

    controller = ContinuousController(
        config, dt_s, bias=bias, threshold=threshold if threshold is not None else config.dead_zone
    )
    commands = np.array(
        [controller.update(float(p), float(e)).command for p, e in zip(probs, engagement, strict=True)]
    )
    active = commands != 0.0

    # === Idle behaviour ===
    idle_segments = _segments(intents, lambda a: a == 0)
    idle_samples = sum(stop - start for start, stop in idle_segments)
    false_activations = 0
    idle_active_samples = 0
    for start, stop in idle_segments:
        seg = active[start:stop]
        if seg.size:
            false_activations += int(np.sum(seg[1:] & ~seg[:-1]) + (1 if seg[0] else 0))
            idle_active_samples += int(np.sum(seg))
    idle_seconds = idle_samples * dt_s
    false_hz = false_activations / idle_seconds if idle_seconds > 0 else 0.0
    idle_active_fraction = idle_active_samples / idle_samples if idle_samples else 0.0

    # === Active behaviour ===
    intent_segments = _segments(intents, lambda a: a != 0)
    commit_times: list[float] = []
    committed = 0
    flips = 0
    wrong = 0
    active_samples = 0

    for start, stop in intent_segments:
        want = np.sign(intents[start])
        seg = commands[start:stop]
        active_samples += stop - start

        correct = np.flatnonzero(np.sign(seg) == want)
        if correct.size:
            committed += 1
            commit_times.append(float(correct[0]) * dt_s)

        signs = np.sign(seg[seg != 0.0])
        if signs.size > 1:
            flips += int(np.sum(signs[1:] != signs[:-1]))
        wrong += int(np.sum((seg != 0.0) & (np.sign(seg) != want)))

    active_seconds = active_samples * dt_s
    n_intents = len(intent_segments)

    return ControlMetrics(
        false_activation_hz=false_hz,
        idle_active_fraction=idle_active_fraction,
        commit_rate=committed / n_intents if n_intents else 0.0,
        median_time_to_commit_s=float(np.median(commit_times)) if commit_times else None,
        sign_flip_hz=flips / active_seconds if active_seconds > 0 else 0.0,
        wrong_direction_fraction=wrong / active_samples if active_samples else 0.0,
        idle_seconds=idle_seconds,
        active_seconds=active_seconds,
    )
