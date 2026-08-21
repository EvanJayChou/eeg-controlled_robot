"""The steering law, as pure functions.

Two classes do not mean two commands. Treating the classifier output as a dial
rather than a switch gives smooth proportional turning *and* a natural idle
state: the dead zone in the middle is "do nothing", so a binary decoder yields
three behaviours.

The chain, applied once per decoding window:

    s_raw = 2 * p(right) - 1                              center to [-1, 1]
    s     = clip(s_raw - bias, -1, 1)                     remove per-subject offset
    s_f   = ema(s, tau)                                   smooth
    u     = 0                              if |s_f| < th  dead zone == idle
          = sign(s_f)*(|s_f|-th)/(1-th)    otherwise      rescale to full range
    u     = slew_limit(u, u_prev, max_d)                  no instant reversals

Everything here is stateless and side-effect free so the properties that make
control feel good -- boundedness, monotonicity, and **continuity at the dead-zone
edge** -- can be tested directly. That last one matters: a step discontinuity at
the threshold is exactly what makes a BCI feel twitchy, because outputs near the
boundary jump between zero and a finite value on classifier noise alone.
"""

from __future__ import annotations

import numpy as np


def center(p_right: float | np.ndarray, bias: float = 0.0) -> float | np.ndarray:
    """Map a probability to a signed intent in [-1, 1].

    `bias` removes a per-subject resting offset, estimated from the idle block
    by `eegbot.decoding.calibrate`. It is not optional in practice: essentially
    every subject's classifier sits slightly off-center, and an uncentered dial
    makes the robot drift constantly while the rider fights it.
    """
    s = 2.0 * np.asarray(p_right, dtype=float) - 1.0 - bias
    out = np.clip(s, -1.0, 1.0)
    return float(out) if np.isscalar(p_right) or np.ndim(p_right) == 0 else out


def dead_zone(s: float | np.ndarray, threshold: float) -> float | np.ndarray:
    """Zero out small intents and rescale the remainder to span [-1, 1].

    The rescaling by ``1 / (1 - threshold)`` is what keeps full deflection
    reachable: without it, a 0.3 dead zone would cap the command at 0.7 and the
    robot could never turn at full rate.

    Continuous at ``|s| == threshold`` (both branches give 0) and monotone in
    `s`, both of which are asserted in the test suite.
    """
    if not 0.0 <= threshold < 1.0:
        raise ValueError(f"threshold must be in [0, 1), got {threshold}")

    s_arr = np.asarray(s, dtype=float)
    magnitude = np.abs(s_arr)
    scaled = np.where(
        magnitude < threshold,
        0.0,
        np.sign(s_arr) * (magnitude - threshold) / (1.0 - threshold),
    )
    out = np.clip(scaled, -1.0, 1.0)
    return float(out) if np.isscalar(s) or np.ndim(s) == 0 else out


def ema_alpha(tau_s: float, dt_s: float) -> float:
    """Smoothing coefficient for an exponential moving average.

    ``alpha = 1 - exp(-dt/tau)``, the exact discretization of a first-order
    low-pass, so the effective time constant stays `tau_s` regardless of the
    update rate. Using a hardcoded alpha instead would silently change the
    smoothing whenever the hop size is retuned.
    """
    if tau_s <= 0 or dt_s <= 0:
        raise ValueError("tau_s and dt_s must be positive")
    return float(1.0 - np.exp(-dt_s / tau_s))


def slew_limit(target: float, previous: float, max_delta: float) -> float:
    """Cap the per-tick change in command.

    Prevents a full -1 -> +1 reversal in a single update, which would read as a
    twitch rather than an intention. At 10 Hz with `max_delta` 0.15, a complete
    reversal takes about 1.3 seconds.
    """
    if max_delta <= 0:
        raise ValueError(f"max_delta must be positive, got {max_delta}")
    delta = float(np.clip(target - previous, -max_delta, max_delta))
    return previous + delta


def apply_law(
    p_right: float,
    *,
    bias: float,
    threshold: float,
    smoothed_prev: float | None,
    command_prev: float,
    alpha: float,
    max_delta: float,
) -> tuple[float, float]:
    """One full pass of the law.

    Returns ``(smoothed_intent, command)``. Kept separate from the stateful
    `ContinuousController` so the arithmetic can be tested without constructing
    an object or replaying a stream.
    """
    s = float(center(p_right, bias))
    s_f = s if smoothed_prev is None else smoothed_prev + alpha * (s - smoothed_prev)
    u_target = float(dead_zone(s_f, threshold))
    u = slew_limit(u_target, command_prev, max_delta)
    return s_f, u
