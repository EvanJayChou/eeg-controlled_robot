"""Stateful steering controller.

The boundary of this project. `ControlCommand` is what a robot layer would
consume; nothing here knows about motors, gaits, or kinematics, and that seam is
deliberate -- it keeps the decoder testable in isolation and lets the robot work
happen in a separate session without touching any of this.
"""

from __future__ import annotations

from dataclasses import dataclass

from eegbot.config import ControlConfig
from eegbot.control.law import apply_law, ema_alpha, slew_limit


@dataclass(frozen=True)
class ControlCommand:
    """A single steering command.

    Attributes
    ----------
    turn_rate
        Angular velocity in rad/s. Positive is right, matching
        `constants.POSITIVE_CLASS`.
    throttle
        Forward speed in [0, 1]. Zero while idle. Currently a placeholder --
        with two classes there is no independent speed channel, so this is
        simply on-when-steering. The robot session may replace it.
    intent
        The smoothed, centered classifier intent in [-1, 1], before the dead
        zone. Useful for plotting and debugging; not for driving.
    command
        The post-dead-zone, post-slew normalized command in [-1, 1].
    is_idle
        True when no command is being issued -- either the intent fell inside
        the dead zone, or the idle gate judged the subject not to be imagining.
    engagement
        p(engaged) from the idle gate, or 1.0 when no gate is in use.
    gated
        True when the idle gate suppressed a command the dead zone would have
        allowed. Useful for diagnosing whether the gate is too strict.
    """

    turn_rate: float
    throttle: float
    intent: float
    command: float
    is_idle: bool
    engagement: float = 1.0
    gated: bool = False


class ContinuousController:
    """Turns a stream of p(right_hand) into a stream of `ControlCommand`.

    Parameters
    ----------
    config
        Smoothing, dead zone, slew and gain settings.
    dt_s
        Seconds between updates, i.e. the decoder's hop. Used to convert the
        EMA time constant into a coefficient, so retuning the hop does not
        silently change how much smoothing is applied.
    bias, threshold
        Per-subject calibration from `eegbot.decoding.calibrate`. When omitted,
        falls back to no bias and the configured default dead zone -- fine for
        tests, wrong for a real rider.
    """

    def __init__(
        self,
        config: ControlConfig,
        dt_s: float,
        *,
        bias: float = 0.0,
        threshold: float | None = None,
    ) -> None:
        self.config = config
        self.dt_s = dt_s
        self.bias = bias
        self.threshold = config.dead_zone if threshold is None else threshold
        self.alpha = ema_alpha(config.ema_tau_s, dt_s)
        self._smoothed: float | None = None
        self._command: float = 0.0
        self._engagement: float | None = None
        self._gated: bool = True  # start disengaged; earn the first command

    def reset(self) -> None:
        """Clear smoothing and command state. Call between runs."""
        self._smoothed = None
        self._command = 0.0
        self._engagement = None
        self._gated = True

    def update(self, p_right: float, engagement: float = 1.0) -> ControlCommand:
        """Advance one tick.

        Parameters
        ----------
        p_right
            Steering decoder output, p(right_hand).
        engagement
            p(engaged) from an `IdleGate`, or 1.0 when no gate is in use.

            The raw value is smoothed with the same time constant as the steering
            intent and then applied with **hysteresis**: commands start only once
            smoothed engagement clears `engagement_threshold`, and stop only once
            it falls below the lower `engagement_release`. A single threshold on
            a raw per-window probability chatters whenever engagement hovers near
            it, which shows up as a burst of brief false activations -- the
            binding constraint on calibration in practice.

            When gated, the command is driven back toward zero *through the slew
            limiter*, so disengaging decelerates smoothly rather than cutting
            out; an abrupt stop reads as a fault to the rider.
        """
        smoothed, command = apply_law(
            p_right,
            bias=self.bias,
            threshold=self.threshold,
            smoothed_prev=self._smoothed,
            command_prev=self._command,
            alpha=self.alpha,
            max_delta=self.config.max_slew_per_tick,
        )

        self._engagement = (
            engagement
            if self._engagement is None
            else self._engagement + self.alpha * (engagement - self._engagement)
        )
        if self._gated:
            self._gated = self._engagement < self.config.engagement_threshold
        else:
            self._gated = self._engagement < self.config.engagement_release
        gated = self._gated

        if gated:
            command = slew_limit(0.0, self._command, self.config.max_slew_per_tick)

        self._smoothed = smoothed
        self._command = command

        is_idle = command == 0.0
        return ControlCommand(
            turn_rate=self.config.turn_rate_gain * command,
            throttle=0.0 if is_idle else 1.0,
            intent=smoothed,
            command=command,
            is_idle=is_idle,
            engagement=engagement,
            gated=gated,
        )
