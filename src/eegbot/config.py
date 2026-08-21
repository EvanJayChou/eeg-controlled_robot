"""YAML configuration loading.

Configs are plain YAML deserialized into frozen dataclasses that validate in
`__post_init__`. No config framework -- the surface is small enough that one
would cost more than it saves, and a typo'd key should be an immediate
``TypeError``, not a silently ignored field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from eegbot.sigproc.spec import PreprocessSpec

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file into a dict, resolving names against `configs/`."""
    p = Path(path)
    if not p.exists() and not p.is_absolute():
        candidate = CONFIG_ROOT / p
        if candidate.exists():
            p = candidate
    if not p.exists():
        raise FileNotFoundError(f"config not found: {path} (also looked in {CONFIG_ROOT})")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{p} must contain a YAML mapping at the top level")
    return data


def load_preprocess_spec(path: str | Path = "preprocess/default.yaml") -> PreprocessSpec:
    return PreprocessSpec.from_dict(load_yaml(path))


# === Control ===


@dataclass(frozen=True)
class ControlConfig:
    """Parameters of the steering law. See `eegbot.control.law`."""

    #: Exponential-moving-average time constant, seconds.
    ema_tau_s: float = 0.4

    #: Fallback dead-zone half-width, used when no idle block is available to
    #: calibrate against. Calibration should almost always override this.
    dead_zone: float = 0.25

    #: Upper bound on the calibrated dead zone.
    #:
    #: Calibration only sees idle data, so left unbounded it will happily widen
    #: the dead zone until the subject can barely issue a command at all -- it
    #: has no way to observe the cost. Past roughly 0.6 the control is unusable
    #: regardless of how quiet it looks at rest, so the search stops there and
    #: reports honestly that the target was not met.
    max_dead_zone: float = 0.60

    #: Target false-activation *onset* rate (activations per second) during
    #: idle. Catches a twitchy decoder that keeps starting to move.
    target_false_activation_hz: float = 0.05

    #: Maximum fraction of idle time spent issuing any command.
    #:
    #: Needed alongside the onset rate because the two catch different
    #: failures. A decoder stuck permanently off-center produces exactly one
    #: onset for the whole block -- an excellent onset rate -- while steering
    #: continuously and being completely undrivable. Only the duty cycle sees
    #: that.
    max_idle_active_fraction: float = 0.05

    #: Maximum change in normalized command per update tick. Prevents instant
    #: full-scale reversals.
    max_slew_per_tick: float = 0.15

    #: Scales the normalized command to a turn rate in rad/s.
    turn_rate_gain: float = 1.0

    #: Smoothed p(engaged) required to *start* issuing commands.
    #:
    #: Only used when a gate is supplied. Without one, a 2-class decoder emits
    #: confident output even at rest -- it has no way to say "neither" -- and no
    #: dead-zone width fixes that. See `eegbot.decoding.idle`.
    engagement_threshold: float = 0.60

    #: Smoothed p(engaged) below which commands *stop*.
    #:
    #: Deliberately lower than `engagement_threshold`: a single threshold makes
    #: the gate chatter every time engagement hovers near it, producing a burst
    #: of brief false activations. Hysteresis means you must clearly engage to
    #: start and clearly disengage to stop.
    engagement_release: float = 0.40

    def __post_init__(self) -> None:
        if not 0.0 <= self.dead_zone < 1.0:
            raise ValueError(f"dead_zone must be in [0, 1), got {self.dead_zone}")
        if self.ema_tau_s <= 0:
            raise ValueError("ema_tau_s must be positive")
        if self.max_slew_per_tick <= 0:
            raise ValueError("max_slew_per_tick must be positive")
        if not 0.0 < self.max_idle_active_fraction <= 1.0:
            raise ValueError(
                f"max_idle_active_fraction must be in (0, 1], got {self.max_idle_active_fraction}"
            )
        if self.engagement_release > self.engagement_threshold:
            raise ValueError(
                f"engagement_release ({self.engagement_release}) must be <= "
                f"engagement_threshold ({self.engagement_threshold}); inverting them "
                f"would make the gate chatter rather than hold"
            )


def load_control_config(path: str | Path = "control/default.yaml") -> ControlConfig:
    return ControlConfig(**load_yaml(path))


# === Paradigm ===


@dataclass(frozen=True)
class ParadigmConfig:
    """Recording-session structure. See `eegbot.session`."""

    fixation_s: float = 2.0
    cue_s: float = 1.0
    imagery_s: float = 4.0
    rest_range_s: tuple[float, float] = (2.0, 3.5)

    n_runs: int = 6
    trials_per_run: int = 20
    max_consecutive_same: int = 3

    alpha_block_s: float = 60.0
    idle_block_s: float = 60.0

    cue_script_version: str = "v1"
    instructions: str = ""

    def __post_init__(self) -> None:
        if self.trials_per_run % 2 != 0:
            raise ValueError(
                f"trials_per_run must be even so classes balance within a run, "
                f"got {self.trials_per_run}"
            )
        if self.max_consecutive_same < 1:
            raise ValueError("max_consecutive_same must be >= 1")
        lo, hi = self.rest_range_s
        if not 0 < lo <= hi:
            raise ValueError(f"rest_range_s must satisfy 0 < low <= high, got {self.rest_range_s}")

    @property
    def n_trials(self) -> int:
        return self.n_runs * self.trials_per_run

    @property
    def trial_duration_s(self) -> float:
        """Mean wall-clock seconds per trial, for estimating session length."""
        return self.fixation_s + self.cue_s + self.imagery_s + sum(self.rest_range_s) / 2


def load_paradigm_config(path: str | Path = "paradigm/mi_2class.yaml") -> ParadigmConfig:
    d = load_yaml(path)
    if "rest_range_s" in d:
        d["rest_range_s"] = tuple(d["rest_range_s"])
    return ParadigmConfig(**d)


# === Hardware ===


@dataclass(frozen=True)
class HardwareConfig:
    """How to reach the headset. Ignored entirely when replaying."""

    lsl_stream_name: str = "WS-default"
    lsl_stream_type: str = "EEG"
    marker_stream_name: str = "eegbot-markers"
    serial_port: str = "COM4"
    connect_timeout_s: float = 10.0
    extra: dict[str, Any] = field(default_factory=dict)


def load_hardware_config(path: str | Path = "hardware/dsi7.yaml") -> HardwareConfig:
    return HardwareConfig(**load_yaml(path))
