"""Common container for decodable data.

Every dataset -- Lee2019 or our own recordings -- ends up as an `EpochSet`, so
the decoding and evaluation code never learns where its data came from.

The `groups` field carries the **originating trial index** for each row and is
the single most important thing in this module. Cropped training produces many
overlapping windows per trial; if crops from one trial land in both the train
and test folds, cross-validation reports something like 95% accuracy that is
entirely fake. Every CV splitter in `eegbot.evaluation` is group-aware, and a
test asserts no trial ever spans a fold.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from eegbot.constants import CLASS_NAMES, EVENT_IDS
from eegbot.sigproc.spec import PreprocessSpec


@dataclass
class EpochSet:
    """Windows of EEG ready for decoding.

    Attributes
    ----------
    X
        ``(n_epochs, n_channels, n_times)`` in volts, channels in canonical
        DSI-7 order.
    y
        ``(n_epochs,)`` integer class labels, using `constants.EVENT_IDS`.
    groups
        ``(n_epochs,)`` originating trial id. Crops of the same trial share a
        value. Never split these across CV folds.
    subjects, sessions
        ``(n_epochs,)`` provenance, used by the cross-session and LOSO
        protocols.
    """

    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    subjects: np.ndarray
    sessions: np.ndarray
    spec: PreprocessSpec
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=float)
        if self.X.ndim != 3:
            raise ValueError(f"X must be (n_epochs, n_channels, n_times), got {self.X.shape}")

        n = self.X.shape[0]
        for name in ("y", "groups", "subjects", "sessions"):
            arr = np.asarray(getattr(self, name))
            if arr.shape != (n,):
                raise ValueError(
                    f"{name} must have shape ({n},) to match X, got {arr.shape}"
                )
            setattr(self, name, arr)

        if self.X.shape[1] != self.spec.n_channels:
            raise ValueError(
                f"X has {self.X.shape[1]} channels but spec expects "
                f"{self.spec.n_channels} ({', '.join(self.spec.channels)})"
            )

    def __len__(self) -> int:
        return self.X.shape[0]

    @property
    def n_trials(self) -> int:
        """Distinct source trials -- the real sample size, not `len(self)`."""
        return len(np.unique(self.groups))

    @property
    def class_counts(self) -> dict[str, int]:
        return {
            name: int(np.sum(self.y == EVENT_IDS[name]))
            for name in CLASS_NAMES
            if name in EVENT_IDS
        }

    def select(self, mask: np.ndarray) -> EpochSet:
        """Return a new EpochSet restricted to `mask` (boolean or index array)."""
        return replace(
            self,
            X=self.X[mask],
            y=self.y[mask],
            groups=self.groups[mask],
            subjects=self.subjects[mask],
            sessions=self.sessions[mask],
        )

    def summary(self) -> str:
        counts = ", ".join(f"{k}={v}" for k, v in self.class_counts.items())
        return (
            f"EpochSet: {len(self)} windows from {self.n_trials} trials "
            f"({counts}) | {self.X.shape[1]}ch x {self.X.shape[2]} samples "
            f"@ {self.spec.sfreq:g} Hz | "
            f"{len(np.unique(self.subjects))} subject(s), "
            f"{len(np.unique(self.sessions))} session(s)"
        )


def concat(sets: list[EpochSet]) -> EpochSet:
    """Concatenate EpochSets, offsetting `groups` so trial ids stay unique.

    Without the offset, trial 0 of subject 1 and trial 0 of subject 2 would
    share a group id and the CV splitter would treat them as the same trial.
    """
    if not sets:
        raise ValueError("nothing to concatenate")

    specs = {s.spec for s in sets}
    if len(specs) > 1:
        raise ValueError("cannot concatenate EpochSets built with different PreprocessSpecs")

    offset = 0
    groups: list[np.ndarray] = []
    for s in sets:
        groups.append(s.groups + offset)
        offset += int(s.groups.max()) + 1 if len(s) else 0

    return EpochSet(
        X=np.concatenate([s.X for s in sets], axis=0),
        y=np.concatenate([s.y for s in sets], axis=0),
        groups=np.concatenate(groups, axis=0),
        subjects=np.concatenate([s.subjects for s in sets], axis=0),
        sessions=np.concatenate([s.sessions for s in sets], axis=0),
        spec=sets[0].spec,
        meta={"n_sources": len(sets)},
    )
