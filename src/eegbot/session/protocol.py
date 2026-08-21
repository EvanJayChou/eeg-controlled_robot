"""Trial scheduling.

Three constraints, each for a reason:

**Balanced within every run, not just overall.** If a run happened to be 15 left
and 5 right, and that run gets used as a calibration block, the classifier
inherits the imbalance as a prior and the steering dial sits off-center before
the subject has done anything.

**No long same-class streaks.** Subjects notice runs of four or five identical
cues and start predicting, which changes what they are doing well before the
arrow appears. Capping streaks keeps the cue genuinely informative.

**Jittered rest.** A fixed inter-trial interval lets subjects entrain to the
rhythm, producing anticipatory activity time-locked to the cue -- which a
classifier will happily decode instead of the imagery.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eegbot.config import ParadigmConfig
from eegbot.constants import EVENT_IDS


@dataclass(frozen=True)
class Trial:
    index: int
    run: int
    label: int
    label_name: str
    rest_s: float


def balanced_labels(
    n: int,
    max_consecutive: int,
    rng: np.random.Generator,
    *,
    max_attempts: int = 1000,
) -> np.ndarray:
    """Shuffle a balanced label vector until no streak exceeds `max_consecutive`.

    Rejection sampling rather than a constructive algorithm: at these sizes it
    converges in a handful of attempts, and it keeps the resulting order
    genuinely uniform among valid sequences rather than biased by a greedy
    construction.
    """
    if n % 2 != 0:
        raise ValueError(f"n must be even to balance two classes, got {n}")

    codes = list(EVENT_IDS.values())
    if len(codes) != 2:
        raise ValueError("balanced_labels assumes exactly two classes")

    base = np.array([codes[0]] * (n // 2) + [codes[1]] * (n // 2))

    for _ in range(max_attempts):
        labels = base.copy()
        rng.shuffle(labels)
        if _max_streak(labels) <= max_consecutive:
            return labels

    raise RuntimeError(
        f"could not generate {n} balanced labels with streaks <= {max_consecutive} "
        f"in {max_attempts} attempts"
    )


def _max_streak(labels: np.ndarray) -> int:
    best = current = 1
    for i in range(1, len(labels)):
        current = current + 1 if labels[i] == labels[i - 1] else 1
        best = max(best, current)
    return best


def build_schedule(config: ParadigmConfig, seed: int = 0) -> list[Trial]:
    """Generate the full trial list for a session."""
    rng = np.random.default_rng(seed)
    name_by_code = {v: k for k, v in EVENT_IDS.items()}
    lo, hi = config.rest_range_s

    trials: list[Trial] = []
    index = 0
    for run in range(config.n_runs):
        labels = balanced_labels(config.trials_per_run, config.max_consecutive_same, rng)
        for label in labels:
            trials.append(
                Trial(
                    index=index,
                    run=run,
                    label=int(label),
                    label_name=name_by_code[int(label)],
                    rest_s=float(rng.uniform(lo, hi)),
                )
            )
            index += 1
    return trials


def estimate_duration_s(config: ParadigmConfig) -> float:
    """Wall-clock estimate for the whole session, quality blocks included.

    Worth printing before a session starts: subjects fatigue, and MI decoding
    degrades noticeably in a tired subject, so knowing up front that the plan
    runs to 40 minutes is actionable.
    """
    trial_time = config.n_trials * config.trial_duration_s
    blocks = 2 * config.alpha_block_s + config.idle_block_s
    breaks = max(config.n_runs - 1, 0) * 30.0  # assume ~30 s per self-paced break
    return trial_time + blocks + breaks
