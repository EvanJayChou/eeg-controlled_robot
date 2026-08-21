"""Scoring and reporting.

## Two accuracies, both reported

**Crop-level** accuracy scores each 1-second window independently. It is what
the control law actually consumes, so it is the number that predicts how the
robot will feel.

**Trial-level** accuracy averages the crop probabilities within a trial and
scores the result. This is the number comparable to published Lee2019 results,
where a whole 4-second trial gets one decision. It is always higher than
crop-level, and quoting it as though it described online performance would be
misleading.

## Always per subject

A grand mean over subjects hides the thing you most need to know. Motor-imagery
performance is bimodal -- a substantial minority of people sit at chance no
matter how good the pipeline is. A mean of 70% could be everyone at 70%, or half
the group at 85% and half at 55%, and those two worlds call for completely
different next steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score
from sklearn.pipeline import Pipeline

from eegbot.constants import EVENT_IDS, POSITIVE_CLASS
from eegbot.datasets.base import EpochSet
from eegbot.evaluation.protocols import Split, assert_no_group_leak


@dataclass
class SplitResult:
    name: str
    subject: str
    n_train_trials: int
    n_test_trials: int
    crop_accuracy: float
    trial_accuracy: float
    kappa: float
    auc: float
    extra: dict[str, Any] = field(default_factory=dict)


def adapt_alignment(pipe: Pipeline, X_test: np.ndarray) -> None:
    """Re-home the Riemannian alignment onto the test session.

    Uses only the test *features*, never its labels, so this is not leakage --
    it is exactly what the online loop does when a subject sits down: compute
    the session's mean covariance from unlabelled data and align to it.

    No-op for pipelines without an alignment step.
    """
    if "align" not in getattr(pipe, "named_steps", {}):
        return
    Xt = X_test
    for name, step in pipe.steps:
        if name == "align":
            step.adapt(Xt)
            return
        Xt = step.transform(Xt)


def _trial_level(y_true: np.ndarray, proba: np.ndarray, groups: np.ndarray) -> float:
    """Average crop probabilities within each trial, then score."""
    correct = 0
    trials = np.unique(groups)
    positive_code = EVENT_IDS[POSITIVE_CLASS]
    for trial in trials:
        mask = groups == trial
        mean_p = float(np.mean(proba[mask]))
        predicted = positive_code if mean_p >= 0.5 else _other_code(positive_code)
        truth = y_true[mask][0]
        correct += int(predicted == truth)
    return correct / len(trials)


def _other_code(code: int) -> int:
    others = [v for v in EVENT_IDS.values() if v != code]
    return others[0]


def score_split(
    pipe: Pipeline,
    epochs: EpochSet,
    split: Split,
    *,
    check_leak: bool = True,
) -> SplitResult:
    """Fit on `split.train`, score on `split.test`."""
    if check_leak:
        assert_no_group_leak(epochs, split)

    model = clone(pipe)
    X_train, y_train = epochs.X[split.train], epochs.y[split.train]
    X_test, y_test = epochs.X[split.test], epochs.y[split.test]

    model.fit(X_train, y_train)
    adapt_alignment(model, X_test)

    positive_code = EVENT_IDS[POSITIVE_CLASS]
    class_index = list(model.classes_).index(positive_code)
    proba = model.predict_proba(X_test)[:, class_index]
    predictions = model.predict(X_test)

    subjects = np.unique(epochs.subjects[split.test])
    return SplitResult(
        name=split.name,
        subject=subjects[0] if len(subjects) == 1 else "mixed",
        n_train_trials=len(np.unique(epochs.groups[split.train])),
        n_test_trials=len(np.unique(epochs.groups[split.test])),
        crop_accuracy=float(accuracy_score(y_test, predictions)),
        trial_accuracy=_trial_level(y_test, proba, epochs.groups[split.test]),
        kappa=float(cohen_kappa_score(y_test, predictions)),
        auc=float(roc_auc_score((y_test == positive_code).astype(int), proba)),
    )


def results_frame(results: list[SplitResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "split": r.name,
                "subject": r.subject,
                "train_trials": r.n_train_trials,
                "test_trials": r.n_test_trials,
                "crop_acc": r.crop_accuracy,
                "trial_acc": r.trial_accuracy,
                "kappa": r.kappa,
                "auc": r.auc,
            }
            for r in results
        ]
    )


def per_subject(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("subject")[["crop_acc", "trial_acc", "kappa", "auc"]]
        .mean()
        .sort_values("trial_acc", ascending=False)
    )


def report(df: pd.DataFrame, *, decoder: str, protocol: str) -> str:
    """Render a markdown report, with the sanity checks that matter inline."""
    subject_df = per_subject(df)
    crop_mean = df["crop_acc"].mean()
    trial_mean = df["trial_acc"].mean()
    trial_std = df["trial_acc"].std()
    near_chance = int((subject_df["trial_acc"] < 0.60).sum())

    lines = [
        f"# {decoder} / {protocol}",
        "",
        f"- subjects: **{len(subject_df)}**, splits: **{len(df)}**",
        f"- crop-level accuracy (what the controller sees): **{crop_mean:.3f}**",
        f"- trial-level accuracy (comparable to published): **{trial_mean:.3f}** "
        f"(sd {trial_std:.3f})",
        f"- subjects below 0.60 trial accuracy: **{near_chance}/{len(subject_df)}**",
        "",
    ]

    if trial_mean > 0.85:
        lines += [
            "> **Treat this as a leak until proven otherwise.** 7-channel MI does not",
            "> normally reach this level. Check, in order: crops of one trial split",
            "> across folds; a trial window that includes cue onset (the classifier",
            "> learns the arrow, not the imagery); EMG contamination.",
            "",
        ]
    elif trial_mean < 0.55:
        lines += [
            "> At chance. Before touching the model, verify the ERD is present at all",
            "> (`notebooks/01_erd_sanity.ipynb`) -- a harmonization bug looks exactly",
            "> like this.",
            "",
        ]

    lines += ["## Per subject", "", _as_table(subject_df.round(3)), ""]
    return "\n".join(lines)


def _as_table(df: pd.DataFrame) -> str:
    """Markdown table, falling back to plain text without `tabulate`.

    A missing optional formatter should never lose you a completed evaluation
    run -- especially one that took a long time to produce.
    """
    try:
        return df.to_markdown()
    except ImportError:
        return "```\n" + df.to_string() + "\n```"
