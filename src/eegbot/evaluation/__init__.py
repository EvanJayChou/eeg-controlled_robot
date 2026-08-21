"""Evaluation protocols, scoring, and control simulation."""

from eegbot.evaluation.control_sim import ControlMetrics, simulate_control
from eegbot.evaluation.metrics import SplitResult, report, results_frame, score_split
from eegbot.evaluation.protocols import Split, assert_no_group_leak, get_protocol

__all__ = [
    "ControlMetrics",
    "Split",
    "SplitResult",
    "assert_no_group_leak",
    "get_protocol",
    "report",
    "results_frame",
    "score_split",
    "simulate_control",
]
