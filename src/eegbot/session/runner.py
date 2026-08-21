"""The recording session.

Session time with a human subject is the scarcest resource in this project, and
a flawed session is unrecoverable -- you find out during analysis, days later,
with the subject long gone. So the runner is built around catching problems
while the person is still in the chair:

* **Alpha bookends.** 60 s eyes-open then eyes-closed, at the start *and* end.
  Posterior alpha should rise visibly on eye closure at P3/Pz/P4. If it does not,
  contact is bad and the session is worthless -- better to know in minute two
  than in the analysis a week later. Running it again at the end catches an
  electrode that drifted loose partway through.
* **Impedance before every run**, prompted and written to the sidecar. Logged,
  not merely checked, so a bad run can be traced afterwards.
* **An idle block.** Not a third class -- this is what the dead zone and the
  centering bias get fitted from.
* **Self-paced breaks.** MI decoding degrades sharply in a tired subject.

The runner is display- and marker-agnostic, so the entire protocol can be
rehearsed headless before PsychoPy or the headset exist.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from eegbot.config import ParadigmConfig
from eegbot.session import markers as mk
from eegbot.session.cues import Display, HeadlessDisplay
from eegbot.session.markers import MarkerSink, MemoryMarkerSink
from eegbot.session.protocol import Trial, build_schedule, estimate_duration_s

log = logging.getLogger(__name__)

#: Asked before every run. Returns a per-channel impedance reading in kOhm.
ImpedanceProbe = Callable[[], dict[str, float]]


@dataclass
class SessionLog:
    """What happened, for the sidecar."""

    impedances: dict[str, dict[str, float]] = field(default_factory=dict)
    completed_trials: int = 0
    completed_runs: int = 0
    notes: str = ""


class SessionRunner:
    """Drives one full recording session.

    Parameters
    ----------
    impedance_probe
        Called before each run. In practice this prompts the operator to read
        values off DSI-Streamer; a stub returning ``{}`` is fine for dry runs.
    """

    def __init__(
        self,
        config: ParadigmConfig,
        display: Display | None = None,
        sink: MarkerSink | None = None,
        *,
        seed: int = 0,
        impedance_probe: ImpedanceProbe | None = None,
    ) -> None:
        self.config = config
        self.display = display if display is not None else HeadlessDisplay()
        self.sink = sink if sink is not None else MemoryMarkerSink()
        self.seed = seed
        self.impedance_probe = impedance_probe
        self.log = SessionLog()

    # === Blocks ===

    def run_alpha_block(self, position: str) -> None:
        """Eyes-open / eyes-closed bookend. The electrode-quality ground truth."""
        duration = self.config.alpha_block_s

        self.display.message(
            f"[{position}] Relax with your eyes OPEN, looking at the cross.\n"
            "Try not to move. Press a key to begin.",
            wait_for_key=True,
        )
        self.sink.push(mk.ALPHA_EYES_OPEN)
        self.display.fixation()
        self.display.wait(duration)
        self.sink.push(mk.BLOCK_END)

        self.display.message(
            "Now close your eyes and stay relaxed.\n"
            "You will be told when to open them. Press a key to begin.",
            wait_for_key=True,
        )
        self.sink.push(mk.ALPHA_EYES_CLOSED)
        self.display.blank()
        self.display.wait(duration)
        self.sink.push(mk.BLOCK_END)
        self.display.message("Please open your eyes.", wait_for_key=True)

    def run_idle_block(self) -> None:
        """Relaxed no-imagery rest -- the calibration data for the dead zone."""
        self.display.message(
            "Rest. Look at the cross and let your mind wander.\n"
            "Do NOT imagine any movement. Press a key to begin.",
            wait_for_key=True,
        )
        self.sink.push(mk.IDLE_BLOCK)
        self.display.fixation()
        self.display.wait(self.config.idle_block_s)
        self.sink.push(mk.BLOCK_END)

    def run_trial(self, trial: Trial) -> None:
        self.sink.push(mk.FIXATION)
        self.display.fixation()
        self.display.wait(self.config.fixation_s)

        direction = "left" if trial.label_name == "left_hand" else "right"
        self.sink.push(mk.cue_marker(trial.label_name))
        self.display.arrow(direction)
        self.display.wait(self.config.cue_s)

        # The arrow stays up through imagery: removing it would create a second
        # visual transient in the middle of the decoding window.
        self.sink.push(mk.IMAGERY_START)
        self.display.wait(self.config.imagery_s)
        self.sink.push(mk.IMAGERY_END)

        self.sink.push(mk.REST)
        self.display.blank()
        self.display.wait(trial.rest_s)

        self.log.completed_trials += 1

    def run_block(self, run_index: int, trials: list[Trial]) -> None:
        if self.impedance_probe is not None:
            readings = self.impedance_probe()
            self.log.impedances[f"run-{run_index}"] = readings
            bad = {ch: v for ch, v in readings.items() if v > 20.0}
            if bad:
                log.warning("high impedance before run %d: %s", run_index, bad)

        self.display.message(
            f"Run {run_index + 1} of {self.config.n_runs}.\n"
            f"{len(trials)} trials. Press a key when you are ready.",
            wait_for_key=True,
        )
        self.sink.push(mk.RUN_START)
        for trial in trials:
            self.run_trial(trial)
        self.sink.push(mk.RUN_END)
        self.log.completed_runs += 1

    # === Whole session ===

    def run(self) -> SessionLog:
        schedule = build_schedule(self.config, seed=self.seed)
        minutes = estimate_duration_s(self.config) / 60.0

        try:
            self.display.message(
                self.config.instructions or "Motor imagery session.",
                wait_for_key=True,
            )
            self.display.message(
                f"This session takes about {minutes:.0f} minutes, with breaks.\n"
                "Press a key to start.",
                wait_for_key=True,
            )

            self.run_alpha_block("start")
            self.run_idle_block()

            for run_index in range(self.config.n_runs):
                trials = [t for t in schedule if t.run == run_index]
                self.run_block(run_index, trials)
                if run_index < self.config.n_runs - 1:
                    self.display.message(
                        "Break. Rest as long as you like.\nPress a key to continue.",
                        wait_for_key=True,
                    )

            # Repeated at the end to catch an electrode that drifted loose
            # partway through -- otherwise indistinguishable from a subject who
            # simply performed poorly.
            self.run_alpha_block("end")

            self.display.message("Session complete. Thank you!", wait_for_key=True)
        finally:
            self.sink.close()
            self.display.close()

        return self.log
