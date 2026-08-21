"""Cue presentation.

Two backends behind one interface. `HeadlessDisplay` does everything the real
one does except draw, which is what makes the session runner testable and lets
the team rehearse the whole protocol -- timings, markers, operator prompts --
before PsychoPy is installed or the headset arrives.

PsychoPy is imported lazily inside `PsychoPyDisplay`, so importing this module
costs nothing in an analysis environment.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable


@runtime_checkable
class Display(Protocol):
    """Minimal presentation surface for the MI paradigm."""

    def fixation(self) -> None: ...
    def arrow(self, direction: str) -> None: ...
    def blank(self) -> None: ...
    def message(self, text: str, wait_for_key: bool = False) -> None: ...
    def wait(self, seconds: float) -> None: ...
    def close(self) -> None: ...


class HeadlessDisplay:
    """No-op display that records what it was asked to show.

    `speed` compresses waits so a full 40-minute session dry-runs in seconds;
    set it to 1.0 to rehearse at real time.
    """

    def __init__(self, speed: float = 0.0, verbose: bool = False) -> None:
        self.speed = speed
        self.verbose = verbose
        self.shown: list[tuple[str, str]] = []

    def _record(self, kind: str, value: str = "") -> None:
        self.shown.append((kind, value))
        if self.verbose:
            print(f"[display] {kind} {value}".rstrip())

    def fixation(self) -> None:
        self._record("fixation")

    def arrow(self, direction: str) -> None:
        self._record("arrow", direction)

    def blank(self) -> None:
        self._record("blank")

    def message(self, text: str, wait_for_key: bool = False) -> None:
        self._record("message", text[:40])

    def wait(self, seconds: float) -> None:
        if self.speed > 0:
            time.sleep(seconds / self.speed)

    def close(self) -> None:
        self._record("close")


class PsychoPyDisplay:
    """Real cue presentation.

    Arrows are drawn as vector shapes rather than text so the visual transient
    at cue onset is identical for left and right. A left/right difference in
    stimulus energy would land in the EEG as a visual evoked response that
    correlates perfectly with the label -- and the classifier would find it. The
    0.5 s offset in `PreprocessSpec.trial_window` is the other half of that
    defense.
    """

    def __init__(
        self,
        fullscreen: bool = True,
        size: tuple[int, int] = (1280, 720),
        background: str = "black",
    ) -> None:
        from psychopy import event, visual  # noqa: F401

        self._event = event
        self._win = visual.Window(
            size=size, fullscr=fullscreen, color=background, units="height"
        )
        self._fixation = visual.ShapeStim(
            self._win,
            vertices=((0, -0.04), (0, 0.04), (0, 0), (-0.04, 0), (0.04, 0)),
            lineWidth=4,
            closeShape=False,
            lineColor="white",
        )
        self._arrows = {
            direction: visual.ShapeStim(
                self._win,
                vertices=self._arrow_vertices(direction),
                fillColor="white",
                lineColor="white",
            )
            for direction in ("left", "right")
        }
        self._text = visual.TextStim(self._win, text="", color="white", height=0.04, wrapWidth=1.2)

    @staticmethod
    def _arrow_vertices(direction: str) -> tuple[tuple[float, float], ...]:
        sign = -1.0 if direction == "left" else 1.0
        return (
            (sign * 0.15, 0.0),
            (sign * 0.05, 0.08),
            (sign * 0.05, 0.03),
            (-sign * 0.12, 0.03),
            (-sign * 0.12, -0.03),
            (sign * 0.05, -0.03),
            (sign * 0.05, -0.08),
        )

    def fixation(self) -> None:
        self._fixation.draw()
        self._win.flip()

    def arrow(self, direction: str) -> None:
        self._arrows[direction].draw()
        self._win.flip()

    def blank(self) -> None:
        self._win.flip()

    def message(self, text: str, wait_for_key: bool = False) -> None:
        self._text.text = text
        self._text.draw()
        self._win.flip()
        if wait_for_key:
            self._event.waitKeys()

    def wait(self, seconds: float) -> None:
        from psychopy import core

        core.wait(seconds)

    def close(self) -> None:
        self._win.close()
