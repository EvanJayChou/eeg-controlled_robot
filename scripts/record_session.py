#!/usr/bin/env python
"""Run a recording session, or rehearse one headless.

Equivalent to `eegbot record`.

    python scripts/record_session.py --dry-run --speed 1   # rehearse, real timings
    python scripts/record_session.py                       # live (needs PsychoPy + LSL)

Rehearse before every real session. The dry run exercises the full protocol --
timings, marker order, operator prompts, impedance checks -- with no headset, so
the first time a subject sits down is not also the first time the code runs.
See the runbook in README.md.
"""

import sys

from eegbot.cli import main

if __name__ == "__main__":
    sys.exit(main(["record", *sys.argv[1:]]))
