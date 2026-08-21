#!/usr/bin/env python
"""Electrode-quality check on a recorded session.

Equivalent to `eegbot check-signal`.

    python scripts/check_signal.py data/raw/sub-01/ses-1

Run this on the alpha bookend blocks immediately after recording, while the
subject is still available. Posterior alpha should rise clearly on eye closure
at P3/Pz/P4; if it does not, the session is probably unusable and a refit-and-
retry is far cheaper than discovering it days later.
"""

import sys

from eegbot.cli import main

if __name__ == "__main__":
    sys.exit(main(["check-signal", *sys.argv[1:]]))
