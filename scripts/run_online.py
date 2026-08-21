#!/usr/bin/env python
"""Live decoding. Prints the turn rate; does not drive anything.

Equivalent to `eegbot online`.

    python scripts/run_online.py --model model.joblib --replay data/raw/sub-01/ses-1/eeg_raw.fif

No robot output by design -- `ControlCommand` is the project boundary, and
wiring it to motors is separate work.
"""

import sys

from eegbot.cli import main

if __name__ == "__main__":
    sys.exit(main(["online", *sys.argv[1:]]))
