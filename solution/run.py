"""Our submission: data in, possession file out. The one command the deliverables ask for.

    python solution/run.py --team <team_name>        # or: make submission TEAM=<team_name>

Writes evaluation/results/<team_name>.csv and validates it. The method is `v7_keep2_control` (D39; `dm_k2ctl` until D40):
the semi-Markov decoder (solution/decoder.py, DECISIONS D8 and D9) reading the touch engine's
holder as well (solution/touch.py, D10), with each half dead ball until the kickoff found from
the players (solution/kickoff.py, D13, D14), the teams' width difference as team evidence
(solution/gapteam.py, D16), where the ball is seen the per-frame team model of
solution/framemodel.py (C3, D23), the learned dead-ball model of solution/deadmodel.py in place
of the speed likelihood (D26, D37), the decoder's priors fitted on controlled-possession
relabels of the IDSSE labels (solution/relabel.py, keep2, D21, D38) and the control model of
solution/controlmodel.py that keeps a possession through brief contacts (D35, D39), on the
BAR-ATH tactical ATD. `v6_deadball_model` (D37; `touch_c3_seen_w1_dm` until D40) stays registered as the fallback. Its priors
and the three models are fitted at run time on the five IDSSE dev matches, never the held-out
ones (D3), so the external data must be downloaded first: `make external`. Freezing them into
committed files is a Phase 4 job.

Comparison files. `--method <name>` (or `make submission METHOD=<name> TEAM=<file name>`) writes
the file of any other registered method (solution/methods.py), for instance a registered
candidate whose BAR-ATH check failed, so a person can set it against the video. Without
`--method` the output is the submission and is byte-identical to before this flag existed. A
file written with another method is a comparison file, not the submission: switching what we
submit is a decision (docs/DECISIONS.md), never a flag. The TV cut-away and ball-merge methods
(`*_tv*`, `*_fused`) read the broadcast alignment cache that `make broadcast-align` writes.

Rules worth keeping in view while editing this file (RULES.md):
  - no hand labels or frame-specific constants for this game -- everything comes out of code
  - the public 5 minutes are for checking the pipeline, never for tuning
  - the output must be byte-identical on every run: seed anything random
"""

import argparse
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
import baseline  # noqa: E402
from solution.match import pm  # noqa: E402
from solution.methods import METHODS  # noqa: E402
from solution.sources import atd, idsse  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
METHOD = "v7_keep2_control"


def require_external(data_dir=idsse.DATA_DIR):
    """Stop with a clear message if the IDSSE dev matches the priors are fitted on are missing."""
    missing = [m for m in idsse.DEV_IDS if not idsse.available(m, data_dir)]
    if missing:
        raise SystemExit(
            "solution/run.py fits its decoder priors on the IDSSE dev matches, and {} of them "
            "are not downloaded ({}).\nRun `make external` first (about 2.5 GB, pinned and "
            "checksummed), then run this again.".format(len(missing), ", ".join(missing)))


def needs_alignment(method):
    """True for the registered methods that read the broadcast alignment cache (E13)."""
    settings = getattr(METHODS[method], "settings", {})
    config = getattr(METHODS[method], "config", {})
    return (bool(settings.get("tv_w", 0.0) > 0) or bool(config.get("merge_ball"))
            or method.endswith("_fused"))


def require_alignment(method):
    """Stop with a clear message if `method` needs the broadcast alignment and it is missing."""
    if not needs_alignment(method):
        return
    from solution.sources.fused import ALIGN_PATH
    if not os.path.isfile(ALIGN_PATH):
        raise SystemExit(
            "{} reads the broadcast alignment, and {} does not exist.\nRun `make broadcast-align` "
            "first (needs data/; a few minutes), then run this again.".format(
                method, os.path.relpath(ALIGN_PATH, ROOT)))


def predict(match=None, method=METHOD):
    """One label per tactical video frame: an array of 'H' / 'A' / 'D' / 'X', length 199319.

    X exactly outside the graded halves, H / A / D everywhere inside them. `method` names an
    entry of solution.methods.METHODS; the default is the submission.
    """
    if method not in METHODS:
        raise ValueError("unknown method {!r}; registered: {}".format(method, sorted(METHODS)))
    if match is None:
        match = atd.load()
    codes = METHODS[method](match)
    return np.array(pm.LABELS)[codes]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--team", default=os.environ.get("TEAM", "our_team"),
                        help="file name for the submission; the board names the row after it")
    parser.add_argument("--out", default=RESULTS_DIR, help="where to write")
    parser.add_argument("--method", default=METHOD, choices=sorted(METHODS), metavar="NAME",
                        help="registered method to write a comparison file with (default: the "
                             "submission, {})".format(METHOD))
    args = parser.parse_args()

    require_external()
    require_alignment(args.method)
    if args.method != METHOD:
        print("comparison file: method {} (the submission stays {})".format(args.method, METHOD))
    path = baseline.write_submission(os.path.join(args.out, args.team + ".csv"),
                                     predict(method=args.method))
    print("wrote", os.path.relpath(path, ROOT), "with", args.method)
    sys.exit(subprocess.call([sys.executable,
                              os.path.join(ROOT, "evaluation", "validate_submission.py"), path]))
