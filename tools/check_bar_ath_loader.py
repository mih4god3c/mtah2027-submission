"""Check the BAR-ATH loader against the committed baselines. Prints agreement shares only.

    .venv/bin/python tools/check_bar_ath_loader.py            # needs data/, about two minutes

For a HUMAN to run: it reads the gated match data. Its output is aggregate (percentages and
counts), so it is safe to paste back to the team.

What it proves: solution/sources/atd.py reads the tracking the same way starter/baseline.py
does. It rebuilds the four published baselines from a Match and compares them, frame by frame
over the graded halves, with the committed evaluation/results/baseline_*.csv. Expect 100.000%,
or a handful of frames where two players are equidistant from the ball to float precision.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution.match import pm  # noqa: E402
from solution.methods import METHODS  # noqa: E402
from solution.sources import atd  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baselines", default=os.path.join(ROOT, "evaluation", "results"),
                        help="folder holding baseline_*.csv")
    parser.add_argument("--refresh", action="store_true", help="re-parse instead of the cache")
    args = parser.parse_args()

    match = atd.load(refresh=args.refresh)
    in_play = match.in_play_mask()
    print("loaded {}: {} frames, {} track slots, halves {}".format(
        match.name, match.n_frames, len(match.team), match.halves))
    print("ball present on {:.1f}% of graded frames".format(100 * match.ball_seen[in_play].mean()))

    worst = 1.0
    for name in ("baseline_raw", "baseline_smooth25", "baseline_smooth50", "baseline_smooth125"):
        ours = METHODS[name](match)
        published = pm.label_codes(pm.read_label_csv(os.path.join(args.baselines, name + ".csv")))
        agree = float(np.mean(ours[in_play] == published[in_play]))
        worst = min(worst, agree)
        print("  {:<20} agrees on {:.3f}% of graded frames ({} differ)".format(
            name, 100 * agree, int(np.sum(ours[in_play] != published[in_play]))))
    print("\nOK: the loader matches the published baseline." if worst > 0.9999 else
          "\nMISMATCH: the loader reads the tracking differently from starter/baseline.py.")
    return 0 if worst > 0.9999 else 1


if __name__ == "__main__":
    sys.exit(main())
