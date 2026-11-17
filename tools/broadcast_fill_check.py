"""E13 ball merge, what it is worth on the proxy league: a paired run with the ball put back into
the simulated gaps the way the broadcast fills BAR-ATH's. No BAR-ATH data needed.

    .venv/bin/python tools/broadcast_fill_check.py                      # D12 stress, seeds 0 1 2
    .venv/bin/python tools/broadcast_fill_check.py --methods v5_frame_model --plain

The proxy matches have no TV feed, so the merge is imitated. Each match is degraded exactly as
`make proxy` does (simulator v1, then the D12 ball stress). The "filled" input differs in one
step: between the two, part of each ball gap gets its clean ball back. That share comes from
FILL_PROFILE, per gap length: the share of BAR-ATH gap frames the broadcast filled
(EXPERIMENTS section 17, section E of tools/broadcast_align.py). It is restored as one chunk at
the gap's start or end (a coin flip): the TV is on the pitch as the ball is lost or found again,
and cut away in between. Chunks shorter than broadcast.FILL_MIN_RUN are dropped, as fuse_ball
drops them. The D12 noise then applies to the restored ball as to any other.

Both inputs go to the same method, and the gain is judged by D4's rule with proxy_league's
paired bootstrap (first seed), exactly as `make proxy` judges two methods. Settings: FILL_PROFILE
and the placement were fixed from the BAR-ATH aggregates before the first run.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import broadcast as bc  # noqa: E402
from solution.match import pm  # noqa: E402
from solution.methods import METHODS  # noqa: E402
from solution.simulate import add_ball_noise, degrade  # noqa: E402
from solution.sources import idsse  # noqa: E402
from tools.deadball_check import STRESS  # noqa: E402
from tools.proxy_league import bootstrap, f1_from_flags, spell_flags  # noqa: E402

FILL_PROFILE = bc.FILL_PROFILE
fill_gaps = bc.imitate_fill


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--methods", nargs="+", default=["v5_frame_model"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--labels", default="keep2")
    parser.add_argument("--plain", action="store_true", help="no D12 ball stress")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    raws = [idsse.load(m, labels=args.labels) for m in idsse.DEV_IDS]
    flags = {(name, kind): [] for name in args.methods for kind in ("base", "filled")}
    per_match = {(name, kind): np.zeros(len(raws)) for name in args.methods
                 for kind in ("base", "filled")}
    seen_share = {"base": [], "filled": []}
    for s, seed in enumerate(args.seeds):
        parts = {key: ([], 0) for key in flags}
        for i, raw in enumerate(raws):
            base = degrade(raw, seed)
            inputs = {"base": base, "filled": fill_gaps(base, raw, seed)}
            for kind, match in inputs.items():
                if not args.plain:
                    match = add_ball_noise(match, *STRESS[:2], seed=seed, smooth_s=STRESS[2])
                graded = match.in_play_mask()
                seen_share[kind].append(float(match.ball_seen[graded].mean()))
                for name in args.methods:
                    pred = METHODS[name](match)
                    f, n = spell_flags(raw.labels, pred)
                    per_match[(name, kind)][i] += pm.evaluate(raw.labels, pred)["spell_f1_mean"]
                    got, total = parts[(name, kind)]
                    got.append(f)
                    parts[(name, kind)] = (got, total + n)
        for key, (got, total) in parts.items():
            flags[key].append((np.concatenate(got, axis=1), total))

    print("proxy: {} dev matches x seeds {}, {}, truth {}".format(
        len(raws), args.seeds, "plain v1" if args.plain else "v1 + D12 ball stress",
        args.labels))
    print("ball seen on graded frames: base {:.3f}, filled {:.3f}".format(
        np.mean(seen_share["base"]), np.mean(seen_share["filled"])))
    for name in args.methods:
        pooled = {kind: float(np.mean([f1_from_flags(f, n) for f, n in flags[(name, kind)]]))
                  for kind in ("base", "filled")}
        low, high = bootstrap(*flags[(name, "base")][0], args.bootstrap,
                              other=flags[(name, "filled")][0])
        gain = pooled["filled"] - pooled["base"]
        wins = int(np.sum(per_match[(name, "filled")] > per_match[(name, "base")]))
        if high < 0:
            verdict = "WORSE"
        elif gain >= 0.02 and low > 0 and wins * 3 >= 2 * len(raws):
            verdict = "ADOPT"
        elif low > 0:
            verdict = "better, below the adoption bar"
        else:
            verdict = "within noise"
        print("{}: base {:.4f}, filled {:.4f}, gain {:+.4f}, interval {:+.4f}..{:+.4f}, "
              "wins {}/{}  {}".format(name, pooled["base"], pooled["filled"], gain, low, high,
                                      wins, len(raws), verdict))


if __name__ == "__main__":
    main()
