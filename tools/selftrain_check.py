"""C6 self-training checks on the proxy league (PLAN.md C6, E9). Aggregates only.

    .venv/bin/python tools/selftrain_check.py --diag                   # Step 0, frame level
    .venv/bin/python tools/selftrain_check.py --ceiling -- --methods v5_frame_model \
        c6o_gbm_hold_gaps c6o_gbm_hold_all ...                         # Step 0, proxy league

--diag: on the IDSSE dev matches degraded as `make proxy` does (simulator v1 + the D12 ball
  stress, tools/deadball_check.inputs), truth --labels (keep2 by default, D21). Per adapted
  model (selftrain.KINDS x selections), against the base frame model: sign accuracy and
  log-loss on alive frames where the ball is not seen (all, and by seconds since the ball was
  last seen), on seen alive frames not picked, and on the picked ones (cross-fitted, so not
  in-sample); plus how many frames are picked.
--ceiling: registers the Step 0 oracle methods (names c6o_<kind>_<select>_<reading>, below)
  and runs tools/proxy_league.py with the arguments after `--`. Each is v5_frame_model
  (D23) with the frame term's log-odds replaced:
    _gaps: the adapted model on frames without a seen ball, the base model where it is seen
           (so the frame term is read everywhere);
    _all:  the adapted model on every frame;
    _seen: the adapted model where the ball is seen, nothing in gaps (as v5_frame_model).
  The oracle reads match.labels: proxy only, never a submission.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import selftrain  # noqa: E402
from solution.match import pm  # noqa: E402
from solution.methods import _GAPTEAM, METHODS, semi_markov  # noqa: E402

SELECTS = ("hold", "seen")
READINGS = ("gaps", "all", "seen")
SINCE_BINS_S = (0.0, 2.0, 5.0, 15.0, np.inf)


def oracle_method(kind, select, reading):
    params = dict(selftrain.PARAMS, kind=kind)

    def frame_fn(match):
        adapted = selftrain.oracle_log_odds(match, select, params)
        if reading == "gaps":
            base, _ = selftrain.base_inputs(match)
            return selftrain.where_seen(match, base, adapted)
        return adapted
    name = "c6o_{}_{}_{}".format(kind, select, reading)
    return name, semi_markov(name, frame_w=1.0, frame_where="seen" if reading == "seen"
                             else "all", frame_fn=frame_fn, **_GAPTEAM)


def register_oracles():
    for kind in selftrain.KINDS:
        for select in SELECTS:
            for reading in READINGS:
                name, method = oracle_method(kind, select, reading)
                METHODS[name] = method


def _scores(z, y):
    """(n, right, summed log-loss) of log-odds z against 0/1 targets y."""
    ok = np.isfinite(z)
    z, y = z[ok], y[ok]
    right = int(np.sum((z > 0) == (y == 1)))
    loss = float(np.sum(np.where(y == 1, np.logaddexp(0.0, -z), np.logaddexp(0.0, z))))
    return len(z), right, loss


def diag(seeds, labels, kinds=selftrain.KINDS, selects=SELECTS):
    from solution import touch
    from solution.sources import idsse
    from tools.deadball_check import inputs
    from tools.gap_check import bin_label, since_seen
    variants = [(k, s) for k in kinds for s in selects]
    splits = ["gap, all"] + ["gap, missing " + bin_label(lo, hi)
                             for lo, hi in zip(SINCE_BINS_S[:-1], SINCE_BINS_S[1:])]
    splits += ["seen, not picked", "seen, picked"]
    totals = {}
    picked_n = alive_seen_n = graded = 0
    for truth, match, _, seed in inputs(idsse.DEV_IDS, seeds, False, labels):
        match.labels = truth
        base, _ = selftrain.base_inputs(match)
        holder = touch.detect(match).holder
        picked = selftrain.pickable(match, holder)
        alive = np.isin(truth, pm.POSSESSION_LABELS) & match.in_play_mask()
        y = (truth == pm.H).astype(np.int8)
        since = since_seen(match)
        gap = alive & ~match.ball_seen
        masks = {"gap, all": gap,
                 "seen, not picked": alive & match.ball_seen & ~picked,
                 "seen, picked": alive & picked}
        for lo, hi in zip(SINCE_BINS_S[:-1], SINCE_BINS_S[1:]):
            masks["gap, missing " + bin_label(lo, hi)] = gap & (since >= lo) & (since < hi)
        picked_n += int(np.sum(alive & picked))
        alive_seen_n += int(np.sum(alive & match.ball_seen))
        graded += int(match.in_play_mask().sum())
        signals = {"base": base}
        for kind, select in variants:
            signals[(kind, select)] = selftrain.oracle_log_odds(
                match, select, dict(selftrain.PARAMS, kind=kind))
        for split, mask in masks.items():
            for name, z in signals.items():
                n, right, loss = _scores(z[mask], y[mask])
                t = totals.setdefault((split, name), [0, 0, 0.0])
                t[0] += n
                t[1] += right
                t[2] += loss
    print("C6 Step 0, frame level: IDSSE dev x seeds {}, simulator v1 + D12 ball stress; truth "
          "{}; adapted models cross-fitted ({} folds of {:g} s blocks)".format(
              seeds, labels, selftrain.PARAMS["folds"], selftrain.PARAMS["block_s"]))
    print("picked (ball seen, holder stable >= {:g} s, truth alive): {:.3f} of seen alive "
          "frames, {:.3f} of graded frames".format(selftrain.PARAMS["min_hold_s"],
                                                   picked_n / float(alive_seen_n),
                                                   picked_n / float(graded)))
    names = ["base"] + variants
    print("{:<22} {:>9}".format("split", "frames") + "".join(
        "{:>18}".format(n if n == "base" else "{} {}".format(*n)) for n in names))
    for split in splits:
        n = totals[(split, "base")][0]
        cells = []
        for name in names:
            t = totals[(split, name)]
            cells.append("{:.3f} / {:.3f}".format(t[1] / max(t[0], 1.0), t[2] / max(t[0], 1.0)))
        print("{:<22} {:>9}".format(split, n) + "".join("{:>18}".format(c) for c in cells))
    print("(cells: sign accuracy / mean log-loss against the true team; 'select seen' trains "
          "on every seen alive frame, so its 'seen' rows are cross-fitted but not a fair test "
          "of a pseudo-labeller's coverage)")


def main():
    argv = sys.argv[1:]
    rest = []
    if "--" in argv:
        i = argv.index("--")
        argv, rest = argv[:i], argv[i + 1:]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--diag", action="store_true")
    parser.add_argument("--ceiling", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--labels", default="keep2")
    parser.add_argument("--kinds", nargs="+", default=list(selftrain.KINDS),
                        choices=selftrain.KINDS)
    parser.add_argument("--selects", nargs="+", default=list(SELECTS), choices=SELECTS)
    args = parser.parse_args(argv)
    if args.diag:
        diag(args.seeds, args.labels, args.kinds, args.selects)
    if args.ceiling:
        register_oracles()
        from tools import proxy_league
        sys.argv = ["proxy_league.py"] + rest
        proxy_league.main()


if __name__ == "__main__":
    main()
