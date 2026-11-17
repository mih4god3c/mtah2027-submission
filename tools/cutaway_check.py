"""E13 cut-away cue, what it is worth on the proxy league: the submitted reference against the
same method reading a synthetic TV cut-away cue (solution/cutaway.py), paired on the same inputs.
No BAR-ATH data needed.

    .venv/bin/python tools/cutaway_check.py                    # keep2 truth, D12, seeds 0 1 2
    .venv/bin/python tools/cutaway_check.py --labels dfl       # the D21 "not WORSE" check

Inputs are degraded exactly as `make proxy` with the D12 ball stress does. Each input gets a
synthetic cue made from the truth by cutaway.synthetic, per variant: "calibrated" (decides,
D31; the default), "pessimistic" and "optimistic" (section 17's first run). The reference
ignores it; touch_c3_seen_w1_tv_cal / _tv / _tv_opt read it with that variant's table. Gains are judged by D4's rule with
proxy_league's paired bootstrap (first seed), as `make proxy` judges two methods.

It also prints, per variant, the cue against the truth and against the reference's own
prediction. The second is directly comparable with BAR-ATH's real cue (EXPERIMENTS section
17, tools/broadcast_align.py section F: off pitch on 61.2% of predicted D, 9.6% of predicted
H or A): if the synthetic split is far from that, the imitation is off.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import cutaway  # noqa: E402
from solution.match import pm  # noqa: E402
from solution.methods import METHODS  # noqa: E402
from solution.simulate import add_ball_noise, degrade  # noqa: E402
from solution.sources import idsse  # noqa: E402
from tools.deadball_check import STRESS  # noqa: E402
from tools.proxy_league import bootstrap, f1_from_flags, spell_flags  # noqa: E402

REFERENCE = "v5_frame_model"
VARIANT_METHOD = {"pessimistic": "touch_c3_seen_w1_tv", "optimistic": "touch_c3_seen_w1_tv_opt",
                  "calibrated": "touch_c3_seen_w1_tv_cal"}


def verdict(gain, low, high, wins, n):
    if high < 0:
        return "WORSE"
    if gain >= 0.02 and low > 0 and wins * 3 >= 2 * n:
        return "ADOPT"
    return "better, below the adoption bar" if low > 0 else "within noise"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--labels", default="keep2")
    parser.add_argument("--variants", nargs="+", default=["calibrated"],
                        choices=list(VARIANT_METHOD))
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    raws = [idsse.load(m, labels=args.labels) for m in idsse.DEV_IDS]
    names = [REFERENCE] + [VARIANT_METHOD[v] for v in args.variants]
    parts = {name: [] for name in names}            # parts[name][seed] = (flags, n_pred)
    per_match = {name: np.zeros(len(raws)) for name in names}
    rates = {v: np.zeros((2, 2, 2)) for v in args.variants}   # [truth|pred][alive/dead][on/off]
    for seed in args.seeds:
        got = {name: ([], 0) for name in names}
        for i, raw in enumerate(raws):
            match = add_ball_noise(degrade(raw, seed), *STRESS[:2], seed=seed,
                                   smooth_s=STRESS[2])
            ref = METHODS[REFERENCE](match)
            preds = {REFERENCE: ref}
            graded = np.isin(raw.labels, (pm.H, pm.A, pm.D))
            for v in args.variants:
                tv = cutaway.synthetic(raw, raw.labels, seed, v)
                with_tv = match.copy()
                with_tv.notes["tv"] = tv
                preds[VARIANT_METHOD[v]] = METHODS[VARIANT_METHOD[v]](with_tv)
                known = graded & (tv >= 0)
                for k, dead in enumerate((raw.labels == pm.D, ref == pm.D)):
                    for row, mask in ((0, known & ~dead), (1, known & dead)):
                        rates[v][k, row, 1] += (mask & (tv == 1)).sum()
                        rates[v][k, row, 0] += (mask & (tv == 0)).sum()
            for name, pred in preds.items():
                f, n = spell_flags(raw.labels, pred)
                per_match[name][i] += pm.evaluate(raw.labels, pred)["spell_f1_mean"]
                flags, total = got[name]
                flags.append(f)
                got[name] = (flags, total + n)
        for name, (flags, total) in got.items():
            parts[name].append((np.concatenate(flags, axis=1), total))

    print("proxy: {} dev matches x seeds {}, v1 + D12 ball stress, truth {}".format(
        len(raws), args.seeds, args.labels))
    for v in args.variants:
        off = rates[v][:, :, 1] / rates[v].sum(axis=2)
        print("synthetic cue, {}: off pitch on true D {:.3f}, true live {:.3f}; on predicted D "
              "{:.3f}, predicted live {:.3f}  (BAR-ATH real: predicted D 0.612, live 0.096)"
              .format(v, off[0, 1], off[0, 0], off[1, 1], off[1, 0]))
        table = np.exp(cutaway.idsse_table("", v))[:, 1]
        print("  decoder table (all five dev matches): P(off | alive) {:.4f}, P(off | dead) "
              "{:.4f}".format(table[0], table[1]))
    pooled = {name: float(np.mean([f1_from_flags(f, n) for f, n in parts[name]]))
              for name in names}
    print("{}: {:.4f}".format(REFERENCE, pooled[REFERENCE]))
    for v in args.variants:
        name = VARIANT_METHOD[v]
        low, high = bootstrap(*parts[REFERENCE][0], args.bootstrap, other=parts[name][0])
        gain = pooled[name] - pooled[REFERENCE]
        wins = int(np.sum(per_match[name] > per_match[REFERENCE]))
        print("{} ({}): {:.4f}, gain {:+.4f}, interval {:+.4f}..{:+.4f}, wins {}/{}  {}".format(
            name, v, pooled[name], gain, low, high, wins, len(raws),
            verdict(gain, low, high, wins, len(raws))))


if __name__ == "__main__":
    main()
