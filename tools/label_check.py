"""D20 Step 0 / PLAN L2: what first-touch semantics put in the DFL labels, and what a
controlled-possession relabelling (solution/relabel.py) makes of them.

    .venv/bin/python tools/label_check.py              # IDSSE dev matches, clean labels
    .venv/bin/python tools/label_check.py --public     # + the public window, structure only

For each label variant: possessions per 90 graded minutes, the share under 1 and 2 s, median and
mean length, and direct H<->A changes per alive minute (the units of EXPERIMENTS section 4).
Then, per DFL possession by length, what the relabelling does to it:
  - kept: the same frames are still one possession of the same team;
  - absorbed: its frames now belong to the other team (a touch, not control);
  - grown: it is now part of a longer possession of its own team, which swallowed a touch.

--public adds the public window and that window put through the same rule. It is there only
to see whether the window already looks like controlled possession (a rule that barely
changes it), never to choose a variant or a value (PLAN L2, D20).
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import relabel  # noqa: E402
from solution.match import pm  # noqa: E402
from solution.sources import idsse  # noqa: E402

PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
BINS = ((0, 1), (1, 2), (2, 5), (5, 10), (10, np.inf))


def fate(dfl, new, fps):
    """{bin: [n, kept, absorbed, grown]} for every DFL possession."""
    runs = pm.valid_runs(dfl)
    new_spells = {(a, b, k) for a, b, k in pm.possession_spells(new, runs)}
    out = {b: [0, 0, 0, 0] for b in BINS}
    for a, b, k in pm.possession_spells(dfl, runs):
        seconds = (b - a) / fps
        row = out[next(x for x in BINS if x[0] <= seconds < x[1])]
        row[0] += 1
        if (a, b, k) in new_spells:
            row[1] += 1
        elif np.all(new[a:b] == k):
            row[3] += 1
        else:
            row[2] += 1
    return out


def print_rows(rows):
    print("{:<28} {:>6} {:>7} {:>7} {:>7} {:>8} {:>7} {:>12}".format(
        "labels", "poss", "per 90", "<1 s", "<2 s", "median s", "mean s", "H<->A/alive"))
    for name, s in rows:
        print("{:<28} {:>6} {:>7.0f} {:>7.3f} {:>7.3f} {:>8.1f} {:>7.1f} {:>12.2f}".format(
            name, s["n"], s["per90"], s["under1"], s["under2"], s["median"], s["mean"],
            s["direct_per_alive_min"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matches", nargs="+", default=list(idsse.DEV_IDS))
    parser.add_argument("--public", action="store_true",
                        help="also describe the public window (structure only, never choose)")
    parser.add_argument("--keep", nargs="+", type=float, default=[1.0, 2.0, 3.0],
                        help="keep_s values for the sensitivity rows (described, not chosen)")
    args = parser.parse_args()
    assert not set(args.matches) & set(idsse.HOLDOUT_IDS), "held-out matches stay locked (D3)"

    raw = [idsse.load(m, labels="dfl") for m in args.matches]
    fps = raw[0].fps
    confirms = [relabel.pass_confirm(idsse.completed_passes(m), fps) for m in args.matches]
    variants = [("dfl", lambda m, c: m.labels)]
    for keep in args.keep:
        variants.append(("keep{:g}".format(keep),
                         lambda m, c, k=keep: relabel.controlled(m.labels, fps, k)))
        variants.append(("keep{:g}pass".format(keep),
                         lambda m, c, k=keep: relabel.controlled(m.labels, fps, k, c)))
    labelled = {name: [fn(m, c) for m, c in zip(raw, confirms)] for name, fn in variants}

    print("IDSSE dev, clean labels: {} matches. keep2 and keep2pass are the registered "
          "variants (relabel.VARIANTS); the others are sensitivity rows.\n".format(len(raw)))
    print_rows([(name, relabel.summary([relabel.stats(x, fps) for x in labelled[name]]))
                for name, _ in variants])

    passes = sum(len(idsse.completed_passes(m)[0]) for m in args.matches)
    print("\ncompleted passes in the event feed: {} ({:.1f} per match)".format(
        passes, passes / len(args.matches)))

    for name in ("keep2", "keep2pass"):
        if name not in labelled:
            continue
        print("\nwhat {} does to the DFL possessions, by DFL length".format(name))
        print("{:<10} {:>6} {:>7} {:>9} {:>7}   {:>8}".format(
            "length s", "n", "kept", "absorbed", "grown", "removed"))
        total = {b: [0, 0, 0, 0] for b in BINS}
        for dfl, new in zip(labelled["dfl"], labelled[name]):
            for b, row in fate(dfl, new, fps).items():
                total[b] = [x + y for x, y in zip(total[b], row)]
        all_rows = np.sum([total[b] for b in BINS], axis=0)
        for b in BINS + ("all",):
            n, kept, absorbed, grown = all_rows if b == "all" else total[b]
            label = "all" if b == "all" else ("{:g}-{:g}".format(*b) if np.isfinite(b[1])
                                               else ">{:g}".format(b[0]))
            print("{:<10} {:>6} {:>7} {:>9} {:>7}   {:>8.3f}".format(
                label, n, kept, absorbed, grown, absorbed / max(n, 1)))
        changed = np.mean([np.mean(d[d != pm.X] != x[x != pm.X])
                           for d, x in zip(labelled["dfl"], labelled[name])])
        print("frames changed: {:.4f} of graded frames".format(changed))

    if args.public:
        gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT)["label"].to_numpy())
        print("\npublic window (structure only, PLAN L2: never used to choose)")
        print_rows([("public truth", relabel.summary([relabel.stats(gt, fps)])),
                    ("public truth -> keep2",
                     relabel.summary([relabel.stats(relabel.controlled(gt, fps), fps)]))])
        print("public frames keep2 changes: {}".format(
            int(np.sum(relabel.controlled(gt, fps) != gt))))


if __name__ == "__main__":
    main()
