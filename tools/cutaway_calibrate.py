"""D31: calibrate the synthetic cut-away cue to BAR-ATH's real split. Reads no score.

    .venv/bin/python tools/cutaway_calibrate.py

On the proxy inputs `tools/cutaway_check.py` uses (D12, dev x seeds 0-2), it predicts with the
reference once, then for each (spill_p, false_share) on a fixed grid builds the synthetic cue
(all other settings as the section 17 variants) and measures its off-pitch share on the
reference's predicted D and predicted live frames. The pair closest to BAR-ATH's real 0.612 and
0.096 (EXPERIMENTS section 17, tools/broadcast_align.py section F; sum of squared differences)
is printed, to be written into cutaway.VARIANTS["calibrated"]. Labels are read only to make the
cue (as cutaway.synthetic always does), never to score.
"""

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

TARGET = (0.612, 0.096)              # BAR-ATH: off pitch on predicted D, on predicted live
SPILL_GRID = np.round(np.arange(0.0, 0.501, 0.05), 3)
FALSE_GRID = np.round(np.arange(0.0, 0.02501, 0.0025), 4)
SEEDS = (0, 1, 2)
REFERENCE = "v5_frame_model"


def main():
    base = dict(cutaway.VARIANTS["pessimistic"])
    inputs = []
    for m in idsse.DEV_IDS:
        raw = idsse.load(m, labels="keep2")
        for seed in SEEDS:
            match = add_ball_noise(degrade(raw, seed), *STRESS[:2], seed=seed,
                                   smooth_s=STRESS[2])
            pred = METHODS[REFERENCE](match)
            graded = np.isin(raw.labels, (pm.H, pm.A, pm.D))
            inputs.append((raw, seed, graded & (pred == pm.D),
                           graded & np.isin(pred, (pm.H, pm.A))))
        print("predicted", m, flush=True)
    rows = []
    for spill in SPILL_GRID:
        for false in FALSE_GRID:
            spec = dict(base, spill_p=float(spill), false_share=float(false))
            counts = np.zeros((2, 2))
            for raw, seed, pdead, plive in inputs:
                tv = cutaway.synthetic(raw, raw.labels, seed, spec)
                for k, mask in enumerate((pdead, plive)):
                    known = mask & (tv >= 0)
                    counts[k] += ((known & (tv == 1)).sum(), known.sum())
            share = counts[:, 0] / counts[:, 1]
            loss = float(np.sum((share - TARGET) ** 2))
            rows.append((loss, float(spill), float(false), share[0], share[1]))
    rows.sort()
    print("\ntarget: off on predicted D {:.3f}, predicted live {:.3f}".format(*TARGET))
    print("best pairs (spill_p, false_share -> off on predicted D, predicted live, loss):")
    for loss, spill, false, d, live in rows[:8]:
        print("  {:.2f}  {:.4f}  ->  {:.3f}  {:.3f}   {:.6f}".format(spill, false, d, live, loss))
    loss, spill, false, _, _ = rows[0]
    print("\nchosen: spill_p {:.2f}, false_share {:.4f}".format(spill, false))


if __name__ == "__main__":
    main()
