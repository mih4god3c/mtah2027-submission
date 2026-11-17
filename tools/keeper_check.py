"""Is the keeper's ball lost by the tracker, and does the keeper model read the zone? Aggregates only.

    .venv/bin/python tools/keeper_check.py                 # proxy: gap lengths + frame-level model
    .venv/bin/python tools/keeper_check.py --bar-ath       # (needs data/) gap lengths on BAR-ATH
    make keeper-check [ARGS=--bar-ath]

Part 1, label-free (solution/keeper.py gaps_by_keeper): every ball gap inside a half, split by
whether the sighting just before it, and separately just after it, is a keeper on the ball in
his own area (within 2 m); and the ball-seen share inside and outside the keeper zone
(keepermodel.zone). denis's reading is that BAR-ATH loses the ball in the keeper's hands: then
gaps that end with the ball at the keeper are more common and longer there than on the proxy,
and the zone's seen share is lower (the proxy's simulator drops the ball blind to the play). The proxy rows are given with the keeper's ball hidden at
keep 1 (no stress), 0.5 and 0, so the BAR-ATH row can be placed between them.

Part 2, proxy only (IDSSE dev x seeds, simulator v1 + D12, keep2 truth): the keeper model
(solution/keepermodel.py, leave-one-match-out) in its zone, argmax state against the truth, for
all zone frames, the keeper on the ball (clean match, truth alive), and true D stretches ending
in a goal kick; with the "always dead" baseline (the zone's majority class) beside it.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import keeper, keepermodel  # noqa: E402
from solution.match import pm  # noqa: E402
from solution.simulate import add_ball_noise, degrade  # noqa: E402

STRESS = (0.05, 3.4, 0.26)
HIDE = (1.0, 0.5, 0.0)


def gap_row(name, results, seen_zone=None):
    """One line per side: gaps next to a keeper sighting against the rest."""
    for side in ("before", "after"):
        lengths = np.concatenate([r[side][0] for r in results])
        flags = np.concatenate([r[side][1] for r in results])
        parts = ["keeper {} share {:.4f}".format(side, np.mean(flags))]
        for label, g in (("keeper", lengths[flags]), ("other", lengths[~flags])):
            if not len(g):
                parts.append("{} --".format(label))
                continue
            parts.append("{} n {} p50 {:.1f} s p90 {:.1f} s >2 s {:.3f}".format(
                label, len(g), np.median(g), np.percentile(g, 90), np.mean(g > 2)))
        print("  {:<26} {}".format(name if side == "before" else "", "  |  ".join(parts)))
    if seen_zone is not None:
        print("  {:<26} ball seen in the keeper zone {:.3f}, outside it {:.3f}".format(
            "", *seen_zone))


def seen_shares(match):
    z = keepermodel.zone(match)
    inside = match.in_play_mask()
    return (float(np.mean(match.ball_seen[inside & (z >= 0)])),
            float(np.mean(match.ball_seen[inside & (z < 0)])))


def proxy(seeds, gaps_only=False):
    from solution.sources import idsse
    from tools.deadball_check import dead_stretches, restart_kinds
    print("Part 1, ball gaps after a keeper sighting (label-free), IDSSE dev x seeds {}, "
          "v1 + D12".format(seeds))
    rows = {k: [] for k in HIDE}
    acc = {k: {"all": [0, 0, 0], "keeper on ball": [0, 0, 0], "goal kick": [0, 0, 0]}
           for k in HIDE}
    for match_id in idsse.DEV_IDS:
        raw = idsse.load(match_id, labels="keep2")
        events = idsse.restarts(match_id)
        truth = raw.labels
        runs = pm.valid_runs(truth)
        true_d = dead_stretches(truth, runs)
        gk = np.zeros(raw.n_frames, dtype=bool)
        for (a, b), kind in zip(true_d, restart_kinds(true_d, events, raw.fps)):
            if kind == "goal_kick":
                gk[a:b] = True
        hold = np.zeros(raw.n_frames, dtype=bool)
        for a, b, _ in keeper.on_ball(raw):
            hold[a:b] = True
        hold &= truth != pm.D
        for seed in seeds:
            base = add_ball_noise(degrade(raw, seed), *STRESS[:2], seed=seed,
                                  smooth_s=STRESS[2])
            for keep in HIDE:
                match = keeper.hide_keeper_ball(base, raw, keep, seed)
                rows[keep].append((keeper.gaps_by_keeper(match), seen_shares(match)))
                if gaps_only:
                    continue
                p, z = keepermodel.probabilities(match)
                zone = z >= 0
                y = keepermodel.oriented_target(truth, z)
                pred = np.argmax(np.nan_to_num(p, nan=-1.0), axis=1)
                for name, mask in (("all", zone), ("keeper on ball", zone & hold),
                                   ("goal kick", zone & gk)):
                    use = mask & (y >= 0)
                    acc[keep][name][0] += int(np.sum(pred[use] == y[use]))
                    acc[keep][name][1] += int(np.sum(use))
                    acc[keep][name][2] += int(np.sum(y[use] == 2))
        print("  done", match_id, flush=True)
    for keep in HIDE:
        gap_row("proxy, keeper kept p {:g}".format(keep), [r[0] for r in rows[keep]],
                tuple(np.mean([r[1] for r in rows[keep]], axis=0)))
    if gaps_only:
        return
    print("Part 2, keeper model in its zone (argmax vs truth; 'always D' beside it):")
    for keep in HIDE:
        print("  keeper's ball kept p {:g}".format(keep))
        for name, (right, n, dead) in acc[keep].items():
            print("    {:<16} frames {:>8}  model {:.3f}  always D {:.3f}".format(
                name, n, right / float(max(n, 1)), dead / float(max(n, 1))))


def bar_ath():
    from solution.sources import atd
    match = atd.load()
    print("Part 1 on BAR-ATH (label-free): ball gaps next to a keeper sighting")
    gap_row("BAR-ATH", [keeper.gaps_by_keeper(match)], seen_shares(match))
    z = keepermodel.zone(match)
    print("  keeper zone share of graded frames {:.3f}".format(
        np.mean(z[match.in_play_mask()] >= 0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bar-ath", action="store_true", help="(needs data/) label-free only")
    parser.add_argument("--gaps-only", action="store_true", help="proxy Part 1 only (no model)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    if args.bar_ath:
        bar_ath()
    else:
        proxy(args.seeds, args.gaps_only)


if __name__ == "__main__":
    main()
