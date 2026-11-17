"""Where does a method lose possessions? The boundary report behind `make proxy BOUNDARIES=1`.

    make proxy METHODS="v1_decoder" BOUNDARIES=1
    .venv/bin/python tools/proxy_league.py --methods semi_markov --boundaries

Step 0 of E3/E6 (PLAN.md C2a, C5): before building touch evidence or boundary snapping, measure
whether the loss is lag, jitter, wrong flips or missed short possessions. Everything here goes
through the official possession_metrics code: transitions() on the abstention-filled
prediction, and match_transitions() (one-to-one, matched on the destination label) for pairing.

It prints, pooled over matches and simulator seeds:
  - per true transition kind (hand-over H<->A, restart D->team, to dead): recall within the
    tolerance, and the latency of the matched ones (pred - true, frames: mean, sd, p10/p50/p90),
    split by whether the degraded ball is seen on the true transition frame. Only transitions
    with the ball seen can be found from ball kinematics;
  - spurious predicted transitions by kind: the wrong flips;
  - true possessions by duration, with how many are matched at IoU 0.5, against the predicted
    count in the same bins: missed short possessions show here;
  - an oracle snap: each predicted transition matched to a true one within +-1.5 s is moved
    onto the true frame, then spellF1 is rescored. That is the most C5 boundary snapping could
    gain with perfect events, the ceiling for E6.
"""

import numpy as np

from solution.match import pm

KINDS = ("hand-over", "restart", "to dead")
SNAP_TOL = 37                         # +-1.5 s at 25 fps, the C5 search window
DURATION_BINS_S = (0.0, 2.0, 5.0, 15.0, np.inf)


def kind(transition):
    _, before, after = transition
    if after == pm.D:
        return "to dead"
    return "restart" if before == pm.D else "hand-over"


def transition_records(truth, pred, seen, tol):
    """([(kind, ball seen, latency or None)] per true transition, [kind] per spurious one)."""
    runs = pm.valid_runs(truth)
    gt = pm.transitions(truth, runs)
    predicted = pm.transitions(pm.fill_abstentions(pred, runs), runs)
    pairs, _, spurious = pm.match_transitions(gt, predicted, tol)
    latency = {g: p[0] - g[0] for g, p in pairs}
    records = [(kind(g), bool(seen[g[0]]), latency.get(g)) for g in gt]
    return records, [kind(p) for p in spurious]


def spell_bins(truth, pred):
    """(n_gt, n_matched at IoU 0.5, n_pred) per duration bin, as int arrays."""
    runs = pm.valid_runs(truth)
    gt = pm.possession_spells(truth, runs)
    predicted = pm.possession_spells(pm.fill_abstentions(pred, runs), runs)
    overlaps = pm.spell_overlaps(gt, predicted)
    matched = pm.matched_gt_flags(overlaps, len(gt), pm.SPELL_IOU_HEADLINE)
    edges = np.asarray(DURATION_BINS_S) * pm.FPS

    def binned(spells, weights=None):
        lengths = np.array([b - a for a, b, _ in spells], dtype=float)
        return np.histogram(lengths, edges, weights=weights)[0].astype(int)
    return binned(gt), binned(gt, matched.astype(float)), binned(predicted)


def snap_to_truth(truth, pred, kinds, tol=SNAP_TOL):
    """pred with each matched transition of a true kind in `kinds` moved onto the true frame.

    A move that would cross a neighbouring boundary is skipped, so segments keep their order.
    """
    runs = pm.valid_runs(truth)
    filled = pm.fill_abstentions(pred, runs)
    gt = pm.transitions(truth, runs)
    predicted = pm.transitions(filled, runs)
    pairs, _, _ = pm.match_transitions(gt, predicted, tol)
    return move_transitions(filled, runs, predicted,
                            {p: g[0] for g, p in pairs if kind(g) in kinds})


def move_transitions(filled, runs, predicted, target):
    """`filled` codes with each predicted transition p in `target` moved to frame target[p].

    `predicted` is pm.transitions(filled, runs). A move that would cross a neighbouring
    boundary is skipped, so segments keep their order.
    """
    out = filled.copy()
    for start, stop in runs:
        inside = [p for p in predicted if start <= p[0] < stop]
        frames = [p[0] for p in inside]
        for i, p in enumerate(inside):
            if p not in target:
                continue
            low = frames[i - 1] if i else start
            high = frames[i + 1] if i + 1 < len(frames) else stop
            if low < target[p] < high:
                frames[i] = target[p]
        bounds = [start] + frames + [stop]
        codes = [filled[start]] + [p[2] for p in inside]
        for a, b, code in zip(bounds[:-1], bounds[1:], codes):
            out[a:b] = code
    return out


def _latency_cell(values):
    if not len(values):
        return "{:>6} {:>6} {:>18}".format("--", "--", "--")
    v = np.asarray(values, dtype=float)
    return "{:>+6.1f} {:>6.1f} {:>18}".format(
        v.mean(), v.std(), "{:+.0f} / {:+.0f} / {:+.0f}".format(*np.percentile(v, (10, 50, 90))))


def report(name, truths, predictions, seen, tols=(25, 75)):
    """Print the boundary report for one method.

    predictions[s][i] and seen[s][i] are seed s, match i, as the proxy league holds them.
    """
    from tools.proxy_league import f1_from_flags, spell_flags  # here, to avoid an import cycle

    seeds, n_matches = len(predictions), len(truths)
    print("\nboundary report: {}  ({} matches x {} seeds pooled)".format(name, n_matches, seeds))
    for tol in tols:
        records, spurious = [], []
        for s in range(seeds):
            for i in range(n_matches):
                r, sp = transition_records(truths[i], predictions[s][i], seen[s][i], tol)
                records += r
                spurious += sp
        print("  transitions matched within +-{} frames (official rule: destination must "
              "agree)".format(tol))
        print("    {:<10} {:<8} {:>6} {:>6} {:>7} {:>6} {:>6} {:>18}".format(
            "true kind", "ball", "n", "share", "recall", "mean", "sd", "p10 / p50 / p90"))
        for k in KINDS:
            mine = [r for r in records if r[0] == k]
            for label, keep in (("all", None), ("seen", True), ("missing", False)):
                rows = [r for r in mine if keep is None or r[1] == keep]
                if not rows:
                    continue
                lat = [r[2] for r in rows if r[2] is not None]
                print("    {:<10} {:<8} {:>6} {:>6.2f} {:>7.3f} {}".format(
                    k if label == "all" else "", label, len(rows) // seeds,
                    len(rows) / float(len(mine)), len(lat) / float(len(rows)),
                    _latency_cell(lat)))
        counts = {k: spurious.count(k) // seeds for k in KINDS}
        print("    spurious predicted transitions per pass: " +
              ", ".join("{} {}".format(k, counts[k]) for k in KINDS))

    gt_bins = matched_bins = pred_bins = 0
    for s in range(seeds):
        for i in range(n_matches):
            g, m, p = spell_bins(truths[i], predictions[s][i])
            gt_bins, matched_bins, pred_bins = gt_bins + g, matched_bins + m, pred_bins + p
    labels = ["<2 s", "2-5 s", "5-15 s", ">=15 s"]
    print("  possessions by duration (per pass; matched = true ones found at IoU 0.5)")
    print("    {:<8} {:>7} {:>8} {:>7} {:>7}".format("bin", "true", "matched", "recall",
                                                       "pred"))
    for b, label in enumerate(labels):
        print("    {:<8} {:>7} {:>8} {:>7.3f} {:>7}".format(
            label, gt_bins[b] // seeds, matched_bins[b] // seeds,
            matched_bins[b] / float(max(gt_bins[b], 1)), pred_bins[b] // seeds))
    print("    count ratio predicted / true: {:.3f}".format(pred_bins.sum() /
                                                        float(gt_bins.sum())))

    print("  oracle snap within +-{} frames, then rescored (ceiling for E6)".format(SNAP_TOL))
    for label, kinds in (("as decoded", ()), ("snap hand-overs", ("hand-over",)),
                         ("snap hand-overs + restarts", ("hand-over", "restart")),
                         ("snap every transition", KINDS)):
        scores = []
        for s in range(seeds):
            parts, n_pred = [], 0
            for i in range(n_matches):
                snapped = snap_to_truth(truths[i], predictions[s][i], kinds)
                flags, n = spell_flags(truths[i], snapped)
                parts.append(flags)
                n_pred += n
            scores.append(f1_from_flags(np.concatenate(parts, axis=1), n_pred))
        print("    {:<28} spellF1 {:.4f}".format(label, float(np.mean(scores))))
