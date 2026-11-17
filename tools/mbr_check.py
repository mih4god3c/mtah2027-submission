"""#6 Step 0: is decoding for spellF1 itself (minimum Bayes risk) worth building? Proxy only.

    .venv/bin/python tools/mbr_check.py                                # v5_frame_model, keep2, D12
    .venv/bin/python tools/mbr_check.py --labels dfl
    .venv/bin/python tools/mbr_check.py --temperatures 1 2 4 --samples 200 --candidates 50

The decoder outputs its single most likely labelling (Viterbi). spellF1 punishes a wrong split
harder than a missed one, so the labelling with the best EXPECTED spellF1 can differ from the most
likely one. This tool samples labellings from the decoder's own posterior (forward filtering,
backward sampling: decoder.forward_sums / backward_sample, method.sample) and reports, pooled over
the IDSSE dev matches exactly as the proxy league pools them (simulator v1 + the D12 ball stress
unless --plain):

  - viterbi      the method as registered (byte-identical to `make proxy`);
  - one sample   a single posterior draw (how good a typical draw is);
  - marginal     the per-frame majority of the samples (posterior decoding);
  - mbr          the candidate (Viterbi + the first --candidates samples) with the highest mean
                 spellF1 against the other samples, the samples standing in for the truth. This
                 is what Step 1 would build; here it is only a preview;
  - oracle best  the sample with the highest spellF1 against the TRUTH, per match. A ceiling
                 that reads the truth: if even it barely beats Viterbi, MBR cannot;
  - merge s<x    the cheaper version: Viterbi with every possession whose posterior support
                 (share of samples holding a same-team spell over it at IoU >= 0.5) is below x
                 absorbed by a neighbour (merge_spells);
  - merge oracle the same with the TRUTH deciding: every Viterbi possession matching no true one
                 at IoU >= 0.5 is absorbed. The ceiling of merging alone.

Temperature divides every decoder score. It leaves Viterbi unchanged and only widens (> 1) the
posterior the samples come from; the decoder's evidence assumes independent 0.2 s blocks, so it is
probably overconfident at 1. Calibration is reported two ways:
  - per frame: graded frames binned by the share of samples that agree with Viterbi there, and
    Viterbi's accuracy against the truth in each bin;
  - per metric: the decoder's own estimate of its spellF1 (Viterbi against its samples) next to
    its actual spellF1 against the truth.
Aggregates only. Nothing here reads data/.
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution.match import pm  # noqa: E402
from solution.methods import METHODS  # noqa: E402
from tools.deadball_check import STRESS, inputs  # noqa: E402
from tools.proxy_league import bootstrap, f1_from_flags, spell_flags  # noqa: E402

CONF_EDGES = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.0)


# -- spellF1 between two labellings, fast --------------------------------------------------------

def spell_arrays(codes, runs):
    """(starts, stops, codes) int arrays of pm.possession_spells(codes, runs), in frame order."""
    spells = np.array(pm.possession_spells(codes, runs), dtype=np.int64).reshape(-1, 3)
    return spells[:, 0], spells[:, 1], spells[:, 2]


def spell_f1(ref, cand):
    """spellF1 of `cand` scored against `ref` as the truth, both from spell_arrays.

    The same number as pm.evaluate(...)["spell_f1_mean"] (tests/test_mbr.py checks it), computed
    on interval arrays: spells are disjoint and in order, so the ones overlapping a true spell are
    a contiguous range found by searchsorted. One greedy pass over the IoU-descending overlaps
    gives every threshold at once, because greedy decisions on a prefix never depend on what
    follows it.
    """
    gs, ge, gc = ref
    ps, pe, pc = cand
    n_gt, n_pred = len(gs), len(ps)
    lo = np.searchsorted(pe, gs, side="right")
    hi = np.searchsorted(ps, ge, side="left")
    counts = np.maximum(hi - lo, 0)
    gi = np.repeat(np.arange(n_gt), counts)
    offsets = np.arange(int(counts.sum())) - np.repeat(np.cumsum(counts) - counts, counts)
    pj = np.repeat(lo, counts) + offsets
    keep = gc[gi] == pc[pj]
    gi, pj = gi[keep], pj[keep]
    inter = np.minimum(ge[gi], pe[pj]) - np.maximum(gs[gi], ps[pj])
    union = np.maximum(ge[gi], pe[pj]) - np.minimum(gs[gi], ps[pj])
    iou = inter / union.astype(float)
    order = np.lexsort((pj, gi, -iou))
    used_gt = np.zeros(n_gt, dtype=bool)
    used_pred = np.zeros(n_pred, dtype=bool)
    matched = []
    for index in order:
        g, p = gi[index], pj[index]
        if used_gt[g] or used_pred[p]:
            continue
        used_gt[g] = used_pred[p] = True
        matched.append(iou[index])
    matched = np.asarray(matched)
    scores = []
    for threshold in pm.SPELL_IOUS:
        tp = int(np.sum(matched >= threshold))
        fp, fn = max(0, n_pred - tp), n_gt - tp
        scores.append(2.0 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(scores))


# -- one match ------------------------------------------------------------------------------------

def best_overlap(ref, spells):
    """Per spell of `spells`: its highest IoU with a same-team spell of `ref` (0 if none)."""
    gs, ge, gc = ref
    ps, pe, pc = spells
    out = np.zeros(len(ps))
    lo = np.searchsorted(ge, ps, side="right")
    hi = np.searchsorted(gs, pe, side="left")
    for j in range(len(ps)):
        for i in range(lo[j], hi[j]):
            if gc[i] == pc[j]:
                inter = min(ge[i], pe[j]) - max(gs[i], ps[j])
                union = max(ge[i], pe[j]) - min(gs[i], ps[j])
                out[j] = max(out[j], inter / float(union))
    return out


def merge_spells(codes, spells, drop, runs):
    """Copy of `codes` with the possessions `drop` (bool per spell) absorbed by a neighbour.

    A dropped spell takes the label of the frame before it if that is a team (H A H -> H H H,
    H A D -> H H D), else of the frame after it if that is a team, else D. Neighbours are read
    from the original codes, so the result does not depend on the order of the drops.
    """
    out = codes.copy()
    starts, stops, _ = spells
    for j in np.flatnonzero(drop):
        start, stop = int(starts[j]), int(stops[j])
        run = next((a, b) for a, b in runs if a <= start < b)
        before = codes[start - 1] if start - 1 >= run[0] else pm.X
        after = codes[stop] if stop < run[1] else pm.X
        if before in (pm.H, pm.A):
            out[start:stop] = before
        elif after in (pm.H, pm.A):
            out[start:stop] = after
        else:
            out[start:stop] = pm.D
    return out


def to_codes(states, match):
    codes = states.astype(np.int8).copy()
    codes[..., ~match.in_play_mask()] = pm.X
    return codes


def mbr_pick(candidates, references, self_index):
    """Index of the candidate with the highest mean spellF1 against the references.

    self_index[i] is the reference that IS candidate i (or -1); it is left out of that
    candidate's mean, so a sample is not credited for agreeing with itself. Ties go to the
    lowest index (Viterbi first). Returns (index, expected spellF1 of every candidate).
    """
    expected = np.empty(len(candidates))
    for i, cand in enumerate(candidates):
        values = [spell_f1(ref, cand) for r, ref in enumerate(references) if r != self_index[i]]
        expected[i] = np.mean(values)
    return int(np.argmax(expected)), expected


def analyse(truth, match, method, inp, viterbi_states, temperature, n_samples, n_candidates,
            seed, merges=()):
    """Per-variant codes plus calibration counts for one degraded match at one temperature."""
    samples = method.sample(match, inp, n_samples, seed=seed, temperature=temperature)
    vit = to_codes(viterbi_states, match)
    codes = to_codes(samples, match)
    graded = truth != pm.X
    # Per-frame calibration on graded frames.
    agree = np.mean(codes == vit[None, :], axis=0)[graded]
    right = (vit == truth)[graded]
    which = np.clip(np.searchsorted(CONF_EDGES, agree, side="right") - 1, 0,
                    len(CONF_EDGES) - 2)
    which[agree >= 1.0] = len(CONF_EDGES) - 2
    calib = np.zeros((len(CONF_EDGES) - 1, 3))           # frames, right, sum of agreement
    np.add.at(calib, which, np.column_stack((np.ones_like(agree), right, agree)))
    counts = np.stack([(codes == k).sum(axis=0) for k in (pm.H, pm.A, pm.D)])
    marginal = to_codes(np.argmax(counts, axis=0), match)
    # MBR over Viterbi + the first candidates, the samples standing in for the truth.
    runs = [tuple(h) for h in match.halves]
    refs = [spell_arrays(c, runs) for c in codes]
    cand_codes = [vit] + [codes[i] for i in range(min(n_candidates, n_samples))]
    cands = [spell_arrays(vit, runs)] + refs[:len(cand_codes) - 1]
    pick, expected = mbr_pick(cands, refs, [-1] + list(range(len(cand_codes) - 1)))
    truth_runs = pm.valid_runs(truth)
    truth_spells = spell_arrays(truth, truth_runs)
    actual = [spell_f1(truth_spells, spell_arrays(c, truth_runs)) for c in codes]
    out = {"viterbi": vit, "one sample": codes[0], "marginal": marginal,
           "mbr": cand_codes[pick], "oracle best": codes[int(np.argmax(actual))]}
    # The cheaper version: support of each Viterbi possession = share of samples holding a
    # same-team spell over it at IoU >= 0.5; merge the weakly supported ones into a neighbour.
    vit_spells = cands[0]
    support = np.mean([best_overlap(ref, vit_spells) >= 0.5 for ref in refs], axis=0)
    true_iou = best_overlap(spell_arrays(truth, truth_runs), vit_spells)
    for threshold in merges:
        out["merge s<{:g}".format(threshold)] = merge_spells(vit, vit_spells,
                                                             support < threshold, runs)
    out["merge oracle"] = merge_spells(vit, vit_spells, true_iou < 0.5, runs)
    return out, {"calib": calib, "self_estimate": expected[0],
                 "sample_mean": float(np.mean(actual)), "mbr_is_viterbi": pick == 0,
                 "sample_poss": float(np.mean([len(r[0]) for r in refs])),
                 "support": support, "true_iou": true_iou,
                 "length_s": (vit_spells[1] - vit_spells[0]) / float(match.fps)}


# -- pooling and report ---------------------------------------------------------------------------

def pooled(results, variant, seeds, n_matches):
    """(mean pooled spellF1 over seeds, per-match means, seed-0 (flags, n_pred), mean n_pred)."""
    by_seed, per_match, first, n_preds = [], np.zeros((len(seeds), n_matches)), None, []
    for s, seed in enumerate(seeds):
        parts, n_pred = [], 0
        for m in range(n_matches):
            truth, codes = results[(seed, m)]["truth"], results[(seed, m)]["codes"][variant]
            flags, n = spell_flags(truth, codes)
            parts.append(flags)
            n_pred += n
            per_match[s, m] = f1_from_flags(flags, n)
        flags = np.concatenate(parts, axis=1)
        by_seed.append(f1_from_flags(flags, n_pred))
        n_preds.append(n_pred)
        if s == 0:
            first = (flags, n_pred)
    return float(np.mean(by_seed)), per_match.mean(axis=0), first, float(np.mean(n_preds))


def report(results, seeds, n_matches, temperature, draws):
    print("\n== temperature {} ==".format(temperature))
    variants = list(next(iter(results.values()))["codes"])
    rows = {v: pooled(results, v, seeds, n_matches) for v in variants}
    base_value, base_match, base_first, _ = rows["viterbi"]
    n_gt = base_first[0].shape[1]
    print("{:<12} {:>8} {:>7} {:>9} {:>19} {:>6}  {}".format(
        "variant", "spellF1", "poss", "gain", "paired interval", "wins", "verdict"))
    for variant in variants:
        value, per_match, first, n_pred = rows[variant]
        if variant == "viterbi":
            print("{:<12} {:>8.4f} {:>7.0f}   (truth {} possessions over {} seeds)".format(
                variant, value, n_pred, n_gt, len(seeds)))
            continue
        low, high = bootstrap(base_first[0], base_first[1], draws, other=first)
        wins = int(np.sum(per_match > base_match))
        gain = value - base_value
        if variant in ("oracle best", "merge oracle"):
            verdict = "ceiling (reads the truth)"
        elif high < 0:
            verdict = "WORSE"
        elif gain >= 0.02 and low > 0 and wins * 3 >= 2 * n_matches:
            verdict = "would clear D4 (preview, not a decision)"
        elif low > 0:
            verdict = "better, below the bar"
        else:
            verdict = "within noise"
        print("{:<12} {:>8.4f} {:>7.0f} {:>+9.4f}  {:>+8.4f}..{:<+8.4f} {:>3}/{}  {}".format(
            variant, value, n_pred, gain, low, high, wins, n_matches, verdict))
    stats = [r["stats"] for r in results.values()]
    print("decoder's own spellF1 estimate for Viterbi {:.4f} (actual {:.4f}); a typical sample "
          "scores {:.4f} against the truth".format(
              np.mean([s["self_estimate"] for s in stats]), base_value,
              np.mean([s["sample_mean"] for s in stats])))
    print("MBR kept Viterbi on {}/{} match-seeds; possessions per sample {:.0f} (Viterbi {:.0f})"
          .format(sum(s["mbr_is_viterbi"] for s in stats), len(stats),
                  np.mean([s["sample_poss"] for s in stats]), rows["viterbi"][3] / n_matches))
    calib = sum(s["calib"] for s in stats)
    total = calib[:, 0].sum()
    print("per-frame calibration (graded frames; agreement = share of samples equal to Viterbi)")
    print("  {:<12} {:>8} {:>10} {:>14}".format("agreement", "frames", "mean agr", "Viterbi right"))
    for b in range(len(CONF_EDGES) - 1):
        n = calib[b, 0]
        label = ("[{:.2f}, {:.2f})".format(CONF_EDGES[b], CONF_EDGES[b + 1])
                 if b < len(CONF_EDGES) - 2 else "[0.99, 1.00]")
        if n:
            print("  {:<12} {:>8.3f} {:>10.3f} {:>14.3f}".format(
                label, n / total, calib[b, 2] / n, calib[b, 1] / n))
    ece = np.sum(np.abs(calib[:, 2] - calib[:, 1])) / total
    print("  expected calibration error {:.3f}; Viterbi right on {:.3f} of graded frames".format(
        ece, calib[:, 1].sum() / total))
    support = np.concatenate([s["support"] for s in stats])
    true_iou = np.concatenate([s["true_iou"] for s in stats])
    length = np.concatenate([s["length_s"] for s in stats])
    print("Viterbi possessions by posterior support (share of samples with a same-team spell at "
          "IoU >= 0.5); 'true' = a same-team true spell at IoU >= 0.5")
    print("  {:<12} {:>7} {:>7} {:>12}".format("support", "share", "true", "median s"))
    for lo, hi in ((0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)):
        pick = (support >= lo) & (support < hi)
        if pick.any():
            print("  {:<12} {:>7.3f} {:>7.3f} {:>12.1f}".format(
                "[{:.1f}, {:.1f})".format(lo, min(hi, 1.0)), pick.mean(),
                np.mean(true_iou[pick] >= 0.5), np.median(length[pick])))


def main():
    from solution.sources import idsse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", default="v5_frame_model", choices=sorted(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--plain", action="store_true", help="no ball stress (plain v1)")
    parser.add_argument("--labels", default="keep2",
                        help="truth label variant (keep2, the decision truth, D21; or dfl)")
    parser.add_argument("--prior-labels", default="dfl",
                        help="labels the priors are fitted on (the proxy league's default)")
    parser.add_argument("--temperatures", nargs="+", type=float, default=[1.0, 2.0, 4.0])
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--candidates", type=int, default=50)
    parser.add_argument("--merge", nargs="+", type=float, default=[0.3, 0.5, 0.7],
                        help="support thresholds for the merge variants")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    idsse.LABELS = args.prior_labels
    method = METHODS[args.method]
    if not hasattr(method, "sample"):
        raise SystemExit("{} is not a semi_markov method".format(args.method))
    started = time.time()
    print("MBR check: {} on IDSSE dev x seeds {}, simulator v1{}, {} truth, priors on {}; {} "
          "samples, {} MBR candidates".format(
              args.method, args.seeds, "" if args.plain else " + D12 ball stress {}".format(
                  STRESS), args.labels, args.prior_labels, args.samples, args.candidates))
    results = {t: {} for t in args.temperatures}
    match_index = {}
    for truth, match, _, seed in inputs(idsse.DEV_IDS, args.seeds, args.plain, args.labels):
        m = match_index.setdefault(match.name, len(match_index))
        inp = method.prepare(match)
        states = method.decode(match, inp)
        for t in args.temperatures:
            codes, stats = analyse(truth, match, method, inp, states, t, args.samples,
                                   args.candidates, seed, args.merge)
            results[t][(seed, m)] = {"truth": truth, "codes": codes, "stats": stats}
        print("  {} seed {} done ({:.0f} s)".format(match.name, seed, time.time() - started))
        sys.stdout.flush()
    for t in args.temperatures:
        report(results[t], args.seeds, len(match_index), t, args.bootstrap)
    print("\n({:.0f} s)".format(time.time() - started))


if __name__ == "__main__":
    main()
