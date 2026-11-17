"""What do ball gaps cost? Step 0 of E5 (PLAN.md E5). Aggregates only.

    .venv/bin/python tools/gap_check.py                       # v3_kickoff, D12 ball stress
    .venv/bin/python tools/gap_check.py --plain               # simulator v1, clean ball
    .venv/bin/python tools/gap_check.py --methods v3_kickoff <new>
    .venv/bin/python tools/gap_check.py --cues                # E5 team cues on the proxy
    .venv/bin/python tools/gap_check.py --cues --bar-ath      # (needs data/) label-free, for a
                                                              # human to run
    .venv/bin/python tools/gap_check.py --methods v4_spread --timing [--capacity]
                                                    # E5 part 2 Step 0: hand-over timing
    .venv/bin/python tools/gap_check.py --decoded             # Step 2 on the stressed proxy
    .venv/bin/python tools/gap_check.py --decoded --bar-ath   # (needs data/) Step 2 on BAR-ATH
    .venv/bin/python tools/gap_check.py --events [--noise 0.1] [--speed 0.9] [--rates]
                                                    # Step A: hand-over moment features
    .venv/bin/python tools/gap_check.py --events --bar-ath    # (needs data/) their event rates
    .venv/bin/python tools/gap_check.py --methods v4_spread --seen [--feature dadvance0.5]
                                                    # Step B: short possessions, ball seen

--events (Step A of the C2c-or-touch choice, after D17) generalises --timing's part (c) to any
  per-frame feature of solution/handover.py. No decoding. Per feature: the share of true
  open-play hand-overs inside gaps with an event (towards the new team, if signed) within +-1
  and +-3 s, the lag of the nearest one, the false-alarm rate (events per minute of alive
  in-gap play farther than FA_CLEAR_S from any true transition), the hit share events at that
  rate would get by chance, the lift (hit / chance), and the precision (events on alive gap
  frames within +-1 s of a hand-over to their team). --bar-ath gives the label-free rates.
--seen (Step B) takes the true possessions under 5 s lying mostly where the ball is seen and,
  by length: the decoder's recall and team accuracy; the team, snap and reach oracles of
  short_oracles (reach: only the possessions whose start the touch engine sees, against a
  random subset of the same size); what the engine sees at their starts; how often its
  hand-over claims are right. Then the same oracles with a handover feature's events as the
  evidence (--feature), on the same possessions, with touches added, and on short
  possessions lying mostly in a gap.

--timing (E5 part 2, Step 0) asks whether what is left of the in-gap ceiling is timing:
  (a) the timing oracle moves each decoded H <-> A change inside a gap onto the true open-play
      hand-over it matches (same destination team, one-to-one; within +-1.5 s, +-3 s or
      anywhere in the same gap), teams otherwise as decoded, against the team oracle above.
      The team oracle is also split: gap frames within GAP_PAD_S of a true in-gap hand-over
      against the rest, and the frames of short true possessions lying mostly in a gap.
      --capacity decodes v4_spread with its spread cue replaced, on alive gap frames, by
      the true team's sign at a realistic size (CAPACITY): what the decoder makes of ideal
      evidence, which separates the decoder's limits from the evidence's;
  (b) per true open-play hand-over inside a gap: a decoded change to the new team within +-1
      and +-3 s (official matching) and its latency, the nearest one within the gap, or none;
  (c) when the spread cue, signed towards the new team, turns positive relative to the
      hand-over (lead or lag), or never within FLIP_WINDOW_S;
  split by gap length and by the length of the possession the hand-over starts; then the
  possession duration priors: minimum durations, log P(d), the prior cost of a short
  possession inside a long one.
--decoded (Step 2, no labels): per method, the home share of decoded team frames and the
  dead-ball share inside gaps, possessions per 90 (all, mostly in a gap, those under 2 s) and
  H <-> A changes inside gaps per 90; on the proxy also the truth's numbers. With --bar-ath,
  agreement with the team of the Smart Tagging attacking phases (each defending code mirrors
  one with the team swapped, FINDINGS point 5; the mirror check is printed).

On the IDSSE dev matches, degraded as `make proxy` does (simulator v1, then the D12 ball stress
unless --plain; tools/deadball_check.inputs), a ball gap is a maximal run of frames inside a
half where the degraded ball is not seen. Per method it reports:
  - gaps per seed, their share of graded frames, and the method's team accuracy on the gap
    frames where truth and prediction are both alive;
  - oracle ceilings, pooled spellF1 exactly as the proxy league pools it: inside the chosen gaps,
    every frame where truth and the prediction are both alive (H or A) takes the true team.
    Dead ball stays as predicted, so the repair isolates the team question from dead ball (C2b,
    tools/deadball_check.py). One row repairs every gap, the others one gap-length bin
    (GAP_BINS_S) or one kind of gap:
      "change": the true team in possession changes inside the gap. The team is carried
        through dead ball, so a gap holding H, D, A changes and one holding H, D, H does not;
        a gap that opens a half changes if it holds both teams;
      "no change": every other gap. Carrying the last owner through it is right;
      "change restart": every change in the gap comes at a restart (D to the other team);
      "change in play": the gap holds at least one direct H <-> A hand-over;
  - true possessions by length (as tools/boundaries.py: matched = found at IoU 0.5), split by
    whether most of the possession lies inside a gap, before and after the full repair. This
    shows how short possessions fare and how much of their loss is the team inside gaps.

With --cues (Step 1), for each team cue of solution/gapteam.py: on the frames without a seen
ball, the share where the cue is known, its size (|c| p50 and p90) and how often it is beyond
the middle bin of CUE_BINS ("fires"), and its mean (positive: towards home); on the proxy also, among true-alive gap frames where it
fires, how often its sign points at the true team, and the mean log-likelihood ratio of the
true team against the other under the fitted tables. With --bar-ath the label-free part on
BAR-ATH, for the firing rates to be compared. Nothing frame-level is printed.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution.match import pm  # noqa: E402
from solution.methods import METHODS  # noqa: E402
from tools.deadball_check import STRESS, inputs  # noqa: E402
from tools.proxy_league import f1_from_flags, spell_flags  # noqa: E402

GAP_BINS_S = (0.0, 2.0, 10.0, 30.0, np.inf)
SPELL_BINS_S = (0.0, 2.0, 5.0, 15.0, np.inf)
# The truth's label variant (solution/relabel.py), set by --labels. keep2, controlled
# possession, as `make proxy` decides on (D21); dfl reproduces sections 9 to 11.
LABELS = "keep2"


def bin_label(lo, hi):
    return ">{:g} s".format(lo) if hi == np.inf else "<{:g} s".format(hi) if lo == 0 \
        else "{:g}-{:g} s".format(lo, hi)


def gap_spans(match):
    """[(start, stop)] maximal runs of frames without a seen ball, inside each half."""
    out = []
    for start, stop in match.halves:
        missing = ~match.ball_seen[start:stop]
        edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
        out += [(start + a, start + b) for a, b in zip(np.flatnonzero(edges == 1),
                                                        np.flatnonzero(edges == -1))]
    return out


def carried_team(truth, runs):
    """Per frame, the true team in possession carried forward through dead ball; -1 before
    the first possession of a half."""
    out = np.full(len(truth), -1, dtype=np.int8)
    for start, stop in runs:
        window = truth[start:stop]
        alive = np.flatnonzero(np.isin(window, pm.POSSESSION_LABELS))
        last = np.searchsorted(alive, np.arange(stop - start), side="right") - 1
        out[start:stop] = np.where(last >= 0, window[alive[np.maximum(last, 0)]], -1)
    return out


def change_kind(truth, carried, start, stop, half_start):
    """How the carried true team changes inside [start, stop): "no change", "restart" (every
    change comes at a restart, D to the other team) or "in play" (some H <-> A hand-over)."""
    lo = max(start, half_start + 1)
    f = np.arange(lo, stop)
    changes = f[(carried[f] != carried[f - 1]) & (carried[f - 1] >= 0)]
    if start == half_start and not len(changes):
        window = carried[start:stop]
        if len(set(window[window >= 0].tolist())) < 2:
            return "no change"
    elif not len(changes):
        return "no change"
    return "restart" if np.all(truth[changes - 1] == pm.D) else "in play"


def team_repair(truth, pred, runs, mask):
    """pred with the true team on the `mask` frames where truth and pred are both alive."""
    filled = pm.fill_abstentions(pred, runs)
    out = filled.copy()
    fix = mask & np.isin(truth, pm.POSSESSION_LABELS) & np.isin(filled, pm.POSSESSION_LABELS)
    out[fix] = truth[fix]
    return out


def spell_records(truth, pred, gap_mask):
    """[(seconds, mostly in a gap, matched at IoU 0.5)] per true possession."""
    runs = pm.valid_runs(truth)
    gt = pm.possession_spells(truth, runs)
    predicted = pm.possession_spells(pm.fill_abstentions(pred, runs), runs)
    overlaps = pm.spell_overlaps(gt, predicted)
    matched = pm.matched_gt_flags(overlaps, len(gt), pm.SPELL_IOU_HEADLINE)
    return [((b - a) / float(pm.FPS), bool(gap_mask[a:b].mean() > 0.5), bool(m))
            for (a, b, _), m in zip(gt, matched)]


def analyse(truth, pred, match):
    """Per-gap records and {repair name: repaired codes} for one method on one match."""
    runs = pm.valid_runs(truth)
    filled = pm.fill_abstentions(pred, runs)
    carried = carried_team(truth, runs)
    run_starts = np.array([start for start, _ in runs])
    both = np.isin(truth, pm.POSSESSION_LABELS) & np.isin(filled, pm.POSSESSION_LABELS)
    gaps, masks = [], {}
    n = len(truth)
    for a, b in gap_spans(match):
        seconds = (b - a) / float(match.fps)
        length = next(bin_label(lo, hi) for lo, hi in zip(GAP_BINS_S[:-1], GAP_BINS_S[1:])
                      if lo <= seconds < hi)
        half_start = run_starts[np.searchsorted(run_starts, a, side="right") - 1]
        how = change_kind(truth, carried, a, b, half_start)
        kind = "no change" if how == "no change" else "change"
        gaps.append({"how": how, "seconds": seconds, "bin": length, "kind": kind, "frames": b - a,
                     "both": int(both[a:b].sum()),
                     "true_alive": int(np.isin(truth[a:b], pm.POSSESSION_LABELS).sum()),
                     "pred_alive": int(np.isin(filled[a:b], pm.POSSESSION_LABELS).sum()),
                     "right": int(np.sum(both[a:b] & (filled[a:b] == truth[a:b])))})
        for key in ("all gaps", "length " + length, kind, "change " + how):
            masks.setdefault(key, np.zeros(n, dtype=bool))[a:b] = True
    repairs = {key: team_repair(truth, pred, runs, mask) for key, mask in masks.items()}
    gap_mask = masks.get("all gaps", np.zeros(n, dtype=bool))
    spells = {"as is": spell_records(truth, pred, gap_mask),
              "repaired": spell_records(truth, repairs["all gaps"], gap_mask)}
    graded = int(sum(stop - start for start, stop in runs))
    return gaps, repairs, spells, graded


def pooled(rows, key):
    """Pooled spellF1 of the codes under `key` ('as is' = the prediction), mean over seeds."""
    per_seed = []
    for seed in sorted(set(row["seed"] for row in rows)):
        parts, n_pred = [], 0
        for row in rows:
            if row["seed"] != seed:
                continue
            codes = row["pred"] if key == "as is" else row["repairs"].get(key, row["pred"])
            flags, n = spell_flags(row["truth"], codes)
            parts.append(flags)
            n_pred += n
        per_seed.append(f1_from_flags(np.concatenate(parts, axis=1), n_pred))
    return float(np.mean(per_seed))


def report(name, rows, n_seeds):
    gaps = [g for row in rows for g in row["gaps"]]
    graded = sum(row["graded"] for row in rows)
    print("\n== {} ==  ({:.0f} gaps per seed, {:.3f} of graded frames)".format(
        name, len(gaps) / float(n_seeds), sum(g["frames"] for g in gaps) / float(graded)))
    print("{:<16} {:>8} {:>9} {:>9} {:>11} {:>11} {:>11} {:>10} {:>10}".format(
        "gaps", "n/seed", "frames", "median s", "true alive", "pred alive", "both alive",
        "team acc", "ceiling"))
    scores = {"as is": pooled(rows, "as is")}
    groups = [("all gaps", gaps)]
    groups += [("length " + bin_label(lo, hi), [g for g in gaps if g["bin"] == bin_label(lo, hi)])
               for lo, hi in zip(GAP_BINS_S[:-1], GAP_BINS_S[1:])]
    groups += [(kind, [g for g in gaps if g["kind"] == kind]) for kind in ("change", "no change")]
    groups += [("change " + how, [g for g in gaps if g["how"] == how])
               for how in ("restart", "in play")]
    for key, group in groups:
        if not group:
            continue
        scores[key] = pooled(rows, key)
        frames = sum(g["frames"] for g in group)
        both = sum(g["both"] for g in group)
        print("{:<16} {:>8.1f} {:>9.3f} {:>9.1f} {:>11.3f} {:>11.3f} {:>11.3f} {:>10.3f} "
              "{:>+10.4f}".format(
            key.replace("length ", ""), len(group) / float(n_seeds), frames / float(graded),
            np.median([g["seconds"] for g in group]),
            sum(g["true_alive"] for g in group) / float(max(frames, 1)),
            sum(g["pred_alive"] for g in group) / float(max(frames, 1)),
            both / float(max(frames, 1)),
            sum(g["right"] for g in group) / float(max(both, 1)),
            scores[key] - scores["as is"]))
    print("spellF1 as is {:.4f}; every gap repaired {:.4f}".format(scores["as is"],
                                                                 scores["all gaps"]))
    print("by length and kind, gaps per seed (team acc):  " + "  ".join(
        "{} {}: {:.0f} ({:.3f})".format(
            bin_label(lo, hi), kind, len(group) / float(n_seeds),
            sum(g["right"] for g in group) / float(max(sum(g["both"] for g in group), 1)))
        for kind in ("change", "no change") for lo, hi in zip(GAP_BINS_S[:-1], GAP_BINS_S[1:])
        for group in [[g for g in gaps if g["kind"] == kind and g["bin"] == bin_label(lo, hi)]]))

    print("true possessions by length, recall at IoU 0.5 (as is -> every gap repaired)")
    print("{:<10} {:>18} {:>18} {:>18}".format("length", "all", "mostly in a gap",
                                               "mostly ball seen"))
    for lo, hi in zip(SPELL_BINS_S[:-1], SPELL_BINS_S[1:]):
        cells = []
        for where in (None, True, False):
            counts = []
            for key in ("as is", "repaired"):
                found = total = 0
                for row in rows:
                    for seconds, in_gap, matched in row["spells"][key]:
                        if lo <= seconds < hi and (where is None or in_gap == where):
                            total += 1
                            found += matched
                counts.append((found, total))
            cells.append("{:.3f}->{:.3f} ({:.0f})".format(
                counts[0][0] / float(max(counts[0][1], 1)),
                counts[1][0] / float(max(counts[1][1], 1)), counts[0][1] / float(n_seeds)))
        print("{:<10} {:>18} {:>18} {:>18}".format(bin_label(lo, hi), *cells))
    return scores


# -- Step 0 of E5 part 2: hand-over timing inside gaps ---------------------------------------

TIMING_TOLS_S = (1.0, 3.0)        # a decoded change counts as found within these
SNAP_TOLS = (("+-1.5 s", 1.5), ("+-3 s", 3.0), ("same gap", None))
FLIP_WINDOW_S = 10.0              # the spread cue's flip is looked for this far either side
FLIP_SMOOTH_S = (1.0, 4.0)        # centred mean windows of the spread cue
FLIP_PROBE_S = 2.0                # "before" and "after" read this far from the hand-over
GAP_PAD_S = 3.0                   # a decoded change to the new team counts within the gap +- this
NEW_BINS_S = (0.0, 2.0, 5.0, np.inf)   # length of the possession a hand-over starts
CAPACITY = ((12.0, 0.5), (25.0, 0.5), (12.0, 1.0))   # (|cue| m, team_w) of the capacity oracle


def gap_ids(match):
    """(per-frame gap index or -1, [(start, stop)] gaps, [length bin] per gap)."""
    spans = gap_spans(match)
    out = np.full(match.n_frames, -1, dtype=np.int64)
    bins = []
    for i, (a, b) in enumerate(spans):
        out[a:b] = i
        seconds = (b - a) / float(match.fps)
        bins.append(next(bin_label(lo, hi) for lo, hi in zip(GAP_BINS_S[:-1], GAP_BINS_S[1:])
                         if lo <= seconds < hi))
    return out, spans, bins


def is_handover(transition):
    return transition[1] in pm.POSSESSION_LABELS and transition[2] in pm.POSSESSION_LABELS


def timing_snap(truth, pred, gap_id, gap_bins, tol_s, fps, only_bin=None):
    """pred with each decoded H <-> A change inside a gap moved onto the true open-play
    hand-over it matches (destination team agreeing, one-to-one), true one inside a gap too.

    tol_s None matches within the same gap at any distance. Teams stay as decoded: only the
    frame of a change moves (tools/boundaries.move_transitions). `only_bin` restricts it to
    gaps of one length bin.
    """
    from tools.boundaries import move_transitions
    runs = pm.valid_runs(truth)
    filled = pm.fill_abstentions(pred, runs)
    predicted = pm.transitions(filled, runs)

    def chosen(t):
        g = gap_id[t[0]]
        return is_handover(t) and g >= 0 and (only_bin is None or gap_bins[g] == only_bin)
    gts = [t for t in pm.transitions(truth, runs) if chosen(t)]
    preds = [t for t in predicted if chosen(t)]
    if tol_s is None:
        pairs = []
        for g in sorted(set(gap_id[t[0]] for t in gts)):
            mine = [t for t in preds if gap_id[t[0]] == g]
            if mine:
                pairs += pm.match_transitions([t for t in gts if gap_id[t[0]] == g], mine,
                                              len(truth))[0]
    else:
        pairs = pm.match_transitions(gts, preds, int(round(tol_s * fps)))[0]
    return move_transitions(filled, runs, predicted, {p: g[0] for g, p in pairs})


def first_crossing(value, f, lo, hi):
    """Signed frames from f to the nearest frame in [lo, hi) where `value` turns from <= 0 to
    > 0 (NaN frames skipped); None if it never does."""
    window = np.arange(lo, hi)
    keep = np.isfinite(value[window])
    frames, v = window[keep], value[window][keep]
    up = frames[1:][(v[:-1] <= 0) & (v[1:] > 0)]
    if not len(up):
        return None
    return int(up[np.argmin(np.abs(up - f))] - f)


def timing_records(truth, pred, match, spread_smooth):
    """Per true open-play hand-over inside a gap: how the decoder and the spread cue time it."""
    runs = pm.valid_runs(truth)
    filled = pm.fill_abstentions(pred, runs)
    gt = pm.transitions(truth, runs)
    predicted = pm.transitions(filled, runs)
    gap_id, spans, gap_bins = gap_ids(match)
    fps = match.fps
    latency = {}
    for tol_s in TIMING_TOLS_S:
        pairs = pm.match_transitions(gt, predicted, int(round(tol_s * fps)))[0]
        latency[tol_s] = {g: p[0] - g[0] for g, p in pairs}
    pad, window, probe = (int(round(s * fps)) for s in (GAP_PAD_S, FLIP_WINDOW_S, FLIP_PROBE_S))
    run_of = [(s, e) for s, e in runs]
    true_frames = np.array([t[0] for t in gt], dtype=np.int64)
    out = []
    for g in gt:
        if not is_handover(g) or gap_id[g[0]] < 0:
            continue
        f, new = g[0], g[2]
        a, b = spans[gap_id[f]]
        start, stop = next((s, e) for s, e in run_of if s <= f < e)
        after = true_frames[(true_frames > f) & (true_frames < stop)]
        new_s = ((after[0] if len(after) else stop) - f) / float(fps)
        to_new = [p[0] - f for p in predicted
                  if p[2] == new and a - pad <= p[0] < b + pad and start <= p[0] < stop]
        rec = {"bin": gap_bins[gap_id[f]], "seconds": (b - a) / float(fps),
               "new": next(bin_label(lo, hi) for lo, hi in zip(NEW_BINS_S[:-1], NEW_BINS_S[1:])
                           if lo <= new_s < hi),
               "lat": {t: latency[t].get(g) for t in TIMING_TOLS_S},
               "gap_lat": min(to_new, key=abs) if to_new else None, "flip": {}, "probe": {}}
        sign = 1.0 if new == pm.H else -1.0
        lo, hi = max(start, f - window), min(stop, f + window + 1)
        for s, cue in spread_smooth.items():
            v = cue * sign
            rec["flip"][s] = first_crossing(v, f, lo, hi)
            before = v[f - probe] if f - probe >= start else np.nan
            after = v[f + probe] if f + probe < stop else np.nan
            rec["probe"][s] = (before, after)
        out.append(rec)
    return out


def timing_analyse(truth, pred, match):
    """(hand-over records, {(snap tol, bin): codes}) for one method on one match."""
    from solution import gapteam
    from solution.kickoff import _moving_mean
    gap_id, _, gap_bins = gap_ids(match)
    raw = gapteam.spread(match)
    smooth = {s: _moving_mean(raw, max(1, int(round(s * match.fps)))) for s in FLIP_SMOOTH_S}
    records = timing_records(truth, pred, match, smooth)
    snaps = {}
    for label, tol_s in SNAP_TOLS:
        for only in [None] + [bin_label(lo, hi) for lo, hi in zip(GAP_BINS_S[:-1],
                                                                   GAP_BINS_S[1:])]:
            snaps[(label, only)] = timing_snap(truth, pred, gap_id, gap_bins, tol_s, match.fps,
                                               only)
    # Where the team ceiling sits: the true team on gap frames near a true in-gap hand-over,
    # on the frames of short true possessions lying mostly in a gap, and on everything else.
    runs = pm.valid_runs(truth)
    gap = gap_id >= 0
    near = np.zeros(len(truth), dtype=bool)
    pad = int(round(GAP_PAD_S * match.fps))
    for f, before, after in pm.transitions(truth, runs):
        if gap[f] and is_handover((f, before, after)):
            near[max(0, f - pad):f + pad] = True
    short = {s: np.zeros(len(truth), dtype=bool) for s in (2.0, 5.0)}
    for a, b, _ in pm.possession_spells(truth, runs):
        if gap[a:b].mean() > 0.5:
            for s, mask in short.items():
                if (b - a) < s * match.fps:
                    mask[a:b] = True
    parts = {"near hand-over": gap & near, "far from one": gap & ~near}
    parts.update({"in-gap poss <{:g} s".format(s): gap & m for s, m in short.items()})
    for key, mask in parts.items():
        snaps[("team", key)] = team_repair(truth, pred, runs, mask)
    return records, snaps


def ideal_sign(truth, match, lag_s=0.0, flip=None, seed=0):
    """Per frame +1 (home) / -1 (away) / NaN (not alive) of the true team, as a per-frame
    model at its best could give it. `lag_s` > 0 makes it trail the truth by that much inside
    each half (the evidence a posture model sees reacts late). `flip` = (chunk s, p): the sign
    is reversed on whole chunks of that length with probability p, from a stream seeded by
    `seed` (a model's errors come in runs, not single frames)."""
    sign = np.where(truth == pm.H, 1.0, np.where(truth == pm.A, -1.0, np.nan))
    lag = int(round(lag_s * match.fps))
    if lag > 0:
        lagged = np.full(len(sign), np.nan)
        for start, stop in match.halves:
            lagged[start + lag:stop] = sign[start:stop - lag]
        sign = np.where(np.isfinite(sign), lagged, np.nan)
    if flip is not None:
        chunk = max(1, int(round(flip[0] * match.fps)))
        rng = np.random.default_rng(seed)
        draws = rng.random(-(-len(sign) // chunk)) < flip[1]
        sign = np.where(np.repeat(draws, chunk)[:len(sign)], -sign, sign)
    return sign


def capacity_oracle(match, truth, magnitude, team_w, mask=None, sign=None):
    """v4_spread with its spread cue replaced, on gap frames where truth is alive, by the
    true team's sign times `magnitude` metres: perfect timing at a realistic strength. What
    the decoder (priors, transition costs, team_w) makes of ideal evidence. `mask` limits the
    replacement to those frames; `sign` (ideal_sign) replaces the perfect sign."""
    from solution import gapteam, methods
    settings = {"cues": ("spread",)}
    gapteam.idsse_log(match.name, tuple(sorted(settings.items())))   # fit before patching
    real = gapteam.cues
    if sign is None:
        sign = ideal_sign(truth, match)
    ideal = ~match.ball_seen & np.isfinite(sign)
    if mask is not None:
        ideal &= mask

    def patched(m, params=gapteam.PARAMS):
        out = real(m, params)
        if m is match:
            out[ideal, 0] = sign[ideal] * magnitude
        return out
    gapteam.cues = patched
    try:
        return methods.semi_markov("capacity", touch_w=1.0, kickoff_params={}, team_w=team_w,
                                   team_where="all", team_dead="max",
                                   team_params=settings)(match)
    finally:
        gapteam.cues = real


def pooled_codes(rows, codes_of):
    """Pooled spellF1 of codes_of(row), mean over seeds."""
    per_seed = []
    for seed in sorted(set(row["seed"] for row in rows)):
        parts, n_pred = [], 0
        for row in rows:
            if row["seed"] == seed:
                flags, n = spell_flags(row["truth"], codes_of(row))
                parts.append(flags)
                n_pred += n
        per_seed.append(f1_from_flags(np.concatenate(parts, axis=1), n_pred))
    return float(np.mean(per_seed))


def _pcts(values, fps):
    if not len(values):
        return "--"
    p = np.percentile(np.asarray(values, dtype=float) / fps, [10, 50, 90])
    return "{:+.1f}/{:+.1f}/{:+.1f}".format(*p)


def timing_report(name, rows, n_seeds, fps):
    """Step 0 (a), (b), (c) for one method; rows carry "records", "snaps" and "repairs"."""
    bins = [bin_label(lo, hi) for lo, hi in zip(GAP_BINS_S[:-1], GAP_BINS_S[1:])]
    base = pooled_codes(rows, lambda r: r["pred"])
    print("\n== {}: hand-over timing inside gaps ==  (spellF1 as is {:.4f})".format(name, base))
    print("(a) oracle gains: timing = decoded H<->A changes in gaps moved onto the matching true "
          "hand-over, teams as decoded; team = the true team on both-alive gap frames")
    print("{:<12}".format("gaps") + "".join("{:>16}".format(c) for c in (
        ["timing " + label for label, _ in SNAP_TOLS] + ["team", "team in play"])))
    for only in [None] + bins:
        cells = [pooled_codes(rows, lambda r, k=(label, only): r["snaps"][k]) - base
                 for label, _ in SNAP_TOLS]
        if only is None:
            team = [pooled(rows, "all gaps"), pooled(rows, "change in play")]
        else:
            team = [pooled(rows, "length " + only), float("nan")]
        cells += [t - base for t in team]
        print("{:<12}".format("all" if only is None else only) +
              "".join("{:>+16.4f}".format(c) for c in cells))
    print("team oracle on part of the gap frames (gain): " + "  ".join(
        "{} {:+.4f}".format(key, pooled_codes(rows, lambda r, k=key: r["snaps"][("team", k)])
                            - base)
        for key in ("near hand-over", "far from one", "in-gap poss <2 s", "in-gap poss <5 s")) +
        "  (near: within +-{:g} s of a true in-gap hand-over)".format(GAP_PAD_S))
    if "capacity" in rows[0]:
        print("capacity oracle, the decoder fed a perfect-sign spread cue on alive gap frames "
              "(gain; recall of true possessions mostly in a gap <2 s, 2-5 s; predicted / true "
              "possessions):")
        for key in CAPACITY:
            gain = pooled_codes(rows, lambda r, k=key: r["capacity"][k]) - base
            recall, count = [], [0, 0]
            for lo, hi in ((0.0, 2.0), (2.0, 5.0)):
                hit = total = 0
                for row in rows:
                    for seconds, in_gap, matched in spell_records(
                            row["truth"], row["capacity"][key], row["gap_mask"]):
                        if in_gap and lo <= seconds < hi:
                            total += 1
                            hit += matched
                recall.append(hit / float(max(total, 1)))
            for row in rows:
                runs = pm.valid_runs(row["truth"])
                count[0] += len(pm.possession_spells(
                    pm.fill_abstentions(row["capacity"][key], runs), runs))
                count[1] += len(pm.possession_spells(row["truth"], runs))
            print("  |cue| {:g} m, team_w {:g}: {:+.4f}; recall {:.3f}, {:.3f}; ratio {:.3f}".format(
                key[0], key[1], gain, recall[0], recall[1], count[0] / float(count[1])))

    records = [r for row in rows for r in row["records"]]
    groups = [("all", lambda r: True)] + [
        (b, lambda r, b=b: r["bin"] == b) for b in bins] + [
        ("new " + b, lambda r, b=b: r["new"] == b)
        for b in [bin_label(lo, hi) for lo, hi in zip(NEW_BINS_S[:-1], NEW_BINS_S[1:])]]
    print("(b) true open-play hand-overs inside gaps: decoded change to the new team found "
          "within +-1 s and +-3 s (official one-to-one matching), latency s p10/p50/p90; "
          "nearest decoded change to the new team within the gap +-{:g} s, and none".format(
              GAP_PAD_S))
    print("{:<14} {:>7} {:>8} {:>8} {:>16} {:>9} {:>16} {:>9}".format(
        "gap", "n/seed", "+-1 s", "+-3 s", "lat +-3 s", "in gap", "lat in gap", "none"))
    for only, keep in groups:
        group = [r for r in records if keep(r)]
        if not group:
            continue
        n = float(len(group))
        lat3 = [r["lat"][3.0] for r in group if r["lat"][3.0] is not None]
        lat_gap = [r["gap_lat"] for r in group if r["gap_lat"] is not None]
        print("{:<14} {:>7.1f} {:>8.3f} {:>8.3f} {:>16} {:>9.3f} {:>16} {:>9.3f}".format(
            only, n / n_seeds, sum(r["lat"][1.0] is not None for r in group) / n,
            len(lat3) / n, _pcts(lat3, fps), len(lat_gap) / n, _pcts(lat_gap, fps),
            1.0 - len(lat_gap) / n))

    print("(c) the spread cue (signed towards the new team, centred mean over s seconds) turning "
          "positive nearest the hand-over within +-{:g} s: share that flips, lead (-) or lag (+) "
          "s p10/p50/p90, share within +-1 s and +-3 s; at -{:g} s favours the old team, at "
          "+{:g} s the new".format(FLIP_WINDOW_S, FLIP_PROBE_S, FLIP_PROBE_S))
    print("{:<14} {:>6} {:>7} {:>8} {:>18} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
        "gap", "mean s", "n/seed", "flips", "lead/lag", "+-1 s", "+-3 s", "never", "old -2",
        "new +2"))
    for s in FLIP_SMOOTH_S:
        for only, keep in groups:
            group = [r for r in records if keep(r)]
            if not group:
                continue
            n = float(len(group))
            flips = [r["flip"][s] for r in group if r["flip"][s] is not None]
            before = np.array([r["probe"][s][0] for r in group])
            after = np.array([r["probe"][s][1] for r in group])
            with np.errstate(invalid="ignore"):
                print("{:<14} {:>6g} {:>7.1f} {:>8.3f} {:>18} {:>8.3f} {:>8.3f} {:>8.3f} "
                      "{:>8.3f} {:>8.3f}".format(
                          only, s, n / n_seeds, len(flips) / n, _pcts(flips, fps),
                          np.sum(np.abs(flips) <= fps) / n, np.sum(np.abs(flips) <= 3 * fps) / n,
                          1.0 - len(flips) / n, np.mean(before < 0), np.mean(after > 0)))


def prior_report(rows, n_seeds):
    """Can the duration priors represent short possessions inside gaps? Minimum durations and
    log P(d) of the fitted possession prior, per left-out match, and the true possessions that
    lie mostly in a gap by length against them."""
    from solution import decoder
    from solution.sources import idsse
    block_s = decoder.BLOCK / 25.0
    max_blocks = int(round(decoder.MAX_SECONDS / block_s))
    probes = (0.2, 0.4, 0.6, 1.0, 2.0, 5.0, 10.0, 20.0)
    print("\nduration priors (possession, log-normal, leave-one-match-out), log P(d) at d s")
    print("{:<14} {:>7} {:>7} {:>8} {:>7}".format("left out", "min s", "blocks", "median s",
                                                  "sigma") +
          "".join("{:>8}".format("{:g} s".format(d)) for d in probes))
    mins = []
    for mid in idsse.DEV_IDS:
        p = decoder.idsse_priors("idsse_" + mid, 2, "rank")["durations"]["possession"]
        log = decoder.duration_log_prior(p["mu"], p["sigma"], p["min_s"], block_s, max_blocks)
        mins.append(p["min_s"])
        print("{:<14} {:>7.2f} {:>7d} {:>8.2f} {:>7.2f}".format(
            mid, p["min_s"], max(1, int(np.floor(p["min_s"] / block_s))), np.exp(p["mu"]),
            p["sigma"]) + "".join("{:>8.2f}".format(log[int(round(d / block_s)) - 1])
                                  for d in probes))
    # An extra possession of d s inside one of L s: log P(L) - [log P(x) + log P(d) + log P(x)]
    p = decoder.idsse_priors("idsse_" + idsse.DEV_IDS[0], 2, "rank")["durations"]["possession"]
    log = decoder.duration_log_prior(p["mu"], p["sigma"], p["min_s"], block_s, max_blocks)

    def lp(seconds):
        return log[min(max_blocks, max(1, int(round(seconds / block_s)))) - 1]
    print("prior cost (nats) of splitting a possession of L s by a short one of d s in its "
          "middle (L, d): " + "  ".join(
              "({:g}, {:g}) {:.1f}".format(L, d, lp(L) - (2 * lp((L - d) / 2.0) + lp(d)))
              for L in (10.0, 30.0) for d in (0.4, 1.0, 2.0)) +
          "; moving a change costs only the difference of two neighbours' log P")
    lengths = np.array([sec for row in rows for sec, in_gap, _ in row["spells"]["as is"]
                        if in_gap])
    print("true possessions mostly in a gap: {:.0f} per seed; share under the minimum duration "
          "({:.2f}..{:.2f} s) {:.3f}; under 0.4 s {:.3f}, 1 s {:.3f}, 2 s {:.3f}".format(
              len(lengths) / float(n_seeds), min(mins), max(mins),
              np.mean(lengths < np.median(mins)), np.mean(lengths < 0.4),
              np.mean(lengths < 1.0), np.mean(lengths < 2.0)))


# -- Step A of the C2c-or-touch choice: does any players-only feature mark the hand-over? ------

EVENT_TOLS_S = (1.0, 3.0)          # an event counts as a hit within these of the hand-over
FA_CLEAR_S = 3.0                   # false alarms: events this far from every true transition
NEW_SHORT_S = 2.0                  # hand-overs starting a possession shorter than this


def event_scores(truth, match, frames, teams):
    """How one feature's events (frames, teams; team -1 = unsigned) time true open-play
    hand-overs inside ball gaps, and how often they fire away from any true transition.

    Per hand-over: the signed distance to the nearest event (of the new team, if signed) in the
    same half within FLIP_WINDOW_S, or None. False alarms: events on in-gap frames where truth
    is alive and no true transition (of any kind) lies within FA_CLEAR_S. Precision: of the
    events on alive in-gap frames, the share within +-1 s of an in-gap hand-over to their team.
    """
    fps = match.fps
    runs = pm.valid_runs(truth)
    gt = pm.transitions(truth, runs)
    gap = gap_ids(match)[0] >= 0
    window, clear = int(round(FLIP_WINDOW_S * fps)), int(round(FA_CLEAR_S * fps))
    signed = bool(len(teams)) and bool(np.all(teams >= 0))
    true_frames = np.array([t[0] for t in gt], dtype=np.int64)
    lags, short = [], []
    hand = []
    for start, stop in runs:
        inside = (frames >= start) & (frames < stop)
        f_run, k_run = frames[inside], teams[inside]
        for g in gt:
            f, new = g[0], g[2]
            if not (start <= f < stop) or not is_handover(g) or not gap[f]:
                continue
            hand.append((f, new))
            mine = f_run if not signed else f_run[k_run == new]
            near = mine[np.abs(mine - f) <= window]
            lags.append(int(near[np.argmin(np.abs(near - f))] - f) if len(near) else None)
            after = true_frames[(true_frames > f) & (true_frames < stop)]
            short.append(((after[0] if len(after) else stop) - f) < NEW_SHORT_S * fps)
    # Frames far from every true transition, in a gap, truth alive.
    far = np.ones(len(truth), dtype=bool)
    for f in true_frames:
        far[max(0, f - clear):f + clear + 1] = False
    alive_gap = gap & np.isin(truth, pm.POSSESSION_LABELS)
    region = alive_gap & far
    on_alive = alive_gap[frames] if len(frames) else np.zeros(0, dtype=bool)
    right = 0
    tol = int(round(EVENT_TOLS_S[0] * fps))
    hand_f = np.array([h[0] for h in hand], dtype=np.int64)
    hand_k = np.array([h[1] for h in hand], dtype=np.int64)
    for e, k in zip(frames[on_alive], teams[on_alive]):
        close = np.abs(hand_f - e) <= tol
        right += bool(np.any(close & ((hand_k == k) if signed else True)))
    inside = match.in_play_mask()
    return {"lags": lags, "short": short, "signed": signed,
            "fa_events": int(region[frames].sum()) if len(frames) else 0,
            "fa_frames": int(region.sum()),
            "alive_events": int(on_alive.sum()), "alive_right": right,
            "events": int(inside[frames].sum()) if len(frames) else 0,
            "frames": int(inside.sum())}


def events_report(rows, n_seeds, fps):
    """One line per feature: Step A's numbers."""
    from solution import handover
    print("\nStep A: players-only features against true open-play hand-overs inside ball gaps "
          "(hand-overs per seed: {:.1f}; new <{:g} s: {:.1f})".format(
              sum(len(r["spread"]["lags"]) for r in rows) / float(n_seeds), NEW_SHORT_S,
              sum(sum(r["spread"]["short"]) for r in rows) / float(n_seeds)))
    print("hit: an event (towards the new team, if signed) within +-1 / +-3 s; lag: nearest "
          "within +-{:g} s, s p10/p50/p90; FA: events per minute of alive in-gap play farther "
          "than {:g} s from any true transition; chance: hit share of events at the FA rate "
          "placed at random (signed: half the rate per direction); lift = hit / chance; "
          "prec: share of events on alive gap frames within +-1 s of a hand-over to their "
          "team".format(FLIP_WINDOW_S, FA_CLEAR_S))
    print("{:<12} {:>6} {:>7} {:>7} {:>7} {:>18} {:>7} {:>8} {:>7} {:>8} {:>7} {:>8} {:>7}".format(
        "feature", "kind", "ev/min", "hit 1s", "hit 3s", "lag p10/p50/p90", "FA/min",
        "chance1", "lift1", "chance3", "lift3", "<2s 1s", "prec1"))
    for name in handover.FEATURES:
        recs = [r[name] for r in rows]
        lags = [x for r in recs for x in r["lags"]]
        short = [s for r in recs for s in r["short"]]
        n = float(len(lags))
        found = np.array([x for x in lags if x is not None], dtype=float)
        hit = [np.sum(np.abs(found) <= t * fps) / n for t in EVENT_TOLS_S]
        hit_short = sum(1 for x, s in zip(lags, short) if s and x is not None and
                        abs(x) <= EVENT_TOLS_S[0] * fps) / float(max(sum(short), 1))
        fa_min = sum(r["fa_events"] for r in recs) / (sum(r["fa_frames"] for r in recs) /
                                                       float(fps) / 60.0)
        rate = fa_min / 60.0 / (2.0 if recs[0]["signed"] else 1.0)
        chance = [1.0 - np.exp(-rate * (2.0 * t + 1.0 / fps)) for t in EVENT_TOLS_S]
        ev_min = sum(r["events"] for r in recs) / (sum(r["frames"] for r in recs) /
                                                  float(fps) / 60.0)
        prec = sum(r["alive_right"] for r in recs) / float(max(sum(r["alive_events"]
                                                                  for r in recs), 1))
        print("{:<12} {:>6} {:>7.1f} {:>7.3f} {:>7.3f} {:>18} {:>7.1f} {:>8.3f} {:>7.2f} {:>8.3f} "
              "{:>7.2f} {:>8.3f} {:>7.3f}".format(
                  name, handover.FEATURES[name][0] + ("" if recs[0]["signed"] else "*"), ev_min,
                  hit[0], hit[1], _pcts(found, fps), fa_min, chance[0], hit[0] / chance[0],
                  chance[1], hit[1] / chance[1], hit_short, prec))
    print("(* unsigned: an event of either team counts)")


def event_rates(match):
    """Label-free: per feature, events per minute of in-play frames and of in-gap frames."""
    from solution import handover
    inside = match.in_play_mask()
    gap = inside & ~match.ball_seen
    out = {}
    for name in handover.FEATURES:
        frames, _ = handover.events(match, name)
        out[name] = (int(inside[frames].sum()), int(gap[frames].sum()))
    return out, int(inside.sum()), int(gap.sum())


def print_event_rates(title, rows, fps):
    from solution import handover
    print("\n" + title)
    print("{:<12} {:>12} {:>12}".format("feature", "ev/min all", "ev/min gaps"))
    frames = sum(r[1] for r in rows) / float(fps) / 60.0
    gaps = sum(r[2] for r in rows) / float(fps) / 60.0
    for name in handover.FEATURES:
        print("{:<12} {:>12.2f} {:>12.2f}".format(name, sum(r[0][name][0] for r in rows) / frames,
                                                  sum(r[0][name][1] for r in rows) / gaps))


def events_check_bar_ath():
    from solution.sources import atd
    match = atd.load()
    print_event_rates("Step A on BAR-ATH, no labels: feature events per minute (compare with "
                      "--events --rates on the proxy)", [event_rates(match)], match.fps)


def events_check(seeds, plain, noise=0.0, speed=1.0, rates=False):
    from solution import handover
    from solution.simulate import add_position_noise, scale_speeds
    from solution.sources import idsse
    print("Step A: IDSSE dev x seeds {}, simulator v1{}{}{}".format(
        seeds, "" if plain else " + D12 ball stress {} (only 'leave' and 'crowd' read the "
        "last-seen ball)".format(STRESS), ", position noise {:g} m".format(noise) if noise
        else "", ", speed x{:g}".format(speed) if speed != 1 else ""))
    rows, rate_rows = [], []
    for truth, match, _, seed in inputs(idsse.DEV_IDS, seeds, plain, LABELS):
        match = add_position_noise(scale_speeds(match, speed), noise, seed=seed)
        if rates:
            rate_rows.append(event_rates(match))
        row = {}
        for name in handover.FEATURES:
            frames, teams = handover.events(match, name)
            row[name] = event_scores(truth, match, frames, teams)
        rows.append(row)
    events_report(rows, len(seeds), pm.FPS)
    if rates:
        print_event_rates("label-free event rates on the proxy (as --events --bar-ath prints)",
                          rate_rows, pm.FPS)


# -- Step B: short possessions with the ball seen; what touch or E6 work could reach ----------

SEEN_BINS_S = (0.0, 1.0, 2.0, 5.0)        # possession length bins of Step B
SEEN_GROUPS = (("<1 s", 0.0, 1.0), ("1-2 s", 1.0, 2.0), ("<2 s", 0.0, 2.0), ("2-5 s", 2.0, 5.0),
               ("<5 s", 0.0, 5.0))
TOUCH_TOL_S = 1.0                         # a touch "sees" a start within this
SEEN_SNAP_S = 1.5                         # snap oracle tolerance (the C5 window)


def engine_claims(touches):
    """(frames, teams) where the touch engine's holder changes to a team: its hand-over claims,
    from a touch or a control episode."""
    holder = touches.holder
    change = np.flatnonzero((holder[1:] != holder[:-1]) & (holder[1:] >= 0)) + 1
    return change.astype(np.int64), holder[change].astype(np.int8)


def _nearest(frames, f, tol):
    near = frames[np.abs(frames - f) <= tol]
    return int(near[np.argmin(np.abs(near - f))] - f) if len(near) else None


def short_oracles(truth, pred, match, in_gap, sees):
    """(records, {key: codes}) for the true possessions under 5 s lying mostly in a gap
    (`in_gap`) or mostly where the ball is seen (not `in_gap`).

    Per possession: its length, how it starts (hand-over from the other team, restart from D,
    or the half's first), whether the decoder finds it (IoU 0.5), the decoder's team accuracy
    on its both-alive frames, and what `sees(start, end, team)` returns (a dict whose "seen"
    says whether the evidence sees its start). Oracles per group: "team" puts the true team on
    the group's frames where truth and pred are both alive; "snap" moves the decoded changes
    matching the group's true starts and ends (+-SEEN_SNAP_S, destination agreeing) onto them;
    "reach" is "team" on the possessions whose start the evidence sees; "random" is "team" on
    as many of the group's possessions drawn at random (fixed seed), which shows how much of
    the gap between "reach" and "team" is only that short possessions come in clusters.
    """
    from tools.boundaries import move_transitions
    fps = match.fps
    runs = pm.valid_runs(truth)
    filled = pm.fill_abstentions(pred, runs)
    gap = gap_ids(match)[0] >= 0
    gt = pm.possession_spells(truth, runs)
    matched = pm.matched_gt_flags(pm.spell_overlaps(gt, pm.possession_spells(filled, runs)),
                                  len(gt), pm.SPELL_IOU_HEADLINE)
    starts = set(s for s, _ in runs)
    both = np.isin(truth, pm.POSSESSION_LABELS) & np.isin(filled, pm.POSSESSION_LABELS)
    records = []
    members = {g: [] for g, _, _ in SEEN_GROUPS}
    for (a, b, code), found in zip(gt, matched):
        seconds = (b - a) / float(fps)
        if seconds >= SEEN_BINS_S[-1] or (gap[a:b].mean() > 0.5) != in_gap:
            continue
        start = "half" if a in starts else "restart" if truth[a - 1] == pm.D else "hand-over"
        rec = dict(sees(a, b, code), seconds=seconds, start=start, found=bool(found),
                   both=int(both[a:b].sum()),
                   right=int(np.sum(both[a:b] & (filled[a:b] == truth[a:b]))))
        records.append(rec)
        for g, lo, hi in SEEN_GROUPS:
            if lo <= seconds < hi:
                members[g].append((a, b, rec["seen"]))
    out = {}
    n = len(truth)
    predicted = pm.transitions(filled, runs)
    true_t = pm.transitions(truth, runs)
    rng = np.random.default_rng(n)                      # fixed per match: deterministic
    for g, _, _ in SEEN_GROUPS:
        mask, reach, rand = (np.zeros(n, dtype=bool) for _ in range(3))
        picked = rng.permutation(len(members[g]))[:sum(s for _, _, s in members[g])]
        for a, b, seen in members[g]:
            mask[a:b] = True
            reach[a:b] |= seen
        for i in picked:
            a, b, _ = members[g][i]
            rand[a:b] = True
        out[("team", g)] = team_repair(truth, pred, runs, mask)
        out[("reach", g)] = team_repair(truth, pred, runs, reach)
        out[("random", g)] = team_repair(truth, pred, runs, rand)
        bounds = set(f for a, b, _ in members[g] for f in (a, b))
        mine = [t for t in true_t if t[0] in bounds]
        pairs = pm.match_transitions(mine, predicted, int(round(SEEN_SNAP_S * fps)))[0]
        out[("snap", g)] = move_transitions(filled, runs, predicted, {p: t[0] for t, p in pairs})
    return records, out


def seen_analyse(truth, pred, match, touches):
    """Step B: short_oracles on possessions mostly where the ball is seen, with the touch
    engine as the evidence: the nearest touch and the nearest engine claim by the
    possession's team to its start (within +-TOUCH_TOL_S), whether the ball is usable at the
    start, and the share of its frames the engine's holder gives its team. Plus the engine's
    claims on seen, alive frames, and how many sit near a true change to their team."""
    fps = match.fps
    claim_f, claim_k = engine_claims(touches)
    tol = int(round(TOUCH_TOL_S * fps))

    def sees(a, b, code):
        touch = _nearest(touches.touch_frame[touches.touch_team == code], a, tol)
        claim = _nearest(claim_f[claim_k == code], a, tol)
        return {"touch": touch, "claim": claim, "seen_at": bool(touches.usable[a]),
                "held": float(np.mean(touches.holder[a:b] == code)),
                "seen": touch is not None or claim is not None}
    records, out = short_oracles(truth, pred, match, False, sees)
    runs = pm.valid_runs(truth)
    true_t = pm.transitions(truth, runs)
    # The engine's claims on seen, alive frames: how many sit near a true change to their team.
    alive_seen = touches.usable & np.isin(truth, pm.POSSESSION_LABELS)
    into = {}
    for f, _, after in true_t:
        into.setdefault(after, []).append(f)
    into = {k: np.array(v, dtype=np.int64) for k, v in into.items()}
    on = alive_seen[claim_f]
    right = sum(1 for f, k in zip(claim_f[on], claim_k[on])
                if k in into and np.any(np.abs(into[k] - f) <= tol))
    claims = {"claims": int(on.sum()), "claims_right": right,
              "seen_alive_frames": int(alive_seen.sum())}
    return records, out, claims


def oracle_table(rows, rec_key, codes_key, n_seeds, base):
    """short_oracles' gains per length group; 'seen' is the share of starts the evidence sees."""
    print("{:<8} {:>7} {:>8} {:>9} {:>7} {:>9} {:>9} {:>9} {:>9}".format(
        "length", "n/seed", "recall", "team acc", "seen", "team", "snap", "reach", "random"))
    records = [r for row in rows for r in row[rec_key]]
    for g, lo, hi in SEEN_GROUPS:
        group = [r for r in records if lo <= r["seconds"] < hi]
        if not group:
            continue
        gains = [pooled_codes(rows, lambda r, k=(kind, g): r[codes_key][k]) - base
                 for kind in ("team", "snap", "reach", "random")]
        print("{:<8} {:>7.1f} {:>8.3f} {:>9.3f} {:>7.3f} {:>+9.4f} {:>+9.4f} {:>+9.4f} "
              "{:>+9.4f}".format(
                  g, len(group) / float(n_seeds), np.mean([r["found"] for r in group]),
                  sum(r["right"] for r in group) / float(max(sum(r["both"] for r in group), 1)),
                  np.mean([r["seen"] for r in group]), *gains))


def gap_event_analyse(truth, pred, match, name="dadvance0.5", in_gap=True, touches=None):
    """Step A's analogue of Step B's reach: short_oracles on possessions mostly in a gap (or,
    with in_gap False, mostly where the ball is seen), with one handover feature's events as
    the evidence (an event of the possession's team within +-TOUCH_TOL_S of its start; any
    event if the feature is unsigned). With `touches`, a touch or engine claim by the team
    within +-TOUCH_TOL_S also counts: the union of both kinds of evidence."""
    from solution import handover
    frames, teams = handover.events(match, name)
    tol = int(round(TOUCH_TOL_S * match.fps))
    if touches is not None:
        claim_f, claim_k = engine_claims(touches)
        frames = np.concatenate((frames, touches.touch_frame, claim_f))
        teams = np.concatenate((teams, touches.touch_team, claim_k))

    def sees(a, b, code):
        mine = frames if not np.all(teams >= 0) else frames[teams == code]
        return {"seen": _nearest(mine, a, tol) is not None}
    return short_oracles(truth, pred, match, in_gap, sees)


def seen_report(name, rows, n_seeds, fps):
    base = pooled_codes(rows, lambda r: r["pred"])
    print("\n== {}: Step B, short true possessions lying mostly where the ball is seen ==  "
          "(spellF1 as is {:.4f}; D12 ball stress unless --plain)".format(name, base))
    print("oracle gains, pooled spellF1: team = the true team on the group's both-alive frames; "
          "snap = decoded changes matching the group's true starts and ends (+-{:g} s) moved "
          "onto them; reach = team, only on possessions whose start the touch engine sees (a "
          "touch or a holder change by their team within +-{:g} s)".format(SEEN_SNAP_S,
                                                                          TOUCH_TOL_S))
    oracle_table(rows, "seen_records", "seen", n_seeds, base)
    records = [r for row in rows for r in row["seen_records"]]
    print("what the touch engine sees at their starts (share of possessions): ball usable at the "
          "start; a touch by their team within +-0.5 / +-1 s; an engine claim (holder change to "
          "their team) within +-1 s, its lag s p10/p50/p90; mean share of their frames held by "
          "their team")
    print("{:<8} {:<10} {:>7} {:>8} {:>9} {:>9} {:>9} {:>18} {:>8}".format(
        "length", "start", "n/seed", "usable", "touch .5", "touch 1", "claim 1",
        "claim lag", "held"))
    for g, lo, hi in SEEN_GROUPS:
        for start in ("all", "hand-over", "restart"):
            group = [r for r in records if lo <= r["seconds"] < hi and
                     (start == "all" or r["start"] == start)]
            if not group:
                continue
            n = float(len(group))
            lags = [r["claim"] for r in group if r["claim"] is not None]
            print("{:<8} {:<10} {:>7.1f} {:>8.3f} {:>9.3f} {:>9.3f} {:>9.3f} {:>18} {:>8.3f}".format(
                g, start, n / n_seeds, sum(r["seen_at"] for r in group) / n,
                sum(r["touch"] is not None and abs(r["touch"]) <= 0.5 * fps for r in group) / n,
                sum(r["touch"] is not None for r in group) / n, len(lags) / n,
                _pcts(lags, fps), np.mean([r["held"] for r in group])))
    claims = [row["claims"] for row in rows]
    total = sum(c["claims"] for c in claims)
    print("engine claims on usable-ball alive frames: {:.1f} per minute; share within +-{:g} s of "
          "a true change to their team {:.3f}".format(
              total / (sum(c["seen_alive_frames"] for c in claims) / float(fps) / 60.0),
              TOUCH_TOL_S, sum(c["claims_right"] for c in claims) / float(max(total, 1))))
    if "gap_records" in rows[0]:
        feature = rows[0]["gap_feature"]
        for key, text in (
                ("seen_dadv", "the same possessions (mostly ball seen); 'seen' = a {} event of "
                              "their team within +-{:g} s of their start"),
                ("seen_union", "the same; 'seen' = a {} event, a touch or an engine claim by "
                               "their team within +-{:g} s"),
                ("gap", "Step A analogue: short true possessions lying mostly in a gap; 'seen' = "
                        "a {} event of their team within +-{:g} s of their start")):
            print("\n" + text.format(feature, TOUCH_TOL_S))
            oracle_table(rows, key + "_records", key, n_seeds, base)


# -- Step 0 of the transition bonus: what do the false alarms cost? ---------------------------

BONUS_WEIGHTS = (2.0, 4.0, 7.0)              # nats; a short possession costs about 7 (D17)
BONUS_SHIFTS_S = (0.0, -0.7)                 # dadvance lags hand-overs by a median 0.7 s
BONUS_RIGHT_S = 1.0                          # "right" events: this close to a true hand-over


def true_handover_events(truth):
    """(frames, teams) of the true H <-> A hand-overs: the perfect event source."""
    out = [(f, after) for f, before, after in pm.transitions(truth, pm.valid_runs(truth))
           if is_handover((f, before, after))]
    return (np.array([f for f, _ in out], dtype=np.int64),
            np.array([k for _, k in out], dtype=np.int8))


def right_events(frames, teams, truth, fps):
    """The events within BONUS_RIGHT_S of a true hand-over to their team: an oracle filter."""
    true_f, true_k = true_handover_events(truth)
    tol = int(round(BONUS_RIGHT_S * fps))
    keep = np.array([np.any((np.abs(true_f - f) <= tol) & (true_k == k))
                     for f, k in zip(frames, teams)], dtype=bool)
    return frames[keep], teams[keep]


def bonus_settings():
    """[(label, source, weight, shift)] of the Step 0 grid."""
    out = []
    for source in ("all", "right"):
        for shift in BONUS_SHIFTS_S:
            for w in BONUS_WEIGHTS:
                out.append(("dadvance {} w {:g} shift {:+g}".format(source, w, shift),
                            source, w, shift))
    out += [("perfect w {:g}".format(w), "perfect", w, 0.0) for w in BONUS_WEIGHTS]
    return out


def bonus_codes(match, truth, source, weight, shift):
    """v4_spread with the switch bonus fed by one event source."""
    from solution import methods
    if source == "perfect":
        def fn(m):
            return true_handover_events(truth)
    elif source == "right":
        def fn(m):
            return right_events(*methods.dadvance_events(m), truth, m.fps)
    else:
        fn = methods.dadvance_events
    return methods.semi_markov("bonus", touch_w=1.0, kickoff_params={}, team_w=0.5,
                               team_where="all", team_dead="max",
                               team_params={"cues": ("spread",)}, bonus_fn=fn,
                               bonus_w=weight, bonus_shift_s=shift)(match)


def bonus_check(seeds, plain):
    """Step 0: the switch bonus fed by every dadvance event, by only its right ones (an oracle
    filter), and by every true hand-over, against v4_spread."""
    from solution.sources import idsse
    settings = bonus_settings()
    print("transition bonus Step 0: IDSSE dev x seeds {}, simulator v1{}; against "
          "v4_spread".format(seeds, "" if plain else " + D12 ball stress"))
    rows = []
    for truth, match, _, seed in inputs(idsse.DEV_IDS, seeds, plain, LABELS):
        gap_mask = gap_ids(match)[0] >= 0
        row = {"truth": truth, "seed": seed, "gap_mask": gap_mask,
               "pred": METHODS["v4_spread"](match), "codes": {}}
        for label, source, w, shift in settings:
            row["codes"][label] = bonus_codes(match, truth, source, w, shift)
        rows.append(row)
    base = pooled_codes(rows, lambda r: r["pred"])
    print("v4_spread {:.4f}".format(base))
    print("{:<34} {:>9} {:>8} {:>12} {:>12} {:>8}".format(
        "bonus", "spellF1", "gain", "recall <2 s", "gap <2 s", "ratio"))

    def line(label, codes_of):
        recall = {True: [0, 0], False: [0, 0], None: [0, 0]}
        count = [0, 0]
        for row in rows:
            codes = codes_of(row)
            for seconds, in_gap, matched in spell_records(row["truth"], codes, row["gap_mask"]):
                if seconds < 2.0:
                    for key in (None, in_gap):
                        recall[key][0] += matched
                        recall[key][1] += 1
            runs = pm.valid_runs(row["truth"])
            count[0] += len(pm.possession_spells(pm.fill_abstentions(codes, runs), runs))
            count[1] += len(pm.possession_spells(row["truth"], runs))
        score = pooled_codes(rows, codes_of)
        print("{:<34} {:>9.4f} {:>+8.4f} {:>12.3f} {:>12.3f} {:>8.3f}".format(
            label, score, score - base, recall[None][0] / float(max(recall[None][1], 1)),
            recall[True][0] / float(max(recall[True][1], 1)), count[0] / float(count[1])))
    line("v4_spread", lambda r: r["pred"])
    for label, _, _, _ in settings:
        line(label, lambda r, k=label: r["codes"][k])


# -- Step 2: label-free decoded aggregates inside gaps, for BAR-ATH ---------------------------

# Smart Tagging phase codes whose Team tag has the ball; each defending code mirrors one of
# them exactly with the team swapped (FINDINGS point 5), so it only checks the side mapping.
ATTACK_CODES = ("build up", "progression", "creating chances", "attacking transition")
DEFEND_CODES = ("defending in attacking third", "defending in middle third",
                "defending in defensive third", "defensive transition")


def decoded_aggregates(match, codes, tag_team=None):
    """Label-free counts of one method's codes on one match: decoded H, A, D inside ball gaps
    and with the ball seen, possessions (all, mostly in a gap, and those under 2 s), H <-> A
    changes inside gaps, and, with `tag_team` (per frame H, A or -1), agreement with it."""
    runs = [tuple(h) for h in match.halves]
    inside = match.in_play_mask()
    gap = inside & ~match.ball_seen
    seen = inside & match.ball_seen
    spells = pm.possession_spells(codes, runs)
    in_gap = [(b - a, gap[a:b].mean() > 0.5) for a, b, _ in spells]
    changes = [f for f, before, after in pm.transitions(codes, runs)
               if gap[f] and is_handover((f, before, after))]
    out = {"minutes": inside.sum() / float(match.fps) / 60.0, "frames": int(inside.sum()),
           "poss": len(spells), "poss_gap": sum(g for _, g in in_gap),
           "short_gap": sum(g and n < 2 * match.fps for n, g in in_gap),
           "changes_gap": len(changes)}
    for where, mask in (("gap", gap), ("seen", seen)):
        for name, code in (("H", pm.H), ("A", pm.A), ("D", pm.D)):
            out[where + name] = int(np.sum(mask & (codes == code)))
        if tag_team is not None:
            use = mask & (tag_team >= 0) & np.isin(codes, pm.POSSESSION_LABELS)
            out[where + "_tagged"] = int(use.sum())
            out[where + "_agree"] = int(np.sum(codes[use] == tag_team[use]))
    return out


def print_decoded(title, rows_by_method):
    print("\n" + title)
    print("{:<22} {:>9} {:>9} {:>9} {:>8} {:>10} {:>10} {:>11} {:>10} {:>10}".format(
        "method", "gap H/HA", "gap D", "seen D", "poss/90", "gap pos/90", "gap <2s/90",
        "gap chg/90", "tag gap", "tag seen"))
    for name, rows in rows_by_method.items():
        def total(key):
            return float(sum(r.get(key, 0) for r in rows))
        per90 = 90.0 / total("minutes")
        gap = total("gapH") + total("gapA") + total("gapD")
        seen = total("seenH") + total("seenA") + total("seenD")
        tags = ["{:.3f}".format(total(w + "_agree") / total(w + "_tagged"))
                if total(w + "_tagged") else "--" for w in ("gap", "seen")]
        print("{:<22} {:>9.3f} {:>9.3f} {:>9.3f} {:>8.1f} {:>10.1f} {:>10.1f} {:>11.1f} {:>10} "
              "{:>10}".format(name[:22], total("gapH") / max(total("gapH") + total("gapA"), 1),
                              total("gapD") / max(gap, 1), total("seenD") / max(seen, 1),
                              total("poss") * per90, total("poss_gap") * per90,
                              total("short_gap") * per90, total("changes_gap") * per90, *tags))


def smart_tag_team(match):
    """(per frame H, A or -1 from the attacking phase tags, {check: count}) on BAR-ATH.

    A frame inside attacking phases of both teams is left -1. The defending codes give the
    same per-frame team with the side swapped; the counts say how often they agree with it.
    """
    sys.path.insert(0, os.path.join(ROOT, "starter"))
    import baseline
    import load_smart_tagging
    from games import GAME, require
    home = baseline.team_names(require(["atd_meta"], GAME)[0])[0].strip().lower()
    n = match.n_frames
    votes = {code: np.zeros((n, 2), dtype=bool) for code in ("attack", "defend")}
    counts = {"attack tags": 0, "defend tags": 0, "no team": 0}
    for tag in load_smart_tagging.load():
        code = (tag["code"] or "").strip().lower()
        kind = "attack" if code in ATTACK_CODES else "defend" if code in DEFEND_CODES else None
        if kind is None:
            continue
        team = (tag["tags"].get("Team") or "").strip().lower()
        if team in ("", "n/a"):
            counts["no team"] += 1
            continue
        side = pm.H if team == home else pm.A
        if kind == "defend":
            side = 1 - side                                  # the other team has the ball
        votes[kind][max(0, tag["start_frame"]):max(0, tag["end_frame"]), side] = True
        counts[kind + " tags"] += 1
    inside = match.in_play_mask()

    def team_of(v):
        return np.where(v[:, 0] & ~v[:, 1], pm.H, np.where(v[:, 1] & ~v[:, 0], pm.A, -1))
    attack, defend = team_of(votes["attack"]), team_of(votes["defend"])
    both = inside & (attack >= 0) & (defend >= 0)
    counts["frames tagged"] = int(np.sum(inside & (attack >= 0)))
    counts["mirror agree"] = "{:.3f}".format(np.mean(attack[both] == defend[both])
                                             if both.any() else float("nan"))
    counts["home share"] = "{:.3f}".format(np.mean(attack[inside & (attack >= 0)] == pm.H))
    return np.where(inside, attack, -1), counts


def decoded_check_bar_ath(methods):
    from solution.sources import atd
    match = atd.load()
    tag_team, counts = smart_tag_team(match)
    print("Smart Tagging attacking phases: " + ", ".join(
        "{} {}".format(k, v) for k, v in counts.items()) +
        " (mirror agree: the defending codes give the same team; home share: of tagged frames)")
    print_decoded("BAR-ATH, no labels: decoded inside ball gaps (gap H/HA: home share of decoded "
                  "team frames; tag: agreement with the attacking phase's team, on decoded team "
                  "frames)",
                  {name: [decoded_aggregates(match, METHODS[name](match), tag_team)]
                   for name in methods})


def decoded_check(methods, seeds, plain):
    from solution.sources import idsse
    rows = {name: [] for name in methods + ["truth"]}
    for truth, match, _, _ in inputs(idsse.DEV_IDS, seeds, plain, LABELS):
        for name in methods:
            rows[name].append(decoded_aggregates(match, METHODS[name](match)))
        rows["truth"].append(decoded_aggregates(match, truth))
    print_decoded("IDSSE dev x seeds {}{}: the same aggregates (no tags; 'truth' is the target)"
                  .format(seeds, "" if plain else ", D12 stress"), rows)


# -- Step 1: team cues -----------------------------------------------------------------------

ALL_CUES = ("advance", "line", "spread", "pressure")


def cue_rows(match, truth=None, tables=None):
    """Per cue, aggregates over the match's graded frames without a seen ball."""
    from solution import gapteam
    gap = match.in_play_mask() & ~match.ball_seen
    values = gapteam.cues(match, dict(gapteam.PARAMS, cues=ALL_CUES))
    out = {}
    for k, name in enumerate(ALL_CUES):
        value = values[gap, k]
        known = np.isfinite(value)
        half, n_bins = gapteam.CUE_BINS[name]
        fire = known & (np.abs(np.where(known, value, 0.0)) > half / n_bins)
        row = {"frames": int(gap.sum()), "known": int(known.sum()), "fires": int(fire.sum()),
               "size": np.abs(value[known]), "sum": float(np.sum(value[known]))}
        if truth is not None:
            labels = truth[gap]
            alive = np.isin(labels, pm.POSSESSION_LABELS)
            sign = np.where(labels == pm.H, 1.0, -1.0)
            pick = alive & fire
            row["alive_fires"] = int(pick.sum())
            row["right"] = int(np.sum(np.sign(value[pick]) == sign[pick]))
            use = alive & known
            log = gapteam.state_log(value[use], tables[k], name)
            true_team = (labels[use] == pm.A).astype(int)
            row["llr"] = float(np.sum(log[np.arange(len(log)), true_team] -
                                      log[np.arange(len(log)), 1 - true_team]))
            row["alive_known"] = int(use.sum())
        out[name] = row
    return out


def print_cues(title, rows):
    print("\n" + title)
    labelled = "right" in rows[0][ALL_CUES[0]]
    print("{:<10} {:>8} {:>8} {:>10} {:>10} {:>10}".format("cue", "known", "fires", "|c| p50",
                                                           "|c| p90", "mean c")
          + ("{:>14} {:>14}".format("sign right", "LLR / frame") if labelled else ""))
    for name in ALL_CUES:
        group = [r[name] for r in rows]
        frames = sum(r["frames"] for r in group)
        size = np.concatenate([r["size"] for r in group])
        line = "{:<10} {:>8.3f} {:>8.3f} {:>10.2f} {:>10.2f} {:>+10.2f}".format(
            name, sum(r["known"] for r in group) / float(frames),
            sum(r["fires"] for r in group) / float(frames),
            *(np.percentile(size, [50, 90]) if len(size) else [np.nan] * 2),
            sum(r["sum"] for r in group) / float(max(sum(r["known"] for r in group), 1)))
        if labelled:
            line += "{:>14.3f} {:>14.3f}".format(
                sum(r["right"] for r in group) / float(max(sum(r["alive_fires"] for r in group), 1)),
                sum(r["llr"] for r in group) / float(max(sum(r["alive_known"] for r in group), 1)))
        print(line)


def cue_check(seeds, plain):
    from solution import gapteam
    from solution.sources import idsse
    rows = []
    for truth, match, _, _ in inputs(idsse.DEV_IDS, seeds, plain, LABELS):
        tables = gapteam.idsse_log(match.name, (("cues", ALL_CUES),))
        rows.append(cue_rows(match, truth, tables))
    print_cues("team cues on frames without a seen ball, IDSSE dev x seeds {}{} (sign right and "
               "LLR on true-alive frames; LLR fitted leave-one-match-out)".format(
                   seeds, "" if plain else ", D12 stress"), rows)


def cue_check_bar_ath():
    from solution.sources import atd
    print_cues("team cues on frames without a seen ball, BAR-ATH (no labels)",
               [cue_rows(atd.load())])


# -- after D22's BAR-ATH check: why do gaps disagree with the tags? --------------------------

TAG_METHODS = ("v4_spread", "touch_c3", "touch_c3_w1")
SINCE_BINS_S = (0.0, 2.0, 5.0, 15.0, np.inf)     # time since the ball was last seen
THIN_PLAYERS = 5                                  # a team with fewer tracked players is "thin"


def since_seen(match):
    """Per frame, seconds since the ball was last seen in the half (inf if not yet); 0 where
    it is seen."""
    out = np.full(match.n_frames, np.inf)
    for start, stop in match.halves:
        index = np.where(match.ball_seen[start:stop], np.arange(start, stop), -1)
        last = np.maximum.accumulate(index)
        out[start:stop] = np.where(last >= 0, (np.arange(start, stop) - last) / float(match.fps),
                                   np.inf)
    return out


def thin_tracking(match, minimum=THIN_PLAYERS):
    """Per frame, True where either team has fewer than `minimum` tracked players."""
    from solution.match import AWAY, HOME
    known = np.isfinite(match.player_x) & np.isfinite(match.player_y)
    return ((known[:, match.team == HOME].sum(axis=1) < minimum) |
            (known[:, match.team == AWAY].sum(axis=1) < minimum))


def tag_rows(match, ref, methods):
    """Label-free counts on in-play gap frames where `ref` (per frame H, A or -1: the Smart
    Tagging team on BAR-ATH, the true team on the proxy) is known, per split: frames, ref home
    frames, and per signal (frame model sign, spread sign, each method's decoded team) the
    frames it reads and agrees on."""
    from solution import framemodel, gapteam
    gap = match.in_play_mask() & ~match.ball_seen & (ref >= 0)
    since = since_seen(match)
    thin = thin_tracking(match)
    model = framemodel.log_odds(match)
    spread = gapteam.spread(match)
    signals = {"model": np.where(np.isfinite(model) & (model != 0),
                                 np.where(model > 0, pm.H, pm.A), -1),
               "spread": np.where(np.isfinite(spread) & (spread != 0),
                                  np.where(spread > 0, pm.H, pm.A), -1)}
    for name in methods:
        codes = METHODS[name](match)
        signals[name] = np.where(np.isin(codes, pm.POSSESSION_LABELS), codes, -1)
    splits = {"all gap frames": gap, "full tracking": gap & ~thin, "thin tracking": gap & thin}
    for lo, hi in zip(SINCE_BINS_S[:-1], SINCE_BINS_S[1:]):
        splits["missing " + bin_label(lo, hi)] = gap & (since >= lo) & (since < hi)
    out = {}
    for key, mask in splits.items():
        row = {"frames": int(mask.sum()), "home": int(np.sum(mask & (ref == pm.H))),
               "all_gap": int(gap.sum())}
        for name, value in signals.items():
            use = mask & (value >= 0)
            row[name] = (int(use.sum()), int(np.sum(value[use] == ref[use])))
        out[key] = row
    return out


def print_tag_rows(title, rows, signals, ref_name):
    print("\n" + title)
    print("{:<18} {:>8} {:>9}".format("split", "share", ref_name + " H") + "".join(
        "{:>16}".format(s[:16]) for s in signals))
    for key in rows[0]:
        group = [r[key] for r in rows]
        frames = sum(g["frames"] for g in group)
        cells = []
        for s in signals:
            read = sum(g[s][0] for g in group)
            agree = sum(g[s][1] for g in group)
            cells.append("{:.3f} ({:.2f})".format(agree / float(max(read, 1)),
                                                  read / float(max(frames, 1))))
        print("{:<18} {:>8.3f} {:>9.3f}".format(
            key, frames / float(max(sum(g["all_gap"] for g in group), 1)),
            sum(g["home"] for g in group) / float(max(frames, 1))) +
            "".join("{:>16}".format(c) for c in cells))
    print("(share: of gap frames with the reference known; cells: agreement (share of the "
          "split's frames the signal reads))")


def tag_check(methods, seeds, plain, bar_ath):
    if bar_ath:
        from solution.sources import atd
        match = atd.load()
        tag_team, _ = smart_tag_team(match)
        print_tag_rows("BAR-ATH, no labels: agreement with the Smart Tagging attacking-phase "
                       "team on frames without a seen ball", [tag_rows(match, tag_team, methods)],
                       ["model", "spread"] + list(methods), "tag")
        return
    from solution.sources import idsse
    rows = []
    for truth, match, _, _ in inputs(idsse.DEV_IDS, seeds, plain, LABELS):
        ref = np.where(np.isin(truth, pm.POSSESSION_LABELS), truth, -1)
        rows.append(tag_rows(match, ref, methods))
    print_tag_rows("IDSSE dev x seeds {}{}: the same, against the true team ({}, alive frames)"
                   .format(seeds, "" if plain else ", D12 stress", LABELS), rows,
                   ["model", "spread"] + list(methods), "true")


# -- visible-ball Step 0: what is the loss where the ball is seen made of? -------------------

BOUNDARY_S = 1.5             # frames this close to a true transition count as boundary errors


def matched_pred_flags(overlaps, n_pred, threshold=pm.SPELL_IOU_HEADLINE):
    """Boolean array over predicted spells, True where one was matched (pm's greedy)."""
    flags = np.zeros(n_pred, dtype=bool)
    used = set()
    for iou, gt_index, pred_index in overlaps:
        if iou < threshold:
            break
        if gt_index in used or flags[pred_index]:
            continue
        used.add(gt_index)
        flags[pred_index] = True
    return flags


def error_kinds(truth, pred, match, fps):
    """{kind: frame mask} of team errors (truth and prediction both alive, teams differ) on
    frames where the ball is seen, by kind, first that applies: "boundary" (within BOUNDARY_S
    of a true transition), "invented" (inside a decoded possession matching no true one at
    IoU 0.5), "inside" (the rest: the wrong team inside a matched possession); plus dead-ball
    errors "true D, pred team" and "true team, pred D" (not repaired by the team oracle)."""
    runs = pm.valid_runs(truth)
    filled = pm.fill_abstentions(pred, runs)
    seen = match.in_play_mask() & match.ball_seen
    alive_t = np.isin(truth, pm.POSSESSION_LABELS)
    alive_p = np.isin(filled, pm.POSSESSION_LABELS)
    wrong = seen & alive_t & alive_p & (filled != truth)
    near = np.zeros(len(truth), dtype=bool)
    pad = int(round(BOUNDARY_S * fps))
    for f, _, _ in pm.transitions(truth, runs):
        near[max(0, f - pad):f + pad] = True
    gt = pm.possession_spells(truth, runs)
    predicted = pm.possession_spells(filled, runs)
    matched = matched_pred_flags(pm.spell_overlaps(gt, predicted), len(predicted))
    invented = np.zeros(len(truth), dtype=bool)
    for (a, b, _), m in zip(predicted, matched):
        if not m:
            invented[a:b] = True
    kinds = {"boundary": wrong & near}
    kinds["invented"] = wrong & ~near & invented
    kinds["inside"] = wrong & ~near & ~invented
    kinds["true D, pred team"] = seen & (truth == pm.D) & alive_p
    kinds["true team, pred D"] = seen & alive_t & (filled == pm.D)
    return kinds, int(seen.sum()), invented & seen


def seen_ceiling_check(methods, seeds, plain):
    """Visible-ball Step 0: team errors on seen frames by kind, and the team oracle on each."""
    from solution.sources import idsse
    print("visible-ball Step 0: IDSSE dev x seeds {}, simulator v1{}; truth {}".format(
        seeds, "" if plain else " + D12 ball stress {}".format(STRESS), LABELS))
    rows = {name: [] for name in methods}
    for truth, match, _, seed in inputs(idsse.DEV_IDS, seeds, plain, LABELS):
        runs = pm.valid_runs(truth)
        for name in methods:
            if hasattr(METHODS[name], "prepare"):
                pred, records = invented_records(truth, match, METHODS[name])
            else:
                pred, records = METHODS[name](match), []
            kinds, seen, invented = error_kinds(truth, pred, match, match.fps)
            repairs = {k: team_repair(truth, pred, runs, m) for k, m in kinds.items()
                       if k in ("boundary", "invented", "inside")}
            repairs["all seen"] = team_repair(truth, pred, runs,
                                              match.in_play_mask() & match.ball_seen)
            # Invented possessions removed: their seen frames take the true team (alive) where
            # truth is alive, which merges them into their neighbours.
            repairs["invented, whole"] = team_repair(truth, pred, runs, invented)
            # The same, only on the invented possessions truth gives the other team (phantom
            # hand-overs), and only those of them the holder started with a touch hand-over.
            for key, keep in INVENTED_ORACLES:
                mask = np.zeros(len(truth), dtype=bool)
                for r in records:
                    if keep(r):
                        mask[r["span"][0]:r["span"][1]] = True
                repairs[key] = team_repair(truth, pred, runs, mask & invented)
            rows[name].append({"truth": truth, "pred": pred, "seed": seed, "seen": seen,
                               "counts": {k: int(m.sum()) for k, m in kinds.items()},
                               "repairs": repairs, "invented": records})
    for name in methods:
        rs = rows[name]
        base = pooled_codes(rs, lambda r: r["pred"])
        seen = float(sum(r["seen"] for r in rs))
        print("\n== {} ==  spellF1 {:.4f}".format(name, base))
        print("{:<22} {:>14} {:>12}".format("kind", "share of seen", "oracle gain"))
        for kind in ("boundary", "invented", "inside", "invented, whole") + tuple(
                key for key, _ in INVENTED_ORACLES if key in rs[0]["repairs"]) + (
                "all seen", "true D, pred team", "true team, pred D"):
            share = sum(r["counts"].get(kind, 0) for r in rs) / seen if kind in rs[0]["counts"] \
                else float("nan")
            gain = (pooled_codes(rs, lambda r, k=kind: r["repairs"][k]) - base
                    if kind in rs[0]["repairs"] else float("nan"))
            print("{:<22} {:>14.3f} {:>+12.4f}".format(kind, share, gain))
        print("(team kinds count frames where truth and prediction are both alive and differ; "
              "'invented, whole': the true team on every seen frame of decoded possessions "
              "that match none; dead-ball kinds are not repaired)")
        invented_report(rs, len(seeds))


# -- visible-ball Step 0, part 2: what makes the decoder invent a possession? ----------------

SOURCES = ("nearest", "holder", "spread", "frame")      # the decoder's team evidence terms
ABLATE = {"nearest": None, "holder": {"touch_w": 0.0}, "spread": {"team_w": 0.0},
          "frame": {"frame_w": 0.0}}                    # None: the nearest-player codes blanked
INVENTED_BINS_S = (0.0, 1.0, 2.0, 5.0, 10.0, np.inf)
START_S = 1.0                # "at its start": the evidence over its first second
NEAR_S = 1.5                 # a true transition this close to a start or end counts as next to it
REMOVED = 0.5                # an ablation removes a spell if under this share stays its team
INVENTED_ORACLES = (         # "invented, whole" restricted to a group of invented possessions
    ("inv. other team", lambda r: r["truth"] == "other team"),
    ("inv. other, touch h-o", lambda r: r["truth"] == "other team" and
     r["holder"] == "touch hand-over"))


def evidence_llr(method, match, inp):
    """{source: per-frame log P(evidence | H) - log P(evidence | A)} as the decoder weighs it
    (decoder.block_emissions, w included), each block's value spread evenly over its frames.
    Each source is isolated by switching it off and differencing, so they add up exactly to
    the full H - A emission difference (checked)."""
    from solution import decoder
    settings = method.settings
    out = {s: np.zeros(match.n_frames) for s in SOURCES}
    for start, stop in inp["spans"]:
        (nearest, seen, speed, priors, holder, cue, team, frame,
         deadm) = method.half_inputs(inp, start, stop)
        n = stop - start
        size = np.bincount(np.arange(n) // decoder.BLOCK).astype(float)

        def diff(params, near=nearest, spd=speed):
            e = decoder.block_emissions(near, seen, spd, priors, params, holder, cue, team, frame,
                                        deadm)
            return e[:, decoder.H] - e[:, decoder.A]
        full = diff(settings)
        parts = {"nearest": diff(dict(settings, touch_w=0.0, frame_w=0.0, team_w=0.0,
                                      deadball_w=0.0), spd=np.full(n, np.nan))}
        for source in ("holder", "spread", "frame"):
            parts[source] = full - diff(dict(settings, **ABLATE[source]))
        assert np.allclose(sum(parts.values()), full, atol=1e-6), "evidence does not add up"
        for source, values in parts.items():
            out[source][start:stop] = np.repeat(values / size, decoder.BLOCK)[:n]
    return out


def holder_start(touches, a, b, k, fps):
    """How the touch engine's holder came to give team k the possession [a, b): the holder
    change to k nearest its start within +-TOUCH_TOL_S, by the event that made it ("touch
    hand-over": a touch_takes hand-over from the other team; "touch from none"; "control"),
    or "held before" (k already holds at a - TOUCH_TOL_S), or "none" (no such change)."""
    holder = touches.holder
    tol = int(round(TOUCH_TOL_S * fps))
    lo, hi = max(1, a - tol), min(len(holder), a + tol + 1)
    f = np.arange(lo, hi)
    changes = f[(holder[f] == k) & (holder[f - 1] != k)]
    if not len(changes):
        return "held before" if holder[max(0, a - tol)] == k and np.any(holder[a:b] == k) \
            else "none"
    f = int(changes[np.argmin(np.abs(changes - a))])
    if np.any((touches.control[:, 0] == f) & (touches.control[:, 2] == k)):
        return "control"
    if np.any((touches.touch_frame == f) & (touches.touch_team == k)):
        return "touch hand-over" if holder[f - 1] == 1 - k else "touch from none"
    return "other"


def invented_records(truth, match, method):
    """(pred codes, [record]) for the decoded possessions of `method` that match no true one at
    IoU 0.5 and lie mostly on seen in-play frames. Per record: length, what truth holds under
    it (shares of its own team, the other team, D), its wrong-team seen frames (the
    'invented' error kind), the summed evidence towards its team per source over its first
    START_S and over all of it, whether each single-source ablation removes it, how the
    holder started it, and its distance to true transitions."""
    from solution.methods import to_scorer_codes
    fps = match.fps
    inp = method.prepare(match)
    states = method.decode(match, inp)
    pred = to_scorer_codes(np.array(["H", "A", "D"])[states], match)
    runs = pm.valid_runs(truth)
    filled = pm.fill_abstentions(pred, runs)
    seen = match.in_play_mask() & match.ball_seen
    gt = pm.possession_spells(truth, runs)
    predicted = pm.possession_spells(filled, runs)
    matched = matched_pred_flags(pm.spell_overlaps(gt, predicted), len(predicted))
    candidates = [(a, b, k) for (a, b, k), m in zip(predicted, matched)
                  if not m and seen[a:b].mean() > 0.5]
    if not candidates:
        return pred, []
    llr = evidence_llr(method, match, inp)
    ablated = {}
    for source, overrides in ABLATE.items():
        if overrides is None:
            ablated[source] = method.decode(match, dict(inp, nearest=np.full(match.n_frames, -1,
                                                                             dtype=np.int8)))
        else:
            ablated[source] = method.decode(match, inp, overrides)
    true_t = np.array([f for f, _, _ in pm.transitions(truth, runs)], dtype=np.int64)
    first = int(round(START_S * fps))
    near = int(round(NEAR_S * fps))
    alive_t = np.isin(truth, pm.POSSESSION_LABELS)
    records = []
    for a, b, k in candidates:
        sign = 1.0 if k == pm.H else -1.0
        whole = {s: sign * float(llr[s][a:b].sum()) for s in SOURCES}
        start = {s: sign * float(llr[s][a:min(b, a + first)].sum()) for s in SOURCES}
        own, other = float(np.mean(truth[a:b] == k)), float(np.mean(truth[a:b] == 1 - k))

        def gap_to(f):
            return int(np.min(np.abs(true_t - f))) if len(true_t) else 10 ** 9
        records.append({
            "span": (a, b), "seconds": (b - a) / float(fps), "own": own, "other": other,
            "dead": 1.0 - own - other,
            "truth": "other team" if other > 0.5 else "own team" if own > 0.5 else
                     "dead ball" if 1.0 - own - other > 0.5 else "mixed",
            "wrong": int(np.sum(seen[a:b] & alive_t[a:b] & (truth[a:b] != k))),
            "whole": whole, "start": start,
            "removed": {s: bool(np.mean(ablated[s][a:b] == k) < REMOVED) for s in SOURCES},
            "holder": holder_start(inp["touches"], a, b, k, fps),
            "held": float(np.mean(inp["holder"][a:b] == k)),
            "start_near": gap_to(a) <= near, "end_near": gap_to(b) <= near,
            "contains": bool(np.any((true_t > a + near) & (true_t < b - near)))})
    return pred, records


def _dominant(values):
    """The source with the largest positive evidence towards the spell's team, or "none"."""
    best = max(SOURCES, key=lambda s: values[s])
    return best if values[best] > 0 else "none"


def invented_report(rows, n_seeds):
    """Step 0 of the visible-ball work: the invented possessions on seen frames, their causes."""
    records = [r for row in rows for r in row.get("invented", [])]
    if not records:
        return
    groups = (("all", lambda r: True), ("other team", lambda r: r["truth"] == "other team"),
              ("own team", lambda r: r["truth"] == "own team"),
              ("dead ball", lambda r: r["truth"] == "dead ball"),
              ("mixed", lambda r: r["truth"] == "mixed"))
    wrong_total = float(max(sum(r["wrong"] for r in records), 1))
    print("\ninvented possessions mostly on seen frames (decoded, matching no true one at IoU "
          "0.5), by what truth holds under them (majority): count per seed, share of their "
          "wrong-team seen frames, length share by bin, start / end within +-{:g} s of a true "
          "transition, a true transition inside".format(NEAR_S))
    bins = list(zip(INVENTED_BINS_S[:-1], INVENTED_BINS_S[1:]))
    print("{:<11} {:>7} {:>7} {:>7}".format("truth", "n/seed", "wrong", "p50 s") + "".join(
        "{:>8}".format(bin_label(lo, hi)) for lo, hi in bins) +
        "{:>8} {:>7} {:>7}".format("start", "end", "inside"))
    for label, keep in groups:
        g = [r for r in records if keep(r)]
        if not g:
            continue
        n = float(len(g))
        print("{:<11} {:>7.1f} {:>7.3f} {:>7.1f}".format(
            label, n / n_seeds, sum(r["wrong"] for r in g) / wrong_total,
            np.median([r["seconds"] for r in g])) + "".join(
            "{:>8.3f}".format(sum(lo <= r["seconds"] < hi for r in g) / n) for lo, hi in bins) +
            "{:>8.3f} {:>7.3f} {:>7.3f}".format(np.mean([r["start_near"] for r in g]),
                                                np.mean([r["end_near"] for r in g]),
                                                np.mean([r["contains"] for r in g])))
    print("\nwhat put them there, per group. 'start' / 'held': the source with the largest "
          "evidence towards the spell's team over its first {:g} s / all of it ('none': no "
          "source points there); 'removed': the share gone (under {:.0%} of it still that team) "
          "when that source alone is switched off and the match re-decoded; median nats "
          "towards the spell's team over all of it (H and A share the duration prior and "
          "transitions cost 0, so the total over a decoded team spell is never negative)"
          .format(START_S, REMOVED))
    for label, keep in groups:
        g = [r for r in records if keep(r)]
        if not g:
            continue
        n = float(len(g))
        print("-- {} ({:.1f} per seed)".format(label, n / n_seeds))
        print("{:<9} {:>7} {:>7} {:>8} {:>10} {:>9}".format("source", "start", "held",
                                                            "removed", "nats p50", "positive"))
        for s in SOURCES + ("none",):
            if s == "none":
                print("{:<9} {:>7.3f} {:>7.3f} {:>8.3f} {:>10} {:>9}".format(
                    s, sum(_dominant(r["start"]) == s for r in g) / n,
                    sum(_dominant(r["whole"]) == s for r in g) / n,
                    sum(not any(r["removed"].values()) for r in g) / n, "--", "--"))
                continue
            print("{:<9} {:>7.3f} {:>7.3f} {:>8.3f} {:>+10.2f} {:>9.3f}".format(
                s, sum(_dominant(r["start"]) == s for r in g) / n,
                sum(_dominant(r["whole"]) == s for r in g) / n,
                sum(r["removed"][s] for r in g) / n,
                np.median([r["whole"][s] for r in g]),
                sum(r["whole"][s] > 0 for r in g) / n))
        totals = [sum(r["whole"].values()) for r in g]
        print("total nats towards their team p10/p50/p90: {:.1f} / {:.1f} / {:.1f}; removed by "
              "exactly one source {:.3f}, by more than one {:.3f}".format(
                  *np.percentile(totals, [10, 50, 90]),
                  sum(sum(r["removed"].values()) == 1 for r in g) / n,
                  sum(sum(r["removed"].values()) > 1 for r in g) / n))
        kinds = ("touch hand-over", "touch from none", "control", "held before", "other", "none")
        print("holder at the start: " + ", ".join(
            "{} {:.3f}".format(kind, sum(r["holder"] == kind for r in g) / n) for kind in kinds) +
            "; mean share of frames held by their team {:.3f}".format(
                np.mean([r["held"] for r in g])))


# -- C3 Step 0: how much could a per-frame team model add inside ball gaps? -----------------

# (label, |cue| m, team_w, lag s, flip): the capacity oracle's settings. The first is the
# realistic one (spread's p50 size at v4_spread's weight, D17) and is also split.
CEILING_CAPACITY = (("ideal 12 m, w 0.5", 12.0, 0.5, 0.0, None),
                    ("ideal 25 m, w 0.5", 25.0, 0.5, 0.0, None),
                    ("ideal 12 m, w 1", 12.0, 1.0, 0.0, None),
                    ("lag 1 s, 12 m, w 0.5", 12.0, 0.5, 1.0, None),
                    ("lag 3 s, 12 m, w 0.5", 12.0, 0.5, 3.0, None),
                    ("20% wrong 2 s runs, 12 m, w 0.5", 12.0, 0.5, 0.0, (2.0, 0.2)))


def ceiling_masks(truth, match):
    """{split name: frame mask}: every gap, gaps by length (GAP_BINS_S), and the gap frames of
    true possessions by their length (SPELL_BINS_S); with per-split counts."""
    gap_id, spans, gap_bins = gap_ids(match)
    gap = gap_id >= 0
    masks, counts = {"all gaps": gap}, {"all gaps": len(spans)}
    for lo, hi in zip(GAP_BINS_S[:-1], GAP_BINS_S[1:]):
        label = "gap " + bin_label(lo, hi)
        keep = np.array([b == bin_label(lo, hi) for b in gap_bins] + [False])
        masks[label] = gap & keep[gap_id]          # gap_id -1 reads the trailing False
        counts[label] = int(keep.sum())
    runs = pm.valid_runs(truth)
    spells = pm.possession_spells(truth, runs)
    for lo, hi in zip(SPELL_BINS_S[:-1], SPELL_BINS_S[1:]):
        label = "poss " + bin_label(lo, hi)
        mask = np.zeros(len(truth), dtype=bool)
        n = 0
        for a, b, _ in spells:
            if lo <= (b - a) / float(match.fps) < hi and gap[a:b].any():
                mask[a:b] = True
                n += 1
        masks[label] = gap & mask
        counts[label] = n
    return masks, counts


def ceiling_check(seeds, plain):
    """C3 Step 0 against v4_spread: the team oracle (true team on both-alive frames) and
    the capacity oracle (CEILING_CAPACITY[0], ideal evidence fed to the decoder), on every gap,
    by gap length and by true possession length; then every capacity setting on all gaps."""
    import zlib
    from solution.sources import idsse
    print("C3 Step 0: IDSSE dev x seeds {}, simulator v1{}; truth {}; against v4_spread"
          .format(seeds, "" if plain else " + D12 ball stress {}".format(STRESS), LABELS))
    rows = []
    for truth, match, _, seed in inputs(idsse.DEV_IDS, seeds, plain, LABELS):
        pred = METHODS["v4_spread"](match)
        runs = pm.valid_runs(truth)
        masks, counts = ceiling_masks(truth, match)
        row = {"truth": truth, "pred": pred, "seed": seed, "counts": counts,
               "gap_mask": masks["all gaps"], "team": {}, "capacity": {}}
        for key, mask in masks.items():
            row["team"][key] = team_repair(truth, pred, runs, mask)
        first = CEILING_CAPACITY[0]
        for key, mask in masks.items():
            if key != "all gaps":
                row["capacity"][(first[0], key)] = capacity_oracle(match, truth, first[1],
                                                                   first[2], mask)
        for label, size, weight, lag, flip in CEILING_CAPACITY:
            sign = ideal_sign(truth, match, lag, flip,
                              seed=zlib.crc32(match.name.encode()) + seed)
            row["capacity"][(label, "all gaps")] = capacity_oracle(match, truth, size, weight,
                                                                  sign=sign)
        rows.append(row)
    n_seeds = float(len(seeds))
    base = pooled_codes(rows, lambda r: r["pred"])
    print("v4_spread spellF1 {:.4f}".format(base))
    print("{:<16} {:>9} {:>12} {:>20}".format("split", "n/seed", "team oracle",
                                               CEILING_CAPACITY[0][0]))
    for key in rows[0]["counts"]:
        team = pooled_codes(rows, lambda r, k=key: r["team"][k]) - base
        cap = pooled_codes(rows, lambda r, k=key: r["capacity"][(CEILING_CAPACITY[0][0], k)]) \
            - base
        print("{:<16} {:>9.1f} {:>+12.4f} {:>+20.4f}".format(
            key, sum(r["counts"][key] for r in rows) / n_seeds, team, cap))
    print("(gap rows: gaps per seed; poss rows: true possessions per seed that touch a gap; "
          "the oracles act only on that split's gap frames)")
    print("\ncapacity settings on every gap: gain; recall at IoU 0.5 of true possessions "
          "mostly in a gap, by length; predicted / true possessions")
    bins = list(zip(SPELL_BINS_S[:-1], SPELL_BINS_S[1:]))
    print("{:<34} {:>9}".format("setting", "gain") + "".join(
        "{:>10}".format(bin_label(lo, hi)) for lo, hi in bins) + "{:>8}".format("ratio"))

    def describe(label, codes_of):
        gain = pooled_codes(rows, codes_of) - base
        hits = np.zeros(len(bins))
        totals = np.zeros(len(bins))
        n_pred = n_true = 0
        for row in rows:
            codes = codes_of(row)
            for seconds, in_gap, matched in spell_records(row["truth"], codes, row["gap_mask"]):
                for i, (lo, hi) in enumerate(bins):
                    if in_gap and lo <= seconds < hi:
                        totals[i] += 1
                        hits[i] += matched
            runs = pm.valid_runs(row["truth"])
            n_pred += len(pm.possession_spells(pm.fill_abstentions(codes, runs), runs))
            n_true += len(pm.possession_spells(row["truth"], runs))
        print("{:<34} {:>+9.4f}".format(label, gain) + "".join(
            "{:>10}".format("{:.3f} ({:.0f})".format(h / max(t, 1.0), t / n_seeds))
            for h, t in zip(hits, totals)) + "{:>8.3f}".format(n_pred / float(n_true)))
    describe("v4_spread as is", lambda r: r["pred"])
    describe("team oracle, every gap", lambda r: r["team"]["all gaps"])
    for label, *_ in CEILING_CAPACITY:
        describe(label, lambda r, k=(label, "all gaps"): r["capacity"][k])


def main():
    from solution.sources import idsse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--methods", nargs="+", default=["v3_kickoff"], choices=sorted(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--plain", action="store_true", help="no ball stress (plain v1)")
    parser.add_argument("--cues", action="store_true", help="E5 team cue aggregates")
    parser.add_argument("--bar-ath", action="store_true",
                        help="(needs data/) with --cues: label-free BAR-ATH, for a human to run")
    parser.add_argument("--timing", action="store_true",
                        help="E5 part 2 Step 0: hand-over timing inside gaps, duration priors")
    parser.add_argument("--capacity", action="store_true",
                        help="with --timing and v4_spread: the decoder fed an ideal cue")
    parser.add_argument("--decoded", action="store_true",
                        help="Step 2: label-free decoded aggregates inside gaps (with --bar-ath "
                             "on BAR-ATH, plus Smart Tagging agreement); default methods "
                             "v3_kickoff v4_spread")
    parser.add_argument("--events", action="store_true",
                        help="Step A: players-only hand-over features (solution/handover.py) "
                             "against true in-gap hand-overs; no decoding")
    parser.add_argument("--seen", action="store_true",
                        help="Step B: short possessions with the ball seen, oracle ceilings "
                             "and what the touch engine sees at their starts")
    parser.add_argument("--feature", default="dadvance0.5",
                        help="with --seen: the handover feature for the in-gap reach oracle")
    parser.add_argument("--rates", action="store_true",
                        help="with --events: also the label-free event rates per minute")
    parser.add_argument("--noise", type=float, default=0.0,
                        help="with --events: white position noise on the players, metres")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="with --events: positions shrunk by this factor (SPEED=0.9)")
    parser.add_argument("--bonus", action="store_true",
                        help="transition bonus Step 0: dadvance events (all / right only) and "
                             "perfect events, against v4_spread")
    parser.add_argument("--labels", choices=("dfl", "keep2", "keep2pass"), default="keep2",
                        help="truth labels: keep2, controlled possession (default, as make "
                             "proxy, D21), or dfl (first touch, sections 9 to 11)")
    parser.add_argument("--prior-labels", choices=("dfl", "keep2", "keep2pass"), default="dfl",
                        help="labels the methods' priors are fitted on: dfl, as submitted (D21)")
    parser.add_argument("--ceiling", action="store_true",
                        help="C3 Step 0: team and capacity oracles split by gap length and by "
                             "true possession length, against v4_spread")
    parser.add_argument("--tagdiag", action="store_true",
                        help="after D22: agreement inside gaps by tracking and time since the "
                             "ball was seen, for the frame model, spread and each method "
                             "(with --bar-ath against the Smart Tagging team, label-free)")
    parser.add_argument("--seen-ceiling", action="store_true",
                        help="visible-ball Step 0: team errors where the ball is seen, by kind, "
                             "and the oracle on each")
    args = parser.parse_args()
    global LABELS
    LABELS = args.labels
    # Priors are fitted lazily from idsse.load(), so this must be set before any method runs.
    idsse.LABELS = args.prior_labels
    print("truth labels {}, priors on {} labels".format(LABELS, idsse.LABELS))
    if args.ceiling:
        ceiling_check(args.seeds, args.plain)
        return
    picked = args.methods if args.methods != ["v3_kickoff"] else None
    if args.tagdiag:
        tag_check(picked or list(TAG_METHODS), args.seeds, args.plain, args.bar_ath)
        return
    if args.seen_ceiling:
        seen_ceiling_check(picked or ["v4_spread", "touch_c3_w1"], args.seeds, args.plain)
        return
    if args.bonus:
        bonus_check(args.seeds, args.plain)
        return
    if args.events:
        if args.bar_ath:
            events_check_bar_ath()
        else:
            events_check(args.seeds, args.plain, args.noise, args.speed, args.rates)
        return
    if args.decoded:
        methods = args.methods if args.methods != ["v3_kickoff"] else [
            "v3_kickoff", "v4_spread"]
        if args.bar_ath:
            decoded_check_bar_ath(methods)
        else:
            decoded_check(methods, args.seeds, args.plain)
        return
    if args.cues:
        if args.bar_ath:
            cue_check_bar_ath()
        else:
            cue_check(args.seeds, args.plain)
        return
    print("gap check: IDSSE dev x seeds {}, simulator v1{}".format(
        args.seeds, "" if args.plain else " + D12 ball stress {}".format(STRESS)))
    rows = {name: [] for name in args.methods}
    for truth, match, _, seed in inputs(idsse.DEV_IDS, args.seeds, args.plain, LABELS):
        for name in args.methods:
            pred = METHODS[name](match)
            gaps, repairs, spells, graded = analyse(truth, pred, match)
            row = {"gaps": gaps, "repairs": repairs, "spells": spells, "graded": graded,
                   "truth": truth, "pred": pred, "seed": seed}
            if args.seen:
                from solution import touch
                touches = touch.detect(match)
                row["seen_records"], row["seen"], row["claims"] = seen_analyse(
                    truth, pred, match, touches)
                row["gap_records"], row["gap"] = gap_event_analyse(truth, pred, match,
                                                                   args.feature)
                row["seen_dadv_records"], row["seen_dadv"] = gap_event_analyse(
                    truth, pred, match, args.feature, in_gap=False)
                row["seen_union_records"], row["seen_union"] = gap_event_analyse(
                    truth, pred, match, args.feature, in_gap=False, touches=touches)
                row["gap_feature"] = args.feature
            if args.timing:
                row["records"], row["snaps"] = timing_analyse(truth, pred, match)
                row["gap_mask"] = gap_ids(match)[0] >= 0
                if name == "v4_spread" and args.capacity:
                    row["capacity"] = {key: capacity_oracle(match, truth, *key)
                                       for key in CAPACITY}
            rows[name].append(row)
    for name in args.methods:
        report(name, rows[name], len(args.seeds))
        if args.seen:
            seen_report(name, rows[name], len(args.seeds), pm.FPS)
        if args.timing:
            timing_report(name, rows[name], len(args.seeds), pm.FPS)
    if args.timing:
        prior_report(rows[args.methods[0]], len(args.seeds))


if __name__ == "__main__":
    main()
