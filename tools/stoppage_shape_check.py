"""Step 0 for denis's video review (2026-11-11): keeper holds, goal kicks and team shape. Aggregates only.

    .venv/bin/python tools/stoppage_shape_check.py                    # v7_keep2_control, D12 stress
    .venv/bin/python tools/stoppage_shape_check.py --methods v6_deadball_model v7_keep2_control

denis watched the whole BAR-ATH match against the decoded blocks and reported, among others:
  (a) the keeper holding the ball (alive, the keeper's team) is decoded D, while a goal kick
      (dead from the ball going out) is first decoded as the restarting team, D only later;
  (b) celebrations, the drinks break and the walk-offs are decoded as possession while the
      players stand far from the middle of the pitch; a team's centre normally stays in a
      "belt" along the pitch, off it only at stoppages.
This tool measures both on the proxy (IDSSE dev, simulator v1 + D12 ball stress, keep2 truth),
before anything is built:

Part A, keeper holds and goal kicks.
  - A keeper hold is defined on the CLEAN match from the truth: team T has the ball (keep2), the
    clean ball is within HOLD_M of T's keeper (its deepest player on average over the half),
    who is inside his own penalty area; runs
    with gaps up to GAP_S, kept when at least MIN_HOLD_S long. Checked against the DFL keeper
    events (BallClaiming BallHeld, saved shots held): the share of those within
    EVENT_S of a hold.
  - Per hold: share of its frames decoded D ("inverted"), the right team, the other team.
  - Per true D stretch by restart type (deadball_check.restart_kinds): the share decoded as a
    team before the first decoded D ("team first"), and decoded possessions lying wholly inside
    the stretch ("invented").
  - Oracle ceilings, pooled spellF1 as the proxy league pools it: "hold release" gives the hold
    frames decoded D the nearest decoded team; "hold release+" does it for the whole predicted
    D stretches that touch a hold, on their truly alive frames.
Part B, team shape (on the degraded match, as the methods see it), swap-invariant per frame:
  lat_max   largest |team centre y - W/2| of the two teams (metres; >= MIN_TEAM tracked)
  lat_all   |centre y of all tracked players - W/2|
  gyr_min   smaller team radius of gyration (a huddle is small)
  beyond    share of tracked players more than 0.5 m beyond a touchline
  Percentiles by class: true alive, true D, and true D that the method decodes alive ("missed D",
  the frames a cue would have to fix). Then each feature's within-match rank (the "belt" read
  from the match itself, D9) held for SUSTAIN_S: the share of each class it flags, and, as an
  indicator only (not a decision), the pooled spellF1 change if those frames were forced D.
"""

import argparse
import os
import sys
import warnings
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution.gapteam import attack_sign  # noqa: E402
from solution.match import AWAY, HOME, pm  # noqa: E402
from solution.methods import METHODS  # noqa: E402
from solution.simulate import add_ball_noise, degrade  # noqa: E402
from tools.deadball_check import (KINDS, STRESS, dead_stretches, repair,  # noqa: E402
                                  restart_kinds)
from tools.proxy_league import f1_from_flags, spell_flags  # noqa: E402

HOLD_M = 2.0
GAP_S = 0.4
MIN_HOLD_S = 1.0
EVENT_S = 3.0
BOX_DEPTH, BOX_HALF_WIDTH = 16.5, 20.16
MIN_TEAM = 4
SUSTAIN_S = 3.0
QUANTILES = (0.95, 0.98, 0.99)
FEATURES = ("lat_max", "lat_all", "gyr_min", "beyond")
SMALL_IS_ODD = {"gyr_min"}          # for these a LOW rank is the unusual end


def runs_of(mask):
    """[(start, stop)] of True runs."""
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[::2], edges[1::2]))


def keeper_holds(raw, truth):
    """[(start, stop, team)] keeper holds on the clean match (see the docstring)."""
    fps, (length, width) = raw.fps, raw.pitch
    sign = attack_sign(raw)
    out = []
    for team, code in ((HOME, pm.H), (AWAY, pm.A)):
        cols = np.flatnonzero(raw.team == team)
        x = raw.player_x[:, cols].astype(np.float64)
        y = raw.player_y[:, cols].astype(np.float64)
        # depth from the team's own goal line: home defends x = 0 where sign is +1
        own_at_zero = (sign >= 0) if team == HOME else (sign < 0)
        depth = np.where(own_at_zero[:, None], x, length - x)
        known = np.isfinite(depth) & np.isfinite(y)
        # the keeper: the team's deepest player on average over each half (an identity, so a
        # defender dropping behind him for a moment does not take his place)
        k = np.zeros(len(depth), dtype=np.int64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for start, stop in raw.halves:
                mean = np.nanmean(depth[start:stop], axis=0)
                k[start:stop] = int(np.nanargmin(np.where(np.isfinite(mean), mean, np.inf)))
        rows = np.arange(len(k))
        known = known[rows, k]
        kd, kx, ky = depth[rows, k], x[rows, k], y[rows, k]
        dist = np.hypot(raw.ball_x - kx, raw.ball_y - ky)
        with np.errstate(invalid="ignore"):
            mask = ((truth == code) & known & (dist < HOLD_M) & (kd < BOX_DEPTH) &
                    (np.abs(ky - width / 2.0) < BOX_HALF_WIDTH) & (sign != 0))
        # close short gaps, then keep long enough runs
        gap = int(round(GAP_S * fps))
        spans = runs_of(mask)
        merged = []
        for a, b in spans:
            if merged and a - merged[-1][1] <= gap:
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        out += [(a, b, code) for a, b in merged if b - a >= MIN_HOLD_S * fps]
    return sorted(out)


def keeper_events(match_id):
    """Frames of DFL keeper-hold events: BallClaiming BallHeld, saves held (BallClaimed is any
    player winning the ball, so it is left out)."""
    from solution.sources import idsse
    frames = []
    for frame, event in idsse._events(match_id):
        child = next(iter(event), None)
        if child is None:
            continue
        if child.tag == "BallClaiming" and child.get("Type") == "BallHeld":
            frames.append(frame)
        elif child.tag == "ShotAtGoal" and any(
                g.tag == "SavedShot" and g.get("SaveResult") == "held" for g in child):
            frames.append(frame)
    return frames


def shape_features(match):
    """{name: (T,) float} team-shape features, unchanged by a team swap."""
    width = match.pitch[1]
    x, y = match.player_x.astype(np.float64), match.player_y.astype(np.float64)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        lats, gyrs = [], []
        for team in (HOME, AWAY):
            cols = match.team == team
            tx, ty = x[:, cols], y[:, cols]
            n = np.sum(np.isfinite(tx) & np.isfinite(ty), axis=1)
            cx, cy = np.nanmean(tx, axis=1), np.nanmean(ty, axis=1)
            ok = n >= MIN_TEAM
            lats.append(np.where(ok, np.abs(cy - width / 2.0), np.nan))
            gyrs.append(np.where(ok, np.sqrt(np.nanmean((tx - cx[:, None]) ** 2 +
                                                        (ty - cy[:, None]) ** 2, axis=1)), np.nan))
        out["lat_max"] = np.fmax(lats[0], lats[1])
        out["gyr_min"] = np.fmin(gyrs[0], gyrs[1])
        n_all = np.sum(np.isfinite(x) & np.isfinite(y), axis=1)
        cy_all = np.nanmean(y, axis=1)
        out["lat_all"] = np.where(n_all >= 2 * MIN_TEAM, np.abs(cy_all - width / 2.0), np.nan)
        beyond = (y < -0.5) | (y > width + 0.5)
        out["beyond"] = np.where(n_all >= 2 * MIN_TEAM,
                                 np.sum(beyond, axis=1) / np.maximum(n_all, 1), np.nan)
    return out


def within_rank(values, mask):
    """values as their quantile among the finite `mask` frames (deadmodel._rank)."""
    out = np.full(len(values), np.nan)
    use = mask & np.isfinite(values)
    if use.any():
        ordered = np.sort(values[use])
        out[use] = np.searchsorted(ordered, values[use], side="right") / float(len(ordered))
    return out


def sustained(flag, frames):
    """True where `flag` holds over a whole window of `frames` containing the frame."""
    f = flag.astype(np.int32)
    c = np.concatenate([[0], np.cumsum(f)])
    full = np.zeros(len(f), dtype=bool)
    if len(f) >= frames:
        whole = (c[frames:] - c[:-frames]) == frames          # window starting at i is all True
        starts = np.flatnonzero(whole)
        diff = np.zeros(len(f) + 1, dtype=np.int32)
        diff[starts] += 1
        diff[starts + frames] -= 1
        full = np.cumsum(diff)[:-1] > 0
    return full


def pooled(rows, key):
    per_seed = []
    for seed in sorted(set(r["seed"] for r in rows)):
        parts, n_pred = [], 0
        for r in rows:
            if r["seed"] == seed:
                flags, n = spell_flags(r["truth"], r["variants"].get(key, r["pred"]))
                parts.append(flags)
                n_pred += n
        per_seed.append(f1_from_flags(np.concatenate(parts, axis=1), n_pred))
    return float(np.mean(per_seed)), per_seed


def analyse(raw, truth, match, pred, events, kevents):
    fps = match.fps
    inside = match.in_play_mask()
    runs = pm.valid_runs(truth)
    filled = pm.fill_abstentions(pred, runs)
    alive_t = (truth == pm.H) | (truth == pm.A)
    n = len(truth)
    row = {"variants": {}}

    # -- Part A: keeper holds
    holds = keeper_holds(raw, truth)
    hold_mask = np.zeros(n, dtype=bool)
    per_hold = []
    for a, b, code in holds:
        hold_mask[a:b] = True
        seg = filled[a:b]
        per_hold.append({"s": (b - a) / float(fps), "d": float(np.mean(seg == pm.D)),
                         "right": float(np.mean(seg == code)),
                         "other": float(np.mean((seg != code) & (seg != pm.D)))})
    row["holds"] = per_hold
    starts = np.array([a for a, _, _ in holds])
    stops = np.array([b for _, b, _ in holds])
    tol = EVENT_S * fps
    row["kevents"] = len(kevents)
    row["kevents_hit"] = sum(1 for f in kevents if len(holds) and
                             np.any((starts - tol <= f) & (f <= stops + tol)))
    row["variants"]["hold release"] = repair(truth, pred, runs, np.zeros(n, dtype=bool),
                                             hold_mask & alive_t)
    pred_d = dead_stretches(filled, runs)
    touch = np.zeros(n, dtype=bool)
    for a, b in pred_d:
        if hold_mask[a:b].any():
            touch[a:b] = True
    row["variants"]["hold release+"] = repair(truth, pred, runs, np.zeros(n, dtype=bool),
                                              touch & alive_t)
    row["variants"]["hold true team"] = np.where(hold_mask, truth, filled).astype(filled.dtype)

    # -- Part A: team first / invented, by restart type
    true_d = dead_stretches(truth, runs)
    kinds = restart_kinds(true_d, events, fps)
    per_stop = []
    for (a, b), kind in zip(true_d, kinds):
        seg = filled[a:b]
        dead = np.flatnonzero(seg == pm.D)
        first = dead[0] if len(dead) else len(seg)
        restarter = truth[b] if b < n and truth[b] in (pm.H, pm.A) else -1
        lead = seg[:first]
        # decoded possessions wholly inside the stretch
        invented = 0
        for s, e, k in pm.segments(filled, runs):
            if k in (pm.H, pm.A) and s >= a and e <= b:
                invented += 1
        per_stop.append({"kind": kind, "s": (b - a) / float(fps),
                         "team_first_s": first / float(fps) if len(dead) else np.nan,
                         "lead_restarter": float(np.mean(lead == restarter)) if len(lead) else
                         np.nan,
                         "covered": float(np.mean(seg == pm.D)), "invented": invented})
    row["stops"] = per_stop
    gk = np.zeros(n, dtype=bool)
    for (a, b), kind in zip(true_d, kinds):
        if kind == "goal_kick":
            gk[a:b] = True
    near = np.zeros(n, dtype=bool)
    for a, b in pred_d:
        if gk[a:b].any():
            near[a:b] = True
    row["variants"]["goal kick D"] = repair(truth, pred, runs, gk, near & ~(truth == pm.D))
    row["variants"]["goal kick + hold"] = repair(truth, row["variants"]["goal kick D"], runs,
                                                 np.zeros(n, dtype=bool), touch & alive_t)

    # -- Part B: team shape
    feats = shape_features(match)
    dead_t = truth == pm.D
    missed = dead_t & (filled != pm.D)
    classes = {"alive": alive_t & inside, "D": dead_t & inside, "missed D": missed & inside}
    row["feat"] = {name: {c: v[m & np.isfinite(v)] for c, m in classes.items()}
                   for name, v in feats.items()}
    flags = {}
    w = int(round(SUSTAIN_S * fps))
    for name, v in feats.items():
        r = within_rank(v, inside)
        if name in SMALL_IS_ODD:
            r = 1.0 - r
        for q in QUANTILES:
            f = sustained(np.nan_to_num(r, nan=0.0) >= q, w) & inside
            flags[(name, q)] = {c: (int(np.sum(f & m)), int(np.sum(m))) for c, m in
                                classes.items()}
            forced = filled.copy()
            forced[f] = pm.D
            row["variants"]["force D {} q{:g}".format(name, q)] = forced
    row["flags"] = flags
    row.update(truth=truth, pred=pred)
    return row


def report(name, rows, n_seeds):
    print("\n== {} ==  ({} seeds)".format(name, n_seeds))
    holds = [h for r in rows for h in r["holds"]]
    ke = sum(r["kevents"] for r in rows)
    kh = sum(r["kevents_hit"] for r in rows)
    print("Part A. keeper holds: {:.1f} per match-seed, length s p50 {:.1f} (p10 {:.1f}, p90 "
          "{:.1f}); DFL keeper events within {:g} s of a hold: {}/{} ({:.2f})".format(
              len(holds) / float(len(rows)), *np.percentile([h["s"] for h in holds], [50, 10, 90]),
              EVENT_S, kh, ke, kh / float(max(ke, 1))))
    frames = np.array([h["s"] for h in holds])
    for key in ("d", "right", "other"):
        vals = np.array([h[key] for h in holds])
        print("  hold frames decoded {:<6} {:.3f} (frame-weighted)   holds with any: {:.3f}   "
              "with over half: {:.3f}".format(key, np.sum(vals * frames) / frames.sum(),
                                               np.mean(vals > 0), np.mean(vals > 0.5)))
    stops = [s for r in rows for s in r["stops"]]
    print("  true D stretches by restart type: n/match-seed, median s, decoded D covered,")
    print("  'team first' = decoded as a team before the first decoded D (share of stretches, "
          "median s when it is), lead frames on the restarting team, invented possessions/stretch")
    by = defaultdict(list)
    for s in stops:
        by[s["kind"]].append(s)
    for kind in KINDS + ("all",):
        group = stops if kind == "all" else by.get(kind, [])
        if not group:
            continue
        tf = np.array([s["team_first_s"] for s in group])
        has = tf[np.isfinite(tf) & (tf > 0)]
        lead = np.array([s["lead_restarter"] for s in group], dtype=float)
        print("  {:<13} {:>5.1f} {:>6.1f} s  covered {:.3f}  team first {:.3f} (p50 {:>4.1f} s) "
              " lead on restarter {:.3f}  invented {:.2f}".format(
                  kind, len(group) / float(len(rows)), np.median([s["s"] for s in group]),
                  np.mean([s["covered"] for s in group]), len(has) / float(len(group)),
                  np.median(has) if len(has) else np.nan,
                  np.nanmean(lead) if np.isfinite(lead).any() else np.nan,
                  np.mean([s["invented"] for s in group])))

    print("Part B. team shape, percentiles p50 / p90 / p99 by class (degraded players):")
    for feat in FEATURES:
        parts = []
        for c in ("alive", "D", "missed D"):
            v = np.concatenate([r["feat"][feat][c] for r in rows])
            parts.append("{} {:.2f}/{:.2f}/{:.2f}".format(c, *np.percentile(v, [50, 90, 99])))
        print("  {:<8} {}".format(feat, "   ".join(parts)))
    print("  within-match rank >= q held {:g} s: share of each class flagged".format(SUSTAIN_S))
    for feat in FEATURES:
        for q in QUANTILES:
            tot = defaultdict(lambda: [0, 0])
            for r in rows:
                for c, (k, m) in r["flags"][(feat, q)].items():
                    tot[c][0] += k
                    tot[c][1] += m
            flagged = sum(tot[c][0] for c in ("alive", "D"))
            print("  {:<8} q{:<5g} alive {:.4f}  D {:.4f}  missed D {:.4f}  precision (D among "
                  "flagged) {:.3f}".format(feat, q, *(tot[c][0] / float(max(tot[c][1], 1))
                                                       for c in ("alive", "D", "missed D")),
                                           tot["D"][0] / float(max(flagged, 1))))

    base, base_seeds = pooled(rows, "as is")
    print("pooled spellF1 (as is {:.4f}); gain, seeds won:".format(base))
    keys = sorted(set(k for r in rows for k in r["variants"]))
    for key in keys:
        score, seeds = pooled(rows, key)
        wins = sum(s > b for s, b in zip(seeds, base_seeds))
        tag = "oracle" if not key.startswith("force") else "rule  "
        print("  {} {:<26} {:.4f}  {:+.4f}  {}/{}".format(tag, key, score, score - base, wins,
                                                        len(seeds)))


def main():
    from solution.sources import idsse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--methods", nargs="+", default=["v7_keep2_control"],
                        choices=sorted(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--labels", default="keep2")
    parser.add_argument("--matches", nargs="+", default=None, help="a subset of the dev matches")
    args = parser.parse_args()
    print("stoppage shape check: IDSSE dev x seeds {}, simulator v1 + D12 {}, {} truth".format(
        args.seeds, STRESS, args.labels))
    rows = {m: [] for m in args.methods}
    for match_id in args.matches or idsse.DEV_IDS:
        assert match_id in idsse.DEV_IDS, "dev matches only (D3)"
        raw = idsse.load(match_id, labels=args.labels)
        events = idsse.restarts(match_id)
        kevents = keeper_events(match_id)
        for seed in args.seeds:
            match = add_ball_noise(degrade(raw, seed), *STRESS[:2], seed=seed,
                                   smooth_s=STRESS[2])
            for m in args.methods:
                row = analyse(raw, raw.labels, match, METHODS[m](match), events, kevents)
                row["seed"] = seed
                rows[m].append(row)
        print("  done", match_id, flush=True)
    for m in args.methods:
        report(m, rows[m], len(args.seeds))


if __name__ == "__main__":
    main()
