"""Step 0 of the whistle veto: can "no whistle" remove false stoppages in mid-pitch? Proxy only.

    .venv/bin/python tools/whistle_veto_check.py                    # v7_keep2_control, D12, keep2
    .venv/bin/python tools/whistle_veto_check.py --methods v7_keep2_control dm_stack3_dead4
    .venv/bin/python tools/whistle_veto_check.py --labels dfl

The idea (denis, 2026-11-12): away from the lines almost every stoppage is whistled (fouls,
offsides, injuries, dropped balls, goals); throw-ins, corners and goal kicks happen at a line and
are not. So a stoppage the method decodes in mid-pitch with no whistle near it is likely a player
holding the ball, and should be live. D26 closed the whistle as a cue that PLACES whistled
stoppages (+0.012 to +0.015); this measures it as a VETO on decoded ones, which goes after the
largest loss (false stoppages, +0.046 on v5, EXPERIMENTS section 16).

On the IDSSE dev matches degraded as the proxy league does (simulator v1, D12 ball stress unless
--plain), per method:
  - whistles: the start frame of every true D stretch whose restart is WHISTLED
    (tools/deadball_check: free kicks incl. fouls and offsides, penalties, goals, dropped balls),
    a perfect detector. Then each whistle dropped with probability `miss` (a detector's misses),
    and FAN_RATES extra whistles per minute at random frames (its false alarms).
  - veto: a decoded D stretch is released (its frames take the team of the nearest frame the
    method labelled H or A, as deadball_check's repairs do) when its ball sits at least M metres
    from every line and no whistle falls in [start - BEFORE_S, stop]. The ball: the median of
    its seen positions inside the stretch, else its last seen position up to LOOKBACK_S before;
    a stretch with neither is kept. M = 0 vetoes regardless of place (shows what the place test
    is for).
  - gain: pooled spellF1 exactly as the proxy league pools it, mean over seeds, against the
    method as it is; "false-stoppage oracle" is deadball_check's repair of every false stoppage
    (the ceiling of anything that only removes them).
Also: decoded stretches vetoed per seed, how many touch no true D frame (false stoppages), and
the released frames that were truly live against truly dead.

Settings fixed before the first run (2026-11-12): BEFORE_S 5, LOOKBACK_S 10, M in
(0, 5, 10, 15, 20), miss in (0, 0.1, 0.2, 0.3, 0.5), FAN_RATES (0, 0.5)/min, seeds 0-4. The
decision this informs: a perfect-whistle gain under D4's 0.02 closes the idea; otherwise the
miss rate where the gain falls under 0.02 is what a detector must beat.
Aggregates only; no BAR-ATH data is read.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution.match import pm  # noqa: E402
from solution.methods import METHODS  # noqa: E402
from tools.deadball_check import (WHISTLED, analyse, dead_stretches, inputs,  # noqa: E402
                                  repair, restart_kinds)
from tools.proxy_league import f1_from_flags, spell_flags  # noqa: E402

BEFORE_S = 5.0          # a whistle up to this long before a decoded stoppage starts covers it
LOOKBACK_S = 10.0       # no ball seen inside: its last seen position up to this long before
MARGINS_M = (0.0, 5.0, 10.0, 15.0, 20.0)
MISS = (0.0, 0.1, 0.2, 0.3, 0.5)
FAN_RATES = (0.0, 0.5)  # extra whistles per minute of play
SEEDS = (0, 1, 2, 3, 4)


def whistle_frames(truth, events, fps):
    """Start frames of the true D stretches a referee whistles (a perfect detector)."""
    runs = pm.valid_runs(truth)
    true_d = dead_stretches(truth, runs)
    kinds = restart_kinds(true_d, events, fps)
    return np.array(sorted(a for (a, _), k in zip(true_d, kinds) if k in WHISTLED), dtype=int)


def detector(whistles, in_play, fps, miss, fan_rate, rng):
    """The perfect whistles with each dropped at probability `miss`, plus fan whistles at
    `fan_rate` per minute of play at uniformly random in-play frames."""
    kept = whistles[rng.random_sample(len(whistles)) >= miss]
    frames = np.flatnonzero(in_play)
    n_fan = rng.poisson(fan_rate * len(frames) / float(fps) / 60.0)
    fans = rng.choice(frames, n_fan, replace=False) if n_fan else np.array([], dtype=int)
    return np.sort(np.concatenate((kept, fans)).astype(int))


def stretch_margin(match, a, b, lookback):
    """Distance in metres from the stretch's ball to the nearest line, or None if unknown."""
    seen = match.ball_seen
    inside = np.flatnonzero(seen[a:b])
    if len(inside):
        x = float(np.median(match.ball_x[a:b][inside]))
        y = float(np.median(match.ball_y[a:b][inside]))
    else:
        lo = max(0, a - lookback)
        before = np.flatnonzero(seen[lo:a])
        if not len(before):
            return None
        k = lo + before[-1]
        x, y = float(match.ball_x[k]), float(match.ball_y[k])
    length, width = match.pitch
    return min(x, length - x, y, width - y)


def veto_mask(stretches, margins, whistles, n, margin_m, before):
    """Frames of the decoded D stretches the veto releases."""
    out = np.zeros(n, dtype=bool)
    for (a, b), m in zip(stretches, margins):
        if m is None or m < margin_m:
            continue
        lo, hi = np.searchsorted(whistles, a - before), np.searchsorted(whistles, b)
        if hi > lo:
            continue
        out[a:b] = True
    return out


def pooled(rows, key):
    """Mean over seeds of the pooled spellF1 of rows[...][key] (as the proxy league pools)."""
    scores = []
    for seed in sorted(set(r["seed"] for r in rows)):
        parts, n_pred = [], 0
        for r in rows:
            if r["seed"] != seed:
                continue
            flags, n = spell_flags(r["truth"], r[key])
            parts.append(flags)
            n_pred += n
        scores.append(f1_from_flags(np.concatenate(parts, axis=1), n_pred))
    return float(np.mean(scores)), scores


def variants():
    for margin in MARGINS_M:
        yield ("perfect", margin, 0.0, 0.0)
    for miss in MISS[1:]:
        yield ("miss", 10.0, miss, 0.0)
    for fan in FAN_RATES[1:]:
        for miss in (0.0, 0.3):
            yield ("fans", 10.0, miss, fan)


def name_of(margin, miss, fan):
    return "M {:>2.0f} m, miss {:.1f}, fans {:.1f}/min".format(margin, miss, fan)


def run(method, seeds, plain, labels):
    from solution.sources import idsse
    rows = []
    for truth, match, events, seed in inputs(idsse.DEV_IDS, seeds, plain, labels):
        pred = METHODS[method](match)
        runs = pm.valid_runs(truth)
        filled = pm.fill_abstentions(pred, runs)
        decoded = dead_stretches(filled, runs)
        lookback = int(round(LOOKBACK_S * match.fps))
        margins = [stretch_margin(match, a, b, lookback) for a, b in decoded]
        whistles = whistle_frames(truth, events, match.fps)
        _, false, repairs = analyse(truth, pred, events, match.fps)
        false_set = set(false)
        row = {"seed": seed, "truth": truth, "as is": pred, "oracle": repairs["false"],
               "fps": match.fps, "n_decoded": len(decoded), "n_false": len(false),
               "n_whistles": len(whistles), "minutes": match.in_play_mask().sum() / match.fps / 60.0,
               "unknown_place": sum(m is None for m in margins), "stats": {}}
        before = int(round(BEFORE_S * match.fps))
        for k, (_, margin, miss, fan) in enumerate(variants()):
            rng = np.random.RandomState(1000 * seed + 17 * k + len(truth) % 997)
            heard = detector(whistles, match.in_play_mask(), match.fps, miss, fan, rng)
            release = veto_mask(decoded, margins, heard, len(truth), margin * 1.0, before)
            key = name_of(margin, miss, fan)
            row[key] = repair(truth, pred, runs, np.zeros(len(truth), dtype=bool), release)
            vetoed = [(a, b) for a, b in decoded if release[a]]
            row["stats"][key] = {
                "vetoed": len(vetoed),
                "vetoed_false": sum(s in false_set for s in vetoed),
                "live_frames": int(np.sum(release & (truth != pm.D) & (filled == pm.D))),
                "dead_frames": int(np.sum(release & (truth == pm.D)))}
        # the place of false stoppages against decoded stoppages that touch true dead ball
        row["false_margins"] = [m for s, m in zip(decoded, margins) if s in false_set]
        row["true_margins"] = [m for s, m in zip(decoded, margins) if s not in false_set]
        rows.append(row)
    return rows


def report(method, rows, n_seeds, labels, plain):
    print("\n== {} ==  IDSSE dev x seeds {}, {} truth, {}".format(
        method, sorted(set(r["seed"] for r in rows)), labels,
        "plain v1" if plain else "v1 + D12 ball stress"))
    per = lambda key: sum(r[key] for r in rows) / float(n_seeds)
    print("per seed: {:.1f} decoded stoppages, {:.1f} false (touch no true D), {:.1f} whistled "
          "true stoppages ({:.2f}/min), {:.1f} decoded with no ball place".format(
              per("n_decoded"), per("n_false"), per("n_whistles"),
              sum(r["n_whistles"] for r in rows) / sum(r["minutes"] for r in rows),
              per("unknown_place")))
    for label, key in (("false", "false_margins"), ("touching true D", "true_margins")):
        values = np.array([m for r in rows for m in r[key] if m is not None])
        print("  ball's distance to the nearest line, decoded stoppages {:<16}: n/seed {:.1f}; "
              "share >= 5 / 10 / 15 / 20 m: {}".format(
                  label, len(values) / float(n_seeds),
                  " / ".join("{:.2f}".format(np.mean(values >= m)) for m in MARGINS_M[1:])))
    base, base_seeds = pooled(rows, "as is")
    oracle, _ = pooled(rows, "oracle")
    print("as is {:.4f}; false-stoppage oracle {:+.4f}".format(base, oracle - base))
    print("  {:<34} | gain    | wins | vetoed/seed | of them false | released live s/seed | "
          "released dead s/seed".format("variant"))
    for _, margin, miss, fan in variants():
        key = name_of(margin, miss, fan)
        score, seeds = pooled(rows, key)
        wins = sum(s > b for s, b in zip(seeds, base_seeds))
        stats = [r["stats"][key] for r in rows]
        vetoed = sum(s["vetoed"] for s in stats)
        fps = rows[0]["fps"]
        print("  {:<34} | {:+.4f} | {}/{}  | {:11.1f} | {:13.2f} | {:20.0f} | {:20.0f}".format(
            key, score - base, wins, len(seeds), vetoed / float(n_seeds),
            sum(s["vetoed_false"] for s in stats) / float(max(vetoed, 1)),
            sum(s["live_frames"] for s in stats) / float(fps * n_seeds),
            sum(s["dead_frames"] for s in stats) / float(fps * n_seeds)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--methods", nargs="+", default=["v7_keep2_control"],
                        choices=sorted(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--plain", action="store_true", help="no ball stress (plain v1)")
    parser.add_argument("--labels", default="keep2", help="truth: keep2 (D21) or dfl")
    args = parser.parse_args()
    print("whistle veto Step 0: before {} s, lookback {} s".format(BEFORE_S, LOOKBACK_S))
    for method in args.methods:
        rows = run(method, args.seeds, args.plain, args.labels)
        report(method, rows, len(args.seeds), args.labels, args.plain)


if __name__ == "__main__":
    main()
