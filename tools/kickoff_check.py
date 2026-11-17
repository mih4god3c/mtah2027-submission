"""Where does the kickoff rule fire? Aggregates only (solution/kickoff.py, PLAN finding 9).

    .venv/bin/python tools/kickoff_check.py              # IDSSE dev, plain and with a 117 s pre-kickoff
    .venv/bin/python tools/kickoff_check.py --profile busy   # players mill about until late
    .venv/bin/python tools/kickoff_check.py --grid       # the same for a small grid of settings
    .venv/bin/python tools/kickoff_check.py --bar-ath    # (needs data/) BAR-ATH, for a human to run
    .venv/bin/python tools/kickoff_check.py --diagnose [--bar-ath] [--profile busy]   # why it misses

The proxy league cannot show this rule working: IDSSE halves start at the kick. So this prints,
on IDSSE dev degraded by simulator v1 with the D12 ball stress:
  - plain halves: how often the rule fires at all. It should never fire there (a false kickoff
    costs a whole possession or more), and `kickoff-like` counts the formation holds
    anywhere in a half that pass the quiet and rise cues, so it shows what the guards hold back
    (kickoffs after goals, mostly); the guard is what holds them back;
  - with simulate.add_prekickoff in front of each second half: whether the kickoff is found, the
    error against the true kick in seconds, and how much of the pre-kickoff stretch v2_touch
    decodes as D without the rule.
With --bar-ath, for each half: the detected kickoff as seconds from the half's first graded
frame and, before it, v2_touch's decoded D share and possession count, the share of frames
with the ball seen and the median collective speed; then how many candidates passed each cue.
The synthetic pre-kickoff prints the same aggregates, so the two can be compared and the
profile calibrated. Nothing frame-level is printed.
"""

import argparse
import itertools
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import kickoff  # noqa: E402
from solution.match import pm  # noqa: E402
from solution.methods import METHODS  # noqa: E402
from solution.simulate import (PREKICKOFF_PROFILES, add_ball_noise,  # noqa: E402
                               add_prekickoff, degrade, mean_player_speed)

STRESS = (0.05, 3.4, 0.26)          # D12: sd m, jumps per visible minute, smoothing s
GRID = {"own_share": (0.7, 0.9), "hold_s": (5.0, 12.0), "back_s": (1.0, 3.0)}


def kickoff_like(match, params):
    """Formation holds anywhere in each half that pass quiet and rise: no window, no guard."""
    wide = dict(params, search_s=1e9)
    form = kickoff.formation(match, wide)
    speed = mean_player_speed(match)
    count = 0
    for start, stop in match.halves:
        for a, b in kickoff.holds(start, stop, form, match.fps, wide):
            k = kickoff.kick_frame(a, b, speed, match.fps, wide)
            result = kickoff.checks(k, start, stop, speed, match.fps, wide)
            count += result["quiet"] and result["rise"]
    return count


def inputs(match_ids, seeds, prekickoff, profile):
    from solution.sources import idsse
    for match_id in match_ids:
        raw = idsse.load(match_id)
        for seed in seeds:
            clean = add_prekickoff(raw, prekickoff, seed, profile=profile)
            yield add_ball_noise(degrade(clean, seed), *STRESS[:2], seed=seed, smooth_s=STRESS[2])


def stretch(match, codes, start, kick):
    """Aggregates of the frames before a kickoff: what the rule turns into dead ball."""
    if kick <= start:
        return None
    part = codes[start:kick]
    alive = np.isin(part, (pm.H, pm.A)).astype(np.int8)
    speed = mean_player_speed(match)[start:kick]
    return {"seconds": (kick - start) / float(match.fps),
            "d_share": float(np.mean(part == pm.D)),
            "possessions": int(alive[0] + np.sum(np.diff(alive) == 1)),
            "ball_seen": float(np.mean(match.ball_seen[start:kick])),
            "speed": float(np.nanmedian(speed)) if np.isfinite(speed).any() else float("nan")}


def describe(row):
    return ("{seconds:.1f} s; v2_touch D share {d_share:.3f} and {possessions} possessions "
            "there, ball seen {ball_seen:.3f}, median collective speed {speed:.2f} m/s"
            .format(**row))


def idsse_rows(params, seeds, prekickoff, profile, decode=True):
    from solution.sources import idsse
    rows = []
    for match in inputs(idsse.DEV_IDS, seeds, prekickoff, profile):
        kicks = kickoff.detect(match, params)
        info = match.notes.get("prekickoff")
        row = {"halves": len(kicks), "like": kickoff_like(match, params), "false": 0,
               "found": None, "error_s": None, "before": None}
        for i, kick in enumerate(kicks):
            if info and i == info["half"]:
                row["found"] = kick is not None
                if kick is not None:
                    row["error_s"] = (kick - info["kick"]) / float(match.fps)
            elif kick is not None:
                row["false"] += 1
        if info and decode:
            row["before"] = stretch(match, METHODS["v2_touch"](match),
                                    match.halves[info["half"]][0], info["kick"])
        rows.append(row)
    return rows


def summary(title, rows):
    halves = sum(r["halves"] for r in rows)
    false = sum(r["false"] for r in rows)
    like = sum(r["like"] for r in rows)
    print("{} false kickoffs {}/{} halves, kickoff-like {}".format(title, false, halves, like))
    found = [r for r in rows if r["found"] is not None]
    if found:
        errors = np.array([r["error_s"] for r in found if r["found"]])
        line = "    found {}/{}".format(len(errors), len(found))
        if len(errors):
            line += ", error s median {:+.2f}, range {:+.2f}..{:+.2f}".format(
                np.median(errors), errors.min(), errors.max())
        print(line)
        before = [r["before"] for r in found if r["before"]]
        if before:
            mean = {key: float(np.mean([b[key] for b in before])) for key in before[0]}
            mean["possessions"] = int(round(mean["possessions"]))
            print("    true pre-kickoff (mean): " + describe(mean))


def _longest(mask):
    """Longest run of True, in elements."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    runs = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    return int(runs.max()) if len(runs) else 0


def window_profile(match, start, end, params):
    """Aggregates of what the cues read over frames [start, end): percentiles and shares only."""
    fps = float(match.fps)
    distance = kickoff.ball_to_spot(match)[start:end]
    seen = np.isfinite(distance)
    sighted = distance[seen]
    out = {"seconds": (end - start) / fps, "ball_seen": float(np.mean(seen))}
    if len(sighted):
        out["ball_to_spot_p"] = np.percentile(sighted, (1, 5, 50))
        for radius in (2, 3, 5):
            near = sighted <= radius
            out["within_{}m".format(radius)] = float(np.mean(near))
            # Longest stretch of consecutive sightings this close, gaps skipped, in seconds.
            out["rest_{}m_s".format(radius)] = _longest(near) / fps
    x = match.player_x[start:end].astype(np.float64) - match.pitch[0] / 2.0
    sides, counts, shares = [], [], []
    for team in (0, 1):
        xs = x[:, match.team == team]
        visible = np.isfinite(xs)
        count = visible.sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            side = np.sign(np.nanmedian(xs, axis=1))
        with np.errstate(invalid="ignore"):
            inside = visible & (xs * side[:, None] >= -params["line_m"])
        counts.append(count)
        shares.append(inside.sum(axis=1) / np.maximum(count, 1))
        sides.append(side)
    out["tracked_p"] = [np.percentile(c, (5, 50, 95)) for c in counts]
    out["enough_players"] = float(np.mean((counts[0] >= params["min_players"]) &
                                          (counts[1] >= params["min_players"])))
    out["opposite_sides"] = float(np.mean(sides[0] == -sides[1]))
    out["own_share_p50"] = [float(np.median(v[c > 0])) if np.any(c > 0) else float("nan")
                            for v, c in zip(shares, counts)]
    out["formation"] = float(np.mean(kickoff.formation(match, params)[start:end]))
    speed = mean_player_speed(match)[start:end]
    out["speed_p"] = (np.nanpercentile(speed, (10, 50, 90)) if np.isfinite(speed).any()
                      else np.full(3, np.nan))
    out.update(formation_hold(match, start, end, params))
    return out


def formation_hold(match, start, end, params):
    """The longest stretch with both teams in their own halves (1 s majority), as aggregates:
    its length, where it ends, play speed in it and just after, and the ball during it."""
    fps = match.fps
    form = kickoff.formation(match, params)[start:end].astype(float)
    window = np.ones(fps) / fps
    held = np.convolve(form, window, mode="same") >= 0.5
    edges = np.diff(np.concatenate(([0], held.astype(np.int8), [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    out = {"holds_2s": int(np.sum(stops - starts >= 2 * fps))}
    if not len(starts):
        return out
    i = int(np.argmax(stops - starts))
    a, b = start + starts[i], start + stops[i]
    speed = mean_player_speed(match)
    distance = kickoff.ball_to_spot(match)[a:b]
    sighted = distance[np.isfinite(distance)]
    five = 5 * fps
    out.update({
        "hold_s": (b - a) / float(fps), "hold_end_s": (b - start) / float(fps),
        "hold_speed": float(np.nanmedian(speed[a:b])),
        "speed_before_end": float(np.nanmean(speed[max(a, b - five):b])),
        "speed_after_end": float(np.nanmean(speed[b:min(end, b + five)])) if b < end
        else float("nan"),
        "hold_ball_seen": float(np.mean(np.isfinite(distance))),
        "hold_ball_p": (np.percentile(sighted, (10, 50)) if len(sighted)
                        else np.full(2, np.nan))})
    return out


def print_profile(title, row):
    print(title)
    print("    {:.0f} s, ball seen {:.3f}".format(row["seconds"], row["ball_seen"]))
    if "ball_to_spot_p" in row:
        print("    ball to centre mark m p1/p5/p50 {}; seen within 2/3/5 m {:.3f}/{:.3f}/{:.3f}; "
              "longest run of sightings within 2/3/5 m {:.1f}/{:.1f}/{:.1f} s".format(
                  "/".join("{:.1f}".format(v) for v in row["ball_to_spot_p"]),
                  row["within_2m"], row["within_3m"], row["within_5m"],
                  row["rest_2m_s"], row["rest_3m_s"], row["rest_5m_s"]))
    print("    tracked per team p5/p50/p95 home {} away {}; both >= min_players {:.3f}".format(
        *["/".join("{:.0f}".format(v) for v in p) for p in row["tracked_p"]],
        row["enough_players"]))
    print("    teams on opposite sides {:.3f}; own-half share p50 home {:.2f} away {:.2f}; "
          "formation holds {:.3f}".format(row["opposite_sides"], *row["own_share_p50"],
                                          row["formation"]))
    print("    collective speed m/s p10/p50/p90 {}".format(
        "/".join("{:.2f}".format(v) for v in row["speed_p"])))
    if "hold_s" in row:
        print("    formation holds >= 2 s: {:.0f}; longest {:.1f} s, ending at {:.1f} s; speed in "
              "it {:.2f}, last 5 s {:.2f}, next 5 s {:.2f} m/s; ball seen {:.3f}, to the mark "
              "p10/p50 {}".format(
                  row["holds_2s"], row["hold_s"], row["hold_end_s"], row["hold_speed"],
                  row["speed_before_end"], row["speed_after_end"], row["hold_ball_seen"],
                  "/".join("{:.1f}".format(v) for v in row["hold_ball_p"])))


def diagnose(params, bar, seeds, prekickoff, profile):
    """Window profiles of each half's first search_s: BAR-ATH, or the synthetic halves."""
    reach = int(round(params["search_s"]))
    if bar:
        from solution.sources import atd
        match = atd.load()
        print("BAR-ATH, first {} s of each half".format(reach))
        for h, (start, stop) in enumerate(match.halves):
            end = min(stop, start + reach * match.fps)
            print_profile("half {}".format(h + 1), window_profile(match, start, end, params))
        return
    from solution.sources import idsse
    print("IDSSE dev x seeds {}, v1 + ball stress, {:.0f} s {} pre-kickoff; first {} s of the "
          "second half, then the true pre-kickoff alone (means over halves)".format(
              seeds, prekickoff, profile, reach))
    for which in ("window", "pre-kickoff"):
        rows = []
        for match in inputs(idsse.DEV_IDS, seeds, prekickoff, profile):
            info = match.notes["prekickoff"]
            start = match.halves[info["half"]][0]
            end = (min(match.halves[info["half"]][1], start + reach * match.fps)
                   if which == "window" else info["kick"])
            rows.append(window_profile(match, start, end, params))
        mean = {key: np.mean([r[key] for r in rows], axis=0) for key in rows[0]}
        mean["tracked_p"] = list(mean["tracked_p"])
        mean["own_share_p50"] = list(mean["own_share_p50"])
        print_profile(which, mean)


def bar_ath(params):
    from solution.sources import atd
    match = atd.load()
    codes = METHODS["v2_touch"](match)
    print("BAR-ATH, kickoff rule at kickoff.PARAMS; seconds from each half's first graded frame")
    for h, ((start, stop), (counts, kick)) in enumerate(zip(match.halves,
                                                            kickoff.explain(match, params))):
        cues = ", ".join("{} {}".format(key, counts[key]) for key in counts)
        if kick is None:
            print("half {}: no kickoff found in the first {:.0f} s".format(
                h + 1, params["search_s"]))
        else:
            row = stretch(match, codes, start, kick)
            print("half {}: kickoff at {}".format(h + 1, describe(row) if row else "0.0 s"))
        print("    candidates passing each cue: " + cues)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bar-ath", action="store_true",
                        help="(needs data/) BAR-ATH aggregates, for a human to run")
    parser.add_argument("--grid", action="store_true", help="a small grid of settings")
    parser.add_argument("--diagnose", action="store_true",
                        help="what the cues read over each half's first search_s: ball distance "
                             "to the mark, tracked players, formation, speed (aggregates)")
    parser.add_argument("--prekickoff", type=float, default=117.0, metavar="S")
    parser.add_argument("--profile", choices=sorted(PREKICKOFF_PROFILES), default="calm")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    if args.diagnose:
        diagnose(kickoff.PARAMS, args.bar_ath, args.seeds, args.prekickoff, args.profile)
        return
    if args.bar_ath:
        bar_ath(kickoff.PARAMS)
        return
    settings = [{}]
    if args.grid:
        settings = [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())]
    print("IDSSE dev x seeds {}, simulator v1 + ball stress {}".format(args.seeds, STRESS))
    for change in settings:
        params = dict(kickoff.PARAMS, **change)
        print(", ".join("{} {}".format(k, v) for k, v in change.items()) or "kickoff.PARAMS")
        summary("  plain:", idsse_rows(params, args.seeds, 0, args.profile, decode=False))
        summary("  {:.0f} s {} pre-kickoff:".format(args.prekickoff, args.profile),
                idsse_rows(params, args.seeds, args.prekickoff, args.profile,
                           decode=not args.grid))
    if args.grid:
        print("{} settings tried".format(len(settings)))


if __name__ == "__main__":
    main()
