"""Visible-ball Step 3, Step 0: does a learned players-only P(dead) beat the speed rank at the
edges of stoppages? Frame level only, no decoding. Aggregates only.

    .venv/bin/python tools/deadmodel_check.py                 # D12 ball stress, seeds 0 1 2
    .venv/bin/python tools/deadmodel_check.py --plain
    .venv/bin/python tools/deadmodel_check.py --tempo 0.9      # the D9 tempo check (section 16)
    .venv/bin/python tools/deadmodel_check.py --speed 0.9      # shrunk pitch, kept for reference
    .venv/bin/python tools/deadmodel_check.py --bar-ath        # (needs data/) for a human to run
    .venv/bin/python tools/deadmodel_check.py --proxy-side     # the same on the proxy
    .venv/bin/python tools/deadmodel_check.py --ceiling -- --methods v5_frame_model \
        dmo_sign dmo_edges v6_deadball_model                 # capacity oracles, proxy league

On the IDSSE dev matches degraded as `make proxy` does, it scores solution/deadmodel.py's P(dead)
(trained without the match) against the decoder's dead-ball evidence today, 1 - the speed rank
(decoder.speed_feature, D9), on graded frames with truth D against H or A:
  - AUC by region: "start" = within EDGE_S of a true alive -> D change, "end" = within EDGE_S of
    a true D -> alive change, "interior" = the rest, and all graded frames. Inside an edge window
    both sides of the change are present, so the AUC there measures timing;
  - per region, how many matches (mean over seeds) the model's AUC beats the speed rank on;
  - edge timing at a fixed call rate: each score, smoothed over SMOOTH_S, calls the top CALL_SHARE
    of the match's graded frames dead (the middle of the IDSSE dev range, D9). For every true
    stoppage of at least MIN_STOP_S: the nearest called alive -> dead change within LAG_REACH_S of
    its start, and dead -> alive of its end; p50 |lag| and the share within 1 s.
The bar for decoder work, fixed before the first run: pooled AUC at least BAR above the speed
rank at both starts and ends, and higher on at least 4 of the 5 matches in both. It is applied
to each model version (deadmodel.DROPPED: v1 the D25 model, v2 without absolute positions), and
must hold under the D12 stress and under --tempo 0.9 (EXPERIMENTS section 16).

--ceiling registers the capacity oracles and runs tools/proxy_league.py with the arguments after
`--`. Each is v6_deadball_model (deadm_w 1) with the model's log-ratio r replaced:
  dmo_sign:  |r| with the truth's sign (+ on D, - on H or A): perfect knowledge at the model's
             own strength, frame by frame;
  dmo_edges: r, with the truth's sign only within EDGE_S of a true change into or out of D.
They read match.labels: proxy only, never a submission. The perfect-dead-ball repair of
tools/deadball_check.py is the ceiling above both.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import deadmodel, decoder  # noqa: E402
from solution.framemodel import _window_mean  # noqa: E402
from solution.match import pm  # noqa: E402
from tools.deadball_check import STRESS, inputs  # noqa: E402

EDGE_S = 2.0
SMOOTH_S = 1.0
CALL_SHARE = 0.45
MIN_STOP_S = 2.0
LAG_REACH_S = 5.0
BAR = 0.03
REGIONS = ("start", "end", "interior", "all")


def auc(score, dead):
    """Mann-Whitney AUC of `score` for dead (True) against alive; NaN if a class is empty."""
    from sklearn.metrics import roc_auc_score
    if dead.all() or not dead.any():
        return float("nan")
    return float(roc_auc_score(dead, score))


def regions(truth, runs, fps):
    """{region: mask} over the graded frames (truth H, A or D inside the halves)."""
    graded = np.isin(truth, (pm.H, pm.A, pm.D))
    pad = int(round(EDGE_S * fps))
    start = np.zeros(len(truth), dtype=bool)
    end = np.zeros(len(truth), dtype=bool)
    for f, before, after in pm.transitions(truth, runs):
        if after == pm.D and before != pm.D:
            start[max(0, f - pad):f + pad] = True
        elif before == pm.D and after != pm.D:
            end[max(0, f - pad):f + pad] = True
    return {"start": graded & start, "end": graded & end,
            "interior": graded & ~start & ~end, "all": graded}


def called(score, graded, fps):
    """Dead calls: the smoothed score's top CALL_SHARE of graded frames."""
    w = int(round(SMOOTH_S * fps))
    smooth = _window_mean(score, -(w // 2), w - w // 2, cut=False)
    threshold = np.quantile(smooth[graded & np.isfinite(smooth)], 1.0 - CALL_SHARE)
    return graded & (smooth >= threshold)


def edge_lags(truth, runs, calls, fps):
    """([start lags], [end lags]) in frames, called change minus true change; None if none."""
    reach = int(round(LAG_REACH_S * fps))
    on = np.flatnonzero(calls[1:] & ~calls[:-1]) + 1
    off = np.flatnonzero(~calls[1:] & calls[:-1]) + 1
    starts, ends = [], []
    for a, b, code in pm.segments(truth, runs):
        if code != pm.D or (b - a) < MIN_STOP_S * fps:
            continue
        for frames, f, out in ((on, a, starts), (off, b, ends)):
            near = frames[np.abs(frames - f) <= reach]
            out.append(int(near[np.argmin(np.abs(near - f))] - f) if len(near) else None)
    return starts, ends


# -- Step 2: label-free aggregates of the model itself, for BAR-ATH ------------------------------

SHIFT_FEATURES = ("tracked", "gyration", "touchline_min", "goalline_min", "in_box",
                  "beyond_past3", "speed_cv")


def model_aggregates(match):
    """Label-free numbers about deadmodel v1 on one match, over the frames inside the halves:
    the share it leans dead (log-ratio > 0), its log-ratio p10 / p50 / p90, its Spearman
    correlation with the slowness rank, the agreement of its 45% dead calls with the speed
    rank's (both unsmoothed), and the medians (means for counts of beyond / box) of the
    position features it reads, where BAR-ATH's tracking could differ from the simulator's."""
    inside = match.in_play_mask()
    r = deadmodel.log_ratio(match)
    slow = 1.0 - decoder.speed_feature(match, 2, "rank")
    use = inside & np.isfinite(r) & np.isfinite(slow)
    a, b = r[use], slow[use]
    rank_a = np.argsort(np.argsort(a)) / float(len(a))
    rank_b = np.argsort(np.argsort(b)) / float(len(b))
    calls_a = rank_a >= 1.0 - CALL_SHARE
    calls_b = rank_b >= 1.0 - CALL_SHARE
    X, names = deadmodel.features(match)
    out = {"lean dead": float(np.mean(a > 0)),
           "ratio p10": float(np.percentile(a, 10)), "ratio p50": float(np.percentile(a, 50)),
           "ratio p90": float(np.percentile(a, 90)),
           "spearman vs slow": float(np.corrcoef(rank_a, rank_b)[0, 1]),
           "calls agree": float(np.mean(calls_a == calls_b))}
    for name in SHIFT_FEATURES:
        values = X[inside, names.index(name)]
        values = values[np.isfinite(values)]
        mean = name in ("in_box", "beyond_past3")
        out[("mean " if mean else "p50 ") + name] = float(np.mean(values) if mean
                                                           else np.median(values))
    return out


def print_aggregates(title, rows):
    print("\n" + title)
    keys = list(rows[0])
    for key in keys:
        values = np.array([row[key] for row in rows])
        print("  {:<22} {:>8.3f}".format(key, np.mean(values)) +
              ("   (range over match-seeds {:.3f}..{:.3f})".format(values.min(), values.max())
               if len(values) > 1 else ""))


def bar_ath_aggregates():
    from solution.sources import atd
    print_aggregates("BAR-ATH (no labels): deadmodel v1 aggregates", [model_aggregates(atd.load())])


def proxy_aggregates(seeds, plain, tempo):
    from solution.sources import idsse
    rows = [model_aggregates(match)
            for _, match, _, _ in inputs(idsse.DEV_IDS, seeds, plain, "keep2", tempo)]
    print_aggregates("IDSSE dev x seeds {}{}{}: deadmodel v1 aggregates (no labels read)".format(
        seeds, "" if plain else ", D12 stress", ", tempo x{:g}".format(tempo) if tempo != 1
        else ""), rows)


def oracle_ratio(match, edges_only):
    """The capacity oracles' log-ratio: the model's size, the truth's sign (proxy only)."""
    r = deadmodel.log_ratio(match)
    truth = match.labels
    signed = np.where(truth == pm.D, np.abs(r), -np.abs(r))
    if not edges_only:
        return signed
    masks = regions(truth, pm.valid_runs(truth), match.fps)
    edge = masks["start"] | masks["end"]
    return np.where(edge, signed, r)


def register_oracles():
    from solution.methods import _GAPTEAM, METHODS, semi_markov
    for name, edges_only in (("dmo_sign", False), ("dmo_edges", True)):
        METHODS[name] = semi_markov(name, frame_w=1.0, frame_where="seen", deadm_w=1.0,
                                    deadm_fn=lambda m, e=edges_only: oracle_ratio(m, e),
                                    **_GAPTEAM)


def main():
    from solution.sources import idsse
    if "--ceiling" in sys.argv:
        at = sys.argv.index("--ceiling")
        rest = sys.argv[at + 1:]
        rest = rest[1:] if rest[:1] == ["--"] else rest
        register_oracles()
        from tools import proxy_league
        sys.argv = ["proxy_league.py"] + rest
        proxy_league.main()
        return
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--plain", action="store_true", help="no ball stress (plain v1)")
    parser.add_argument("--noise", type=float, default=0.0,
                        help="white position noise on the players, metres (NOISE=0.1)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="positions shrunk by this factor (simulate.scale_speeds, SPEED=0.9)")
    parser.add_argument("--tempo", type=float, default=1.0,
                        help="clean match replayed this many times as fast before degrading "
                             "(simulate.retime, TEMPO=0.9): the D9 tempo check, section 16")
    parser.add_argument("--bar-ath", action="store_true",
                        help="(needs data/) label-free aggregates of the model on BAR-ATH")
    parser.add_argument("--proxy-side", action="store_true",
                        help="the --bar-ath aggregates on the proxy, for comparison")
    args = parser.parse_args()
    if args.bar_ath:
        bar_ath_aggregates()
        return
    if args.proxy_side:
        proxy_aggregates(args.seeds, args.plain, args.tempo)
        return
    from solution.simulate import add_position_noise, scale_speeds
    print("Step 3 / Step 0: IDSSE dev x seeds {}, simulator v1{}{}{}{}; P(dead) vs speed rank"
          .format(args.seeds, "" if args.plain else " + D12 ball stress {}".format(STRESS),
                  ", position noise {:g} m".format(args.noise) if args.noise else "",
                  ", speed x{:g}".format(args.speed) if args.speed != 1 else "",
                  ", tempo x{:g} (retimed)".format(args.tempo) if args.tempo != 1 else ""))
    # "speed" (1 - the rank, as the decoder reads it per frame) decides; the smoothed ranks are
    # reference rows only: the decoder integrates speed over whole segments, so the model's
    # context features may add less there than against a single frame.
    models = tuple("model " + v for v in sorted(deadmodel.DROPPED))
    scorers = models + ("speed", "speed 1 s", "speed 3 s")
    pooled = {(s, r): ([], []) for s in scorers for r in REGIONS}
    per_match = {}
    lags = {s: {"start": [], "end": []} for s in scorers}
    for truth, match, _, seed in inputs(idsse.DEV_IDS, args.seeds, args.plain, "keep2",
                                        args.tempo):
        match = add_position_noise(scale_speeds(match, args.speed), args.noise, seed=seed)
        runs = pm.valid_runs(truth)
        masks = regions(truth, runs, match.fps)
        slow = 1.0 - decoder.speed_feature(match, 2, "rank")
        scores = {"model " + v: deadmodel.dead_probability(match, version=v)
                  for v in deadmodel.DROPPED}
        scores["speed"] = slow
        for s in (1.0, 3.0):
            w = int(round(s * match.fps))
            scores["speed {:g} s".format(s)] = _window_mean(slow, -(w // 2), w - w // 2,
                                                            cut=False)
        dead = truth == pm.D
        for s, score in scores.items():
            for r, mask in masks.items():
                use = mask & np.isfinite(score)
                pooled[(s, r)][0].append(score[use])
                pooled[(s, r)][1].append(dead[use])
                per_match.setdefault((match.name, s, r), []).append(auc(score[use], dead[use]))
            starts, ends = edge_lags(truth, runs, called(score, masks["all"], match.fps),
                                     match.fps)
            lags[s]["start"] += starts
            lags[s]["end"] += ends
    names = sorted(set(k[0] for k in per_match))
    print("\nAUC of dead against alive, pooled over matches and seeds; 'wins': matches where the "
          "model's mean AUC over seeds beats the speed rank's ({} matches)".format(len(names)))
    print("{:<9} {:<9} {:>10} {:>9} {:>9} {:>9} {:>6} {:>10} {:>10}".format(
        "model", "region", "frames", "model", "speed", "gain", "wins", "speed 1 s", "speed 3 s"))
    verdict = {m: [] for m in models}
    for model in models:
        for r in REGIONS:
            a = {s: auc(np.concatenate(pooled[(s, r)][0]), np.concatenate(pooled[(s, r)][1]))
                 for s in (model, "speed", "speed 1 s", "speed 3 s")}
            wins = sum(np.nanmean(per_match[(m, model, r)]) >
                       np.nanmean(per_match[(m, "speed", r)]) for m in names)
            n = sum(len(v) for v in pooled[(model, r)][0])
            print("{:<9} {:<9} {:>10d} {:>9.4f} {:>9.4f} {:>+9.4f} {:>4d}/{} {:>10.4f} {:>10.4f}"
                  .format(model[6:], r, n, a[model], a["speed"], a[model] - a["speed"], wins,
                          len(names), a["speed 1 s"], a["speed 3 s"]))
            if r in ("start", "end"):
                verdict[model].append(a[model] - a["speed"] >= BAR and wins >= 4)
    fps = pm.FPS
    print("\nedge timing at a {:.0%} dead call rate (score smoothed over {:g} s), true stoppages "
          ">= {:g} s: found within +-{:g} s, p50 |lag| s, share within 1 s, p50 signed lag s"
          .format(CALL_SHARE, SMOOTH_S, MIN_STOP_S, LAG_REACH_S))
    print("{:<9} {:<6} {:>7} {:>7} {:>8} {:>9} {:>9}".format("scorer", "edge", "n", "found",
                                                              "|lag|", "<=1 s", "lag"))
    for s in scorers:
        for edge in ("start", "end"):
            values = lags[s][edge]
            found = np.array([v for v in values if v is not None], dtype=float) / fps
            print("{:<9} {:<6} {:>7d} {:>7.3f} {:>8.2f} {:>9.3f} {:>+9.2f}".format(
                s, edge, len(values), len(found) / max(len(values), 1),
                np.median(np.abs(found)) if len(found) else float("nan"),
                np.mean(np.abs(found) <= 1.0) if len(found) else float("nan"),
                np.median(found) if len(found) else float("nan")))
    for model in models:
        print("\nbar for decoder work (fixed before the first run), {}: AUC gain >= {:g} and wins "
              ">= 4 at both starts and ends -> {}".format(
                  model, BAR, "CLEARS" if all(verdict[model]) else "does not clear"))


if __name__ == "__main__":
    main()
