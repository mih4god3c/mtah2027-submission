"""Step 0 of the self-calibrating whistle detector on BAR-ATH (solution/whistle.py, D47).

    .venv/bin/python tools/whistle_selfcal_check.py            # (needs data/) tactical audio + tracking
    .venv/bin/python tools/whistle_selfcal_check.py --cross    # also the broadcast (make broadcast-align)
    make whistle-selfcal [ARGS="--cross"]

Everything comes out of code: the kickoffs from the tracking, the whistle's pitch from the audio
around them, the checks from the tracking and the decoder. Nobody notes, picks or judges a moment
of the match, and this tool prints no frame, no time and no listing -- aggregates only, so its
whole output may be shared.

The first run decodes the tactical audio (a few minutes) and caches the per-frame peaks in
cache/whistle/ (gitignored: derived from match data); the BAR-ATH decode for E takes a few
minutes more.

Prints, with the pass criteria fixed in D47 before the first run:
  A. Anchors: kickoffs whistle.kickoff_anchors finds from the tracking, per half (counts).
  B. Calibration: kind (pair / single), tone pitches, anchors with the signature, its share in
     the chance windows and the ratio.                       Criterion 1: a calibration exists.
  C. Leave one anchor out: of the re-calibrations, how many give the same kind and every tone
     within CAL_BIN_HZ.             Criterion 2: all of them (with at least 3 anchors; with 2,
                                    reported as not testable).
  D. Whistles per minute of each half.                      Criterion 3: 0.2 to 1.5 in each half.
  E. Timing against the tracking: the share of whistle onsets (grouped as whistlecue does) with
     the start of a v7_keep2_control decoded stoppage within STOP_WINDOW_S of it, against the
     same share with the onsets shifted by CHANCE_SHIFTS_S (chance).
                                                    Criterion 4: at least 2 x chance in each half.
  F. --cross (context, no criterion): the broadcast audio through the same calibrated bands, its
     whistles mapped to the tactical timeline and paired with the tactical ones, against the
     chance rate with the broadcast shifted by CROSS_SHIFTS_S.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import games  # noqa: E402
from solution import whistle  # noqa: E402
from solution.match import CACHE_DIR, pm  # noqa: E402

GAME = games.GAME
STOP_WINDOW_S = (-2.0, 6.0)                   # a decoded stoppage starting this close to a whistle
CHANCE_SHIFTS_S = (-60.0, -30.0, 30.0, 60.0)  # onsets moved this far keep only chance agreement
RATE_RANGE = (0.2, 1.5)                       # criterion 3, whistles per minute
CHANCE_FACTOR = 2.0                           # criterion 4
CROSS_SHIFTS_S = (-20.0, -10.0, 10.0, 20.0)   # no real cross-feed pairs at these offsets
ALIGN_PATH = os.path.join(CACHE_DIR, "broadcast", GAME + "_align.npz")


def cross_counts(tactical, broadcast, fps, shifts_s=CROSS_SHIFTS_S):
    """tactical: whistles on the tactical timeline; broadcast: whistles already map_events()ed
    onto it. Returns (lag frames, lag pair count, pairs at the lag, pair counts at each shift)."""
    tac = [dict(e, strict=True) for e in tactical]
    bro = [dict(e, strict=True) for e in sorted(broadcast, key=lambda e: e["start"])]
    lag, lags = whistle.feed_lag(tac, bro, fps)
    pairs = whistle.pair_feeds(tac, bro, fps, lag)
    chance = [len(whistle.pair_feeds(tac, bro, fps, lag, shift=int(round(s * fps))))
              for s in shifts_s]
    return lag, len(lags), pairs, chance


def same_calibration(a, b):
    return (a is not None and b is not None and a["kind"] == b["kind"]
            and all(abs(x - y) <= whistle.CAL_BIN_HZ for x, y in zip(a["tones"], b["tones"])))


def stop_share(onsets, stop_starts, lo, hi, shift, fps):
    """Share of onsets (shifted by `shift` frames, kept inside [lo, hi)) with a stoppage start
    within STOP_WINDOW_S; None if no shifted onset stays in the half."""
    moved = onsets + shift
    moved = moved[(moved >= lo) & (moved < hi)]
    if not len(moved):
        return None
    a = np.searchsorted(stop_starts, moved + int(round(STOP_WINDOW_S[0] * fps)))
    b = np.searchsorted(stop_starts, moved + int(round(STOP_WINDOW_S[1] * fps)), side="right")
    return float(np.mean(b > a))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cross", action="store_true",
                        help="also the broadcast audio (needs make broadcast-align)")
    args = parser.parse_args()

    from solution.methods import METHODS
    from solution.sources import atd
    from solution.whistlecue import group_onsets

    path = games.video_path(GAME)
    if not os.path.isfile(path):
        sys.exit("no tactical video at {} (set MTAH_DATA_DIR)".format(path))
    match = atd.load()
    fps = match.fps
    features = whistle.peak_video_features(path, match.n_frames, fps)
    graded = match.in_play_mask()
    verdict = {}

    anchors = whistle.kickoff_anchors(match)
    print("{} self-calibrating whistle detector, Step 0 (D47); aggregates only".format(GAME))
    print("\nA. kickoffs found from the tracking: {} ({})".format(len(anchors), ", ".join(
        "half {}: {}".format(h + 1, sum(a <= k < b for k in anchors))
        for h, (a, b) in enumerate(match.halves))))

    cal = whistle.calibrate(features, anchors, graded, fps)
    if cal is None:
        print("\nB. calibration: none (no signature at {} or more anchors at {} x chance)".format(
            whistle.CAL_MIN_ANCHORS, whistle.CAL_MIN_RATIO))
        print("\nCriterion 1 fails: Step 0 fails; the detector finds no whistles on this match.")
        return
    print("\nB. calibration: {} tone(s) at {} Hz; at {} of {} anchors; in {:.2f} of {} chance "
          "windows; ratio {:.1f}".format(
              cal["kind"], " + ".join("{:.0f}".format(t) for t in cal["tones"]),
              cal["anchors_hit"], cal["n_anchors"], cal["chance"], cal["n_chance_windows"],
              cal["ratio"]))
    verdict[1] = True

    if len(anchors) >= 3:
        same = sum(same_calibration(cal, whistle.calibrate(
            features, anchors[:k] + anchors[k + 1:], graded, fps)) for k in range(len(anchors)))
        print("\nC. leave one anchor out: {} of {} re-calibrations agree".format(same, len(anchors)))
        verdict[2] = same == len(anchors)
    else:
        print("\nC. leave one anchor out: not testable with {} anchors".format(len(anchors)))

    found = [e for e in whistle.whistle_events(features, cal, fps) if e["whistle"]]
    onsets = group_onsets(found, fps)
    print("\nD. whistles per minute of each half (criterion 3: {} to {}):".format(*RATE_RANGE))
    rates = []
    for h, (a, b) in enumerate(match.halves):
        n = sum(a <= e["start"] < b for e in found)
        g = int(np.sum((onsets >= a) & (onsets < b)))
        rate = n / ((b - a) / float(fps) / 60.0)
        rates.append(rate)
        print("   half {}: {:.2f} whistles/min ({} whistles, {} grouped onsets)".format(
            h + 1, rate, n, g))
    verdict[3] = all(RATE_RANGE[0] <= r <= RATE_RANGE[1] for r in rates)

    codes = METHODS["v7_keep2_control"](match)
    starts = np.flatnonzero((codes == pm.D) & (np.concatenate(([pm.X], codes[:-1])) != pm.D))
    print("\nE. onsets with a decoded stoppage starting {} s to {} s from them (v7_keep2_control):"
          .format(*STOP_WINDOW_S))
    ok4 = True
    for h, (a, b) in enumerate(match.halves):
        inside = onsets[(onsets >= a) & (onsets < b)]
        real = stop_share(inside, starts, a, b, 0, fps)
        chance = [stop_share(inside, starts, a, b, int(round(s * fps)), fps)
                  for s in CHANCE_SHIFTS_S]
        chance = [c for c in chance if c is not None]
        null = float(np.mean(chance)) if chance else float("nan")
        passed = real is not None and np.isfinite(null) and real >= CHANCE_FACTOR * null
        ok4 &= bool(passed)
        print("   half {}: {} of onsets, chance {} (shifts {} s); {}".format(
            h + 1, "n/a" if real is None else "{:.2f}".format(real),
            "n/a" if not np.isfinite(null) else "{:.2f}".format(null), list(CHANCE_SHIFTS_S),
            "x{:.1f}".format(real / null) if real is not None and null > 0 else "-"))
    verdict[4] = ok4

    if args.cross:
        cross_report(cal, found, fps)

    print("\nVerdict (D47): " + ", ".join(
        "criterion {} {}".format(k, "pass" if v else "FAIL") for k, v in sorted(verdict.items())))
    print("Step 0 {}.".format("passes" if all(verdict.values()) else "fails"))


def cross_report(cal, tactical, fps):
    """F: the broadcast through the same calibrated bands, against the tactical whistles."""
    if not os.path.isfile(ALIGN_PATH):
        print("\nF. --cross: no broadcast alignment at {}; run `make broadcast-align`".format(
            os.path.relpath(ALIGN_PATH, ROOT)))
        return
    align = np.load(ALIGN_PATH, allow_pickle=False)
    live_t, bro_frame = align["live_t"], align["bro_frame"]
    n_tac, n_bro = games.source_frames("tactical"), games.source_frames("broadcast")
    path = games.video_path(GAME, source="broadcast")
    if not os.path.isfile(path) or len(live_t) != n_bro or len(bro_frame) != n_tac:
        print("\nF. --cross: broadcast video or alignment cache missing / stale")
        return
    feats = whistle.peak_video_features(path, n_bro, fps)
    bro_found = [e for e in whistle.whistle_events(feats, cal, fps) if e["whistle"]]
    bro, dropped = whistle.map_events(bro_found, live_t)
    bro = [e for e in bro if e["start"] < n_tac]
    tac = [e for e in tactical if bro_frame[e["start"]] >= 0]
    lag, n_lag, pairs, chance = cross_counts(tac, bro, fps)
    null = float(np.mean(chance))
    print("\nF. cross-feed (context): broadcast whistles {} ({} mapped, {} unaligned); tactical "
          "whistles where the broadcast airs {}".format(len(bro_found), len(bro), dropped, len(tac)))
    print("   lag {} frames from {} pairs; paired {} ({:.0%} of aired tactical, {:.0%} of mapped "
          "broadcast); chance {} (mean {:.1f}), real / chance {}".format(
              lag, n_lag, len(pairs), len(pairs) / max(1, len(tac)),
              len(pairs) / max(1, len(bro)), chance, null,
              "{:.1f}".format(len(pairs) / null) if null > 0 else "inf"))


if __name__ == "__main__":
    main()
