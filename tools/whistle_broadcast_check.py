"""Whistle Step 0 with both audio tracks (D44): a whistle heard in the tactical AND the broadcast
audio. Aggregates first; listings only on request.

    make broadcast-align                                         # once: writes the alignment cache
    .venv/bin/python tools/whistle_broadcast_check.py            # summary + end of each half
    .venv/bin/python tools/whistle_broadcast_check.py --spot 10  # every 10th paired whistle
    .venv/bin/python tools/whistle_broadcast_check.py --around FRAME [FRAME ...]
    make whistle-broadcast [ARGS="--spot 10"]

Why two tracks. On the tactical audio alone the detector (solution/whistle.py, STRICT and LOOSE
fixed before any BAR-ATH run) found the half-time and kickoff whistles, but fired ~4.5 times a
minute, mostly on crowd whistling at 2.3-2.9 kHz (D33, EXPERIMENTS section 20). The broadcast
mix carries the referee and commentary. Commentary
puts harmonics into the whistle band, so the broadcast alone is not safe either
(tests/test_whistle_broadcast.py). A tone in both, at the same moment and pitch, is the cue.

How. Features of both tracks (cached in cache/whistle/, gitignored, derived from match data).
Broadcast events go to the tactical timeline through broadcast.align's live_t (the live moment
each broadcast frame airs during: replays and cut-aways carry live sound). whistle.feed_lag
estimates the audio lag between the feeds from the events themselves; whistle.pair_feeds pairs
them. The chance rate is the same pairing with the broadcast shifted by SHIFTS_S: pairs there
can only be coincidences, so every BOTH rate is printed beside it.

Printed:
  A. Both audio streams (codec, rate, channels, duration) and the share of graded frames the
     broadcast covers. Rates below are over the graded frames the broadcast covers, for all rows.
  B. Per half, events per minute: tactical STRICT / LOOSE, broadcast STRICT / LOOSE, BOTH, and
     the chance rate. Then the pitch of STRICT events per feed and of BOTH, in 0.5 kHz bins:
     whether the 2.3-2.9 kHz crowd cluster is missing from the broadcast is denis's hypothesis.
  C. The lag (pairs, median, p10 / p90).
  D. Kickoffs: the BOTH events in the 10 s before each kick (D33's criterion 4).
  E. End of each half: D33's candidate pick rule unchanged (tools/whistle_check.end_pick: the
     last group of blasts with gaps <= 3 s, at least 45:00 after the half's first graded frame),
     applied to BOTH events, and the BOTH events in the half's last --tail seconds.
  F. --spot N: every Nth BOTH event in the graded halves, no picking, with both feeds' frame and
     time. --around: BOTH and per-feed events near given frames.
The listings print video times of audio events, for the person running this to check against
the video; share the summary lines (A to C, the pick's offsets), not the listings.

Step 0 passes, fixed before the first run (D44), when both hold:
  1. Of `--spot 10`'s list (every 10th BOTH event, no picking), at least 0.8 are the referee's
     whistle.
  2. The BOTH rate is at least 3 times its chance rate in each half.
  Context, reported: kickoff whistles found (D); the end-of-half picks (E); the pitch tables (B).
If both pass, Step 1 is D33's rule (dead ball from the pick to the half's last graded frame,
nothing when no pick exists), adopted under D14 like D33. A failure closes the route again.
Nothing here is a label, and no setting is to be tuned on what this prints.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import games  # noqa: E402
from solution import whistle  # noqa: E402
from solution.match import CACHE_DIR  # noqa: E402
from tools.whistle_check import clock, end_pick  # noqa: E402

GAME = "BAR-ATH"
FPS = games.GAMES[GAME]["fps"]
ALIGN_PATH = os.path.join(CACHE_DIR, "broadcast", GAME + "_align.npz")
SHIFTS_S = (-20.0, -10.0, 10.0, 20.0)     # chance pairing: no real whistle pairs at these lags
HZ_BINS = (2000, 2500, 3000, 3500, 4000, 4500, 5001)
KICK_WINDOW_S = (10.0, 1.0)               # a kickoff whistle sits up to 10 s before the kick


def stream_info(path):
    """'codec, rate Hz, channels, duration s' of a file's first audio track (metadata only)."""
    import av
    with av.open(path) as container:
        if not container.streams.audio:
            return "no audio track"
        audio = container.streams.audio[0]
        seconds = (float(audio.duration * audio.time_base) if audio.duration is not None
                   else float(container.duration) / 1e6 if container.duration else float("nan"))
        return "{}, {} Hz, {} channel(s), {:.1f} s".format(
            audio.codec_context.name, audio.codec_context.sample_rate,
            audio.codec_context.channels, seconds)


def per_minute(events, covered, start, end):
    """Events starting on a covered frame of [start, end], per covered minute."""
    n = sum(1 for e in events if start <= e["start"] <= end and covered[e["start"]])
    minutes = covered[start:end + 1].sum() / float(FPS) / 60.0
    return n / minutes if minutes else float("nan"), n


def in_halves(events, covered=None):
    out = []
    for e in events:
        if any(s <= e["start"] <= t for _, s, t in games.half_video_blocks(GAME)):
            if covered is None or covered[e["start"]]:
                out.append(e)
    return out


def pitch_row(events):
    hz = np.array([e["hz"] for e in events])
    counts, _ = np.histogram(hz, bins=HZ_BINS)
    total = max(1, len(hz))
    return " | ".join("{:5.1%}".format(c / float(total)) for c in counts) + " | {:5d}".format(len(hz))


def bclock(frame):
    return "-" if frame is None or frame < 0 else clock(frame)


HEADER = ("    tactical frame | tactical time | broadcast frame | broadcast time | s before | "
          "length s | Hz tac | Hz bro | strict tac/bro")


def print_pairs(pairs, ref=None):
    print(HEADER)
    for p in pairs:
        before = "" if ref is None else "{:8.1f}".format((ref - p["start"]) / float(FPS))
        print("    {:14d} | {} | {:15d} | {}  | {:>8} | {:8.2f} | {:6.0f} | {:6.0f} | {}/{}".format(
            p["start"], clock(p["start"]), p["bstart"], bclock(p["bstart"]), before,
            p["seconds"], p["hz"], p["bhz"], "y" if p["tactical_strict"] else "n",
            "y" if p["broadcast_strict"] else "n"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tail", type=float, default=120.0, metavar="S",
                        help="list BOTH events in each half's last S seconds (default 120)")
    parser.add_argument("--spot", type=int, default=0, metavar="N",
                        help="list every Nth BOTH event in the graded halves (spot check)")
    parser.add_argument("--around", nargs="+", type=int, default=[], metavar="FRAME",
                        help="list BOTH and per-feed events within --radius s of these frames")
    parser.add_argument("--radius", type=float, default=30.0, metavar="S")
    parser.add_argument("--broadcast-video", default=None, metavar="PATH",
                        help="the broadcast video, when it is not at games.video_path's name")
    args = parser.parse_args()

    tac_path = games.video_path(GAME)
    bro_path = args.broadcast_video or games.video_path(GAME, source="broadcast")
    for path in (tac_path, bro_path):
        if not os.path.isfile(path):
            sys.exit("no video at {} (set MTAH_DATA_DIR)".format(path))
    if not os.path.isfile(ALIGN_PATH):
        sys.exit("no broadcast alignment at {}: run `make broadcast-align` first".format(
            os.path.relpath(ALIGN_PATH, ROOT)))
    align = np.load(ALIGN_PATH, allow_pickle=False)
    live_t, bro_frame = align["live_t"], align["bro_frame"]

    n_tac, n_bro = games.source_frames(), games.source_frames("broadcast")
    if len(live_t) != n_bro or len(bro_frame) != n_tac:
        sys.exit("alignment cache does not match the frame counts; re-run make broadcast-align")
    print("A. audio")
    print("  tactical:  " + stream_info(tac_path))
    print("  broadcast: " + stream_info(bro_path))
    tac_f = whistle.video_features(tac_path, n_tac, FPS)
    bro_f = whistle.video_features(bro_path, n_bro, FPS)
    print("  features on {:.4f} of tactical and {:.4f} of broadcast frames".format(
        np.isfinite(tac_f[:, 0]).mean(), np.isfinite(bro_f[:, 0]).mean()))

    tac = whistle.annotate(whistle.events(tac_f, FPS, whistle.STRICT),
                           whistle.events(tac_f, FPS, whistle.LOOSE))
    bro_raw = whistle.annotate(whistle.events(bro_f, FPS, whistle.STRICT),
                               whistle.events(bro_f, FPS, whistle.LOOSE))
    bro, dropped = whistle.map_events(bro_raw, live_t)
    bro = sorted(bro, key=lambda e: e["start"])
    bro = [e for e in bro if e["start"] < n_tac]
    lag, lags = whistle.feed_lag(tac, bro, FPS)
    pairs = whistle.pair_feeds(tac, bro, FPS, lag)
    chance = [whistle.pair_feeds(tac, bro, FPS, lag, shift=int(s * FPS)) for s in SHIFTS_S]

    covered = bro_frame >= 0
    graded = np.zeros(n_tac, dtype=bool)
    for _, s, t in games.half_video_blocks(GAME):
        graded[s:t + 1] = True
    print("  broadcast covers {:.1%} of graded tactical frames; {} of {} broadcast events "
          "unaligned (dropped)".format(covered[graded].mean(), dropped, len(bro_raw)))
    print("  settings: STRICT {}, LOOSE {}; pairing tol {} s, pitch {} Hz, chance shifts {} s"
          .format(whistle.STRICT, whistle.LOOSE, whistle.PAIR_TOL_S, whistle.PAIR_HZ,
                  list(SHIFTS_S)))

    print("\nB. events per minute, over the graded frames the broadcast covers")
    print("  half | minutes | tac STRICT | tac LOOSE | bro STRICT | bro LOOSE |  BOTH | chance "
          "| BOTH/chance")
    for half, s, t in games.half_video_blocks(GAME):
        minutes = covered[s:t + 1].sum() / float(FPS) / 60.0
        rates = [per_minute(x, covered, s, t)[0] for x in
                 ([e for e in tac if e["strict"]], tac, [e for e in bro if e["strict"]], bro,
                  pairs)]
        null = np.mean([per_minute(c, covered, s, t)[0] for c in chance])
        ratio = rates[4] / null if null > 0 else float("inf")
        print("  {:4d} | {:7.1f} | {:10.2f} | {:9.2f} | {:10.2f} | {:9.2f} | {:5.2f} | {:6.2f} "
              "| {:11.1f}".format(half, minutes, *(rates + [null, ratio])))
    print("\n  pitch of events in the graded, covered frames (share per bin, then count)")
    print("  {:16s} | ".format("") + " | ".join("{:.1f}-{:.1f}".format(a / 1e3, b / 1e3)
                                                 for a, b in zip(HZ_BINS[:-1], HZ_BINS[1:])) +
          " |     n")
    print("  {:16s} | {}".format("tactical STRICT",
                                 pitch_row(in_halves([e for e in tac if e["strict"]], covered))))
    print("  {:16s} | {}".format("broadcast STRICT",
                                 pitch_row(in_halves([e for e in bro if e["strict"]], covered))))
    print("  {:16s} | {}".format("BOTH", pitch_row(in_halves(pairs, covered))))

    print("\nC. audio lag of the tactical feed behind the broadcast (frames; + = tactical later)")
    if lags:
        print("  {} pairs within {} s: median {:+.1f} ({:+.2f} s), p10 {:+.1f}, p90 {:+.1f}; "
              "used {:+d}{}".format(len(lags), whistle.LAG_SEARCH_S, np.median(lags),
                                    np.median(lags) / FPS, np.percentile(lags, 10),
                                    np.percentile(lags, 90), lag,
                                    "" if len(lags) >= whistle.MIN_LAG_PAIRS else
                                    " (too few pairs: 0)"))
    else:
        print("  no pairs: lag 0")

    print("\nD. kickoff whistles: BOTH events {:.0f} s to {:.0f} s before each kick".format(
        *KICK_WINDOW_S))
    for name in ("kickoff", "kickoff_2nd"):
        kick = games.anchor_video_frame(games.GAMES[GAME][name])
        near = [p for p in pairs if kick - KICK_WINDOW_S[0] * FPS <= p["start"]
                <= kick + KICK_WINDOW_S[1] * FPS]
        print("  {} (frame {}): {}".format(name, kick, "; ".join(
            "{:.2f} s before, {:.0f} Hz, {:.2f} s".format((kick - p["start"]) / float(FPS),
                                                         p["hz"], p["seconds"])
            for p in near) or "none"))

    print("\nE. end of each half: D33's pick rule on BOTH events")
    for half, s, t in games.half_video_blocks(GAME):
        pick = end_pick(pairs, s, t)
        if pick is None:
            print("  half {}: no pick (no BOTH event after 45:00)".format(half))
        else:
            print("  half {}: pick at tactical frame {} ({}), broadcast frame {} ({}), {:.1f} s "
                  "before the half's last graded frame; {} blast(s), {:.2f} s of whistle".format(
                      half, pick[0]["start"], clock(pick[0]["start"]), pick[0]["bstart"],
                      bclock(pick[0]["bstart"]), (t - pick[0]["start"]) / float(FPS), len(pick),
                      sum(p["seconds"] for p in pick)))
        tail = [p for p in pairs if t - args.tail * FPS <= p["start"] <= t]
        print("  half {}: {} BOTH events in the last {:.0f} s, chance {:.1f}".format(
            half, len(tail), args.tail, np.mean([sum(1 for p in c if t - args.tail * FPS
                                                     <= p["start"] <= t) for c in chance])))
        if tail:
            print_pairs(tail, t)

    if args.spot:
        listed = in_halves(pairs)[::args.spot]
        print("\nF. every {}th BOTH event in the graded halves ({} listed, of {})".format(
            args.spot, len(listed), len(in_halves(pairs))))
        print_pairs(listed)
    for frame in args.around:
        radius = int(args.radius * FPS)
        print("\naround tactical frame {} ({}), +-{:.0f} s ('s before' = before that frame)".format(
            frame, clock(frame), args.radius))
        print("  BOTH:")
        print_pairs([p for p in pairs if abs(p["start"] - frame) <= radius], frame)
        for name, events in (("tactical", tac), ("broadcast (mapped)", bro)):
            near = [e for e in events if abs(e["start"] - frame) <= radius]
            print("  {}: ".format(name) + ("; ".join(
                "{:+.2f} s {:.0f} Hz {:.2f} s{}".format((e["start"] - frame) / float(FPS), e["hz"],
                                                       e["seconds"], "" if e["strict"] else " (L)")
                for e in near) or "none"))


if __name__ == "__main__":
    main()
