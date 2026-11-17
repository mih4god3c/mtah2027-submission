"""Whistle detector Step 0 on BAR-ATH: what it finds.

    .venv/bin/python tools/whistle_check.py                     # (needs data/) end of each half
    .venv/bin/python tools/whistle_check.py --around FRAME [FRAME ...]  # events near given frames
    .venv/bin/python tools/whistle_check.py --all               # every STRICT event, for spot checks
    .venv/bin/python tools/whistle_check.py --source broadcast  # the broadcast's audio on its own
    .venv/bin/python tools/whistle_check.py --source broadcast --from 52:00 --to 54:00

--source broadcast runs the same detector, settings unchanged, on the broadcast video's audio
alone: no alignment, no pairing (D44's pairing is tools/whistle_broadcast_check.py). Frames and
times are the broadcast video's own. The graded halves are tactical frames, so the end-of-half
sections are skipped; it prints the event rate per 10 minutes and the pitch of STRICT events, and
lists events with --all, --around (broadcast frames) or --from/--to (h:mm:ss, mm:ss or seconds).

The first run decodes the tactical video's audio (a few minutes) and caches per-frame features
in cache/whistle/ (gitignored: derived from match data). Later runs take seconds.

Prints, per graded half (games.half_video_blocks): events per minute under STRICT and LOOSE
(solution/whistle.py), then every event in the half's last --tail seconds with its video frame,
video time (h:mm:ss.ff, for scrubbing a player), seconds before the half's last graded frame,
length, pitch, tonality and rise, and whether STRICT finds it (LOOSE-only rows are near-misses).

Candidate end-of-half pick, fixed before any BAR-ATH run (2026-11-06), printed per half: group
STRICT events whose gaps are at most GROUP_S; the pick is the last group that starts at least
45:00 after the half's first graded frame (Law 7; for half 2 that frame is ~2 min before the
kick, so the window opens a little early) and before its last graded frame. Its first event's
start is where a rule would begin dead ball. It is printed with its blasts and total length.
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

GAME = "BAR-ATH"
FPS = games.GAMES[GAME]["fps"]
GROUP_S = 3.0                       # blasts this close belong to one signal
LAW7_S = 45 * 60.0


def clock(frame):
    seconds = frame / float(FPS)
    hours, rest = divmod(seconds, 3600)
    minutes, rest = divmod(rest, 60)
    return "{:d}:{:02d}:{:05.2f}".format(int(hours), int(minutes), rest)


def merged(strict, loose):
    """LOOSE events, each marked with whether a STRICT event overlaps it, plus any STRICT event
    no LOOSE event covers (not expected, but never dropped)."""
    rows = []
    for event in loose:
        hit = any(s["start"] < event["stop"] and event["start"] < s["stop"] for s in strict)
        rows.append(dict(event, strict=hit))
    for s in strict:
        if not any(r["start"] < s["stop"] and s["start"] < r["stop"] for r in rows):
            rows.append(dict(s, strict=True))
    return sorted(rows, key=lambda r: r["start"])


def groups(events, gap_s=GROUP_S):
    """Lists of events whose gaps are at most gap_s."""
    out = []
    for event in events:
        if out and event["start"] - out[-1][-1]["stop"] <= gap_s * FPS:
            out[-1].append(event)
        else:
            out.append([event])
    return out


HEADER = ("    video frame | video time    | s before end | length s |   Hz | tonal dB | "
          "rise dB | strict")


def print_rows(rows, end=None):
    print(HEADER)
    for r in rows:
        before = "" if end is None else "{:12.1f}".format((end - r["start"]) / float(FPS))
        print("    {:11d} | {} | {:>12} | {:8.2f} | {:4.0f} | {:8.1f} | {:7.1f} | {}".format(
            r["start"], clock(r["start"]), before, r["seconds"], r["hz"], r["tonal_db"],
            r["rise_db"], "yes" if r["strict"] else "loose only"))


def end_pick(strict, start, end):
    """The candidate pick: the last STRICT group in [start + 45:00, end]."""
    window = [e for e in strict if start + LAW7_S * FPS <= e["start"] <= end]
    found = groups(window)
    return found[-1] if found else None


PITCH_BINS = (2000, 2500, 3000, 3500, 4000, 4500, 5001)


def parse_time(text):
    """'h:mm:ss', 'mm:ss' or seconds (decimals allowed) -> video frame."""
    seconds = 0.0
    for part in text.split(":"):
        seconds = seconds * 60 + float(part)
    return int(round(seconds * FPS))


def whole_file_report(source, rows, strict, loose, n_frames, args):
    """Summary and listings over a whole file, on its own timeline (no graded halves)."""
    minutes = n_frames / float(FPS) / 60.0
    lengths = np.array([e["seconds"] for e in strict]) if strict else np.zeros(1)
    print("\nwhole file ({:.1f} min): STRICT {} events ({:.2f}/min, length p50 {:.2f} s, p90 "
          "{:.2f} s, max {:.2f} s), LOOSE {} ({:.2f}/min)".format(
              minutes, len(strict), len(strict) / minutes, np.median(lengths),
              np.percentile(lengths, 90), lengths.max(), len(loose), len(loose) / minutes))
    hz = np.array([e["hz"] for e in strict])
    counts, _ = np.histogram(hz, bins=PITCH_BINS)
    print("  pitch of STRICT events: " + ", ".join(
        "{:.1f}-{:.1f} kHz {:d}".format(a / 1e3, b / 1e3, c)
        for a, b, c in zip(PITCH_BINS[:-1], PITCH_BINS[1:], counts)))
    print("  STRICT events per 10 minutes of {} video time (all pitches / 3.5 kHz and up):".format(
        source))
    step = 10 * 60 * FPS
    for a in range(0, n_frames, step):
        inside = [e for e in strict if a <= e["start"] < a + step]
        high = sum(1 for e in inside if e["hz"] >= 3500.0)
        print("    {} - {}: {:4d} / {:4d}".format(clock(a), clock(min(a + step, n_frames)),
                                                 len(inside), high))
    if args.window_from is not None or args.window_to is not None:
        a = parse_time(args.window_from) if args.window_from else 0
        b = parse_time(args.window_to) if args.window_to else n_frames
        print("\nevery event from {} to {}".format(clock(a), clock(b)))
        print_rows([r for r in rows if a <= r["start"] <= b])
    for frame in args.around:
        radius = int(args.radius * FPS)
        print("\naround {} video frame {} ({}), +-{:.0f} s; 's before end' = seconds before that "
              "frame".format(source, frame, clock(frame), args.radius))
        print_rows([r for r in rows if abs(r["start"] - frame) <= radius], frame)
    if args.all:
        print("\nevery STRICT event in the {} file".format(source))
        print_rows([dict(e, strict=True) for e in strict])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tail", type=float, default=180.0, metavar="S",
                        help="list events in each half's last S seconds (default 180)")
    parser.add_argument("--around", nargs="+", type=int, default=[], metavar="FRAME",
                        help="list events within --radius seconds of these video frames")
    parser.add_argument("--radius", type=float, default=30.0, metavar="S")
    parser.add_argument("--all", action="store_true",
                        help="list every STRICT event in the graded halves (broadcast: the file)")
    parser.add_argument("--source", choices=("tactical", "broadcast"), default="tactical",
                        help="whose audio (default tactical)")
    parser.add_argument("--video", default=None, metavar="PATH",
                        help="the video file, when it is not at games.video_path's name "
                             "(e.g. data/broadcast/BAR-ATH_broadcast-001.mp4)")
    parser.add_argument("--from", dest="window_from", default=None, metavar="TIME",
                        help="list every event from this video time (h:mm:ss, mm:ss or s)")
    parser.add_argument("--to", dest="window_to", default=None, metavar="TIME",
                        help="... to this video time")
    args = parser.parse_args()

    path = args.video or games.video_path(GAME, source=args.source)
    if not os.path.isfile(path):
        sys.exit("no {} video at {} (set MTAH_DATA_DIR)".format(args.source, path))
    n_frames = games.source_frames(args.source)
    features = whistle.video_features(path, n_frames, FPS)
    strict = whistle.events(features, FPS, whistle.STRICT)
    loose = whistle.events(features, FPS, whistle.LOOSE)
    rows = merged(strict, loose)
    covered = float(np.mean(np.isfinite(features[:, 0])))
    print("{} {} audio: features on {:.4f} of {} video frames; STRICT {}, LOOSE {}".format(
        GAME, args.source, covered, n_frames, whistle.STRICT, whistle.LOOSE))
    if args.source == "broadcast":
        whole_file_report(args.source, rows, strict, loose, n_frames, args)
        return
    if args.window_from is not None or args.window_to is not None:
        a = parse_time(args.window_from) if args.window_from else 0
        b = parse_time(args.window_to) if args.window_to else n_frames
        print("\nevery event from {} to {}".format(clock(a), clock(b)))
        print_rows([r for r in rows if a <= r["start"] <= b])

    for half, start, end in games.half_video_blocks(GAME):
        minutes = (end + 1 - start) / float(FPS) / 60.0
        in_half = [e for e in strict if start <= e["start"] <= end]
        loose_n = sum(1 for e in loose if start <= e["start"] <= end)
        lengths = np.array([e["seconds"] for e in in_half]) if in_half else np.zeros(1)
        print("\nhalf {}: video frames {}..{} ({:.1f} min); STRICT {} events ({:.2f}/min, length "
              "p50 {:.2f} s, p90 {:.2f} s, max {:.2f} s), LOOSE {}".format(
                  half, start, end, minutes, len(in_half), len(in_half) / minutes,
                  np.median(lengths), np.percentile(lengths, 90), lengths.max(), loose_n))
        tail_from = end - int(args.tail * FPS)
        print("  events in the last {:.0f} s (seconds before frame {}, the half's last graded "
              "frame):".format(args.tail, end))
        print_rows([r for r in rows if tail_from <= r["start"] <= end], end)
        pick = end_pick(strict, start, end)
        if pick is None:
            print("  candidate end pick: none (no STRICT event after 45:00)")
        else:
            print("  candidate end pick: frame {} ({}), {:.1f} s before the end; {} blast(s), "
                  "{:.2f} s of whistle".format(
                      pick[0]["start"], clock(pick[0]["start"]),
                      (end - pick[0]["start"]) / float(FPS), len(pick),
                      sum(e["seconds"] for e in pick)))

    for frame in args.around:
        radius = int(args.radius * FPS)
        print("\naround video frame {} ({}), +-{:.0f} s; 's before end' here is seconds before "
              "that frame (negative: after it)".format(frame, clock(frame), args.radius))
        print_rows([r for r in rows if abs(r["start"] - frame) <= radius], frame)

    if args.all:
        print("\nevery STRICT event in the graded halves")
        blocks = games.half_video_blocks(GAME)
        print_rows([dict(e, strict=True) for e in strict
                    if any(s <= e["start"] <= t for _, s, t in blocks)])


if __name__ == "__main__":
    main()
