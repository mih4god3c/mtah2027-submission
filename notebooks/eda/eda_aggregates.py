"""Aggregate-only exploration of the match data, for a HUMAN to run. Prints no rows.

    .venv/bin/python notebooks/eda/eda_aggregates.py                 # everything, a few minutes
    .venv/bin/python notebooks/eda/eda_aggregates.py --skip broadcast video
    .venv/bin/python notebooks/eda/eda_aggregates.py --only ball gaps

Why this exists: the terms of use forbid sending the match data, or any extract of it, to a hosted
LLM, so the AI assistant on this team never reads data/. This script measures what PLAN.md
Appendix A asks for and prints counts, shares, percentiles and histograms only -- never a frame,
a row or a coordinate. Read the output before sharing it, and paste only what you are comfortable
with into notebooks/eda/FINDINGS.md.

Nothing printed here is a label, and nothing here may become a frame-specific constant in
solution/ (RULES.md). It is a throwaway script, not part of the submission pipeline.

Sections (names for --only / --skip):
    schema     A1  channel names declared in the ATD metadata, ball-group field count
    ball       A2  ball visibility, gap lengths, out-of-pitch share, speeds, stationary stretches
    players    A3  players visible per team, slot lifetimes, duplicates, speeds
    gaps       A4  what is happening around ball gaps
    tagging    A5  Smart Tagging durations, team tags, set pieces, overlaps
    video      A6  ffprobe stream metadata for both videos (no decoding)
    broadcast  A7  broadcast ATD ball visibility and valid-homography stretches
    direction  A8  each team's mean x per half, to check an automatic direction detector
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, asset_paths, half_video_blocks, require, source_frames  # noqa: E402
import baseline  # noqa: E402  (team_of_each_track)
import load_homography  # noqa: E402
import load_smart_tagging  # noqa: E402

SECTIONS = ("schema", "ball", "players", "gaps", "tagging", "video", "broadcast", "direction")

FPS = GAMES[GAME]["fps"]
LENGTH, WIDTH = GAMES[GAME]["pitch"]          # metres
BLOCK = 5 * 60 * FPS                          # 5-minute blocks
TELEPORT_MS = 40.0                            # implied ball speed treated as a tracking jump
STATIONARY_MS = 0.5                           # ball speed below this counts as still
STATIONARY_MIN = FPS                          # ... for at least one second
DUPLICATE_M = 0.7                             # same-team points closer than this
BOX_DEPTH, BOX_HALF_WIDTH = 16.5, 20.16       # penalty area, metres
WIDE_M = 15.0                                 # "wide" channel: within this of a touchline


# -- small helpers -------------------------------------------------------------------------------

def pct(values, qs=(5, 25, 50, 75, 95), unit=""):
    """'n=.. p5=.. p25=..' over the finite values, or 'n=0'."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return "n=0"
    return "n={}  ".format(len(values)) + "  ".join(
        "p{}={:.2f}{}".format(q, np.percentile(values, q), unit) for q in qs)


def share(numerator, denominator):
    return "{:.1f}%".format(100.0 * numerator / denominator) if denominator else "--"


def halves(n_frames):
    """[(half, start, stop)] half-open video-frame spans of the graded halves."""
    return [(half, first, min(last + 1, n_frames))
            for half, first, last in half_video_blocks(GAME)]


def graded_mask(n_frames):
    mask = np.zeros(n_frames, dtype=bool)
    for _, start, stop in halves(n_frames):
        mask[start:stop] = True
    return mask


def true_runs(mask, n_frames):
    """[(start, stop)] runs of True, computed inside each graded half so halftime never merges."""
    out = []
    for _, start, stop in halves(n_frames):
        window = np.concatenate(([False], mask[start:stop], [False])).astype(np.int8)
        edges = np.diff(window)
        out.extend(zip((start + np.flatnonzero(edges == 1)).tolist(),
                       (start + np.flatnonzero(edges == -1)).tolist()))
    return out


def header(code, title):
    print("\n" + "=" * 78)
    print("{}  {}".format(code, title))
    print("=" * 78)


# -- loading -------------------------------------------------------------------------------------

def parse_atd(raw_path, n_slots, n_frames, ball_only=False):
    """Players (T, slots, 3) and ball (T, k) as float32, NaN where missing. Never printed."""
    players = None if ball_only else np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    ball = None
    stats = Counter()
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\r\n").split(":")
            if len(parts) != 3:
                stats["malformed lines"] += 1
                continue
            frame = int(parts[0]) - 1          # ATD frame_id counts from 1, video frames from 0
            if not 0 <= frame < n_frames:
                stats["frame ids out of range"] += 1
                continue
            stats["records"] += 1

            fields = parts[2].split(",")
            if ball is None:
                ball = np.full((n_frames, len(fields)), np.nan, dtype=np.float32)
            for k in range(min(len(fields), ball.shape[1])):
                try:
                    ball[frame, k] = float(fields[k])
                except ValueError:
                    stats["unparseable ball fields"] += 1
            if len(fields) != ball.shape[1]:
                stats["ball groups with a different field count"] += 1

            if ball_only:
                continue
            groups = parts[1].split(";")
            if len(groups) != n_slots:
                stats["lines whose slot count differs from the metadata"] += 1
            for slot in range(min(len(groups), n_slots)):
                group = groups[slot]
                if not group or group[0] == "N":
                    continue
                values = group.split(",")
                for k in range(min(len(values), 3)):
                    try:
                        players[frame, slot, k] = float(values[k])
                    except ValueError:
                        stats["unparseable player fields"] += 1
    return players, ball, stats


class Match(object):
    """Arrays for the tactical recording, in metres. Built once, shared by the sections."""

    def __init__(self, data_dir):
        meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, data_dir)
        self.meta_path = meta_path
        self.team = baseline.team_of_each_track(meta_path)
        self.n_frames = source_frames("tactical", GAME)
        started = time.time()
        print("parsing tactical ATD ({} slots) ...".format(len(self.team)), flush=True)
        players, ball, self.parse_stats = parse_atd(raw_path, len(self.team), self.n_frames)
        print("  done in {:.0f} s".format(time.time() - started))
        self.norm_x = players[:, :, 0]                     # 0-1, kept for direction checks
        self.px = players[:, :, 0] * LENGTH
        self.py = players[:, :, 1] * WIDTH
        self.ps = players[:, :, 2]                          # speed field as delivered, m/s
        self.ball_raw = ball
        self.bx = ball[:, 0] * LENGTH
        self.by = ball[:, 1] * WIDTH if ball.shape[1] > 1 else np.full(self.n_frames, np.nan)
        self.ball_seen = np.isfinite(self.bx) & np.isfinite(self.by)
        self.graded = graded_mask(self.n_frames)
        self.present = np.isfinite(self.px) & np.isfinite(self.py)


# -- sections ------------------------------------------------------------------------------------

def strip_ns(tag):
    return tag.split("}", 1)[-1]


def section_schema(match):
    header("A1", "ATD schema (names only, no values)")
    root = ET.parse(match.meta_path).getroot()
    tags = Counter(strip_ns(el.tag) for el in root.iter())
    print("XML element names and counts:")
    for tag, count in sorted(tags.items()):
        print("  {:<40} {}".format(tag, count))

    # Channel identifiers, with digits collapsed so 46 per-slot copies print as one pattern.
    channel_ids = Counter()
    for el in root.iter():
        if "channel" not in strip_ns(el.tag).lower():
            continue
        for key, value in el.attrib.items():
            if "id" in key.lower() or "channel" in key.lower():
                channel_ids["{}.{}={}".format(strip_ns(el.tag), key, re.sub(r"\d+", "#", value))] += 1
        name = (el.findtext("Name") or "").strip()
        if name:
            channel_ids["{}.Name={}".format(strip_ns(el.tag), re.sub(r"\d+", "#", name))] += 1
    print("\nchannel identifiers (digits shown as #):")
    for key, count in sorted(channel_ids.items()):
        print("  {:<60} x{}".format(key, count))

    print("\ntrack slots in metadata: {}  (home {}, away {})".format(
        len(match.team), int(np.sum(match.team == 0)), int(np.sum(match.team == 1))))
    print("parse: " + ", ".join("{} {}".format(k, v) for k, v in sorted(match.parse_stats.items())))

    ball = match.ball_raw
    print("\nball group has {} comma-separated fields".format(ball.shape[1]))
    graded = match.graded
    for k in range(ball.shape[1]):
        column = ball[graded, k].astype(np.float64)
        print("  field {}: NaN on {} of graded frames; {}".format(
            k + 1, share(np.sum(~np.isfinite(column)), len(column)),
            pct(column, qs=(1, 50, 99))))
    print("  (fields 1-2 are x, y in 0-1. A third field near 0 with a long upper tail looks like")
    print("   height; one tracking ball speed looks like m/s. Compare with section A2 speeds.)")


def section_ball(match):
    header("A2", "ball")
    seen, graded, n = match.ball_seen, match.graded, match.n_frames
    print("ball visible on {} of all frames, {} of graded frames".format(
        share(seen.sum(), n), share((seen & graded).sum(), graded.sum())))
    for half, start, stop in halves(n):
        window = seen[start:stop]
        blocks = [window[i:i + BLOCK].mean() for i in range(0, len(window), BLOCK)
                  if len(window[i:i + BLOCK]) >= BLOCK // 2]
        print("  half {}: {} visible; per 5-min block: {}".format(
            half, share(window.sum(), len(window)), " ".join("{:.2f}".format(b) for b in blocks)))

    gaps = true_runs(~seen, n)
    lengths = np.array([stop - start for start, stop in gaps], dtype=np.float64) / FPS
    print("\nball-missing runs inside graded halves: {}".format(len(gaps)))
    print("  length (s): " + pct(lengths, qs=(10, 25, 50, 75, 90, 99)) +
          "  max={:.1f}".format(lengths.max() if len(lengths) else 0))
    if len(lengths):
        missing = lengths.sum()
        for threshold in (1, 5, 10, 30):
            print("  runs >= {:>2} s: {:>4}  carrying {} of missing frames".format(
                threshold, int(np.sum(lengths >= threshold)),
                share(lengths[lengths >= threshold].sum(), missing)))

    visible = seen & graded
    out_lines = visible & ((match.bx < 0) | (match.bx > LENGTH) | (match.by < 0) | (match.by > WIDTH))
    out_margin = visible & ((match.bx < -1) | (match.bx > LENGTH + 1) |
                            (match.by < -1) | (match.by > WIDTH + 1))
    print("\nvisible graded frames with the ball beyond a line: {}  (beyond by > 1 m: {})".format(
        share(out_lines.sum(), visible.sum()), share(out_margin.sum(), visible.sum())))

    both = seen[1:] & seen[:-1]
    speed = np.full(n, np.nan)
    speed[1:][both] = np.hypot(np.diff(match.bx)[both], np.diff(match.by)[both]) * FPS
    graded_speed = speed[graded]
    print("ball speed from consecutive positions (m/s): " +
          pct(graded_speed, qs=(50, 90, 95, 99)) +
          "  max={:.1f}".format(np.nanmax(graded_speed) if np.isfinite(graded_speed).any() else 0))
    print("  jumps above {:.0f} m/s (tracking teleports): {}".format(
        TELEPORT_MS, int(np.nansum(graded_speed > TELEPORT_MS))))

    still = np.nan_to_num(speed, nan=np.inf) < STATIONARY_MS
    stretches = [(a, b) for a, b in true_runs(still, n) if b - a >= STATIONARY_MIN]
    still_lengths = np.array([b - a for a, b in stretches], dtype=np.float64) / FPS
    print("stationary-ball stretches (< {} m/s for >= 1 s): {}".format(STATIONARY_MS, len(stretches)))
    print("  length (s): " + pct(still_lengths, qs=(10, 50, 90)) +
          "  covering {} of graded frames".format(share(still_lengths.sum() * FPS, graded.sum())))
    if match.ball_raw.shape[1] >= 3:
        third = match.ball_raw[:, 2].astype(np.float64)
        ok = graded & np.isfinite(third) & np.isfinite(speed)
        if ok.sum() > 100:
            print("  correlation of ball field 3 with computed speed: {:.2f}".format(
                np.corrcoef(third[ok], speed[ok])[0, 1]))


def section_players(match):
    header("A3", "players")
    graded, present, team = match.graded, match.present, match.team
    total = present[graded].sum(axis=1)
    print("tracked objects per graded frame: mean {:.2f}; {}".format(total.mean(), pct(total)))
    for code, name in ((0, "home"), (1, "away")):
        counts = present[:, team == code][graded].sum(axis=1)
        print("  {}: mean {:.2f}; {}; frames with > 11: {}; with < 9: {}".format(
            name, counts.mean(), pct(counts), share(np.sum(counts > 11), len(counts)),
            share(np.sum(counts < 9), len(counts))))

    lifetimes = []
    for slot in range(present.shape[1]):
        lifetimes.extend(b - a for a, b in true_runs(present[:, slot], match.n_frames))
    lifetimes = np.array(lifetimes, dtype=np.float64) / FPS
    print("\ntracklets (contiguous runs of one slot) inside graded halves: {}".format(len(lifetimes)))
    print("  lifetime (s): " + pct(lifetimes, qs=(10, 25, 50, 75, 90)))

    frames = np.flatnonzero(graded)
    frames_with_duplicate = pairs = 0
    for begin in range(0, len(frames), 4000):
        chunk = frames[begin:begin + 4000]
        has = np.zeros(len(chunk), dtype=bool)
        for code in (0, 1):
            x = match.px[chunk][:, team == code]
            y = match.py[chunk][:, team == code]
            distance = np.hypot(x[:, :, None] - x[:, None, :], y[:, :, None] - y[:, None, :])
            upper = np.triu(np.ones(distance.shape[1:], dtype=bool), k=1)
            close = (distance < DUPLICATE_M) & upper[None]
            pairs += int(close.sum())
            has |= close.any(axis=(1, 2))
        frames_with_duplicate += int(has.sum())
    print("same-team pairs closer than {} m: {} of graded frames have one; {:.3f} pairs per frame".format(
        DUPLICATE_M, share(frames_with_duplicate, len(frames)), pairs / max(1, len(frames))))

    speeds = match.ps[graded][present[graded]]
    print("player speed field (m/s): " + pct(speeds, qs=(50, 90, 99)) +
          "  max={:.1f}".format(np.nanmax(speeds) if len(speeds) else 0))


def team_spread(match, code):
    x = match.px[:, match.team == code]
    y = match.py[:, match.team == code]
    with np.errstate(invalid="ignore"), _quiet():
        cx, cy = np.nanmean(x, axis=1), np.nanmean(y, axis=1)
        return np.nanmean(np.hypot(x - cx[:, None], y - cy[:, None]), axis=1)


class _quiet(object):
    """Silence 'mean of empty slice' warnings for frames with nobody tracked."""

    def __enter__(self):
        import warnings
        self._catch = warnings.catch_warnings()
        self._catch.__enter__()
        warnings.simplefilter("ignore", category=RuntimeWarning)

    def __exit__(self, *exc):
        self._catch.__exit__(*exc)


def zone(x, y):
    if not (np.isfinite(x) and np.isfinite(y)):
        return "unknown"
    if x < 0 or x > LENGTH or y < 0 or y > WIDTH:
        return "beyond a line"
    if (x < BOX_DEPTH or x > LENGTH - BOX_DEPTH) and abs(y - WIDTH / 2) < BOX_HALF_WIDTH:
        return "penalty box"
    if y < WIDE_M or y > WIDTH - WIDE_M:
        return "wide channel"
    return "middle"


def section_gaps(match):
    header("A4", "context of ball gaps")
    graded, seen = match.graded, match.ball_seen
    with _quiet():
        mean_speed = np.nanmean(np.where(match.present, match.ps, np.nan), axis=1)
    inside, outside = graded & ~seen, graded & seen
    print("mean player speed (m/s): inside gaps {:.2f}, ball visible {:.2f}".format(
        np.nanmean(mean_speed[inside]), np.nanmean(mean_speed[outside])))
    for code, name in ((0, "home"), (1, "away")):
        spread = team_spread(match, code)
        print("{} spread, mean distance to centroid (m): inside gaps {:.2f}, ball visible {:.2f}".format(
            name, np.nanmean(spread[inside]), np.nanmean(spread[outside])))

    zones, lengths, jumps = Counter(), {}, []
    for start, stop in true_runs(~seen, match.n_frames):
        before, after = start - 1, stop
        if before < 0 or not seen[before]:
            label = "unknown"
        else:
            label = zone(match.bx[before], match.by[before])
            if after < match.n_frames and seen[after]:
                jumps.append(np.hypot(match.bx[after] - match.bx[before],
                                      match.by[after] - match.by[before]))
        zones[label] += 1
        lengths.setdefault(label, []).append((stop - start) / float(FPS))
    print("\nwhere the ball was last seen before each gap:")
    for label, count in zones.most_common():
        print("  {:<14} {:>4} gaps   median length {:.1f} s".format(
            label, count, np.median(lengths[label])))
    print("distance between last-seen and first-seen ball across a gap (m): " +
          pct(jumps, qs=(10, 50, 90)))


def section_tagging(match, data_dir):
    header("A5", "Smart Tagging")
    events = load_smart_tagging.load(GAME, data_dir)
    n = match.n_frames
    codes = sorted({event["code"] for event in events})
    print("{:<32} {:>4}  {:<42} {}".format("code", "n", "duration (s)", "Team tag counts"))
    for code in codes:
        rows = [e for e in events if e["code"] == code]
        durations = [e["end_s"] - e["start_s"] for e in rows]
        teams = Counter(e["tags"].get("Team") or "(none)" for e in rows)
        print("{:<32} {:>4}  {:<42} {}".format(
            code, len(rows), pct(durations, qs=(10, 50, 90)),
            ", ".join("{} {}".format(k, v) for k, v in sorted(teams.items()))))

    set_pieces = [e for e in events if (e["code"] or "").strip().lower() == "set pieces"]
    table = Counter((e["tags"].get("Type") or "(none)", e["tags"].get("Half") or "(none)")
                    for e in set_pieces)
    print("\nset pieces by type and half:")
    for (kind, half), count in sorted(table.items()):
        print("  {:<24} {:<10} {}".format(kind, half, count))

    cover = {}
    for code in codes:
        mask = np.zeros(n, dtype=bool)
        for e in events:
            if e["code"] == code:
                mask[max(0, e["start_frame"]):min(n, e["end_frame"])] = True
        cover[code] = mask
    graded = match.graded
    any_code = np.zeros(n, dtype=bool)
    for mask in cover.values():
        any_code |= mask
    print("\ngraded frames covered by any tag: {}".format(share((any_code & graded).sum(), graded.sum())))
    short = {code: "".join(word[0] for word in code.split())[:4] for code in codes}
    print("overlap matrix: share of the ROW code's frames also covered by the COLUMN code")
    print("  " + " " * 6 + "".join("{:>6}".format(short[c]) for c in codes))
    for row in codes:
        denominator = cover[row].sum()
        print("  {:<6}".format(short[row]) + "".join(
            "{:>6.2f}".format((cover[row] & cover[col]).sum() / float(denominator)
                              if denominator else 0.0) for col in codes))
    print("  key: " + ", ".join("{}={}".format(short[c], c) for c in codes))


def section_video(data_dir):
    header("A6", "video stream metadata (ffprobe, nothing decoded)")
    probe = shutil.which("ffprobe")
    if not probe:
        print("ffprobe not found; install ffmpeg (brew install ffmpeg) and re-run this section")
        return
    for source in ("tactical", "broadcast"):
        path = asset_paths(GAME, data_dir, source)["video"]
        if not os.path.isfile(path):
            print("{}: video not found".format(source))
            continue
        output = subprocess.run(
            [probe, "-v", "error", "-show_entries",
             "stream=index,codec_type,codec_name,width,height,avg_frame_rate,nb_frames,"
             "sample_rate,channels:format=duration", "-of", "compact", path],
            capture_output=True, text=True).stdout.strip()
        print("{}:".format(source))
        for line in output.splitlines():
            print("  " + line)
    print("(an audio stream line means the whistle-detection idea, backlog E10, is possible)")


def section_broadcast(data_dir):
    header("A7", "broadcast recording")
    try:
        meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, data_dir, "broadcast")
    except IOError as error:
        print(error)
        return
    n_frames = source_frames("broadcast", GAME)
    n_slots = len(baseline.team_of_each_track(meta_path))
    print("parsing broadcast ATD, ball only ...", flush=True)
    _, ball, stats = parse_atd(raw_path, n_slots, n_frames, ball_only=True)
    seen = np.isfinite(ball[:, 0])
    print("  records {}; ball visible on {} of frames".format(stats["records"], share(seen.sum(), n_frames)))

    _, _, valid = load_homography.load(GAME, data_dir, "broadcast")
    for label, mask in (("valid", valid), ("invalid", ~valid)):
        window = np.concatenate(([False], mask, [False])).astype(np.int8)
        edges = np.diff(window)
        lengths = (np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)) / float(FPS)
        print("  {} homography stretches: {}; length (s): {}".format(
            label, len(lengths), pct(lengths, qs=(10, 50, 90, 99))))
    print("  ball visible where homography valid: {}; where invalid: {}".format(
        share((seen & valid).sum(), valid.sum()), share((seen & ~valid).sum(), (~valid).sum())))


def section_direction(match):
    header("A8", "direction of play (mean normalised x, 0-1)")
    for half, start, stop in halves(match.n_frames):
        first = slice(start, min(stop, start + 30 * FPS))
        whole = slice(start, stop)
        with _quiet():
            values = ["{} first 30 s {:.3f}, whole half {:.3f}".format(
                name, np.nanmean(match.norm_x[first][:, match.team == code]),
                np.nanmean(match.norm_x[whole][:, match.team == code]))
                for code, name in ((0, "home"), (1, "away"))]
        print("  half {}: {}".format(half, ";  ".join(values)))
    print("  (a team well below 0.5 at the start of a half defends the x = 0 goal)")


# -- entry point ---------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="+", choices=SECTIONS, help="run just these sections")
    parser.add_argument("--skip", nargs="+", choices=SECTIONS, default=[], help="skip these")
    parser.add_argument("--data-dir", default=None, help="defaults to MTAH_DATA_DIR or ./data")
    args = parser.parse_args()
    wanted = [s for s in (args.only or SECTIONS) if s not in args.skip]

    print("AGGREGATES ONLY. Review before sharing; never paste rows, frames or plots of the data.")
    started = time.time()
    needs_match = {"schema", "ball", "players", "gaps", "tagging", "direction"}
    match = Match(args.data_dir) if needs_match & set(wanted) else None

    if "schema" in wanted:
        section_schema(match)
    if "ball" in wanted:
        section_ball(match)
    if "players" in wanted:
        section_players(match)
    if "gaps" in wanted:
        section_gaps(match)
    if "tagging" in wanted:
        section_tagging(match, args.data_dir)
    if "video" in wanted:
        section_video(args.data_dir)
    if "broadcast" in wanted:
        section_broadcast(args.data_dir)
    if "direction" in wanted:
        section_direction(match)
    print("\nfinished in {:.0f} s".format(time.time() - started))


if __name__ == "__main__":
    main()
