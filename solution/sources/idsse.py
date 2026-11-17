"""IDSSE / DFL open data -> Match, with per-frame possession labels.

The DFL position files carry, on every ball frame, `BallStatus` (1 alive, 0 dead) and
`BallPossession` (1 home, 2 away). That is the same kind of signal the BAR-ATH ground truth is
derived from -- a corrected tracking product's dead flag and owning team -- so the conversion is
mechanical, exactly as CHALLENGE.md describes for the competition truth:

    D where BallStatus == 0, else H if BallPossession == 1, else A.

Caveat that matters for every number this produces: DFL and Metrica need not define "owning
team" identically. PLAN.md section 6 (L2) is where that gets checked.

Parsing is a line-oriented stream over the XML rather than kloppy: every <Frame> sits on its own
line, so a regex pass reads a 400 MB file in seconds, where a DOM parse needs minutes and
gigabytes. Positions come centred in metres and are shifted to the corner origin Match uses.
"""

import glob
import json
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np

from solution.match import AWAY, CACHE_DIR, HOME, Match, cache_path, cached, pm, validate

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT, "external_data", "idsse")
MATCH_IDS = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
# Two matches are held out until Phase 4 (PLAN.md section 6): nothing may be tuned or chosen on
# them, so they can tell us at the end how much we over-fitted the other five. One first-division
# (DFL-COM-000001) and one second-division match, fixed on 2026-10-28 before any method was
# tuned. Do not change this split; if you need more data, add matches instead.
HOLDOUT_IDS = ("J03WN1", "J03WR9")
DEV_IDS = tuple(m for m in MATCH_IDS if m not in HOLDOUT_IDS)
FPS = 25
HALF_ORDER = ("firstHalf", "secondHalf")   # extra time and penalties, if present, are ignored
# Frames of X between halves in the stitched timeline. Any positive number keeps the halves as
# separate scorer runs; 250 (10 s) also keeps windowed features from reaching across.
HALF_GAP = 250
# Bump when build() changes what it produces, so stale caches are not reused. 2: referees dropped.
CACHE_VERSION = 2

_FRAMESET = re.compile(r'GameSection="(\w+)".*?TeamId="([^"]+)".*?PersonId="([^"]+)"')
_FRAME = re.compile(r'<Frame N="(\d+)".*? X="(-?[\d.]+)" Y="(-?[\d.]+)"')
_POSSESSION = re.compile(r'BallPossession="(\d)"')
_STATUS = re.compile(r'BallStatus="(\d)"')


def files(match_id, data_dir=DATA_DIR):
    def one(kind):
        found = glob.glob(os.path.join(data_dir, "DFL_{}_*DFL-MAT-{}.xml".format(kind, match_id)))
        if len(found) != 1:
            raise IOError("IDSSE {}: expected one {} file in {}, found {}. Run "
                          "`make external` first.".format(match_id, kind, data_dir, len(found)))
        return found[0]
    return one("02_01_matchinformation"), one("04_03_positions_raw_observed")


def team_ids(info_path):
    """(home TeamId, guest TeamId) from the match information file."""
    roles = {team.get("Role"): team.get("TeamId")
             for team in ET.parse(info_path).getroot().iter("Team")}
    if "home" not in roles or "guest" not in roles:
        raise ValueError("{}: expected a home and a guest team, found {}".format(
            info_path, sorted(roles)))
    return roles["home"], roles["guest"]


def pitch_size(positions_path):
    with open(positions_path) as handle:
        for line in handle:
            if "<PitchSize" in line:
                return (float(re.search(r'X="([\d.]+)"', line).group(1)),
                        float(re.search(r'Y="([\d.]+)"', line).group(1)))
            if line.startswith("<FrameSet"):
                break
    raise ValueError("{}: no PitchSize".format(positions_path))


def parse_positions(path):
    """{half: {"ball": (N, x, y, possession, status), (team, person): (N, x, y)}} as arrays."""
    out, current, rows, is_ball = {}, None, [], False
    with open(path) as handle:
        for line in handle:
            if line.startswith("<Frame "):
                found = _FRAME.match(line)
                if is_ball:
                    possession = _POSSESSION.search(line)
                    status = _STATUS.search(line)
                    rows.append((int(found.group(1)), float(found.group(2)), float(found.group(3)),
                                 int(possession.group(1)) if possession else 0,
                                 int(status.group(1)) if status else -1))
                else:
                    rows.append((int(found.group(1)), float(found.group(2)),
                                 float(found.group(3))))
            elif line.startswith("<FrameSet"):
                half, team, person = _FRAMESET.search(line).groups()
                is_ball = team == "BALL"
                current, rows = (half, "ball" if is_ball else (team, person)), []
            elif line.startswith("</FrameSet"):
                half, key = current
                out.setdefault(half, {})[key] = np.array(rows, dtype=np.float64)
                current, rows = None, []
    return out


def build(match_id, data_dir=DATA_DIR):
    info_path, positions_path = files(match_id, data_dir)
    home, guest = team_ids(info_path)
    length, width = pitch_size(positions_path)
    sections = parse_positions(positions_path)

    halves_present = [h for h in HALF_ORDER if h in sections and "ball" in sections[h]]
    # Only the two teams' players. Some matches also track the referees (TeamId "referee");
    # counted as players they would take the ball in nearest-player rules.
    people = sorted({key for h in halves_present for key in sections[h]
                     if key != "ball" and key[0] in (home, guest)},
                    key=lambda k: (k[0] != home, k[1]))
    column = {person: i for i, person in enumerate(people)}
    team = np.array([HOME if t == home else AWAY for t, _ in people], dtype=np.int8)

    spans, offset = [], 0
    for half in halves_present:
        frames = sections[half]["ball"][:, 0]
        first, n = int(frames.min()), int(frames.max() - frames.min() + 1)
        spans.append((half, first, offset, n))
        offset += n + HALF_GAP
    total = offset - HALF_GAP

    player_x = np.full((total, len(people)), np.nan, dtype=np.float32)
    player_y = np.full((total, len(people)), np.nan, dtype=np.float32)
    ball_x = np.full(total, np.nan, dtype=np.float32)
    ball_y = np.full(total, np.nan, dtype=np.float32)
    labels = np.full(total, pm.X, dtype=np.int8)
    halves = []

    for half, first, start, n in spans:
        halves.append((start, start + n))
        for key, rows in sections[half].items():
            if key != "ball" and key not in column:
                continue
            index = start + (rows[:, 0].astype(np.int64) - first)
            if key == "ball":
                ball_x[index] = rows[:, 1] + length / 2
                ball_y[index] = rows[:, 2] + width / 2
                status, possession = rows[:, 4], rows[:, 3]
                labels[index] = np.where(status == 0, pm.D,
                                         np.where(possession == 1, pm.H,
                                                  np.where(possession == 2, pm.A, pm.X)))
            else:
                player_x[index, column[key]] = rows[:, 1] + length / 2
                player_y[index, column[key]] = rows[:, 2] + width / 2

    match = Match("idsse_" + match_id, FPS, (length, width), player_x, player_y, team,
                  ball_x, ball_y, halves, labels, source="idsse")
    unknown = int(np.sum(labels[match.in_play_mask()] == pm.X))
    if unknown:
        raise ValueError("{}: {} in-play frames have no usable BallStatus/BallPossession".format(
            match.name, unknown))
    return validate(match)


def available(match_id, data_dir=DATA_DIR):
    """True if load(match_id) can work: its cache is built, or its raw files are downloaded."""
    if os.path.isfile(cache_path("idsse_{}_v{}".format(match_id, CACHE_VERSION))):
        return True
    try:
        files(match_id, data_dir)
    except IOError:
        return False
    return True


# Which possession labels load() returns (D20). "dfl" is DFL's BallPossession, which flips at
# the first touch; "keep2" and "keep2pass" are solution/relabel.py's controlled-possession
# relabellings. Set by tools/proxy_league.py --labels (or MTAH_PROXY_LABELS) before anything
# loads, so the truth AND every prior fitted from load() (decoder, touch, gapteam, deadball)
# follow the same labels. The default stays "dfl" until a decision says otherwise.
LABELS = os.environ.get("MTAH_PROXY_LABELS", "dfl")


def relabelled(match, match_id, variant):
    """match with its labels turned into `variant` (solution/relabel.py); a copy unless dfl."""
    from solution import relabel
    if variant == "dfl":
        return match
    if variant not in relabel.VARIANTS:
        raise ValueError("unknown label variant {!r}; one of {}".format(variant, relabel.VARIANTS))
    confirm = None
    if variant == "keep2pass":
        confirm = relabel.pass_confirm(completed_passes(match_id), match.fps)
    out = match.copy()
    out.labels = relabel.controlled(match.labels, match.fps, relabel.KEEP_S, confirm)
    return validate(out)


def load(match_id, refresh=False, labels=None):
    """The clean match. `labels` (default the module's LABELS) picks the possession labels."""
    match = validate(cached("idsse_{}_v{}".format(match_id, CACHE_VERSION),
                            lambda: build(match_id), refresh))
    return relabelled(match, match_id, LABELS if labels is None else labels)


# -- restart events, for measuring dead-ball errors by restart type (C2b) ----------------------

# DFL event elements that restart play, and the names tools print. A KickOff that is not the
# first of its half follows a goal, so it is reported as "goal".
RESTART_KINDS = {"KickOff": "kickoff", "ThrowIn": "throw_in", "GoalKick": "goal_kick",
                 "CornerKick": "corner", "FreeKick": "free_kick", "Penalty": "penalty",
                 "RefereeBall": "referee_ball"}
EARLY_S = 5.0     # an event this far before a half's first frame still belongs to it


def _seconds(stamp):
    return datetime.fromisoformat(stamp).timestamp()


def half_clocks(match_id, data_dir=DATA_DIR):
    """[(half, first N, last N, epoch seconds of first N)] of the ball, in HALF_ORDER.

    One pass over the positions file, cached as JSON: build()'s timeline is fixed by the first
    and last ball frame of each half, so this reproduces it without parsing every frame.
    """
    path = os.path.join(CACHE_DIR, "idsse_clock_{}.json".format(match_id))
    if os.path.isfile(path):
        with open(path) as handle:
            return [tuple(row) for row in json.load(handle)]
    _, positions_path = files(match_id, data_dir)
    found, half, first, last = {}, None, None, None
    with open(positions_path) as handle:
        for line in handle:
            if line.startswith("<FrameSet"):
                section, team, _ = _FRAMESET.search(line).groups()
                half = section if team == "BALL" else None
                first = last = None
            elif half and line.startswith("<Frame "):
                if first is None:
                    first = (int(_FRAME.match(line).group(1)),
                             _seconds(re.search(r' T="([^"]+)"', line).group(1)))
                last = line
            elif half and line.startswith("</FrameSet"):
                found[half] = (half, first[0], int(_FRAME.match(last).group(1)), first[1])
                half = None
    rows = [found[h] for h in HALF_ORDER if h in found]
    if not os.path.isdir(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    with open(path, "w") as handle:
        json.dump(rows, handle)
    return rows


def _events(match_id, data_dir=DATA_DIR):
    """[(frame, Event element)] on load(match_id)'s timeline, for events inside a half.

    Frames come from EventTime against the ball's frame clock, so they carry the event feed's
    timing error (manual, about a second). Deleted events are not in the feed.
    """
    found = glob.glob(os.path.join(data_dir, "DFL_03_02_events_raw_*DFL-MAT-{}.xml".format(
        match_id)))
    if len(found) != 1:
        raise IOError("IDSSE {}: expected one events file, found {}. Run `make external` "
                      "first.".format(match_id, len(found)))
    spans, offset = [], 0
    for _, first, last, clock in half_clocks(match_id, data_dir):
        spans.append((offset, last - first + 1, clock))
        offset += last - first + 1 + HALF_GAP
    out = []
    for event in ET.parse(found[0]).getroot().iter("Event"):
        when = _seconds(event.get("EventTime"))
        for start, n, clock in spans:
            index = int(round((when - clock) * FPS))
            if -EARLY_S * FPS <= index < n:
                out.append((start + max(0, index), event))
                break
    return out


def completed_passes(match_id, data_dir=DATA_DIR):
    """(frames, codes) of DFL `Play` events evaluated successfullyCompleted, codes pm.H / pm.A.

    A completed pass reaches a team-mate, so the passing team had the ball under control
    (relabel.pass_confirm). Cached as JSON: the events file takes a second or two to parse.
    """
    path = os.path.join(CACHE_DIR, "idsse_passes_{}.json".format(match_id))
    if os.path.isfile(path):
        with open(path) as handle:
            frames, codes = json.load(handle)
        return np.array(frames, dtype=np.int64), np.array(codes, dtype=np.int8)
    info_path, _ = files(match_id, data_dir)
    home, guest = team_ids(info_path)
    frames, codes = [], []
    for frame, event in _events(match_id, data_dir):
        for play in event.iter("Play"):
            if play.get("Evaluation") == "successfullyCompleted" and \
                    play.get("Team") in (home, guest):
                frames.append(int(frame))
                codes.append(int(pm.H if play.get("Team") == home else pm.A))
    if not os.path.isdir(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    with open(path, "w") as handle:
        json.dump([frames, codes], handle)
    return np.array(frames, dtype=np.int64), np.array(codes, dtype=np.int8)


def restarts(match_id, data_dir=DATA_DIR):
    """[(frame, kind)] of the restart events, on load(match_id)'s timeline, in time order.

    Frames come from EventTime against the ball's frame clock, so they carry the event feed's
    timing error (manual, about a second). Deleted events are not in the feed as restarts.
    """
    spans, offset = [], 0
    for _, first, last, _ in half_clocks(match_id, data_dir):
        spans.append((offset, last - first + 1))
        offset += last - first + 1 + HALF_GAP
    out = []
    for frame, event in _events(match_id, data_dir):
        child = next(iter(event), None)
        if child is not None and child.tag in RESTART_KINDS:
            out.append((frame, RESTART_KINDS[child.tag]))
    out.sort()
    seen_halves, named = set(), []
    for frame, kind in out:
        half = next(i for i, (start, n) in enumerate(spans) if frame < start + n)
        if kind == "kickoff":
            kind = "kickoff" if half not in seen_halves else "goal"
            seen_halves.add(half)
        named.append((frame, kind))
    return named


def load_all(refresh=False, labels=None):
    return [load(match_id, refresh, labels) for match_id in MATCH_IDS]
