"""BAR-ATH tactical ATD -> Match. The competition match, in the same schema as external ones.

Reads the raw FIFA/EPTS text file with the standard library, exactly as starter/baseline.py
does, so it needs no kloppy and agrees with the published baseline's reading of the file:

  - video_frame = frame_id - 1 (load_atd.py);
  - slot teams from baseline.team_of_each_track (first team in the metadata is home);
  - positions are normalised 0-1 and scaled by games.GAMES["BAR-ATH"]["pitch"] to metres;
  - halves are games.half_video_blocks, the published graded spans.

labels is always None here: the full-match truth is private and the public 5 minutes are not
for training or tuning. tools/check_bar_ath_loader.py verifies this loader against the committed
baseline files without printing any data.
"""

import os
import sys

import numpy as np

from solution.match import ROOT, Match, cached, validate

sys.path.insert(0, os.path.join(ROOT, "starter"))
import baseline  # noqa: E402
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402

FIELD_SEP, GROUP_SEP = ":", ";"


def parse_raw(raw_path, n_slots, n_frames, length, width):
    player_x = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    player_y = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    ball_x = np.full(n_frames, np.nan, dtype=np.float32)
    ball_y = np.full(n_frames, np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            if not 0 <= frame < n_frames:
                continue
            ball = parts[2]
            if ball[0] != "N":
                fields = ball.split(",")
                ball_x[frame] = float(fields[0]) * length
                ball_y[frame] = float(fields[1]) * width
            for slot, group in enumerate(parts[1].split(GROUP_SEP)[:n_slots]):
                if group and group[0] != "N":
                    fields = group.split(",")
                    player_x[frame, slot] = float(fields[0]) * length
                    player_y[frame, slot] = float(fields[1]) * width
    return player_x, player_y, ball_x, ball_y


def build(game=GAME, data_dir=None):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    team = baseline.team_of_each_track(meta_path)
    length, width = GAMES[game]["pitch"]
    n_frames = source_frames("tactical", game)
    player_x, player_y, ball_x, ball_y = parse_raw(raw_path, len(team), n_frames, length, width)
    halves = [(first, last + 1) for _, first, last in half_video_blocks(game)]
    return validate(Match(game, GAMES[game]["fps"], (length, width), player_x, player_y,
                          team.astype(np.int8), ball_x, ball_y, halves, None, source="atd"))


def load(refresh=False, data_dir=None):
    return validate(cached(GAME, lambda: build(GAME, data_dir), refresh))


def build_broadcast(game=GAME, data_dir=None):
    """The broadcast ATD as a Match on its own timeline (metres, one 'half' spanning it all).
    Not aligned to the tactical video: solution/broadcast.py does that."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir, "broadcast")
    team = baseline.team_of_each_track(meta_path)
    length, width = GAMES[game]["pitch"]
    n_frames = source_frames("broadcast", game)
    player_x, player_y, ball_x, ball_y = parse_raw(raw_path, len(team), n_frames, length, width)
    return validate(Match(game + "-broadcast", GAMES[game]["fps"], (length, width), player_x,
                          player_y, team.astype(np.int8), ball_x, ball_y, [(0, n_frames)], None,
                          source="atd-broadcast"))


def load_broadcast(refresh=False, data_dir=None):
    return validate(cached(GAME + "-broadcast", lambda: build_broadcast(GAME, data_dir), refresh))
