"""One schema for every match we touch: BAR-ATH and the external matches we validate on.

Everything downstream -- features, models, the decoder, the proxy league -- takes a Match, so
the same code runs unchanged on the competition match and on every external one. That is the
whole point of this module: without one schema there is no proxy league (PLAN.md, section 6).

Conventions
-----------
- Frames are video frames at `fps` (25 for everything we use), indexed from 0.
- Positions are metres, origin at a pitch corner: x in [0, length], y in [0, width]. No attack
  direction normalisation here; that is a later step (C1 in PLAN.md).
- Missing observations are NaN, for players and ball alike.
- `team[s]` is 0 for home and 1 for away, the same codes starter/baseline.py uses.
- `labels`, when known, uses the scorer's codes (possession_metrics: H=0, A=1, D=2, X=3) and is X
  outside the halves, so possession_metrics.valid_runs() recovers the halves from it.
- `halves` are half-open [start, stop) frame spans of play.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _path in (ROOT, os.path.join(ROOT, "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
import possession_metrics as pm  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "cache")       # gitignored; derived arrays, never committed
HOME, AWAY = 0, 1


@dataclass
class Match:
    name: str
    fps: int
    pitch: Tuple[float, float]                 # (length, width) in metres
    player_x: np.ndarray                       # (T, S) float32
    player_y: np.ndarray                       # (T, S) float32
    team: np.ndarray                           # (S,) int8, 0 home / 1 away
    ball_x: np.ndarray                         # (T,) float32
    ball_y: np.ndarray                         # (T,) float32
    halves: List[Tuple[int, int]]
    labels: Optional[np.ndarray] = None        # (T,) int8 scorer codes, or None if unknown
    source: str = ""
    notes: dict = field(default_factory=dict)

    @property
    def n_frames(self):
        return len(self.ball_x)

    @property
    def ball_seen(self):
        return np.isfinite(self.ball_x) & np.isfinite(self.ball_y)

    def in_play_mask(self):
        mask = np.zeros(self.n_frames, dtype=bool)
        for start, stop in self.halves:
            mask[start:stop] = True
        return mask

    def copy(self):
        return Match(self.name, self.fps, self.pitch, self.player_x.copy(), self.player_y.copy(),
                     self.team.copy(), self.ball_x.copy(), self.ball_y.copy(), list(self.halves),
                     None if self.labels is None else self.labels.copy(), self.source,
                     dict(self.notes))

    # -- cache ---------------------------------------------------------------------------------

    def save(self, path):
        folder = os.path.dirname(path)
        if folder and not os.path.isdir(folder):
            os.makedirs(folder)
        np.savez(path, name=self.name, fps=self.fps, pitch=np.array(self.pitch),
                 player_x=self.player_x, player_y=self.player_y, team=self.team,
                 ball_x=self.ball_x, ball_y=self.ball_y, halves=np.array(self.halves),
                 labels=self.labels if self.labels is not None else np.array([], dtype=np.int8),
                 source=self.source)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        labels = z["labels"]
        return cls(str(z["name"]), int(z["fps"]), tuple(float(v) for v in z["pitch"]),
                   z["player_x"], z["player_y"], z["team"], z["ball_x"], z["ball_y"],
                   [tuple(int(v) for v in row) for row in z["halves"]],
                   labels if len(labels) else None, str(z["source"]))


def cache_path(name):
    return os.path.join(CACHE_DIR, "matches", name + ".npz")


def cached(name, build, refresh=False):
    """Match from the cache, or build() it and cache it. build() must be deterministic."""
    path = cache_path(name)
    if os.path.isfile(path) and not refresh:
        return Match.load(path)
    match = build()
    match.save(path)
    return match


def validate(match):
    """Raise on anything that would make scores silently wrong. Cheap; call after every load."""
    n, s = match.n_frames, len(match.team)
    for name in ("player_x", "player_y"):
        if getattr(match, name).shape != (n, s):
            raise ValueError("{}: {} has shape {}, expected {}".format(
                match.name, name, getattr(match, name).shape, (n, s)))
    if match.ball_y.shape != (n,):
        raise ValueError("{}: ball arrays disagree in length".format(match.name))
    if not set(np.unique(match.team).tolist()) <= {HOME, AWAY}:
        raise ValueError("{}: team codes outside 0/1".format(match.name))
    previous = 0
    for start, stop in match.halves:
        if not previous <= start < stop <= n:
            raise ValueError("{}: bad half span {}".format(match.name, (start, stop)))
        previous = stop
    if match.labels is not None:
        if match.labels.shape != (n,):
            raise ValueError("{}: labels length {} != {}".format(match.name, len(match.labels), n))
        outside = ~match.in_play_mask()
        if np.any(match.labels[outside] != pm.X):
            raise ValueError("{}: labels outside the halves must be X".format(match.name))
        runs = pm.valid_runs(match.labels)
        if len(runs) != len(match.halves):
            # A half that starts or ends in X would shift the scorer's runs; catch it here.
            raise ValueError("{}: labels give runs {} but halves are {}".format(
                match.name, runs, match.halves))
    return match
