"""Tests for the better-tracker what-if, simulate.reveal_gaps / degrade(ball_visible=). No data.

    .venv/bin/python -m unittest tests.test_ball_visible -v
"""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import simulate  # noqa: E402
from tests.test_proxy_league import synthetic_match  # noqa: E402


def visible_share(match):
    return float(np.mean([match.ball_seen[a:b].mean() for a, b in match.halves]))


def long_match():
    """Two long halves, so each holds dozens of BAR-ATH-length gaps."""
    match = synthetic_match(n_frames=60000, seed=4)
    match.halves = [(0, 29000), (30000, 60000)]
    return match


class RevealGaps(unittest.TestCase):

    def test_none_is_identity(self):
        rng = np.random.RandomState(0)
        hide = rng.rand(1000) < 0.4
        self.assertTrue(np.array_equal(simulate.reveal_gaps(hide, None, rng), hide))

    def test_only_whole_gaps_are_revealed(self):
        hide = np.zeros(1000, dtype=bool)
        for a, b in ((50, 150), (300, 420), (600, 700), (800, 990)):
            hide[a:b] = True
        out = simulate.reveal_gaps(hide, 0.8, np.random.RandomState(3))
        self.assertFalse(np.any(out & ~hide))                  # nothing new hidden
        for a, b in ((50, 150), (300, 420), (600, 700), (800, 990)):
            self.assertIn(int(out[a:b].sum()), (0, b - a))     # each gap all or nothing
        self.assertLess(out.sum(), hide.sum())

    def test_already_visible_enough_is_unchanged(self):
        hide = np.zeros(1000, dtype=bool)
        hide[100:150] = True
        out = simulate.reveal_gaps(hide, 0.9, np.random.RandomState(0))
        self.assertTrue(np.array_equal(out, hide))


class DegradeBallVisible(unittest.TestCase):

    def test_default_is_byte_identical(self):
        match = synthetic_match()
        for version in simulate.VERSIONS:
            a = simulate.degrade(match, 2, version)
            b = simulate.degrade(match, 2, version, ball_visible=None)
            for field in ("ball_x", "ball_y", "player_x", "player_y", "team"):
                self.assertTrue(np.array_equal(getattr(a, field), getattr(b, field),
                                               equal_nan=field != "team"), field)

    def test_share_moves_to_target(self):
        match = long_match()
        for version in simulate.VERSIONS:
            base = simulate.degrade(match, 1, version)
            for target in (0.76, 0.9):
                out = simulate.degrade(match, 1, version, ball_visible=target)
                self.assertGreater(visible_share(out), visible_share(base))
                self.assertAlmostEqual(visible_share(out), target, delta=0.04)
                # only reveals: every frame seen before is still seen, players untouched
                self.assertFalse(np.any(base.ball_seen & ~out.ball_seen))
                self.assertTrue(np.array_equal(base.player_x, out.player_x, equal_nan=True))
                revealed = out.ball_seen & ~base.ball_seen
                self.assertTrue(np.array_equal(out.ball_x[revealed], match.ball_x[revealed]))

    def test_seeded(self):
        match = long_match()
        a = simulate.degrade(match, 5, "v1", ball_visible=0.9)
        b = simulate.degrade(match, 5, "v1", ball_visible=0.9)
        self.assertTrue(np.array_equal(a.ball_x, b.ball_x, equal_nan=True))


if __name__ == "__main__":
    unittest.main()
