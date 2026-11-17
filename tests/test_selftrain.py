"""Tests for C6 self-training (solution/selftrain.py). Synthetic inputs only, no data needed:
    .venv/bin/python -m unittest tests.test_selftrain -v        # or: make test
"""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import selftrain  # noqa: E402
from solution.match import Match  # noqa: E402

FPS = 25


def blank_match(n=3000, seen=None):
    ball = np.zeros(n, dtype=np.float32)
    if seen is not None:
        ball[~seen] = np.nan
    x = np.zeros((n, 2), dtype=np.float32)
    return Match("synthetic_c6", FPS, (105.0, 68.0), x, x.copy(), np.array([0, 1], np.int8),
                 ball, ball.copy(), [(0, n)])


class HolderRunsTest(unittest.TestCase):
    def test_only_long_runs_of_a_team(self):
        holder = np.array([-1] * 10 + [0] * 25 + [1] * 10 + [-1] * 30 + [1] * 30, np.int8)
        runs = selftrain.holder_runs(holder, FPS, 1.0)
        expected = np.zeros(len(holder), bool)
        expected[10:35] = True
        expected[75:105] = True
        np.testing.assert_array_equal(runs, expected)

    def test_pickable_needs_a_seen_ball(self):
        n = 200
        seen = np.ones(n, bool)
        seen[50:100] = False
        match = blank_match(n, seen)
        holder = np.zeros(n, np.int8)
        picked = selftrain.pickable(match, holder)
        np.testing.assert_array_equal(picked, seen)


class AdaptTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        n = 6000
        self.match = blank_match(n)
        self.target = (np.arange(n) // 300 % 2).astype(np.int8)        # alternating teams
        # One feature knows the team; the base model is uninformative.
        self.X = np.column_stack((self.target + rng.normal(0, 0.5, n),
                                  rng.normal(0, 1, n))).astype(np.float32)
        self.base = np.zeros(n)

    def test_learns_from_picked_frames_and_is_cross_fitted(self):
        picked = np.ones(self.match.n_frames, bool)
        for kind in ("gbm", "logit"):          # "pooled" needs the IDSSE training set
            params = dict(selftrain.PARAMS, kind=kind)
            out = selftrain.adapt(self.match, self.base, self.X, picked, self.target, params)
            accuracy = np.mean((out > 0) == (self.target == 1))
            self.assertGreater(accuracy, 0.8, kind)

    def test_deterministic(self):
        picked = np.ones(self.match.n_frames, bool)
        a = selftrain.adapt(self.match, self.base, self.X, picked, self.target)
        b = selftrain.adapt(self.match, self.base, self.X, picked, self.target)
        np.testing.assert_array_equal(a, b)

    def test_too_few_frames_keeps_the_base(self):
        picked = np.zeros(self.match.n_frames, bool)
        picked[:100] = True
        base = np.linspace(-1, 1, self.match.n_frames)
        out = selftrain.adapt(self.match, base, self.X, picked, self.target)
        np.testing.assert_array_equal(out, base)


if __name__ == "__main__":
    unittest.main()
