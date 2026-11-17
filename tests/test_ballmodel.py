"""Tests for E15, ball inference in gaps (solution/ballmodel.py).

Synthetic inputs only, no data needed:
    .venv/bin/python -m unittest tests.test_ballmodel -v        # or: make test
"""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import ballmodel  # noqa: E402
from solution.match import Match, pm  # noqa: E402

FPS = 25


def walking_match(n=3000, seed=0, hide=((500, 900), (1500, 2400))):
    """10 players drifting; the ball follows player 0 (home) then player 5 (away); gaps hidden."""
    rng = np.random.default_rng(seed)
    team = np.array([0] * 5 + [1] * 5, dtype=np.int8)
    x = rng.uniform(10, 95, 10) + np.cumsum(rng.normal(0, 0.1, (n, 10)), axis=0)
    y = rng.uniform(5, 63, 10) + np.cumsum(rng.normal(0, 0.1, (n, 10)), axis=0)
    owner = np.where(np.arange(n) < n // 2, 0, 5)
    bx = x[np.arange(n), owner] + 0.5
    by = y[np.arange(n), owner]
    labels = np.where(owner == 0, pm.H, pm.A).astype(np.int8)
    clean = Match("synthetic_ball", FPS, (105.0, 68.0), x.astype(np.float32),
                  y.astype(np.float32), team, bx.astype(np.float32), by.astype(np.float32),
                  [(0, n)], labels)
    degraded = clean.copy()
    for a, b in hide:
        degraded.ball_x[a:b] = np.nan
        degraded.ball_y[a:b] = np.nan
    return degraded, clean


class GapEndpoints(unittest.TestCase):
    def test_prev_and_next_seen_within_half(self):
        seen = np.array([1, 0, 0, 1, 1, 0, 0, 1, 0, 1], dtype=bool)
        prev, nxt = ballmodel._gap_endpoints(seen, [(0, 5), (5, 10)], 10)
        self.assertEqual(prev.tolist(), [0, 0, 0, 3, 4, -1, -1, 7, 7, 9])
        self.assertEqual(nxt.tolist(), [0, 3, 3, 3, 4, 7, 7, 7, 9, 9])


class Reader(unittest.TestCase):
    def test_symmetric_and_signed(self):
        rng = np.random.default_rng(1)
        z = rng.normal(0, 8, 5000)
        labels = np.where(z + rng.normal(0, 4, 5000) > 0, pm.H, pm.A).astype(np.int8)
        table = ballmodel.fit_reader([(z, labels)])
        self.assertTrue(np.allclose(table, -table[::-1]))      # swapping teams flips the sign
        self.assertGreater(table[-1], 1.0)                     # home much closer: home
        self.assertLess(table[0], -1.0)

    def test_read_only_where_asked(self):
        table = np.linspace(-1, 1, len(ballmodel.Z_EDGES) - 1)
        z = np.array([5.0, np.nan, -5.0])
        out = ballmodel.read(table, z, np.array([True, True, False]))
        self.assertTrue(np.isfinite(out[0]))
        self.assertTrue(np.isnan(out[1]) and np.isnan(out[2]))


class Imputer(unittest.TestCase):
    def test_imputes_only_gaps_and_is_deterministic(self):
        degraded, clean = walking_match()
        params = dict(ballmodel.PARAMS, max_iter=30, step=2)
        models = ballmodel.fit_imputer([walking_match(seed=s) for s in (2, 3)], params)
        bx, by = ballmodel.impute(models, degraded)
        seen = degraded.ball_seen
        self.assertTrue(np.array_equal(bx[seen], degraded.ball_x[seen].astype(float)))
        self.assertTrue(np.isfinite(bx).all() and np.isfinite(by).all())
        again = ballmodel.impute(ballmodel.fit_imputer([walking_match(seed=s) for s in (2, 3)],
                                                       params), degraded)
        self.assertTrue(np.array_equal(bx, again[0]) and np.array_equal(by, again[1]))

    def test_oracle_without_noise_is_the_truth(self):
        degraded, clean = walking_match()
        bx, by = ballmodel.oracle_ball(degraded, clean, 0.0, 0.0)
        self.assertTrue(np.allclose(bx, clean.ball_x) and np.allclose(by, clean.ball_y))

    def test_closeness_sign(self):
        degraded, clean = walking_match()
        z = ballmodel.closeness(clean, clean.ball_x.astype(float), clean.ball_y.astype(float))
        n = clean.n_frames
        self.assertGreater(np.mean(z[: n // 2] > 0), 0.9)     # ball on a home player
        self.assertGreater(np.mean(z[n // 2:] < 0), 0.9)     # ball on an away player


if __name__ == "__main__":
    unittest.main()
