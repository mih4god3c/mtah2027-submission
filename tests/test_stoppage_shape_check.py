"""tools/stoppage_shape_check.py's helpers on a synthetic match: no data needed."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from solution.match import AWAY, HOME, Match  # noqa: E402
from tools import stoppage_shape_check as sc  # noqa: E402


def toy_match(n=100, fps=10):
    """Home on the left (defends x = 0), away on the right; 5 players each, keeper deepest."""
    team = np.array([HOME] * 5 + [AWAY] * 5, np.int8)
    px = np.tile(np.array([2.0, 20, 30, 40, 50, 103, 85, 75, 65, 55], np.float32), (n, 1))
    py = np.tile(np.array([34.0, 20, 30, 40, 50, 34, 20, 30, 40, 50], np.float32), (n, 1))
    bx = np.full(n, 52.5, np.float32)
    by = np.full(n, 34.0, np.float32)
    return Match("toy", fps, (105.0, 68.0), px, py, team, bx, by, [(0, n)])


class StoppageShapeTest(unittest.TestCase):
    def test_runs_and_sustained(self):
        mask = np.array([0, 1, 1, 0, 1, 1, 1, 1, 0], bool)
        self.assertEqual(sc.runs_of(mask), [(1, 3), (4, 8)])
        # a window of 3 fits only the second run, and marks all of it
        self.assertEqual(sc.sustained(mask, 3).tolist(),
                         [False, False, False, False, True, True, True, True, False])

    def test_keeper_hold_needs_ball_team_and_box(self):
        m = toy_match()
        truth = np.full(m.n_frames, pm.H, np.int8)
        # the home keeper holds the ball for 2 s (frames 20-39), then it goes upfield
        m.ball_x[20:40], m.ball_y[20:40] = 2.5, 34.0
        holds = sc.keeper_holds(m, truth)
        self.assertEqual(holds, [(20, 40, pm.H)])
        # the same frames with the truth on away: not a hold of either keeper
        self.assertEqual(sc.keeper_holds(m, np.full(m.n_frames, pm.A, np.int8)), [])
        # too short (under MIN_HOLD_S): nothing
        m2 = toy_match()
        m2.ball_x[20:25], m2.ball_y[20:25] = 2.5, 34.0
        self.assertEqual(sc.keeper_holds(m2, truth), [])

    def test_shape_features_swap_invariant(self):
        m = toy_match()
        m.player_y[:, 5:] = 60.0                       # away crowds one touchline
        f = sc.shape_features(m)
        swapped = toy_match()
        swapped.player_y[:, 5:] = 60.0
        swapped.team = 1 - swapped.team
        g = sc.shape_features(swapped)
        for name in sc.FEATURES:
            np.testing.assert_allclose(f[name], g[name])
        self.assertAlmostEqual(float(f["lat_max"][0]), 26.0, places=4)


if __name__ == "__main__":
    unittest.main()
