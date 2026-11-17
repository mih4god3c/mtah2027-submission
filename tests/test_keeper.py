"""solution/keeper.py, solution/keepermodel.py's zone and features, the decoder's keeper term:
on synthetic matches, no data needed."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from solution import decoder, keeper, keepermodel  # noqa: E402
from solution.match import AWAY, HOME, Match  # noqa: E402


def toy_match(n=200, fps=10):
    """Home defends x = 0 (keeper at x 2), away defends x = 105 (keeper at x 103)."""
    team = np.array([HOME] * 5 + [AWAY] * 5, np.int8)
    px = np.tile(np.array([2.0, 20, 30, 40, 50, 103, 85, 75, 65, 55], np.float32), (n, 1))
    py = np.tile(np.array([34.0, 20, 30, 40, 50, 34, 20, 30, 40, 50], np.float32), (n, 1))
    bx = np.full(n, 52.5, np.float32)
    by = np.full(n, 34.0, np.float32)
    return Match("toy", fps, (105.0, 68.0), px, py, team, bx, by, [(0, n)])


def mirrored(m):
    """The same play with the teams swapped and the pitch turned: a swap of colours and sides."""
    out = m.copy()
    out.team = (1 - m.team).astype(np.int8)
    out.player_x = (105.0 - m.player_x).astype(np.float32)
    out.player_y = (68.0 - m.player_y).astype(np.float32)
    out.ball_x = (105.0 - m.ball_x).astype(np.float32)
    out.ball_y = (68.0 - m.ball_y).astype(np.float32)
    return out


class KeeperGeometryTest(unittest.TestCase):
    def test_keepers_and_depth(self):
        m = toy_match()
        k = keeper.keepers(m)
        self.assertTrue(np.all(k[:, HOME] == 0) and np.all(k[:, AWAY] == 5))
        self.assertAlmostEqual(float(keeper.depth(m, AWAY, np.array([103.0]))[0]), 2.0)

    def test_on_ball_and_hide(self):
        m = toy_match()
        m.ball_x[50:80], m.ball_y[50:80] = 2.5, 34.0          # home keeper has it for 3 s
        m.ball_x[120:125], m.ball_y[120:125] = 103.0, 34.5    # away keeper, too short
        self.assertEqual(keeper.on_ball(m), [(50, 80, HOME)])
        hidden = keeper.hide_keeper_ball(m, m, 0.0)
        self.assertTrue(np.all(np.isnan(hidden.ball_x[50:80])))
        self.assertTrue(np.all(np.isfinite(hidden.ball_x[:50])))
        self.assertTrue(np.array_equal(keeper.hide_keeper_ball(m, m, 1.0).ball_x, m.ball_x))

    def test_lose_ball_near_goal(self):
        m = toy_match()
        m.ball_x[40:60], m.ball_y[40:60] = -1.0, 30.0         # out behind the home goal
        m.ball_x[60:70], m.ball_y[60:70] = 15.0, 34.0         # carried back (no one near)
        m.ball_x[70:100], m.ball_y[70:100] = 2.5, 34.0        # with the home keeper, 3 s
        lost = keeper.lose_ball_near_goal(m, m, 0.0)
        # out: hidden from the crossing until the keeper has it; hold: all but the last 1 s
        self.assertTrue(np.all(np.isnan(lost.ball_x[40:90])))
        self.assertTrue(np.all(np.isfinite(lost.ball_x[90:100])))
        self.assertTrue(np.all(np.isfinite(lost.ball_x[:40])))
        only_out = keeper.lose_ball_near_goal(m, m, 0.0, keep_hold=1.0)
        self.assertTrue(np.all(np.isnan(only_out.ball_x[40:70])))
        self.assertTrue(np.all(np.isfinite(only_out.ball_x[70:100])))

    def test_gaps_by_keeper(self):
        m = toy_match()
        m.ball_x[40:60], m.ball_y[40:60] = 10.0, 34.0         # ball near the box, then lost
        m.ball_x[60:90] = np.nan
        m.ball_x[90:100], m.ball_y[90:100] = 2.5, 34.0        # reappears with the home keeper
        g = keeper.gaps_by_keeper(m)
        lengths, after = g["after"]
        self.assertEqual(lengths.tolist(), [3.0])
        self.assertEqual(after.tolist(), [True])
        self.assertEqual(g["before"][1].tolist(), [False])


class KeeperModelFeaturesTest(unittest.TestCase):
    def test_zone_and_mirror(self):
        m = toy_match()
        m.ball_x[50:80], m.ball_y[50:80] = 3.0, 34.0          # with the home keeper
        m.ball_x[80:120] = np.nan                             # then lost for 4 s
        z = keepermodel.zone(m)
        self.assertTrue(np.all(z[50:120] == HOME))
        self.assertTrue(np.all(z[:50] == -1))
        X, z1 = keepermodel.features(m)
        Y, z2 = keepermodel.features(mirrored(m))
        self.assertTrue(np.all(z2[50:120] == AWAY))
        np.testing.assert_allclose(X[50:120], Y[50:120], atol=1e-4)

    def test_oriented_target(self):
        labels = np.array([pm.H, pm.A, pm.D, pm.H, pm.A, pm.D, pm.H], np.int8)
        z = np.array([HOME, HOME, HOME, AWAY, AWAY, AWAY, -1], np.int8)
        self.assertEqual(keepermodel.oriented_target(labels, z).tolist(), [0, 1, 2, 1, 0, 2, -1])


class DecoderKeeperTermTest(unittest.TestCase):
    def test_zero_weight_and_effect(self):
        n = 50
        nearest = np.full(n, -1, np.int8)
        seen = np.zeros(n, bool)
        speed = np.full(n, 0.5)
        priors = {"agree": 0.8, "speed_log": np.zeros((2, 3)), "speed_edges": np.array([0.0, 1.0])}
        keeper_ll = np.full((n, 3), np.nan)
        keeper_ll[:25] = [0.0, -2.0, -1.0]
        params = dict(decoder.DEFAULTS)
        base = decoder.block_emissions(nearest, seen, speed, priors, params)
        off = decoder.block_emissions(nearest, seen, speed, priors, params, keeper=keeper_ll)
        np.testing.assert_array_equal(base, off)            # keeper_w 0: unchanged
        on = decoder.block_emissions(nearest, seen, speed, priors, dict(params, keeper_w=1.0),
                                     keeper=keeper_ll)
        diff = on - base
        np.testing.assert_allclose(diff[:5], [[0.0, -2.0, -1.0]] * 5)
        np.testing.assert_allclose(diff[5:], 0.0)


if __name__ == "__main__":
    unittest.main()
