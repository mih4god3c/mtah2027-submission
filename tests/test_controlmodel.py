"""Tests for the #3 control model (solution/controlmodel.py). Synthetic inputs only, no data:
    .venv/bin/python -m unittest tests.test_controlmodel -v       # or: make test
"""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import controlmodel, touch  # noqa: E402
from solution.match import pm  # noqa: E402
from tests.test_touch import ARRIVE, FPS, KICK, N, kick_match  # noqa: E402


def team_swapped(match):
    out = match.copy()
    out.team = (1 - match.team).astype(np.int8)
    return out


class Verdicts(unittest.TestCase):

    def test_majority_of_the_next_second_decides(self):
        truth = np.full(200, pm.H, dtype=np.int8)
        truth[100:] = pm.A
        truth[150:] = pm.D
        frames = np.array([10, 10, 90, 130, 160])
        teams = np.array([pm.H, pm.A, pm.A, pm.A, pm.H], dtype=np.int8)
        got = controlmodel.verdicts(truth, frames, teams, FPS)
        C, A, U = controlmodel.CONTRADICTED, controlmodel.AGREED, controlmodel.NEUTRAL
        # 90: 10 H frames then 15 A frames in [90, 115): A has the majority. 130: 20 A frames
        # then 5 D: A. 160: all D: neutral.
        self.assertEqual(got.tolist(), [A, C, A, A, U])

    def test_teamless_events_are_neutral(self):
        truth = np.full(50, pm.H, dtype=np.int8)
        got = controlmodel.verdicts(truth, np.array([5]), np.array([-1], dtype=np.int8), FPS)
        self.assertEqual(got.tolist(), [controlmodel.NEUTRAL])


class Features(unittest.TestCase):

    def test_shape_names_and_the_reception(self):
        match = kick_match()
        touches = touch.detect(match)
        X, names = controlmodel.features(match, touches)
        self.assertEqual(X.shape, (len(touches.touch_frame), len(names)))
        self.assertEqual(len(set(names)), len(names))
        # The away receiver's touch: away is nearest the ball for the next second, and the home
        # team held it just before.
        i = int(np.argmin(np.abs(touches.touch_frame - ARRIVE) +
                          100 * (touches.touch_team != 1)))
        row = dict(zip(names, X[i]))
        self.assertGreater(row["own_share_0_1"], 0.9)
        self.assertEqual(row["held_before"], -1.0)
        self.assertLess(row["own_m"], row["opp_m"])

    def test_relative_to_the_touching_team(self):
        match = kick_match()
        a, _ = controlmodel.features(match, touch.detect(match))
        swapped = team_swapped(match)
        b, _ = controlmodel.features(swapped, touch.detect(swapped))
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)     # 1 - share rounds

    def test_deterministic(self):
        match = kick_match()
        a, _ = controlmodel.features(match, touch.detect(match))
        b, _ = controlmodel.features(match, touch.detect(match))
        np.testing.assert_array_equal(a, b)


class Fit(unittest.TestCase):

    def test_fit_is_deterministic_and_skips_neutral(self):
        rng = np.random.RandomState(0)
        X = rng.normal(size=(400, 3))
        v = np.where(X[:, 0] > 0.5, controlmodel.CONTRADICTED, controlmodel.AGREED)
        v[:20] = controlmodel.NEUTRAL
        X[:20] = 1e6                                    # would wreck the fit if used
        params = dict(controlmodel.PARAMS, max_iter=20, min_samples_leaf=10)
        one = controlmodel.fit([(X, v)], params).predict_proba(X[20:])[:, 1]
        two = controlmodel.fit([(X, v)], params).predict_proba(X[20:])[:, 1]
        np.testing.assert_array_equal(one, two)
        judged = v[20:]
        self.assertGreater(one[judged == controlmodel.CONTRADICTED].mean(),
                           one[judged == controlmodel.AGREED].mean() + 0.5)


if __name__ == "__main__":
    unittest.main()
