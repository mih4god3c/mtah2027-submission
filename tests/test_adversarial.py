"""tools/adversarial_check.py: the helpers it rests on. Synthetic inputs only."""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools import adversarial_check as adv  # noqa: E402
from tests.test_proxy_league import synthetic_match  # noqa: E402


class RunLengths(unittest.TestCase):
    def test_lengths_and_positions(self):
        length, pos = adv.run_lengths(np.array([1, 1, 1, 0, 0, 1], dtype=bool))
        np.testing.assert_array_equal(length, [3, 3, 3, 2, 2, 1])
        np.testing.assert_allclose(pos, [0, 0.5, 1, 0, 1, 0])


class BalancedWeights(unittest.TestCase):
    def test_classes_and_matches_weigh_the_same(self):
        target = np.array([True] * 10 + [False] * 30)
        match = np.array(["t"] * 10 + ["a"] * 25 + ["b"] * 5, dtype=object)
        w = adv.balanced_weights(target, match)
        self.assertAlmostEqual(w[target].sum(), w[~target].sum())
        self.assertAlmostEqual(w[match == "a"].sum(), w[match == "b"].sum())


class ContextFeatures(unittest.TestCase):
    def test_shape_and_unseen_ball(self):
        match = synthetic_match()
        match.ball_x[100:200] = np.nan
        X = adv.context_features(match)
        self.assertEqual(X.shape, (match.n_frames, len(adv.CONTEXT_NAMES)))
        for name in ("ball_beyond", "ball_still", "ball_nearest_m"):
            self.assertTrue(np.isnan(X[100:200, adv.CONTEXT_NAMES.index(name)]).all())
        run_s = X[:, adv.CONTEXT_NAMES.index("run_s")]
        self.assertAlmostEqual(run_s[150], 100 / match.fps)

    def test_feature_groups_name_real_columns(self):
        from solution import deadmodel
        names = set(deadmodel.FEATURE_NAMES) | set(adv.CONTEXT_NAMES)
        for members in adv.GROUPS.values():
            self.assertTrue(set(members) <= names)
        self.assertEqual(set(sum(adv.GROUPS.values(), [])), names)


class Separates(unittest.TestCase):
    def test_auc_near_half_for_same_and_high_for_shifted(self):
        rng = np.random.RandomState(0)
        n = 4000
        X = rng.normal(size=(n, 3))
        y = np.arange(n) % 2 == 0
        groups = np.array(["g{}".format(i // 40) for i in range(n)], dtype=object)
        w = np.ones(n)
        same = adv.wauc(y, adv.cross_fit(X, y, w, groups)[0], w)
        self.assertLess(abs(same - 0.5), 0.05)
        X[y, 0] += 2.0
        shifted = adv.wauc(y, adv.cross_fit(X, y, w, groups)[0], w)
        self.assertGreater(shifted, 0.85)


if __name__ == "__main__":
    unittest.main()
