"""Tests for C3, the per-frame team model (solution/framemodel.py), and its decoder term.

Synthetic inputs only, no data needed:
    .venv/bin/python -m unittest tests.test_framemodel -v        # or: make test
"""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import decoder, framemodel  # noqa: E402
from solution.match import Match, pm  # noqa: E402
from tests.test_decoder import synthetic_evidence, synthetic_priors  # noqa: E402

FPS = 25
LENGTH, WIDTH = 105.0, 68.0


def moving_match(n=1500, seed=0, labels=None):
    """12 players drifting at random; home (first 6) deeper on the left, so it attacks +x."""
    rng = np.random.default_rng(seed)
    team = np.array([0] * 6 + [1] * 6, dtype=np.int8)
    start_x = np.concatenate((rng.uniform(5, 50, 6), rng.uniform(55, 100, 6)))
    start_y = rng.uniform(5, 63, 12)
    x = start_x + np.cumsum(rng.normal(0, 0.15, (n, 12)), axis=0)
    y = start_y + np.cumsum(rng.normal(0, 0.15, (n, 12)), axis=0)
    x[rng.random((n, 12)) < 0.05] = np.nan            # dropped players
    ball = np.full(n, np.nan, dtype=np.float32)
    return Match("synthetic_c3", FPS, (LENGTH, WIDTH), x.astype(np.float32),
                 y.astype(np.float32), team, ball, ball.copy(), [(0, n)], labels)


def team_swapped(match):
    out = match.copy()
    out.team = (1 - match.team).astype(np.int8)
    if match.labels is not None:
        labels = match.labels.copy()
        labels[match.labels == pm.H] = pm.A
        labels[match.labels == pm.A] = pm.H
        out.labels = labels
    return out


class FeatureTest(unittest.TestCase):
    def test_swap_matches_exchanging_the_teams(self):
        match = moving_match()
        X, names, kinds = framemodel.features(match)
        Y, _, _ = framemodel.features(team_swapped(match))
        self.assertEqual(X.shape[1], len(names))
        np.testing.assert_allclose(framemodel.swap(X, kinds), Y, rtol=1e-4, atol=1e-3)

    def test_swap_is_an_involution(self):
        X, _, kinds = framemodel.features(moving_match(300))
        np.testing.assert_array_equal(framemodel.swap(framemodel.swap(X, kinds), kinds), X)

    def test_pairs_come_home_then_away(self):
        _, names, kinds = framemodel.features(moving_match(100))
        for i, kind in enumerate(kinds):
            if kind == "pair" and names[i].endswith("_home"):
                self.assertEqual(names[i + 1], names[i][:-5] + "_away")
                self.assertEqual(kinds[i + 1], "pair")

    def test_window_mean_past_and_future(self):
        v = np.arange(10, dtype=float)
        past = framemodel._window_mean(v, -3, 0)
        future = framemodel._window_mean(v, 0, 3)
        self.assertTrue(np.isnan(past[:3]).all())
        self.assertEqual(past[5], 3.0)                 # mean of 2, 3, 4
        self.assertEqual(future[5], 6.0)               # mean of 5, 6, 7
        self.assertTrue(np.isnan(future[8:]).all())


class ModelTest(unittest.TestCase):
    PARAMS = dict(framemodel.PARAMS, max_iter=20, min_samples_leaf=20, step=2)

    def labelled(self, seed):
        labels = np.full(1500, pm.H, dtype=np.int8)
        labels[500:1000] = pm.A
        labels[1200:1300] = pm.D
        return moving_match(seed=seed, labels=labels)

    def test_fit_is_deterministic_and_symmetric(self):
        train = [self.labelled(1), self.labelled(2)]
        one = framemodel.fit(train, self.PARAMS)
        two = framemodel.fit(train, self.PARAMS)
        test = self.labelled(3)
        p1 = framemodel.home_probability(one, test, self.PARAMS)
        p2 = framemodel.home_probability(two, test, self.PARAMS)
        self.assertEqual(p1.tobytes(), p2.tobytes())
        # Exchanging the teams of the match gives exactly the complementary probability.
        swapped = framemodel.home_probability(one, team_swapped(test), self.PARAMS)
        np.testing.assert_allclose(p1 + swapped, 1.0, atol=1e-3)
        self.assertTrue((p1 >= self.PARAMS["clip"]).all())
        self.assertTrue((p1 <= 1 - self.PARAMS["clip"]).all())

    def test_probability_is_nan_outside_the_halves(self):
        model = framemodel.fit([self.labelled(1)], self.PARAMS)
        test = self.labelled(3)
        test.halves = [(0, 1000)]
        p = framemodel.home_probability(model, test, self.PARAMS)
        self.assertTrue(np.isfinite(p[:1000]).all())
        self.assertTrue(np.isnan(p[1000:]).all())


class DecoderTermTest(unittest.TestCase):
    def setUp(self):
        truth = np.repeat(np.array([decoder.H, decoder.A, decoder.D, decoder.H]), 250)
        self.nearest, self.seen, self.speed = synthetic_evidence(truth, np.random.RandomState(0))
        self.priors = synthetic_priors()

    def emissions(self, frame, **params):
        return decoder.block_emissions(self.nearest, self.seen, self.speed, self.priors,
                                       dict(decoder.DEFAULTS, **params), frame=frame)

    def test_weight_zero_changes_nothing(self):
        frame = np.full(len(self.nearest), 3.0)
        np.testing.assert_array_equal(self.emissions(frame, frame_w=0.0),
                                      self.emissions(None))

    def test_frame_term_favours_home_and_has_no_say_on_dead_ball(self):
        n = len(self.nearest)
        frame = np.full(n, np.log(0.9 / 0.1))
        base = self.emissions(None)
        out = self.emissions(frame, frame_w=1.0)
        delta = out - base
        np.testing.assert_allclose(delta[:, decoder.H], np.log(0.9), atol=1e-9)
        np.testing.assert_allclose(delta[:, decoder.A], np.log(0.1), atol=1e-9)
        np.testing.assert_allclose(delta[:, decoder.D], delta[:, decoder.H], atol=1e-9)

    def test_gaps_reads_only_frames_without_the_ball(self):
        n = len(self.nearest)
        frame = np.full(n, 2.0)
        frame[:decoder.BLOCK] = np.nan             # unknown on the first block
        out = self.emissions(frame, frame_w=0.5, frame_where="gaps")
        base = self.emissions(None)
        index = np.arange(n) // decoder.BLOCK
        read = np.bincount(index, weights=(~self.seen & np.isfinite(frame)).astype(float))
        changed = np.any(out != base, axis=1)
        np.testing.assert_array_equal(changed, read > 0)
        seen_out = self.emissions(frame, frame_w=0.5, frame_where="seen")
        read_seen = np.bincount(index, weights=(self.seen & np.isfinite(frame)).astype(float))
        np.testing.assert_array_equal(np.any(seen_out != base, axis=1), read_seen > 0)
        with self.assertRaises(ValueError):
            self.emissions(frame, frame_w=0.5, frame_where="nowhere")


if __name__ == "__main__":
    unittest.main()
