"""Simulator v2 (solution/simulate_v2.py): players lost where BAR-ATH loses them (no data needed)."""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import simulate, simulate_v2  # noqa: E402
from solution.match import Match, validate  # noqa: E402

FPS = 25


def pitch_match(n=30000, seed=0):
    """Slots 0-1 stand 0.5 m apart (one behind the other from the camera); 2-9 are spread out."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n, 10), dtype=np.float32)
    y = np.zeros((n, 10), dtype=np.float32)
    drift = np.cumsum(rng.normal(0, 0.02, n))
    x[:, 0], y[:, 0] = 50.0 + drift, 30.0
    x[:, 1], y[:, 1] = 50.0 + drift, 30.5                  # behind slot 0, seen from y = 0
    for s in range(2, 10):
        x[:, s] = 15.0 + 10.0 * (s - 2) + np.cumsum(rng.normal(0, 0.02, n))
        y[:, s] = 15.0 + 5.0 * (s % 3)
    bx = (50.0 + np.zeros(n)).astype(np.float32)
    by = (34.0 + np.zeros(n)).astype(np.float32)
    team = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8)
    return Match("synthetic_v2", FPS, (105.0, 68.0), x, y, team, bx, by,
                 [(0, n // 2), (n // 2, n)])


class SimulatorV2(unittest.TestCase):
    def setUp(self):
        self.match = pitch_match()
        self.out = simulate.degrade(self.match, 0, "v2")

    def test_valid_deterministic_and_input_untouched(self):
        validate(self.out)
        again = simulate.degrade(self.match, 0, "v2")
        np.testing.assert_array_equal(self.out.player_x, again.player_x)
        self.assertFalse(np.isnan(self.match.player_x).any())
        self.assertEqual(self.out.notes["degraded"], {"version": "v2", "seed": 0})

    def test_same_ball_as_v1(self):
        v1 = simulate.degrade(self.match, 0, "v1")
        np.testing.assert_array_equal(self.out.ball_seen, v1.ball_seen)

    def test_occluded_player_is_lost_more(self):
        params = dict(simulate_v2.V2, episode_every=1e9)    # individual losses only
        x, _, _ = simulate_v2.degrade_players_v2(
            self.match, self.out.ball_seen, simulate._rng(self.match, 0, 2), params)
        lost = np.isnan(x[:, :10]).mean(axis=0)
        self.assertGreater(lost[1], 5 * lost[0])            # behind, against in front
        self.assertGreater(lost[1], 5 * lost[2:].mean())    # against isolated players

    def test_episodes_take_the_crowd(self):
        # The pair is the only crowd, so episodes centre on it and take both players.
        lost = np.isnan(self.out.player_x[:, :10]).mean(axis=0)
        self.assertGreater(lost[0], 3 * lost[2:].mean())
        self.assertGreater(lost[1], 3 * lost[2:].mean())

    def test_no_track_ends_soon_after_it_starts(self):
        min_on = simulate_v2.V2["min_on_frames"]
        on = np.isfinite(self.out.player_x[:, :10])
        for start, stop in self.match.halves:
            for s in range(10):
                window = np.concatenate(([False], on[start:stop, s], [False])).astype(np.int8)
                edges = np.diff(window)
                starts = np.flatnonzero(edges == 1)
                ends = np.flatnonzero(edges == -1)
                inner = (starts > 0) & (ends < stop - start)   # runs cut by a half edge aside
                self.assertTrue(np.all((ends - starts)[inner] > min_on))

    def test_never_frozen_exactly_before_a_drop(self):
        x, y = self.out.player_x[:, :10], self.out.player_y[:, :10]
        on = np.isfinite(x)
        t, s = np.nonzero(on[1:-1] & on[:-2] & ~on[2:])
        t = t + 1
        step = np.hypot(x[t, s] - x[t - 1, s], y[t, s] - y[t - 1, s])
        self.assertGreater(len(t), 0)
        self.assertTrue(np.all(step >= 0.001 - 1e-6))

    def test_players_false_keeps_clean_players(self):
        kept = simulate.degrade(self.match, 0, "v2", players=False)
        np.testing.assert_array_equal(kept.player_x, self.match.player_x)
        self.assertEqual(kept.notes["degraded"]["players"], "clean")


class LossRate(unittest.TestCase):
    def test_terms_add_up(self):
        match = pitch_match(n=200)
        coef = simulate_v2.V2["loss"]
        seen = np.ones(match.n_frames, dtype=bool)
        rate = simulate_v2.loss_log_rate(match, seen, coef)
        # slot 1 is 0.5 m behind slot 0: nearest 0-1 m; slot 0 has nobody in front
        self.assertAlmostEqual(rate[0, 1] - rate[0, 0], coef["nearest, any team = 0-1 m"], 5)
        unseen = simulate_v2.loss_log_rate(match, ~seen, coef)
        self.assertAlmostEqual(unseen[0, 0] - rate[0, 0], coef["ball = unseen"], 5)
        self.assertAlmostEqual(unseen[0, 1] - rate[0, 1], coef["ball = unseen"]
                               + coef["nearest < 2 m x ball unseen = yes"], 5)


class TrainVersion(unittest.TestCase):
    def test_default_env_and_refusal(self):
        saved = os.environ.pop(simulate.TRAIN_SIM_ENV, None)
        try:
            self.assertEqual(simulate.train_version(), simulate.DEFAULT_VERSION)
            os.environ[simulate.TRAIN_SIM_ENV] = "v2"
            self.assertEqual(simulate.train_version(), "v2")
            os.environ[simulate.TRAIN_SIM_ENV] = "v9"
            with self.assertRaises(ValueError):
                simulate.train_version()
        finally:
            os.environ.pop(simulate.TRAIN_SIM_ENV, None)
            if saved is not None:
                os.environ[simulate.TRAIN_SIM_ENV] = saved


if __name__ == "__main__":
    unittest.main()
