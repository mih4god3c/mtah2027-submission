"""Tests for the hand-over moment features (solution/handover.py) and the Step A / Step B
diagnostics of tools/gap_check.py (--events, --seen).

Synthetic inputs only, no data needed:
    .venv/bin/python -m unittest tests.test_handover -v        # or: make test
"""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import handover, touch  # noqa: E402
from solution.match import pm  # noqa: E402
from tests.test_gapteam import FPS, LENGTH, formation, gap_match, match_with  # noqa: E402
from tools import gap_check  # noqa: E402

H, A, D = pm.H, pm.A, pm.D


def running_match(n, turn, speed=3.0):
    """Every player runs towards the away goal (+x, home attacks +x) at `speed`, until frame
    `turn`, then everyone runs back. Home starts deeper, so home attacks +x."""
    x, y = formation(n, 30.0, 70.0, 40.0, 40.0)
    t = np.arange(n, dtype=float)
    path = np.where(t < turn, t, 2 * turn - t) * speed / FPS
    x = x + path[:, None]
    return match_with(x, y)


class PrimitiveTest(unittest.TestCase):
    def test_crossings_skip_nan_and_give_the_direction(self):
        value = np.array([-1, 1, np.nan, -2, 0, 0, 3, np.nan, np.nan, -1], dtype=float)
        frames, teams = handover.crossings(value, 0, len(value))
        np.testing.assert_array_equal(frames, [1, 3, 6, 9])
        np.testing.assert_array_equal(teams, [H, A, H, A])
        frames, _ = handover.crossings(value, 2, 7)   # inside a half: 3 has no earlier value
        np.testing.assert_array_equal(frames, [6])

    def test_peaks_keep_the_largest_above_threshold_ties_earliest(self):
        score = np.array([0, 5, 4, 0, 0, 0, 2, 0, 0, 3, 3, 0], dtype=float)
        np.testing.assert_array_equal(handover.peaks(score, 0, 12, 2.5, 2), [1, 9])
        np.testing.assert_array_equal(handover.peaks(score, 0, 12, 1.0, 2), [1, 6, 9])
        np.testing.assert_array_equal(handover.peaks(score, 0, 12, 1.0, 3), [1, 9])

    def test_past_minus_future_peaks_at_a_step(self):
        values = np.where(np.arange(40) >= 20, 5.0, 1.0)
        out = handover.past_minus_future(values, 4)
        self.assertEqual(int(np.nanargmax(out)), 20)
        self.assertAlmostEqual(out[20], 4.0)
        self.assertTrue(np.isnan(out[:4]).all() and np.isnan(out[-3:]).all())
        np.testing.assert_allclose(out[8:16], 0.0)


class FeatureTest(unittest.TestCase):
    def test_reversal_flips_when_both_teams_turn_and_is_nan_when_they_disagree(self):
        match = running_match(200, 100)
        value = handover.reversal(match)
        self.assertTrue((value[20:90] > 0).all())        # both run towards away's goal
        self.assertTrue((value[110:180] < 0).all())
        x = match.player_x.copy()
        x[:, 6:] = x[0, 6:]                               # away stands still: no agreement
        self.assertTrue(np.isnan(handover.reversal(match_with(x, match.player_y))).all())

    def test_dadvance_fires_once_at_the_turn_towards_away(self):
        match = running_match(300, 150)
        frames, teams = handover.events(match, "dadvance0.5")
        self.assertTrue(len(frames) >= 1)
        best = np.argmin(np.abs(frames - 150))
        self.assertLessEqual(abs(int(frames[best]) - 150), 3)
        self.assertEqual(teams[best], A)                 # everyone now runs to home's goal

    def test_leave_is_positive_when_home_moves_away_from_the_last_seen_ball(self):
        n = 200
        x, y = formation(n, 50.0, 55.0, 10.0, 10.0)
        t = np.arange(n, dtype=float) / FPS
        bx = np.full(n, np.nan)
        by = np.full(n, np.nan)
        bx[:50], by[:50] = 52.0, 34.0                    # seen until frame 50, then a gap
        x[:, 1:6] = 50.0 - 2.0 * t[:, None]              # home outfield runs away from it
        x[:, 7:12] = 60.0 - 2.0 * t[:, None]             # away closes in from the far side
        match = match_with(x, y, bx, by)
        value = handover.leave(match)
        self.assertTrue((value[40:110] > 0).all())
        # NaN beyond hold_s, plus the half-width of the 0.5 s centred mean
        self.assertTrue(np.isnan(value[49 + 3 * FPS + 8:]).all())

    def test_every_registered_feature_runs_and_is_deterministic(self):
        match = gap_match(400, (150, 300))
        for name in handover.FEATURES:
            a, b = handover.events(match, name), handover.events(match, name)
            np.testing.assert_array_equal(a[0], b[0])
            np.testing.assert_array_equal(a[1], b[1])
            self.assertEqual(len(handover.feature(match, name)), 400)


class EventScoreTest(unittest.TestCase):
    def test_hit_lag_and_false_alarms(self):
        n = 1500
        truth = np.full(n, H, dtype=np.int8)
        truth[600:] = A                                  # one hand-over, at 600, in the gap
        match = gap_match(n, (200, 1400), labels=truth)
        frames = np.array([610, 1100, 1200], dtype=np.int64)
        teams = np.array([A, H, A], dtype=np.int8)
        out = gap_check.event_scores(truth, match, frames, teams)
        self.assertEqual(out["lags"], [10])
        self.assertTrue(out["signed"])
        self.assertEqual(out["fa_events"], 2)            # 1100 and 1200: far from 600
        self.assertEqual(out["alive_events"], 3)
        self.assertEqual(out["alive_right"], 1)
        wrong = gap_check.event_scores(truth, match, frames, np.array([H, H, A], np.int8))
        self.assertEqual(wrong["lags"], [None])          # nothing towards A within 10 s
        unsigned = gap_check.event_scores(truth, match, frames, np.full(3, -1, np.int8))
        self.assertEqual(unsigned["lags"], [10])


class ShortOracleTest(unittest.TestCase):
    def setUp(self):
        n = 1000
        self.truth = np.full(n, H, dtype=np.int8)
        for a in (200, 500):                             # two 1 s possessions of A, ball seen
            self.truth[a:a + 25] = A
        self.match = gap_match(n, (900, 950), labels=self.truth)
        self.pred = np.full(n, H, dtype=np.int8)         # the decoder misses both

    def test_team_reach_and_random(self):
        seen = {200: True, 500: False}
        records, out = gap_check.short_oracles(
            self.truth, self.pred, self.match, False, lambda a, b, k: {"seen": seen[a]})
        self.assertEqual(len(records), 2)
        self.assertEqual([r["start"] for r in records], ["hand-over", "hand-over"])
        np.testing.assert_array_equal(out[("team", "<2 s")], self.truth)
        reach = out[("reach", "<2 s")]
        self.assertTrue((reach[200:225] == A).all() and (reach[500:525] == H).all())
        self.assertEqual(int(np.sum(out[("random", "<2 s")] == A)), 25)   # one of the two
        np.testing.assert_array_equal(out[("snap", "<2 s")], self.pred)   # nothing to move

    def test_snap_moves_a_late_decoded_possession_onto_the_true_one(self):
        pred = self.pred.copy()
        pred[210:235] = A                                # found 0.4 s late
        _, out = gap_check.short_oracles(self.truth, pred, self.match, False,
                                         lambda a, b, k: {"seen": False})
        snapped = out[("snap", "<2 s")]
        self.assertTrue((snapped[200:225] == A).all() and (snapped[225:240] == H).all())

    def test_engine_claims_are_the_holder_changes_to_a_team(self):
        holder = np.array([-1, -1, 0, 0, -1, 1, 1, 0], dtype=np.int8)
        touches = touch.Touches(np.zeros(0, np.int64), np.zeros(0, np.int8),
                                np.zeros((0, 3), np.int64), holder, np.ones(8, bool))
        frames, teams = gap_check.engine_claims(touches)
        np.testing.assert_array_equal(frames, [2, 5, 7])
        np.testing.assert_array_equal(teams, [0, 1, 0])


class SwitchBonusTest(unittest.TestCase):
    """The decoder's H <-> A switch bonus (decoder.switch_bonus, bonus_w)."""

    def params(self, **kw):
        from solution import decoder
        return dict(decoder.DEFAULTS, **kw)

    def test_bonus_covers_the_spread_around_the_shifted_event(self):
        from solution import decoder
        p = self.params(bonus_w=3.0, bonus_shift_s=-0.4, bonus_spread_s=0.2)
        out = decoder.switch_bonus(np.array([60, 10]), np.array([A, D]), 100, p, FPS)
        self.assertEqual(out.shape, (20, 2))
        # frame 60 - 10 = 50 is block 10; spread 0.2 s is 1 block either side
        np.testing.assert_array_equal(np.flatnonzero(out[:, 1]), [9, 10, 11])
        self.assertTrue((out[:, 0] == 0).all())          # D events are ignored
        self.assertEqual(out[10, 1], 3.0)

    def test_zero_weight_is_byte_identical_and_a_large_bonus_forces_a_switch(self):
        from solution import decoder
        from tests.test_decoder import synthetic_evidence, synthetic_priors
        rng = np.random.RandomState(0)
        truth = np.zeros(1000, dtype=np.int8)
        nearest, seen, speed = synthetic_evidence(truth, rng, flip=0.1, gaps=1)
        priors = synthetic_priors()
        plain = decoder.decode_half(nearest, seen, speed, priors, self.params(), FPS)
        events = (np.array([500, 525]), np.array([A, H]))
        off = decoder.decode_half(nearest, seen, speed, priors, self.params(), FPS,
                                  events=events)
        np.testing.assert_array_equal(plain, off)
        on = decoder.decode_half(nearest, seen, speed, priors,
                                 self.params(bonus_w=50.0, bonus_spread_s=0.0), FPS,
                                 events=events)
        self.assertTrue((on[500:525] == A).all())
        self.assertTrue((on[:500] == H).all() and (on[525:] == H).all())


if __name__ == "__main__":
    unittest.main()
