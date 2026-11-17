"""Tests for the touch engine (solution/touch.py), the ball-noise stress and the boundary report.

Synthetic inputs only, no data needed:
    .venv/bin/python -m unittest tests.test_touch -v       # or: make test
"""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import decoder, touch  # noqa: E402
from solution.match import Match, pm  # noqa: E402
from solution.simulate import add_ball_noise  # noqa: E402
from tools import boundaries  # noqa: E402

FPS = 25
KICK, ARRIVE, N = 100, 124, 250


def kick_match():
    """Home player 0 dribbles, kicks at KICK; away player 2 receives at ARRIVE and dribbles on.

    Slots: 0 home dribbler, 1 home far away, 2 away receiver, 3 away far away.
    """
    t = np.arange(N, dtype=np.float64)
    px = np.zeros((N, 4))
    py = np.full((N, 4), 34.0)
    px[:, 0] = 30.0 + 2.5 * np.minimum(t, KICK) / FPS          # stops after the kick
    px[:, 1], py[:, 1] = 20.0, 10.0
    px[:, 2] = 55.0                                            # waits, then runs along y
    py[:, 2] = 34.0 + 2.0 * np.maximum(t - ARRIVE, 0) / FPS
    px[:, 3], py[:, 3] = 80.0, 60.0
    bx, by = px[:, 0] + 0.5, py[:, 0].copy()                    # at the dribbler's feet
    flight = (t > KICK) & (t < ARRIVE)
    start_x = px[KICK, 0] + 0.5
    bx[flight] = start_x + (55.0 - 0.5 - start_x) * (t[flight] - KICK) / (ARRIVE - KICK)
    after = t >= ARRIVE
    bx[after], by[after] = px[after, 2] - 0.5, py[after, 2]
    return Match("synthetic_kick", FPS, (105.0, 68.0), px.astype(np.float32),
                 py.astype(np.float32), np.array([0, 0, 1, 1], dtype=np.int8),
                 bx.astype(np.float32), by.astype(np.float32), [(0, N)])


class Touches(unittest.TestCase):

    def test_touches_found_at_the_kick_and_the_reception(self):
        found = touch.detect(kick_match())
        near_kick = np.abs(found.touch_frame - KICK) <= 3
        near_arrival = np.abs(found.touch_frame - ARRIVE) <= 3
        self.assertTrue(np.any(near_kick & (found.touch_team == 0)))
        self.assertTrue(np.any(near_arrival & (found.touch_team == 1)))
        # Nothing during the steady dribble or the clean flight.
        steady = ((found.touch_frame > 10) & (found.touch_frame < KICK - 5)) | \
                 ((found.touch_frame > KICK + 5) & (found.touch_frame < ARRIVE - 5))
        self.assertFalse(np.any(steady))

    def test_control_and_holder_follow_the_ball(self):
        found = touch.detect(kick_match())
        teams = set(found.control[:, 2].tolist())
        self.assertEqual(teams, {0, 1})
        self.assertTrue(np.all(found.holder[10:KICK - 5] == 0))
        self.assertTrue(np.all(found.holder[ARRIVE + 5:N - 10] == 1))

    def test_deterministic(self):
        first, second = touch.detect(kick_match()), touch.detect(kick_match())
        for name in ("touch_frame", "touch_team", "control", "holder", "usable"):
            np.testing.assert_array_equal(getattr(first, name), getattr(second, name))

    def test_nan_players_do_not_crash_or_create_touches(self):
        match = kick_match()
        match.player_x[40:60] = np.nan
        match.player_y[40:60] = np.nan
        found = touch.detect(match)
        self.assertFalse(np.any((found.touch_frame >= 40) & (found.touch_frame < 60)))
        # The hold carries through the blind stretch, as through flight.
        self.assertTrue(np.all(found.holder[40:60] == 0))
        everyone_gone = kick_match()
        everyone_gone.player_x[:] = np.nan
        found = touch.detect(everyone_gone)
        self.assertEqual(len(found.touch_frame), 0)
        self.assertTrue(np.all(found.holder == touch.NONE))

    def test_a_still_ghost_of_the_other_team_does_not_take_the_ball(self):
        match = kick_match()
        ghost_x = np.full((N, 1), np.nan, dtype=np.float32)
        ghost_y = np.full((N, 1), np.nan, dtype=np.float32)
        ghost_x[30:70], ghost_y[30:70] = match.ball_x[50], match.ball_y[50] + 0.2
        match.player_x = np.hstack((match.player_x, ghost_x))
        match.player_y = np.hstack((match.player_y, ghost_y))
        match.team = np.append(match.team, np.int8(1))
        found = touch.detect(match)
        self.assertTrue(np.all(found.holder[10:KICK - 5] == 0))

    def test_teleport_hops_are_cut_out(self):
        match = kick_match()
        match.ball_x[60:63] += 3.0                                 # a 3-frame hop of 3 m
        found = touch.detect(match)
        self.assertFalse(np.any(found.usable[60:63]))
        self.assertFalse(np.any((found.touch_frame >= 55) & (found.touch_frame < 68)))
        self.assertTrue(np.all(found.holder[10:KICK - 5] == 0))

    def test_ball_beyond_a_line_is_never_used(self):
        match = kick_match()
        match.ball_y[:] = -1.0                                     # just over the touchline
        match.player_y[:] = -0.5
        found = touch.detect(match)
        self.assertFalse(found.usable.any())
        self.assertEqual(len(found.touch_frame), 0)

    def test_hysteresis_waits_for_control(self):
        control = np.array([[0, 20, 0]])
        frames, teams = np.array([50]), np.array([1], dtype=np.int8)
        usable = np.ones(200, dtype=bool)
        takes = touch.sweep_holder(200, frames, teams, control, usable, [(0, 200)], FPS,
                                   dict(touch.PARAMS, touch_takes=True))
        waits = touch.sweep_holder(200, frames, teams, control, usable, [(0, 200)], FPS,
                                   dict(touch.PARAMS, touch_takes=False))
        self.assertTrue(np.all(takes[:50] == 0) and np.all(takes[50:150] == 1))
        self.assertTrue(np.all(waits[:50] == 0) and np.all(waits[50:] == touch.NONE))
        # Holds expire expire_s after the team's last touch.
        expire = int(touch.PARAMS["expire_s"] * FPS)
        self.assertTrue(np.all(takes[50 + expire:] == touch.NONE))

    def test_long_ball_gaps_end_the_hold(self):
        usable = np.ones(400, dtype=bool)
        usable[60:200] = False
        holder = touch.sweep_holder(400, np.array([], dtype=np.int64), np.array([], np.int8),
                                    np.array([[0, 50, 0]]), usable, [(0, 400)], FPS)
        carry = int(touch.PARAMS["carry_gap_s"] * FPS)
        self.assertTrue(np.all(holder[:60 + carry] == 0))
        self.assertTrue(np.all(holder[60 + carry:] == touch.NONE))

    def test_keep_masks_drop_events_and_default_to_all(self):
        control = np.array([[0, 20, 0]])
        frames, teams = np.array([50]), np.array([1], dtype=np.int8)
        usable = np.ones(200, dtype=bool)
        args = (200, frames, teams, control, usable, [(0, 200)], FPS, touch.PARAMS)
        everything = touch.sweep_holder(*args)
        self.assertTrue(np.array_equal(
            everything, touch.sweep_holder(*args, np.array([True]), np.array([True]))))
        # Without the other team's touch, team 0 keeps the ball until its hold expires.
        kept = touch.sweep_holder(*args, np.array([False]), None)
        expire = int(touch.PARAMS["expire_s"] * FPS)
        self.assertTrue(np.all(kept[:20 + expire] == 0))
        self.assertFalse(np.any(kept == 1))
        # Without the control episode, the hold starts at the touch.
        alone = touch.sweep_holder(*args, None, np.array([False]))
        self.assertTrue(np.all(alone[:50] == touch.NONE) and np.all(alone[50:150] == 1))


class Snapping(unittest.TestCase):

    def test_hand_over_moves_onto_the_new_teams_touch_and_never_crosses(self):
        states = np.array([0] * 100 + [1] * 100 + [0] * 100, dtype=np.int8)
        events = touch.Touches(np.array([90, 185, 210]), np.array([1, 0, 0], dtype=np.int8),
                               np.zeros((0, 3), dtype=np.int64), np.full(300, -1, np.int8),
                               np.ones(300, dtype=bool))
        out = touch.snap_transitions(states, events, [(0, 300)], FPS)
        self.assertEqual(int(np.flatnonzero(np.diff(out))[0]) + 1, 90)     # H->A at 90
        self.assertEqual(int(np.flatnonzero(np.diff(out))[1]) + 1, 185)    # A->H at 185
        # No event in the window: unchanged. Restarts move only when asked.
        dead = np.array([2] * 100 + [1] * 100, dtype=np.int8)
        self.assertTrue(np.array_equal(
            touch.snap_transitions(dead, events, [(0, 200)], FPS), dead))
        self.assertEqual(int(np.flatnonzero(np.diff(touch.snap_transitions(
            dead, events, [(0, 200)], FPS, restarts=True)))[0]) + 1, 90)


class HolderTerm(unittest.TestCase):

    def test_off_by_default_and_symmetric_when_on(self):
        rng = np.random.RandomState(0)
        n = 200
        nearest = rng.randint(-1, 2, n).astype(np.int8)
        seen = nearest >= 0
        speed = rng.rand(n)
        priors = {"agree": 0.78, "holder_agree": 0.95,
                  "speed_log": np.log(np.full((2, len(decoder.RANK_EDGES)), 0.05)),
                  "speed_edges": decoder.RANK_EDGES}
        plain = decoder.block_emissions(nearest, seen, speed, priors, decoder.DEFAULTS)
        holder = np.where(np.arange(n) < 100, 0, 1).astype(np.int8)
        same = decoder.block_emissions(nearest, seen, speed, priors, decoder.DEFAULTS, holder)
        np.testing.assert_array_equal(plain, same)                 # touch_w 0: ignored
        on = decoder.block_emissions(nearest, seen, speed, priors,
                                     dict(decoder.DEFAULTS, touch_w=1.0), holder)
        self.assertTrue(np.all((on - plain)[:20, decoder.H] > (on - plain)[:20, decoder.A]))
        self.assertTrue(np.all((on - plain)[20:, decoder.A] > (on - plain)[20:, decoder.H]))


class BallNoise(unittest.TestCase):

    def test_keeps_the_missing_mask_and_is_deterministic(self):
        match = kick_match()
        match.ball_x[150:180] = np.nan
        match.ball_y[150:180] = np.nan
        a = add_ball_noise(match, 0.05, 30.0, seed=1)
        b = add_ball_noise(match, 0.05, 30.0, seed=1)
        np.testing.assert_array_equal(a.ball_x, b.ball_x)
        np.testing.assert_array_equal(a.ball_seen, match.ball_seen)
        self.assertFalse(np.array_equal(a.ball_x, match.ball_x))
        off = add_ball_noise(match, 0.0, 0.0)
        np.testing.assert_array_equal(off.ball_x, match.ball_x)

    def test_smoothing_keeps_gaps_and_spreads_the_kick(self):
        match = kick_match()
        match.ball_x[150:180] = np.nan
        match.ball_y[150:180] = np.nan
        smooth = add_ball_noise(match, 0.0, 0.0, smooth_s=0.26)
        np.testing.assert_array_equal(smooth.ball_seen, match.ball_seen)
        accel = np.abs(np.diff(match.ball_x[:150].astype(np.float64), 2))
        smooth_accel = np.abs(np.diff(smooth.ball_x[:150].astype(np.float64), 2))
        self.assertLess(smooth_accel.max(), 0.5 * accel.max())
        # The engine still finds the kick and the reception on the smoothed ball.
        found = touch.detect(smooth)
        self.assertTrue(np.any((np.abs(found.touch_frame - KICK) <= 8) & (found.touch_team == 0)))
        self.assertTrue(np.any((np.abs(found.touch_frame - ARRIVE) <= 8) &
                               (found.touch_team == 1)))

    def test_hops_are_teleports_of_the_bar_ath_size(self):
        n = 25 * 60 * 30                                           # 30 minutes, ball still
        match = Match("still", FPS, (105.0, 68.0), np.zeros((n, 1), np.float32),
                      np.zeros((n, 1), np.float32), np.array([0], np.int8),
                      np.full(n, 50.0, np.float32), np.full(n, 30.0, np.float32), [(0, n)])
        noisy = add_ball_noise(match, 0.0, 3.4, seed=0)
        step = np.hypot(np.diff(noisy.ball_x), np.diff(noisy.ball_y)) * FPS
        jumps = step[step > 1.0]
        self.assertTrue(np.all((jumps >= 40.0 - 1e-3) & (jumps <= 110.0 + 1e-3)))
        self.assertAlmostEqual(len(jumps) / 30.0, 3.4, delta=1.0)  # per minute, roughly


class BoundaryReport(unittest.TestCase):

    def test_oracle_snap_recovers_shifted_boundaries(self):
        truth = np.array([3] * 10 + [0] * 100 + [1] * 60 + [2] * 50 + [0] * 80 + [3] * 10,
                         dtype=np.int8)
        late = truth.copy()
        late[110:120], late[220:245] = 0, 2           # hand-over 10 late, restart 25 late
        snapped = boundaries.snap_to_truth(truth, late, boundaries.KINDS)
        np.testing.assert_array_equal(snapped, truth)
        only = boundaries.snap_to_truth(truth, late, ("hand-over",))
        np.testing.assert_array_equal(only[:220], truth[:220])
        self.assertTrue(np.all(only[220:245] == 2))

    def test_records_latency_kind_and_ball(self):
        truth = np.array([3] * 10 + [0] * 100 + [1] * 60 + [2] * 50 + [0] * 80 + [3] * 10,
                         dtype=np.int8)
        late = truth.copy()
        late[110:116] = 0
        seen = np.ones(len(truth), dtype=bool)
        seen[220] = False
        records, spurious = boundaries.transition_records(truth, late, seen, 25)
        self.assertEqual(records, [("hand-over", True, 6), ("to dead", True, 0),
                                   ("restart", False, 0)])
        self.assertEqual(spurious, [])


if __name__ == "__main__":
    unittest.main()
