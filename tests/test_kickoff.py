"""Tests for the kickoff grammar (solution/kickoff.py) and the pre-kickoff stress.

Synthetic inputs only, no data needed:
    .venv/bin/python -m unittest tests.test_kickoff -v       # or: make test
"""

import os
import sys
import unittest
import warnings
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import decoder, kickoff, methods  # noqa: E402
from solution.match import Match, pm, validate  # noqa: E402
from solution.simulate import add_ball_noise, add_prekickoff, mean_player_speed  # noqa: E402
from tests.test_decoder import synthetic_priors  # noqa: E402

FPS = 25
LENGTH, WIDTH = 105.0, 68.0
CX, CY = LENGTH / 2, WIDTH / 2


def formation(rng):
    """(x, y) of 22 players in their own halves: slots 0-10 home on the left, 11-21 away."""
    x = np.concatenate([rng.uniform(15, CX - 0.5, 11), rng.uniform(CX + 0.5, 90, 11)])
    y = rng.uniform(5, WIDTH - 5, 22)
    return x, y


def standing(n, x, y, rng):
    return (np.tile(x, (n, 1)) + rng.normal(0, 0.03, (n, 22)),
            np.tile(y, (n, 1)) + rng.normal(0, 0.03, (n, 22)))


def open_play(n, x, y, rng):
    """Players running at about 3 m/s on smooth random headings, the ball moving among them."""
    heading = np.cumsum(rng.normal(0, 0.05, (n, 22)), axis=0) + rng.uniform(0, 6.3, 22)
    px = np.clip(x + np.cumsum(3.0 / FPS * np.cos(heading), axis=0), 1, LENGTH - 1)
    py = np.clip(y + np.cumsum(3.0 / FPS * np.sin(heading), axis=0), 1, WIDTH - 1)
    carrier = np.repeat(rng.randint(0, 22, n // 50 + 1), 50)[:n]
    return px, py, px[np.arange(n), carrier] + 0.5, py[np.arange(n), carrier]


def kick_away(n, start_x, start_y, rng):
    """A kickoff from the mark: the ball rolls back at 12 m/s, play speeds up, and the home
    team runs forward at 4 m/s for 3 s, so those near the line cross it, as in a real kickoff."""
    px, py, bx, by = open_play(n, start_x, start_y, rng)
    px[:, :11] += 4.0 / FPS * np.minimum(np.arange(1, n + 1), 3 * FPS)[:, None]
    roll = min(n, 2 * FPS)
    f = np.arange(1, roll + 1) / float(roll)
    bx[:roll], by[:roll] = CX - 24.0 * f, CY - 4.0 * f
    return px, py, bx, by


def build(pieces, name="synthetic_kickoff", labels=None):
    px = np.vstack([p[0] for p in pieces]).astype(np.float32)
    py = np.vstack([p[1] for p in pieces]).astype(np.float32)
    bx = np.concatenate([p[2] for p in pieces]).astype(np.float32)
    by = np.concatenate([p[3] for p in pieces]).astype(np.float32)
    team = np.array([0] * 11 + [1] * 11, dtype=np.int8)
    return validate(Match(name, FPS, (LENGTH, WIDTH), px, py, team, bx, by, [(0, len(bx))],
                          labels))


def pre_kickoff_half(pre_s=60, post_s=60, seed=0):
    """(match, kick frame): players stand in formation with the ball on the mark, then kickoff."""
    rng = np.random.RandomState(seed)
    x, y = formation(rng)
    n_pre, n_post = pre_s * FPS, post_s * FPS
    sx, sy = standing(n_pre, x, y, rng)
    rest = (sx, sy, np.full(n_pre, CX + 0.2), np.full(n_pre, CY - 0.1))
    labels = np.concatenate([np.full(n_pre, pm.D), np.full(n_post, pm.H)]).astype(np.int8)
    return build([rest, kick_away(n_post, x, y, rng)], labels=labels), n_pre


class Kickoff(unittest.TestCase):

    def test_kickoff_found_where_it_is(self):
        match, kick = pre_kickoff_half()
        found = kickoff.detect(match)
        self.assertEqual(len(found), 1)
        self.assertIsNotNone(found[0])
        # Early is the safe side: back_s (2 s) before where play speeds up.
        self.assertTrue(kick - 3 * FPS <= found[0] <= kick + 3)

    def test_the_ball_is_not_read(self):
        match, kick = pre_kickoff_half(seed=1)
        noisy = add_ball_noise(match, 0.05, 3.4, seed=0, smooth_s=0.26)
        blind = match.copy()
        blind.ball_x[:] = np.nan
        blind.ball_y[:] = np.nan
        self.assertEqual(kickoff.detect(noisy), kickoff.detect(match))
        self.assertEqual(kickoff.detect(blind), kickoff.detect(match))

    def test_no_kickoff_mid_half(self):
        """Open play, a goal, a re-set with the ball on the mark, a kickoff: not the half's."""
        rng = np.random.RandomState(3)
        x, y = formation(rng)
        play = open_play(100 * FPS, x, y, rng)
        sx, sy = standing(30 * FPS, x, y, rng)
        reset = (sx, sy, np.full(30 * FPS, CX), np.full(30 * FPS, CY))
        match = build([play, reset, kick_away(60 * FPS, x, y, rng)])
        self.assertEqual(kickoff.detect(match), [None])
        # The guard on play before the kick is what holds it back: without it, it would fire.
        unguarded = dict(kickoff.PARAMS, play_share=1.0)
        found = kickoff.detect(match, unguarded)[0]
        self.assertTrue(130 * FPS - 3 * FPS <= found <= 130 * FPS + 3)

    def test_half_that_starts_at_the_kick_has_none(self):
        """IDSSE halves start at the kick: no formation hold before it, nothing to find."""
        rng = np.random.RandomState(4)
        x, y = formation(rng)
        match = build([kick_away(90 * FPS, x, y, rng)])
        self.assertEqual(kickoff.detect(match), [None])

    def test_no_speed_up_no_kickoff(self):
        """The ball rolled off the mark while everyone stands in formation: not a kickoff."""
        rng = np.random.RandomState(5)
        x, y = formation(rng)
        sx, sy = standing(90 * FPS, x, y, rng)
        bx, by = np.full(90 * FPS, CX), np.full(90 * FPS, CY)
        bx[60 * FPS:] = CX - np.minimum(np.arange(30 * FPS) * 0.4, 20.0)
        self.assertEqual(kickoff.detect(build([(sx, sy, bx, by)])), [None])

    def test_deterministic(self):
        match, _ = pre_kickoff_half(seed=6)
        self.assertEqual(kickoff.detect(match), kickoff.detect(match.copy()))
        self.assertEqual(kickoff.explain(match), kickoff.explain(match.copy()))

    def test_nan_and_ghost_players(self):
        match, kick = pre_kickoff_half(seed=7)
        rng = np.random.RandomState(0)
        holes = rng.rand(*match.player_x.shape) < 0.2
        match.player_x[holes] = np.nan
        match.player_y[holes] = np.nan
        # Ghosts: stale copies of three home and two away players, left where they stood.
        source = [0, 3, 5, 12, 15]
        ghosts_x = np.tile(match.player_x[kick - 200, source] + 0.3, (match.n_frames, 1))
        ghosts_y = np.tile(match.player_y[kick - 200, source], (match.n_frames, 1))
        match.player_x = np.hstack((match.player_x, ghosts_x))
        match.player_y = np.hstack((match.player_y, ghosts_y))
        match.team = np.concatenate((match.team, match.team[source]))
        found = kickoff.detect(validate(match))[0]
        self.assertIsNotNone(found)
        self.assertTrue(kick - 3 * FPS <= found <= kick + 3)

    def test_no_players_tracked_finds_nothing_quietly(self):
        match, _ = pre_kickoff_half(seed=8)
        nobody = match.copy()
        nobody.player_x[:] = np.nan
        nobody.player_y[:] = np.nan
        few = match.copy()
        few.player_x[:, 5:11] = np.nan                  # five home players: below min_players
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertEqual(kickoff.detect(nobody), [None])
            self.assertEqual(kickoff.detect(few), [None])

    def test_decoder_is_dead_ball_until_the_kickoff(self):
        match, kick = pre_kickoff_half(seed=9)
        # A team keeps near the ball through the pre-kickoff: the trap the grammar is for.
        match.player_x[:kick, 0] = match.ball_x[:kick] - 0.5
        match.player_y[:kick, 0] = match.ball_y[:kick]
        with mock.patch.object(decoder, "idsse_priors", lambda *args: synthetic_priors()):
            codes = methods.semi_markov(kickoff_params={}, speed_scale="ms")(match)
            plain = methods.semi_markov(speed_scale="ms")(match)
        found = kickoff.detect(match)[0]
        # D up to the decoding block that holds the kick (on the half's 5-frame grid).
        self.assertTrue(np.all(codes[:found // decoder.BLOCK * decoder.BLOCK] == pm.D))
        self.assertEqual(codes.shape, plain.shape)
        with self.assertRaises(ValueError):
            methods.semi_markov(kickoff_params={"spot": 2.0})


class PreKickoffStress(unittest.TestCase):

    def base(self):
        match, kick = pre_kickoff_half(pre_s=0, post_s=120, seed=10)
        # Two halves with X between, as in the IDSSE timeline; the stretch goes before the second.
        n = match.n_frames
        two = Match("synthetic_two", FPS, (LENGTH, WIDTH),
                    np.vstack([match.player_x, np.full((50, 22), np.nan, np.float32),
                               match.player_x]),
                    np.vstack([match.player_y, np.full((50, 22), np.nan, np.float32),
                               match.player_y]),
                    match.team,
                    np.concatenate([match.ball_x, np.full(50, np.nan, np.float32), match.ball_x]),
                    np.concatenate([match.ball_y, np.full(50, np.nan, np.float32), match.ball_y]),
                    [(0, n), (n + 50, 2 * n + 50)],
                    np.concatenate([match.labels, np.full(50, pm.X, np.int8), match.labels]))
        return validate(two)

    def test_shape_labels_and_halves(self):
        match = self.base()
        out = validate(add_prekickoff(match, 60, seed=0))
        n = 60 * FPS
        info = out.notes["prekickoff"]
        a = match.halves[1][0]
        self.assertEqual(out.n_frames, match.n_frames + n)
        self.assertEqual(out.halves, [match.halves[0], (a, match.halves[1][1] + n)])
        self.assertEqual((info["half"], info["frames"]), (1, n))
        self.assertTrue(a < info["kick"] < a + n)
        self.assertTrue(np.all(out.labels[a:info["kick"]] == pm.D))
        self.assertTrue(np.all(out.labels[info["kick"]:a + n] == match.labels[a]))
        np.testing.assert_array_equal(out.labels[a + n:], match.labels[a:])
        np.testing.assert_array_equal(out.ball_x[:a], match.ball_x[:a])

    def test_deterministic_and_labels_do_not_depend_on_the_seed(self):
        match = self.base()
        one, again = add_prekickoff(match, 60, seed=1), add_prekickoff(match, 60, seed=1)
        other = add_prekickoff(match, 60, seed=2)
        np.testing.assert_array_equal(one.ball_x, again.ball_x)
        np.testing.assert_array_equal(one.player_x, again.player_x)
        np.testing.assert_array_equal(one.labels, other.labels)
        self.assertFalse(np.array_equal(one.ball_x, other.ball_x, equal_nan=True))

    def test_detector_finds_the_synthetic_kick(self):
        out = add_prekickoff(self.base(), 90, seed=3)
        found = kickoff.detect(out)
        self.assertIsNone(found[0])                      # the first half starts at its kick
        kick = out.notes["prekickoff"]["kick"]
        self.assertTrue(kick - 3 * FPS <= found[1] <= kick + 3)

    def test_busy_profile_mills_about_and_is_still_found(self):
        out = add_prekickoff(self.base(), 90, seed=4, profile="busy")
        info, a = out.notes["prekickoff"], out.halves[1][0]
        speed = mean_player_speed(out)
        self.assertGreater(np.nanmedian(speed[a:a + 60 * FPS]), 0.8)    # not standing still
        found = kickoff.detect(out)[1]
        self.assertIsNotNone(found)
        self.assertTrue(info["kick"] - 3 * FPS <= found <= info["kick"] + 3)

    def test_zero_seconds_is_a_copy(self):
        match = self.base()
        out = add_prekickoff(match, 0)
        np.testing.assert_array_equal(out.ball_x, match.ball_x)
        self.assertNotIn("prekickoff", out.notes)


if __name__ == "__main__":
    unittest.main()
