"""Tests for the C2b dead-ball cues (solution/deadball.py), the beyond-line thinning stress,
the IDSSE restart events and tools/deadball_check.py's oracle repairs.

Synthetic inputs only, no data needed:
    .venv/bin/python -m unittest tests.test_deadball -v       # or: make test
"""

import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import deadball, decoder  # noqa: E402
from solution.match import Match, pm  # noqa: E402
from solution.simulate import thin_beyond  # noqa: E402
from solution.sources import idsse  # noqa: E402
from tests.test_decoder import synthetic_evidence, synthetic_priors  # noqa: E402
from tools import deadball_check  # noqa: E402

FPS = 25
LENGTH, WIDTH = 105.0, 68.0


def match_with(player_x, player_y, ball_x, ball_y, labels=None):
    n = len(ball_x)
    team = np.array([0, 1], dtype=np.int8)[:player_x.shape[1]]
    return Match("synthetic_deadball", FPS, (LENGTH, WIDTH), player_x.astype(np.float32),
                 player_y.astype(np.float32), team, ball_x.astype(np.float32),
                 ball_y.astype(np.float32), [(0, n)], labels)


def two_players(n):
    """One player standing mid-pitch, one jogging along x at 3 m/s on the halfway line."""
    x = np.column_stack([np.full(n, 30.0), 20.0 + 3.0 * np.arange(n) / FPS])
    y = np.column_stack([np.full(n, 30.0), np.full(n, 34.0)])
    return x, y


class CueTest(unittest.TestCase):
    def test_slow_player_beyond_a_touchline(self):
        n = 200
        x, y = two_players(n)
        y[50:150, 0] = -1.0                     # the standing player steps behind the touchline
        match = match_with(x, y, np.full(n, 50.0), np.full(n, 30.0))
        cue = deadball.beyond_line(match)
        self.assertTrue(cue[70:130].all())        # clear of the 1 s speed window round the steps
        self.assertFalse(cue[:40].any())
        self.assertFalse(cue[160:].any())

    def test_fast_player_beyond_a_line_does_not_count(self):
        n = 200
        x, y = two_players(n)
        y[:, 1] = WIDTH + 1.0                  # the jogging player runs behind the far touchline
        match = match_with(x, y, np.full(n, 50.0), np.full(n, 30.0))
        self.assertFalse(deadball.beyond_line(match).any())

    def test_player_on_the_line_does_not_count(self):
        n = 100
        x, y = two_players(n)
        x[:, 0] = LENGTH + 0.3                 # within beyond_m of the goal line
        match = match_with(x, y, np.full(n, 50.0), np.full(n, 30.0))
        self.assertFalse(deadball.beyond_line(match).any())

    def test_ball_beyond_by_more_than_the_margin(self):
        n = 10
        x, y = two_players(n)
        bx = np.array([50, -0.5, -1.5, LENGTH + 2, 50, 50, 50, np.nan, 50, 50], dtype=float)
        by = np.array([30, 30, 30, 30, -2, WIDTH + 0.5, 30, np.nan, 30, 30], dtype=float)
        cue = deadball.ball_beyond(match_with(x, y, bx, by))
        np.testing.assert_array_equal(cue, [0, 0, 1, 1, 1, 0, 0, 0, 0, 0])

    def test_cues_stacks_in_the_order_asked(self):
        n = 100
        x, y = two_players(n)
        x[:, 0] = -1.0
        match = match_with(x, y, np.full(n, 50.0), np.full(n, 30.0))
        both = deadball.cues(match, dict(deadball.PARAMS, cues=("ball", "player")))
        self.assertEqual(both.shape, (n, 2))
        self.assertFalse(both[:, 0].any())
        self.assertTrue(both[20:80, 1].all())

    def test_fit_reads_the_cue_as_dead_ball_evidence(self):
        n = 400
        x, y = two_players(n)
        labels = np.full(n, pm.H, dtype=np.int8)
        labels[100:200] = pm.D
        x[100:200, 0] = -1.0                   # a taker stands behind the goal line in the stoppage
        match = match_with(x, y, np.full(n, 50.0), np.full(n, 30.0), labels)
        table = deadball.fit([match])
        self.assertEqual(table.shape, (1, 2, 2))
        np.testing.assert_allclose(np.exp(table).sum(axis=2), 1.0)
        self.assertGreater(table[0, 1, 1], table[0, 0, 1])     # P(cue | D) > P(cue | alive)


class DecoderTermTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(3)
        self.truth = np.repeat(rng.randint(0, 3, 40), 50).astype(np.int8)
        self.nearest, self.seen, self.speed = synthetic_evidence(self.truth, rng)
        self.priors = dict(synthetic_priors(), cue_log=np.log(np.array([[[0.99, 0.01],
                                                                          [0.8, 0.2]]])))
        self.cue = (self.truth == decoder.D)[:, None] & (rng.rand(len(self.truth), 1) < 0.3)

    def test_weight_zero_leaves_emissions_byte_identical(self):
        params = dict(decoder.DEFAULTS)
        base = decoder.block_emissions(self.nearest, self.seen, self.speed, self.priors, params)
        with_cue = decoder.block_emissions(self.nearest, self.seen, self.speed, self.priors,
                                           params, cue=self.cue)
        np.testing.assert_array_equal(base, with_cue)

    def test_cue_blocks_move_towards_dead_ball(self):
        params = dict(decoder.DEFAULTS, deadball_w=1.0)
        base = decoder.block_emissions(self.nearest, self.seen, self.speed, self.priors,
                                       dict(decoder.DEFAULTS))
        with_cue = decoder.block_emissions(self.nearest, self.seen, self.speed, self.priors,
                                           params, cue=self.cue)
        lift = (with_cue[:, decoder.D] - with_cue[:, decoder.H]) - \
            (base[:, decoder.D] - base[:, decoder.H])
        index = np.arange(len(self.truth)) // decoder.BLOCK
        fired = np.bincount(index, weights=self.cue[:, 0].astype(float)) > 0
        self.assertTrue((lift[fired] > 0).all())
        self.assertTrue((lift[~fired] < 0).all())             # no cue is weak alive evidence

    def test_decoding_is_deterministic(self):
        params = dict(decoder.DEFAULTS, deadball_w=1.0)
        runs = [decoder.decode_half(self.nearest, self.seen, self.speed, self.priors, params,
                                    FPS, cue=self.cue) for _ in range(2)]
        np.testing.assert_array_equal(runs[0], runs[1])


class ThinBeyondTest(unittest.TestCase):
    def setUp(self):
        n = 300
        x, y = two_players(n)
        bx, by = np.full(n, 50.0), np.full(n, 30.0)
        by[20:40] = -2.0                       # three stretches beyond a line by 2 m
        bx[100:130] = LENGTH + 2.0
        by[200:220] = WIDTH + 0.5              # and one only 0.5 m beyond
        self.match = match_with(x, y, bx, by)

    def test_keep_one_is_identity(self):
        out = thin_beyond(self.match, 1.0, seed=0)
        np.testing.assert_array_equal(out.ball_x, self.match.ball_x)

    def test_keep_zero_hides_every_stretch_beyond_the_margin(self):
        out = thin_beyond(self.match, 0.0, seed=0, margin_m=1.0)
        self.assertFalse(out.ball_seen[20:40].any())
        self.assertFalse(out.ball_seen[100:130].any())
        self.assertTrue(out.ball_seen[200:220].all())          # only 0.5 m beyond: kept
        self.assertTrue(out.ball_seen[:20].all() and out.ball_seen[40:100].all())
        self.assertTrue(self.match.ball_seen.all())            # the input is not touched

    def test_same_seed_same_result(self):
        a = thin_beyond(self.match, 0.5, seed=4)
        b = thin_beyond(self.match, 0.5, seed=4)
        np.testing.assert_array_equal(a.ball_seen, b.ball_seen)


EVENTS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<PutDataRequest>
<Event EventTime="2023-05-27T15:30:00.000+02:00"><KickOff GameSection="firstHalf"/></Event>
<Event EventTime="2023-05-27T15:30:10.000+02:00"><ThrowIn Team="t"/></Event>
<Event EventTime="2023-05-27T15:30:20.000+02:00"><Delete/></Event>
<Event EventTime="2023-05-27T15:30:30.000+02:00"><Play Team="t"/></Event>
<Event EventTime="2023-05-27T15:30:40.000+02:00"><KickOff GameSection="firstHalf"/></Event>
<Event EventTime="2023-05-27T16:30:02.000+02:00"><KickOff GameSection="secondHalf"/></Event>
<Event EventTime="2023-05-27T16:30:04.000+02:00"><CornerKick Team="t"/></Event>
</PutDataRequest>
"""


class RestartsTest(unittest.TestCase):
    def test_events_land_on_the_stitched_timeline(self):
        folder = tempfile.mkdtemp()
        try:
            with open(os.path.join(folder, "DFL_03_02_events_raw_DFL-MAT-TEST.xml"), "w") as f:
                f.write(EVENTS_XML)
            first = idsse._seconds("2023-05-27T13:30:00.000+00:00")
            second = idsse._seconds("2023-05-27T14:30:00.000+00:00")
            clocks = [("firstHalf", 10000, 10000 + 60 * FPS - 1, first),
                      ("secondHalf", 100000, 100000 + 60 * FPS - 1, second)]
            with mock.patch.object(idsse, "half_clocks", lambda *args: clocks):
                found = idsse.restarts("TEST", folder)
        finally:
            shutil.rmtree(folder)
        offset = 60 * FPS + idsse.HALF_GAP
        self.assertEqual(found, [(0, "kickoff"), (10 * FPS, "throw_in"), (40 * FPS, "goal"),
                                 (offset + 2 * FPS, "kickoff"), (offset + 4 * FPS, "corner")])


class OracleRepairTest(unittest.TestCase):
    def setUp(self):
        h, a, d = pm.H, pm.A, pm.D
        self.truth = np.array([h] * 10 + [d] * 10 + [a] * 10 + [d] * 5 + [a] * 10, dtype=np.int8)
        # Late into the first stoppage, a false stoppage, the second one missed.
        self.pred = np.array([h] * 13 + [d] * 9 + [a] * 4 + [d] * 2 + [a] * 17, dtype=np.int8)
        self.runs = pm.valid_runs(self.truth)

    def test_dead_stretches_and_overlaps(self):
        true_d = deadball_check.dead_stretches(self.truth, self.runs)
        pred_d = deadball_check.dead_stretches(self.pred, self.runs)
        self.assertEqual(true_d, [(10, 20), (30, 35)])
        self.assertEqual(pred_d, [(13, 22), (26, 28)])
        self.assertEqual(deadball_check.overlaps(true_d, pred_d), [[0], []])

    def test_restart_kinds_are_one_to_one(self):
        kinds = deadball_check.restart_kinds([(10, 20), (30, 35)],
                                             [(21, "throw_in"), (22, "corner"), (36, "goal")],
                                             FPS)
        self.assertEqual(kinds, ["throw_in", "goal"])

    def test_perfect_repair_gives_the_true_dead_ball_and_keeps_the_teams(self):
        records, false, repairs = deadball_check.analyse(self.truth, self.pred, [], FPS)
        perfect = repairs["perfect"]
        np.testing.assert_array_equal(perfect == pm.D, self.truth == pm.D)
        self.assertEqual(perfect[20], pm.A)    # wrongly D: the nearest frame the method labelled
        self.assertEqual(perfect[27], pm.A)
        self.assertEqual(false, [(26, 28)])
        self.assertEqual([r["found"] for r in records], [True, False])
        self.assertAlmostEqual(records[0]["start_s"], 3.0 / FPS)
        self.assertAlmostEqual(records[0]["end_s"], 2.0 / FPS)

    def test_single_repairs_touch_only_their_error(self):
        _, _, repairs = deadball_check.analyse(self.truth, self.pred, [], FPS)
        np.testing.assert_array_equal(repairs["false"][26:28], [pm.A, pm.A])
        np.testing.assert_array_equal(repairs["false"][:26], self.pred[:26])
        np.testing.assert_array_equal(repairs["missed"][30:35], [pm.D] * 5)
        np.testing.assert_array_equal(repairs["missed"][:30], self.pred[:30])
        np.testing.assert_array_equal(repairs["edge start"][10:13], [pm.D] * 3)
        self.assertEqual(repairs["edge start"][21], pm.D)      # the end is left alone
        self.assertEqual(repairs["edge end"][21], pm.A)

    def test_whistle_repairs_only_whistled_restarts(self):
        # First stoppage ends in a free kick (whistled), the missed one in a throw-in.
        events = [(20, "free_kick"), (35, "throw_in")]
        _, _, repairs = deadball_check.analyse(self.truth, self.pred, events, FPS)
        np.testing.assert_array_equal(repairs["whistle start"][10:13], [pm.D] * 3)
        self.assertEqual(repairs["whistle start"][21], pm.D)    # its end is left alone
        np.testing.assert_array_equal(repairs["whistle start+missed"][30:35], self.pred[30:35])
        np.testing.assert_array_equal(repairs["whistle all"] == pm.D,
                                      repairs["type free_kick"] == pm.D)
        # Swap the kinds: now only the missed stretch is whistled.
        events = [(20, "throw_in"), (35, "free_kick")]
        _, _, repairs = deadball_check.analyse(self.truth, self.pred, events, FPS)
        filled = pm.fill_abstentions(self.pred, self.runs)
        np.testing.assert_array_equal(repairs["whistle start"], filled)
        np.testing.assert_array_equal(repairs["whistle start+missed"][30:35], [pm.D] * 5)
        np.testing.assert_array_equal(repairs["whistle start+missed"][:30], filled[:30])


if __name__ == "__main__":
    unittest.main()
