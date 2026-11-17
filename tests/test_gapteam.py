"""Tests for the E5 team cues (solution/gapteam.py), their decoder term and tools/gap_check.py.

Synthetic inputs only, no data needed:
    .venv/bin/python -m unittest tests.test_gapteam -v        # or: make test
"""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import decoder, gapteam  # noqa: E402
from solution.match import Match, pm  # noqa: E402
from tests.test_decoder import synthetic_evidence, synthetic_priors  # noqa: E402
from tools import gap_check  # noqa: E402

FPS = 25
LENGTH, WIDTH = 105.0, 68.0
TEAM = np.array([0] * 6 + [1] * 6, dtype=np.int8)


def formation(n, home_x, away_x, home_width, away_width):
    """(x, y) for 6 home and 6 away players: a keeper at each goal, five outfield in a column
    at the given depth spread over the given width, centred on the pitch."""
    x = np.zeros((n, 12))
    y = np.zeros((n, 12))
    for columns, keeper_x, depth, width in ((slice(0, 6), 2.0, home_x, home_width),
                                            (slice(6, 12), LENGTH - 2.0, away_x, away_width)):
        x[:, columns] = depth
        x[:, columns.start] = keeper_x
        y[:, columns.start] = WIDTH / 2.0
        y[:, columns.start + 1:columns.stop] = WIDTH / 2.0 + np.linspace(-width / 2.0,
                                                                         width / 2.0, 5)
    return x, y


def match_with(x, y, ball_x=None, ball_y=None, halves=None, labels=None):
    n = len(x)
    ball_x = np.full(n, np.nan) if ball_x is None else ball_x
    ball_y = np.full(n, np.nan) if ball_y is None else ball_y
    return Match("synthetic_gapteam", FPS, (LENGTH, WIDTH), x.astype(np.float32),
                 y.astype(np.float32), TEAM, ball_x.astype(np.float32),
                 ball_y.astype(np.float32), halves or [(0, n)], labels)


class CueTest(unittest.TestCase):
    def test_attack_direction_follows_each_half(self):
        x, y = formation(200, 30.0, 70.0, 40.0, 40.0)
        x[100:] = LENGTH - x[100:]                 # the teams change ends at half time
        sign = gapteam.attack_sign(match_with(x, y, halves=[(0, 100), (100, 200)]))
        self.assertTrue((sign[:100] == 1).all())
        self.assertTrue((sign[100:] == -1).all())

    def test_spread_is_home_width_minus_away_width_in_both_halves(self):
        x, y = formation(200, 40.0, 60.0, 50.0, 20.0)
        x[100:] = LENGTH - x[100:]
        match = match_with(x, y, halves=[(0, 100), (100, 200)])
        np.testing.assert_allclose(gapteam.spread(match), 30.0, atol=1e-4)
        gyration = gapteam.spread(match, dict(gapteam.PARAMS, spread="gyration"))
        self.assertTrue((gyration > 0).all())

    def test_spread_leaves_out_the_keeper_and_needs_enough_players(self):
        x, y = formation(10, 40.0, 60.0, 30.0, 30.0)
        y[:, 0] = 0.0                              # the home keeper stands far wide
        x[5:, 1:4] = np.nan                        # three home outfielders lost: two left
        spread = gapteam.spread(match_with(x, y))
        np.testing.assert_allclose(spread[:5], 0.0, atol=1e-4)
        self.assertTrue(np.isnan(spread[5:]).all())

    def test_line_is_positive_when_home_pushes_up(self):
        x, y = formation(50, 60.0, 80.0, 40.0, 40.0)   # home line 60 m up, away 25 m up
        np.testing.assert_allclose(gapteam.line(match_with(x, y)), 35.0, atol=1e-4)

    def test_advance_is_signed_by_the_attack_direction(self):
        n = 200
        x, y = formation(n, 40.0, 60.0, 40.0, 40.0)
        drift = 2.0 * np.arange(n) / FPS
        x[:100] += drift[:100, None]               # half 1: everyone runs +x, home attacks +x
        x[100:] = LENGTH - x[100:] - drift[100:, None]   # half 2: ends changed, everyone runs -x
        value = gapteam.advance(match_with(x, y, halves=[(0, 100), (100, 200)]))
        np.testing.assert_allclose(value[30:70], 2.0, atol=1e-4)
        np.testing.assert_allclose(value[130:170], 2.0, atol=1e-4)

    def test_pressure_counts_round_the_last_seen_ball_for_the_hold(self):
        n = 200
        x, y = formation(n, 40.0, 60.0, 40.0, 40.0)
        x[:, 1:3] = 50.0                           # two home players at the ball
        y[:, 1:3] = WIDTH / 2.0
        bx, by = np.full(n, np.nan), np.full(n, np.nan)
        bx[:20], by[:20] = 50.0, WIDTH / 2.0       # seen for 20 frames, then lost
        match = match_with(x, y, bx, by, halves=[(0, 120), (120, 200)])
        value = gapteam.pressure(match)
        hold = int(gapteam.PARAMS["pressure_hold_s"] * FPS)
        self.assertTrue((value[:20 + hold] >= 2).all())
        self.assertTrue(np.isnan(value[20 + hold:]).all())   # incl. the second half

    def test_bins_mirror_around_zero(self):
        for name, (half, n) in gapteam.CUE_BINS.items():
            values = np.linspace(-1.3 * half, 1.3 * half, 97) + 1e-3
            np.testing.assert_array_equal(gapteam.bins(-values, name),
                                          n - 1 - gapteam.bins(values, name))
            self.assertEqual(int(gapteam.bins(np.array([0.0]), name)[0]), n // 2)


class FitTest(unittest.TestCase):
    def setUp(self):
        n = 1000
        x, y = formation(n, 40.0, 60.0, 40.0, 40.0)
        labels = np.full(n, pm.H, dtype=np.int8)
        labels[400:800] = pm.A
        labels[800:] = pm.D
        y[:400, 1:6] = WIDTH / 2.0 + np.linspace(-30, 30, 5)    # home wide while it has the ball
        y[400:800, 7:12] = WIDTH / 2.0 + np.linspace(-30, 30, 5)
        self.match = match_with(x, y, labels=labels)
        self.params = dict(gapteam.PARAMS, cues=("spread",))
        self.table = gapteam.fit([self.match], self.params)[0]

    def test_possession_row_favours_the_wide_team(self):
        half, n = gapteam.CUE_BINS["spread"]
        middle = n // 2
        self.assertGreater(self.table[0][middle + 1:].max(), self.table[0][:middle].max())
        np.testing.assert_allclose(self.table[1], self.table[1][::-1])  # dead row symmetric

    def test_state_log_mirrors_away_and_max_gives_dead_no_say(self):
        values = np.array([-20.0, 0.0, 20.0])
        log = gapteam.state_log(values, self.table, "spread")
        np.testing.assert_allclose(log[:, 0], log[::-1, 1])
        self.assertGreater(log[2, 0], log[2, 1])
        top = gapteam.state_log(values, self.table, "spread", dead="max")
        np.testing.assert_allclose(top[:, 2], np.maximum(top[:, 0], top[:, 1]))


class DecoderTermTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(5)
        self.truth = np.repeat(rng.randint(0, 2, 40), 50).astype(np.int8)
        self.nearest, self.seen, self.speed = synthetic_evidence(self.truth, rng)
        table = np.log(np.array([[0.05, 0.05, 0.1, 0.3, 0.5],      # signed towards the team
                                 [0.1, 0.2, 0.4, 0.2, 0.1]]))      # dead ball
        self.priors = dict(synthetic_priors(), team_log=[table], team_names=("spread",))
        sign = np.where(self.truth == decoder.H, 1.0, -1.0)
        self.team = (sign * 20.0 + rng.normal(0, 5, len(self.truth)))[:, None]
        self.old_bins = gapteam.CUE_BINS["spread"]
        gapteam.CUE_BINS["spread"] = (25.0, 5)

    def tearDown(self):
        gapteam.CUE_BINS["spread"] = self.old_bins

    def emissions(self, **params):
        return decoder.block_emissions(self.nearest, self.seen, self.speed, self.priors,
                                       dict(decoder.DEFAULTS, **params), team=self.team)

    def test_weight_zero_leaves_emissions_byte_identical(self):
        base = decoder.block_emissions(self.nearest, self.seen, self.speed, self.priors,
                                       dict(decoder.DEFAULTS))
        np.testing.assert_array_equal(base, self.emissions())

    def test_gaps_only_touches_blocks_without_the_ball(self):
        base, with_cue = self.emissions(), self.emissions(team_w=1.0)
        index = np.arange(len(self.truth)) // decoder.BLOCK
        any_gap = np.bincount(index, weights=(~self.seen).astype(float)) > 0
        np.testing.assert_array_equal(base[~any_gap], with_cue[~any_gap])
        home = np.bincount(index, weights=(self.truth == decoder.H).astype(float)) > 0
        lift = (with_cue[:, decoder.H] - with_cue[:, decoder.A]) - \
            (base[:, decoder.H] - base[:, decoder.A])
        self.assertTrue((lift[any_gap & home] > 0).all())
        self.assertTrue((lift[any_gap & ~home] < 0).all())

    def test_all_frames_and_max_leave_dead_ball_its_best_team_score(self):
        out = self.emissions(team_w=1.0, team_where="all", team_dead="max")
        base = self.emissions()
        gain = out - base
        np.testing.assert_allclose(gain[:, decoder.D], gain[:, :2].max(axis=1))

    def test_decoding_is_deterministic(self):
        params = dict(decoder.DEFAULTS, team_w=1.0, team_where="all")
        runs = [decoder.decode_half(self.nearest, self.seen, self.speed, self.priors, params,
                                    FPS, team=self.team) for _ in range(2)]
        np.testing.assert_array_equal(runs[0], runs[1])


class GapCheckTest(unittest.TestCase):
    H, A, D, X = pm.H, pm.A, pm.D, pm.X

    def test_gap_spans_stay_inside_the_halves(self):
        x, y = formation(30, 40.0, 60.0, 40.0, 40.0)
        bx = np.full(30, 50.0)
        bx[5:15] = np.nan
        bx[18:30] = np.nan
        match = match_with(x, y, bx, np.full(30, 30.0), halves=[(0, 20), (22, 30)])
        self.assertEqual(gap_check.gap_spans(match), [(5, 15), (18, 20), (22, 30)])

    def test_change_kinds(self):
        H, A, D = self.H, self.A, self.D
        truth = np.array([H, H, H, D, D, A, A, A, H, H], dtype=np.int8)
        runs = [(0, 10)]
        carried = gap_check.carried_team(truth, runs)
        np.testing.assert_array_equal(carried, [H, H, H, H, H, A, A, A, H, H])
        self.assertEqual(gap_check.change_kind(truth, carried, 2, 7, 0), "restart")
        self.assertEqual(gap_check.change_kind(truth, carried, 6, 10, 0), "in play")
        self.assertEqual(gap_check.change_kind(truth, carried, 0, 3, 0), "no change")
        same = np.array([H, H, D, D, H, H], dtype=np.int8)
        self.assertEqual(gap_check.change_kind(same, gap_check.carried_team(same, [(0, 6)]),
                                               1, 6, 0), "no change")

    def test_team_repair_fixes_only_both_alive_frames_inside_the_mask(self):
        H, A, D = self.H, self.A, self.D
        truth = np.array([H, H, D, A, A, A, H, H], dtype=np.int8)
        pred = np.array([H, H, H, D, H, H, H, A], dtype=np.int8)
        mask = np.array([0, 1, 1, 1, 1, 1, 1, 0], dtype=bool)
        out = gap_check.team_repair(truth, pred, [(0, 8)], mask)
        np.testing.assert_array_equal(out, [H, H, H, D, A, A, H, A])


def gap_match(n, gap, labels=None):
    """A synthetic match with the ball seen everywhere except frames [gap[0], gap[1])."""
    x, y = formation(n, 40.0, 60.0, 40.0, 40.0)
    bx = np.full(n, 50.0)
    bx[gap[0]:gap[1]] = np.nan
    return match_with(x, y, bx, np.full(n, 30.0), labels=labels)


class TimingTest(unittest.TestCase):
    """E5 part 2, Step 0: hand-over timing inside gaps (tools/gap_check.py --timing)."""
    H, A = pm.H, pm.A

    def codes(self, change, n=400):
        out = np.full(n, self.H, dtype=np.int8)
        out[change:] = self.A
        return out

    def snap(self, match, truth, pred, tol_s):
        gap_id, _, bins = gap_check.gap_ids(match)
        return gap_check.timing_snap(truth, pred, gap_id, bins, tol_s, FPS)

    def test_snap_moves_a_late_change_onto_the_true_hand_over_in_a_gap(self):
        truth, pred = self.codes(200), self.codes(230)
        out = self.snap(gap_match(400, (150, 300)), truth, pred, 1.5)
        np.testing.assert_array_equal(out, truth)

    def test_snap_leaves_hand_overs_outside_gaps_and_beyond_the_tolerance(self):
        truth, pred = self.codes(200), self.codes(230)
        np.testing.assert_array_equal(self.snap(gap_match(400, (0, 100)), truth, pred, 1.5),
                                      pred)
        far = self.codes(290)                  # 3.6 s late: beyond +-3 s, inside the gap
        match = gap_match(400, (150, 300))
        np.testing.assert_array_equal(self.snap(match, truth, far, 3.0), far)
        np.testing.assert_array_equal(self.snap(match, truth, far, None), truth)

    def test_snap_needs_the_same_destination_team(self):
        truth = self.codes(200)
        pred = np.full(400, self.A, dtype=np.int8)
        pred[230:] = self.H                    # a change the other way round
        np.testing.assert_array_equal(self.snap(gap_match(400, (150, 300)), truth, pred, 3.0),
                                      pred)

    def test_first_crossing_is_the_nearest_upward_one(self):
        value = np.array([-1, 1, -1, -1, np.nan, 2, 2, -1, 3], dtype=float)
        self.assertEqual(gap_check.first_crossing(value, 4, 0, 9), 1)   # frame 5, NaN skipped
        self.assertEqual(gap_check.first_crossing(value, 0, 0, 9), 1)
        self.assertIsNone(gap_check.first_crossing(np.ones(5), 2, 0, 5))

    def test_records_time_the_decoder_and_the_new_possession(self):
        truth = self.codes(200)
        truth[220:] = self.H                   # a 0.8 s possession of A inside the gap
        match = gap_match(400, (150, 300), labels=truth)
        pred = self.codes(210)
        pred[240:] = self.H
        smooth = {1.0: np.where(np.arange(400) >= 205, -10.0, 10.0)}  # flips 5 frames late
        records = gap_check.timing_records(truth, pred, match, smooth)
        self.assertEqual(len(records), 2)      # H->A and A->H, both inside the gap
        first = records[0]
        self.assertEqual(first["new"], "<2 s")
        self.assertEqual(first["lat"][1.0], 10)
        self.assertEqual(first["gap_lat"], 10)
        self.assertEqual(first["flip"][1.0], 5)


class DecodedTest(unittest.TestCase):
    """Step 2: label-free decoded aggregates and the Smart Tagging team (no data needed)."""
    H, A, D = pm.H, pm.A, pm.D

    def test_aggregates_count_possessions_and_changes_inside_gaps(self):
        H, A, D = self.H, self.A, self.D
        n = 3000                                 # 2 minutes
        match = gap_match(n, (1000, 2000))
        codes = np.full(n, H, dtype=np.int8)
        codes[1500:1525] = A                     # 1 s possession of A inside the gap
        codes[2500:2600] = D
        tag = np.full(n, -1)
        tag[1400:1600] = H                       # the tag says H throughout
        out = gap_check.decoded_aggregates(match, codes, tag)
        self.assertEqual(out["poss"], 4)         # H, A, H, then H after the dead ball
        self.assertEqual(out["poss_gap"], 1)
        self.assertEqual(out["short_gap"], 1)
        self.assertEqual(out["changes_gap"], 2)
        self.assertEqual((out["gapH"], out["gapA"], out["gapD"]), (975, 25, 0))
        self.assertEqual(out["seenD"], 100)
        self.assertEqual((out["gap_tagged"], out["gap_agree"]), (200, 175))
        self.assertAlmostEqual(out["minutes"], 2.0)

    def test_smart_tag_team_maps_home_and_mirrors_defending_codes(self):
        import types
        from unittest import mock

        def tag(code, team, a, b):
            return {"code": code, "tags": {"Team": team}, "start_frame": a, "end_frame": b}
        tags = [tag("BUILD UP", "Barcelona", 0, 50),
                tag("DEFENDING IN ATTACKING THIRD", "Atletic Club", 0, 50),
                tag("PROGRESSION", "Atletic Club", 60, 80),
                tag("Defending in Middle Third", "Barcelona", 60, 80),
                tag("ATTACKING TRANSITION", "Barcelona", 75, 90),     # clashes on 75..80
                tag("SET PIECES", "N/A", 0, 100),
                tag("BUILD UP", "N/A", 90, 100)]
        fakes = {"load_smart_tagging": types.SimpleNamespace(load=lambda: tags),
                 "baseline": types.SimpleNamespace(team_names=lambda path: ("Barcelona", "x")),
                 "games": types.SimpleNamespace(GAME="g", require=lambda keys, game: ["meta"])}
        x, y = formation(100, 40.0, 60.0, 40.0, 40.0)
        with mock.patch.dict(sys.modules, fakes):
            team, counts = gap_check.smart_tag_team(match_with(x, y))
        np.testing.assert_array_equal(team[:50], self.H)
        np.testing.assert_array_equal(team[60:75], self.A)
        np.testing.assert_array_equal(team[75:80], -1)
        np.testing.assert_array_equal(team[80:90], self.H)
        np.testing.assert_array_equal(team[90:], -1)
        self.assertEqual(counts["mirror agree"], "1.000")
        self.assertEqual(counts["no team"], 1)


class CeilingTest(unittest.TestCase):
    """C3 Step 0 helpers (tools/gap_check.py --ceiling)."""
    H, A, D = pm.H, pm.A, pm.D

    def test_ideal_sign_lags_inside_each_half_and_skips_dead_ball(self):
        truth = np.full(200, self.H, dtype=np.int8)
        truth[60:] = self.A
        truth[90:100] = self.D
        x, y = formation(200, 40.0, 60.0, 40.0, 40.0)
        match = match_with(x, y, halves=[(0, 100), (100, 200)])
        sign = gap_check.ideal_sign(truth, match)
        self.assertEqual(sign[59], 1.0)
        self.assertEqual(sign[60], -1.0)
        self.assertTrue(np.isnan(sign[95]))
        lagged = gap_check.ideal_sign(truth, match, lag_s=0.4)      # 10 frames
        self.assertEqual(lagged[69], 1.0)
        self.assertEqual(lagged[70], -1.0)
        self.assertTrue(np.isnan(lagged[:10]).all())               # nothing to trail yet
        self.assertTrue(np.isnan(lagged[100:110]).all())           # nor after half time
        self.assertTrue(np.isnan(lagged[95]))

    def test_ideal_sign_flips_whole_chunks_deterministically(self):
        truth = np.full(1000, self.H, dtype=np.int8)
        x, y = formation(1000, 40.0, 60.0, 40.0, 40.0)
        match = match_with(x, y)
        one = gap_check.ideal_sign(truth, match, flip=(0.4, 0.3), seed=7)
        two = gap_check.ideal_sign(truth, match, flip=(0.4, 0.3), seed=7)
        np.testing.assert_array_equal(one, two)
        chunks = one.reshape(-1, 10)
        self.assertTrue((chunks == chunks[:, :1]).all())            # whole chunks only
        self.assertTrue(0.1 < np.mean(one < 0) < 0.5)

    def test_ceiling_masks_split_gaps_and_possessions(self):
        H, A = self.H, self.A
        truth = np.full(2000, H, dtype=np.int8)
        truth[1000:1030] = A                       # a 1.2 s possession inside the long gap
        truth[1030:] = H
        x, y = formation(2000, 40.0, 60.0, 40.0, 40.0)
        bx = np.full(2000, 50.0)
        bx[100:125] = np.nan                       # a 1 s gap
        bx[900:1900] = np.nan                      # a 40 s gap
        match = match_with(x, y, bx, np.full(2000, 30.0), labels=truth)
        masks, counts = gap_check.ceiling_masks(truth, match)
        self.assertEqual(counts["all gaps"], 2)
        self.assertEqual(counts["gap <2 s"], 1)
        self.assertEqual(counts["gap >30 s"], 1)
        self.assertEqual(masks["gap <2 s"].sum(), 25)
        self.assertEqual(masks["gap >30 s"].sum(), 1000)
        self.assertEqual(counts["poss <2 s"], 1)
        np.testing.assert_array_equal(np.flatnonzero(masks["poss <2 s"]), np.arange(1000, 1030))
        self.assertFalse(masks["poss <2 s"][:1000].any())


class DiagnosticTest(unittest.TestCase):
    """tools/gap_check.py --tagdiag and --seen-ceiling helpers."""
    H, A, D = pm.H, pm.A, pm.D

    def test_since_seen_restarts_each_half(self):
        x, y = formation(20, 40.0, 60.0, 40.0, 40.0)
        bx = np.full(20, 50.0)
        bx[3:10] = np.nan
        bx[10:13] = np.nan                         # the second half starts without the ball
        match = match_with(x, y, bx, np.full(20, 30.0), halves=[(0, 10), (10, 20)])
        since = gap_check.since_seen(match)
        self.assertEqual(since[2], 0.0)
        self.assertAlmostEqual(since[5], 3.0 / FPS)
        self.assertTrue(np.isinf(since[10:13]).all())
        self.assertEqual(since[13], 0.0)

    def test_thin_tracking_needs_both_teams(self):
        x, y = formation(4, 40.0, 60.0, 40.0, 40.0)
        x[2:, 0:2] = np.nan                        # home down to 4 tracked players
        thin = gap_check.thin_tracking(match_with(x, y))
        np.testing.assert_array_equal(thin, [False, False, True, True])

    def test_matched_pred_flags_mirror_the_scorer(self):
        overlaps = [(0.9, 0, 1), (0.6, 1, 1), (0.55, 1, 0), (0.3, 2, 2)]
        np.testing.assert_array_equal(gap_check.matched_pred_flags(overlaps, 3),
                                      [True, True, False])

    def test_error_kinds(self):
        H, A = self.H, self.A
        n = 400
        truth = np.full(n, H, dtype=np.int8)
        truth[200:] = A
        pred = truth.copy()
        pred[180:200] = A                          # early hand-over: boundary errors
        pred[50:70] = A                            # an invented away possession
        pred[300:310] = H                          # wrong team inside the long away spell
        match = gap_match(n, (0, 0))               # ball seen everywhere
        kinds, seen, invented = gap_check.error_kinds(truth, pred, match, FPS)
        self.assertEqual(seen, n)
        self.assertEqual(kinds["boundary"].sum(), 20)
        self.assertEqual(kinds["invented"].sum(), 30)   # both short decoded spells match nothing
        self.assertEqual(kinds["inside"].sum(), 0)
        self.assertTrue(invented[50:70].all() and invented[300:310].all())


if __name__ == "__main__":
    unittest.main()
