"""Tests for the proxy league and what it rests on. Standard library unittest, no data needed.

    .venv/bin/python -m unittest discover -s tests -v       # or: make test

Synthetic inputs only, so the suite runs on a fresh clone in seconds and never touches the
competition data or the external downloads.
"""

import os
import sys
import tempfile
import unittest
import zlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution.match import Match, pm, validate  # noqa: E402
from solution import simulate  # noqa: E402
from solution.sources import idsse  # noqa: E402
from tools import proxy_league  # noqa: E402


def random_labels(rng, halves, n_frames, p_dead=0.3):
    """Scorer codes made of random-length segments inside each half, X elsewhere."""
    codes = np.full(n_frames, pm.X, dtype=np.int8)
    for start, stop in halves:
        position = start
        while position < stop:
            length = int(rng.randint(5, 400))
            codes[position:min(stop, position + length)] = (
                pm.D if rng.rand() < p_dead else rng.choice([pm.H, pm.A]))
            position += length
    return codes


def noisy_copy(rng, codes, flip=0.01):
    """A prediction: the truth with random short flips, and some X inside the halves."""
    pred = codes.copy()
    alive = np.flatnonzero(codes != pm.X)
    for frame in rng.choice(alive, size=int(flip * len(alive)), replace=False):
        pred[frame:frame + int(rng.randint(1, 60))] = rng.choice([pm.H, pm.A, pm.D, pm.X])
    pred[codes == pm.X] = pm.X
    return pred


def synthetic_match(name="synthetic", n_frames=6000, n_slots=6, seed=0):
    rng = np.random.RandomState(seed)
    halves = [(0, 2800), (3000, 6000)]
    shape = (n_frames, n_slots)
    return Match(name, 25, (105.0, 68.0),
                 (rng.rand(*shape) * 105).astype(np.float32),
                 (rng.rand(*shape) * 68).astype(np.float32),
                 np.array([0, 0, 0, 1, 1, 1], dtype=np.int8),
                 (rng.rand(n_frames) * 105).astype(np.float32),
                 (rng.rand(n_frames) * 68).astype(np.float32),
                 halves, random_labels(rng, halves, n_frames), source="test")


class PooledScoreMatchesOfficialScorer(unittest.TestCase):
    """The proxy league's pooled number must equal the official scorer on the matches laid end
    to end. Everything we decide rests on this identity."""

    def test_pooled_equals_concatenated_official_score(self):
        rng = np.random.RandomState(1)
        gts, preds = [], []
        for _ in range(3):
            gt = random_labels(rng, [(100, 3000), (3300, 6000)], 6200)
            gts.append(gt)
            preds.append(noisy_copy(rng, gt))

        flags, n_pred = [], 0
        for gt, pred in zip(gts, preds):
            f, n = proxy_league.spell_flags(gt, pred)
            self.assertAlmostEqual(proxy_league.f1_from_flags(f, n),
                                   pm.evaluate(gt, pred)["spell_f1_mean"], places=12)
            flags.append(f)
            n_pred += n
        pooled = proxy_league.f1_from_flags(np.concatenate(flags, axis=1), n_pred)

        gap = np.full(250, pm.X, dtype=np.int8)               # any X gap keeps matches apart
        gt_all = np.concatenate([gts[0], gap, gts[1], gap, gts[2]])
        pred_all = np.concatenate([preds[0], gap, preds[1], gap, preds[2]])
        self.assertAlmostEqual(pooled, pm.evaluate(gt_all, pred_all)["spell_f1_mean"], places=12)

    def test_paired_interval_of_a_method_against_itself_is_zero(self):
        rng = np.random.RandomState(2)
        gt = random_labels(rng, [(0, 5000)], 5000)
        flags, n = proxy_league.spell_flags(gt, noisy_copy(rng, gt))
        self.assertEqual(proxy_league.bootstrap(flags, n, 50, other=(flags, n)), (0.0, 0.0))


def run_lengths(mask):
    """(lengths of True runs, lengths of False runs), each sorted."""
    edges = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    pieces = np.split(mask, edges)
    return (sorted(len(p) for p in pieces if p[0]), sorted(len(p) for p in pieces if not p[0]))


class Simulator(unittest.TestCase):

    def test_same_seed_same_output_and_labels_untouched(self):
        match = synthetic_match()
        for version in simulate.VERSIONS:
            a, b = simulate.degrade(match, 3, version), simulate.degrade(match, 3, version)
            for field in ("ball_x", "player_x", "player_y", "team"):
                np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
            np.testing.assert_array_equal(a.labels, match.labels)
            validate(a)
        self.assertFalse(np.isnan(match.ball_x).any(), "degrade must not modify its input")
        self.assertFalse(np.isnan(match.player_x).any(), "degrade must not modify its input")

    def test_retime_slows_every_speed_and_keeps_positions(self):
        n = 6000
        frames = np.arange(n, dtype=np.float64)
        match = synthetic_match()
        # Smooth tracks with known speeds: slot s moves along x at (s + 1) / 1000 m per frame.
        match.player_x = (10 + np.outer(frames, np.arange(1, 7)) / 1000.0).astype(np.float32)
        match.player_x[100:110, 2] = np.nan
        match.ball_x = (20 + frames / 500.0).astype(np.float32)
        slow = simulate.retime(match, 0.9)
        validate(slow)
        self.assertEqual(slow.notes["tempo"], 0.9)
        self.assertAlmostEqual(slow.n_frames / float(n), 1 / 0.9, places=3)
        # Speeds x 0.9, per player and for the ball.
        fast_v = np.diff(match.player_x[:, 4])[:50].mean()
        slow_v = np.diff(slow.player_x[:, 4])[:50].mean()
        self.assertAlmostEqual(slow_v / fast_v, 0.9, places=3)
        self.assertAlmostEqual(np.diff(slow.ball_x)[:50].mean() / (1 / 500.0), 0.9, places=3)
        # Positions unchanged: same range, every new position lies on the old track.
        self.assertAlmostEqual(float(np.nanmin(slow.player_x[:, 0])), 10.0, places=4)
        self.assertAlmostEqual(float(np.nanmax(slow.player_x[:, 5])),
                               float(match.player_x[-1, 5]), places=2)
        # Missing stays missing (and the neighbours that mix with it).
        gone = np.isnan(slow.player_x[:, 2])
        self.assertTrue(gone.any())
        self.assertTrue(np.all((np.floor(np.flatnonzero(gone) * 0.9) >= 99) &
                               (np.floor(np.flatnonzero(gone) * 0.9) < 110)))
        # Labels are the old frame's; halves agree with them (validate checked the runs).
        old = np.floor(np.arange(slow.n_frames) * 0.9).astype(int)
        np.testing.assert_array_equal(slow.labels, match.labels[old])
        self.assertEqual(slow.halves[1][0], simulate.retime_frame(3000, 0.9))
        # Durations of labelled stretches grow by 1 / 0.9.
        self.assertAlmostEqual(np.sum(slow.labels == pm.D) / float(np.sum(match.labels == pm.D)),
                               1 / 0.9, places=2)
        # Factor 1 is a plain copy; the input is untouched.
        same = simulate.retime(match, 1.0)
        np.testing.assert_array_equal(same.player_x, match.player_x)
        self.assertEqual(match.n_frames, n)
        validate(simulate.degrade(slow, 0))

    def test_different_seed_different_mask(self):
        match = synthetic_match()
        for version in simulate.VERSIONS:
            a, b = simulate.degrade(match, 0, version), simulate.degrade(match, 1, version)
            self.assertFalse(np.array_equal(np.isnan(a.ball_x), np.isnan(b.ball_x)))

    def test_unknown_version_is_refused(self):
        with self.assertRaises(ValueError):
            simulate.degrade(synthetic_match(), 0, "v9")

    def test_v0_is_unchanged(self):
        # v0 stays reproducible: the ball mask of the original implementation, players untouched.
        match = synthetic_match()
        out = simulate.degrade(match, 2, "v0")
        rng = np.random.RandomState((zlib.crc32(match.name.encode()) + 7919 * 2) % (2 ** 32))
        for start, stop in match.halves:
            np.testing.assert_array_equal(np.isnan(out.ball_x[start:stop]),
                                          simulate.ball_dropout_mask(stop - start, rng))
        np.testing.assert_array_equal(out.player_x, match.player_x)

    def test_v1_reorders_v0_stretches_without_changing_their_lengths(self):
        n = 40000
        speed = np.random.RandomState(0).rand(n) * 4
        drawn = simulate.ball_dropout_mask(n, np.random.RandomState(7))
        mask = simulate.ball_gap_mask_v1(speed, np.random.RandomState(7), 25)
        self.assertEqual(run_lengths(mask), run_lengths(drawn))
        self.assertFalse(np.array_equal(mask, drawn))

    def test_v1_puts_gaps_in_slow_play(self):
        # Alternating slow and fast spells; v1 should hide the ball mostly in the slow ones.
        rng = np.random.RandomState(1)
        speed = np.concatenate([np.full(int(rng.randint(100, 1500)), level)
                                for _ in range(80) for level in (0.8, 2.6)])[:70000]
        ratios = {}
        for weight in (0.0, 1.0):
            params = dict(simulate.V1, speed_weight=weight)
            hidden = np.concatenate([simulate.ball_gap_mask_v1(speed, np.random.RandomState(s),
                                                               25, params) for s in range(4)])
            tiled = np.tile(speed, 4)
            ratios[weight] = tiled[hidden].mean() / tiled[~hidden].mean()
        self.assertAlmostEqual(ratios[0.0], 1.0, delta=0.1)
        self.assertLess(ratios[1.0], 0.7)

    def test_v1_players_lose_points_and_gain_same_team_ghosts(self):
        match = synthetic_match(n_frames=40000)
        match.halves = [(0, 19000), (20000, 40000)]
        match.labels = random_labels(np.random.RandomState(0), match.halves, match.n_frames)
        out = simulate.degrade(match, 0, "v1")
        slots = match.team.size
        clean_x = out.player_x[:, :slots]
        kept = np.isfinite(clean_x)
        self.assertLess(kept.mean(), 0.99)                          # some points are dropped
        np.testing.assert_array_equal(clean_x[kept], match.player_x[kept])  # never moved
        np.testing.assert_array_equal(out.team[:slots], match.team)
        self.assertGreater(out.team.size, slots)                    # ghost slots were added
        self.assertTrue(set(out.team.tolist()) <= {0, 1})
        self.assertEqual(out.player_x.shape, (match.n_frames, out.team.size))
        ghost_x = out.player_x[:, slots:]
        outside = ~match.in_play_mask()
        self.assertFalse(np.isfinite(ghost_x[outside]).any(), "ghosts stay inside the halves")

    def test_coverage_close_to_bar_ath(self):
        # Long synthetic halves so the alternating process averages out.
        mask_share = []
        rng = np.random.RandomState(0)
        for _ in range(20):
            mask_share.append(simulate.ball_dropout_mask(80000, rng).mean())
        _, _, visible_share = simulate.ball_profile()
        self.assertAlmostEqual(1 - np.mean(mask_share), visible_share, delta=0.05)


POSITIONS = """<?xml version="1.0" encoding="UTF-8"?>
<PutDataRequest>
<Positions>
<MetaData MatchId="DFL-MAT-TEST01" Type="pitch-size">
<PitchSize X="105.00" Y="68.00" />
</MetaData>
<FrameSet GameSection="firstHalf" MatchId="DFL-MAT-TEST01" TeamId="CLU-HOME" PersonId="P-H1">
<Frame N="10000" T="t" X="-10.00" Y="5.00" D="0" S="0" A="0" M="1"/>
<Frame N="10001" T="t" X="-9.00" Y="5.00" D="0" S="0" A="0" M="1"/>
<Frame N="10002" T="t" X="-8.00" Y="5.00" D="0" S="0" A="0" M="1"/>
</FrameSet>
<FrameSet GameSection="firstHalf" MatchId="DFL-MAT-TEST01" TeamId="CLU-AWAY" PersonId="P-A1">
<Frame N="10000" T="t" X="10.00" Y="-5.00" D="0" S="0" A="0" M="1"/>
<Frame N="10002" T="t" X="12.00" Y="-5.00" D="0" S="0" A="0" M="1"/>
</FrameSet>
<FrameSet GameSection="firstHalf" MatchId="DFL-MAT-TEST01" TeamId="referee" PersonId="P-REF">
<Frame N="10000" T="t" X="0.50" Y="0.00" D="0" S="0" A="0" M="1"/>
<Frame N="10001" T="t" X="1.50" Y="0.00" D="0" S="0" A="0" M="1"/>
</FrameSet>
<FrameSet GameSection="firstHalf" MatchId="DFL-MAT-TEST01" TeamId="BALL" PersonId="BALL1">
<Frame N="10000" T="t" X="0.00" Y="0.00" Z="0.1" D="0" S="0" A="0" M="1" BallPossession="1" BallStatus="1"/>
<Frame N="10001" T="t" X="1.00" Y="0.00" Z="0.1" D="0" S="0" A="0" M="1" BallPossession="2" BallStatus="1"/>
<Frame N="10002" T="t" X="2.00" Y="0.00" Z="0.1" D="0" S="0" A="0" M="1" BallPossession="2" BallStatus="0"/>
</FrameSet>
<FrameSet GameSection="secondHalf" MatchId="DFL-MAT-TEST01" TeamId="BALL" PersonId="BALL1">
<Frame N="100000" T="t" X="0.00" Y="0.00" Z="0.1" D="0" S="0" A="0" M="1" BallPossession="2" BallStatus="1"/>
<Frame N="100001" T="t" X="0.00" Y="0.00" Z="0.1" D="0" S="0" A="0" M="1" BallPossession="1" BallStatus="1"/>
</FrameSet>
</Positions>
</PutDataRequest>
"""

INFO = """<?xml version="1.0" encoding="UTF-8"?>
<PutDataRequest><MatchInformation><Teams>
<Team TeamId="CLU-AWAY" TeamName="Away" Role="guest"/>
<Team TeamId="CLU-HOME" TeamName="Home" Role="home"/>
</Teams></MatchInformation></PutDataRequest>
"""


class IdsseConversion(unittest.TestCase):

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        for kind, text in (("02_01_matchinformation", INFO), ("04_03_positions_raw_observed", POSITIONS)):
            with open(os.path.join(self.folder.name,
                                   "DFL_{}_DFL-COM-X_DFL-MAT-TEST01.xml".format(kind)), "w") as h:
                h.write(text)
        self.match = idsse.build("TEST01", self.folder.name)

    def tearDown(self):
        self.folder.cleanup()

    def test_labels_follow_ball_status_then_possession(self):
        m = self.match
        first, second = m.halves
        self.assertEqual(list(m.labels[first[0]:first[1]]), [pm.H, pm.A, pm.D])
        self.assertEqual(list(m.labels[second[0]:second[1]]), [pm.A, pm.H])

    def test_halves_are_separated_by_x(self):
        m = self.match
        self.assertEqual(m.halves, [(0, 3), (3 + idsse.HALF_GAP, 5 + idsse.HALF_GAP)])
        self.assertTrue(np.all(m.labels[3:3 + idsse.HALF_GAP] == pm.X))
        self.assertEqual(len(pm.valid_runs(m.labels)), 2)

    def test_referees_are_not_players(self):
        self.assertEqual(self.match.player_x.shape[1], 2)
        self.assertEqual(list(self.match.team), [0, 1])

    def test_home_from_role_and_positions_shifted_to_corner_origin(self):
        m = self.match
        self.assertEqual(list(m.team), [0, 1])                  # home person sorts first
        self.assertAlmostEqual(float(m.player_x[0, 0]), -10.0 + 52.5, places=4)
        self.assertAlmostEqual(float(m.player_y[0, 1]), -5.0 + 34.0, places=4)
        self.assertTrue(np.isnan(m.player_x[1, 1]))             # away frame 10001 missing
        self.assertAlmostEqual(float(m.ball_x[1]), 1.0 + 52.5, places=4)


class MatchContainer(unittest.TestCase):

    def test_save_load_roundtrip(self):
        match = synthetic_match()
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "m.npz")
            match.save(path)
            back = validate(Match.load(path))
        for field in ("player_x", "player_y", "team", "ball_x", "ball_y", "labels"):
            np.testing.assert_array_equal(getattr(back, field), getattr(match, field))
        self.assertEqual(back.halves, match.halves)

    def test_validate_rejects_labels_outside_halves(self):
        match = synthetic_match()
        match.labels[2900] = pm.H                               # between the halves
        with self.assertRaises(ValueError):
            validate(match)


if __name__ == "__main__":
    unittest.main()
