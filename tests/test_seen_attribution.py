"""Tests for the visible-ball Step 0 attribution in tools/gap_check.py (--seen-ceiling).

Synthetic inputs only, no data needed:
    .venv/bin/python -m unittest tests.test_seen_attribution -v        # or: make test
"""

import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import decoder, touch  # noqa: E402
from tests.test_decoder import synthetic_evidence, synthetic_priors  # noqa: E402
from tools import gap_check  # noqa: E402

FPS = 25


def fake_method(settings):
    """What evidence_llr reads from a registered method: its settings and half_inputs."""
    def half_inputs(inp, start, stop):
        def cut(values):
            return None if values is None else values[start:stop]
        return (cut(inp["nearest"]), cut(inp["seen"]), cut(inp["speed"]), inp["priors"],
                cut(inp["holder"]), None, None, cut(inp["frame"]), None)
    return SimpleNamespace(settings=settings, half_inputs=half_inputs)


def synthetic_inputs(n=1003, seed=0):
    rng = np.random.RandomState(seed)
    truth = np.repeat(rng.randint(0, 3, n // 100 + 1), 100)[:n].astype(np.int8)
    nearest, seen, speed = synthetic_evidence(truth, rng)
    holder = np.where(rng.rand(n) < 0.5, truth, -1).astype(np.int8)
    holder[truth == 2] = -1
    frame = rng.normal(0, 2, n)
    priors = dict(synthetic_priors(), holder_agree=0.8)
    inp = {"nearest": nearest, "seen": seen, "speed": speed, "priors": priors,
           "holder": holder, "frame": frame, "spans": [(0, 500), (500, n)]}
    settings = dict(decoder.DEFAULTS, touch_w=1.0, frame_w=1.0, frame_where="seen")
    return inp, settings


class EvidenceTest(unittest.TestCase):
    def test_sources_add_up_to_the_decoders_difference(self):
        inp, settings = synthetic_inputs()
        llr = gap_check.evidence_llr(fake_method(settings), SimpleNamespace(n_frames=1003), inp)
        total = sum(llr.values())
        for start, stop in inp["spans"]:
            e = decoder.block_emissions(inp["nearest"][start:stop], inp["seen"][start:stop],
                                        inp["speed"][start:stop], inp["priors"], settings,
                                        inp["holder"][start:stop], None, None,
                                        inp["frame"][start:stop])
            index = np.arange(stop - start) // decoder.BLOCK
            per_block = np.bincount(index, weights=total[start:stop])
            np.testing.assert_allclose(per_block, e[:, 0] - e[:, 1], atol=1e-9)
        self.assertFalse(np.any(llr["spread"]))           # no team cue in these settings

    def test_frame_source_is_the_mean_log_odds_on_seen_frames(self):
        inp, settings = synthetic_inputs()
        llr = gap_check.evidence_llr(fake_method(settings), SimpleNamespace(n_frames=1003), inp)
        start, stop = inp["spans"][0]
        block = slice(start, start + decoder.BLOCK)
        use = inp["seen"][block]
        expected = inp["frame"][block][use].sum() / decoder.BLOCK
        self.assertAlmostEqual(llr["frame"][block].sum(), expected, places=9)

    def test_holder_off_means_no_holder_source(self):
        inp, settings = synthetic_inputs()
        llr = gap_check.evidence_llr(fake_method(dict(settings, touch_w=0.0)),
                                     SimpleNamespace(n_frames=1003), inp)
        self.assertFalse(np.any(llr["holder"]))


class HolderStartTest(unittest.TestCase):
    def touches(self, touch_frame, touch_team, control):
        n = 400
        control = np.array(control, dtype=np.int64).reshape(-1, 3)
        usable = np.ones(n, dtype=bool)
        holder = touch.sweep_holder(n, np.array(touch_frame), np.array(touch_team, dtype=np.int8),
                                    control, usable, [(0, n)], FPS)
        return touch.Touches(np.array(touch_frame, dtype=np.int64),
                             np.array(touch_team, dtype=np.int8), control, holder, usable)

    def test_a_touch_by_the_other_team_is_a_hand_over(self):
        t = self.touches([100, 110], [0, 1], [])
        self.assertEqual(gap_check.holder_start(t, 112, 200, 1, FPS), "touch hand-over")

    def test_a_control_episode(self):
        t = self.touches([100], [0], [(140, 160, 1)])
        self.assertEqual(gap_check.holder_start(t, 145, 200, 1, FPS), "control")

    def test_a_touch_after_the_hold_expired(self):
        t = self.touches([10, 250], [0, 1], [])          # team 0's hold expires after 4 s
        self.assertEqual(gap_check.holder_start(t, 250, 300, 1, FPS), "touch from none")

    def test_no_holder(self):
        t = self.touches([100], [0], [])
        self.assertEqual(gap_check.holder_start(t, 300, 350, 1, FPS), "none")

    def test_held_before(self):
        t = self.touches([100], [1], [])
        self.assertEqual(gap_check.holder_start(t, 150, 180, 1, FPS), "held before")


if __name__ == "__main__":
    unittest.main()


class DeadModelFeatureTest(unittest.TestCase):
    """solution/deadmodel.py (visible-ball Step 3): team-symmetric, players only."""

    def test_unchanged_by_a_team_swap(self):
        from solution import deadmodel
        from tests.test_framemodel import moving_match, team_swapped
        match = moving_match(600)
        X, names = deadmodel.features(match)
        Y, _ = deadmodel.features(team_swapped(match))
        self.assertEqual(X.shape[1], len(names))
        np.testing.assert_array_equal(X, Y)

    def test_never_reads_the_ball(self):
        from solution import deadmodel
        from tests.test_framemodel import moving_match
        match = moving_match(600)
        other = match.copy()
        other.ball_x = np.full(match.n_frames, 50.0, dtype=np.float32)
        other.ball_y = np.full(match.n_frames, 30.0, dtype=np.float32)
        np.testing.assert_array_equal(deadmodel.features(match)[0], deadmodel.features(other)[0])

    def test_v2_reads_no_absolute_positions(self):
        from solution import deadmodel
        from tests.test_framemodel import moving_match
        match = moving_match(600)
        X, names = deadmodel.features(match)
        self.assertEqual(names, deadmodel.FEATURE_NAMES)
        np.testing.assert_array_equal(deadmodel.select(X, "v1"), X)
        kept = [names[i] for i in deadmodel.columns("v2")]
        self.assertEqual(len(kept), len(names) - 6)
        for name in ("gyration", "tilt_abs", "tracked", "touchline_min", "goalline_min",
                     "in_box"):
            self.assertNotIn(name, kept)
        for name in kept:   # what is left: ranks, ratios and the beyond-line windows
            self.assertTrue(name.startswith(("speed_rank", "rank_", "speed_cv", "cv_", "beyond_"))
                            or name.endswith("_rank"), name)
        # Moving every player 3 m along x (no one crosses a line) leaves v2's columns alone.
        shifted = match.copy()
        shifted.player_x = shifted.player_x + np.float32(3.0)
        self.assertTrue(np.nanmin(match.player_x) > 1.0 and np.nanmax(shifted.player_x) < 104.0)
        np.testing.assert_allclose(deadmodel.select(deadmodel.features(shifted)[0], "v2"),
                                   deadmodel.select(X, "v2"), rtol=1e-5, atol=1e-5)
