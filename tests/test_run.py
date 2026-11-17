"""Tests for solution/run.py's path to a submission. Synthetic inputs only, no data needed.

    .venv/bin/python -m unittest tests.test_run -v       # or: make test
"""

import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from solution import deadmodel, decoder, framemodel, run  # noqa: E402
from solution.match import Match, pm  # noqa: E402
from solution.sources import idsse  # noqa: E402
from test_decoder import synthetic_priors  # noqa: E402


def bar_ath_like(seed=0):
    """A synthetic match shaped like the ATD: ungraded frames before, between and after halves."""
    rng = np.random.RandomState(seed)
    n, halves = 9000, [(400, 4300), (4800, 8700)]
    slots = 46
    x = (rng.rand(n, slots) * 105).astype(np.float32)
    y = (rng.rand(n, slots) * 68).astype(np.float32)
    x[rng.rand(n, slots) < 0.5] = np.nan
    ball_x = (rng.rand(n) * 105).astype(np.float32)
    ball_x[rng.rand(n) < 0.4] = np.nan
    return Match("BAR-ATH", 25, (105.0, 68.0), x, y,
                 np.array([0] * 23 + [1] * 23, dtype=np.int8), ball_x,
                 (rng.rand(n) * 68).astype(np.float32), halves, source="test")


def tiny_model():
    from test_framemodel import moving_match
    labels = np.full(1500, pm.H, dtype=np.int8)
    labels[500:1000] = pm.A
    return framemodel.fit([moving_match(seed=1, labels=labels)],
                          dict(framemodel.PARAMS, max_iter=10, min_samples_leaf=20, step=2))


def tiny_dead_model():
    from test_framemodel import moving_match
    labels = np.full(1500, pm.H, dtype=np.int8)
    labels[500:800] = pm.D
    labels[800:] = pm.A
    return deadmodel.fit([moving_match(seed=2, labels=labels)],
                         dict(deadmodel.PARAMS, max_iter=10, min_samples_leaf=20, step=2))


TINY_MODEL = tiny_model()
TINY_DEAD_MODEL = tiny_dead_model()


class Submission(unittest.TestCase):

    def test_one_valid_letter_per_frame_with_x_exactly_outside_the_halves(self):
        match = bar_ath_like()
        with mock.patch.object(decoder, "idsse_priors", lambda *args: dict(
                synthetic_priors(), speed_edges=decoder.RANK_EDGES,
                speed_log=np.log(np.full((2, len(decoder.RANK_EDGES)),
                                         1.0 / len(decoder.RANK_EDGES))))):
            # The frame model (C3) and the dead-ball model (D37) would train on the external
            # matches; give them small synthetic ones so the test needs no data.
            with mock.patch.object(framemodel, "idsse_model", lambda *args: TINY_MODEL), \
                    mock.patch.object(deadmodel, "idsse_model", lambda *args: TINY_DEAD_MODEL):
                labels = run.predict(match)
                again = run.predict(match)
        self.assertEqual(labels.shape, (match.n_frames,))
        np.testing.assert_array_equal(labels, again)
        graded = match.in_play_mask()
        self.assertTrue(np.all(labels[~graded] == "X"))
        self.assertTrue(np.all(np.isin(labels[graded], ["H", "A", "D"])))

        # The written file passes the official format check and reads back unchanged.
        with tempfile.TemporaryDirectory() as folder:
            path = run.baseline.write_submission(os.path.join(folder, "t.csv"), labels)
            frame = pm.read_label_csv(path)
            self.assertIsNone(pm.validate_submission(frame, match.n_frames))
            np.testing.assert_array_equal(pm.label_codes(frame), pm.label_codes(labels))

    def test_method_switch_dispatches_and_the_default_is_the_submission(self):
        from solution import methods
        match = bar_ath_like()
        calls = []

        def fake(m):
            calls.append(m.name)
            codes = np.full(m.n_frames, pm.D, dtype=np.int8)
            codes[~m.in_play_mask()] = pm.X
            return codes
        with mock.patch.dict(methods.METHODS, {"fake_method": fake}):
            labels = run.predict(match, method="fake_method")
        self.assertEqual(calls, ["BAR-ATH"])
        graded = match.in_play_mask()
        self.assertTrue(np.all(labels[graded] == "D"))
        self.assertTrue(np.all(labels[~graded] == "X"))
        self.assertEqual(run.METHOD, "v7_keep2_control")                  # D39
        self.assertIn(run.METHOD, methods.METHODS)
        with self.assertRaises(ValueError):
            run.predict(match, method="no_such_method")

    def test_alignment_is_required_only_by_the_broadcast_methods(self):
        self.assertFalse(run.needs_alignment("v5_frame_model"))
        self.assertFalse(run.needs_alignment("v6_deadball_model"))
        self.assertFalse(run.needs_alignment("v7_keep2_control"))          # the submission, D39
        self.assertTrue(run.needs_alignment("v6_plus_ball"))            # merge flag, no _fused suffix
        self.assertTrue(run.needs_alignment("v6_plus_stack"))
        self.assertTrue(run.needs_alignment("touch_c3_seen_w1_tv_cal"))
        self.assertTrue(run.needs_alignment("touch_c3_seen_w1_fused"))
        self.assertTrue(run.needs_alignment("touch_c3_seen_w1_tv_cal_fused"))
        with mock.patch("solution.sources.fused.ALIGN_PATH", "/nonexistent/align.npz"):
            run.require_alignment("v5_frame_model")        # no cache needed, no error
            with self.assertRaises(SystemExit) as stop:
                run.require_alignment("touch_c3_seen_w1_tv_cal_fused")
        self.assertIn("make broadcast-align", str(stop.exception))

    def test_missing_external_data_stops_with_a_clear_message(self):
        with tempfile.TemporaryDirectory() as empty, \
                mock.patch.object(idsse, "cache_path", lambda name: os.path.join(empty, name)):
            with self.assertRaises(SystemExit) as stop:
                run.require_external(data_dir=empty)
        self.assertIn("make external", str(stop.exception))
        for match_id in idsse.DEV_IDS:
            self.assertIn(match_id, str(stop.exception))
        for match_id in idsse.HOLDOUT_IDS:
            self.assertNotIn(match_id, str(stop.exception))


if __name__ == "__main__":
    unittest.main()
