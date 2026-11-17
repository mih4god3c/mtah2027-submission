"""Tests for tools/moment_check.py on a synthetic match. No data needed.

    .venv/bin/python -m unittest tests.test_moment_check -v       # or: make test
"""

import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from solution import deadmodel, decoder, framemodel  # noqa: E402
from solution.methods import METHODS  # noqa: E402
from test_decoder import synthetic_priors  # noqa: E402
from test_run import TINY_DEAD_MODEL, TINY_MODEL, bar_ath_like  # noqa: E402
from tools import moment_check as mc  # noqa: E402


def patched():
    priors = dict(synthetic_priors(), speed_edges=decoder.RANK_EDGES,
                  speed_log=np.log(np.full((2, len(decoder.RANK_EDGES)),
                                           1.0 / len(decoder.RANK_EDGES))))
    return (mock.patch.object(decoder, "idsse_priors", lambda *args: priors),
            mock.patch.object(framemodel, "idsse_model", lambda *args: TINY_MODEL),
            mock.patch.object(deadmodel, "idsse_model", lambda *args: TINY_DEAD_MODEL))


class ParseTime(unittest.TestCase):

    def test_clock_and_frames(self):
        self.assertEqual(mc.parse_time("11:59"), 11 * 60 * 25 + 59 * 25)
        self.assertEqual(mc.parse_time("00:01.2"), 30)
        self.assertEqual(mc.parse_time("17980"), 17980)
        self.assertEqual(mc.clock(17980), "11:59.2")


class Inspect(unittest.TestCase):

    def test_rows_cover_the_window_and_the_parts_add_up(self):
        match = bar_ath_like()
        method = METHODS["v6_deadball_model"]
        gt = np.full(match.n_frames, 3, dtype=np.int8)
        gt[1000:1500] = 2
        a, b, c = patched()
        with a, b, c:
            rows = mc.inspect(match, method, 1000, 1125, gt=gt, alignment=None)
            out = io.StringIO()
            with redirect_stdout(out):
                mc.print_tables(rows, "v6_deadball_model")
        self.assertEqual([r["frame"] for r in rows], [1000, 1025, 1050, 1075, 1100])
        self.assertTrue(all(r["gt"] == "D" for r in rows))
        for r in rows:
            self.assertIn(r["decoded"], "HAD")
            self.assertTrue(0.0 <= r["seen"] <= 1.0)
            self.assertAlmostEqual(r["near_h"] + r["near_a"], r["seen"], places=6)
            # The term parts are differences against the full margin, so each term's column
            # exists exactly for the weights the method sets.
            for key, name in mc.TERMS:
                self.assertEqual("team_" + name in r, method.settings.get(key, 0.0) > 0)
        text = out.getvalue()
        self.assertIn("1. Inputs per second", text)
        self.assertIn("3. Dead-ball evidence", text)
        self.assertIn("deadm", text)

    def test_other_methods_and_alignment_columns(self):
        match = bar_ath_like()
        method = METHODS["v5_frame_model"]
        bro = np.arange(match.n_frames) + 100
        state = np.full(match.n_frames, 1, dtype=np.int8)
        state[1050:1100] = 2
        a, b, c = patched()
        with a, b, c:
            rows = mc.inspect(match, method, 1000, 1100, gt=None, alignment=(bro, state),
                              others=["baseline_smooth50"])
        self.assertEqual([r["tv_state"] for r in rows], ["live", "live", "cut", "cut"])
        self.assertEqual(rows[0]["broadcast"], mc.clock(1100))
        self.assertTrue(all(r["other_baseline_smooth50"] in "HAD" for r in rows))
        self.assertTrue(all(r["gt"] == "?" for r in rows))

    def test_window_outside_the_halves_is_refused(self):
        match = bar_ath_like()
        a, b, c = patched()
        with a, b, c:
            with self.assertRaises(SystemExit):
                mc.inspect(match, METHODS["v5_frame_model"], 10, 50)


if __name__ == "__main__":
    unittest.main()
