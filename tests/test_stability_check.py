"""Tests for tools/stability_check.py on a synthetic match. No data needed.

    .venv/bin/python -m unittest tests.test_stability_check -v       # or: make test
"""

import math
import os
import sys
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from solution import decoder, framemodel, methods  # noqa: E402
from solution.match import pm  # noqa: E402
from test_decoder import synthetic_priors  # noqa: E402
from test_run import TINY_MODEL, bar_ath_like  # noqa: E402
from tools import stability_check as sc  # noqa: E402


def fake_priors(*args):
    return dict(synthetic_priors(), speed_edges=decoder.RANK_EDGES,
                speed_log=np.log(np.full((2, len(decoder.RANK_EDGES)),
                                         1.0 / len(decoder.RANK_EDGES))))


class Stability(unittest.TestCase):

    def test_perturbed_priors_move_only_the_asked_median(self):
        priors = synthetic_priors()
        out = sc.perturbed_priors(priors, "dead", 1.1)
        self.assertAlmostEqual(out["durations"]["dead"]["mu"],
                               priors["durations"]["dead"]["mu"] + math.log(1.1))
        self.assertEqual(out["durations"]["possession"], priors["durations"]["possession"])
        self.assertEqual(priors["durations"]["dead"]["mu"], synthetic_priors()["durations"]["dead"]["mu"])

    def test_counts_and_verdict(self):
        codes = np.array([pm.X, pm.H, pm.H, pm.D, pm.D, pm.A, pm.H, pm.X], dtype=np.int8)
        c = sc.counts(codes, [(1, 7)], 25)
        self.assertEqual((c["poss"], c["stops"]), (3, 1))
        self.assertAlmostEqual(c["d_share"], 2 / 6.0)
        self.assertAlmostEqual(c["home"], 3 / 4.0)
        table = [{"name": "reference", "poss": 100}, {"name": "w x0.8", "poss": 112},
                 {"name": "stoppage length x1.1", "poss": 160}]
        worst, label = sc.verdict(table)
        self.assertAlmostEqual(worst, 0.12)          # the prior row does not count
        self.assertTrue(label.startswith("watch"))

    def test_variants_decode_and_the_reference_matches_the_method(self):
        match = bar_ath_like()
        few = [{"name": "w x1.2", "decoder": {"w": 1.2}},
               {"name": "stoppage length x0.9", "prior": ("dead", 0.9)},
               {"name": "touch_m x1.2", "touch": {"touch_m": 1.2}}]
        with mock.patch.object(decoder, "idsse_priors", fake_priors), \
                mock.patch.object(framemodel, "idsse_model", lambda *a: TINY_MODEL), \
                mock.patch.object(methods.touch, "idsse_holder_agree", lambda *a: 0.8), \
                mock.patch.object(methods.gapteam, "idsse_log",
                                  lambda *a: [np.log(np.full((2, 15), 1.0 / 15))]):
            predict = methods.METHODS["v5_frame_model"]
            variants = sc.decode_variants(predict, match, few)
            np.testing.assert_array_equal(variants["reference"], predict(match))
        self.assertEqual(set(variants), {"reference"} | {p["name"] for p in few})
        graded = match.in_play_mask()
        for codes in variants.values():
            self.assertTrue(np.all(codes[~graded] == pm.X))
            self.assertTrue(np.all(np.isin(codes[graded], [pm.H, pm.A, pm.D])))


if __name__ == "__main__":
    unittest.main()
