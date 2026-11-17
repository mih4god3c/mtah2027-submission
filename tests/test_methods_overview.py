"""Tests for tools/methods_overview.py. No data needed.

    .venv/bin/python -m unittest tests.test_methods_overview -v       # or: make test
"""

import contextlib
import io
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution.methods import METHODS  # noqa: E402
from tools import methods_overview as mo  # noqa: E402


class Describe(unittest.TestCase):

    def test_the_submission_lists_its_parts(self):
        row = mo.describe(METHODS["v6_deadball_model"])
        self.assertEqual(row["touch"], "w1")
        self.assertEqual(row["kickoff"], "yes")
        self.assertEqual(row["spread"], "w0.5 all max")
        self.assertEqual(row["frame"], "w1 seen")
        self.assertEqual(row["deadm"], "w1 all")
        self.assertEqual((row["cue"], row["tv"], row["ball"], row["control"], row["priors"],
                          row["costs"]), ("", "", "", "", "", ""))

    def test_other_parts_show_where_set(self):
        self.assertEqual(mo.describe(METHODS["touch_markov_hold"])["touch"], "w1 hold")
        self.assertEqual(mo.describe(METHODS["touch_deadball"])["cue"], "w0.5 player+ball")
        self.assertEqual(mo.describe(METHODS["touch_c3_seen_w1_tv_cal_fused"])["tv"],
                         "w1 calibrated")
        self.assertEqual(mo.describe(METHODS["touch_c3_seen_w1_tv_cal_fused"])["ball"], "merged")
        self.assertEqual(mo.describe(METHODS["touch_gapteam_k2"])["priors"], "keep2")
        self.assertEqual(mo.describe(METHODS["c3s_dead4_sw2"])["costs"], "switch 2, dead 4")
        self.assertEqual(mo.describe(METHODS["c3s_dmin2"])["costs"], "min stop 2 s")
        self.assertEqual(mo.describe(METHODS["v1_decoder"])["touch"], "")

    def test_former_names_are_aliases_and_listed_once(self):
        from solution.methods import ALIASES, canonical
        for old, new in ALIASES.items():
            self.assertIs(METHODS[old], METHODS[new])
            self.assertEqual(METHODS[new].__name__, new)
        self.assertEqual(canonical("touch_c3_seen_w1_dm"), "v6_deadball_model")
        self.assertEqual(canonical("v7_keep2_control"), "v7_keep2_control")
        lines = mo.table(["v6_deadball_model"])
        self.assertIn("(was `touch_c3_seen_w1_dm`)", lines[2])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            mo.main(["--no-legend"])
        rows = [l for l in out.getvalue().splitlines() if l.startswith("| `")]
        names = [l.split("`")[1] for l in rows]
        self.assertEqual(len(names), len(set(names)))
        self.assertNotIn("dm_k2ctl", names)
        self.assertIn("v7_keep2_control", names)
        self.assertIsNone(mo.describe(METHODS["baseline_smooth50"]))

    def test_table_has_a_row_per_method(self):
        lines = mo.table(list(METHODS))
        self.assertEqual(len(lines), len(METHODS) + 2)
        self.assertIn("majority vote over +-50 frames", lines[2 + list(METHODS).index(
            "baseline_smooth50")])


if __name__ == "__main__":
    unittest.main()
