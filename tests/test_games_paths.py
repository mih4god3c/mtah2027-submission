"""games.asset_paths falls back to a single copy-suffixed file ("-001") when the documented one
is missing. No data needed: .venv/bin/python -m unittest tests.test_games_paths -v"""

import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import games  # noqa: E402


def touch(folder, *names):
    os.makedirs(folder, exist_ok=True)
    for name in names:
        open(os.path.join(folder, name), "w").close()


class TestCopySuffix(unittest.TestCase):
    def paths(self, source, *names):
        root = tempfile.mkdtemp()
        touch(os.path.join(root, source), *names)
        return {k: os.path.basename(v)
                for k, v in games.asset_paths(data_dir=root, source=source).items()}

    def test_single_copy_is_used(self):
        got = self.paths("tactical", "BAR-ATH_tactical-001.mp4", "BAR-ATH_tactical_atd.xml")
        self.assertEqual(got["video"], "BAR-ATH_tactical-001.mp4")
        self.assertEqual(got["atd_meta"], "BAR-ATH_tactical_atd.xml")

    def test_documented_name_wins(self):
        got = self.paths("broadcast", "BAR-ATH_broadcast.mp4", "BAR-ATH_broadcast-001.mp4")
        self.assertEqual(got["video"], "BAR-ATH_broadcast.mp4")

    def test_two_copies_are_ambiguous(self):
        got = self.paths("broadcast", "BAR-ATH_broadcast-001.mp4", "BAR-ATH_broadcast (2).mp4")
        self.assertEqual(got["video"], "BAR-ATH_broadcast.mp4")

    def test_missing_folder_keeps_the_documented_path(self):
        got = games.asset_paths(data_dir=os.path.join(tempfile.mkdtemp(), "nope"))
        self.assertTrue(got["video"].endswith("BAR-ATH_tactical.mp4"))


if __name__ == "__main__":
    unittest.main()
