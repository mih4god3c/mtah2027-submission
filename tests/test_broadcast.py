"""Tests for solution/broadcast.py, the broadcast -> tactical alignment. No data.

    .venv/bin/python -m unittest tests.test_broadcast -v

A synthetic tactical match, and a broadcast cut from it the way a TV feed is: live stretches at
one offset, a cut-away, a replay of an earlier moment, a skip in time to a new offset. The camera
shows only the players near the ball, with a little position noise, and its own slot numbering.
"""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import broadcast as bc  # noqa: E402
from solution.match import Match  # noqa: E402

PITCH = (105.0, 68.0)
N_TAC = 8000
N_SLOTS = 22
# (kind, broadcast start, stop, offset): tactical frame = broadcast frame + offset
SEGMENTS = [("live", 0, 1500, 300), ("cut", 1500, 1700, 300), ("replay", 1700, 1950, -500),
            ("cut", 1950, 2050, 300), ("live", 2050, 3500, 300), ("cut", 3500, 3600, None),
            ("live", 3600, 5600, 2000)]
N_BRO = 5600


def wave(rng, n, centre, amp, periods):
    t = np.arange(n) / 25.0
    out = np.full(n, float(centre))
    for a, p in zip(amp, periods):
        out += a * np.sin(2 * np.pi * t / p + rng.rand() * 2 * np.pi)
    return out


def tactical(seed=0):
    rng = np.random.RandomState(seed)
    px = np.empty((N_TAC, N_SLOTS))
    py = np.empty((N_TAC, N_SLOTS))
    for s in range(N_SLOTS):
        px[:, s] = wave(rng, N_TAC, rng.uniform(15, 90), rng.uniform(3, 8, 3), rng.uniform(4, 40, 3))
        py[:, s] = wave(rng, N_TAC, rng.uniform(10, 58), rng.uniform(3, 8, 3), rng.uniform(4, 40, 3))
    bx = wave(rng, N_TAC, 52, rng.uniform(10, 18, 3), rng.uniform(3, 20, 3))
    by = wave(rng, N_TAC, 34, rng.uniform(6, 10, 3), rng.uniform(3, 20, 3))
    bx, by = np.clip(bx, 0, 105), np.clip(by, 0, 68)
    true_ball = (bx.copy(), by.copy())
    hide = wave(rng, N_TAC, 0, [1.0], [7.0]) < -0.3                  # ~40% gone, in runs
    bx[hide] = by[hide] = np.nan
    team = np.array([0] * 11 + [1] * 11, dtype=np.int8)
    match = Match("tac", 25, PITCH, np.clip(px, 0, 105).astype(np.float32),
                  np.clip(py, 0, 68).astype(np.float32), team, bx.astype(np.float32),
                  by.astype(np.float32), [(0, N_TAC)])
    match.notes["true_ball"] = true_ball
    return match


def broadcast(tac, seed=1, view_m=25.0, noise=0.3):
    """(broadcast match, valid) cut from `tac` by SEGMENTS."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N_SLOTS)                                   # broadcast slot -> tactical
    px = np.full((N_BRO, N_SLOTS), np.nan)
    py = np.full((N_BRO, N_SLOTS), np.nan)
    bx = np.full(N_BRO, np.nan)
    by = np.full(N_BRO, np.nan)
    valid = np.zeros(N_BRO, dtype=bool)
    # the camera follows the true ball, which it also sees inside the tactical ball's gaps
    centre_x, centre_y = tac.notes["true_ball"]
    bro_ball_hide = rng.rand(N_BRO) < 0.2
    for kind, a, b, d in SEGMENTS:
        if kind == "cut":
            continue
        valid[a:b] = True
        for f in range(a, b):
            t = f + d
            cx, cy = centre_x[t], centre_y[t]
            for s in range(N_SLOTS):
                x, y = tac.player_x[t, perm[s]], tac.player_y[t, perm[s]]
                if np.hypot(x - cx, y - cy) <= view_m:
                    px[f, s] = x + rng.randn() * noise
                    py[f, s] = y + rng.randn() * noise
            if not bro_ball_hide[f]:
                bx[f] = cx + rng.randn() * noise
                by[f] = cy + rng.randn() * noise
    team = tac.team[perm]
    return Match("bro", 25, PITCH, px.astype(np.float32), py.astype(np.float32), team,
                 bx.astype(np.float32), by.astype(np.float32), [(0, N_BRO)]), valid


class Fixture:
    tac = bro = valid = out = None

    @classmethod
    def get(cls):
        if cls.out is None:
            cls.tac = tactical()
            cls.bro, cls.valid = broadcast(cls.tac)
            cls.out = bc.align(cls.tac, cls.bro, cls.valid)
        return cls


class Runs(unittest.TestCase):

    def test_runs_and_min_len(self):
        mask = np.array([1, 1, 0, 1, 1, 1, 0, 0, 1], dtype=bool)
        self.assertEqual(bc.runs(mask), [(0, 2), (3, 6), (8, 9)])
        self.assertEqual(bc.runs(mask, 3), [(3, 6)])
        self.assertEqual(bc.runs(np.zeros(4, dtype=bool)), [])


class Alignment(unittest.TestCase):

    def test_true_offset_is_cheapest(self):
        f = Fixture.get()
        frames = bc.cost_frames(f.bro, 100, 1400, 24)
        curve = bc.cost_curve(f.tac, f.bro, frames, np.arange(250, 351))
        self.assertEqual(int(np.nanargmin(curve)) + 250, 300)
        self.assertLess(np.nanmin(curve), 1.0)

    def test_live_stretches_found_at_their_offsets(self):
        f = Fixture.get()
        by_start = {r["start"]: (r, k) for r, k in zip(f.out["results"], f.out["kinds"])}
        for start, d in ((0, 300), (2050, 300), (3600, 2000)):
            r, kind = by_start[start]
            self.assertTrue(r["confident"])
            self.assertEqual(kind, "live")
            self.assertLessEqual(abs(r["d"] - d), 1)
            if r["drift"] is not None:
                self.assertLessEqual(abs(r["drift"]), 1)

    def test_replay_is_found_and_classified(self):
        f = Fixture.get()
        by_start = {r["start"]: (r, k) for r, k in zip(f.out["results"], f.out["kinds"])}
        r, kind = by_start[1700]
        self.assertTrue(r["confident"])
        self.assertEqual(kind, "replay")
        self.assertLessEqual(abs(r["d"] + 500), 1)

    def test_live_map_fills_cut_aways_but_not_skips(self):
        f = Fixture.get()
        live_t = f.out["live_t"]
        for b in (10, 1600, 1800, 2000, 3000, 4000):
            want = b + (2000 if b >= 3600 else 300)
            self.assertLessEqual(abs(live_t[b] - want), 1, b)
        self.assertTrue(np.all(live_t[3500:3600] == -1))            # offsets disagree: a skip

    def test_tactical_states(self):
        f = Fixture.get()
        state, bro_frame = f.out["state"], f.out["bro_frame"]
        self.assertEqual(state[300 + 700], bc.LIVE)
        self.assertEqual(state[300 + 1600], bc.CUT)
        self.assertEqual(state[300 + 1800], bc.REPLAY)
        self.assertEqual(state[100], bc.NOT_COVERED)                 # before the broadcast
        self.assertLessEqual(abs(bro_frame[300 + 700] - 700), 1)
        replay = f.out["replay_frame"]
        self.assertLessEqual(abs(replay[1700 - 500 + 100] - 1800), 1)

    def test_stretch_without_ball_aligns_from_its_neighbours(self):
        f = Fixture.get()
        bro = f.bro.copy()
        bro.ball_x[2050:3500] = np.nan
        bro.ball_y[2050:3500] = np.nan
        results = bc.align_stretches(f.tac, bro, f.valid)
        r = next(r for r in results if r["start"] == 2050)
        self.assertEqual(r["votes"], 0)
        self.assertTrue(r["confident"])
        self.assertEqual(r["pass"], 2)
        self.assertLessEqual(abs(r["d"] - 300), 1)

    def test_broadcast_ball_fills_tactical_gaps(self):
        f = Fixture.get()
        live = f.out["state"] == bc.LIVE
        t = np.flatnonzero(live & ~f.tac.ball_seen)
        seen = f.bro.ball_seen[f.out["bro_frame"][t]]
        self.assertGreater(seen.mean(), 0.7)                         # 80% minus edge frames

    def test_team_codes_agree(self):
        f = Fixture.get()
        share, pairs = bc.team_agreement(f.tac, f.bro, f.out["results"], f.out["chain"])
        self.assertGreater(pairs, 100)
        self.assertGreater(share, 0.95)


class FuseBall(unittest.TestCase):

    def test_fills_only_tactical_gaps_and_never_overwrites(self):
        f = Fixture.get()
        fused, stats = bc.fuse_ball(f.tac, f.bro, f.out["state"], f.out["bro_frame"])
        seen = f.tac.ball_seen
        self.assertTrue(np.array_equal(fused.ball_x[seen], f.tac.ball_x[seen]))
        filled = fused.ball_seen & ~seen
        self.assertEqual(int(filled.sum()), stats["filled"])
        self.assertGreater(stats["filled"], 0)
        self.assertTrue(np.all(f.out["state"][filled] == bc.LIVE))
        true_x, true_y = f.tac.notes["true_ball"]
        err = np.hypot(fused.ball_x[filled] - true_x[filled], fused.ball_y[filled] - true_y[filled])
        self.assertLess(np.median(err), 1.0)
        for a, b in bc.runs(filled):
            self.assertGreaterEqual(b - a, bc.FILL_MIN_RUN)
        self.assertIsNot(fused.ball_x, f.tac.ball_x)

    def test_speed_gate_rejects_a_far_ball(self):
        f = Fixture.get()
        bro = f.bro.copy()
        bro.ball_x[:] = np.where(np.isfinite(bro.ball_x), 0.0, np.nan)   # all at the goal line
        bro.ball_y[:] = np.where(np.isfinite(bro.ball_y), 0.0, np.nan)
        _, stats = bc.fuse_ball(f.tac, bro, f.out["state"], f.out["bro_frame"])
        self.assertGreater(stats["speed_rejected"], 0.5 * stats["candidates"])


class FillGaps(unittest.TestCase):

    def test_restores_part_of_gaps_from_clean(self):
        from tools import broadcast_fill_check as fc
        from tests.test_proxy_league import synthetic_match
        clean = synthetic_match(n_frames=6000, seed=3)
        degraded = clean.copy()
        hide = np.zeros(6000, dtype=bool)
        for a, b in ((100, 150), (400, 1000), (3100, 3300)):     # 2 s, 24 s, 8 s
            hide[a:b] = True
        degraded.ball_x[hide] = np.nan
        degraded.ball_y[hide] = np.nan
        out = fc.fill_gaps(degraded, clean, seed=0)
        back = out.ball_seen & hide
        self.assertTrue(np.array_equal(out.ball_x[back], clean.ball_x[back]))
        self.assertTrue(np.array_equal(out.ball_seen[~hide], degraded.ball_seen[~hide]))
        self.assertEqual(int(back[400:1000].sum()), int(round(0.078 * 600)))
        self.assertEqual(int(back[3100:3300].sum()), int(round(0.322 * 200)))
        for a, b in ((100, 150), (400, 1000), (3100, 3300)):        # one chunk at an edge
            got = bc.runs(back[a:b])
            self.assertLessEqual(len(got), 1)
            if got:
                self.assertTrue(got[0][0] == 0 or got[0][1] == b - a)
        again = fc.fill_gaps(degraded, clean, seed=0)
        self.assertTrue(np.array_equal(again.ball_seen, out.ball_seen))


class Merged(unittest.TestCase):

    def test_refuses_what_it_cannot_imitate(self):
        from solution.sources import fused
        from tests.test_proxy_league import synthetic_match
        m = synthetic_match(n_frames=6000, seed=3)
        with self.assertRaises(ValueError):
            fused.merged(m)                                    # not degraded, not IDSSE
        m.name = "idsse_X"
        m.notes["degraded"] = {"version": "v1", "seed": 0}
        m.notes["tempo"] = 0.9
        with self.assertRaises(ValueError):
            fused.merged(m)                                    # timeline moved


class Orientation(unittest.TestCase):

    def test_mirrored_feed_is_unmirrored(self):
        f = Fixture.get()
        flip, rows = bc.choose_flip(f.tac, bc.flipped(f.bro, "xy"), f.valid, n=3)
        self.assertEqual(flip, "xy")
        self.assertEqual([row["flip"] for row in rows], list(bc.FLIPS))

    def test_unmirrored_feed_stays(self):
        f = Fixture.get()
        flip, _ = bc.choose_flip(f.tac, f.bro, f.valid, n=3)
        self.assertEqual(flip, "none")


class ReportRuns(unittest.TestCase):
    """tools/broadcast_align.py's sections on the synthetic pair: they run and print."""

    def test_sections(self):
        import contextlib
        import io
        import tempfile
        from unittest import mock
        from tools import broadcast_align as tool
        f = Fixture.get()
        graded = f.tac.in_play_mask()
        buffer = io.StringIO()
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(tool, "OUT_PATH", os.path.join(folder, "x", "align.npz")), \
                mock.patch.object(tool.load_smart_tagging, "load", return_value=[
                    {"code": "SET PIECES", "start_frame": 2400, "end_frame": 2600, "tags": {"Type": "Throw-in"}}]), \
                contextlib.redirect_stdout(buffer):
            flip = tool.section_orientation(f.tac, f.bro, f.valid)
            tool.section_alignment(f.tac, f.bro, f.valid, f.out)
            tool.section_coverage(f.tac, f.out, graded)
            tool.section_ball(f.tac, f.bro, f.out, graded)
            tool.section_cutaway(f.tac, f.out, graded, "none", None)
            codes = np.full(N_TAC, 0, dtype=np.int8)                # H everywhere ...
            codes[300 + 1500:300 + 2050] = 2                             # ... D over the cut
            tool.section_cutaway(f.tac, f.out, graded, "m", codes)
            tool.section_timing(f.tac, f.out, graded, codes)
            with mock.patch.object(tool, "predict", return_value=codes):
                tool.section_merge(f.tac, f.bro, f.out, graded, "m", codes)
            tool.save(f.out, flip)
            self.assertTrue(os.path.isfile(tool.OUT_PATH))
        text = buffer.getvalue()
        self.assertIn("using: none", text)
        self.assertIn("live segments: 2", text)
        self.assertIn("off-pitch lift", text)
        self.assertIn("around SET PIECES", text)
        self.assertIn("predicted D runs", text)
        self.assertIn("ball merge", text)


if __name__ == "__main__":
    unittest.main()
