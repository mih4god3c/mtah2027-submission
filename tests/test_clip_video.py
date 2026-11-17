"""Tests for tools/clip_video.py on a synthetic video. No data needed.

    .venv/bin/python -m unittest tests.test_clip_video -v       # or: make test
"""

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from fractions import Fraction
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools import clip_video as cv  # noqa: E402

FPS = 25
W, H = 64, 48
N = 120          # 4.8 s: several GOPs of 12 frames, so seeking lands on an inner keyframe
RATE = 48000


def luma(i):
    """A distinct flat grey level per frame, so a decoded frame names its own index."""
    return 20 + (i * 7) % 200


def make_video(path, n=N, audio=True, noise=None):
    """n flat frames at 25 fps, lossless H.264 with B-frames and short GOPs, plus silent AAC.

    With `noise` (a Generator) the frames carry random texture instead, which no encoder can
    compress away; that is what a size cap has to work against.
    """
    import av
    with av.open(path, "w") as out:
        vs = out.add_stream("libx264", rate=FPS,
                            options={"crf": "0", "g": "12", "bf": "2", "preset": "ultrafast"})
        vs.width, vs.height, vs.pix_fmt = W, H, "yuv420p"
        aud = None
        if audio:
            aud = out.add_stream("aac", rate=RATE)
            aud.codec_context.format = "fltp"
            aud.codec_context.layout = "mono"
        for i in range(n):
            y = np.full((H, W), luma(i), dtype=np.uint8)
            if noise is not None:
                y = noise.integers(0, 256, (H, W), dtype=np.uint8)
            planes = np.concatenate([y, np.full((H // 2, W), 128, dtype=np.uint8)])
            frame = av.VideoFrame.from_ndarray(planes, format="yuv420p")
            frame.pts = i
            frame.time_base = Fraction(1, FPS)
            for pkt in vs.encode(frame):
                out.mux(pkt)
            if aud is not None:
                samples = RATE // FPS
                aframe = av.AudioFrame.from_ndarray(np.zeros((1, samples), dtype=np.float32),
                                                    format="fltp", layout="mono")
                aframe.sample_rate = RATE
                aframe.pts = i * samples
                aframe.time_base = Fraction(1, RATE)
                for pkt in aud.encode(aframe):
                    out.mux(pkt)
        for pkt in vs.encode(None):
            out.mux(pkt)
        if aud is not None:
            for pkt in aud.encode(None):
                out.mux(pkt)


def decode_luma(path):
    """Mean luma of every frame of a file, in order."""
    import av
    with av.open(path) as inp:
        return [float(f.to_ndarray(format="yuv420p")[:H].mean()) for f in inp.decode(video=0)]


def audio_seconds(path):
    import av
    with av.open(path) as inp:
        if not inp.streams.audio:
            return None
        aud = inp.streams.audio[0]
        return sum(f.samples for f in inp.decode(aud)) / float(aud.rate)


class Windows(unittest.TestCase):

    def test_parse_time_and_clock(self):
        self.assertEqual(cv.parse_time("12:27.4"), 18685)
        self.assertEqual(cv.parse_time("00:01.2"), 30)
        self.assertEqual(cv.parse_time("18685"), 18685)
        self.assertEqual(cv.clock(18685), "12:27.4")
        self.assertEqual(cv.clock(26185), "17:27.4")

    def test_public_window_and_the_one_after(self):
        first, last = cv.public_window()
        self.assertEqual((first, last), (11185, 18684))
        self.assertEqual(cv.next_window(), (18685, 26184))
        self.assertEqual(26184 - 18685 + 1, last - first + 1)


class Clip(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.src = os.path.join(cls.tmp.name, "src.mp4")
        make_video(cls.src)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def out(self, name):
        return os.path.join(self.tmp.name, name)

    def check_frames(self, path, first, last, tolerance=3.0):
        got = decode_luma(path)
        self.assertEqual(len(got), last - first + 1)
        for k, value in enumerate(got):
            self.assertAlmostEqual(value, luma(first + k), delta=tolerance,
                                   msg="clip frame {} is not source frame {}".format(k, first + k))

    def test_frame_accurate_with_seek(self):
        first, last = 30, 77  # starts mid-GOP, ends mid-GOP
        n = cv.clip(self.src, first, last, self.out("seek.mp4"))
        self.assertEqual(n, 48)
        self.check_frames(self.out("seek.mp4"), first, last)
        self.assertAlmostEqual(audio_seconds(self.out("seek.mp4")), 48 / 25.0, delta=0.1)

    def test_same_without_seek(self):
        first, last = 30, 77
        cv.clip(self.src, first, last, self.out("noseek.mp4"), seek=False)
        self.check_frames(self.out("noseek.mp4"), first, last)

    def test_edges_and_no_audio(self):
        cv.clip(self.src, 0, 5, self.out("head.mp4"), audio=False)
        self.check_frames(self.out("head.mp4"), 0, 5)
        self.assertIsNone(audio_seconds(self.out("head.mp4")))
        cv.clip(self.src, N - 7, N - 1, self.out("tail.mp4"))
        self.check_frames(self.out("tail.mp4"), N - 7, N - 1)

    def test_stamp_marks_the_corner_only(self):
        import av
        first, last = 40, 44
        cv.clip(self.src, first, last, self.out("stamp.mp4"), stamp=True, crf=0)
        with av.open(self.out("stamp.mp4")) as inp:
            frames = [f.to_ndarray(format="yuv420p")[:H].astype(float) for f in inp.decode(video=0)]
        self.assertEqual(len(frames), 5)
        for k, y in enumerate(frames):
            corner = y[:H // 2, :W // 2]
            rest = y[H // 2:, W // 2:]
            self.assertGreater(float(np.abs(corner - luma(first + k)).max()), 40,
                               "the stamp is not drawn")
            self.assertLess(float(np.abs(rest - luma(first + k)).max()), 4,
                             "the stamp spilled over the frame")

    def test_bitrate_for(self):
        self.assertEqual(cv.bitrate_for(480, 7500, 25), int(480e6 * 8 * cv.MARGIN / 300))
        self.assertEqual(cv.bitrate_for(480, 7500, 25, audio_bps=128000),
                         int(480e6 * 8 * cv.MARGIN / 300 - 128000))
        self.assertLess(cv.MARGIN, 1 / 1.12)
        self.assertEqual(cv.bitrate_for(0.001, 7500, 25), 200_000)

    def test_max_mb_shrinks_the_file(self):
        src = self.out("noise_src.mp4")
        make_video(src, n=60, audio=False, noise=np.random.default_rng(0))
        cv.clip(src, 6, 53, self.out("noise_free.mp4"))
        free = os.path.getsize(self.out("noise_free.mp4"))
        cap_mb = free / 2 / 1e6
        cv.clip(src, 6, 53, self.out("noise_capped.mp4"), max_mb=cap_mb)
        capped = os.path.getsize(self.out("noise_capped.mp4"))
        self.assertLess(capped, 0.8 * free)
        self.assertEqual(len(decode_luma(self.out("noise_capped.mp4"))), 48)

    def test_bad_windows(self):
        with self.assertRaises(ValueError):
            cv.clip(self.src, 10, 9, self.out("bad.mp4"))
        with self.assertRaises(ValueError):
            cv.clip(self.src, 10, N + 5, self.out("bad.mp4"))
        with self.assertRaises(ValueError):
            cv.clip(self.src, -1, 5, self.out("bad.mp4"))

    def test_cli(self):
        out = self.out("cli.mp4")
        with mock.patch.object(cv, "require", lambda names, **kw: [self.src]):
            text = io.StringIO()
            with redirect_stdout(text):
                code = cv.main(["--from", "00:01", "--to", "00:01.8", "--out", out, "--no-audio"])
        self.assertEqual(code, 0)
        self.check_frames(out, 25, 45)
        report = text.getvalue()
        self.assertIn("frames 25..45 (21 frames", report)
        self.assertIn("clip frame k = video frame 25 + k = ATD frame_id 26 + k", report)

    def test_cli_default_window_is_after_the_public_truth(self):
        seen = {}

        def fake_clip(src, first, last, out, **kw):
            seen.update(first=first, last=last, out=out)
            open(out, "wb").close()
            return last - first + 1

        with mock.patch.object(cv, "require", lambda names, **kw: [self.src]), \
                mock.patch.object(cv, "clip", fake_clip), \
                mock.patch.object(cv, "CLIP_DIR", self.tmp.name), redirect_stdout(io.StringIO()):
            cv.main([])
        self.assertEqual((seen["first"], seen["last"]), (18685, 26184))
        self.assertEqual(seen["out"], os.path.join(self.tmp.name, "src_f18685-26184.mp4"))


if __name__ == "__main__":
    unittest.main()
