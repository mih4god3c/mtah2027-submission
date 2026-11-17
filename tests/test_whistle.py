"""Tests for the whistle detector (solution/whistle.py) on synthetic stadium audio.

Synthetic inputs only, no data needed:
    .venv/bin/python -m unittest tests.test_whistle -v       # or: make test
"""

import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import whistle  # noqa: E402

FPS = 25
RATE = whistle.RATE
SECONDS = 30.0
# (start s, length s): a short blast, a long one, and an end-of-half pair 0.25 s apart
BLASTS = [(3.0, 0.3), (10.0, 1.2), (20.0, 0.4), (20.65, 1.5)]


def crowd(rng, seconds=SECONDS, rate=RATE):
    """Pink-ish crowd noise with a slow swell, a 2 s roar at 5 s (+15 dB, broadband) and a
    'commentator' (harmonics of 180 Hz under 1.5 kHz) from 12 s to 18 s."""
    n = int(seconds * rate)
    white = rng.normal(0, 1, n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1.0 / rate)
    spectrum /= np.sqrt(np.maximum(freqs, 20.0))
    noise = np.fft.irfft(spectrum, n)
    noise /= noise.std()
    t = np.arange(n) / float(rate)
    noise *= 1.0 + 0.3 * np.sin(2 * np.pi * 0.1 * t)
    roar = (t >= 5.0) & (t < 7.0)
    noise[roar] *= 10 ** (15 / 20.0)
    voice = (t >= 12.0) & (t < 18.0)
    for k in range(1, 8):
        noise[voice] += 0.8 / k * np.sin(2 * np.pi * 180 * k * t[voice])
    return noise * 0.05


def blast(rate, start, length, total, amplitude=0.08, base_hz=3200.0):
    """A warbling pea-whistle tone with a 2nd harmonic and 20 ms ramps."""
    n = int(total * rate)
    t = np.arange(n) / float(rate)
    on = (t >= start) & (t < start + length)
    hz = base_hz + 120.0 * np.sin(2 * np.pi * 35.0 * t)
    phase = 2 * np.pi * np.cumsum(hz) / rate
    ramp = np.clip(np.minimum(t - start, start + length - t) / 0.02, 0, 1) * on
    return amplitude * ramp * (np.sin(phase) + 0.3 * np.sin(2 * phase))


def scene(seed=0, blasts=BLASTS):
    rng = np.random.RandomState(seed)
    audio = crowd(rng)
    for start, length in blasts:
        audio = audio + blast(RATE, start, length, SECONDS)
    return audio.astype(np.float32)


def detect(audio, setting=whistle.STRICT, blocks=None):
    n_frames = int(SECONDS * FPS)
    parts = [audio] if blocks is None else np.array_split(audio, blocks)
    features = whistle.frame_features(iter(parts), n_frames, FPS)
    return features, whistle.events(features, FPS, setting)


class TestDetector(unittest.TestCase):
    def test_finds_each_blast_on_time(self):
        _, found = detect(scene())
        self.assertEqual(len(found), len(BLASTS))
        for event, (start, length) in zip(found, BLASTS):
            self.assertLessEqual(abs(event["start"] - start * FPS), 2)
            self.assertLessEqual(abs(event["seconds"] - length), 0.12)
            self.assertLess(abs(event["hz"] - 3200.0), 200.0)

    def test_noise_roar_and_voice_alone_give_nothing(self):
        for setting in (whistle.STRICT, whistle.LOOSE):
            for seed in range(3):
                _, found = detect(scene(seed, blasts=[]), setting)
                self.assertEqual(found, [], (setting, seed))

    def test_blast_under_the_roar_is_reported_by_loose_only(self):
        """A known limit: 11 dB under a crowd roar, STRICT misses the blast and LOOSE finds only
        part of it (late). LOOSE exists to show such near-misses to a human."""
        _, strict = detect(scene(blasts=[(5.5, 0.8)]))
        _, loose = detect(scene(blasts=[(5.5, 0.8)]), whistle.LOOSE)
        self.assertEqual(strict, [])
        self.assertEqual(len(loose), 1)
        self.assertGreaterEqual(loose[0]["start"], 5.5 * FPS)
        self.assertLessEqual(loose[0]["stop"], 6.3 * FPS + 1)

    def test_block_sizes_do_not_change_the_features(self):
        audio = scene()
        whole, _ = detect(audio)
        split, _ = detect(audio, blocks=977)
        np.testing.assert_allclose(whole, split, rtol=0, atol=1e-4)

    def test_frames_past_the_audio_are_nan(self):
        audio = scene()[:int(20 * RATE)]
        features = whistle.frame_features(iter([audio]), int(SECONDS * FPS), FPS)
        self.assertTrue(np.isfinite(features[:20 * FPS - 1]).all())
        self.assertTrue(np.isnan(features[20 * FPS + 1:]).all())


class TestDecode(unittest.TestCase):
    def test_aac_file_round_trip(self):
        """Encode the scene as AAC in an .mp4 with PyAV, decode it back, find the blasts."""
        import av
        audio = scene()
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "scene.mp4")
            try:
                with av.open(path, "w") as container:
                    stream = container.add_stream("aac", rate=48000)
                    stream.layout = "mono"
                    up = np.repeat(audio, 3).astype(np.float32)      # 16 kHz -> 48 kHz
                    for i in range(0, len(up), 1024):
                        chunk = up[i:i + 1024].reshape(1, -1)
                        frame = av.AudioFrame.from_ndarray(chunk, format="flt", layout="mono")
                        frame.sample_rate = 48000
                        for packet in stream.encode(frame):
                            container.mux(packet)
                    for packet in stream.encode(None):
                        container.mux(packet)
            except (av.error.FFmpegError, ValueError) as error:
                self.skipTest("no AAC encoder: {}".format(error))
            features = whistle.frame_features(whistle.audio_blocks(path), int(SECONDS * FPS),
                                              FPS)
        found = whistle.events(features, FPS)
        self.assertEqual(len(found), len(BLASTS))
        for event, (start, _) in zip(found, BLASTS):
            self.assertLessEqual(abs(event["start"] - start * FPS), 2)


if __name__ == "__main__":
    unittest.main()
