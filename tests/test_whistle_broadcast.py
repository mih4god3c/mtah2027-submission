"""Tests for the two-feed whistle pairing (solution/whistle.py, D44) on synthetic audio.

The situation the pairing is for: the tactical feed hears the referee and crowd whistling but no
commentary; the broadcast hears the referee and a close-miked commentator, whose harmonics reach
into the whistle band, and no crowd whistling. Synthetic inputs only, no data needed:
    .venv/bin/python -m unittest tests.test_whistle_broadcast -v       # or: make test
"""

import functools
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from solution import whistle  # noqa: E402
from test_whistle import blast, crowd  # noqa: E402

FPS = 25
RATE = whistle.RATE
SECONDS = 120.0
REFEREE_HZ = 3800.0
# Referee blasts (start s, length s) on the live timeline; 18, so feed_lag has its pairs. The
# broadcast starts 12 s into live time (BROADCAST_OFFSET), so the first is at 13 s.
REFEREE = [(13.0 + 6.0 * k, 0.25 + 0.1 * (k % 4)) for k in range(18)]
# Crowd whistling (tactical only): pitch 2.3-2.9 kHz, between the referee's blasts.
CROWD = [(16.0 + 6.0 * k, 0.4 + 0.1 * (k % 3), 2300.0 + 35.0 * k) for k in range(18)]
SOUND_DELAY = 5          # frames: the tactical microphone hears everything 0.2 s late
BROADCAST_OFFSET = 300   # tactical frame = broadcast frame + 300
GAP = (45 * FPS, 46 * FPS)  # broadcast frames with no alignment (live 57-58 s: no blast)
# The broadcast voice of the pairing tests: a single feed under it misses blasts. Louder or
# brighter voices hide the blasts from the broadcast entirely (TestLimits).
VOICE = {"f0": (110.0, 220.0), "level": 0.15, "tilt_db_per_octave": -9.0}
HARSH_VOICE = {"f0": (180.0, 320.0), "level": 0.3, "tilt_db_per_octave": -9.0}


def commentator(rng, seconds=SECONDS, rate=RATE, f0=(180.0, 320.0), level=0.3,
                tilt_db_per_octave=-9.0):
    """Close-miked speech in 8 s phrases with 2 s pauses: gliding, vibrating f0, harmonics to
    6 kHz under a spectral tilt with formant bumps at 2.7 and 3.5 kHz, a 4 Hz syllable envelope,
    and sibilant noise bursts at 4-8 kHz."""
    n = int(seconds * rate)
    t = np.arange(n) / float(rate)
    on = (t % 10.0) < 8.0
    f = f0[0] + (f0[1] - f0[0]) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * t + rng.uniform(0, 6)))
    f = f * (1 + 0.02 * np.sin(2 * np.pi * 5.5 * t))
    phase = 2 * np.pi * np.cumsum(f) / rate
    out = np.zeros(n)
    for k in range(1, 60):
        fk = k * f
        gain = (fk / 200.0) ** (tilt_db_per_octave / 6.02)
        formant = 1 + 2.0 * np.exp(-((fk - 2700) / 300) ** 2) + 1.5 * np.exp(-((fk - 3500) / 300) ** 2)
        out += np.where(fk < 6000, gain * formant, 0.0) * np.sin(k * phase)
    out *= np.clip(np.sin(2 * np.pi * 4.0 * t + rng.uniform(0, 6)), 0, 1) ** 0.5
    hiss = rng.normal(0, 1, n)
    spectrum = np.fft.rfft(hiss)
    freqs = np.fft.rfftfreq(n, 1.0 / rate)
    spectrum[(freqs < 4000) | (freqs > 8000)] = 0
    hiss = np.fft.irfft(spectrum, n)
    hiss /= hiss.std()
    out = out / out[on].std() + 0.4 * hiss * (np.sin(2 * np.pi * 1.3 * t) > 0.93)
    return level * out * on


def tones(items, delay_s=0.0, amplitude=0.04):
    total = np.zeros(int(SECONDS * RATE))
    for item in items:
        start, length = item[0], item[1]
        hz = item[2] if len(item) > 2 else REFEREE_HZ
        total += blast(RATE, start + delay_s, length, SECONDS, amplitude=amplitude, base_hz=hz)
    return total


def features(audio):
    return whistle.frame_features(iter([audio.astype(np.float32)]), int(SECONDS * FPS), FPS)


def annotated(feats):
    return whistle.annotate(whistle.events(feats, FPS, whistle.STRICT),
                            whistle.events(feats, FPS, whistle.LOOSE))


@functools.lru_cache(maxsize=None)
def feeds(seed=0, voice="normal"):
    """(tactical annotated events, broadcast events mapped to the tactical timeline, the count
    map_events dropped). Live time L is tactical frame L and broadcast frame L - BROADCAST_OFFSET;
    the broadcast's frames GAP are unaligned. `voice` is "normal" (VOICE) or "harsh"."""
    rng = np.random.RandomState(seed)
    delay = SOUND_DELAY / float(FPS)
    tactical = crowd(rng, SECONDS) + tones(REFEREE, delay) + tones(CROWD, delay, amplitude=0.05)
    talk = commentator(rng, **(VOICE if voice == "normal" else HARSH_VOICE))
    broadcast = crowd(rng, SECONDS) * 0.5 + talk + tones(REFEREE)
    tac = annotated(features(tactical))
    cut = BROADCAST_OFFSET * (RATE // FPS)       # the broadcast file starts 12 s into live time
    bro = annotated(features(np.concatenate((broadcast[cut:], np.zeros(cut)))))
    live_t = np.arange(int(SECONDS * FPS)) + BROADCAST_OFFSET
    live_t[GAP[0]:GAP[1]] = -1
    mapped, dropped = whistle.map_events(bro, live_t)
    return tac, mapped, dropped


def referee_frames():
    """Tactical frames where the tactical microphone hears each referee blast start."""
    return [int(round(s * FPS)) + SOUND_DELAY for s, _ in REFEREE]


def at_referee(frame, tol=3, delay=SOUND_DELAY):
    return any(abs(frame - (f - SOUND_DELAY + delay)) <= tol for f in referee_frames())


def both(seed=0, voice="normal", shift=0):
    tac, mapped, _ = feeds(seed, voice)
    lag, _ = whistle.feed_lag(tac, mapped, FPS)
    return whistle.pair_feeds(tac, mapped, FPS, lag, shift=shift)


SHIFTS_S = (-20, -10, 10, 20)


class TestSingleFeedLimits(unittest.TestCase):
    """Why one feed is not enough, on the synthetic scenes."""

    def test_tactical_alone_fires_on_crowd_whistling(self):
        tac, _, _ = feeds()
        crowd_hits = [e for e in tac if e["strict"] and e["hz"] < 3000.0]
        self.assertGreaterEqual(len(crowd_hits), len(CROWD) // 2)

    def test_broadcast_alone_under_the_harsh_voice_misses_blasts_and_fires_on_it(self):
        _, mapped, _ = feeds(0, "harsh")
        strict = [e for e in mapped if e["strict"]]
        found = sum(any(abs(e["start"] - (f - SOUND_DELAY)) <= 3 for e in strict)
                    for f in referee_frames())
        false = sum(not at_referee(e["start"], tol=8, delay=0) for e in strict)
        self.assertLess(found, len(REFEREE) // 2)
        self.assertGreater(false, 0)


class TestPairing(unittest.TestCase):
    def test_both_feeds_keep_the_referee_and_drop_crowd_and_voice(self):
        for seed in range(3):
            pairs = both(seed)
            hits = [p for p in pairs if at_referee(p["start"])]
            self.assertGreaterEqual(len(hits), len(REFEREE) - 2, seed)
            self.assertLessEqual(len(pairs) - len(hits), 1, (seed, pairs))
            for p in pairs:
                self.assertGreater(p["hz"], 3400.0)

    def test_lag_recovers_the_sound_delay(self):
        tac, mapped, dropped = feeds()
        lag, pairs = whistle.feed_lag(tac, mapped, FPS)
        self.assertGreaterEqual(len(pairs), whistle.MIN_LAG_PAIRS)
        self.assertLessEqual(abs(lag - SOUND_DELAY), 1)
        self.assertEqual(dropped, 0)          # GAP holds no broadcast event

    def test_shifted_pairing_is_chance_only(self):
        for shift_s in SHIFTS_S:
            self.assertLessEqual(len(both(shift=int(shift_s * FPS))), 1, shift_s)

    def test_under_the_harsh_voice_the_chance_rate_flags_the_pairs(self):
        """A known limit: a loud, bright voice whose formant sits on the crowd's pitch pairs with
        the crowd's whistling. The chance pairing at shifted lags then runs at a sizeable share of
        the real one, which is what tools/whistle_broadcast_check.py reports beside every rate."""
        pairs = both(0, "harsh")
        null = np.mean([len(both(0, "harsh", int(s * FPS))) for s in SHIFTS_S])
        self.assertGreater(len(pairs) - sum(at_referee(p["start"]) for p in pairs), 0)
        self.assertGreaterEqual(null, 0.2 * len(pairs))


class TestMapping(unittest.TestCase):
    def test_map_events_offsets_and_drops_unmapped(self):
        live_t = np.arange(100) + 1000
        live_t[10:20] = -1
        events = [{"start": 5, "stop": 9, "hz": 3800.0}, {"start": 12, "stop": 15, "hz": 3800.0},
                  {"start": 18, "stop": 25, "hz": 3800.0}]
        mapped, dropped = whistle.map_events(events, live_t)
        self.assertEqual(dropped, 2)
        self.assertEqual(len(mapped), 1)
        self.assertEqual((mapped[0]["start"], mapped[0]["stop"]), (1005, 1009))
        self.assertEqual((mapped[0]["bstart"], mapped[0]["bstop"]), (5, 9))

    def test_lag_falls_back_to_zero_with_few_pairs(self):
        tac = [{"start": 100, "stop": 110, "seconds": 0.4, "hz": 3800.0, "strict": True}]
        bro = [{"start": 95, "stop": 105, "hz": 3800.0, "strict": True}]
        lag, pairs = whistle.feed_lag(tac, bro, FPS)
        self.assertEqual((lag, pairs), (0, [5]))

    def test_pair_needs_pitch_agreement_and_one_strict_side(self):
        tac = [{"start": 100, "stop": 110, "seconds": 0.4, "hz": 3800.0, "strict": False},
               {"start": 200, "stop": 210, "seconds": 0.4, "hz": 2600.0, "strict": True},
               {"start": 300, "stop": 310, "seconds": 0.4, "hz": 3800.0, "strict": True}]
        bro = [{"start": 100, "hz": 3810.0, "strict": False},
               {"start": 200, "hz": 3800.0, "strict": True},
               {"start": 302, "hz": 3750.0, "strict": False}]
        both = whistle.pair_feeds(tac, bro, FPS)
        self.assertEqual([p["start"] for p in both], [300])
        self.assertEqual(both[0]["offset"], -2)


if __name__ == "__main__":
    unittest.main()
