"""Referee-whistle detector on a video's audio track (E10 revisited for the end of a half, Step 0).

Why: BAR-ATH's tracking blocks, which are the graded halves, run on past the half-time and
full-time whistles while the players leave or linger, and the reference decodes possessions
there (tools/halfend_check.py). Players-only cues cannot place that end safely: simulator v1's
dropout episodes make equally deep collapses at the end of plain IDSSE halves. The whistle is the
event that ends a half. IDSSE has no audio, so a whistle cue cannot fire on the proxy league;
it can only be checked on BAR-ATH.

What a referee's pea whistle looks like: a strong, narrow tone, fundamental about 2 to 4.5 kHz
(warbling by tens of Hz), lasting from about 0.15 s (a short blast) to well over a second (the
end of a half is usually two or three blasts, the last one long). Crowd noise and commentary
are broadband or sit lower. So, per video frame (hop = 1 / fps, so feature row f is video frame
f), on a Hann window of WINDOW samples centred on the frame, the band's power spectrum smoothed
over SMOOTH_BINS and taken in dB with a least-squares line removed (crowd noise tilts):
  - tonality: the strongest bin in BAND_HZ above the band's median, in dB;
  - peak: the strongest bin's power in dB, and its frequency;
  - rise: peak minus its rolling median over BACKGROUND_S, in dB (loud against the moment).
A frame is whistle-like when tonality >= tonal_db and rise >= rise_db. Runs of such frames,
bridged over gaps up to gap_s and lasting at least min_s, are events. Two settings are fixed
here before any BAR-ATH run: STRICT (what a rule would use) and LOOSE (near-misses, reported so
a human can see what STRICT leaves out). They are not to be tuned on BAR-ATH. Known limit
(tests/test_whistle.py): a blast about 11 dB under a crowd roar is missed by STRICT and found
only in part, late, by LOOSE.

Audio is decoded with PyAV (pinned in requirements-lock.txt), mixed to mono and resampled to
RATE, and aligned so that sample 0 is the video's first frame. Everything is deterministic:
no randomness, a fixed decoder.
"""

import os

import numpy as np

RATE = 16000                 # Hz after resampling; whistles sit well under its 8 kHz Nyquist
WINDOW = 1024                # samples per spectrum (64 ms)
BAND_HZ = (2000.0, 5000.0)   # where a pea whistle's fundamental sits
SMOOTH_BINS = 5              # spectrum smoothing across frequency (5 x 15.6 Hz)
BACKGROUND_S = 10.0          # rolling-median window for the rise
BLOCK_FRAMES = 4096          # frames per FFT batch (memory and speed; fixed, so deterministic)
STRICT = {"tonal_db": 15.0, "rise_db": 8.0, "min_s": 0.15, "gap_s": 0.12}
LOOSE = {"tonal_db": 10.0, "rise_db": 4.0, "min_s": 0.10, "gap_s": 0.12}
FEATURE_VERSION = 1          # bump when the features change; it keys the cache


def _hop(rate, fps):
    if rate % fps:
        raise ValueError("rate {} is not a multiple of fps {}".format(rate, fps))
    return rate // fps


def frame_features(blocks, n_frames, fps, rate=RATE):
    """Per video frame: (tonality dB, peak dB, peak Hz), float32, from mono sample blocks.

    `blocks` yields float arrays whose concatenation is the audio at `rate`, sample 0 at video
    frame 0. Frames the audio does not reach are NaN.
    """
    hop = _hop(rate, fps)
    half = WINDOW // 2
    taper = np.hanning(WINDOW).astype(np.float64)
    freqs = np.fft.rfftfreq(WINDOW, 1.0 / rate)
    band = np.flatnonzero((freqs >= BAND_HZ[0]) & (freqs <= BAND_HZ[1]))
    # Least-squares line through the band's dB spectrum: (2, bins) projection, so the tilt of
    # crowd noise (pink-ish: louder at the low end) does not read as a tone.
    design = np.stack((np.ones(len(band)), (freqs[band] - freqs[band].mean()) / 1000.0), axis=1)
    project = np.linalg.pinv(design)
    smooth = np.ones(SMOOTH_BINS) / SMOOTH_BINS
    out = np.full((n_frames, 3), np.nan, dtype=np.float32)
    # The buffer starts at sample `origin`; frame f is centred on sample f * hop.
    buffer, origin, done = np.zeros(half, dtype=np.float64), -half, 0

    def emit(buffer, origin, done, final=False):
        available = origin + len(buffer)               # one past the last sample held
        last = (available - half) // hop               # last frame with a full window
        if final:
            last = min(n_frames - 1, (available - 1) // hop)
            pad = max(0, last * hop + half - available)
            buffer = np.concatenate((buffer, np.zeros(pad)))
        last = min(last, n_frames - 1)
        while done <= last:
            stop = min(last + 1, done + BLOCK_FRAMES)
            starts = np.arange(done, stop) * hop - half - origin
            windows = buffer[starts[:, None] + np.arange(WINDOW)[None, :]] * taper
            power = np.abs(np.fft.rfft(windows, axis=1)[:, band]) ** 2
            # Smoothing across a few bins: a warbling whistle spreads over them anyway, while
            # it shrinks the spread of noise bins, whose max over the band would read as a tone.
            power = np.apply_along_axis(np.convolve, 1, power, smooth, mode="same") + 1e-12
            level = 10.0 * np.log10(power)
            residual = level - (level @ project.T) @ design.T
            best = residual.argmax(axis=1)
            rows = np.arange(len(best))
            out[done:stop, 0] = residual[rows, best] - np.median(residual, axis=1)
            out[done:stop, 1] = level[rows, best]
            out[done:stop, 2] = freqs[band][best]
            done = stop
        keep_from = done * hop - half - origin             # first sample the next frame needs
        if keep_from > 0:
            buffer, origin = buffer[keep_from:], origin + keep_from
        return buffer, origin, done

    pending, pending_n = [], 0
    for block in blocks:
        block = np.asarray(block, dtype=np.float64).reshape(-1)
        pending.append(block)
        pending_n += len(block)
        if pending_n >= BLOCK_FRAMES * hop:
            buffer = np.concatenate([buffer] + pending)
            pending, pending_n = [], 0
            buffer, origin, done = emit(buffer, origin, done)
            if done >= n_frames:
                return out
    buffer = np.concatenate([buffer] + pending)
    if done < n_frames and origin + len(buffer) > done * hop:
        emit(buffer, origin, done, final=True)
    return out


def rise(peak_db, fps):
    """Peak dB minus its rolling median over BACKGROUND_S (NaN-safe: NaN stays NaN)."""
    from scipy.ndimage import median_filter
    filled = np.where(np.isfinite(peak_db), peak_db, np.nanmedian(peak_db))
    size = int(round(BACKGROUND_S * fps)) | 1
    return peak_db - median_filter(filled.astype(np.float64), size=size, mode="nearest")


def events(features, fps, setting=STRICT):
    """Whistle events as a list of dicts: start / stop video frame (stop exclusive), seconds,
    median peak Hz, max tonality dB, max rise dB."""
    tonal, peak, hz = features[:, 0], features[:, 1], features[:, 2]
    up = rise(peak, fps)
    with np.errstate(invalid="ignore"):
        hit = (tonal >= setting["tonal_db"]) & (up >= setting["rise_db"])
    edges = np.diff(np.concatenate(([0], hit.astype(np.int8), [0])))
    starts, stops = list(np.flatnonzero(edges == 1)), list(np.flatnonzero(edges == -1))
    merged = []
    for a, b in zip(starts, stops):
        if merged and a - merged[-1][1] <= setting["gap_s"] * fps:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    out = []
    for a, b in merged:
        if b - a < setting["min_s"] * fps:
            continue
        inside = hit[a:b]
        out.append({"start": int(a), "stop": int(b), "seconds": (b - a) / float(fps),
                    "hz": float(np.median(hz[a:b][inside])),
                    "tonal_db": float(np.nanmax(tonal[a:b])),
                    "rise_db": float(np.nanmax(up[a:b]))})
    return out


def audio_blocks(path, rate=RATE):
    """Mono float32 blocks of a file's first audio track at `rate`, sample 0 at the video's
    first frame (leading audio before it is dropped, a late start is padded with silence)."""
    import av
    with av.open(path) as container:
        audio = container.streams.audio[0]
        video = container.streams.video[0] if container.streams.video else None
        v0 = (float(video.start_time * video.time_base)
              if video is not None and video.start_time is not None else 0.0)
        resampler = av.AudioResampler(format="flt", layout="mono", rate=rate)
        drop = None                  # samples still to drop: audio that precedes the video

        def trimmed(outs):
            nonlocal drop
            for out in outs:
                samples = out.to_ndarray().reshape(-1).astype(np.float32)
                cut = min(drop, len(samples))
                drop -= cut
                if len(samples) > cut:
                    yield samples[cut:]

        for frame in container.decode(audio):
            if drop is None:
                first = frame.time if frame.time is not None else v0
                lead = int(round((first - v0) * rate))   # audio starts this late (> 0)
                if lead > 0:
                    yield np.zeros(lead, dtype=np.float32)
                drop = max(0, -lead)
            yield from trimmed(resampler.resample(frame))
        if drop is not None:
            yield from trimmed(resampler.resample(None))


def video_features(path, n_frames, fps, cache_dir=None):
    """frame_features() of a video's audio, cached as .npz (derived from match data: the cache
    folder is gitignored). The cache key is the file's size and mtime and FEATURE_VERSION."""
    if cache_dir is None:
        from solution.match import CACHE_DIR
        cache_dir = os.path.join(CACHE_DIR, "whistle")
    stat = os.stat(path)
    key = "{}_{}_{}_v{}".format(os.path.splitext(os.path.basename(path))[0], stat.st_size,
                                int(stat.st_mtime), FEATURE_VERSION)
    cache = os.path.join(cache_dir, key + ".npz")
    if os.path.isfile(cache):
        return np.load(cache)["features"]
    features = frame_features(audio_blocks(path), n_frames, fps)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(cache, features=features)
    return features


# --- Two feeds (D44): a whistle heard in both the tactical and the broadcast audio. -----------
#
# The two tracks fail in opposite ways. The tactical camera's microphone hears the referee and
# the crowd (crowd whistling at 2.3-2.9 kHz made STRICT fire at ~10x a referee's rate, D33), but
# no commentary. The broadcast mix hears the referee and the commentators, whose harmonics reach
# into the 2-5 kHz band (tests/test_whistle_broadcast.py: under loud bright speech a single feed
# misses blasts and fires on the voice). A
# tone heard in both at the same moment and the same pitch is neither crowd-only nor
# commentary-only. Fixed here before any BAR-ATH run, not to be tuned on BAR-ATH:
LAG_SEARCH_S = 1.0     # pairs used to estimate the feeds' audio lag are at most this far apart
MIN_LAG_PAIRS = 10     # fewer STRICT-STRICT pairs than this: no lag estimate, 0 is used
PAIR_TOL_S = 0.24      # after the lag, onsets this close pair (alignment wander +-3 frames, +-3 onset)
PAIR_HZ = 250.0        # ... and the pitches agree this well (a +-120 Hz warble moves each median)


def annotate(strict, loose):
    """LOOSE events, each with `strict` (a STRICT event overlaps it); STRICT events no LOOSE
    event covers are added (not expected, never dropped). Sorted by start."""
    rows = []
    for event in loose:
        hit = any(s["start"] < event["stop"] and event["start"] < s["stop"] for s in strict)
        rows.append(dict(event, strict=hit))
    for s in strict:
        if not any(r["start"] < s["stop"] and s["start"] < r["stop"] for r in rows):
            rows.append(dict(s, strict=True))
    return sorted(rows, key=lambda r: r["start"])


def map_events(events, live_t):
    """Events on the broadcast timeline -> the tactical one, through broadcast.align's live_t
    (tactical frame of the live moment each broadcast frame airs during, -1 if unknown). The
    broadcast's own frames are kept as `bstart` / `bstop` (to find the moment in that video).
    Events whose first frame is unmapped are dropped. Returns (mapped, dropped count)."""
    live_t = np.asarray(live_t)
    out, dropped = [], 0
    for event in events:
        a, b = event["start"], event["stop"]
        if a >= len(live_t) or live_t[a] < 0:
            dropped += 1
            continue
        start = int(live_t[a])
        last = live_t[min(b, len(live_t)) - 1]
        stop = int(last) + 1 if last >= 0 and last >= start else start + (b - a)
        out.append(dict(event, start=start, stop=stop, bstart=int(a), bstop=int(b)))
    return out, dropped


def feed_lag(tactical, broadcast, fps):
    """Audio lag of the tactical feed behind the broadcast one, in frames (tactical onset minus
    mapped broadcast onset): the median offset of pair_feeds() at lag 0 with the onset tolerance
    widened to LAG_SEARCH_S. Sound takes ~0.15-0.3 s to reach a camera on the stand, so a
    positive lag is expected. With fewer than MIN_LAG_PAIRS pairs the lag is 0.
    Returns (lag, pair offsets)."""
    lags = [p["offset"] for p in pair_feeds(tactical, broadcast, fps, tol_s=LAG_SEARCH_S)]
    lag = int(round(float(np.median(lags)))) if len(lags) >= MIN_LAG_PAIRS else 0
    return lag, lags


def pair_feeds(tactical, broadcast, fps, lag=0, shift=0, tol_s=PAIR_TOL_S):
    """Whistles heard in both feeds. `tactical` and `broadcast` are annotate()d events (the
    broadcast ones already map_events()ed); a pair needs onsets within PAIR_TOL_S after `lag`,
    pitches within PAIR_HZ and STRICT on at least one side. Greedy one-to-one, closest onsets
    first. `shift` moves the broadcast events by that many frames on top of the lag: a pairing
    at a shift of several seconds can only be chance, which measures the chance rate. `tol_s`
    is widened only by feed_lag().

    Each pair: start / stop / seconds / hz of the tactical event (the submission timeline), plus
    `bstart` (broadcast frame), `bhz`, `offset` (tactical minus shifted broadcast onset, frames)
    and which sides are STRICT."""
    tol = tol_s * fps
    candidates = []
    starts = np.array([e["start"] for e in tactical])
    for j, b in enumerate(broadcast):
        at = b["start"] + lag + shift
        lo, hi = np.searchsorted(starts, at - tol), np.searchsorted(starts, at + tol, "right")
        for i in range(lo, hi):
            t = tactical[i]
            if abs(t["hz"] - b["hz"]) > PAIR_HZ or not (t.get("strict") or b.get("strict")):
                continue
            candidates.append((abs(t["start"] - at), i, j))
    used_t, used_b, pairs = set(), set(), []
    for _, i, j in sorted(candidates):
        if i in used_t or j in used_b:
            continue
        used_t.add(i)
        used_b.add(j)
        t, b = tactical[i], broadcast[j]
        pairs.append({"start": t["start"], "stop": t["stop"], "seconds": t["seconds"],
                      "hz": t["hz"], "bstart": b.get("bstart", b["start"]), "bhz": b["hz"],
                      "offset": t["start"] - (b["start"] + shift),
                      "tactical_strict": bool(t.get("strict")),
                      "broadcast_strict": bool(b.get("strict"))})
    return sorted(pairs, key=lambda p: p["start"])


# --- The self-calibrating referee-whistle detector (D47, 2026-11-16). ---------------------------
#
# Nothing in this section comes from BAR-ATH: no frame, no time and no pitch of the match is
# written here, and no person picks or judges a moment of it. RULES.md allows "pseudo-labels
# produced by an algorithm, provided no person decided any of them", and that is how the
# whistle is found:
#
# 1. Features, generic (peak_features): per video frame, the PK_PEAKS strongest tonal peaks in
#    PK_BAND_HZ (D33's band for a pea whistle's fundamental, fixed 2026-11-06), each with its
#    pitch and its local tonality (dB above the median of the PK_LOCAL_HZ of spectrum around it).
# 2. Anchors, from the tracking (kickoff_anchors): every kickoff solution/kickoff.py's Law 8
#    cues find anywhere in a half -- both teams in their own halves, quiet, then play speeds up.
#    That covers the start of each half and the restart after each goal. Law 8: the referee
#    signals the kickoff, so a whistle sounds shortly before each kick.
# 3. Calibration (calibrate): the pitch signature (a pair of tones sounding together, else one
#    tone, in CAL_BIN_HZ cells) present in the CAL_BEFORE_S / CAL_AFTER_S window of the most
#    anchors, and at least CAL_MIN_RATIO times as often there as in the same windows shifted by
#    CAL_SHIFTS_S (its chance rate). Fans whistle everywhere and slide in pitch; the referee's
#    whistle comes back at the same pitch before every kickoff. No signature in CAL_MIN_ANCHORS
#    anchors: no calibration, and no whistles (the cue falls silent; nothing is guessed).
# 4. Detection (whistle_events): runs of frames with a tone in the calibrated band(s), merged over
#    EV_GAP_S and at least EV_MIN_S long; a whistle when its tones are on together on at least
#    EV_MIN_HITS frames and each pitch is steady (std at most EV_STEADY_HZ).
#
# Settings chosen from D33's pre-match values, the laws of the game and synthetic audio only
# (tests/test_whistle_selfcal.py). Fixed before the first run on the match; not to be tuned on
# what a run prints (D47).
PK_BAND_HZ = BAND_HZ               # (2000, 5000) Hz, D33
PK_LOCAL_HZ = 800.0                # tonality reference: median of this much spectrum around a bin
PK_SPAN_HZ = (PK_BAND_HZ[0] - PK_LOCAL_HZ / 2, PK_BAND_HZ[1] + PK_LOCAL_HZ / 2)
PK_PEAKS = 4                       # strongest tonal peaks kept per frame
PK_ON_DB = 10.0                    # a peak this far above its surroundings is a tone
PK_VERSION = 1                     # bump when the features change; it keys the cache
CAL_BEFORE_S = 10.0                # anchor window: the kickoff whistle precedes the kick ...
CAL_AFTER_S = 2.0                  # ... and the kick is placed from the players, give or take
CAL_BIN_HZ = 100.0                 # pitch cells for the signature
CAL_STEADY_S = 0.3                 # a tone counts for calibration when it is still there ...
CAL_STEADY_HZ = 60.0               # ... within this pitch this long before or after (a glide of
                                   # 300 Hz/s moves 90 Hz in 0.3 s, a whistle's warble under 40)
CAL_MIN_FRAMES = 4                 # a signature is present in a window on this many frames
CAL_MIN_ANCHORS = 2                # ... at no fewer than this many anchors
CAL_MIN_RATIO = 3.0                # ... and this many times its rate in the shifted windows
CAL_SHIFTS_S = (-120.0, -90.0, -60.0, -30.0, 30.0, 60.0, 90.0, 120.0)   # chance windows
BAND_HALF_HZ = 150.0               # a calibrated band: the signature's pitch +- this
EV_GAP_S = 0.12                    # D33's gap_s
EV_MIN_S = 0.15                    # D33 STRICT's min_s
EV_MIN_HITS = 2                    # frames with every tone on together
EV_STEADY_HZ = 50.0                # pitch std: a warble of +-20 Hz is 14; a glide across a
                                   # band (2 x BAND_HALF_HZ) is 87


def peak_features(blocks, n_frames, fps, rate=RATE):
    """Per video frame the PK_PEAKS strongest tonal peaks in PK_BAND_HZ: (T, PK_PEAKS, 2) float32
    of (Hz, tonality dB), strongest first; NaN where fewer peaks or past the audio."""
    from scipy.ndimage import median_filter
    hop = _hop(rate, fps)
    half = WINDOW // 2
    taper = np.hanning(WINDOW)
    freqs = np.fft.rfftfreq(WINDOW, 1.0 / rate)
    span = np.flatnonzero((freqs >= PK_SPAN_HZ[0]) & (freqs <= PK_SPAN_HZ[1]))
    f = freqs[span]
    band = np.flatnonzero((f >= PK_BAND_HZ[0]) & (f <= PK_BAND_HZ[1]))
    f_band = f[band]
    width = int(round(PK_LOCAL_HZ / (f[1] - f[0]))) | 1
    smooth = np.ones(3) / 3.0
    out = np.full((n_frames, PK_PEAKS, 2), np.nan, dtype=np.float32)
    buffer, origin, done = np.zeros(half), -half, 0

    def emit(buffer, origin, done, final):
        available = origin + len(buffer)
        last = (available - half) // hop
        if final:
            last = (available - 1) // hop
            buffer = np.concatenate((buffer, np.zeros(max(0, last * hop + half - available))))
        last = min(last, n_frames - 1)
        while done <= last:
            stop = min(last + 1, done + BLOCK_FRAMES)
            starts = np.arange(done, stop) * hop - half - origin
            windows = buffer[starts[:, None] + np.arange(WINDOW)[None, :]] * taper
            power = np.abs(np.fft.rfft(windows, axis=1)[:, span]) ** 2
            power = np.apply_along_axis(np.convolve, 1, power, smooth, mode="same") + 1e-14
            level = 10.0 * np.log10(power)
            tonal = (level - median_filter(level, size=(1, width), mode="nearest"))[:, band]
            peak = np.zeros(tonal.shape, dtype=bool)
            peak[:, 1:-1] = (tonal[:, 1:-1] > tonal[:, :-2]) & (tonal[:, 1:-1] >= tonal[:, 2:])
            score = np.where(peak, tonal, -np.inf)
            best = np.argsort(-score, axis=1, kind="stable")[:, :PK_PEAKS]
            value = np.take_along_axis(score, best, axis=1)
            found = np.isfinite(value)
            out[done:stop, :, 0] = np.where(found, f_band[best], np.nan)
            out[done:stop, :, 1] = np.where(found, value, np.nan)
            done = stop
        keep_from = done * hop - half - origin
        if keep_from > 0:
            buffer, origin = buffer[keep_from:], origin + keep_from
        return buffer, origin, done

    pending, pending_n = [], 0
    for block in blocks:
        pending.append(np.asarray(block, dtype=np.float64).reshape(-1))
        pending_n += len(pending[-1])
        if pending_n >= BLOCK_FRAMES * hop:
            buffer = np.concatenate([buffer] + pending)
            pending, pending_n = [], 0
            buffer, origin, done = emit(buffer, origin, done, False)
            if done >= n_frames:
                return out
    buffer = np.concatenate([buffer] + pending)
    if done < n_frames and origin + len(buffer) > done * hop:
        emit(buffer, origin, done, True)
    return out


def peak_video_features(path, n_frames, fps, cache_dir=None):
    """peak_features() of a video's audio, cached as .npz in cache/whistle/ (gitignored: derived
    from match data). The cache key is the file's size and mtime and PK_VERSION."""
    if cache_dir is None:
        from solution.match import CACHE_DIR
        cache_dir = os.path.join(CACHE_DIR, "whistle")
    stat = os.stat(path)
    key = "{}_{}_{}_pk{}".format(os.path.splitext(os.path.basename(path))[0], stat.st_size,
                                 int(stat.st_mtime), PK_VERSION)
    cache = os.path.join(cache_dir, key + ".npz")
    if os.path.isfile(cache):
        return np.load(cache)["features"]
    features = peak_features(audio_blocks(path), n_frames, fps)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(cache, features=features)
    return features


def kickoff_anchors(match):
    """Kick frames of every kickoff the Law 8 cues find anywhere in each half: solution/kickoff.py
    with no search window and without its 'nothing played before' guard (which is there to keep
    the first kickoff only), so restarts after goals count too. From the tracking only."""
    from solution import kickoff
    from solution.simulate import mean_player_speed
    params = dict(kickoff.PARAMS, search_s=1e9)
    form = kickoff.formation(match, params)
    speed = mean_player_speed(match)
    out = []
    for start, stop in match.halves:
        for a, b in kickoff.holds(start, stop, form, match.fps, params):
            k = kickoff.kick_frame(a, b, speed, match.fps, params)
            passed = kickoff.checks(k, start, stop, speed, match.fps, params)
            if passed["quiet"] and passed["rise"]:
                out.append(int(k))
    return sorted(out)


def _n_bins():
    return int(np.ceil((PK_BAND_HZ[1] - PK_BAND_HZ[0]) / CAL_BIN_HZ)) + 1


def _cell_counts(features, frames, fps):
    """Frames in `frames` with a tone in each pitch cell: (singles (B,), pairs (B, B), upper
    triangle: low cell, high cell). Each frame counts once per cell."""
    n_b = _n_bins()
    singles, pairs = np.zeros(n_b), np.zeros((n_b, n_b))
    frames = np.asarray(frames, dtype=np.int64)
    if not len(frames):
        return singles, pairs
    hz, db = features[frames, :, 0], features[frames, :, 1]
    on = _steady(features, frames, fps)
    cells = np.where(on, np.floor((np.nan_to_num(hz) - PK_BAND_HZ[0]) / CAL_BIN_HZ), -1)
    cells = cells.astype(np.int64)
    marks = np.zeros((len(frames), n_b + 1), dtype=bool)
    marks[np.arange(len(frames))[:, None], np.where(cells >= 0, cells, n_b)] = True
    singles = marks[:, :n_b].sum(axis=0).astype(float)
    codes = []
    for i in range(PK_PEAKS):
        for j in range(i + 1, PK_PEAKS):
            a, b = cells[:, i], cells[:, j]
            ok = (a >= 0) & (b >= 0) & (np.abs(a - b) >= 2)   # two tones, not one split
            codes.append(np.where(ok, np.minimum(a, b) * n_b + np.maximum(a, b), -1))
    codes = np.sort(np.stack(codes, axis=1), axis=1)
    fresh = np.ones(codes.shape, dtype=bool)
    fresh[:, 1:] = codes[:, 1:] != codes[:, :-1]
    valid = codes[(codes >= 0) & fresh]
    pairs = np.bincount(valid, minlength=n_b * n_b).reshape(n_b, n_b).astype(float)
    return singles, pairs


def _steady(features, frames, fps):
    """(len(frames), PK_PEAKS) bool: the peak is a tone (>= PK_ON_DB) and a tone within
    CAL_STEADY_HZ of its pitch sounds CAL_STEADY_S before or after it."""
    d = max(1, int(round(CAL_STEADY_S * fps)))
    n = len(features)
    hz, db = features[frames, :, 0], features[frames, :, 1]
    with np.errstate(invalid="ignore"):
        on = db >= PK_ON_DB
        out = np.zeros(on.shape, dtype=bool)
        for step in (-d, d):
            other = np.clip(frames + step, 0, n - 1)
            o_hz, o_db = features[other, :, 0], features[other, :, 1]
            near = (np.abs(hz[:, :, None] - o_hz[:, None, :]) <= CAL_STEADY_HZ) & \
                   (o_db[:, None, :] >= PK_ON_DB)
            out |= near.any(axis=2)
    return on & out


def _neighbourhood(counts):
    """Sum over each cell and its neighbours (pitch wobbles across a cell edge)."""
    from scipy.ndimage import convolve
    kernel = np.ones((3,) * counts.ndim)
    return convolve(counts, kernel, mode="constant")


def _anchor_frames(anchor, n_frames, fps):
    a = max(0, anchor - int(round(CAL_BEFORE_S * fps)))
    return np.arange(a, min(n_frames, anchor + int(round(CAL_AFTER_S * fps)) + 1))


def _present(counts, k):
    return (_neighbourhood(counts[k]) >= CAL_MIN_FRAMES).astype(float)


def calibrate(features, anchors, graded, fps):
    """The whistle's signature from the anchors (see the section comment), or None.

    `graded`: (T,) bool; anchors and chance windows must lie in it. Returns a dict: kind ("pair"
    or "single"), tones (Hz, low to high), bands ((lo, hi) Hz per tone), anchors_hit, n_anchors,
    chance (share of chance windows with the signature), ratio."""
    n = len(features)
    anchors = sorted(int(a) for a in anchors if 0 <= int(a) < n and graded[int(a)])
    if len(anchors) < CAL_MIN_ANCHORS:
        return None
    windows = [_anchor_frames(a, n, fps) for a in anchors]
    taken = np.zeros(n, dtype=bool)
    for w in windows:
        taken[w] = True
    shifted = []
    for a in anchors:
        for shift in CAL_SHIFTS_S:
            w = _anchor_frames(a + int(round(shift * fps)), n, fps)
            if len(w) and w[0] >= 0 and graded[w].all() and not taken[w].any():
                shifted.append(w)
    per_anchor = [_cell_counts(features, w, fps) for w in windows]
    per_shift = [_cell_counts(features, w, fps) for w in shifted]
    all_window = np.unique(np.concatenate(windows))
    floor = 1.0 / max(len(shifted), 1)
    for kind, k in (("pair", 1), ("single", 0)):
        present = sum(_present(c, k) for c in per_anchor)
        chance = (sum(_present(c, k) for c in per_shift) / len(per_shift) if per_shift
                  else np.zeros_like(present))
        ratio = (present / len(anchors)) / np.maximum(chance, floor)
        ok = (present >= CAL_MIN_ANCHORS) & (ratio >= CAL_MIN_RATIO)
        if not ok.any():
            continue
        flat = np.flatnonzero(ok.ravel())
        best = flat[np.lexsort((flat, -ratio.ravel()[flat], -present.ravel()[flat]))[0]]
        cell = np.unravel_index(best, present.shape)
        tones = _tone_pitches(features, all_window, cell, fps)
        if tones is None:
            continue
        return {"kind": kind, "tones": tones,
                "bands": [(t - BAND_HALF_HZ, t + BAND_HALF_HZ) for t in tones],
                "anchors_hit": int(present[cell]), "n_anchors": len(anchors),
                "n_chance_windows": len(shifted), "chance": float(chance[cell]),
                "ratio": float(ratio[cell])}
    return None


def _tone_pitches(features, frames, cell, fps):
    """Median pitch of each tone of a signature cell (and its neighbours) over `frames`, on the
    frames where all its tones sound together; None if there are none."""
    hz = features[frames, :, 0]
    on = _steady(features, frames, fps)
    with np.errstate(invalid="ignore"):
        cells = np.floor((hz - PK_BAND_HZ[0]) / CAL_BIN_HZ)
    near = [on & (np.abs(cells - c) <= 1) for c in cell]
    together = np.logical_and.reduce([n.any(axis=1) for n in near])
    if not together.any():
        return None
    out = []
    for n in near:
        rows = n[together]
        out.append(float(np.median(hz[together][rows])))
    return sorted(out)


def _band_track(features, band):
    """Per frame: (a tone in `band` is on, its pitch, its dB), the strongest one in the band."""
    hz, db = features[:, :, 0], features[:, :, 1]
    with np.errstate(invalid="ignore"):
        inside = (hz >= band[0]) & (hz <= band[1]) & (db >= PK_ON_DB)
    score = np.where(inside, db, -np.inf)
    pick = score.argmax(axis=1)
    rows = np.arange(len(hz))
    return inside.any(axis=1), hz[rows, pick], db[rows, pick]


def whistle_events(features, cal, fps):
    """Candidates and whistles in the calibrated band(s). Each event: start / stop (video frames,
    stop exclusive), seconds, tones_hz (median pitch per band), hz (the highest tone's, for
    whistle.pair_feeds), together (frames every tone is on), stds, and `whistle`."""
    if cal is None:
        return []
    tracks = [_band_track(features, band) for band in cal["bands"]]
    ons = [t[0] for t in tracks]
    on = np.logical_or.reduce(ons)
    every = np.logical_and.reduce(ons)
    edges = np.diff(np.concatenate(([0], on.astype(np.int8), [0])))
    merged = []
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if merged and a - merged[-1][1] <= EV_GAP_S * fps:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    out = []
    for a, b in merged:
        if b - a < EV_MIN_S * fps:
            continue
        tones, stds = [], []
        for t_on, t_hz, _ in tracks:
            sel = t_on[a:b]
            tones.append(float(np.median(t_hz[a:b][sel])) if sel.any() else float("nan"))
            stds.append(float(np.std(t_hz[a:b][sel])) if sel.any() else float("nan"))
        together = int(every[a:b].sum())
        steady = all(np.isfinite(s) and s <= EV_STEADY_HZ for s in stds)
        out.append({"start": int(a), "stop": int(b), "seconds": (b - a) / float(fps),
                    "tones_hz": tones, "hz": tones[-1], "together": together, "stds": stds,
                    "whistle": bool(together >= EV_MIN_HITS and steady)})
    return out


def match_whistles(features, match):
    """(calibration or None, whistle events) for a match from its own audio features and its
    tracking: kickoff_anchors -> calibrate -> whistle_events. No person in the loop."""
    cal = calibrate(features, kickoff_anchors(match), match.in_play_mask(), match.fps)
    return cal, [e for e in whistle_events(features, cal, match.fps) if e["whistle"]]
