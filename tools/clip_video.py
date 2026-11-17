"""Cut a frame range of a provided video into a clip, frame-accurate, for a person to watch. (needs data/)

    .venv/bin/python tools/clip_video.py --from 18685 --to 26184
    make clip FROM=18685 TO=26184 [SOURCE=tactical] [ARGS="--stamp"]

Frames are video frames, which for the tactical recording are submission rows (ATD frame_id - 1).
The window is inclusive at both ends; mm:ss or mm:ss.s is accepted and converted at 25 fps.
Without --from/--to the window is the five minutes right after the public truth: its last graded
frame plus one, for as many frames as it has (18685..26184 for BAR-ATH, 12:27.4 to 17:27.4).

The clip is decoded and re-encoded (H.264, libx264, CRF 18) so that its frame k is exactly source
frame FROM + k. A container-level stream copy would start on the keyframe before FROM and the
mapping would be off by up to a GOP, which a person reading frame numbers off the clip must not
have. `--stamp` burns the source frame number and video clock into the top-left corner, so the
number does not have to be worked out from the player's clock. The audio track, when the source
has one, is copied over the same time range (--no-audio drops it). `--max-mb 480` keeps the file
under that size by setting the bit rate from the frame count; `--codec h264_videotoolbox` is
several times faster on a Mac but needs that cap, its default rate is large.

Output: data/clips/<stem>_f<from>-<to>.mp4 (gitignored: data/* and *.mp4), or --out.

RULES.md: watching the video is allowed; hand labels on the provided game are not, in any amount
and for any purpose -- not for the submission, not to train a model and not to validate one
(rule 1). A clip is for understanding the data and debugging a method, read beside the
moment_check and frame_report tables.

Nothing about the frames is printed: the tool reports counts, times and the output path.
"""

import argparse
import csv
import os
import sys
import time
from fractions import Fraction

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import DEFAULT_SOURCE, GAME, GAMES, SOURCES, require  # noqa: E402

FPS = GAMES[GAME]["fps"]
GT_PUBLIC = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
CLIP_DIR = os.path.join(ROOT, "data", "clips")


def parse_time(text, fps=FPS):
    """'mm:ss', 'mm:ss.s' or a frame number -> video frame."""
    text = str(text).strip()
    if ":" not in text:
        return int(text)
    minutes, seconds = text.split(":")
    return int(round((int(minutes) * 60 + float(seconds)) * fps))


def clock(frame, fps=FPS):
    """Video frame -> 'mm:ss.s'."""
    seconds = frame / float(fps)
    return "{:02d}:{:04.1f}".format(int(seconds // 60), seconds % 60)


def public_window(path=GT_PUBLIC):
    """(first, last) graded video frame of the public truth: the rows whose label is not X."""
    first = last = None
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["label"].strip().upper() != "X":
                frame = int(row["frame"])
                first = frame if first is None else first
                last = frame
    if first is None:
        raise ValueError("no graded frame in {}".format(path))
    return first, last


def next_window(path=GT_PUBLIC):
    """The window right after the public truth, of the same length."""
    first, last = public_window(path)
    return last + 1, last + (last - first + 1)


class Stamp:
    """Burns 'f <frame>  mm:ss.s' into the top-left corner. Pillow's bundled font, no font file."""

    def __init__(self, height, rate):
        from PIL import ImageDraw, ImageFont
        self.draw = ImageDraw.Draw
        self.rate = rate
        size = max(14, int(height) // 30)
        try:
            self.font = ImageFont.load_default(size=size)
        except TypeError:  # older Pillow: the fixed-size bitmap font
            self.font = ImageFont.load_default()
        self.margin = max(4, size // 3)

    def __call__(self, frame, index):
        import av
        image = frame.to_image()
        draw = self.draw(image)
        text = "f {}  {}".format(index, clock(index, self.rate))
        x, y = self.margin * 2, self.margin
        left, top, right, bottom = draw.textbbox((x, y), text, font=self.font)
        pad = self.margin
        draw.rectangle((left - pad, top - pad, right + pad, bottom + pad), fill=(0, 0, 0))
        draw.text((x, y), text, font=self.font, fill=(255, 255, 255))
        return av.VideoFrame.from_image(image)


MARGIN = 0.85  # share of --max-mb the video bit rate aims at; see bitrate_for


def bitrate_for(max_mb, n_frames, rate, audio_bps=0, margin=MARGIN):
    """Video bit rate (bit/s) that keeps a clip of n_frames under max_mb megabytes (10^6 bytes).

    `margin` leaves room for the container and for an encoder that only holds its average
    rate approximately: on synthetic 1080p noise, the worst case, h264_videotoolbox lands at
    up to 1.12x the rate it was given and libx264 at about 1.08x. The audio track's rate, when
    it is known, comes off the top.
    """
    seconds = float(n_frames / Fraction(rate))
    return max(int(max_mb * 1e6 * 8 * margin / seconds - audio_bps), 200_000)


def clip(src, first, last, out, codec="libx264", crf=18, preset="fast", stamp=False,
         audio=True, seek=True, fps=FPS, bitrate=None, max_mb=None):
    """Write source frames first..last (inclusive) to `out`; return the number written.

    With `seek` the container is positioned on the keyframe before `first` and each decoded
    frame's index is read off its timestamp; frames before `first` are dropped. Without it
    the file is decoded from its start and frames are counted, which is slower but needs no
    timestamps. Either way the window must come out as one run of consecutive indices, or the
    tool stops rather than write a clip whose frame numbers are wrong.

    `bitrate` (bit/s) or `max_mb` (a file size the clip must stay under) puts the encoder in
    average-bit-rate mode; libx264 then ignores `crf`. Without either, libx264 encodes at
    `crf` and other encoders at their own default, which for h264_videotoolbox is large.
    """
    import av

    if last < first:
        raise ValueError("--to {} is before --from {}".format(last, first))
    if first < 0:
        raise ValueError("--from {} is negative".format(first))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    with av.open(src) as inp:
        vs = inp.streams.video[0]
        vs.codec_context.thread_type = "AUTO"
        rate = vs.average_rate or vs.guessed_rate or Fraction(fps)
        if vs.frames and last >= vs.frames:
            raise ValueError("--to {} is past the last frame {} of {}".format(
                last, vs.frames - 1, os.path.basename(src)))
        start_pts = vs.start_time or 0
        aud = inp.streams.audio[0] if (audio and inp.streams.audio) else None
        streams = [vs] + ([aud] if aud is not None else [])
        t_first = Fraction(first) / rate
        t_end = Fraction(last + 1) / rate
        if seek:
            target = start_pts + int(t_first / vs.time_base)
            inp.seek(target, stream=vs, backward=True, any_frame=False)
        if max_mb is not None:
            audio_bps = (aud.bit_rate or 0) if aud is not None else 0
            bitrate = bitrate_for(max_mb, last - first + 1, rate, audio_bps)

        with av.open(out, "w") as outp:
            options = {}
            if codec.startswith("libx26"):
                options["preset"] = preset
                if not bitrate:
                    options["crf"] = str(crf)
            ov = outp.add_stream(codec, rate=rate, options=options)
            ov.width, ov.height = vs.width, vs.height
            ov.pix_fmt = "yuv420p"
            if bitrate:
                ov.codec_context.bit_rate = int(bitrate)
            time_base = 1 / Fraction(rate)
            ov.codec_context.time_base = time_base
            oa = outp.add_stream_from_template(aud) if aud is not None else None
            if aud is not None:
                a_start = aud.start_time or 0
                a_shift = a_start + int(t_first / aud.time_base)
            painter = Stamp(vs.height, rate) if stamp else None
            written = 0
            expected = first
            counter = 0
            for packet in inp.demux(streams):
                if aud is not None and packet.stream_index == aud.index:
                    if packet.pts is None:
                        continue
                    at = (packet.pts - a_start) * aud.time_base
                    if at < t_first or at >= t_end:
                        continue
                    packet.pts -= a_shift
                    if packet.dts is not None:
                        packet.dts -= a_shift
                    packet.stream = oa
                    outp.mux(packet)
                    continue
                for frame in packet.decode():
                    if seek:
                        if frame.pts is None:
                            raise RuntimeError("a decoded frame has no timestamp; rerun with "
                                               "--no-seek")
                        index = int(round((frame.pts - start_pts) * vs.time_base * rate))
                    else:
                        index = counter
                        counter += 1
                    if index < first:
                        continue
                    if index != expected:
                        raise RuntimeError(
                            "decoded frame {} where {} was expected: the stream is not constant "
                            "rate around the window; rerun with --no-seek".format(index, expected))
                    new = painter(frame, index) if painter else frame
                    new.pts = written
                    new.time_base = time_base
                    for pkt in ov.encode(new):
                        outp.mux(pkt)
                    written += 1
                    expected += 1
                    if expected > last:
                        break
                if expected > last:
                    break
            for pkt in ov.encode(None):
                outp.mux(pkt)
    if written != last - first + 1:
        raise RuntimeError("the source ended at frame {}; {} of {} frames written".format(
            expected - 1, written, last - first + 1))
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="first",
                        help="first frame, or mm:ss[.s]; default: the frame after the public truth")
    parser.add_argument("--to", dest="last",
                        help="last frame (inclusive), or mm:ss[.s]; default: the public window's "
                             "length from --from")
    parser.add_argument("--source", default=DEFAULT_SOURCE, choices=sorted(SOURCES))
    parser.add_argument("--data-dir", default=None, help="where the bundle is (default: data/)")
    parser.add_argument("--out", help="output file (default: data/clips/<stem>_f<from>-<to>.mp4)")
    parser.add_argument("--codec", default="libx264",
                        help="encoder (default libx264; h264_videotoolbox is faster on a Mac)")
    parser.add_argument("--crf", type=int, default=18, help="libx264 quality, lower is better")
    parser.add_argument("--preset", default="fast", help="libx264 speed/size trade-off")
    parser.add_argument("--max-mb", type=float, default=None,
                        help="keep the file under this many MB (10^6 bytes) by setting the bit "
                             "rate; libx264 then ignores --crf")
    parser.add_argument("--bitrate", type=float, default=None,
                        help="video bit rate in Mbit/s, instead of --max-mb or --crf")
    parser.add_argument("--stamp", action="store_true",
                        help="burn the source frame number and clock into the frames")
    parser.add_argument("--no-audio", action="store_true", help="do not copy the audio track")
    parser.add_argument("--no-seek", action="store_true",
                        help="decode from the start of the file and count frames (slower)")
    args = parser.parse_args(argv)

    pub_first, pub_last = public_window()
    length = pub_last - pub_first + 1
    first = parse_time(args.first) if args.first else pub_last + 1
    last = parse_time(args.last) if args.last else first + length - 1
    src = require(["video"], data_dir=args.data_dir, source=args.source)[0]
    stem = os.path.splitext(os.path.basename(src))[0]
    out = args.out or os.path.join(CLIP_DIR, "{}_f{}-{}.mp4".format(stem, first, last))

    print("source  {}".format(src))
    print("window  frames {}..{} ({} frames, {} to {}, public truth is {}..{})".format(
        first, last, last - first + 1, clock(first), clock(last + 1), pub_first, pub_last))
    started = time.time()
    written = clip(src, first, last, out, codec=args.codec, crf=args.crf, preset=args.preset,
                   stamp=args.stamp, audio=not args.no_audio, seek=not args.no_seek,
                   bitrate=args.bitrate * 1e6 if args.bitrate else None, max_mb=args.max_mb)
    print("wrote   {} ({} frames, {:.0f} s, {:.0f} MB)".format(
        out, written, time.time() - started, os.path.getsize(out) / 1e6))
    print("mapping clip frame k = video frame {} + k = ATD frame_id {} + k; clip 00:00.0 is "
          "video {}".format(first, first + 1, clock(first)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
