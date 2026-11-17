"""Inspect one moment of BAR-ATH: what the decoder saw there, second by second. (needs data/)

    .venv/bin/python tools/moment_check.py --from 11:50 --to 12:20
    make moment FROM=11:50 TO=12:20 [METHOD=v6_deadball_model] [METHODS="touch_c3_seen_w1_dms"]

For a person to read against the video. RULES.md allows inspection; it forbids writing what was
seen into labels, frame constants or a variant tuned to the moments watched. A decision between
registered methods that a viewing informs goes to docs/DECISIONS.md before run.py changes.

Three tables for the window, one row per second of the tactical video, for the method `--method`
(default: the submission):
  1. inputs: the public truth where graded, the decoded label, the broadcast clock and what the
     TV showed (from cache/broadcast/BAR-ATH_align.npz when it exists), the share of frames with
     the ball seen / beyond a line / usable to the touch engine, the nearest player's team shares,
     the touch engine's holder shares and its touches, the collective speed rank, the dead-ball
     model's mean log P(x | dead) / P(x | alive), the frame model's mean log-odds for home, and
     the share of frames with a player standing beyond a line (deadball.py's cue, whether or not
     the method reads it);
  2. team evidence: the decoder's H minus A per block (mean over the second's five blocks, nats),
     and how much each evidence term moves it (the term's weight set to 0 and the difference
     taken; positive = towards home);
  3. dead-ball evidence: D minus the better alive state, and each term's share of it (positive =
     towards dead). "deadm" is what the learned model adds over the speed rank it replaces.
`--methods` adds the labels other registered methods decode in the window (a full decode each).
Every column is a label, a share, a rank, a count or a log-likelihood; no position is printed.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES  # noqa: E402
from solution import deadball, decoder  # noqa: E402
from solution.match import pm  # noqa: E402
from solution.methods import METHODS  # noqa: E402

FPS = GAMES[GAME]["fps"]
NAME = {pm.H: "H", pm.A: "A", pm.D: "D", pm.X: "X"}
GT_PUBLIC = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
TERMS = (("touch_w", "touch"), ("frame_w", "frame"), ("deadm_w", "deadm"), ("team_w", "team"),
         ("deadball_w", "cue"), ("tv_w", "tv"))
TV_SHORT = {0: "none", 1: "live", 2: "cut", 3: "replay", 4: "unaligned"}


def parse_time(text):
    """'mm:ss', 'mm:ss.s' or a frame number -> video frame."""
    text = str(text).strip()
    if ":" not in text:
        return int(text)
    minutes, seconds = text.split(":")
    return int(round((int(minutes) * 60 + float(seconds)) * FPS))


def clock(frame):
    seconds = frame / float(FPS)
    return "{:02d}:{:04.1f}".format(int(seconds // 60), seconds % 60)


def majority(codes):
    counts = np.bincount(codes.astype(np.int64), minlength=pm.N_LABELS)
    return NAME[int(np.argmax(counts))]


def load_alignment(path=None):
    """(bro_frame, state) from the broadcast alignment cache, or None."""
    if path is None:
        from solution.sources.fused import ALIGN_PATH
        path = ALIGN_PATH
    if not os.path.isfile(path):
        return None
    z = np.load(path, allow_pickle=False)
    return z["bro_frame"], z["state"]


def load_public_gt(path=GT_PUBLIC):
    if not os.path.isfile(path):
        return None
    return pm.label_codes(pm.read_label_csv(path))


def block_margins(method, inp, start, stop):
    """{'full': (n_blocks, 3) emissions, term: emissions without that term} for one span."""
    settings = method.settings
    nearest, seen, speed, priors, holder, cue, team, frame, deadm = method.half_inputs(
        inp, start, stop)
    tv = None if inp.get("tv") is None else inp["tv"][start:stop]

    def emissions(params):
        return decoder.block_emissions(nearest, seen, speed, priors, params, holder, cue, team,
                                       frame, deadm, tv)
    out = {"full": emissions(settings)}
    for key, name in TERMS:
        if settings.get(key, 0.0) > 0:
            out[name] = emissions(dict(settings, **{key: 0.0}))
    return out


def inspect(match, method, first, last, gt=None, alignment=None, others=None):
    """One dict per second of [first, last) with every column the tables print."""
    inp = method.prepare(match)
    states = method.decode(match, inp)
    span = next(((s, t) for s, t in inp["spans"] if s <= first < t), None)
    if span is None:
        raise SystemExit("{} is outside the decoded halves".format(clock(first)))
    start, stop = span
    last = min(last, stop)
    margins = block_margins(method, inp, start, stop)
    seen = match.ball_seen
    beyond = deadball.ball_beyond(match)
    player_beyond = deadball.beyond_line(match)
    usable = None if inp["touches"] is None else inp["touches"].usable
    holder = inp["holder"]
    touches = inp["touches"]
    nearest, speed = inp["nearest"], inp["speed"]
    frame, deadm = inp["frame"], inp.get("deadm")
    tv = inp.get("tv")
    other_states = {name: METHODS[name](match) for name in (others or [])}

    def share(values, lo, hi):
        return float(np.mean(values[lo:hi])) if values is not None else None

    def mean(values, lo, hi):
        if values is None:
            return None
        window = values[lo:hi]
        window = window[np.isfinite(window)]
        return float(window.mean()) if len(window) else None

    rows = []
    for lo in range(first, last, FPS):
        hi = min(lo + FPS, last)
        b0, b1 = (lo - start) // decoder.BLOCK, -(-(hi - start) // decoder.BLOCK)
        row = {
            "frame": lo, "clock": clock(lo),
            "gt": majority(gt[lo:hi]) if gt is not None else "?",
            "decoded": majority(states[lo:hi]),
            "seen": share(seen, lo, hi), "beyond": share(beyond, lo, hi),
            "usable": share(usable, lo, hi),
            "near_h": share(nearest == 0, lo, hi), "near_a": share(nearest == 1, lo, hi),
            "hold_h": share(holder == 0, lo, hi) if holder is not None else None,
            "hold_a": share(holder == 1, lo, hi) if holder is not None else None,
            "touches": "", "speed": mean(speed, lo, hi), "deadm": mean(deadm, lo, hi),
            "frame_odds": mean(frame, lo, hi), "player_beyond": share(player_beyond, lo, hi),
            "tv": None, "broadcast": "", "tv_state": "",
        }
        if touches is not None:
            pick = (touches.touch_frame >= lo) & (touches.touch_frame < hi)
            teams = touches.touch_team[pick]
            row["touches"] = "".join("H" if t == 0 else "A" for t in teams)
        if tv is not None:
            known = tv[lo:hi] >= 0
            row["tv"] = float(np.mean(tv[lo:hi][known] == 1)) if known.any() else None
        if alignment is not None:
            bro_frame, state = alignment
            bf = int(bro_frame[lo])
            row["broadcast"] = clock(bf) if bf >= 0 else "--"
            row["tv_state"] = TV_SHORT.get(int(state[lo]), "?")
        full = margins["full"][b0:b1]
        row["h_minus_a"] = float(np.mean(full[:, 0] - full[:, 1]))
        row["d_minus_alive"] = float(np.mean(full[:, 2] - np.maximum(full[:, 0], full[:, 1])))
        for name, without in margins.items():
            if name == "full":
                continue
            w = without[b0:b1]
            row["team_" + name] = row["h_minus_a"] - float(np.mean(w[:, 0] - w[:, 1]))
            row["dead_" + name] = row["d_minus_alive"] - float(
                np.mean(w[:, 2] - np.maximum(w[:, 0], w[:, 1])))
        for name, codes in other_states.items():
            row["other_" + name] = majority(codes[lo:hi])
        rows.append(row)
    return rows


def fmt(value, width=6, digits=2):
    if value is None:
        return "{:>{w}}".format("--", w=width)
    return "{:>{w}.{d}f}".format(value, w=width, d=digits)


def print_tables(rows, method_name, others=()):
    terms = [name for _, name in TERMS if "team_" + name in rows[0]]
    print("Method {}: evidence terms read: {}".format(method_name, ", ".join(terms) or "none"))

    print("\n1. Inputs per second (shares of the second's frames unless stated)")
    head = "{:>8} {:>9} {:>9}  {:>2} {:>3}  {:>4} {:>6} {:>6}  {:>5} {:>5}  {:>5} {:>5} {:>6}  " \
           "{:>5} {:>6} {:>6} {:>6} {:>5}".format(
               "tactical", "broadcast", "TV", "gt", "dec", "seen", "beyond", "usable",
               "nearH", "nearA", "holdH", "holdA", "touch", "speed", "deadm", "frame", "plyr>", "tvoff")
    print(head)
    print("  (speed = collective speed rank 0..1; deadm = log P(x|dead)/P(x|alive), > 0 says dead; "
          "frame = log-odds home, > 0 says home;\n   plyr> = share with a player standing beyond a "
          "line; tvoff = share of known frames with the TV off the live pitch)")
    for r in rows:
        print("{:>8} {:>9} {:>9}  {:>2} {:>3}  {} {} {}  {} {}  {} {} {:>6}  {} {} {} {} {}".format(
            r["clock"], r["broadcast"], r["tv_state"], r["gt"], r["decoded"],
            fmt(r["seen"], 4, 2), fmt(r["beyond"], 6, 2), fmt(r["usable"], 6, 2),
            fmt(r["near_h"], 5, 2), fmt(r["near_a"], 5, 2),
            fmt(r["hold_h"], 5, 2), fmt(r["hold_a"], 5, 2), r["touches"],
            fmt(r["speed"], 5, 2), fmt(r["deadm"], 6, 2), fmt(r["frame_odds"], 6, 2),
            fmt(r["player_beyond"], 6, 2), fmt(r["tv"], 5, 2)))

    print("\n2. Team evidence: H minus A per block (nats, mean over the second), and each term's "
          "part (positive = towards home)")
    print("{:>8} {:>3} {:>7}  ".format("tactical", "dec", "H-A") +
          "".join("{:>7}".format(t) for t in terms) + "{:>7}".format("rest"))
    for r in rows:
        parts = [r["team_" + t] for t in terms]
        print("{:>8} {:>3} {}  ".format(r["clock"], r["decoded"], fmt(r["h_minus_a"], 7, 2)) +
              "".join(fmt(p, 7, 2) for p in parts) +
              fmt(r["h_minus_a"] - sum(parts), 7, 2))

    print("\n3. Dead-ball evidence: D minus the better alive state per block, and each term's part "
          "(positive = towards dead; 'rest' is the nearest player, missing ball and speed rank)")
    print("{:>8} {:>3} {:>7}  ".format("tactical", "dec", "D-alive") +
          "".join("{:>7}".format(t) for t in terms) + "{:>7}".format("rest"))
    for r in rows:
        parts = [r["dead_" + t] for t in terms]
        print("{:>8} {:>3} {}  ".format(r["clock"], r["decoded"], fmt(r["d_minus_alive"], 7, 2)) +
              "".join(fmt(p, 7, 2) for p in parts) +
              fmt(r["d_minus_alive"] - sum(parts), 7, 2))
    print("  The decoder adds duration priors and transition costs on top; a block's sign alone "
          "does not decide it.")

    if others:
        print("\n4. What other registered methods decode here")
        print("{:>8} {:>2} {:>3}  ".format("tactical", "gt", "dec") +
              "".join("{:>{w}}".format(o, w=len(o) + 2) for o in others))
        for r in rows:
            print("{:>8} {:>2} {:>3}  ".format(r["clock"], r["gt"], r["decoded"]) +
                  "".join("{:>{w}}".format(r["other_" + o], w=len(o) + 2) for o in others))


def main(argv=None):
    from solution.run import METHOD, require_alignment
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="first", required=True, help="mm:ss[.s] or a video frame")
    parser.add_argument("--to", dest="last", required=True, help="mm:ss[.s] or a video frame")
    parser.add_argument("--method", default=METHOD, choices=sorted(METHODS), metavar="NAME",
                        help="registered method to inspect (default: the submission)")
    parser.add_argument("--methods", nargs="*", default=[], metavar="NAME",
                        help="other registered methods whose labels to show in the window")
    args = parser.parse_args(argv)
    first, last = parse_time(args.first), parse_time(args.last)
    if last <= first:
        raise SystemExit("--to must be after --from")
    for name in [args.method] + args.methods:
        if name not in METHODS:
            raise SystemExit("unknown method {}".format(name))
        require_alignment(name)

    from solution.sources import atd
    match = atd.load()
    rows = inspect(match, METHODS[args.method], first, last, load_public_gt(), load_alignment(),
                   args.methods)
    print_tables(rows, args.method, args.methods)
    return 0


if __name__ == "__main__":
    sys.exit(main())
