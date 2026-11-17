"""Frame-by-frame view of result files against the public ground truth.

    .venv/bin/python tools/frame_report.py evaluation/results/our_team.csv [more.csv ...]
    make frame-report FILES="our_team c3_seen_w1" [ARGS="--top 8 --res 1"]

spellF1 scores possessions as objects. This prints what the labels do frame by frame on the
graded window instead: the confusion matrix and per-class precision and recall; every run of
frames where a file and the truth disagree, sorted into the kind of mistake it is (a late or an
early boundary, a missed segment, a spurious segment, a relabelled segment); what each kind costs
in macroF1 when it is repaired with the truth; macroF1 with a margin around every true boundary
forgiven; the truth segment by segment with its head, tail and interior errors; the latency of
every true transition; and a timeline strip. The first file is the reference; others are shown
beside it. It reads result files and the ground truth, nothing under data/.

Correctness and description only (EXPERIMENTS section 4, RULES.md): the window holds 17
possessions, so nothing printed here chooses between methods, and nothing here may be tuned on.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution.match import pm  # noqa: E402
from tools.diff_submissions import diff_runs  # noqa: E402

FPS = 25
NAME = {pm.H: "H", pm.A: "A", pm.D: "D", pm.X: "X"}
GT_PUBLIC = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
KINDS = ("late", "early", "missed", "spurious", "other")
KIND_TEXT = {
    "late": "the file keeps a label past the end of its true segment",
    "early": "the file takes a label before its true segment starts",
    "missed": "a true segment with no right frame (n counts true segments)",
    "spurious": "a file segment with no right frame, its frames outside missed true segments "
                "(n counts file segments)",
    "other": "a disagreement that is none of the above",
}


def clock(frame):
    """Video time of a frame as mm:ss.s from the start of the recording."""
    seconds = frame / float(FPS)
    return "{:02d}:{:04.1f}".format(int(seconds // 60), seconds % 60)


def load(path):
    frame = pm.read_label_csv(path)
    reason = pm.validate_submission(frame, len(frame))
    if reason:
        raise SystemExit("{}: {}".format(path, reason))
    return pm.label_codes(frame)


# -- the disagreement runs ---------------------------------------------------------------------

def classify_frames(gt, pred, runs):
    """Per frame, the index into KINDS of the mistake there, or -1 where the file is right.

    Decided at segment level, so the kind is the same for every frame of a mistake:
    - a true segment with no right frame at all is 'missed' (every frame of it);
    - a file segment with no right frame at all is 'spurious', whether it sits inside one true
      segment or bridges a true boundary; where the two overlap the frames count as missed;
    - what remains lies at the head or the tail of a file segment that is right somewhere: before
      its first right frame the file took the label too soon ('early'), after its last right
      frame it kept the label too long ('late'). A file segment that starts early across a
      missed true segment is early on the frames outside that segment.
    'other' cannot occur by construction and is kept so a bug would show rather than hide.
    """
    index = {kind: i for i, kind in enumerate(KINDS)}
    kind = np.full(len(gt), -1, dtype=np.int8)
    correct = gt == pred
    for first, last in runs:
        missed = np.zeros(last - first, dtype=bool)
        spurious = np.zeros(last - first, dtype=bool)
        for start, stop, _ in pm.segments(gt, [(first, last)]):
            if not correct[start:stop].any():
                missed[start - first:stop - first] = True
        pred_segments = pm.segments(pred, [(first, last)])
        for start, stop, _ in pred_segments:
            if not correct[start:stop].any():
                spurious[start - first:stop - first] = True
        wrong = ~correct[first:last]
        window = kind[first:last]
        window[wrong & missed] = index["missed"]
        window[wrong & ~missed & spurious] = index["spurious"]
        for start, stop, _ in pred_segments:
            right = np.flatnonzero(correct[start:stop])
            if not len(right):
                continue
            head = slice(start - first, start - first + right[0])
            tail = slice(start - first + right[-1] + 1, stop - first)
            for part, name in ((head, "early"), (tail, "late")):
                todo = wrong[part] & (window[part] < 0)
                window[part][todo] = index[name]
        window[wrong & (window < 0)] = index["other"]
    return kind


def pair_name(g, p):
    return "{}>{}".format(NAME[g], NAME[p])


def error_table(gt, pred, runs):
    """[(start, stop, gt_label, pred_label, kind, bridges)] for every disagreement run; bridges
    is True for a spurious segment of the file that spans a true boundary."""
    kinds = classify_frames(gt, pred, runs)
    out = []
    for first, last in runs:
        for start, stop, g, p in diff_runs(gt, pred, [(first, last)]):
            kind = KINDS[kinds[start]]
            bridges = False
            if kind == "spurious":
                left = start
                while left > first and pred[left - 1] == p:
                    left -= 1
                right = stop
                while right < last and pred[right] == p:
                    right += 1
                bridges = len(set(gt[left:right].tolist())) > 1
            out.append((start, stop, g, p, kind, bridges))
    return out


def kind_events(gt, pred, runs, table):
    """{kind: number of mistakes}: true segments with no right frame for missed, file segments
    with no right frame for spurious, disagreement runs for the boundary kinds."""
    correct = gt == pred
    events = {kind: 0 for kind in KINDS}
    for first, last in runs:
        for start, stop, _ in pm.segments(gt, [(first, last)]):
            if not correct[start:stop].any():
                events["missed"] += 1
        for start, stop, _ in pm.segments(pred, [(first, last)]):
            if not correct[start:stop].any():
                events["spurious"] += 1
    for _, _, _, _, kind, _ in table:
        if kind in ("late", "early", "other"):
            events[kind] += 1
    return events


def macro_f1_of(gt, pred, mask=None):
    if mask is not None:
        gt, pred = gt[mask], pred[mask]
    return pm.macro_f1(pm.scored_confusion(pm.confusion(gt, pred)))


def repair_costs(gt, pred, table, graded):
    """{kind: macroF1 gain when every run of that kind is replaced by the truth}."""
    base = macro_f1_of(gt, pred, graded)
    out = {}
    for kind in KINDS + ("late+early",):
        fixed = pred.copy()
        wanted = {"late", "early"} if kind == "late+early" else {kind}
        touched = False
        for start, stop, _, _, k, _ in table:
            if k in wanted:
                fixed[start:stop] = gt[start:stop]
                touched = True
        out[kind] = (macro_f1_of(gt, fixed, graded) - base) if touched else 0.0
    return out


def boundary_mask(gt, runs, margin):
    """Graded frames further than `margin` frames from every true transition."""
    keep = np.zeros(len(gt), dtype=bool)
    for first, last in runs:
        keep[first:last] = True
    for frame, _, _ in pm.transitions(gt, runs):
        keep[max(0, frame - margin):frame + margin] = False
    return keep


# -- segment and transition views ---------------------------------------------------------------

def segment_rows(gt, pred, runs):
    """Per true segment: label, length, head/tail/interior wrong frames, file's labels inside."""
    rows = []
    for index, (start, stop, g) in enumerate(pm.segments(gt, runs), 1):
        window = pred[start:stop]
        wrong = window != g
        n = len(window)
        if wrong.all():
            head, tail, interior = n, 0, 0
        else:
            head = int(np.argmax(~wrong))
            tail = int(np.argmax(~wrong[::-1]))
            interior = int(wrong.sum()) - head - tail
        shares = {NAME[k]: float(np.mean(window == k)) for k in (pm.H, pm.A, pm.D, pm.X)}
        n_pred = len(pm.segments(window, [(0, n)]))
        rows.append({"index": index, "start": start, "stop": stop, "label": NAME[g],
                     "frames": n, "head": head, "tail": tail, "interior": interior,
                     "n_pred_segments": n_pred, "shares": shares})
    return rows


def transition_rows(gt, pred, runs, tol):
    """Per true transition: latency of the matched file transition within +-tol, or None."""
    filled = pm.fill_abstentions(pred, runs)
    gt_trans = pm.transitions(gt, runs)
    pred_trans = pm.transitions(filled, runs)
    pairs, missed, spurious = pm.match_transitions(gt_trans, pred_trans, tol)
    latency = {g: p[0] - g[0] for g, p in pairs}
    rows = [(frame, a, b, latency.get((frame, a, b))) for frame, a, b in gt_trans]
    return rows, spurious


# -- the timeline strip ---------------------------------------------------------------------------

def majority(codes):
    counts = np.bincount(codes, minlength=pm.N_LABELS)
    return int(np.argmax(counts))


def timeline(gt, pred, first, last, res_s, per_line_s):
    """Lines of three strips: truth, file, and where they differ (file letter in lower case
    where at least half the cell is wrong, '.' where some of it is)."""
    cell = max(1, int(round(res_s * FPS)))
    per_line = max(1, int(round(per_line_s / res_s)))
    cells = []
    for start in range(first, last, cell):
        stop = min(last, start + cell)
        g, p = gt[start:stop], pred[start:stop]
        wrong = float(np.mean(g != p))
        pg, pp = majority(g), majority(p)
        diff = " " if wrong == 0 else (NAME[pp].lower() if wrong >= 0.5 else ".")
        cells.append((start, NAME[pg], NAME[pp], diff))
    lines = []
    for i in range(0, len(cells), per_line):
        chunk = cells[i:i + per_line]
        ruler = "".join("|" if (j * res_s) % 10 == 0 else " " for j in range(len(chunk)))
        lines.append("{:>8}  {}".format("", ruler))
        lines.append("{:>8}  {}  truth".format(clock(chunk[0][0]), "".join(c[1] for c in chunk)))
        lines.append("{:>8}  {}  file".format("", "".join(c[2] for c in chunk)))
        lines.append("{:>8}  {}  differs".format("", "".join(c[3] for c in chunk)))
    return lines


# -- printing -------------------------------------------------------------------------------------

def report(name, gt, pred, runs, args):
    graded = np.zeros(len(gt), dtype=bool)
    for first, last in runs:
        graded[first:last] = True
    cm3 = pm.scored_confusion(pm.confusion(gt, pred))
    n_graded = int(cm3.sum())
    table = error_table(gt, pred, runs)
    n_wrong = sum(stop - start for start, stop, _, _, _, _ in table)

    print("=" * 96)
    print("{}: {} graded frames in {}, {} wrong ({:.1%}), in {} runs".format(
        name, n_graded, ", ".join("{}..{}".format(a, b - 1) for a, b in runs), n_wrong,
        n_wrong / float(n_graded), len(table)))
    print("=" * 96)

    print("\nConfusion (rows truth, columns file; count and share of the row)")
    print("{:>6} {:>14} {:>14} {:>14} {:>14}".format("", "H", "A", "D", "X"))
    for k in range(pm.N_SCORED):
        row = cm3[k]
        total = float(row.sum()) or 1.0
        print("{:>6} ".format(NAME[k]) + " ".join(
            "{:>7} {:>6.1%}".format(int(v), v / total) for v in row))
    print("\nPer class                precision  recall      F1   support")
    for k in range(pm.N_SCORED):
        precision, recall, f1 = pm.per_class_prf(cm3, k)
        print("{:>6}                   {:>9.3f} {:>7.3f} {:>7.3f} {:>9}".format(
            NAME[k], precision, recall, f1, int(cm3[k].sum())))
    strict, both = pm.team_accuracy(cm3)
    print("macroF1 {:.4f}   accuracy {:.4f}   kappa {:.4f}   alive/dead F1 {:.4f}   "
          "team right where both alive {:.4f}".format(
              pm.macro_f1(cm3), pm.accuracy(cm3), pm.cohens_kappa(cm3), pm.alive_dead_f1(cm3),
              both))

    print("\nWrong frames by kind of mistake (repair = macroF1 gain if the truth replaced "
          "exactly these frames)")
    costs = repair_costs(gt, pred, table, graded)
    events = kind_events(gt, pred, runs, table)
    print("{:<11} {:>5} {:>7} {:>7} {:>8}   {}".format(
        "kind", "n", "frames", "share", "repair", "what it is"))
    for kind in KINDS:
        rows = [r for r in table if r[4] == kind]
        frames = sum(stop - start for start, stop, _, _, _, _ in rows)
        if not rows:
            continue
        print("{:<11} {:>5} {:>7} {:>6.1%} {:>+8.4f}   {}".format(
            kind, events[kind], frames, frames / float(n_wrong or 1), costs[kind],
            KIND_TEXT[kind]))
    print("{:<11} {:>5} {:>7} {:>6} {:>+8.4f}   both boundary kinds together".format(
        "late+early", "", "", "", costs["late+early"]))

    print("\nWrong frames by label pair (truth>file)")
    pairs = {}
    for start, stop, g, p, _, _ in table:
        pairs[pair_name(g, p)] = pairs.get(pair_name(g, p), 0) + stop - start
    print("  " + "   ".join("{} {} ({:.1%})".format(k, v, v / float(n_wrong or 1))
                            for k, v in sorted(pairs.items(), key=lambda kv: -kv[1])))

    print("\nmacroF1 when frames within a margin of every true boundary are left out")
    print("  " + "   ".join("+-{} f: {:.4f}".format(m, macro_f1_of(gt, pred, boundary_mask(
        gt, runs, m))) for m in (0, 12, 25, 50, 75)))

    lengths = np.array([stop - start for start, stop, _, _, _, _ in table] or [0])
    print("\nRun lengths: median {:.0f} f, mean {:.0f} f, longest {} f; frames in runs over "
          "1 s {:.1%}, over 2 s {:.1%}".format(
              np.median(lengths), lengths.mean(), lengths.max(),
              lengths[lengths > FPS].sum() / float(n_wrong or 1),
              lengths[lengths > 2 * FPS].sum() / float(n_wrong or 1)))

    print("\nLongest {} disagreement runs".format(args.top))
    print("{:>8} {:>8} {:>7} {:>6}  {}".format("start", "clock", "frames", "pair", "kind"))
    for start, stop, g, p, kind, bridges in sorted(table, key=lambda r: r[0] - r[1])[:args.top]:
        note = ""
        if kind == "spurious":
            note = " (bridges a true boundary)" if bridges else " (inside a true segment)"
        print("{:>8} {:>8} {:>7} {:>6}  {}{}".format(
            start, clock(start), stop - start, pair_name(g, p), kind, note))

    print("\nThe truth segment by segment (wrong frames at the head, the tail, and inside)")
    print("{:>3} {:>8} {:>8} {:>6} {:>5} {:>5} {:>5} {:>6} {:>5}  {}".format(
        "#", "start", "clock", "s", "lbl", "head", "tail", "inside", "segs", "file H/A/D/X"))
    for r in segment_rows(gt, pred, runs):
        shares = r["shares"]
        print("{:>3} {:>8} {:>8} {:>6.1f} {:>5} {:>5} {:>5} {:>6} {:>5}  {}".format(
            r["index"], r["start"], clock(r["start"]), r["frames"] / float(FPS), r["label"],
            r["head"], r["tail"], r["interior"], r["n_pred_segments"],
            "/".join("{:.0%}".format(shares[k]) for k in "HADX")))

    rows, spurious = transition_rows(gt, pred, runs, args.tol)
    lat = [r[3] for r in rows if r[3] is not None]
    print("\nTrue transitions and the file's latency (matched one-to-one on the destination "
          "within +-{} frames)".format(args.tol))
    print("{:>8} {:>8} {:>6} {:>9}".format("frame", "clock", "move", "latency"))
    for frame, a, b, latency in rows:
        print("{:>8} {:>8} {:>6} {:>9}".format(
            frame, clock(frame), "{}>{}".format(NAME[a], NAME[b]),
            "none" if latency is None else "{:+d}".format(latency)))
    if lat:
        lat = np.array(lat)
        print("matched {} of {}: median {:+.0f} f, mean {:+.1f} f, mean |latency| {:.1f} f; "
              "late {} early {} exact {}".format(
                  len(lat), len(rows), np.median(lat), lat.mean(), np.abs(lat).mean(),
                  int((lat > 0).sum()), int((lat < 0).sum()), int((lat == 0).sum())))
        print("matched within the scorer's tolerances: {} at +-12 f, {} at +-25 f".format(
            sum(r[3] is not None and abs(r[3]) <= 12 for r in rows),
            sum(r[3] is not None and abs(r[3]) <= 25 for r in rows)))
    if spurious:
        print("file transitions with no true one within +-{} f: {}".format(
            args.tol, ", ".join("{} {}>{}".format(f, NAME[a], NAME[b]) for f, a, b in spurious)))

    print("\nTimeline ({} s per character, majority label per cell; lower case where at least "
          "half the cell is wrong, '.' where some of it is)".format(args.res))
    for first, last in runs:
        for line in timeline(gt, pred, first, last, args.res, args.line):
            print(line)
    print()


def comparison(names, gt, preds, runs, graded):
    print("=" * 96)
    print("Side by side (frame level, graded window only)")
    print("=" * 96)
    header = ["", "macroF1", "acc", "F1 H", "F1 A", "F1 D", "wrong", "runs", "late", "early",
              "missed", "spur.", "lat med"]
    print("{:<18}".format(header[0]) + "".join("{:>8}".format(h) for h in header[1:]))
    print("{:<18}{}".format("", "(wrong frames by kind; lat med = median latency of true "
                             "transitions matched within +-25 f)"))
    for name, pred in zip(names, preds):
        cm3 = pm.scored_confusion(pm.confusion(gt, pred))
        table = error_table(gt, pred, runs)
        by_kind = {k: sum(stop - start for start, stop, _, _, kk, _ in table if kk == k)
                   for k in KINDS}
        rows, _ = transition_rows(gt, pred, runs, 25)
        lat = [r[3] for r in rows if r[3] is not None]
        values = [pm.macro_f1(cm3), pm.accuracy(cm3)] + [
            pm.per_class_prf(cm3, k)[2] for k in range(pm.N_SCORED)]
        print("{:<18}".format(name[:18]) + "".join("{:>8.4f}".format(v) for v in values)
              + "{:>8}{:>8}".format(sum(by_kind.values()), len(table))
              + "".join("{:>8}".format(by_kind[k]) for k in
                        ("late", "early", "missed", "spurious"))
              + "{:>8}".format("{:+.0f}".format(np.median(lat)) if lat else "none"))
    print()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="result CSVs; the first is the reference")
    parser.add_argument("--gt", default=GT_PUBLIC, help="frame,label ground truth")
    parser.add_argument("--top", type=int, default=10, help="longest runs to list")
    parser.add_argument("--tol", type=int, default=125,
                        help="matching tolerance for the latency table, in frames")
    parser.add_argument("--res", type=float, default=1.0, help="timeline seconds per character")
    parser.add_argument("--line", type=float, default=60.0, help="timeline seconds per line")
    args = parser.parse_args(argv)

    gt = load(args.gt)
    runs = pm.valid_runs(gt)
    if not runs:
        raise SystemExit("the ground truth has no graded frames")
    graded = np.zeros(len(gt), dtype=bool)
    for first, last in runs:
        graded[first:last] = True

    names, preds = [], []
    for path in args.files:
        pred = load(path)
        if len(pred) != len(gt):
            raise SystemExit("{}: {} rows, ground truth has {}".format(path, len(pred), len(gt)))
        names.append(os.path.splitext(os.path.basename(path))[0])
        preds.append(pred)

    for name, pred in zip(names, preds):
        report(name, gt, pred, runs, args)
    if len(preds) > 1:
        comparison(names, gt, preds, runs, graded)


if __name__ == "__main__":
    main()
