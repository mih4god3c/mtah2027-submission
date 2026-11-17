"""Controlled-possession relabelling of first-touch possession labels (D20).

CHALLENGE.md ("Possession rules") says possession changes only when the other team gains
*controlled* possession. A simple touch, a deflection that does not intercept, or an aerial duel
won without control is not a change, and a contested or loose ball stays with the last team. DFL's
`BallPossession`, which the proxy league scores against, flips at the first touch instead (D10),
so its truth holds short "possessions" the task's truth does not have.

This module turns first-touch labels into controlled-possession labels. The rule comes from
the definition's own proxy ("a player keeps the ball for a few seconds"), with a round value
fixed before any measurement and never fitted to the public window:

- A team run (a maximal run of H or of A) shorter than `keep_s` is a touch, not control, unless
  `confirm(start, stop, code)` says the team showed control in it (for example a completed pass,
  `pass_confirm`).
- An unconfirmed short run is given to the team that had the ball before it, and merges with its
  neighbours. Runs are absorbed shortest first, so a single touch inside the other team's spell
  goes before the spell it interrupts is judged: `H.. A1.5 H0.3 A4` becomes H then A from the
  first A frame (the H touch did not stop A's control), not A from the 4 s run.
- The first team run of a half and the first after a stoppage are never absorbed: "whenever
  play restarts after a stoppage, there is a clear possessor".
- Dead ball (D) stays exactly as the input marks it, and X too.

Deterministic, numpy plus the standard library, labels in and labels out.
"""

import heapq

import numpy as np

from solution.match import pm

KEEP_S = 2.0          # "keeps the ball for a few seconds": 2 s, the low end of "a few"
PASS_TOL_S = 1.0      # DFL event times are manual, about a second off (sources/idsse.restarts)
VARIANTS = ("dfl", "keep2", "keep2pass")


def controlled(labels, fps, keep_s=KEEP_S, confirm=None):
    """Scorer codes with every unconfirmed team run under keep_s given to the previous team."""
    out = labels.copy()
    keep = int(round(keep_s * fps))
    for half in pm.valid_runs(labels):
        segs = [list(s) for s in pm.segments(labels, [half])]
        n = len(segs)
        prev = list(range(-1, n - 1))
        nxt = list(range(1, n + 1))
        alive = [True] * n                     # segment still exists (not merged away)

        def exempt(i):
            return prev[i] == -1 or segs[prev[i]][2] == pm.D

        def short(i):
            start, stop, code = segs[i]
            if code not in (pm.H, pm.A) or exempt(i) or stop - start >= keep:
                return False
            return not (confirm and confirm(start, stop, code))

        heap = [(segs[i][1] - segs[i][0], segs[i][0], i) for i in range(n) if short(i)]
        heapq.heapify(heap)
        while heap:
            length, start, i = heapq.heappop(heap)
            if not alive[i] or segs[i][0] != start or segs[i][1] - start != length or \
                    not short(i):
                continue                       # stale entry
            p = prev[i]                        # an H/A segment of the other team (not exempt)
            # Give run i to p, then fold i and a same-team successor into p.
            segs[p][1] = segs[i][1]
            alive[i] = False
            nxt[p] = nxt[i]
            if nxt[i] < n:
                prev[nxt[i]] = p
            j = nxt[p]
            if j < n and segs[j][2] == segs[p][2]:
                segs[p][1] = segs[j][1]
                alive[j] = False
                nxt[p] = nxt[j]
                if nxt[j] < n:
                    prev[nxt[j]] = p
            if short(p):
                heapq.heappush(heap, (segs[p][1] - segs[p][0], segs[p][0], p))
        for i in range(n):
            if alive[i]:
                start, stop, code = segs[i]
                out[start:stop] = code
    return out


def pass_confirm(passes, fps, tol_s=PASS_TOL_S):
    """confirm() for controlled(): the run holds a completed pass by its team, +- tol_s.

    `passes` is (frames, codes) of completed passes, codes H or A (pm codes).
    """
    frames, codes = (np.asarray(v) for v in passes)
    order = np.argsort(frames, kind="stable")
    frames, codes = frames[order], codes[order]
    by_team = {code: frames[codes == code] for code in (pm.H, pm.A)}
    tol = int(round(tol_s * fps))

    def confirm(start, stop, code):
        team = by_team[code]
        i = np.searchsorted(team, start - tol, side="left")
        return bool(i < len(team) and team[i] < stop + tol)
    return confirm


def stats(labels, fps):
    """Aggregates of one label array, in the units of EXPERIMENTS section 4 (L2).

    possessions, graded and alive seconds, possession lengths (s), and direct H<->A changes.
    """
    runs = pm.valid_runs(labels)
    spells = pm.possession_spells(labels, runs)
    direct = 0
    for start, stop in runs:
        w = labels[start:stop]
        change = (w[1:] != w[:-1]) & (w[1:] <= pm.A) & (w[:-1] <= pm.A)
        direct += int(np.sum(change))
    graded = sum(stop - start for start, stop in runs)
    alive = int(np.sum((labels == pm.H) | (labels == pm.A)))
    return {"n": len(spells), "graded_s": graded / fps, "alive_s": alive / fps,
            "lengths": np.array([(b - a) / fps for a, b, _ in spells]), "direct": direct}


def summary(parts):
    """Pooled L2 row from stats() dicts: per 90, short shares, median, direct per alive min."""
    lengths = np.concatenate([p["lengths"] for p in parts]) if parts else np.array([])
    graded = sum(p["graded_s"] for p in parts)
    alive = sum(p["alive_s"] for p in parts)
    n = sum(p["n"] for p in parts)
    return {"n": n, "per90": n / graded * 5400.0,
            "under1": float(np.mean(lengths < 1.0)), "under2": float(np.mean(lengths < 2.0)),
            "median": float(np.median(lengths)), "mean": float(np.mean(lengths)),
            "direct_per_alive_min": sum(p["direct"] for p in parts) / alive * 60.0}
