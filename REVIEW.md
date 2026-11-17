# REVIEW.md: an outside review of the Phase 1 solution (2026-11-06)

Written for the team (miha and two teammates). Read-only: no solution code was changed. What was
run: `make test` (220 pass), `validate_submission.py` and `run_evaluation.py --bootstrap 1000` on
the committed result files, and a segment-level analysis of the submitted file on the public
window. That analysis read the public ground truth, the committed `baseline_raw.csv` (its `D`
marks frames where the ATD had no ball) and our own output restricted to the public-window
frames. Nothing under `data/` was read. Every proxy-league number quoted below comes from
`docs/EXPERIMENTS.md` and `docs/DECISIONS.md`; the decision numbers (D-numbers) refer to that log.

**Calendar.** Today is Fri 6 Nov. Features until about Thu 12 Nov, final version Mon 16 Nov,
deadline Tue 17 Nov: ten days.

**The short version.** The pipeline is correct, reproducible on one machine, and far better than
the baselines (public window 0.5818 against 0.3256 for the published floor; proxy league 0.486
against 0.284 on the same truth). It has plateaued because the decision procedure, not the
evidence, is now the bottleneck: the three largest measured gains (the learned dead-ball model,
+0.070; the TV cut-away cue, +0.030; the frame model inside gaps, +0.013) are all built, all ADOPT
on the proxy league, and all held back by label-free BAR-ATH checks whose premises are themselves
unverified. The single biggest loss category on the public window and on the proxy is the same:
**false and misplaced stoppages**. Dead ball is where the next 0.05 is. Short possessions with the
ball visible are the second loss and the harder one.

---

## 1. The metric, in my own words

`spellF1` scores possessions as objects. Inside each half, after every `X` is forward-filled
(back-filled at a half's head), the maximal runs of `H` and of `A` are the predicted possessions.
`D` runs are not possessions; they only split them. Predicted and true possessions of the same team
are matched one-to-one, greedily in descending temporal IoU (ties broken deterministically), and
the F1 of that matching is averaged over IoU thresholds 0.3, 0.4, 0.5, 0.6, 0.7. `macroF1` over
`H`/`A`/`D` frames is the first tie-break; transition F1 and latency come after it.

What that rewards and punishes, beyond the obvious:

- **Dead ball is scored only through its effect on possessions.** A stoppage's exact edges barely
  matter when it sits between two long possessions. A stoppage that does not exist, placed inside
  a 40 s possession, is expensive: it turns one true positive into one true positive with a lower
  IoU plus one false positive. On the public window two such 2.5 s false stoppages cost 0.063.
- **A missed short possession is a double loss when it merges its neighbours.** Missing a 2.8 s
  `A` inside `H ... H` loses that `A` and merges the two `H` possessions into one prediction, which
  can match only one of them.
- **A found-but-late short possession is a double penalty.** A 0.8 s possession predicted 14 frames
  late has IoU 0.11: it counts as a miss and the prediction counts as spurious. Below about 2 s,
  the boundaries must be within a few frames to score at all.
- **The 0.6 and 0.7 thresholds are the latency thresholds.** On perfect labels a uniform 8-frame
  lag scores 0.893 and 1 s scores 0.654. On the public window the submitted file scores 0.667 at
  IoU 0.3 to 0.5 but 0.545 at 0.6 and 0.364 at 0.7: a third of its loss is boundary placement on
  possessions it otherwise finds.
- **One extra predicted possession pays only if its chance of matching exceeds about F1/2.** At
  the current level that is about 0.25 to 0.3; a merge, though, costs the possession on both sides.
- **`X` is "repeat my last label" on the ranking metric.** Free inside a possession, harmful across
  a hand-over, and a miss on the tie-break. The submission writes no `X` on graded frames; that is
  right.
- **Noise.** About 0.04 on the full match. On the public window the 5-95% bootstrap interval of
  the submitted file is 0.40 to 0.75. Nothing on the window separates two versions of the method.

## 2. What the solution does

`solution/run.py` writes `touch_c3_seen_w1` (D23). Step by step:

1. **Load** (`solution/sources/atd.py`): the tactical EPTS file parsed with the standard library,
   positions scaled to metres, slot teams from the metadata, halves = the published graded spans.
2. **Priors fitted at run time** on the five IDSSE dev matches (never the two held out, D3):
   log-normal duration priors per state class with a 1st-percentile minimum; the nearest-player
   agreement; the collective-speed likelihood under alive and dead play, with speed read as a
   within-match rank (D9); the touch holder's agreement; the spread-cue tables. All fitted on DFL
   first-touch labels (`PRIOR_LABELS=dfl`, D21).
3. **Evidence per frame:**
   - the team of the tracked player nearest the ball;
   - the touch engine's holder (`solution/touch.py`, D10/D12): ball cleaned of teleports and
     out-of-pitch frames, touches where the velocity vector changes by 3 m/s or more with a player
     within 3 m at a 0.4 s lag, control episodes, and a holder that changes at the other team's
     first touch (`touch_takes`), survives 1 s gaps and expires after 4 s;
   - the spread cue (`solution/gapteam.py`, D16): home outfield width minus away's, read on every
     frame at weight 0.5, with no say on dead ball;
   - the frame model (`solution/framemodel.py`, D23): a scikit-learn histogram GBM trained at run
     time on the dev matches degraded by the simulator (seed 101) with keep2 labels, 35 players-only
     team-swap-symmetric features, read at weight 1 only on frames where the ball is seen, with no
     say on dead ball;
   - the kickoff grammar (`solution/kickoff.py`, D14): each half is `D` until a formation hold of
     8 s or more ends with a speed rise, players only, erring early.
4. **Decode** (`solution/decoder.py`): an exact semi-Markov Viterbi over {H, A, D} at 5 Hz, block
   emissions summed from the terms above, duration priors, all transition costs zero (D8), blocks
   copied back to frames.
5. **Write**: `H`/`A`/`D` on every graded frame, `X` outside the halves; validated.

Two runs on BAR-ATH give the same sha256 (`f7459db0…bcf8358d`, confirmed on the committed
`c3_seen_w1.csv`), about 18 s each with caches.

**Where it differs from PLAN.md.**

- C5 boundary refinement is built (`touch.snap_transitions`) but unused: on the proxy it was worth
  +0.003 (D10). C6 self-training is closed (D24). C2b dead-ball cues are built and unused (D15).
- The decoder's transition costs were meant to be "tuned directly on spellF1". They are all zero,
  chosen on `semi_markov` with speed-only evidence and DFL truth on 2026-10-29 (D8), and have never
  been re-tuned on the current evidence stack or on keep2 truth. See section 5, item 2.
- The frame model is scikit-learn's HistGBM, not LightGBM. Fine; it is pinned.
- External data is IDSSE only. Metrica sample data (same vendor, the planned label-convention set)
  and SkillCorner were never added.
- E8 (Smart Tagging as evidence) and E7 (restart-team classifier) were never built. E11, E14 and
  E15 neither. Everything else in the backlog has a measured verdict.
- Priors and the frame model are fitted at run time from the 2.5 GB IDSSE download instead of
  shipped weights. Phase 4 job, still open; see section 6.
- `docs/TECHNICAL_REPORT.md` is the empty template.

## 3. Reproduction

| check | result |
|---|---|
| `make test` | 220 tests pass in 16 s |
| `validate_submission.py c3_seen_w1.csv` | OK; H 27.3%, A 17.4%, D 35.0%, X 20.3% (exactly the ungraded 40,462 frames); 332 possessions of mean 10.7 s |
| `run_evaluation.py --bootstrap 1000` | `c3_seen_w1` spellF1 **0.5818** (5-95% 0.400..0.752), macroF1 **0.8372**, 16 possessions of 15.2 s against 17 of 14.2 s, transition F1 0.34 at ±25 frames, mean latency +11.1 frames, share error 0.006 |
| sha256 of the committed file | `f7459db0…bcf8358d`, the D23 run |

Matches EXPERIMENTS section 4 exactly. The `our_team.csv` row on the board (0.4800, 28
possessions) is an older file of unrecorded method; delete it before the final run so the board
cannot be misread.

## 4. Error analysis on the public window

Seventeen possessions, so every number here is a description, not a measurement, and nothing in
this section was used to choose anything. The window is also the best-tracked stretch of the match
(ball seen on 88% of its frames against 59% over the match), so it under-represents ball gaps.

### 4.1 Segment view

| # | team | start | dur s | ball seen | best IoU | start / end offset (frames) | verdict |
|---|---|---|---|---|---|---|---|
| 1 | A | 11185 | 6.6 | 0.99 | 0.57 | +0 / +124 | recovered, fails 0.6 |
| 2 | H | 11351 | 8.1 | 0.98 | 0.07 | | **missed**: merged with #4 because #3 was missed |
| 3 | A | 11553 | 2.8 | 1.00 | 0.00 | | **missed** |
| 4 | H | 11624 | 41.6 | 1.00 | 0.67 | -149 / -240 | recovered, fails 0.7 (cut by a false stoppage) |
| 5 | A | 12665 | 7.4 | 0.89 | 0.54 | +35 / -51 | recovered, fails 0.6 |
| 6 | H | 12989 | 28.6 | 1.00 | 0.98 | +11 / +1 | recovered |
| 7 | A | 13704 | 3.2 | 1.00 | 0.93 | +1 / +5 | recovered |
| 8 | H | 13785 | 42.4 | 1.00 | 0.62 | +5 / -396 | recovered, fails 0.7 (split in 2 by a false stoppage) |
| 9 | A | 14846 | 20.0 | 1.00 | 0.92 | +29 / +14 | recovered |
| 10 | H | 15346 | 0.8 | 1.00 | 0.11 | +14 / +25 | **missed** (found, but 0.6 s late) |
| 11 | A | 15365 | 16.0 | 1.00 | 0.62 | +25 / +204 | recovered, fails 0.7 (ran through #12 and #13) |
| 12 | H | 15766 | 1.3 | 1.00 | 0.00 | | **missed** |
| 13 | A | 15799 | 3.6 | 1.00 | 0.16 | | **missed**: merged into #11 |
| 14 | A | 16400 | 35.7 | 0.59 | 0.97 | +15 / +12 | recovered |
| 15 | H | 17293 | 3.5 | 1.00 | 0.00 | | **missed**: decoded as D 3 s early |
| 16 | A | 17838 | 7.6 | 1.00 | 0.71 | +7 / -47 | recovered |
| 17 | H | 18389 | 11.8 | 1.00 | 0.74 | +76 / +0 | recovered |

Spurious predicted possessions (no match at IoU 0.5):

| predicted | length | what the truth holds under it |
|---|---|---|
| H 12485-12700 | 8.6 s | H 0.84, A 0.16: the second half of #4 after a false stoppage |
| H 12800-12880 | 3.2 s | A 0.64, D 0.36: a phantom hand-over just before a true stoppage |
| H 14515-14875 | 14.4 s | H 0.92: the second half of #8 after a false stoppage |
| H 15360-15390 | 1.2 s | A 0.83: #10 found 14 frames late and 11 frames too long |
| H 17980-18285 | 12.2 s | D 0.85, A 0.15: the stoppage at 18027 missed for 10 s, given to the wrong team |

Tally at IoU 0.5: 11 recovered, 6 missed, 5 spurious. Of the 6 misses, 5 are possessions of 3.6 s
or less and the sixth (#2, 8 s) is lost because the 2.8 s possession next to it was. **All six
misses are on frames with the ball seen.** Of the 5 spurious, 2 are fragments made by false
stoppages, 1 is a missed stoppage start, 1 is a phantom hand-over and 1 is a real 0.8 s possession
found 0.6 s late.

### 4.2 Boundaries

Every matched hand-over is late: +35, +1, +5, +29, +14, +25 frames (median 20, about 0.8 s).
Every matched restart (D to a team) is late: +11, +15, +7, +76. Stoppage starts: +29, +81, -75
(the early one swallowed #15), and one missed by 258 frames. The scorer reports mean latency
+11 frames; on the proxy's clean ball the touch engine placed hand-overs 1.4 frames early (section
5), and under the ball blur its mean is not recorded. Under D12 the blur is 0.26 s; D34 measured
BAR-ATH's at about 0.36 s. A systematic lag on BAR-ATH that the proxy does not show at 0.26 s is
the hypothesis to test (section 5, item 3). It is not to be fixed from this window.

### 4.3 By condition

| frames | n | true D share | predicted D share | frame accuracy | team accuracy, both alive |
|---|---|---|---|---|---|
| ball seen | 6,615 | 0.147 | 0.143 | 0.837 | 0.920 |
| ball missing | 885 | 0.558 | 0.525 | 0.945 | 0.949 |

Dead ball: 4 true stoppages (5.5, 20.4, 18.3, 14.5 s) against 6 predicted (2.4, 4.8, 2.6, 17.8,
21.6, 7.2 s). Two of the predicted ones contain no true dead frame (12425-12485, 14450-14515).
The D share is right; its placement is not. The same picture on the full match: the submission
makes 140 stoppages per 90 minutes of median 9.4 s on BAR-ATH, where the IDSSE truth has 97 of
median 19 s and the window extrapolates to about 72 of median 16 s.

In this window there is no tracking-gap problem to speak of: the gaps are mostly dead ball and the
team is right on 95% of both-alive gap frames. Identity switches are invisible to a per-frame
label comparison; the touch engine attributes by nearest tracked player, so a switch costs at most
one wrong touch.

### 4.4 What each category costs here (oracle repairs that read the truth; illustrative)

| repair on the window | spellF1 gain |
|---|---|
| remove the two false stoppages inside true possessions | +0.063 |
| perfect dead ball (edges, false and missed stoppages), teams as predicted | +0.076 |
| of which: truly dead frames made D (late starts, missed stoppage) | +0.012 |
| true labels on the frames of the 5 spurious possessions | +0.063 |
| snap matched hand-overs onto the true frame (±1.5 s) | +0.085 |
| write in the missed possessions under 4 s | +0.166 |
| write in all missed possessions | +0.229 |
| perfect | +0.418 |

The categories overlap, so they do not add. The proxy league's decomposition of the same
reference (keep2 truth, D12 stress, EXPERIMENTS sections 13 to 16) agrees on the order: a perfect
dead ball is worth +0.147 (false stoppages +0.046, start edges +0.043, end edges +0.038, missed
stoppages +0.017; throw-ins the largest type at +0.045); the true team on visible-ball frames
+0.19 (of which invented possessions +0.14); the true team inside gaps +0.045; hand-over timing
inside gaps +0.006. The window and the proxy disagree on one thing: latency (+0.085 here against a
+0.03 snapping ceiling there), which is the point of item 3 below.

## 5. Over-fitting assessment

**Tuned on the public window: nothing.** The log is explicit at every step (L4 discipline, D1), the
window was never used to pick a setting, and the one case where a method was chosen after a failed
check (`_seen` after D22, `_dms` after D26, the calibrated TV cue after D31) was chosen on
label-free aggregates and the proxy, with the rule written down first. This is better practice
than almost any hackathon entry and it should be said so in the technical report.

**Tuned on the proxy league (5 IDSSE matches, 1,455 controlled possessions):** about 10 settings
are actually selected on scores (decoder weight 1 of 4 tried; zero costs of 7; speed rank of 3;
touch-engine lag, threshold, radius and window of 30; spread weight and placement of 17; frame
model weight and placement of 6), out of roughly 45 constants in the submitted path, most of them
physical values set by reasoning and shown to sit on plateaus. About 180 settings were scored on
the proxy in total. The D4 bar (0.02, a paired interval and 2/3 wins) is a real guard against
garden-of-forking-paths, and the two held-out matches (D3) have never been scored, so the
over-fit to the five dev matches is unmeasured but bounded. Score them once in Phase 4.

**Where the five minutes flatter the ninety.**

- The window has the ball on 88% of frames; the match 59%. The window exercises the touch engine
  and the frame model, not the spread cue. Expect the full-match score well below 0.58: the
  smoothed baselines fall by 25 to 47% from window to match, and the proxy's reference sits at 0.486 (keep2)
  with a 0.26 s ball blur and 0.460 at 0.40 s. A full-match score of 0.40 to 0.48 is my guess.
- The kickoff grammar fires only in half 2 and is not tested by the window at all. Its BAR-ATH
  check (D14) is the only evidence; it looked right.
- The touch engine is tuned for a 0.26 s blur (D12) and BAR-ATH's is about 0.36 s (D34). At 0.40 s
  the same engine loses 0.027 on the proxy. Everything that reads touches is somewhat optimistic
  on the proxy.
- The frame model reads absolute positions and speeds learned on IDSSE. The adversarial check
  (D30) tells BAR-ATH's seen frames from the proxy's at AUC 0.97 (odd-one-out baseline 0.93): a
  real shift, driven by speed spread and players near the goal line. It is read only where the
  ball is seen, where its tag agreement rose on BAR-ATH. Moderate risk, already contained.
- The speed rank assumes a usual share of slow play (D9). BAR-ATH's decoded D share is 0.44,
  inside the IDSSE range. Fine.

**How to validate on the rest of the match without labels.** The team already does the right
things: Smart Tagging team agreement on decoded frames (0.73 in gaps, 0.87 seen), possession
counts and durations against the proxy truth, set-piece windows containing a decoded restart
(0.93 of 58), D share in the IDSSE range, byte-identity of two runs. Three things it does not do
yet and should:

1. **Stability** (PLAN L3): the possession count and D share of the submitted method under small
   perturbations of its own settings (±20% on weights, ±1 block on durations). A configuration
   whose count swings by 30% is fragile. Cheap, label-free, never done.
2. **Per-half consistency**: counts, D share and home share per half of BAR-ATH against each other
   and against the proxy. A method that behaves differently in the two halves has found something
   about the tracking, not about football.
3. **Stoppage plausibility**: stoppages per 90 and their median length against the IDSSE truth
   (97, 19 s) and against football common sense (a LaLiga match has roughly 40 to 50 throw-ins,
   25 to 30 free kicks, 15 to 20 goal kicks, about 10 corners). 140 stoppages of median 9 s is
   not a football match. This is the one check that would have ranked the dead-ball model above
   the reference; see item 1.

Splitting the public window into halves, or using it as held-out data, is not worth doing: 8
possessions per half cannot rank anything.

## 6. Ranked improvement plan

Ranked by expected full-match spellF1 gain times confidence, over the cost in sessions and compute.
Rules: every item is decided on the proxy league (D4) or by a pre-registered label-free check; the
public window is used only to confirm correctness afterwards. All of them are allowed under
RULES.md; the one borderline point is flagged where it occurs.

### 1. Decide the learned dead-ball model on the balance of evidence (expected +0.03 to +0.06; confidence medium-high; cost: one decision and one 40 s run)

`touch_c3_seen_w1_dm` (D26) is ADOPT against the submission on the decision run (+0.0702,
interval +0.057..+0.081, 5/5) and on every stress built in this project: DFL truth +0.048,
TEMPO=0.9 +0.070, plain +0.070, SPEED=0.9 +0.034, NOISE=0.1 +0.072, busy pre-kickoff +0.069. On
the proxy it removes false stoppages (126 to 100 per 90, median 11 to 17 s, truth 97 and 19 s).
On BAR-ATH it does exactly the same (140 to 97 per 90, median 9.4 to 18.1 s), keeps the D share
(0.440 to 0.444), keeps every set piece covered, raises tag agreement in gaps (0.734 to 0.771), and
leaves the gap home share alone. It failed one rule: the *direction* of change of seen D and gap D
on BAR-ATH is the opposite of the proxy's (seen D 0.208 to 0.188, gap D 0.777 to 0.818). That rule
was marked "adapted, miha to confirm" when it was written.

Why I think the rule's premise is weaker than the evidence against it:

- the rule assumes the submission errs the same way on BAR-ATH as on the proxy. On the proxy the
  reference under-calls seen D (0.223 against a truth of 0.256). On the public window the truth's
  seen-D share is 0.147 and the reference's 0.143, in the best-tracked stretch where more dead
  ball has a ball. There is no label-free reason to expect BAR-ATH's seen D to need raising;
- the two things the window and the proxy agree on, too many short stoppages and false stoppages
  being the largest single loss, are exactly what the model fixes, and the fix is visible in
  BAR-ATH's aggregates;
- the adversarial check (D30) found the seen-frame shift is not a simulator knob, so no proxy
  variant can settle this. It is a judgment call, and it should be made as one.

What to do: write D36 before running anything. Add the stoppage-plausibility check (section 5,
item 3) as a criterion, re-weigh D26's rule 4 as a warning rather than a veto, and choose between
`_dm` (larger gain, moves D into gaps on BAR-ATH) and `_dms` (half the gain, +0.036 ADOPT on
every run but SPEED=0.9, gaps untouched, D share 0.426, set pieces 0.95, tag agreement up,
possessions 268 per 90). My pick is `_dm`: its gap move is towards the proxy truth's gap-D of
0.716 only on the proxy, but the BAR-ATH window says gaps are mostly dead ball (0.56) and the
model's gap D (0.818) is still inside anything plausible. If the team wants the conservative
choice, `_dms` keeps two thirds of the case and none of the gap question. D26 and D30 forbid a
*third variant* chosen to pass; they do not forbid re-deciding the two that exist. Phase 4's
hold-out scoring (D3) then measures the choice once; write the reading rule first.

Risk: if BAR-ATH's seen-ball play really is read as "less dead" for a reason that is wrong, the
model would lose some of the +0.04 to +0.05 of seen-D edges while keeping the +0.046 of false
stoppages. Even that case is roughly neutral. Verify without labels: after switching, the
window correctness check must show no new stoppage inside possessions #4, #6, #8, #9 and #14.

### 2. Re-tune the decoder's dead-ball cost and minimum stoppage length on the current stack and on keep2 truth (expected +0.01 to +0.03; confidence medium; cost: one session, 8 proxy runs)

The zero costs were chosen on `semi_markov` (speed-only evidence, DFL truth, D8, 2026-10-29). The
evidence stack has gained four terms since, the truth changed to keep2 (fewer, longer possessions,
D21), and false stoppages are now the largest loss on both the proxy (+0.046, 183 per seed of
median 3.8 s) and the window (+0.063, two of 2.5 s). A `dead` cost of 1 to 4 nats, a `restart`
cost of 0 to 2, and the dead-ball minimum duration at the 1st against the 5th percentile: a
grid of about 8, scored against whatever item 1 leaves as the reference, under D4, DFL not
WORSE. Risk: DFL has true stoppages under 2 s (7% of stretches) that a cost would miss; the proxy
prices that. Together with item 1 this is the dead-ball track; do item 1 first so the grid is
tuned on the right reference.

### 3. Measure and correct the systematic boundary lag; move D12's blur to the measured 0.36 s (expected +0.01 to +0.03; confidence medium; cost: one session, 6 proxy runs)

On the window every matched hand-over and restart is late (median +20 and +13 frames). A uniform
8-frame lag costs 0.107 on perfect labels and the 0.6 and 0.7 thresholds are where the submission
loses a third of its window score. The proxy's mean hand-over latency under the ball blur is not
recorded. Run `make proxy METHODS=touch_c3_seen_w1 BOUNDARIES=1` at `BALL_SMOOTH` 0.26 and 0.36
(with D12's noise and jumps). If the mean latency is positive and grows with the blur, a constant
shift of decoded team changes (H to A, A to H, D to team) earlier by that mean, set on the proxy
at 0.36 s, is a calibration fix the proxy can decide (D4). Setting the shift from the window's
+11 frames would be tuning on the window and is not allowed; the window may only confirm
afterwards that the sign was right.

In the same session, decide D34: move the decision stress to `BALL_SMOOTH=0.36` for every method,
re-run the reference and the registered candidates there, and re-tune the touch engine's `lag_s`,
`touch_m` and `touch_dv` at that blur with a small grid (the D12 grid was 11 settings at 0.26 s).
Section 21 shows what is at stake: the reference loses 0.027 at 0.40 s and a candidate that was
0.0009 under the bar at 0.26 s is ADOPT at 0.40 s. The blur was measured from aggregates with a
statistic D12 was not fitted on, so this is a correction, not a rescue, and D34 says it must be
done for all methods at once.

### 4. Stack the three sub-bar 5/5 gains as one method (expected +0.01 to +0.02; confidence medium-low; cost: one proxy run plus one BAR-ATH check)

Priors fitted on keep2 labels (+0.009 to +0.011 on every keep2 run, 5/5, D25), the broadcast ball
merge (+0.004 to +0.007, 5/5, D27) and the player-beyond-a-line dead-ball cue (+0.005 to +0.008,
5/5, D15) were each rejected by the 0.02 bar alone. Register one method with all three on top of
item 1's reference and score it under D4; a stack of independent small gains is a legitimate
candidate and the bar still applies. Risk: keep2 priors lowered in-gap tag agreement on BAR-ATH
(0.721 to 0.676, D21) before the frame model existed; re-check. The merge makes `run.py` depend on
the broadcast alignment cache, which then has to be built inside `run.py` (section 7).

### 5. Check the Metrica sample data for in-vendor ball-state and possession channels (expected: changes the truth, not the score directly; cost: one short session)

PLAN section 2 listed the three Metrica sample matches as the label-convention set: same vendor,
same EPTS format, and game 3's corrected tracking. I do not know whether that export carries a
ball-status or owning-team channel (BAR-ATH's uncorrected one does not, FINDINGS A1). If it
does, it is the only labelled match in the proxy that uses Metrica's own definition of "dead" and
"owning", the question CHALLENGE.md says matters most, and it would settle whether keep2's 2 s
and `touch_takes` match the vendor's semantics. If it does not, nothing is lost. Allowed: other
games, public data.

### 6. Smart Tagging set pieces as restart anchors only (expected +0.005 to +0.015; confidence low; cost: two sessions)

E8 was never built. Team phases as evidence would consume the project's only label-free team
check and cannot be decided on the proxy (IDSSE has no tags), so I would not build that part. The
58 SET PIECES windows are different: a restart lies inside 0.93 of them already, so their value is
the *start* of the stoppage before the restart, which is the +0.043 start-edge category. A
synthetic tag on the proxy, imitated from the truth at BAR-ATH's measured window shape (35 s,
restart inside), could decide a term that raises D before a tagged restart. The TV cue (D31) is the
template and the warning: its synthetic imitation passed on the proxy and failed on BAR-ATH's
in-gap team. Keep the term off the team question (`team_dead` max style) and it cannot repeat that
failure. Low priority against items 1 to 3.

### 7. Not worth the remaining days

- **MBR decoding, merging low-support possessions** (D28): closed with an oracle that loses.
- **Self-training** (D24): under 0.01 even with true labels.
- **Whistle** (D26, D33): +0.012 at best for in-play stoppages, +0.004 for the half ends.
- **Video ball detector** (D29): coverage is worth +0.011 to +0.022 at best; the value is a sharper
  ball, which needs a tracker nobody will build in ten days.
- **Broadcast cut-away cue and ball merge under it** (D31, D32): +0.030 on the proxy, failed
  BAR-ATH twice on the in-gap team; D32 forbids another variant tuned against those aggregates.
- **Control model on touches** (D35): below the bar and WORSE on DFL; its false drops land on the
  real hand-overs.
- **Restart-team classifier** (E7): the team after a restart is already right on 0.94 of gap
  frames with a restart change (section 9); the whole category is worth +0.001.
- **Temporal network, set transformer, ensembles** (E11, E14, E15): no time, and the gap model
  they would feed is blocked by the in-gap check.
- **Hand-over moment events, snapping to touches** (D17 to D19): ceilings under 0.02.
- **End-of-half walk-off**: +0.005, closed by miha (D33). Polish only.

### Assets, and whether they are underused

Tactical ATD players: used heavily, rightly. Tactical ball: used through the touch engine; its
blur is the limit, not its coverage (D29). Smart Tagging: validation only; item 6 is the one safe
use as evidence. Broadcast: measured and closed. Homography: unused, and correctly so, since the
ATD is already in pitch coordinates; its only remaining use is an internal video explorer. Audio:
measured and closed. External data: IDSSE only; item 5 is the gap. The honest position is that
the assets have been measured and the remaining value is in decisions and calibration, not in
new inputs.

## 7. Must-do before submission (score-neutral, disqualification-relevant)

1. **Freeze the fitted priors, the frame model and (if adopted) the dead-ball model into
   committed files**, so `run.py` runs from a clean clone with the data bundle only, without the
   2.5 GB download and without training at start-up. Keep the deterministic rebuild script.
   Round the model probabilities (PLAN said 1e-4) before decoding; `framemodel.log_odds` does not
   round today. Set the thread count explicitly (`OMP_NUM_THREADS=1`) and verify that the HistGBM
   predictions are identical across thread counts; I would not assume it.
2. **Byte-identity on a second machine and in a Linux container** (PLAN Phase 4). Two runs on one
   Mac is what exists.
3. **Score the two held-out IDSSE matches once** (D3), with the reference and the item-1
   candidates, and record the gap between dev and hold-out as the over-fit estimate in the report.
4. **Write the technical report** from the empty template: approach; assets used and dropped
   (broadcast, audio, Smart Tagging as validation); training data (IDSSE, CC-BY 4.0, 5 matches, DFL
   labels relabelled to controlled possession); possession rules (keep2, first-touch hand-over,
   loose ball stays, kickoff grammar); limitations; reproduction. Decide PLAN question 10 (may
   aggregate figures from the data appear in it) and strip `notebooks/eda/FINDINGS.md`'s raw block
   if the answer is no.
5. **Make the repo public without the data**: `git ls-files data/ external_data/ cache/` must be
   empty; `evaluation/results/*.csv` except the baselines is gitignored, keep it so. Delete the
   stale `our_team.csv`. Name the CSV and the PDF after the team. One Google account that works.
6. **Final public-window correctness check** on the exact file uploaded, with `--bootstrap 1000`,
   logged in EXPERIMENTS section 4 as correctness only.

## 8. Day by day to the deadline

| day | work | decided by |
|---|---|---|
| Sat 7 Nov | Item 1: D36 written, `_dm` and `_dms` re-weighed with the stoppage-plausibility criterion; miha decides; `run.py` switches or not. Item 2 grid on the chosen reference. | D4 on the proxy; label-free BAR-ATH check |
| Sun 8 Nov | Item 3: latency under 0.26 and 0.36 s blur; D34 decision; touch-engine re-tune at the measured blur; a constant shift if the proxy shows one. | D4 at `BALL_SMOOTH=0.36` |
| Mon 9 Nov | Item 4 stack; item 5 Metrica sample check. | D4; L2 description only |
| Tue 10 Nov | **Safety net (PLAN Phase 3):** freeze priors and models into files, `run.py` without the download, two-machine hash check, a valid frozen file in hand. | sha256 |
| Wed 11 to Thu 12 Nov | Item 6 if items 1 to 3 are done; BAR-ATH checks of everything adopted; stability and per-half checks (section 5). | pre-registered rules |
| Fri 13 Nov | **Feature freeze.** Hold-out scoring once (D3). Technical report draft. | D3 reading rule written first |
| Sat 14 to Sun 15 Nov | Linux container reproduction, public-window correctness, report final, repo cleaned. | hashes match |
| Mon 16 Nov | Repo public, form submitted with CSV, URL and PDF. | checklist in TEAM.md |
| Tue 17 Nov | Buffer. Do not touch the method. | |

## 9. The three things I would do first

1. **Re-decide the dead-ball model today (item 1).** It is the largest measured gain in the
   project, built, tested on every stress, and blocked by one adapted rule whose premise the
   public window does not support. Write the criteria, decide, switch, check the window for
   correctness only.
2. **Re-tune the decoder's dead-ball cost on the current stack and keep2 truth (item 2).** Eight
   proxy runs against the largest loss category, on a knob last set with speed-only evidence and
   the wrong truth.
3. **Measure the boundary lag under the measured blur and move D12 to it (item 3).** The window's
   boundaries are all late; the 0.6 and 0.7 thresholds are a third of its loss; the proxy can
   decide a constant shift without the window ever being used to set it.

Then freeze. Items 4 to 6 only if the first three are done by Tuesday.
