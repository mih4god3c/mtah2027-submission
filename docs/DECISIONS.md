# Decisions

Numbered, never renumbered. Each says what was decided, why, and what evidence it rests on, so
nobody (person or agent) undoes it by accident. To reverse one, add a new decision that
supersedes it rather than editing the old one.

## D1. Decide on a proxy league of external matches, not on the public 5 minutes (2026-10-28)

**Decision.** Every "does this help?" question is answered by `make proxy`: external matches
with published per-frame possession labels, degraded to lose the ball like our ATD, scored by
the official `possession_metrics` code. The public window is a correctness check only.
**Why.** The public window has 17 possessions and may not be tuned on (RULES.md). On the
baselines it picks the wrong smoothing window; the proxy league picks the right one
(docs/EXPERIMENTS.md, section 1).

## D2. IDSSE comes from the pinned Hugging Face mirror (2026-10-28)

**Decision.** `tools/download_external.py` fetches IDSSE from `pysport/idsse-data` at revision
`a715a38d…` and verifies every file's published hash.
**Why.** The canonical figshare endpoint answers scripted downloads with a bot check, which we do
not try to get around. The mirror is the same CC-BY data, pinned, so every teammate gets
byte-identical files.

## D3. Two IDSSE matches are held out until Phase 4 (2026-10-28)

**Decision.** `J03WN1` (first division) and `J03WR9` (second division) are excluded from the
default split. Nothing may be tuned or chosen on them. They are scored once, in Phase 4, to
measure how much we over-fitted the five development matches.
**Why.** PLAN.md section 6. Fixed before any method was tuned. Add matches rather than move them.

## D4. Adoption rule for a change (2026-10-28)

**Decision.** Adopt only if the pooled spellF1 gain is at least 0.02, its paired bootstrap
interval excludes 0, and it wins on at least two thirds of the matches. `tools/proxy_league.py`
prints the verdict.
**Why.** PLAN.md section 6, merged with the independent design review. A paired interval judges
both methods on the same possessions, which removes their shared noise.

## D5. External data and caches live outside `data/` (2026-10-28)

**Decision.** External downloads go to `external_data/`, derived arrays to `cache/`. Both are
gitignored.
**Why.** `data/` holds the competition match, which coding agents must never read (CLAUDE.md,
`.claude/settings.json`). Keeping public external data elsewhere lets agents work on it freely
without weakening that guard.

## D6. Develop in a private repo; publish only at submission (2026-10-28)

**Decision.** `origin` is the private, non-fork repo `mih4god3c/mtah2027`. The team of three
pushes and reviews there. The earlier public fork `DenisNeds/more-than-a-hack-2027` holds only
the team-setup commit and gets nothing new. At submission we make a public copy: either switch
this repo to public, or push `main` to a public repo.
**Why.** The fork was public and is listed on the organisers' repo page, so pushing would show
our plan and validation method to other teams. GitHub does not allow a public fork to become
private. The rules require a public repo only at submission (SUBMISSION.md, RULES.md); nothing
requires a fork.

## D7. Simulator v1 is the default for proxy-league decisions (2026-10-28)

**Decision.** `make proxy` degrades with simulator v1 (`solution/simulate.py`). v0 stays
available with `SIM=v0`, for comparison only. Never mix numbers from different versions in one
comparison. v1 keeps v0's ball-stretch lengths exactly and only reorders them. The speed ratio
is calibrated by `V1["speed_weight"]`, and the player model by the remaining `V1` values.
Re-check `make sim-check` after any change to `V1`, and start a new EXPERIMENTS section.

**Why.** Under v0 the ball went missing at random, about 1.0 times the visible-ball player
speed. In BAR-ATH it goes missing in slow play, at 0.54 times. v0's players were clean. v1
meets the FINDINGS.md targets: speed ratio 0.545, 20.7 objects per frame with p5 14, tracklet
median 38 s and duplicates 7.9%. The baselines' macroF1 under v1 sits near the official
full-match level and in the same order; under v0 it did not (EXPERIMENTS section 2).

**Trade-off made on purpose.** The ball is placed by player speed only, and only by reordering
stretches whose lengths stay fixed. That leaves an out-of-play cue the real ATD rarely gives:
the IDSSE ball sits just beyond a line during most dead ball. Degraded, it is seen beyond a line
on 16% of visible frames, against BAR-ATH's 4.2%, and 65% of gaps follow a ball last seen wide or
beyond, against 54%. Adding a beyond-line preference to the reordering barely moved this. Fixing
it would mean changing the length distributions. Ball-position out-of-play rules will therefore
look better on the proxy than they are; check them separately.

## D8. The semi-Markov decoder replaces the majority vote, with all costs at 0 (2026-10-29)

**Decision.** `semi_markov` in `solution/methods.py` ([solution/decoder.py](../solution/decoder.py))
is the reference method from now on. It decodes {H, A, D} at 5 Hz from three inputs: the
nearest-player team, whether the ball is seen, and mean player speed. Its settings are
`decoder.DEFAULTS`: evidence weight w = 1, `switch` = `dead` = `restart` = 0, `no_ball` = 0.
Duration priors, nearest-player agreement and speed likelihoods are fitted
leave-one-match-out on the clean IDSSE dev labels, never on the held-out matches or the public
window. Change a setting only on an ADOPT against `semi_markov` itself. Compare new methods
against `semi_markov`, not `baseline_smooth50`.

**Why.** Under simulator v1 it gains +0.0363 spellF1 over `baseline_smooth50`, with paired
interval +0.0257..+0.0440, and wins 5/5 (ADOPT). macroF1 goes from 0.619 to 0.732. Of 11
settings tried, none with nonzero costs beat zero costs beyond noise (EXPERIMENTS section 2).
Keeping zero costs leaves one tuned number, w, which limits over-fitting the five matches.
`dead` and `restart` cannot be told apart, since every interior stoppage pays both. Ball
position beyond a line is deliberately not used (D7). `no_ball` stays neutral: in BAR-ATH a
missing ball barely signals a stoppage (FINDINGS), while the simulator ties gaps to slow play.

**Caveat.** The whole gain rides on the speed cue: without it the decoder scores 0.2470,
which is WORSE than the baseline. The speed priors are fitted on clean IDSSE positions.
BAR-ATH position noise is not simulated.

## D9. The decoder reads speed as a rank within the match, not in m/s (2026-10-29)

**Decision.** This supersedes the speed part of D8. `decoder.DEFAULTS["speed_scale"]` is
`"rank"`: each frame's collective speed becomes its quantile among the match's graded frames,
and the speed likelihoods are fitted on those quantiles. The speed lag stays 2 frames. The other
D8 settings stand.

**Why.** This choice rests on robustness, not on a D4 gain. On the plain proxy league the rank
scores 0.2993 against 0.2969 in m/s, a gain within noise. The robustness evidence is:
- **BAR-ATH plays about 10% slower than the IDSSE dev matches.** This comes from
  `tools/speed_check.py --bar-ath`: label-free aggregates of the ATD, the same kind of
  evidence the simulator is calibrated to (L3). In m/s the decoder calls half the match dead
  ball, against 0.43 on IDSSE.
- **Under that measured 10% slowdown the m/s decoder drops to 0.2841,** with gain +0.0234. The
  rank keeps 0.2993, with gain +0.0386, wins 5/5, ADOPT.
- **The rank is also immune to position noise.** BAR-ATH shows none, but the m/s decoder lost
  to the baseline at 0.2 m of white noise (EXPERIMENTS section 3).

**Trade-off made on purpose.** The rank assumes BAR-ATH has about the usual share of slow
play. If its true dead-ball share is far from the IDSSE dev range (0.41 to 0.48), the rank
mislabels the margin. On the dev matches at the two ends of that range it lost about 0.002 and
0.006. Revisit this if a label-free BAR-ATH cue, such as tagging set-piece windows, points to an
unusual amount of dead ball.


## D10. touch_markov replaces semi_markov as the reference; a touch hands over the ball (2026-10-30)

**Decision.** `touch_markov` in `solution/methods.py` is the reference method and what
`solution/run.py` submits. It is `semi_markov` (D8, D9) plus the holder from the touch engine,
[solution/touch.py](../solution/touch.py), at weight `touch_w = 1`. The engine runs at its
physical defaults `touch.PARAMS`. In particular `touch_takes = True`: a touch by the other team
hands it the ball at once, with no wait for a control episode. Boundary snapping (E6,
`touch.snap_transitions`) stays off. Compare new methods against `touch_markov`.

**Why.** Under simulator v1 it gains +0.1727 spellF1 over `semi_markov`, with paired interval
+0.1627..+0.1887, and wins 5/5 (ADOPT). The gain holds under the ball-noise stress (D11),
player noise and the 10% slowdown (EXPERIMENTS section 5). Of 19 settings tried, the
hysteresis version (wait for control) gained only +0.034 to +0.058. Among the `touch_takes`
variants, the best (`touch_w = 2`) beat this by +0.011, below the 0.02 bar for a change, so the
untuned weight 1 stays. E6 added +0.003 to +0.005, because the engine already places hand-overs
(latency sd 7.6 frames).

**Risk taken on purpose.** `touch_takes` is a label-semantics choice. DFL possession evidently
flips at the first touch by the other team. Whether Metrica's corrected possession does too is
PLAN.md open question 7. The public window's 1.3 s Barcelona possession, during which no
Barcelona player is ever nearest the ball (PLAN finding 4), suggests it does, but that is one
possession. If the organisers answer that ownership changes only on control, re-test
`touch_takes = False`. Do not choose between the two on the public window.

## D11. Methods that read the ball must hold up under the ball-noise stress (2026-10-30)

**Decision.** Any change that reads ball positions or kinematics is adopted only if it also
gets ADOPT against its reference under `make proxy BALL_NOISE=0.05 BALL_JUMPS=3.4`. Simulator v1
is unchanged (D7). The stress is `simulate.add_ball_noise`: white jitter plus 1.6 to 4.4 m hops,
on its own random stream. Numbers under it are never mixed with plain ones.

**Why.** The IDSSE ball is clean, and DFL freezes it during dead ball. BAR-ATH has 216 jumps
above 40 m/s, all under 110 m/s, and its ball is almost never still (FINDINGS A2). The setting
reproduces the BAR-ATH jump rate and median ball speed on degraded IDSSE. It is pessimistic in
one way: white jitter removes all stationary stretches, where BAR-ATH keeps 2%.

## D12. The touch engine is tuned for a smoothed ball; the ball stress includes smoothing (2026-10-30)

**Decision.** This supersedes D11's stress setting and D10's engine thresholds; the rest of D10
stands (`touch_markov` is the reference, weight 1, a touch hands over the ball).
- **Stress.** The ball stress for adoption is
  `make proxy BALL_NOISE=0.05 BALL_JUMPS=3.4 BALL_SMOOTH=0.26`, which smooths the ball with a
  Gaussian of sd 0.26 s inside each visible stretch. Any method that reads the ball must get
  ADOPT under it. Plain-v1 numbers no longer decide anything that reads the ball. Simulator v1
  itself is unchanged.
- **Engine.** `touch.PARAMS` uses `lag_s` 0.4, `touch_m` 3.0 and `suppress_s` 0.4, with the
  other thresholds unchanged.

**Why.** On BAR-ATH the section 5 detector found 2.3 touches per clean-ball minute against
28.8 on the proxy. `touch_check --diagnose --bar-ath` shows why: the BAR-ATH ball is smoothed,
with acceleration autocorrelation +0.99 and small velocity changes at short lags. A smoothing sd
of 0.24 to 0.28 s reproduces the BAR-ATH dv percentiles at three lags. The touch and control
rates, which were not fitted, land either side of BAR-ATH's. Under that ball the old engine gains
+0.024 and the new one +0.070 over `semi_markov`, ADOPT. It holds at smoothing 0.20 and 0.32 s
(+0.10, +0.04), harsher jitter, player noise and the slowdown (EXPERIMENTS section 6). Of 11
settings tried, the registered one is mid-plateau with round physical values.

**Caveat.** The gain scales with the true smoothing, which is matched only through
aggregates. Confirm with `tools/touch_check.py --bar-ath`: the D12 engine should fire about
20 times per clean-ball minute there, as on the stressed proxy (21.6).

## D13. The kickoff grammar reads players only and errs towards doing nothing; registered, not submitted (2026-11-04)

**Decision.** `touch_kickoff` (`solution/kickoff.py`) is `touch_markov` with each half dead ball
until a kickoff found from the players: the end of a formation hold of at least 8 s (both teams
in their own halves, Law 8), with quiet play before and a speed rise after (`kickoff.PARAMS`).
The ball is not read. It is not what `solution/run.py` submits: on the busy synthetic
pre-kickoff it is "better, below the adoption bar" (+0.0015, interval +0.0006..+0.0021, wins
5/5), not ADOPT (D4). Its safety properties stay as they are:
- when no hold passes every cue, the half is decoded exactly as before;
- the kick is placed `back_s` (2 s) before play speeds up, so it errs early;
- it takes only the first kickoff in a half, and only if play before it stayed quiet (`before`
  guard). Formation holds that pass quiet and rise are common in open play (35 on the plain
  dev halves), so this guard is what keeps false kickoffs at 0 (tests/test_kickoff.py);
- decoding resumes on the half's block grid, so the rest of the half is unchanged.
Loosening any of these needs a re-run of `tools/kickoff_check.py --grid`, which must still show
0 false kickoffs on the plain halves.
**Why.** The BAR-ATH ball is never seen resting on the centre mark (EXPERIMENTS section 7), so a
ball cue finds nothing there. A missed kickoff costs what it costs today; a false one mid-half
would turn real play into D. The rule fixes about one possession per half, so D4's 0.02 bar is
out of reach by construction. Submitting it needs a team decision that supersedes D4 for
grammar rules like this, informed by `kickoff_check --bar-ath`: is the kickoff found near the
end of half 2's formation hold, and how much of the stretch before it does `touch_markov`
decode as a team?

## D14. A narrow exception to D4 for grammar rules; touch_kickoff is submitted (2026-11-04)

**Decision.** This supersedes D4 for one kind of change, and D13's "not submitted".
`solution/run.py` submits `touch_kickoff`, and it is the reference method from now on. A grammar
rule may be adopted without D4's 0.02 gain only if it:
- fixes a specific, documented failure that the proxy league cannot represent;
- leaves the plain proxy byte-identical (gain +0.0000);
- never fires falsely: 0 false fires across its grid of settings on the plain dev halves;
- is not WORSE on any synthetic stress built for it, and is better on at least one;
- is confirmed on BAR-ATH from label-free aggregates: it fires where expected and changes what
  it is meant to change.
Everything else still needs D4's ADOPT.
**Why.** D4's 0.02 bar measures broad gains over about 2,200 possessions. A rule that fixes one
possession per half can never reach it, however right it is. `touch_kickoff` meets every
condition above (EXPERIMENTS section 7):
- plain proxy +0.0000;
- 0 false kickoffs in 90 plain halves and at all 8 grid settings;
- +0.0002 (calm) and +0.0015 (busy, interval +0.0006..+0.0021, wins 5/5);
- on BAR-ATH it finds the second-half kickoff 2 s before the published one and none in half 1.
  It turns into D a stretch where `touch_markov` labelled 46% of frames as a team, in 3 false
  possessions.
Decided by miha on 2026-11-04.

## D15. C2b dead-ball cues are registered, not submitted; the still ball is not a cue (2026-11-04)

**Decision.** `touch_deadball` (`solution/methods.py`) is `touch_kickoff` plus
`solution/deadball.py`'s cues at `deadball_w` 0.5: a player standing beyond a line, and the ball
seen more than 1 m beyond one. It is not what `solution/run.py` submits: under the D12 stress it
gains +0.0084 (interval +0.0034..+0.0111, wins 5/5), below D4's bar. D14's exception does not
apply, since it is not byte-identical on the plain proxy. `touch_kickoff` stays the reference.
Standing rules from this work:
- **Do not use a still or slow ball as dead-ball evidence.** DFL freezes the dead ball, so the
  stressed proxy has a still ball on 13.1% of graded frames, BAR-ATH on 2.0% (FINDINGS A2). It
  looks like the strongest cue on the proxy (on 61% of the frames the decoder misses inside a
  stoppage), and that comes from the feed.
- **Any cue that reads the ball beyond a line is also scored under `simulate.thin_beyond`**:
  `BEYOND_KEEP=0.82 BEYOND_MARGIN=1` (BAR-ATH's 2.7% beyond 1 m) and `BEYOND_KEEP=0.26`
  (its 4.2% beyond at all, the pessimistic bound). It extends D7 and D11.
- `decoder.DEFAULTS["speed_lead_s"]` stays 0: leads of 0.4 to 1.6 s were within noise.
**Why.** `tools/deadball_check.py` puts the perfect-dead-ball ceiling at +0.089 (stress) and
+0.095 (plain), mostly start edges (+0.037), false stoppages (+0.020) and end edges (+0.018).
Throw-ins are the largest type (+0.028). But on the frames that ceiling would fix, players keep
moving, and only the ball shows the stoppage. Of 12 settings (EXPERIMENTS section 8), the fair
cues gave +0.005 to +0.008. Even everything together, the unfair still ball included, gave
+0.018. Submitting `touch_deadball` needs a team decision informed by
`tools/deadball_check.py --bar-ath`. Check that its D share stays in the IDSSE 0.41 to 0.48 range
(D9), that its player cue fires at about the proxy's rate (0.098 of graded frames, which rules out
linesmen tracked as players), and that it restarts inside at least as many Smart Tagging set
pieces as `touch_kickoff`.

## D16. touch_gapteam (E5 spread cue) gets ADOPT; it is submitted once its BAR-ATH cue check passes (2026-11-04)

**Decision.** `touch_gapteam` (`solution/methods.py`) is `touch_kickoff` plus
`solution/gapteam.py`'s spread cue: home's outfield width minus away's, from player positions
only. The decoder reads it on every frame (`team_where` "all") at `team_w` 0.5, with
`team_dead` "max", so the cue chooses between the teams and has no say on dead ball. Under the
D12 stress it gets ADOPT against `touch_kickoff` (D4). It is registered but not yet submitted:
`solution/run.py` switches to it, and it becomes the reference, once
`tools/gap_check.py --cues --bar-ath` shows BAR-ATH's spread cue near the stressed proxy's on
frames without a seen ball. The proxy values are: known 0.98, fires 0.91, |c| p50 12 m, p90 25 m.
Standing rules from this work:
- **Keep `team_w` at 1 or below.** Spread at weight 1.5 gains only noise, and at 2 it is WORSE
  (-0.016): the decoder merges real possessions. 0.5 is chosen over 1, which gains a little
  more (+0.025, within noise of +0.0225), to stay well clear of that cliff.
- **Keep `team_dead` "max"** unless a dead-ball question is being tested on purpose. It keeps
  this cue out of C2b's territory (D15).
- Advance, defensive-line height and pressure near the last-seen ball stay unused. Each alone
  was noise, and together with spread they added nothing.
**Why.** Ball gaps cost `touch_kickoff` up to +0.095 spellF1 (`tools/gap_check.py`, oracle
team inside gaps), almost all in gaps that hold an open-play hand-over. Of 17 settings
(EXPERIMENTS section 9), `touch_gapteam` gains +0.0225 (interval +0.0145..+0.0290, wins 5/5,
ADOPT). It is not WORSE under any stress:
- plain v1: +0.0191;
- SPEED=0.9: +0.0208, ADOPT;
- NOISE=0.1: +0.0175;
- D12 plus busy pre-kickoff: +0.0215, ADOPT.
The cue never reads the ball, so DFL's frozen dead ball cannot flatter it. The risk is in BAR-ATH's
player tracking: dropped wide players, ghosts or tracked linesmen would bias width. That is what
the pending check looks for.
**Update (2026-11-04).** miha ran the BAR-ATH check: spread is known on 0.89 of gap frames,
fires on 0.78, |c| p50 9.8 m, p90 22.4 m, mean +1.09 (EXPERIMENTS section 9). There is no sign of
inflated widths or a flipped direction. miha decided to switch: `solution/run.py` submits
`touch_gapteam`, and it is the reference method from now on.

## D17. Hand-over timing inside ball gaps is closed; in-gap work targets short possessions, and its evidence comes first (2026-11-04)

**Decision.** E5 part 2 stops after Step 0. Do not spend effort on placing or snapping decoded
changes inside gaps: change-point timing, causal or anticausal smoothing of spread, gap-specific
snapping. Any new in-gap cue must create hand-overs the decoder does not make today. Measure
it first with `tools/gap_check.py --timing --capacity`, against these bounds:
- the capacity oracle, a perfect-sign cue at realistic size: +0.050 (`team_w` 0.5) to +0.072
  (`team_w` 1) under the D12 stress;
- the team ceiling of +0.0865.
The cue must fire within about 1 s of true hand-overs, which spread does on 5% of them.
D16's `team_w` <= 1 rule stands; the capacity oracle at `team_w` 1 does not lift it.
**Why.** EXPERIMENTS section 10. The timing oracle gains +0.0007 (+-1.5 s), +0.0025 (+-3 s) and
+0.0055 (anywhere in the gap), against PLAN's 0.02 stop rule. The in-gap loss is possessions
lying inside one gap: +0.074 for those under 5 s. The decoder makes 10 in-gap hand-overs per 90
where truth has 66. Ideal evidence would recover most of the ceiling, so the blocker is the
evidence: spread is a posture signal lagging the hand-over by a median 3 s. Minimum durations
(1 block) are not binding. The prior's cost of about 7 nats per inserted short possession caps
recall under 2 s at 0.13 to 0.28 even with ideal evidence, so a transition bonus (C4's C2
events) would be needed there too.

## D18. Short possessions: touch/E6 ceilings are under 0.02; players-only moment events are measured first, and dadvance is the cue that qualifies (2026-11-04)

**Decision.** These rules come from the evidence. The choice of the next route is miha's (WORKLOG).
- **Do not start touch-engine or E6 work aimed at possessions under 2 s with the ball seen**
  without new evidence. Their ceilings under the D12 stress are under D4's 0.02:
  - snapping decoded changes onto their true starts and ends: +0.0185;
  - the true team on the possessions whose start the touch engine already sees: +0.012, below
    a random subset of the same size (+0.054).
  The touch engine sees mostly what the decoder already decodes.
- **Score any hand-over event cue before the decoder**, with `tools/gap_check.py --events`
  (hit within +-1 s against the chance level at its false-alarm rate, under NOISE=0.1 and
  SPEED=0.9) and `--seen --feature <name>` (reach against random). Only cues the proxy
  represents fairly qualify (D15, D16).
- **`dadvance0.5` is the qualifying moment cue** (`solution/handover.py`). It is the
  past-minus-future difference of all players' velocity along x' over 0.5 s, with peaks above
  the match's 0.9 quantile. Spread, line, the last-seen-ball cues (leave, crowd) and
  run-direction crossings do not qualify: lift 2.3 or less at +-1 s.
**Why.** EXPERIMENTS section 11:
- dadvance0.5 fires within +-1 s of 32% of true in-gap hand-overs, at lift 4.2. The lift holds
  under position noise (4.0) and the slowdown (4.2). Its median lag is +0.7 s, against spread's
  5% and 3 s.
- On the <2 s ball-seen pool it reaches +0.042, against touch's +0.012. In gaps it reaches
  +0.020 under 2 s and +0.037 under 5 s.
**Caveats.**
- Every reach oracle ignores false alarms. dadvance's precision is 0.26; the touch engine's
  claims are right 0.36 of the time.
- dadvance's BAR-ATH event rate is not yet checked (`gap_check.py --events --bar-ath`). Proxy:
  5.1 per minute of play, 2.5 per gap minute.
**Update (2026-11-04).** miha ran `gap_check.py --events --bar-ath`. dadvance0.5 fires 5.01 times
per minute of play (proxy 5.12) and 2.02 per gap minute (proxy 2.52, about 0.8 times). There is
no warning sign. Expect a somewhat smaller effect than on the proxy (EXPERIMENTS section 11).

## D19. The dadvance transition bonus is closed below D4's bar; the bonus mechanism stays, off by default (2026-11-04)

**Decision.** Do not build or tune a method that feeds `dadvance0.5` events (D18) into the
decoder's switch bonus.
- `decoder.DEFAULTS["bonus_w"]` stays 0. With 0, decoding is byte-identical to before. The
  mechanism stays (`decoder.switch_bonus`, `semi_markov(bonus_fn=...)`) for a better event source.
- **Score any future event source with `tools/gap_check.py --bonus` first.** Its "right"
  (false-alarm-free) row must clear 0.02 before the real events are worth a proxy-league run.
**Why.** EXPERIMENTS section 11, 15 settings under the D12 stress, against `touch_gapteam`:
- all dadvance events: at best +0.0043, and -0.022 at 7 nats;
- the oracle-filtered right events: at best +0.0096, so recall is the limit;
- perfect events: +0.024 (4 nats) and +0.040 (7 nats).
An event source needs to recall far more than the third of hand-overs that dadvance finds
within 1 s. With D18's touch/E6 ceilings, short possessions now have no route above 0.02 from
the evidence measured so far.

## D20. The task's possession is controlled possession; the proxy's DFL labels must be relabelled before more splitting decisions (2026-11-04)

**Decision.** This revisits D10's "risk taken on purpose": the answer is now known.
- CHALLENGE.md ("Possession rules") defines a change of possession as the other team gaining
  **controlled** possession. A simple touch, a deflection that does not intercept, or an aerial
  duel won without control is not a change. A loose or contested ball stays with the last team.
  The public ground truth is the reference.
- The proxy league's labels are DFL's `BallPossession`, which flips at the first touch by the
  other team (D10). So the proxy's truth holds short "possessions" the real truth does not have,
  and it rewards splitting.
- **Until a controlled-possession relabelling of the proxy truth exists, do not adopt any change
  whose gain comes from splitting more:** more short possessions, a higher predicted / true
  count, `touch_takes`, switch bonuses.
- The relabelling rule comes from the definition ("keeps the ball for a few seconds", "more than
  two consecutive passes"), never fitted to the public window. The window may only confirm the
  kind of structure (PLAN L2).
- Once it exists, re-test `touch_takes` True against False (D10) and the adopted methods
  (`touch_kickoff`, `touch_gapteam`) on both label versions. A new decision then says which
  labels the proxy decides on.
**Why.**
- The public window has 12% of its possessions under 2 s, median 7.6 s, against DFL's 30% and
  3.7 s (EXPERIMENTS section 4).
- On BAR-ATH `touch_gapteam` decodes 393 possessions per 90, against 306 in the public truth.
- D10's 1.3 s Barcelona possession is one possession and does not outweigh the written rule.
- Sections 9 to 11 found no route above 0.02 on short possessions anyway (D17 to D19).

## D21. The controlled-possession relabelling is fixed; the adopted methods stand on it; the proxy decides on it (2026-11-04)

**Decision.** This answers D20's re-test.
- **The relabelling is fixed, not tunable.** [solution/relabel.py](../solution/relabel.py)
  `keep2`: a team run under 2 s is a touch, not control, and goes to the previous team (shortest
  first). Restart runs are exempt, and D is unchanged. The 2 s comes from CHALLENGE.md's
  wording, set before any measurement. Do not move it, or pick between keep1, keep2 and keep3, on
  the public window or on proxy scores. `keep2pass` agrees with it (spellF1 0.989, the same
  verdicts), so it is a sensitivity check, not a second truth to decide on.
- **The adopted choices stand.** The touch engine (D10), the kickoff grammar (D14) and
  `touch_gapteam` (D16) all keep their verdicts on keep2 truth. `touch_gapteam` gets ADOPT on
  every label set.
- **`touch_takes` stays True.** On controlled truth, waiting for control is within noise, not
  ADOPT. D10's risk did no harm, but the label semantics no longer back it either.
- **`idsse.LABELS` defaults to `dfl`, and `solution/run.py` is unchanged.** `touch_gapteam_k2`
  (priors fitted on keep2) is registered, not submitted: +0.0189 on the decision run, below D4.
  On BAR-ATH it lowers Smart Tagging agreement inside gaps (0.721 to 0.676), so it gets no
  exception.
- **The proxy decides on `LABELS=keep2 PRIOR_LABELS=dfl`**, with the D12 stress; these are now
  `make proxy`'s defaults. Any change must also not be WORSE on `LABELS=dfl`. Decided by miha
  on 2026-11-04. Priors keep following DFL (`idsse.LABELS` stays `dfl`, as `solution/run.py`
  fits). Moving them to keep2 is a method change and needs D4, like `touch_gapteam_k2`. The
  other tools (`gap_check.py` and the rest) still read DFL truth by default.
**Why.** EXPERIMENTS section 12.
- keep2 brackets the public window's structure from the strict side, and DFL from the loose
  side. This is described only, never chosen on.
- First-touch semantics alone cost a perfect DFL labeller 0.24 spellF1 on keep2 truth.
- The decoded/true count on the keep2 proxy (1.26) is the same as on BAR-ATH against the window
  (1.28). On DFL it is 0.84.

## D22. The C3 frame model gets ADOPT; touch_c3_w1 is the candidate, submitted only after its BAR-ATH check (2026-11-04)

**Decision.**
- `touch_c3_w1` (`solution/methods.py`) is `touch_gapteam` plus `solution/framemodel.py`'s
  per-frame P(home | alive) at `frame_w` 1, read on every frame, with dead ball taking the
  better team's likelihood. Under the D12 stress on keep2 truth it gets ADOPT against
  `touch_gapteam` (D4). `touch_c3` (weight 0.5) is the fallback.
- **Registered, not submitted.** `solution/run.py` is unchanged. The switch needs miha's
  decision after `tools/gap_check.py --decoded --bar-ath --methods touch_gapteam touch_c3
  touch_c3_w1`. Its pass criteria, fixed before it is run:
  - Smart Tagging agreement inside gaps at least `touch_gapteam`'s 0.721 (what sank
    `touch_gapteam_k2`, D21); with the ball seen not below 0.80;
  - gap D and seen D within 0.02 of `touch_gapteam`'s (0.776, 0.189);
  - possessions per 90 falling by about the proxy's factor (0.83 for w 1, 0.89 for 0.5: about
    327 and 349 from 393). Far outside 250 to 393 is a warning;
  - the home share of gap team frames within 0.05 of `touch_gapteam`'s 0.486.
  If w 1 fails and 0.5 passes, 0.5 is the one to submit.
- **Standing rules for the model:**
  - Train leave-one-match-out on degraded dev matches only (never the hold-out, D3), with the
    simulator seed `framemodel.TRAIN_SEED`, apart from the evaluation seeds.
  - Keep features players-only (D15) and team-swap symmetric (`framemodel.swap`, tested).
  - The model has no say on dead ball. Keep the weight at 1 or below (D16); nothing above 1
    was tried.
  - The model's settings were set once. Do not tune them on the proxy league without counting
    the settings tried; never on the public window.
- `tools/gap_check.py` now reads keep2 truth by default (`--labels`), like `make proxy`. D21's
  "the other tools still read DFL truth" no longer holds for it.
**Why.** EXPERIMENTS section 13.
- Step 0: ideal per-frame team evidence at spread's strength is worth +0.053 on keep2
  (+0.051 on DFL). It is worth +0.035 at a 1 s lag and nothing at spread's 3 s lag.
- The model is right on 0.873 of alive gap frames (spread's sign 0.834) and turns with a median
  lag of 0.3 s.
- `touch_c3_w1` gains +0.0780 (interval +0.0704..+0.0884, wins 5/5). It gets ADOPT on DFL truth
  (+0.047), plain v1 (+0.051), SPEED=0.9 (+0.059), NOISE=0.1 (+0.051) and the busy pre-kickoff
  (+0.076). Weight 1 beats 0.5 on every run.
- Most of the gain is outside gaps: it removes false possession changes (1,837 to 1,528
  predicted, 1,455 true). Read only in gaps it gains +0.010.
**Risk.** The model reads absolute positions and speeds learned on IDSSE. BAR-ATH has fewer
tracked players inside gaps than the simulator gives. A label-free check can show a shift, not
prove the gain.
**Update (2026-11-04).** miha ran the BAR-ATH check (EXPERIMENTS section 13). Both `touch_c3`
and `touch_c3_w1` fail the in-gap tag rule: 0.683 and 0.674 against 0.721. Every other rule
passes, and tag agreement with the ball seen rises (0.813 to 0.857 and 0.868). **Not switched:**
`touch_gapteam` stays submitted and the reference for the proxy league. The C3 methods stay
registered. A C3 variant may be submitted only once it passes D22's rules unchanged; do not
loosen the rules to fit this result.

## D23. touch_c3_seen_w1 is submitted: the frame model where the ball is seen, at weight 1 (2026-11-04)

**Decision.** This supersedes D22's candidate. `solution/run.py` submits `touch_c3_seen_w1`, and
it is the reference method from now on. It is `touch_gapteam` plus `solution/framemodel.py` at
`frame_w` 1, read only on frames where the ball is seen (`frame_where` "seen"). Gaps stay with
spread, as in D16. D22's standing rules for the model hold.
- **Do not read the frame model inside ball gaps** until a label-free BAR-ATH check explains
  why it agrees with the Smart Tagging team less than spread does in the first 15 s of a gap.
  Missing players are ruled out.
- Weight 1 is at D16's limit. Nothing above 1 was tried; do not go above it.
- The variant was picked after seeing D22's failed check. Its BAR-ATH check used D22's rules
  unchanged.
Chosen by Claude Code at miha's request ("do what you think is better"), on 2026-11-04.
**Why.** EXPERIMENTS section 13.
- Against `touch_gapteam` under the D12 stress on keep2 truth: +0.0655 (interval
  +0.0584..+0.0757), wins 5/5, ADOPT. It gets ADOPT on DFL (+0.033), plain v1 (+0.040),
  SPEED=0.9 (+0.049), NOISE=0.1 (+0.041, 4/5) and the busy pre-kickoff (+0.063).
- Against weight 0.5 directly: +0.0244 (+0.0218..+0.0334), 5/5, ADOPT. D16 chose 0.5 because
  1 was within noise of it; here 1 clears D4.
- It passes every D22 rule on BAR-ATH. Tag agreement rises in gaps (0.721 to 0.734) and with
  the ball seen (0.813 to 0.867); gap D, seen D and the home share hold.
**Caveats.**
- Seen D moves +0.019 against the rule's 0.02.
- BAR-ATH possessions fall to 282 per 90, further than the proxy factor predicts (about 319).
  The visible-ball loss on the proxy is mostly invented possessions, so fewer decoded
  possessions is the expected direction. Whether it overshoots on BAR-ATH is not knowable
  without labels.
- Byte-identical output across two runs is shown on the proxy. On BAR-ATH it needs two
  `make submission` runs with their hashes compared.
**Update (2026-11-04).** miha ran it on BAR-ATH. Two runs are byte-identical (sha256
`f7459db0…bcf8358d`), about 18 s each, and the file is valid. The public-window correctness
check shows no team swap (share error 0.006) and 16 possessions against 17. Transitions are a
median 11 frames late (EXPERIMENTS section 4).

## D24. C6 self-training is closed at Step 0: in-match adaptation is worth under 0.01 even with true labels (2026-11-05)

**Decision.** Do not build the C6 pseudo-labeller (PLAN C6, E9) or a synthetic-occlusion test
for it without new evidence. `solution/run.py` is unchanged; `touch_c3_seen_w1` stays submitted
and the reference (D23). D23's "the frame model stays out of gaps" stands. Adaptation is not a
route to lifting it.
- `solution/selftrain.py` and `tools/selftrain_check.py` stay as the Step 0 tools. The C6
  oracle reads `match.labels`, so it is proxy-only and never registered in `METHODS`.
- `semi_markov(frame_fn=...)` stays (default None, byte-identical) for any future source of
  frame log-odds.
- Reopen only with a route that beats this oracle's bound, for example a different kind of
  match-specific signal, not a different learner. Score it with `selftrain_check.py --ceiling`
  first.
**Why.** EXPERIMENTS section 14, 11 oracle settings, keep2, D12 stress:
- A model fitted on each match's own TRUE labels at pseudo-labellable frames, cross-fitted by
  60 s blocks, gains over the same reading by the base model: in gaps +0.0022, everywhere
  +0.0078, seen only +0.0059.
- The one oracle row over 0.02 against `touch_c3_seen_w1` (+0.0204, `_all`) gets +0.0125 of it
  from reading the unadapted model in gaps (`touch_c3_w1`), not from adaptation.
- The first 2 s of a gap, where BAR-ATH disagreed (D23), do not improve with any learner.
  Three learners (logistic, small GBM, a pooled refit) agree, so the learner is not the limit.
**Caveat.** The proxy's matches are all IDSSE, so the shift from training to target is smaller
than IDSSE to BAR-ATH. Adaptation could matter more on BAR-ATH than this shows. Only a
label-free check could suggest that, and this ceiling gives no reason to build one. Claude Code's
reading at Step 0, as miha's brief asked ("if under 0.02, stop and report"). The headline row's
+0.0204 is technically over the bar, so miha may overrule.

## D25. Visible-ball invented possessions: touch_takes stays True, keep2 priors stay registered, the learned dead-ball model is closed at Step 0 (2026-11-05)

**Decision.** `solution/run.py` is unchanged; `touch_c3_seen_w1` stays submitted and the
reference (D23).
- **`touch_takes` stays True.** This strengthens D21's "within noise". On top of
  `touch_c3_seen_w1`, `touch_takes` False is within noise on the decision run and WORSE on DFL,
  plain v1, SPEED=0.9 and NOISE=0.1. Do not re-test it without a change to the touch engine.
- **keep2-fitted priors stay registered only** (`touch_c3_seen_w1_k2`). They are better on every
  keep2 run (+0.009 to +0.011, 5/5) and never reach D4's 0.02. D21's BAR-ATH warning still
  applies: they lowered in-gap Smart Tagging agreement.
- **A selective fix of the touch engine's hand-overs needs its own ceiling first.** The true team
  on the phantoms the holder starts with a touch hand-over is worth +0.036. A real filter recovers
  a fraction of that, so build one only with evidence that it recalls most of them.
- **The learned dead-ball model (`solution/deadmodel.py`) is closed at Step 0.** No decoder work.
  It cleared the frame-level bar under D12 and lost it at stoppage edges under SPEED=0.9. Reopen
  only with features that hold under SPEED=0.9 (ranks and ratios, no absolute positions), counted
  as a new setting, and checked with `tools/deadmodel_check.py --speed 0.9` before any decoding.
- `semi_markov`'s `prepare` / `decode` split stays, with registered methods byte-identical. It
  also gives E6 snapping the touches it needs (they were lost before; nothing registered used it).
**Why.** EXPERIMENTS section 15. Decoded possessions given to the wrong team on seen frames are
158 per seed and 55% of the invented error frames. The holder is dominant over 0.80 of them, and
switching it off removes 0.49. `touch_takes` False cuts them to 99 per seed but delays real
hand-overs (the boundary oracle rises from +0.038 to +0.052). Claude Code's reading; miha
decides the next lever.

## D26. The learned dead-ball model gets ADOPT; touch_c3_seen_w1_dm is the candidate, submitted only after its BAR-ATH check; TEMPO is the tempo check for position features (2026-11-05)

**Decision.**
- **Tempo stress.** This extends D9 and supersedes the SPEED=0.9 part of D25's reopening rule.
  `make proxy TEMPO=0.9` (`simulate.retime`: the clean match replayed at 0.9 times the speed,
  positions unchanged, truth retimed with it) is the tempo check for any method that reads
  positions. `SPEED=0.9` shrinks the pitch, which moves players off the lines; it stays a
  reported stress, not a deciding one, for such methods. Written into EXPERIMENTS section 16
  before anything was scored with it.
- **E10 (whistle) is closed at Step 0:** a perfect whistle is worth +0.012 to +0.015 (section 16).
- **`touch_c3_seen_w1_dm`** (`solution/methods.py`) is `touch_c3_seen_w1` plus
  `solution/deadmodel.py`'s v1 model (D25's, unchanged) read by the decoder at `deadm_w` 1 in
  place of the speed likelihood. Under keep2 + D12 it gets ADOPT against `touch_c3_seen_w1`
  (D4), and on every other run. **Registered, not submitted:** `solution/run.py` switches only on
  miha's decision after the BAR-ATH check below.
- **Standing rules for the model:** players only (D15); trained leave-one-match-out on degraded
  dev matches with `framemodel.TRAIN_SEED`, never the hold-out (D3); `deadm_w` 1, nothing above it
  tried; settings set once. Do not tune them on the proxy league without counting settings; never
  on the public window. v2 (no absolute positions) failed its frame-level bar; it is not a fallback.

**BAR-ATH pass criteria, fixed before the check is run.** Commands:
`tools/deadball_check.py --bar-ath --methods touch_c3_seen_w1 touch_c3_seen_w1_dm`,
`tools/gap_check.py --decoded --bar-ath --methods touch_c3_seen_w1 touch_c3_seen_w1_dm`,
`tools/deadmodel_check.py --bar-ath`. Every "reference" number comes from the same run. Proxy
values (keep2, D12, `--proxy-side` / `--decoded`) are given for each.
1. (D15) Decoded D share of graded frames within the IDSSE range 0.41 to 0.48. Proxy: 0.435 to
   0.443 (truth 0.441).
2. (D15) Smart Tagging set pieces with a decoded restart inside (all types pooled): at least as
   many as `touch_c3_seen_w1`. Proxy, with DFL restart events as tags: 0.36 to 0.48 of windows
   (throw-ins 0.29 to 0.51, free kicks 0.58 to 0.54).
3. (D22) Smart Tagging team agreement inside gaps at least `touch_c3_seen_w1`'s (D23: 0.734);
   with the ball seen, not below 0.80 (D23: 0.867).
4. (D22, **adapted, miha to confirm**) Gap D and seen D: D22 asked for "within 0.02 of the
   reference". That rule kept a team model away from dead ball; this method's purpose is to move
   dead ball, and on the proxy it moves seen D by +0.026 (0.223 to 0.249, truth 0.256) and gap D by
   -0.019 (0.751 to 0.732, truth 0.716), so the literal rule fails on the proxy itself. Adapted
   rule: each change has the proxy's sign and at most twice its size: gap D change in -0.038..0,
   seen D change in 0..+0.052. Rule 1 bounds the total.
5. (D22) Possessions per 90 fall by about the proxy's factor, 0.93 (281.7 to 262.0, truth 275.0):
   about 262 from D23's 282. Below 240 (0.85x) or any rise is a warning.
6. (D22) Home share of decoded gap team frames within 0.05 of `touch_c3_seen_w1`'s. Proxy moves
   +0.029 (0.526 to 0.555, truth 0.541).
7. (new, the model's own shift; D22's risk) `deadmodel_check.py --bar-ath`, against the proxy's
   range over 15 match-seeds: lean dead 0.40 to 0.50 (proxy 0.424..0.461); Spearman with the
   slowness rank at least 0.75 (0.818..0.861); dead-call agreement with the speed rank at least
   0.80 (0.861..0.884). These three fail the check. The feature medians are warnings only, read
   against the proxy (tracked 21.8, gyration 20.4 m, touchline 3.3 m, goal line 6.4 m, in box
   3.37, beyond 0.099; D15 already measured BAR-ATH's player-beyond rate at 0.7 times the
   proxy's, so beyond about 0.07 is expected).
8. Two `make submission` runs with the switch give byte-identical files; runtime about 40 s (the
   model fit adds about 20 s, scoring about 4 s).
If any of 1 to 7 fails, `touch_c3_seen_w1` stays submitted. Do not loosen these rules to fit the
result.

**Why.** EXPERIMENTS section 16.
- Under TEMPO=0.9 the model's stoppage-edge AUC gains (+0.045 starts, +0.056 ends, 5/5) are as
  large as under D12; D25's SPEED=0.9 failure came from the shrunk pitch.
- Capacity: the term carries the whole perfect-dead-ball ceiling at the model's strength
  (`dmo_sign` +0.1465, ceiling +0.147).
- `touch_c3_seen_w1_dm`: +0.0702 (interval +0.0572..+0.0812), wins 5/5, ADOPT. ADOPT also on DFL
  (+0.048), TEMPO=0.9 (+0.070), plain v1 (+0.070), SPEED=0.9 (+0.034), NOISE=0.1 (+0.072) and
  the busy pre-kickoff (+0.069). On the proxy it brings stoppages per 90 from 126 to 100 (truth
  97) and their median from 11.4 to 17.4 s (truth 19.0): it removes false stoppages.
**Risk.** The model reads absolute positions learned on IDSSE (D22's risk again), and fewer
decoded possessions on BAR-ATH, where D23 already decodes 282 per 90, could overshoot. The
label-free check can show a shift, not prove the gain.
**Update (2026-11-05).** miha ran the BAR-ATH check (EXPERIMENTS section 16). Rule 4 fails: gap D
rises 0.777 to 0.818 and seen D falls 0.208 to 0.188, both against the proxy's direction (D22's
literal 0.02 rule fails too, so the adaptation question is moot). Rules 1 to 3 and 5 to 7 pass;
stoppages fall from 140 to 97 per 90 and lengthen from 9.4 to 18.1 s, as on the proxy.
**Not switched:** `touch_c3_seen_w1` stays submitted and the reference for the proxy league.
`touch_c3_seen_w1_dm` stays registered. A variant may be submitted only once it passes D26's
rules unchanged.
**Update 2 (2026-11-05), fixed before its BAR-ATH check.** miha chose a seen-only variant:
`touch_c3_seen_w1_dms` (`deadm_where` "seen"; the model is read only with a seen ball, gaps keep
the speed likelihood). It was picked after seeing the failed check. It gets ADOPT against
`touch_c3_seen_w1` (+0.0362, interval +0.0272..+0.0457, 5/5) and on every stress but SPEED=0.9
(+0.016, 4/5, below the bar; reported, not deciding, by this decision). It is checked on BAR-ATH
with D26's rules unchanged; the proxy values they compare against are now:
1. D share in 0.41 to 0.48 (proxy 0.435 -> 0.451).
2. set pieces with a decoded restart inside: at least `touch_c3_seen_w1`'s (proxy 0.36 -> 0.43).
3. tag agreement in gaps at least `touch_c3_seen_w1`'s; with the ball seen not below 0.80.
4. gap D: the proxy change is +0.003, under 0.005, so the BAR-ATH change must be within +-0.02
   (the clarification written before the proxy runs, EXPERIMENTS section 16). Seen D: proxy
   +0.025, so the BAR-ATH change must be in 0..+0.050.
5. possessions per 90 about 0.97 times `touch_c3_seen_w1`'s (281.7 -> 273.9): about 274 from 282.
   Below 0.85 times (240) or any rise is a warning.
6. gap home share within 0.05 (proxy +0.000).
7. the model is unchanged, so `deadmodel_check.py --bar-ath` stays as already run: pass.
8. on a switch, two byte-identical `make submission` runs, about 40 s.
**Update 3 (2026-11-05).** miha ran the check of `touch_c3_seen_w1_dms`. Rule 4 fails on seen D
(-0.021 against the proxy's +0.025; D22's literal rule fails too). Rules 1 to 3, 5 to 7 pass, and
gap D holds (-0.002). **Not switched:** `touch_c3_seen_w1` stays submitted and the reference;
`_dm` and `_dms` stay registered. Both variants lower seen D on BAR-ATH while raising it on the
proxy, so the difference is in how the model reads BAR-ATH's seen-ball play, not where it is
read. **Do not try a third variant of where or how much the model is read against these rules:**
two post-hoc variants have now been checked against BAR-ATH, and a third would fit the check.
Reopen only with a label-free explanation of the seen-frame difference (EXPERIMENTS section 16),
or a change to the model or the simulator that is decided on the proxy first.


## D27. E13: the broadcast is not a second ball source; its cut-aways are the E13 track (2026-11-06)

**Decision.**
- **The broadcast alignment is `solution/broadcast.py`**, and its parameters are fixed.
  It is algorithmic: no hand offsets. Any BAR-ATH use of the broadcast goes through it (or
  `cache/broadcast/BAR-ATH_align.npz`, gitignored, written by `make broadcast-align`).
- **The second ball source is closed at Step 0.** The broadcast raises graded ball visibility only
  from 59.3% to 63.6%, because 53% of the tactical ball's missing time is TV cut-away. Even the
  optimistic proxy bound (`BALL_VISIBLE=0.636`, whole gaps at random) gains +0.0035, against
  PLAN's 0.02 stop rule (EXPERIMENTS section 17). Reopen only with a source that sees the ball
  while the TV is cut away.
- **The E13 track is the cut-away cue.** Off-pitch share is 61% on predicted D against 10% on
  predicted live. The implied rates are P(off | D) about 0.71 and P(off | live) about 0.01, under an
  unverified independence assumption. Before any decoder work, its worth is measured on the proxy
  with a synthetic cue. That cue's rates and timing come from label-free BAR-ATH aggregates and are
  fixed before its first proxy run.

**Why.** The ball coverage the broadcast adds sits mostly in 3 to 10 s gaps, and long gaps are
cut-aways. The TV's cut-aways themselves track dead ball strongly, and the signal comes from
editing, not from the position features that broke D26's BAR-ATH check.

**Update (2026-11-06).** miha reopened the ball merge to build it alongside the cut-away cue. It is
built: `broadcast.fuse_ball`, `solution/sources/fused.py`, and the label-free BAR-ATH check in
`tools/broadcast_align.py` section H. The paired proxy run gives +0.0038 (+0.0006..+0.0102, 5/5;
EXPERIMENTS section 17): better, below D4's bar. **It is not submitted.** It may ride along only
if a later change that does get ADOPT is measured with it, and only if the BAR-ATH section H check
shows no harm.

## D28. Decoding for spellF1 (#6) is closed at Step 0; the sampler stays as a diagnostic (2026-11-06)

**Decision.** No MBR decoding and no merging of low-support possessions. The decoder keeps
Viterbi. `decoder.forward_sums` / `backward_sample` / `sample_half`, `predict.sample` and
`tools/mbr_check.py` stay in the repo as diagnostics: posterior support per possession is
available to future work. Do not reopen #6 by trying other temperatures, thresholds or merge
rules on the proxy; reopen only if the evidence changes enough that the decoder's own spellF1
estimate comes near its actual score (`mbr_check.py` prints both).

**Why.** EXPERIMENTS section 18. MBR keeps Viterbi on 29 of 30 match-seed runs; merges lose at
every threshold, and the oracle merge, which reads the truth, loses -0.022 (keep2) and -0.047
(DFL). The decoder estimates its own spellF1 at 0.88 and scores 0.49: its errors are confident
ones in the evidence, which no decoding of the same posterior can undo.

## D29. Better ball tracking (#4): the value is a sharper ball, not more ball; no gap-filling detector (2026-11-06)

**Decision.**
- PLAN E12 (a video ball detector inside ATD gaps) is not pursued: raising the visible share from
  0.60 to 0.76 is worth +0.011, to 0.90 +0.022, an optimistic bound (EXPERIMENTS section 18).
- What remains of #4 is a sharper ball where the ATD already sees it: removing D12's 0.26 s
  smoothing is worth +0.046 (keep2) and +0.061 (DFL), a ceiling for a perfect tracker. Nothing
  is built for it until `tools/ball_lag_check.py --bar-ath` confirms how the BAR-ATH ball is
  blurred or lagged: D12's smoothing was fitted from aggregates, never measured directly.
- `make proxy BALL_VISIBLE=<share>` is a what-if for coverage questions (E13 uses it as an
  optimistic bound). It is not an adoption stress, and D12 stays the deciding ball.

**Why.** The paired gains above, 5/5 matches each. A detector of a sharper ball would bring its
own noise, lag and false balls, and could only be judged on the proxy through a simulated
profile of it, so its realistic gain is below the ceiling.

## D30. #2 adversarial validation is closed: no simulator fix explains D26's seen-D sign; `_dm` / `_dms` stay unsubmitted (2026-11-06)

**Decision.**
- The D26 update 3 reopening path through a label-free explanation is tried and closed. BAR-ATH's
  seen-ball frames differ from the proxy's (speed spread, players near the goal line), but those
  differences do not move the dead-ball model the way BAR-ATH shows. They are also not simulator
  settings. `touch_c3_seen_w1_dm` and `_dms` stay registered, not submitted. D26's ban on a third
  where/how-much variant stands.
- The proxy league stays the decision tool for seen-ball frames: R2 found no simulator-set driver
  there.
- PLAN #1's simulator change "more ball gaps in dead ball" is dropped: seen-frame speed ranks
  already match BAR-ATH. The gap found instead (player losses tied to ball gaps and the
  touchlines) is noted, not built; build it only if a decision that reads players in gaps needs it.
- `tools/adversarial_check.py` stays as the check for any later simulator version. Its reading
  rules are in its docstring; amend them only before a BAR-ATH run, and say so.
**Why.** EXPERIMENTS section 19: median shifts of the driver features move the proxy model by
-0.009 and +0.022 against BAR-ATH's -0.050. Position noise, the one simulator knob that could
raise speed spread, lowers it.

## D31. Cut-away cue: the decision variant becomes one calibrated to BAR-ATH's real split; an exception to the rule fixed before the first run (2026-11-06)

**Decision (miha).**
- **Why an exception.** Section 17's pessimistic variant, which was fixed to decide, missed D4 by
  0.0013. Its own realism check showed it noisier than BAR-ATH's real cue. It is off pitch on 13.0%
  of predicted live against BAR-ATH's 9.6%, and 67.8% of predicted D against 61.2%. The optimistic
  variant errs the other way (7.2%, 57.7%). Choosing the decision variant after seeing results is a
  rule change, recorded here like D14.
- **The calibrated variant, fixed before its first scored run.** It keeps every part of the
  section 17 imitation except the two quantities the pessimistic and optimistic variants
  disagreed on: the spill probability past the restart and the share of live frames with false
  cut-aways. Those are set by `tools/cutaway_calibrate.py`, which reads no score. Against the
  reference's own predictions on the proxy inputs (D12, dev x seeds 0-2), it picks from a fixed
  grid the pair that best reproduces BAR-ATH's 0.612 (predicted D) and 0.096 (predicted live),
  and the pair is written into `cutaway.VARIANTS["calibrated"]`. The matching assumes our method
  errs alike on the proxy and on BAR-ATH. That is the same assumption section 17's implied rates
  made.
- **What passes.** `touch_c3_seen_w1_tv_cal` (`tv_w` 1, calibrated table) against
  `touch_c3_seen_w1` on keep2 + D12 under D4 unchanged (gain >= 0.02, interval excludes 0, wins
  >= 2/3), and DFL not WORSE. One run, no second calibration. If it passes, it is still not
  submitted until a label-free BAR-ATH check on the real cue passes, with criteria fixed before
  that check runs.

**Update (2026-11-06): ADOPT, and the BAR-ATH criteria, fixed before the check runs.**
`touch_c3_seen_w1_tv_cal` gets ADOPT against `touch_c3_seen_w1` on keep2 + D12: +0.0304
(+0.0211..+0.0449, 5/5), reproduced by `make proxy`. DFL: +0.0163, not WORSE (EXPERIMENTS
section 17). On BAR-ATH the method reads the real cue from `cache/broadcast/BAR-ATH_align.npz`
(`cutaway.cue_for`), so `make broadcast-align` must have run on that machine. Commands:
`tools/deadball_check.py --bar-ath --methods touch_c3_seen_w1 touch_c3_seen_w1_tv_cal` and
`tools/gap_check.py --decoded --bar-ath --methods touch_c3_seen_w1 touch_c3_seen_w1_tv_cal`.
Every "reference" number comes from the same run. Proxy values (keep2, D12, `--proxy-side` /
`--decoded`) are in brackets.
1. (D15) Decoded D share within the IDSSE range 0.41 to 0.48. (0.435 -> 0.419; truth 0.441.)
2. Stoppages per 90 fall and their median length rises against the reference. (126.3 -> 98.0,
   11.4 -> 17.2 s; truth 97.1, 19.0 s.)
3. (D15) Smart Tagging set pieces with a decoded restart inside, all types pooled: at least as
   many as the reference. (Restart events as tags: 0.36 -> 0.39.)
4. (D22) Smart Tagging team agreement in gaps at least the reference's; with the ball seen, not
   below 0.80.
5. (D26 rule 4's form) Gap D changes with the proxy's sign and at most twice its size: -0.080..0
   (0.751 -> 0.711). Seen D, unchanged on the proxy (0.223 -> 0.223), moves by at most 0.02
   either way.
6. Warning only: possessions per 90 should fall by about the proxy's factor 0.951
   (281.7 -> 267.9). Below 0.85x of the reference, or any rise, is reported to miha.
7. Warning only: gap home share moves by at most 0.03 (0.526 -> 0.536).
Rules 1 to 5 must all pass for `solution/run.py` to switch to `touch_c3_seen_w1_tv_cal`, and
only on miha's decision. The submission then also depends on the alignment cache, so
`solution/run.py` must build it itself before the final version (Phase 4, byte-identical rule).

**Update (2026-11-06): the BAR-ATH check fails; not submitted.** Rules 1 and 2 pass. Rule 3 fails
(53 against 54 of 58 set pieces), rule 4 fails (tag agreement in gaps 0.734 -> 0.632), and rule 5
fails (gap D +0.004). Warning 7 fires (gap home share 0.480 -> 0.411, against +0.010 on the
proxy). `touch_c3_seen_w1_tv_cal` stays registered. `solution/run.py` is unchanged. Any further
TV variant needs miha's decision and must state, before it runs, what it changes about the
in-gap team shift. A variant chosen to pass these rules after seeing them is not allowed
(D26's precedent).

## D32. The ball merge under the TV method: one variant, its hypothesis stated before it runs (2026-11-06)

**Decision (miha).** Try the ball merge with the TV cue: `touch_c3_seen_w1_tv_cal_fused` =
`touch_c3_seen_w1_tv_cal` reading a merged ball (`solution/sources/fused.merged`). That is
`broadcast.fuse_ball` on BAR-ATH and `broadcast.imitate_fill` on a degraded proxy match (restored
frames there skip the D12 stress: slightly optimistic). `touch_c3_seen_w1_fused`, the merge
alone, is reported alongside. Settings: none new. The fill rule (D27 update) and the TV
method (D31) are used as fixed.

**What it should change, stated before it runs (D31 update's condition).** The TV method failed
BAR-ATH on the team inside ball gaps: tag agreement 0.734 -> 0.632, home share 0.480 -> 0.411,
and short in-gap possessions halved. The merge gives about 31% of the frames of 3 to 10 s gaps a
real ball (section 17, H), and so direct possession evidence there. **Prediction:** on BAR-ATH,
`_tv_cal_fused` keeps more of the in-gap team than `_tv_cal`: tag agreement in gaps and gap home
share closer to the reference's. If they are not, the hypothesis is wrong and E13 stops here.

**What passes.** On the proxy, D4 against `touch_c3_seen_w1` on keep2 + D12 (`make proxy`), and
DFL not WORSE. On BAR-ATH, D31's rules 1 to 5 unchanged, with this method's own proxy values
(`--proxy-side`, `--decoded`, same runs). Warnings 6 and 7 as in D31. One run each; no further
variant of this combination.

**Update (2026-11-06): ADOPT on the proxy; the BAR-ATH check, fixed before it runs.**
`touch_c3_seen_w1_tv_cal_fused` +0.0382 (+0.0311..+0.0560, 5/5) on keep2 + D12; DFL +0.0256, not
WORSE (EXPERIMENTS section 17). Commands (one run, all three methods, so every number comes from
it): `tools/deadball_check.py --bar-ath --methods touch_c3_seen_w1 touch_c3_seen_w1_tv_cal
touch_c3_seen_w1_tv_cal_fused` and `tools/gap_check.py --decoded --bar-ath --methods` with the
same three. D31's rules, against `touch_c3_seen_w1`, with this method's proxy values:
1. D share in 0.41..0.48 (0.435 -> 0.418).
2. Stoppages per 90 fall and their median rises (126.3 -> 98.3, 11.4 -> 17.0 s).
3. Set pieces with a decoded restart: at least as many as the reference (0.36 -> 0.39).
4. Tag agreement in gaps at least the reference's; with the ball seen, not below 0.80.
5. Gap D change in -0.082..0 (0.751 -> 0.710); seen D within 0.02 either way (0.223 -> 0.223).
6. Warning: possessions per 90 fall by about x0.965 (281.7 -> 271.9); below 0.85x, or any rise.
7. Warning: gap home share moves by at most 0.03 (+0.007).
And D32's hypothesis, against `_tv_cal` in the same run: tag agreement in gaps higher and gap home
share nearer the reference's. If the hypothesis fails, E13 stops, whatever rules 1 to 5 say.
Rules 1 to 5 and the hypothesis must all hold for `solution/run.py` to switch, on miha's decision.

**Update (2026-11-06): the hypothesis fails; E13 is closed.** On BAR-ATH,
`touch_c3_seen_w1_tv_cal_fused` fails rules 3, 4 and 5 as `_tv_cal` did, and against `_tv_cal` it
lowers tag agreement in gaps (0.632 -> 0.610) and leaves the gap home share at 0.411. The ball
merge does not repair the TV cue's in-gap team shift. Under this entry's own condition, E13 stops.
`solution/run.py` is unchanged.
- Kept, registered and not submitted: `solution/broadcast.py` (alignment, fuse_ball,
  imitate_fill), `solution/cutaway.py`, `solution/sources/fused.py`, the decoder's `tv_w`
  (off by default), `merge_ball`, the methods `touch_c3_seen_w1_{tv,tv_opt,tv_cal,fused,
  tv_cal_fused}`, and `tools/broadcast_align.py`, `broadcast_fill_check.py`, `cutaway_check.py`,
  `cutaway_calibrate.py`.
- Do not reopen with another TV or merge variant tuned against these BAR-ATH aggregates. Reopen
  only with new evidence about the in-gap team: for instance, a label-free reason to distrust the
  Smart Tagging team in gaps, or a proxy that reproduces BAR-ATH's 12.7 short in-gap possessions
  per 90 (the proxy has 2.6).

## D33. End of half: the graded blocks run past the whistle; a players-only rule is rejected, a whistle detector gets a Step 0 (2026-11-06)

**Failure (documented, not representable on the proxy).** BAR-ATH's tracking blocks, which are the
graded halves, run on after the half-time and full-time whistles while the players leave or
linger. Those frames are D, and `touch_c3_seen_w1` decodes possessions there
(`tools/halfend_check.py --diagnose --bar-ath`, EXPERIMENTS section 20). In half 1, the
reference starts 6 possessions after the whistle. IDSSE tracking stops at the whistle, so the
proxy league has no such frames.

**Decision (miha).**
- **A players-only end-of-half rule is rejected.** Mean player speed does not tell the walk-off
  from slow live play. Sprints are cleaner, but sprint-free stretches still end IDSSE halves.
  Few tracked players to the end of a half looks like the cue, but simulator v1's dropout
  episodes, calibrated to BAR-ATH, make equally deep collapses at plain IDSSE half ends. Players
  beyond the lines also occur on IDSSE for up to 35 s. Stacking guards would fit every threshold
  to one BAR-ATH walk-off, and half 2 has no collapse at all.
- **Whistle Step 0 instead (E10 reopened for this case only; D26's closure of E10 for
  in-play stoppages stands).** `solution/whistle.py` finds tonal 2-5 kHz events in the tactical
  audio. It decodes with PyAV, which is now pinned. Its STRICT and LOOSE settings are fixed in
  the module, from tests on synthetic stadium audio, before any BAR-ATH run, and are not to be
  tuned on BAR-ATH. `tools/whistle_check.py` lists the events at the end of each half, events around given frames, and every STRICT event. It also
  prints a candidate pick, fixed in its docstring: the last group of STRICT blasts (gaps <= 3 s)
  at least 45:00 after the half's first graded frame.

**Step 0, fixed before the run.** One run of `tools/whistle_check.py`. It passes if the
candidate pick lands within 1 s of the half-time and the full-time whistle; the kickoff whistles
found and the share of STRICT events that are the referee's are reported as context. On a pass,
Step 1 builds the rule: D from the pick to the half's last graded frame, nothing changed when no
pick exists, adopted under D14 (the plain proxy stays byte-identical, a synthetic walk-off stress
with a synthetic whistle cue, and a BAR-ATH confirmation). On a failure the whistle route closes.

**Update (2026-11-06): Step 0 fails; the route is closed unless miha supersedes this entry.**
- The detector finds the half-time whistle (two STRICT blasts at about 3.75 kHz, tonality 25-30
  dB) and both kickoff whistles (3.9 kHz, 1.4-1.5 s before the kicks).
- The pick fails. STRICT runs at 4.8 and 4.2 events/min, roughly ten times a referee's rate,
  mostly on crowd whistling after the half-time whistle (2.3-2.9 kHz), so the pick lands 22 s
  before the end of half 1 instead of on the whistle. In half 2, the last STRICT blast is 0.4 s
  before the end.
- Observed, not tested: the referee-like events sit at 3.6-4.0 kHz, the crowd lower. A variant
  keeping only the referee's pitch would be designed after seeing this run, so it needs miha's
  explicit decision (see EXPERIMENTS section 20).

**Update (2026-11-06): closed (miha).** The pitch variant is not built. Its expected gain is
about +0.003 to +0.004 spellF1, most likely in half 1 only, which is under EVALUATION.md's 0.01
noise level. It would also be designed after seeing the run. Kept, registered and not used by any
method: `solution/whistle.py`, `tests/test_whistle.py`, `tools/whistle_check.py`,
`tools/halfend_check.py`, and PyAV in the lock. Reopen only as late polish before the final
version, with evidence that pitch separates the referee from the crowd.


## D34. #4 Better ball tracking: the BAR-ATH ball has no lag, so there is no time-shift fix; D12 stands until miha decides (2026-11-06)

**Decision.**
- **No time shift of the ball.** `tools/ball_lag_check.py --bar-ath` (miha's run, aggregates
  only) puts BAR-ATH's best ball-to-dribbler shift at -0.143 s, the same as clean IDSSE's
  -0.142 s; a past-only tracker would show +0.067 s. Its kick rise is symmetric (late/early 1.02,
  past-only 1.96): the blur is centred. This closes what D29 left of #4 as a cheap fix; a sharper
  ball would need our own tracker, which is not planned.
- **D12's 0.26 s smoothing stays the decision stress for now.** BAR-ATH's kick rise (10-90% p50
  0.692 s, p10 0.548, p90 0.858) sits between D12's 0.26 s (0.535 s) and 0.40 s (0.760 s): about
  0.36 s by interpolation, a statistic D12 was not fitted on. Under 0.40 s the submitted
  reference scores 0.4596 against 0.4864 (EXPERIMENTS section 21), so D12 is somewhat optimistic
  for methods that read touches. Moving it is miha's call, for every method at once, with the
  registered candidates re-run; it must not be done to rescue one method (D35).
**Why.** EXPERIMENTS section 21 ("Why now") and the reference rows of `tools/ball_lag_check.py`.


## D35. #3 control model is closed: two settings tried, below D4 and WORSE on DFL; not submitted (2026-11-06)

**Decision.** `touch_c3_seen_w1` stays submitted and the reference. `touch_c3_seen_w1_ctl` and
`touch_c3_seen_w1_ctl_p05` (`solution/controlmodel.py`: a per-touch P(contradicted), touches above
a threshold left out of the holder) stay registered, not submitted. **Do not try a third
threshold, feature set or training mix on the proxy:** the rule fixed before the diagnostic
allowed one variant and it has run. `touch.sweep_holder`'s keep masks stay (no mask is
byte-identical) for diagnostics.
**Why.** EXPERIMENTS section 21.
- Step 0: dropping every touch the keep2 truth contradicts is worth +0.075; a random imitation at
  70% recall and 10% false drops +0.042. The bar passed.
- Step 1: the model reaches 0.79 recall at 0.11 false drops (AUC 0.93) on held-out matches, yet
  gains only +0.0083 (within noise) and is WORSE on DFL (-0.024).
- The diagnostic: its true drops alone are worth +0.056, but its false drops sit on the real
  hand-overs that matter (costing 0.048 against 0.005 for random ones at the same rate).
- The one allowed variant (5% false-drop target): +0.0191 (+0.0115..+0.0295), 5/5, 0.0009 under
  D4's bar, and WORSE on DFL (-0.013). Under 0.40 s smoothing it is ADOPT (+0.027), reported only.
**Reopen only** if D12 is moved on its own evidence (D34) and the registered `_ctl_p05`, unchanged,
then gets ADOPT with DFL not WORSE; or with a label source where keep2 and DFL do not disagree
on these touches.

## D36. Decoder costs stay 0 after a re-tune on the current evidence; the L3 stability check is a standing check; E8 is open at Step 0 (2026-11-06)

**Decision.**
- **Transition costs stay 0 and `dead_min_s` stays None** (D8 stands). EXPERIMENTS section 22
  scored 9 settings on keep2 + D12: none reached D4. `dead` 4 is the one gain with the same sign
  on keep2 (+0.0066, 5/5) and DFL (+0.0018); the `switch` cost's keep2 gain vanishes on DFL
  because it merges real possessions. Do not run another cost grid. `dead` 4 may be tried once
  more only inside a stack with the other sub-bar 5/5 gains (keep2 priors, ball merge,
  player-beyond-line cue; REVIEW.md item 4), decided under D4 with DFL not WORSE.
- **`tools/stability_check.py` is a standing check** (PLAN L3). Any newly adopted method is run
  through it on the proxy and on BAR-ATH before `run.py` switches to it; a worst possession-count
  change above 30% over the +-20% weight rows blocks the switch, above 10% is reported with the
  decision. `touch_c3_seen_w1`: 6.6% on the proxy, 7.8% on BAR-ATH, plateau (section 23).
- **E8 (set-piece tags as evidence) is open at Step 0, case 2** (section 24): the tag opens about
  3 s before the restart kick. Step 1 may be built only as the section 24 rule says (synthetic
  tag shifted by the measured median, D before the start and a restart bonus after it, D4, DFL
  not WORSE, BAR-ATH rules fixed first); team phases stay a validation signal only. Building it
  is miha's call; the expected gain is +0.01 to +0.02.
**Why.** Sections 22 to 24. The three items were the review's cheap, proxy-decidable ones; the
rules for each were written before its first run, and each has run once.

## D37. The learned dead-ball model is re-decided on the balance of label-free evidence; touch_c3_seen_w1_dm is the candidate, switched after two pre-registered checks (2026-11-06)

**Why a re-decision.** D26 blocked `touch_c3_seen_w1_dm` and `_dms` on one rule (4): the
direction of change of gap D and seen D on BAR-ATH had to match the proxy's. That rule assumes
the submission errs the same way on BAR-ATH as on the proxy, and was marked "adapted, miha to
confirm" when written. Everything else about the model points one way (EXPERIMENTS section 16,
REVIEW.md section 6 item 1): ADOPT on the decision run (+0.0702, 5/5) and on every stress built in
this project, including DFL truth (+0.048), TEMPO (+0.070), NOISE (+0.072) and the busy
pre-kickoff (+0.069); on BAR-ATH it takes the stoppage count from 140 per 90 of median 9.4 s to
97 of median 18.1 s, the IDSSE truth's own figures (97, 19.0 s), keeps the D share (0.444), keeps
every set piece covered, raises tag agreement in gaps (0.734 to 0.771) and leaves the gap home
share alone (0.473). False stoppages are the largest single loss on both the proxy (+0.046) and
the public window (two of 2.5 s, REVIEW.md section 4), and this model removes exactly those. The
public window's truth has a seen-D share of 0.147 against the reference's 0.143 there, so the
window gives no reason to expect BAR-ATH's seen D to need raising, which is what rule 4 demanded.
D26 and D30 forbid a third variant chosen to pass; they do not forbid re-weighing the two that
exist, and no new variant is made here.

**Decision (miha, 2026-11-06, on the review's recommendation).** Rule 4 of D26 becomes a
warning, recorded with the switch, not a veto. The candidate is `touch_c3_seen_w1_dm` (the full
gain; its gap move is the open question). `touch_c3_seen_w1_dms` (+0.0362, gaps untouched) is the
fallback. `solution/run.py` switches only after both checks below pass, run once each:
1. **L3 stability (D36)** of `touch_c3_seen_w1_dm` on the proxy (`tools/stability_check.py
   --method touch_c3_seen_w1_dm`) and on BAR-ATH (`--bar-ath`): worst possession-count change
   over the +-20% weight rows under 30% on both; above 10% is reported with the switch.
2. **Public-window correctness (L4)** on the file `make submission METHOD=touch_c3_seen_w1_dm
   TEAM=our_team_dm` writes: `make eval` shows no team swap (share error under 0.03), the
   abstention count 0, and no new stoppage inside the window's five long possessions (frames
   11624-12665, 12989-13704, 13785-14846, 14846-15346, 16400-17293; the submission has false
   stoppages at 12425-12485 and 14450-14515, which the model may remove). Its spellF1 on the
   window is read and recorded, never compared to choose.
If check 1 fails for `_dm`, the same two checks run for `_dms`; if both fail, `touch_c3_seen_w1`
stays and this entry records why. On a pass: `run.py` switches, two runs are hashed, the
stability and window numbers go into EXPERIMENTS section 16, and `_dm` becomes the reference for
the proxy league. Phase 4's hold-out run (D3) scores the old and new references once.
**Update (2026-11-06): both checks pass; `solution/run.py` submits `touch_c3_seen_w1_dm`.**
Check 1: worst possession-count change 4.5% on the proxy and 6.3% on BAR-ATH (EXPERIMENTS
section 23). Check 2 on `our_team_dm.csv`: share error 0.022, 0 abstentions, no stoppage inside
the five long window possessions; the submission's two false stoppages are gone and the 5.5 s
stoppage at 12851 is missed instead (section 4). Rule 4 of D26 stands recorded as a warning: on
BAR-ATH the model moves D into gaps (+0.041) and off seen frames (-0.020), the proxy's opposite
direction, and the possession count falls to 287 (0.865 times the old 332; proxy factor 0.93).
`touch_c3_seen_w1_dm` is the reference for the proxy league from now on (0.5565 on the decision
run); `touch_c3_seen_w1` stays registered. Two `make submission` runs on BAR-ATH
(miha, 2026-11-06) give the same sha256, `a4c78a40…208a2133`; the file validates with 287
possessions of mean 12.3 s.

## D38. Stacking: the closed parts are re-measured on top of the submission, on the proxy only (2026-11-07)

**Decision (miha, 2026-11-07).** The parts that were closed one at a time (TV cue, merged ball,
beyond-line cue, keep2 priors, dead cost, snapping, control model) are put on top of
`touch_c3_seen_w1_dm`, singly and together, and scored on the proxy league under keep2 + D12 and
DFL (EXPERIMENTS section 25, grid and reading rule written before the run). This reopens the
*measurement* of the TV and merge parts, not D31's or D32's verdicts: any candidate that reads the
broadcast still has to pass D31's BAR-ATH rules 1 to 5, and D32's bar on reopening with a variant
tuned to those aggregates stands. Nothing is submitted from this entry without that check and
miha's decision.
**Update (2026-11-07): the runs are done; nothing switches yet.** EXPERIMENTS section 25: no
closed part reaches D4's bar alone on top of the submission; stacked they do. `dm_all` (merged
ball, TV cue, keep2 priors, beyond-line cue, dead cost, snapping) is +0.0216 on keep2 + D12, 5/5,
and +0.0106 on DFL; `dm_stack` (merged ball, TV cue, keep2 priors, control model) is +0.0372 on
keep2, 5/5, and -0.0009 (noise) on DFL with 0.82 of the truth's possessions. Both are ADOPT under
D4 and both read the broadcast. Before either can be submitted: D31's BAR-ATH rules 1 to 5 on
the candidate (miha runs `tools/deadball_check.py --bar-ath` and `tools/gap_check.py --decoded
--bar-ath` with the reference and the candidate), run.py building the alignment cache itself, and
miha's decision between the safer `dm_all` and the larger, merge-heavier `dm_stack`.
**Update (2026-11-07, after D39): the broadcast stacks fail the BAR-ATH check; not submitted.**
Against `v7_keep2_control`, `v6_plus_stack` (was `dm_stack`) fails rules 3 and 4 and
`v6_plus_all` (was `dm_all`) fails rule 4: tag agreement in gaps 0.790 -> 0.692 and 0.667, gap
home share -0.057 and -0.050 (warning 7), the in-gap team shift of D31 and D32 again and larger
(EXPERIMENTS section 25, last table). The stacking did not change what the TV cue does inside
ball gaps. E13 stays closed under D32's condition; `solution/run.py` stays on
`v7_keep2_control`. The alignment-cache work in run.py is not needed.
**Update (2026-11-07, last): the merged ball alone on v7 (`v7_plus_ball`, section 25 run 4).**
Proxy +0.0041 (keep2) and +0.0053 (DFL), 5/5 each, below D4 and at section 17's size; BAR-ATH
rule 4 fails (tag agreement in gaps 0.790 -> 0.771; short in-gap possessions doubled). D27's
ride-along clause needs no harm on BAR-ATH, so it is refused. Section 25 and the broadcast
question are closed: four label-free measurements on BAR-ATH, all moving the in-gap team the
wrong way. Reopen only under D32's condition.
**Update (2026-11-07, later): the broadcast-free stack passes; the switch is miha's call.**
`dm_k2ctl` = `touch_c3_seen_w1_dm` + keep2 priors + control model (false_drop 0.05), registered
before its run (section 25, run 3): +0.0289 (+0.0203..+0.0397, 5/5) ADOPT on keep2 + D12;
-0.0076 (-0.0141..+0.0039, 1/5) within noise on DFL + D12; 1185 possessions (0.855 of the
reference's). On BAR-ATH (miha's run of the two D31 commands) rules 1 to 5 pass: D share 0.443,
stoppages 95.2 per 90 with median 18.5 s, set pieces with a restart 0.93 of 58 unchanged, tag
agreement in gaps 0.771 -> 0.790 and seen 0.869 -> 0.890, gap D and seen D unchanged. Warning 6
fires: possessions per 90 243.9 -> 200.6, x0.822 against the proxy's x0.855. It reads no
broadcast, so the alignment cache and D31's in-gap question do not arise. Under D4 and this
entry it is the submission candidate; `dm_all` and `dm_stack` stay registered behind it. What
the switch bets on: the organisers label controlled possession (D21), on which the control model
gains, rather than first touches, on which it costs about 0.008. `solution/run.py` switches only
on miha's decision; then `make lock`, the byte-identical check and a `docs/SUBMISSIONS.md` row.

## D39. The submission switches to `dm_k2ctl`: keep2 priors and the control model on top of the dead-ball model (2026-11-07)

**Decision (miha, 2026-11-07).** `solution/run.py`'s default becomes `dm_k2ctl` =
`touch_c3_seen_w1_dm` + the decoder's priors fitted on the keep2 controlled-possession relabels
(D21) + the control model with `false_drop` 0.05 (D35), the stack registered before its run as
EXPERIMENTS section 25, run 3. Evidence: +0.0289 (+0.0203..+0.0397, 5/5) ADOPT on keep2 + D12;
-0.0076 (-0.0141..+0.0039, 1/5) within noise on DFL + D12; on BAR-ATH the label-free rules 1 to 5
of D31 pass and warning 6 fires (possessions per 90 x0.822; the D38 update has the numbers). It
reads no broadcast.
- **What it reopens.** D35 closed the control model measured alone on `touch_c3_seen_w1` (below
  D4, WORSE on DFL) and D21/D25 closed the keep2 priors alone (+0.0189, below D4). This entry is
  their re-measurement together, on top of the dead-ball model, in D38's frame; neither verdict
  is overturned for the part alone.
- **What it bets on.** The organisers label controlled possession (D21, the challenge text). On
  first-touch labels the stack costs about 0.008 on the proxy and removes about a sixth of the
  possessions on BAR-ATH, mostly brief in-gap spells that disagree with the Smart Tagging team.
- **What stays.** `touch_c3_seen_w1_dm` registered as the fallback, its file kept as
  `evaluation/results/our_team_dm.csv` (sha256 `a4c78a40...`). `dm_all` and `dm_stack` registered,
  not submitted: they read the broadcast and D31's rules are unmet (E13 closed, D32).
- **Before Phase 4.** `make submission` twice and the bytes compared; `make eval` as the L4
  sanity check (no team swap on the public window, never a tuning signal); the
  `docs/SUBMISSIONS.md` row. No new dependency: `requirements-lock.txt` is unchanged.
- **Not to undo.** The switch is not reverted on the public window's scores. It is reverted
  only on a recorded proxy result under D4 or on the Phase 4 hold-out run (D3).

**Update (2026-11-07): regenerated and checked.** miha ran `make submission` twice (`our_team`,
`our_team_check`, the latter kept as `evaluation/results/v7_keep2_control.csv`): byte-identical,
sha256 `e3ba84bf…`; 236 possessions of mean 15.0 s by the validator's count. L4 on the public
window (`make eval`, `make frame-report FILES="v6_deadball_model our_team"`): no team swap, share
error 0.022 -> 0.005; macroF1 0.8474 -> 0.8531, wrong frames 13.7% -> 12.9%, team right where
both alive 0.922 -> 0.936; spellF1 0.6533 -> 0.6286. The spell loss is one true possession of
0.8 s at 10:13.8 that the control model folds into the surrounding away possession, the trade
this entry names; with 17 possessions in the window one spell is about 0.03. Under this entry's
own rule the window's score is recorded and does not decide. `solution/run.py` stays on
`v7_keep2_control`.

## D40. Names: the submissions are `v1_decoder` to `v7_keep2_control`; former names stay as aliases (2026-11-07)

**Decision (miha, 2026-11-07).** The lineage and the stacking family are renamed so a reader can
tell from the name what a method is and in which order it was submitted. The result files
follow the same names. Every former name stays registered in `solution/methods.py` as an alias
of the same function (`ALIASES`, `canonical`), so EXPERIMENTS, DECISIONS, the worklog, cached
proxy results and old commands resolve unchanged; tools and `make methods` print the new name
with the former one in brackets. The submission file keeps the organisers' `<team_name>.csv`
rule (`our_team.csv`, `TEAM` in the Makefile).

| new name | former name | result file |
|---|---|---|
| `v1_decoder` | `semi_markov` | `v1_decoder.csv` (was `e1_semi_markov.csv`) |
| `v2_touch` | `touch_markov` | `v2_touch.csv` (was `e3_touch_markov.csv`) |
| `v3_kickoff` | `touch_kickoff` | no file kept |
| `v4_spread` | `touch_gapteam` | no file kept |
| `v5_frame_model` | `touch_c3_seen_w1` | `v5_frame_model.csv` (was `c3_seen_w1.csv`) |
| `v6_deadball_model` | `touch_c3_seen_w1_dm` | `v6_deadball_model.csv` (was `our_team_dm.csv`) |
| `v7_keep2_control` | `dm_k2ctl` | `our_team.csv` once `make submission` has run (D39) |
| `v6_plus_<part>` | `dm_<part>` (tv, ball, tvb -> tv_ball, cue, k2 -> keep2, dead4, snap, ctl -> control, all, stack) | none |
| `touch_c3_seen_w1_tv_cal` | unchanged | `v5_plus_tv_cue.csv` (was `our_team_tvcue.csv`) |
| `touch_c3_seen_w1_tv_cal_fused` | unchanged | `v5_plus_tv_cue_merged_ball.csv` (was `our_team_tv.csv`) |

Closed experiment methods keep their names; `docs/SUBMISSIONS.md` section 4 lists every one with
its parts. Entries before this one use the former names. Do not drop the aliases: the history
is written in them.

## D41. E15 ball inference in gaps is closed: even the true ball adds nothing (2026-11-07)

*Numbered D38 on `denis/ball-impute` (PR #9); renumbered at the merge into `miha/dev` on 2026-11-07 because `miha/dev` had taken D38 to D40. Its EXPERIMENTS sections moved the same way: 25 to 27 there are 26 to 28 here.*

**Decision.** No ball imputation in gaps, by any learner. `solution/ballmodel.py` and the
`touch_c3_seen_w1_dm_ball` / `dm_bor_*` methods stay registered as the record and the oracle
tool; `solution/run.py` is unchanged. Do not reopen E15 (GNN, set transformer, physics fill)
unless the possession reading of a ball changes, since the oracle bounds every imputer.
**Why.** EXPERIMENTS section 26. The true IDSSE ball in every gap, read as the nearer team
through a calibrated table, is within noise of the reference (-0.0009, 2/5); our imputer is
WORSE (-0.0050, 0/5). D29 already put more ball through the whole pipeline at +0.011 to +0.022.
On alive gap frames the C3 frame model reads the team better (0.853) than the true ball does
(0.785). The in-gap team question is also bounded from the model side: the frame model in gaps
is +0.0125 (section 14), its BAR-ATH drop is not missing players (section 13), and self-training
with true labels stays under 0.02 (D24).
Decided by denis on 2026-11-07, with Claude Code.

## D42. The stacked method dm_stack3_dead4 gets ADOPT; it is the candidate, submitted only after BAR-ATH checks fixed here (2026-11-07)

*Numbered D39 on `denis/ball-impute` (PR #9); renumbered at the merge into `miha/dev` on 2026-11-07 because `miha/dev` had taken D38 to D40. Its EXPERIMENTS sections moved the same way: 25 to 27 there are 26 to 28 here.*

**Decision.** `dm_stack3_dead4` (`solution/methods.py`): `touch_c3_seen_w1_dm` plus keep2 priors,
the broadcast ball merge, the player-beyond-a-line cue at 0.5 and the `dead` 4 cost. ADOPT on the
decision run (+0.0206, 5/5), not WORSE on DFL (+0.0097, 5/5). **Registered, not submitted:**
`solution/run.py` switches only on miha and denis's decision after every check below passes.
`dm_stack3` (+0.0195) is not a fallback: it is below the bar.

**BAR-ATH checks, fixed before any is run** (label-free; run by a human; every reference number
from the same run):
- `make broadcast-align` first (the merge reads its cache), then
  `tools/gap_check.py --decoded --bar-ath --methods touch_c3_seen_w1_dm dm_stack3_dead4` and
  `tools/deadball_check.py --bar-ath --methods touch_c3_seen_w1_dm dm_stack3_dead4`.
1. Smart Tagging team agreement inside gaps at least the reference's; with the ball seen not
   below the reference's minus 0.01. (keep2 priors once lowered it in gaps, D21; the merge's
   predecessor failed BAR-ATH on the in-gap team twice, D31, D32.)
2. Decoded D share of graded frames within the IDSSE range 0.41 to 0.48 (D9, D15).
3. Possessions per 90 falling by about the proxy factor (0.957); outside 250 to 393 is a warning.
4. Home share of gap team frames within 0.05 of the reference's.
5. Smart Tagging set pieces with a decoded restart within +-2 s not below the reference's.
6. L3 stability (D36): `tools/stability_check.py` must first be extended to rebuild this method
   faithfully (it supports neither `merge_ball`, `prior_labels` nor the dead-ball cue today);
   worst possession-count change under 30% on the proxy and on BAR-ATH.
7. Public-window correctness (L4, as D37 check 2) on the file `make submission
   METHOD=dm_stack3_dead4 TEAM=our_team_stack` writes: share error under 0.03, 0 abstentions, no
   new stoppage inside the window's five long possessions. Its window spellF1 is recorded,
   never compared to choose.
**On a pass:** `run.py` must build the broadcast alignment itself (or ship it), so a clean clone
reproduces the file (REVIEW section 7); two runs hashed; `dm_stack3_dead4` becomes the reference.
**Why.** EXPERIMENTS section 27. A stack of independent sub-bar gains is a legitimate candidate
(REVIEW item 4), and the bar still applies. It clears it narrowly, so the BAR-ATH checks matter.
Decided by denis on 2026-11-07, with Claude Code.
**Update (2026-11-07): fails rules 1 and 5; not submitted.** denis ran the checks (EXPERIMENTS
section 27): Smart Tagging agreement in gaps 0.768 against the reference's 0.771, set pieces
with a restart 0.90 against 0.93 (throw-ins 0.88 against 0.96); rules 2 to 4 pass. Both
failures are small and both point at the `dead` 4 cost. The rules are not loosened (D22, D32),
and `dm_stack3` is not picked instead: it is below D4's bar, and choosing it after this run
would be choosing on BAR-ATH aggregates. `touch_c3_seen_w1_dm` stays submitted and the
reference. The stack is closed for this phase; do not try further stack variants against these
aggregates.

## D44. Whistle, end of half: reopened with both audio tracks; Step 0 pass criteria fixed before the run (2026-11-10)

**Why reopen (D33 said: only with new evidence).** D33's tactical-audio Step 0
found the referee's whistles but fired about ten times a referee's rate, mostly on crowd
whistling (2.3-2.9 kHz). The broadcast audio is a new input: it has a
different interferer (commentary) and lacks the old one. D26's closure of E10 for in-play
stoppages stands (perfect whistle +0.012 to +0.015, under D4); this entry is the end of half
only, worth about +0.005 (EXPERIMENTS section 20).

**Method, fixed before any BAR-ATH run of it.** `solution/whistle.py` unchanged for one feed
(STRICT, LOOSE). New: `map_events` (broadcast events to tactical frames through
`broadcast.align`'s `live_t`), `feed_lag` (the audio lag between the feeds, from the events), and
`pair_feeds`: a whistle is an event in both feeds, onsets within `PAIR_TOL_S` 0.24 s after the
lag, pitches within `PAIR_HZ` 250 Hz, STRICT on at least one side. The chance rate is the same
pairing with the broadcast shifted by +-10 and +-20 s. Settings chosen on synthetic audio only
(EXPERIMENTS section 31) and not to be tuned on BAR-ATH. `tools/whistle_broadcast_check.py`
(`make whistle-broadcast`, after `make broadcast-align`) prints the report.

**Step 0 passes when both hold.**
1. Of `--spot 10`'s list (every 10th paired event, no picking), at least 0.8 are the referee's
   whistle.
2. The paired rate is at least 3 times its chance rate in each half.
Reported as context: kickoff whistles found; the end-of-half pick per half (D33's rule
unchanged, on paired events); the pitch tables (is the crowd cluster gone?).
**On a pass,** Step 1 is D33's rule (D from the pick to the half's last graded frame, nothing
without a pick), adopted under D14: the plain proxy stays byte-identical (IDSSE has no audio), a
synthetic walk-off stress with a synthetic whistle cue, and a BAR-ATH confirmation. `run.py`
would then decode both audio tracks and build the alignment itself (REVIEW section 7). **On a
failure** the route closes again; no pairing variant is tried against these aggregates.
Decided by denis on 2026-11-10, with Claude Code.

**Update (2026-11-11 to 2026-11-12): a two-tone variant.** The referee's whistle is two steady
tones together. A single-feed two-tone rule (`whistle.two_tone_events`, settings `TT_*`) was
tried on both recordings; its Step 0 did not pass on the broadcast (EXPERIMENTS section 33). The
two scans agree: 46 whistles in both feeds against 0.8 at the shifted offsets.

**Superseded (2026-11-16) by D47.** D47 replaces the two-tone rule with a detector that
calibrates itself from kickoffs found in the tracking, so no setting comes from this match; the
two-tone rule and its scan tool are removed.

## D45. The whistle veto (no whistle, mid-pitch: not dead) is closed at Step 0 (2026-11-12)

**Idea (denis).** Away from the lines almost every stoppage is whistled; throw-ins, corners and
goal kicks are at a line and are not. So a decoded stoppage in mid-pitch with no whistle near it
is likely a player holding the ball, and could be released. This differs from E10 (D26), which
measured whistles placing stoppages, not vetoing false ones.

**Step 0** (`tools/whistle_veto_check.py`, `make whistle-veto`; settings and the closing rule
written in its docstring before the first run): on the proxy (IDSSE dev x 5 seeds, v1 + D12,
keep2), against `v7_keep2_control`, a perfect whistle at the start of every true whistled
stoppage, a veto at M metres from every line. EXPERIMENTS section 34.
- The ceiling is under the bar: repairing every false stoppage of v7 gains +0.0127 (v5: +0.046;
  the dead-ball model took most of it).
- With a perfect whistle the veto loses at every margin (-0.0008 to -0.042, 0 or 1 wins of 5):
  about half the stretches it releases are true stoppages, because decoded stoppages split in
  pieces, the last seen ball before a gap misplaces them, and unmatched restarts count as
  unwhistled. Missed whistles make it worse (-0.0123 at 30% missed).

**Decision.** Closed. Reopen only if a method's false-stoppage oracle (deadball_check's "false"
repair) is at least 0.02 again, and then as a new Step 0. No variant of the veto (window, place
from players, merging pieces) is tried against these numbers: none can beat the +0.0127 ceiling.
Closed by the rule fixed in the tool before the run; recorded by Claude Code on 2026-11-12 for denis.

## D46. The whistle as a decoder cue: registered; reopens D26 (in-play) and D45 at denis's call (2026-11-12)

**Decision (denis).** D26 closed whistles in play on a ceiling and D45 closed the hard whistle
veto. denis reopened both: the methods use information from the real game, built from how
football works (no frame-specific rules), and should be tested on it. `solution/whistlecue.py`
feeds the decoder a fitted per-frame cue (time since the last whistle x the ball's place; table
fitted on the proxy with imitated whistles, never on BAR-ATH). Registered: `v7_whistle`,
`v7_whistle_end`, `v7_whistle_pess`, `v7_place` (control). EXPERIMENTS section 35.

**Results.** Proxy: +0.0081 (keep2, 4/5) and +0.0071 (DFL, 5/5), intervals above 0, below D4's
0.02; the place-only control is noise, so the gain is the whistles'. BAR-ATH label-free, against
D26/D42's yardsticks: D share 0.458 (in range), possessions per 90 x0.95, home share in gaps +0.032 (within 0.05), seen tag
agreement 0.892 (holds); two narrow misses: in-gap tag agreement 0.790 -> 0.775, and one corner
fewer with a decoded restart (53 of 58 against 54).

**Not switched yet.** `solution/run.py` still submits `v7_keep2_control`. Switching to
`v7_whistle_end` is denis and miha's decision (D42). On a switch `run.py` decodes the tactical
audio (about a minute; cached in cache/whistle/), and two `make submission` runs must give
byte-identical files.
Recorded by Claude Code on 2026-11-12 for denis.

**Update (2026-11-15): not adopted.** On the proxy the cue is under D4's 0.02 (+0.008 / +0.007),
so `run.py` keeps `v7_keep2_control`; the methods stay registered for reference. Since D47 they
read the self-calibrating detector on BAR-ATH and the a-priori "prior" variant on the proxy.

## D47. A self-calibrating whistle detector: no setting from this match (2026-11-16)

**Why.** So that no setting of the whistle detector comes from this match. RULES.md forbids
"rules and constants tied to specific frames of the game" and allows "pseudo-labels produced by
an algorithm, provided no person decided any of them". So the whistle is now found by code alone.

**Method** (`solution/whistle.py`, the self-calibrating section; settings fixed here, before any
run on BAR-ATH, chosen from D33's pre-match values, the laws of the game and synthetic audio):
1. Features: the 4 strongest tonal peaks per frame in D33's 2-5 kHz band (pitch, dB above the
   800 Hz of spectrum around it). Generic.
2. Anchors: every kickoff `solution/kickoff.py`'s Law 8 cues find from the tracking, anywhere in
   a half (`kickoff_anchors`). On IDSSE dev x 3 seeds: 35 anchors, all within 10 s of a DFL
   kickoff or goal restart, 0.91 within 3 s (EXPERIMENTS section 36).
3. Calibration: the pitch signature (two tones together, else one; 100 Hz cells; a tone counts
   only if it holds within 60 Hz for 0.3 s) seen in the window 10 s before to 2 s after at least 2
   anchors, and at least 3 times as often there as in the same windows shifted by +-30 to +-120 s.
   No such signature: no whistles at all.
4. Detection: runs in the calibrated band(s) (the tone +-150 Hz), gaps up to 0.12 s, at least
   0.15 s, every tone on together on at least 2 frames, pitch std at most 50 Hz.
The v1 two-tone settings (`TT_*`), `tools/whistle_two_tone_check.py` and
`tools/whistle_spectrum.py` are
removed. `solution/whistlecue.py` reads the new detector on BAR-ATH; its proxy decision variant
is now "prior" (miss 0.3, 0.1 false/min, jitter 0.3 s), set a priori, replacing "measured".

**Step 0 on BAR-ATH** (`tools/whistle_selfcal_check.py`, `make whistle-selfcal`; aggregates only,
no frames, times or listings). Passes when all hold:
1. A calibration exists.
2. Leaving out any one anchor gives the same kind and tones within 100 Hz (with 3 or more
   anchors; with 2 it is not testable and not required).
3. Whistles per minute between 0.2 and 1.5 in each half.
4. In each half, the share of whistle onsets with a `v7_keep2_control` decoded stoppage
   starting 2 s before to 6 s after is at least 2 times its chance (onsets shifted +-30, +-60 s).
Context, no criterion: `--cross`, the broadcast through the same bands, paired against chance.
**On a pass** the whistle methods read these whistles; adoption into `run.py` is still D4 on the
proxy (D46: under the bar), so a pass alone changes no submission. **On a failure** the whistle
stays out, and no setting above is changed against the printed numbers.
Recorded by Claude Code on 2026-11-16 for denis, before any run.

**Step 0 result (2026-11-16): passes.** denis ran `make whistle-selfcal` (aggregates in
EXPERIMENTS section 36). 1: a two-tone signature at 3 of 3 kickoff anchors, 23 x chance. 2: 3 of
3 leave-one-out re-calibrations agree. 3: 0.72 and 0.96 whistles/min. 4: 0.40 of onsets near a
decoded stoppage start against 0.18 and 0.14 by chance (x2.2, x2.9; half 1 is the narrow one).
The whistle methods now read these whistles on BAR-ATH. `run.py` is unchanged (D4).

**Update, registered before its run (2026-11-16): the proxy matched to Step 0.** Up to about 0.5
onsets per minute have no decoded stoppage near them (0.66 and 0.90 onsets/min, 0.60 of them
unmatched): an upper bound on the false rate, 2.5 to 5 times what "prior" (0.1/min) and
"pessimistic" (0.2/min) assume. `v7_whistle_matched` = `v7_whistle` with the "matched" variant
(miss 0.3, false 0.5/min, jitter 0.3 s). On BAR-ATH it reads the same whistles, through a table
fitted for that noisier detector. Reported under D4 like the others; no rate is changed against
its result.
**Result (2026-11-16):** `v7_whistle_matched` +0.0017 on keep2 (interval -0.0040..+0.0110, 3/5)
and +0.0024 on DFL (-0.0023..+0.0094, 4/5): within noise. At the noise Step 0 allows, the
in-play cue is worth about nothing; at the "prior" rate it was +0.0075. Which applies depends
on how many unmatched whistles are real (restarts, repeat blasts), which only `--cross` or audio
from other games with labels can narrow. Not adopted; `run.py` unchanged. What the whistle may
still be worth, the end-of-half rule, the proxy cannot measure (IDSSE halves end at the whistle).

## D48. The end-of-half rule alone (`v7_end`), to be decided under D14 (2026-11-17)

**Why.** The graded halves run past each half's final whistle while the players leave, and v7
decodes possessions there (EXPERIMENTS section 20). Law 7: play ends at the whistle, so every one
of those possessions is wrong, whatever the truth labels say elsewhere. The proxy cannot show it
(IDSSE tracking stops at the whistle), so D4's 0.02 cannot decide it; D14 is the team's route
for exactly such a rule. The in-play whistle cue (D46, D47) stays out: it changes the whole
match and is within noise at the noise Step 0 allows.

**Method, fixed before any run.** `v7_end` = `v7_keep2_control` + `whistlecue.end_of_half` on the
self-calibrated whistles (D47), with no in-play cue (whistle_w 0), so its decode is v7's. Rule
(D33, unchanged): in each half, from the first onset of the last whistle group after 45:00 to the
half's end, D, when that onset is within 120 s of the end. New guard: only if the median
collective speed from the onset to the end is at most 1.7 m/s (halfway between IDSSE's dead-ball
1.02 and open-play 2.33 m/s). Without it, a missed final whistle would let a late foul's whistle
wipe up to 2 minutes of play. On the proxy the whistles are imitated at the noisiest plausible
level ("matched", D47 update).

**Checks** (`tools/halfend_whistle_check.py`, `make halfend-whistle`):
- P1. Without whistles (the plain proxy) nothing changes.
- P2. On plain IDSSE dev halves with imitated whistles ("prior", "matched", "pessimistic" x seeds
  0-4, v1 + D12, keep2), no false fire: no half where the rule turns 1 s or more of truly live
  frames into D.
- P3. On a new walk-off stress (`simulate.add_walkoff`: 30 s and 90 s after each half, players
  linger then walk off at 1.1-1.6 m/s, half the time one carries the ball; all D), not WORSE for
  any variant (mean gain under 0 with at most 1 of 5 seeds winning), better for at least one
  (mean gain over 0 with at least 4 of 5).
- B1, B2 (BAR-ATH, label-free, `--bar-ath`, run by denis). B1: the rule fires in at least one
  half. B2: in each half where it fires, v7 decodes at least one possession after the pick, and
  (where at least 10 s follow the pick) fewer players are inside the pitch after it than in the
  30 s before it.
**On a pass of all five** `v7_end` meets D14, and switching `run.py` to it is denis and miha's
call. **On a failure** it is not adopted, and no setting above changes against the printed
numbers. Recorded by Claude Code on 2026-11-17 for denis, before any run.

**Result (2026-11-17): fails P2 and P3; not adopted.** P1 passes. P2: with the guard, 6 to 10
false fires in 50 plain halves per variant. When the imitated final whistle is missed, a whistle
14-62 s before the true end becomes the pick, play after it is slow enough for the guard, and
4-26 s of live play are lost. P3: on the walk-off stress the gain is +0.0000 or -0.0003, with at
most 2 of 5 seeds winning, because v7 already decodes the synthetic walk-off as D (H/A share 0.0
on its frames). Either the stress is easier than BAR-ATH's walk-off, or v7 no longer has the
failure that section 20 measured on v5. No setting above is changed against these numbers.
Open: `--bar-ath` (denis) shows, label-free, how many possessions v7 still decodes after the
detected end-of-half whistle. If v7 decodes none, the rule has nothing to fix. If it decodes
some, a new rule and stress need a new registration, built from those aggregates and not from
these proxy numbers.
**BAR-ATH (denis, `--bar-ath`, 2026-11-17): B1 passes, B2 fails.** Half 1: the pick is 78.4 s
before the half's end; collective speed after it 1.06 m/s; players inside the pitch 21 -> 4; v7
starts 2 possessions there (H/A share 0.19). Half 2: the pick is 1.3 s before the end and v7
decodes nothing after it, so the rule fires without removing a possession, which fails B2 as
written. What remains of the walk-off failure on v7 is 2 possessions in half 1 (v5 had about 6),
worth about +0.003 spellF1 at most. Recommendation: close the whistle line. A new attempt
would need a new registration, a BAR-ATH-shaped stress and a guard that survives simulator v1's
tracking dropouts, all to chase about +0.003.
