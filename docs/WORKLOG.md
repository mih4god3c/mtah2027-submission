# Work log

One entry per working session, **newest first**. An agent or teammate starting work reads the
top two or three entries before anything else. Keep each entry short: what changed, what we
learned, what is open, what comes next. Results go in docs/EXPERIMENTS.md and lasting choices in
docs/DECISIONS.md; link to them rather than repeating them. Never paste data rows here.

Template:

```
## YYYY-MM-DD — <who> (<agent, if any>) — <branch>
**Goal.** ...
**Done.** ...
**Learned.** ...
**Open / blocked.** ...
**Next.** ...
```

---

## 2026-11-17 — denis (Claude Code) — `denis/keeper-ball`

**Goal.** Adopt only the end-of-half part of the whistle model, under D14 (D48).

**Done.**
- `v7_end` in [solution/methods.py](../solution/methods.py): v7 plus the end-of-half rule on the
  detected whistles, with a tracking guard (`whistlecue.end_picks`; median collective speed after
  the pick at most 1.7 m/s).
- `simulate.add_walkoff` (walk-off stress); the imitated final whistle follows it
  (`whistlecue.walkoff_frame`, `half_ends`).
- [tools/halfend_whistle_check.py](../tools/halfend_whistle_check.py) (`make halfend-whistle`).
- 11 tests in [tests/test_halfend_whistle.py](../tests/test_halfend_whistle.py).

**Learned.** D48 fails P2 and P3 (EXPERIMENTS section 37):
- 6-10 false fires per 50 plain halves: a missed final whistle leaves a late whistle as the pick,
  and the guard does not stop it.
- No gain on the walk-off stress, because v7 already decodes the synthetic walk-off as D.

Not adopted; `run.py` unchanged.

- BAR-ATH `--bar-ath` (denis): B1 pass, B2 fail. v7 still starts 2 possessions after the half-time
  whistle (players inside 21 -> 4), and none after the full-time one. At most about +0.003.

**Open / blocked.** Nothing. Recommended: close the whistle line (D48). Reopening needs a new
registration.

**Next.** The submission stays `v7_keep2_control`.

---

## 2026-11-16 — denis (Claude Code) — `denis/keeper-ball`

**Goal.** A whistle detector with no setting from this match (D47).

**Done.**
- Self-calibrating detector in [solution/whistle.py](../solution/whistle.py): generic tonal
  peaks in 2-5 kHz; kickoffs from the tracking (`kickoff_anchors`); the signature repeated before
  them against chance windows (`calibrate`); detection in those bands (`whistle_events`,
  `match_whistles`). 9 tests on synthetic audio in
  [tests/test_whistle_selfcal.py](../tests/test_whistle_selfcal.py).
- Removed the v1 two-tone settings, `tools/whistle_two_tone_check.py`, its tests, and
  `tools/whistle_spectrum.py`.
- [solution/whistlecue.py](../solution/whistlecue.py) reads the new detector on BAR-ATH. Its
  proxy decision variant is now "prior", set a priori (it replaces "measured").
- [tools/whistle_selfcal_check.py](../tools/whistle_selfcal_check.py) (`make whistle-selfcal`):
  D47's Step 0. Aggregates only; no frames, times or listings.

**Learned.** The tracking's kickoffs are good anchors: on IDSSE all fall within 10 s of a real
restart kickoff, 91% within 3 s. With the a-priori variant the proxy gain is the same as before
(+0.0075 keep2, +0.0069 DFL; below the bar). EXPERIMENTS section 36.

- Step 0 on BAR-ATH (denis): **passes** all four criteria. A two-tone signature at 3 of 3
  kickoff anchors, 23 x chance; stable when any anchor is left out; 0.72 / 0.96 whistles/min;
  0.40 of onsets near a decoded stoppage against 0.14-0.18 by chance.
- Up to ~0.5 onsets/min match no stoppage: an upper bound on false whistles. The proxy at that
  noise (`v7_whistle_matched`, registered before its run): +0.0017 keep2, +0.0024 DFL, within
  noise. EXPERIMENTS section 36, D47.

**Open / blocked.** How many unmatched whistles are real decides between +0.0075 and ~0: denis can
run `make whistle-selfcal ARGS="--cross"` (context). `run.py` is unchanged.

**Next.** The in-play cue is not worth adopting. The open value is the end-of-half rule
(`v7_whistle_end`), which the proxy cannot measure. Adopting it needs a rule other than D4: denis
and miha.

---

## 2026-11-15 — denis (Claude Code) — `denis/keeper-ball`

**Goal.** Settle D46 before review.

**Done.** D46 update: not adopted, since the proxy gain (+0.008 / +0.007) is under D4's bar and
`run.py` keeps `v7_keep2_control`. D44's Step 0 criteria trimmed to the spot check
and the chance rate. The `--around` examples in the whistle tools now take a placeholder frame.

**Open / blocked.** Superseded by the 2026-11-16 entry (D47).

---

## 2026-11-12 — denis (Claude Code) — `denis/keeper-ball` (uncommitted)

**Goal.** Find the referee's whistle in the audio, the broadcast first (D44).

**Done.**
- `tools/whistle_spectrum.py` (removed since, D47): a wide-band look at a few seconds of audio.
- `whistle.two_tone_events` / `two_tone_video_features` in
  [solution/whistle.py](../solution/whistle.py) (settings `TT_*`, D44 update);
  [tools/whistle_two_tone_check.py](../tools/whistle_two_tone_check.py) (`make
  whistle-two-tone`; `--spot`, `--around`, `--cross`); 5 tests in
  [tests/test_whistle_two_tone.py](../tests/test_whistle_two_tone.py).
- `whistle_check.py --source broadcast`; `games.py` finds `-001` copies of the documented files
  (4 tests in tests/test_games_paths.py).
- EXPERIMENTS section 33; D44 Step 0 result.

**Learned.** The two-tone rule runs at 0.6-0.7 whistles/min on both feeds, and the feeds agree
61 times over chance. Spot check: tactical 7 of 8, broadcast 5 of 8, so Step 0 fails as written; the broadcast mix has
some fan whistling after all.

- Whistle veto (denis's idea: no whistle + mid-pitch = not dead): Step 0 on the proxy,
  [tools/whistle_veto_check.py](../tools/whistle_veto_check.py) (`make whistle-veto`, 3 tests in
  tests/test_whistle_veto.py). v7's false-stoppage oracle is only +0.0127, and even a perfect
  whistle veto loses (-0.0008 to -0.042). Closed, D45, EXPERIMENTS section 34.

- The whistle as a decoder cue (denis reopened D26 and D45): [solution/whistlecue.py](../solution/whistlecue.py),
  decoder term `whistle_w`, methods `v7_whistle`, `v7_whistle_end`, `v7_whistle_pess`,
  `v7_place`; 4 tests in tests/test_whistlecue.py. Proxy +0.008 (keep2) / +0.007 (DFL), below the
  bar; BAR-ATH label-free mostly in line, two narrow misses. D46,
  EXPERIMENTS section 35. `tools/voters.py --blocks-csv` adds whistle columns for such methods.

**Open / blocked.** Superseded by the 2026-11-15 entry: the whistle cue is not adopted.

---

## 2026-11-11 — denis (Claude Code) — `denis/keeper-ball` (uncommitted, from `denis/skillcorner-check` 847fa70)

**Goal.** denis watched the whole BAR-ATH match against v7's blocks: H/A transitions mostly
fine (1-3 s late at times); the keeper's ball decoded D and goal kicks first decoded as a team;
celebrations, the drinks break and walk-offs decoded as possession. Measure, then build a
keeper's-ball model.

**Done.**
- [tools/stoppage_shape_check.py](../tools/stoppage_shape_check.py): keeper holds, restart types
  and the team-shape "belt" on the proxy (3 tests).
- [solution/keeper.py](../solution/keeper.py): keeper geometry, `on_ball`, the stresses
  `hide_keeper_ball` and `lose_ball_near_goal`, the label-free `gaps_by_keeper`.
- [solution/keepermodel.py](../solution/keepermodel.py): the keeper-zone state model; decoder
  term `keeper_w` (default 0, registered methods unchanged); methods `v7_keeper`, `v7_keeper2`.
- [tools/keeper_check.py](../tools/keeper_check.py) (`make keeper-check [ARGS=--bar-ath]`);
  proxy league `HIDE_KEEPER=`, `LOSE_NEAR_GOAL=`. 7 tests in tests/test_keeper.py; `make test`
  327 passed before the last 4 were added.
- EXPERIMENTS section 32.

**Learned.** The belt finds stoppages the decoder already has; as a rule it gains nothing on
the proxy (closed there). BAR-ATH loses the ball in the keeper zone (seen 0.30 against the
proxy's 0.42), and gaps ending at a keeper are five times the proxy's share, which looks like
goal kicks. `v7_keeper2` +0.0117 (5/5) on keep2 + D12, +0.0095 on DFL, +0.0096 / +0.0107 under the
two stresses that bracket BAR-ATH: consistent, below D4's 0.02.

**Open / blocked.** Not adopted; no DECISIONS entry. Not run: TEMPO=0.9, a `keeper_w` check,
and label-free BAR-ATH aggregates (needs data/, denis).

**Next.** TEMPO=0.9 for `v7_keeper2`. If it holds, stack it with the next sub-bar gain (stoppage
edges are the largest ceiling left, +0.055) or take it to miha as a candidate under the stacking
rule (D38, D42).

---

## 2026-11-10 — denis (Claude Code) — `denis/skillcorner-check` (uncommitted, on 847fa70)

**Goal.** denis: the tactical-audio whistle detector was swamped by crowd whistling (D33). Get the
whistles from the broadcast audio.

**Done.**
- [solution/whistle.py](../solution/whistle.py): `annotate`, `map_events`, `feed_lag`,
  `pair_feeds` (a whistle heard in both feeds); one-feed detector unchanged.
- [tools/whistle_broadcast_check.py](../tools/whistle_broadcast_check.py) (`make
  whistle-broadcast`, after `make broadcast-align`): rates, pitch tables, lag, kickoffs, D33's
  end-of-half pick on paired events, `--spot N` for spot checks. Smoke-tested on
  fabricated features only.
- 9 tests in [tests/test_whistle_broadcast.py](../tests/test_whistle_broadcast.py).
  D44 (reopening, criteria fixed before the run); EXPERIMENTS section 31 (synthetic results).

**Learned.** On synthetic audio a loud, bright commentator breaks the broadcast feed alone
(missed blasts, false events), so the broadcast is not used on its own. Pairing keeps every
blast the broadcast hears and drops the crowd; its limit is a voice formant on the crowd's
pitch, which the chance-rate column shows. Value stays small: end of half about +0.005 (section
20); in-play whistles stay closed (D26).

**Open / blocked.** Nothing ran on `data/`. denis: `make broadcast-align` (if the cache is
missing), then `make whistle-broadcast ARGS="--spot 10"`; judge D44's criteria.

**Next.** On a pass, Step 1 of D44 (the end-of-half rule, adopted under D14). On a fail, close.

---

## 2026-11-09 — denis (Claude Code) — `denis/skillcorner-check` (from `miha/dev` 89ba9c0)

**Goal.** denis: would more labelled matches help, and do SkillCorner's open data carry
possession labels? (denis found the per-match CSVs.) First look at the data only.

**Done.**
- [tools/skillcorner_check.py](../tools/skillcorner_check.py): downloads SkillCorner's small
  open-data files at a pinned commit into `external_data/skillcorner/` and describes the
  possession structure of its phases of play beside IDSSE (dfl, keep2) and the public window.
  4 tests in tests/test_skillcorner_check.py, no network needed.
- EXPERIMENTS section 30 (numbered after `denis/spell-verify`'s 29).

**Learned.** 20 matches, not 10. Phases of play read as H/A, with D in between, look like
controlled possession: 13.5% under 2 s, median 7.3 s (window 11.8%, 7.6 s; DFL 29.8%, 3.7 s), and
IDSSE-like stoppages (95 per 90, median 20 s). keep2 changes 0.35% of its frames. Tracking is
broadcast at 10 fps, so it would need its own degradation check.

**Open / blocked.** Nothing ran on `data/`. No loader, no proxy-league entry: a setup change,
miha's call, after the Phase 4 freeze. The learning curve on IDSSE (does a 5th training match
still help?) is not run.

**Next.** Show miha section 30. If wanted after the freeze: a SkillCorner loader (tracking via
kloppy or the LFS files, 10 to 25 fps), a degradation check against BAR-ATH, then SkillCorner as a
second label source in the proxy league.

---

## 2026-11-07 — miha (Claude Code) — `miha/dev` (merge of `denis/ball-impute`, PR #9)

**Goal.** miha: merge Denis's PR #9 into `miha/dev`; GitHub showed it conflicting.

**Done.**
- Merged with both sides kept. The conflicts (DECISIONS, EXPERIMENTS, WORKLOG, methods.py) came
  from both branches taking D38, D39 and EXPERIMENTS section 25 on the same day. Denis's entries
  are renumbered, D38 -> D41, D39 -> D42, sections 25 to 27 -> 26 to 28, with a note under each
  heading; every reference in his three docs follows (his code had none). His commit messages
  keep the old numbers.
- [solution/methods.py](../solution/methods.py): his seven methods (`touch_c3_seen_w1_dm_ball`,
  `dm_bor_*`, `dm_stack3`, `dm_stack3_dead4`) sit before the D40 names block; 76 registered.
  [docs/SUBMISSIONS.md](SUBMISSIONS.md) section 4 regenerated (`make methods`), status list extended.
- `make test`: 272 pass (264 plus his 8). Nothing ran on the match data. Pushed to origin.

**Learned.** Two branches numbering decisions in parallel collide within a day. Claim the next
D and section number in WORKLOG before writing, or number at merge time.

**Open / blocked.** The submission stays `v7_keep2_control` (D39): his stack (D42) failed the
BAR-ATH rules and is closed; run.py was not touched by his branch.

**Next.** Tell Denis the new numbers; his next branch starts from this `miha/dev`.

## 2026-11-07 — miha (Claude Code) — `miha/dev` (committed in 3d8c4c9 and follow-ups, on top of 927fb83)

**Goal.** miha: would a two-stage design be better: one per-frame classifier for dead ball,
then, on the alive frames, another for the team?

**Done.** No code, no runs. Read D8, D22, D23, D26, D28, D35, D37, EXPERIMENTS sections 4, 15,
16 and 18, and `decoder.block_emissions`.

**Learned.** The submission already is that design, in its soft form.
[solution/deadmodel.py](../solution/deadmodel.py) is a per-frame dead-against-alive classifier
and [solution/framemodel.py](../solution/framemodel.py) a per-frame team classifier trained on
alive frames only, P(home | alive). The decoder adds log P(x | dead) / P(x | alive) to D and
log P(team | alive) to H and A, the same factorisation, and the duration priors arbitrate
between them. A hard cascade (dead/alive decided first and for good, the team decided inside
the alive runs) is the special case `deadm_w` -> infinity with the team cues silent on the
dead/alive call: it drops that arbitration, needs its own smoother per stage, and makes every
stage-1 error final. The evidence says the wiring is not the limit: the decoder's errors are
confident ones in the evidence (D28), and the dead term at the model's own strength with the
true sign carries the whole perfect-dead-ball ceiling of +0.147 (section 16, `dmo_sign`). On
the public window the team call is right on 0.92 of the frames where both say alive and D
recall is 0.67 (section 4): the dead-ball classifier's accuracy is the lever, not the
pipeline shape.

**Open / blocked.** Whether team cues overpower a dead model that leans dead at the window's
missed stoppage (12:01) is what `make moment FROM=11:50 TO=12:20` reports; not run here (data).
If they do, the fix inside the current design is one decoder setting, decided on the proxy:
`deadm_w` above 1 (never tried, D26) or the team terms scaled by the model's P(alive). If the
model itself leans alive there, no wiring helps; it needs evidence (ball beyond a line, D15;
cut-aways, D31).

**Next.** As before: `make moment` on 11:50 to 12:20; then the D38 stacking decision and the
Phase 4 freeze.

## 2026-11-07 — miha (Claude Code) — `miha/dev` (committed in 3d8c4c9 and follow-ups, on top of 927fb83)

**Goal.** miha: is there anything usable in Guo et al., "Automatic Estimation of Football
Possession via Improved YOLOv8 Detection and DBSCAN-Based Team Classification", Sensors
26(4):1252, Feb 2026 (open access, PMC12944081)?

**Done.** Read the full text (through Europe PMC; mdpi.com refuses fetches) and the layout of
its code repo (AGPL-3.0; the structure of the public football_analysis tutorial). No code, no
runs.

**Learned.** Not applicable. It is a broadcast-only pipeline (YOLOv8 ball and player detectors,
DBSCAN jersey clustering, Norfair tracking) that outputs one possession percentage per match,
scored as RMSE against official statistics on 20 matches (4.87 points). No per-frame truth, no
dead ball (UNKNOWN frames are dropped from the denominator), no pitch coordinates. Its
possession rule is nearest player with distance hysteresis (enter 0.6, exit 1.0 player box
heights), a 0.30 s persistence before a team switch and a 0.20 s hold over missing detections.
Everything it adds sits on tracks we closed: a broadcast ball source (D27), a gap-filling
detector (D29), proximity ownership and the hysteresis holder (D10), control-model variants
(D35); our hold over ball gaps (`carry_gap_s` 1.0 s, `expire_s` 4.0 s in `solution/touch.py`)
was tuned on the proxy for ATD-length gaps, not detection dropouts. Its data links (SoccerNet,
a GitHub video set) carry no possession labels, so no proxy matches either.

**Open / blocked.** Nothing.

**Next.** As before: miha runs `make moment` on 11:50 to 12:20; then the Phase 4 freeze and
the hold-out run (D3).

## 2026-11-07 — miha (Claude Code) — `miha/dev` (committed in 3d8c4c9 and follow-ups, on top of 927fb83)

**Goal.** miha: a clip of the tactical video covering the five minutes after the public window,
to watch.

**Done.**
- [tools/clip_video.py](../tools/clip_video.py) (`make clip`; default window = the five minutes
  after the public truth, video frames 18685..26184, 12:27.4 to 17:27.4; `FROM=`/`TO=` as frames
  or mm:ss, `SOURCE=broadcast`, `ARGS="--stamp"` burns the source frame number and clock into the
  corner): frame-accurate cut with PyAV (already pinned), re-encoded so clip frame k is exactly
  video frame FROM + k, audio copied over the same range; it stops rather than write a clip whose
  frame numbers are off. Output goes to `data/clips/` (gitignored). No new dependency.
- 11 tests on a synthetic video in [tests/test_clip_video.py](../tests/test_clip_video.py); 264 pass.
- The agent's sandbox denies every command naming `data/`, as CLAUDE.md intends, so miha ran
  it: 7500 frames written, 135 s with libx264 and 76 s with `ARGS="--stamp --codec
  h264_videotoolbox"` (the second run overwrites the first, same path). The seek path held on
  the real file. The videotoolbox file came out over 500 MB, it has no rate target by default,
  so `--max-mb N` (and `--bitrate`) were added: the bit rate follows from the frame count with
  a 0.85 margin, because on synthetic 1080p noise the hardware encoder overshoots the rate it
  is given by up to 12% and libx264 by 8%.

**Learned.** ffmpeg is not installed on miha's machine. PyAV 18 bundles libx264, libx265,
h264_videotoolbox and aac but not the drawtext filter, so the stamp is drawn with Pillow (pinned).

**Open / blocked.** Nothing. The clip is for watching, beside `make moment` and `make frame-report`.

**Next.** miha reruns with `ARGS="--stamp --codec h264_videotoolbox --max-mb 480"`, checks the
first stamped frame reads f 18685 / 12:27.4 and the stamp matches the player's clock through
the clip. The frame numbering was verified on synthetic video only;
`ARGS="--no-seek"` decodes from the start if a real file ever trips the seek path.

## 2026-11-07 — miha (Claude Code) — `miha/dev` (committed in 3d8c4c9 and follow-ups, on top of 927fb83)

**Goal.** miha: how does the submission compare to the public 5 minutes frame by frame, not
on spellF1.

**Done.**
- [tools/frame_report.py](../tools/frame_report.py) (`make frame-report FILES="our_team
  c3_seen_w1"`): confusion matrix and per-class precision/recall on the graded window; every
  disagreement run sorted into late / early / missed segment / spurious segment, with the macroF1
  each kind would return if repaired with the truth; macroF1 with a margin around true boundaries
  forgiven; the truth segment by segment (head, tail, interior wrong frames); latency of every
  true transition; a timeline strip. Reads result files and the public truth only.
- 11 tests in [tests/test_frame_report.py](../tests/test_frame_report.py); 245 pass.
- EXPERIMENTS section 4: the frame-level diagnostic of `our_team.csv` (`touch_c3_seen_w1_dm`,
  sha `a4c78a40…`) beside `c3_seen_w1`. Description only; nothing is chosen on it.

**Learned.** (section 4) 13.7% of the window's frames are wrong, in 19 runs, 91% of them in
runs over 1 s. Four causes carry it in roughly equal parts: boundaries called late (median +18
frames on matched transitions, 9 of 12 late), four short true segments never shown (1.3 to
5.5 s), one invented H segment of 12.4 s that bridges the A>D at 12:01 and so starts that
stoppage 10.5 s late, and that same stoppage plus the missed 5.5 s one account for every D>H
frame. D recall 0.67 is the weakest per-class figure; the team call where both say alive is
0.92. With +-1 s around true boundaries forgiven macroF1 is 0.897, with +-3 s 0.940.

**Added later (same session): the TV viewing starts; the 12:01 stoppage.** miha, watching
the broadcast: at 9:22 it cuts to the coach, which is tactical 11:59, where the ball is out of
play for about 12 s, and we give possession. The alignment cache agrees (broadcast 9:22 is
tactical 12:03.6; the cut-away runs 12:04 to 12:18). From the result files alone: every method
of ours (`_dm`, `touch_c3_seen_w1`, `touch_markov`) hands over A>H at 11:59.2 and stays H until
12:11.6; the truth is A until 12:01.1, then D to 12:15.6. The two TV-cue comparison files
(`our_team_tv`, `our_team_tvcue`) turn D at 12:07, 3 s after the cut-away and 6 s after the
truth, and end it 2.4 s late; they also say D on 11:53 to 11:57 under an earlier cut-away
where the truth is A. Why the submission can miss a 12 s out-of-play: its dead-ball evidence
reads players only (speed rank, the learned model; D15, D22), the touch engine drops frames
with the ball beyond a line (D7) while the nearest-player term and the frame model keep
reading them, and the C2b "player beyond a line" cue is registered in `touch_deadball` only.
Which term holds the moment alive is a question for the data: [tools/moment_check.py](../tools/moment_check.py)
(`make moment FROM=11:50 TO=12:20 METHODS="touch_c3_seen_w1_dms touch_c3_seen_w1_tv_cal"`)
prints, per second, the inputs the decoder saw and each evidence term's part of the H-A and
D-alive margins, plus other methods' labels; 4 tests, 249 pass. For miha to run and read;
nothing ran on `data/` here.

**Added later (same session): the submissions overview.** miha asked for every result file
labelled with what went into it. [docs/SUBMISSIONS.md](SUBMISSIONS.md): each file in
`evaluation/results/` with its method, status, date, sha256, public scores and parts; the
lineage of `run.py`'s default (semi_markov 30 Oct, touch_markov 2 Nov, touch_kickoff and
touch_gapteam 4 Nov, touch_c3_seen_w1 4 Nov, touch_c3_seen_w1_dm 6 Nov, from git); and a
table of every registered method generated from the code by
[tools/methods_overview.py](../tools/methods_overview.py) (`make methods`), which reads the new
`predict.config` on `semi_markov` methods. 3 tests; 252 pass.

**Added later (same session): stacking (D38, EXPERIMENTS section 25).** miha: the closed parts
were never tried together or on top of the submission. Nine `dm_*` stacks registered with
settings and a reading rule fixed first; one plain run by mistake (kept, reported), then the
decision setting on both truths, then the rule-composed `dm_stack`. Result: no part clears D4
alone, two stacks do. `dm_all` +0.0216 keep2 / +0.0106 DFL; `dm_stack` +0.0372 keep2 / -0.0009
DFL (noise) with far fewer possessions (1198 of 1455), the control model's doing. Both read the
broadcast, so D31's BAR-ATH check and a self-built alignment cache in run.py stand between them
and the submission. Not run: a broadcast-free stack (keep2 priors + control model). Four proxy
runs, about 70 minutes of compute; nothing ran on `data/`.

**Added later (same session): the broadcast-free stack, run 3.** miha asked what to do with
the two broadcast stacks; the answer was to try the stack without the broadcast parts first,
since the TV cue inside both had already failed the BAR-ATH in-gap rules twice (D31, D32).
`dm_k2ctl` (keep2 priors + control model) registered and pre-registered as run 3, then run on
both truths: +0.0289 ADOPT 5/5 on keep2 + D12, -0.0076 within noise on DFL + D12, 1185
possessions. miha ran the two D31 commands on BAR-ATH in parallel and shared the aggregates:
rules 1 to 5 pass, warning 6 fires (possessions per 90 x0.822, below the 0.85 line; the proxy
has far fewer short in-gap possessions, D32), and tag agreement rises both in gaps and seen.
Recorded in section 25 and the D38 update. Two more proxy runs, about 12 minutes.

**Added later (same session): miha decided: switch (D39).** `solution/run.py`'s `METHOD` is
`dm_k2ctl`; the docstring, `tests/test_run.py` (the pinned default; `needs_alignment` now reads
the merge flag, so `dm_ball` counts as a broadcast method) and `docs/SUBMISSIONS.md` (sections
1, 2 and 4) follow. The file itself is not regenerated here: that needs `data/`, so miha runs
`make submission` twice, and the sha256, the byte comparison and the public-window sanity
score are filled in when they report. No new dependency; the lock file is unchanged.

**Added later (same session): names (D40).** miha: rename the methods and the result files to
something that reads. The lineage is now `v1_decoder` .. `v7_keep2_control`, the stacking
family `v6_plus_<part>`, and the result files follow (`v6_deadball_model.csv` holds the D37
bytes). Former names stay as aliases in `solution/methods.py`; `tools/methods_overview.py`
skips aliases and prints the former name in brackets; `docs/SUBMISSIONS.md` regenerated
(sections 1, 2 and 4). Code, tests, tool defaults and the Makefile help use the new names;
the history files keep the old ones. `make eval` rescored the renamed files: unchanged.

**Added later (same session): the file (D39 update).** miha ran `make submission` twice; the
two files are byte-identical (`e3ba84bf…`), the second kept as `v7_keep2_control.csv`. Public
window against v6: macroF1 0.8474 -> 0.8531 and 13.7% -> 12.9% wrong frames, share error
0.022 -> 0.005, spellF1 0.6533 -> 0.6286 (one 0.8 s possession at 10:13.8 merged; one spell of
17 is about 0.03). Recorded in `docs/SUBMISSIONS.md` and D39; not a decision input.

**Added later (same session): the broadcast stacks checked on BAR-ATH.** miha asked why not try
the merged ball and TV cue; the answer was to run D31's label-free check on `v6_plus_stack` and
`v6_plus_all` against v7, which had not been done. miha ran the two commands and shared the
aggregates: both fail rule 4 (tag agreement in gaps 0.790 -> 0.692 / 0.667, gap home share
-0.05), `v6_plus_stack` also rule 3. The same in-gap team shift as D31/D32, larger; stacking
did not change it. E13 stays closed (D38 update); nothing switches. Recorded in section 25.

**Added later (same session): the merged ball alone on v7 (run 4).** miha asked how more
visible ball can fail to help; the merge alone on top of an adopted change had never been
measured, so `v7_plus_ball` was registered with D27's ride-along clause as the rule. Proxy
+0.0041 / +0.0053 (5/5, below the bar, as section 17 predicted); BAR-ATH rule 4 fails (tag
agreement in gaps 0.790 -> 0.771, short in-gap possessions 1.7 -> 3.4 per 90). Refused.
The broadcast is now measured four ways on BAR-ATH, all the same direction; section 25 closed.
miha asked for a commit: 3d8c4c9 and 4144acd, on `miha/dev`, not pushed.

**Open / blocked.** The two stoppages (08:34 missed, 12:01 10.5 s late) are the window's
largest frame-level losses and are label-free questions for the dead-ball model; the proxy
decides whether anything changes. Nothing ran on `data/`.

**Next.** The Phase 4 freeze and the hold-out run (D3) on `v7_keep2_control`; freezing the
run-time fits into committed files (run.py's docstring). `make moment` on the 11:50 to 12:20
window stays open as a diagnostic. Nothing is committed from this session yet.

## 2026-11-07 — denis (Claude Code) — `denis/ball-impute` (from `miha/dev` 927fb83)

**Goal.** denis: instead of only hiding the IDSSE ball, learn to put it back in the gaps (E15)
and read possession off the inferred ball.

**Done.**
- [solution/ballmodel.py](../solution/ballmodel.py): gap imputer (HistGBM on gap endpoints and
  player summaries, trained against the true IDSSE ball), a symmetric team reader fitted on
  cross-fitted imputations, and a true-ball oracle with noise and lag. Five methods at the end of
  [solution/methods.py](../solution/methods.py); 6 tests in tests/test_ballmodel.py.
- Proxy league (EXPERIMENTS section 26): imputer WORSE (-0.0050); true-ball oracle within noise
  (-0.0009). [D41](DECISIONS.md): E15 closed.

**Learned.** The frame model reads the team in gaps better than the true ball does (0.853 vs
0.785 on alive gap frames). Moving the in-gap team question further is worth about +0.01 at most:
missing players were already ruled out as the cause of the BAR-ATH in-gap drop (section 13).

**Open / blocked.** `make test`: test_whistle needs `av` (pinned in requirements-lock.txt) in the
shared .venv; unrelated to this work. Nothing run on BAR-ATH.

- REVIEW item 4, the stack (EXPERIMENTS section 27, [D42](DECISIONS.md)): `dm_stack3_dead4` gets
  ADOPT (+0.0206, 5/5), DFL not WORSE (+0.0097); `dm_stack3` +0.0195, below the bar.

- D42's BAR-ATH checks (denis): **fail** rules 1 (tag gap 0.768 vs 0.771) and 5 (set pieces
  0.90 vs 0.93, throw-ins 0.88 vs 0.96). Not submitted; the stack is closed (D42 update).
  `make broadcast-align` now has a cache on denis's machine.
- Step 0 for player tracking (EXPERIMENTS section 28): `CLEAN_PLAYERS=1` (new
  `simulate.degrade(players=False)`, tests/test_clean_players.py) gives +0.0205 for perfect
  players. Not actionable before the freeze.
- Desktop helpers for denis (not in the repo): possession segments and subtitles, and a
  per-second tracked-player count for BAR-ATH, both run by denis.

**Next.** Phase 4 must-dos (REVIEW section 7), safety net Tue 10 Nov: freeze priors and models
into committed files, byte-identity on a second machine, the hold-out run (D3), the technical
report. Tell miha about D41 and D42.

---

## 2026-11-06 (night, 2) — miha (Claude Code) — `miha/dev` (committed in 8e27a9b, on top of 876c856)

**Goal.** REVIEW.md item 1: re-decide the learned dead-ball model (D26) on the balance of
label-free evidence. miha: "Ok do that".

**Done.**
- [D37](DECISIONS.md): the re-decision, with two checks written before anything ran: the L3
  stability check (D36) on the proxy and on BAR-ATH, and a public-window correctness check with
  named criteria (share error, abstentions, no stoppage inside the five long possessions).
- Both pass (EXPERIMENTS sections 4 and 23: plateau 4.5% proxy / 6.3% BAR-ATH; share error
  0.022, 0 abstentions, no false stoppage left in the window). **`solution/run.py` now submits
  `touch_c3_seen_w1_dm`**; `touch_c3_seen_w1` stays registered. Tests updated (a tiny synthetic
  dead-ball model stands in for the IDSSE fit); 234 pass. `make stability` added.
- `tools/stability_check.py` accepts the `touch_c3_seen_w1*` family.

**Learned.** On BAR-ATH the model gives 287 possessions and 114 stoppages of median 18.1 s
(old: 332 and 165 of 9.4 s). On the window it removes both false stoppages and misses one 5.5 s
real one; its window score is recorded, never chosen on. D26's rule 4 stays as a warning.

**Open / blocked.**
- **Byte-identity confirmed** (miha): two `make submission TEAM=our_team` runs give sha256
  `a4c78a40…208a2133`, about 40 s each; the file validates (287 possessions of mean 12.3 s).
- `_dm` is now the proxy-league reference (0.5565 on the decision run). Registered candidates
  measured against `touch_c3_seen_w1` keep their old verdicts; a re-run is needed only for one
  about to be stacked on it.
- The `our_team_*` comparison files in `evaluation/results/` are gitignored; delete the stale
  ones before the final board.

**Next.** The Phase 4 freeze (priors and both models into committed
files, second machine), the TV viewing, and the hold-out run (D3) with both references.

---

## 2026-11-06 (night) — miha (Claude Code) — `miha/dev` (committed in 8e27a9b, on top of 876c856)

**Goal.** Three items from REVIEW.md, in order: the decoder cost re-tune, the L3 stability
check, E8 as set-piece anchors (Step 0 only).

**Done.**
- Decoder setting `dead_min_s` (None = byte-identical) in [solution/decoder.py](../solution/decoder.py);
  9 cost variants `c3s_*` registered in [solution/methods.py](../solution/methods.py); grid and
  rule written in EXPERIMENTS section 22 before the run; keep2 and DFL runs done.
- [tools/stability_check.py](../tools/stability_check.py) (16 fixed perturbations, proxy and
  `--bar-ath`); section 23. miha ran the BAR-ATH side.
- [tools/tag_anchor_check.py](../tools/tag_anchor_check.py) (`--proxy-side`, `--bar-ath`) with
  the reading rule in section 24 written before miha's run.
- 9 tests added; 234 pass. D36 records the outcomes.

**Learned.** (sections 22 to 24, D36)
- Costs: no ADOPT. `dead` 4 + `switch` 2 is +0.0165 on keep2 (5/5) but -0.0014 on DFL: a
  merge effect. `dead` 4 alone is the only same-sign gain (+0.007 / +0.002). Minimum stoppage
  length does nothing. D8's zeros stand.
- Stability: plateau on both sides (worst count change 6.6% proxy, 7.8% BAR-ATH).
- Tags: the SET PIECES window opens about 3 s before the restart kick, inside the stoppage;
  the decoded restart team already agrees with the tag's on 0.91. An anchor for restart edges
  and missed throw-ins, not for stoppage starts. Case 2: Step 1 may be built, +0.01 to +0.02.

**Open / blocked.** E8 Step 1 not started (miha's call). The `dead` 4 stack (REVIEW item 4)
not run. Phase 4 freeze work (priors and models into files, second machine, report) still
untouched. Nothing ran on `data/` from this session; miha's two BAR-ATH runs were label-free.

**Next.** miha decides between E8 Step 1, the stack, and the dead-ball model decision (REVIEW
item 1); then the Phase 4 freeze (safety net Tue 10 Nov).

---

## 2026-11-06 (late) — miha (Claude Code) — `miha/dev` (committed in 8e27a9b, on top of 876c856)

**Goal.** miha wants the TV cut-away cue with the ball merge (`touch_c3_seen_w1_tv_cal_fused`,
D31, D32) as a full file to set against the video by eye, with the submission untouched.

**Done.**
- [solution/run.py](../solution/run.py) `--method <registered name>` writes a comparison file
  with any method in `METHODS`; the default is unchanged (`touch_c3_seen_w1`), so the submission
  path is the same code as before. TV and merge methods stop with a message if the alignment
  cache is missing. `make submission METHOD=... TEAM=...` passes it through.
- [tools/diff_submissions.py](../tools/diff_submissions.py) (`make diff A=... B=...`): every run
  of frames where two result files disagree, as tactical video time and, from the alignment
  cache, broadcast time and what the TV showed; per-file possession and stoppage counts.
- 7 tests added ([tests/test_run.py](../tests/test_run.py),
  [tests/test_diff_submissions.py](../tests/test_diff_submissions.py)); 227 pass. The diff tool
  was tried on two committed baselines without the cache. Nothing ran on `data/`.

**Learned.** Nothing new is measured. No new variant: D32's closure stands (the method run is
the one already registered and checked).

**Open / blocked.**
- Not verified here: that `make submission TEAM=our_team` still gives sha256 `f7459db0…bcf8358d`
  (it should; the default path is unchanged). miha checks it on the first run.
- What the viewing may feed back: a decision between registered methods, recorded as a D entry
  before `run.py`'s default changes; never labels, frame constants or a variant tuned to the
  moments watched (RULES.md, D32).

**Next.** miha runs `make submission TEAM=our_team`, `make submission
METHOD=touch_c3_seen_w1_tv_cal_fused TEAM=our_team_tv` (and `METHOD=touch_c3_seen_w1_tv_cal
TEAM=our_team_tvcue` to see the cue without the merge), then `make diff A=our_team B=our_team_tv`
and watches the longest differing runs. `make eval` on the files is a correctness check only.

**Added later (same session): the Metrica sample-data check** (REVIEW.md item 5). The public
Sample Game 3 EPTS tracking has x and y only, no ball-status or owning-team channel, so the
vendor's corrected-tracking convention is not learnable from public files. Its manual events give
a possession structure close to the public window's (EXPERIMENTS section 12, described only):
Metrica's annotators keep possessions under 2 s (17%), which keep2 removes. Not built: a loader
for the match; adding it to the league would be a setup change (new section) and is miha's call.
Files read were public other-game data in the scratch directory; nothing ran on `data/`.

---

## 2026-11-06 — miha (Claude Code) — `miha/dev` (committed in 8e27a9b, on top of 876c856)

**Goal.** An outside, critical review of the solution with a ranked plan to the deadline. No
solution code changed.

**Done.**
- [REVIEW.md](../REVIEW.md): the metric in plain words, what the pipeline does and where it
  differs from [PLAN.md](../PLAN.md), a segment-level error analysis of the submitted file on
  the public window, an over-fitting assessment, a ranked plan, a day-by-day plan to Mon 16 Nov.
- Reproduced the public-window score: `make test` (220 pass), `validate_submission.py`,
  `run_evaluation.py --bootstrap 1000` (into a scratch dir): 0.5818 / 0.8372, interval
  0.40..0.75; the committed file's sha256 is D23's. Aggregates in EXPERIMENTS section 4.
- A scratch script (not committed) read the public GT, `baseline_raw.csv` as the ball-missing
  mask, and our output on the public-window frames only. Nothing ran on `data/`.

**Learned.** (EXPERIMENTS section 4; REVIEW.md sections 4 to 6)
- On the window: 11 of 17 recovered, 6 missed (five under 4 s, all with the ball seen), 5
  spurious (two fragments cut by 2.5 s false stoppages). Every matched hand-over and restart is
  late (median +20 and +13 frames). Oracle repairs: false stoppages +0.063, perfect dead ball
  +0.076, snapped hand-overs +0.085, missed possessions under 4 s +0.166. Same order as the proxy.
- The review's reading: the decision procedure, not the evidence, is the bottleneck. The
  learned dead-ball model (D26) is the largest built gain and only rule 4 blocks it.

**Open / blocked.** No decision taken; D26 to D35 stand. The hold-out (D3) is unscored.
`run.py` still fits priors and the model at run time. TECHNICAL_REPORT.md is the template.
`evaluation/results/our_team.csv` is a stale file of unrecorded method. PLAN.md unchanged.

**Next.** miha reads REVIEW.md and picks items; each goes on its own branch, one at a time, with
a `make proxy` ablation before and after. Recommended first: re-decide `_dm` / `_dms` (write D36
first), the dead-ball cost grid, the boundary-lag measurement at `BALL_SMOOTH=0.36` (D34).

---

## 2026-11-06 — miha (Claude Code) — `miha/dev` (committed on top of f3a5f3b)

**Goal.** Pick the best unbuilt lever (PLAN backlog review). Run #4's ball lag check, then #3
(control model) from Step 0.

**Done.**
- miha ran `tools/ball_lag_check.py --bar-ath`; the IDSSE reference rows were run here (D34).
- [tools/control_check.py](../tools/control_check.py): Step 0 oracle (default), `--model` (bar 1),
  `--diagnose` (rows A-F). [solution/controlmodel.py](../solution/controlmodel.py): per-touch
  P(contradicted), HistGBM, leave-one-match-out, trained at smoothing 0.26 and 0.40 s.
  `touch.sweep_holder` keep masks; `touch_c3_seen_w1_ctl` / `_ctl_p05` registered.
  [tests/test_controlmodel.py](../tests/test_controlmodel.py) (6) and one mask test: 220 pass.

**Learned.** (EXPERIMENTS section 21, D34, D35)
- BAR-ATH ball: no lag, centred blur about 0.36 s (D12 assumes 0.26 s). The reference loses
  0.027 on the proxy at 0.40 s.
- #3: oracle +0.075, imitation +0.042, but the model's false drops land on real hand-overs.
  `_ctl_p05` +0.0191 (5/5, below D4) and WORSE on DFL. Closed; not submitted.

**Open / blocked.** Whether D12 moves to the measured blur is miha's call (D34), for all methods.
PLAN backlog review (this session): unbuilt are E7, E8, E11, E14, E15, Metrica/SkillCorner, the
explorer, and Phase 4's freeze of fitted priors into committed files (`run.py` still fits them
from the 2.5 GB IDSSE download). PLAN.md not edited. Nothing ran on `data/` from this session.

**Next.** miha decides: D12 at the measured blur (re-run the reference and registered
candidates), or E7 Step 0 (restart-team ceiling). The whistle entry below is a parallel session.

---

## 2026-11-06 — miha (Claude Code) — `miha/dev` (committed on top of 0e01ec0)

**Goal.** miha found possessions predicted after the half-time whistle while players walk off.
Fix the end of each half: players-only rule or whistle detector?

**Done.**
- [tools/halfend_check.py](../tools/halfend_check.py) `--diagnose [--bar-ath]`: the last minutes of
  each half in bins (players tracked and inside the lines, speeds, sprints, ball, reference
  labels). miha ran it on BAR-ATH.
- Players-only cues measured on IDSSE dev (scratch scripts): all rejected (EXPERIMENTS section 20).
- [solution/whistle.py](../solution/whistle.py): whistle detector on the video's audio (PyAV
  18.1.0 pinned by `make lock`); 6 tests in [tests/test_whistle.py](../tests/test_whistle.py);
  213 pass. [tools/whistle_check.py](../tools/whistle_check.py) for the Step 0 run.

**Learned.** (EXPERIMENTS section 20, D33)
- Half 1: after the whistle (about 80 s before the block end), the reference starts 6 possessions
  while players leave. Half 2: about 9 s of H/A in the last 40 s.
- Simulator v1's dropout episodes mimic the walk-off collapse at plain IDSSE half ends, so no
  players-only trigger passes D14's zero-false-fire rule without fitting one BAR-ATH event.

- Step 0 (miha's BAR-ATH run): the detector finds the half-time whistle (two blasts) and both kickoff whistles. But STRICT fires about 10x a referee's rate,
  on crowd whistling, so the end pick misses (22 s before the end in half 1). D33 criterion 1 fails.
- The referee's whistle seems to have its own pitch (3.6-4.0 kHz, against the crowd's 2.3-2.9).
  Not tested: a variant would be designed after seeing the run.

**Open / blocked.** Closed by miha (D33 update): about +0.003 to +0.004, below the noise level.
The code is kept and used by no method. The submission is unchanged.

**Next.** A bigger lever: #7 (Smart Tagging restart team) or #3 (control model). The pitch variant
is late polish at most (D33 update says how to reopen).

---

## 2026-11-06 — miha (Claude Code) — `miha/dev` (committed with the E13 work, on top of 400784e)

**Goal.** Improvement plan #2: adversarial validation. Are BAR-ATH's frames different from the
proxy's, and does that explain why the dead-ball model lowers seen D on BAR-ATH (D26 update 3)?
Then decide what is left of #1 (simulator fixes).

**Done.**
- [tools/adversarial_check.py](../tools/adversarial_check.py): BAR-ATH-or-proxy classifier per
  frame, odd-one-out baseline, group shuffle drops, per-feature shifts, covariate-shift lines.
  `--sanity` (v0 as target), `--proxy-only`, `--subsets`, `--alone`. 5 tests in
  [tests/test_adversarial.py](../tests/test_adversarial.py), all pass; full suite not re-run.
- Reading rules amended after the first sanity run, before any BAR-ATH run (docstring, section
  19). miha ran it on BAR-ATH (about 15 minutes). Two proxy-only follow-ups were scratch scripts.

**Learned.** (EXPERIMENTS section 19, D30)
- Seen frames differ (AUC 0.973 against the odd-one-out 0.930), driven by speed spread and players
  near the goal line. Speed composition, visibility and the ball beyond a line match.
- Position noise lowers speed spread, so it is not the cause. Median shifts of the drivers move
  the model -0.009 and +0.022 against BAR-ATH's -0.050: D26's seen-D sign stays unexplained.
- #1's "more gaps in dead ball" is not supported. A real gap in gaps only: BAR-ATH drops a player
  with the ball and keeps players off the touchlines.

**Open / blocked.** #2 is closed (D30); `_dm` / `_dms` stay unsubmitted. The quantile-mapping
attribution on BAR-ATH was not run (judged not worth it). The parallel E13 session's D28 / D29
were renumbered D31 / D32 at commit. PLAN #1 narrows to a noted player-dropout gap.

**Next.** miha picks a score lever. #6 is already closed (D28), and #4 waits on
`ball_lag_check.py --bar-ath`. Candidates: #7 (Smart Tagging restart team) and #3 (control model).

---

## 2026-11-06 — miha (Claude Code, + 2 forked agents) — `miha/dev` (committed on top of be89cdb)

**Goal.** Step 0 of two items from the improvement plan: #4 (better ball tracking from the
video: what is it worth?) and #6 (decode for spellF1: MBR, or merging unsure possessions).

**Done.**
- #4 W0 (fork): `simulate.reveal_gaps`, `make proxy BALL_VISIBLE=<share>` (default off, 0.4864
  reproduced); 6 tests in [tests/test_ball_visible.py](../tests/test_ball_visible.py).
- #6: posterior sampling in [solution/decoder.py](../solution/decoder.py) (`forward_sums`,
  `backward_sample`, `half_model`, `sample_half`; `viterbi` untouched) and `predict.sample` in
  [solution/methods.py](../solution/methods.py); [tools/mbr_check.py](../tools/mbr_check.py);
  6 tests in [tests/test_mbr.py](../tests/test_mbr.py) (sampler = brute-force posterior, fast
  spellF1 = the official scorer).

**Learned.** (EXPERIMENTS section 18, D28, D29)
- A sharper ball is most of #4's value: no smoothing, same gaps +0.046 (keep2), +0.061 (DFL).
  Coverage is worth little: visible 0.76 +0.011, 0.90 +0.022. PLAN E12 (detector inside gaps,
  "up to +0.15") is effectively closed.
- #6 closes at Step 0: MBR keeps Viterbi; merging low-support possessions loses; even the oracle
  merge loses (-0.022 keep2, -0.047 DFL). The decoder is confidently wrong (own spellF1 estimate
  0.88 against 0.49): the errors are in the evidence, not the decoding.

**Open / blocked.** The +0.046 rests on the 0.26 s smoothing fitted in D12. [tools/ball_lag_check.py](../tools/ball_lag_check.py) (fork: dribble
lag, kick rise time and late/early asymmetry; `add_ball_noise(causal=True)` opt-in; 3 tests,
195 pass) has proxy reference rows in its docstring/report; not reviewed line by line. W0's paired intervals come from an uncommitted scratch driver; the means
reproduce with `make proxy`. Nothing ran on `data/`. The submission is unchanged.

**Next.** miha runs `ball_lag_check.py --bar-ath` once it lands: a real lag is a near-free shift
(proxy decides); no lag closes #4. The main lever is dead ball via #2 (an in-progress
`tools/adversarial_check.py` is in the tree, owner to confirm), then #3.

---

## 2026-11-06 — miha (Claude Code) — `miha/dev` (committed on top of 400784e)

**Goal.** E13 Step 0: can the broadcast feed add ball coverage or a dead-ball cue? First align it
to the tactical timeline by algorithm, then measure, before building any fusion.

**Done.**
- [solution/broadcast.py](../solution/broadcast.py): broadcast -> tactical alignment from the
  two ATDs alone (ball-vote candidates, one-way player-distance refinement, confidence against
  rival offsets, a live chain whose offsets never fall, replays below it, orientation check).
  Parameters set once, before any run on BAR-ATH.
- [tools/broadcast_align.py](../tools/broadcast_align.py) (`make broadcast-align`, `METHOD=none`
  skips the predicted-label split): aggregates only. miha ran it on BAR-ATH. It writes
  `cache/broadcast/BAR-ATH_align.npz` (gitignored, derived from match data).
- 12 tests in [tests/test_broadcast.py](../tests/test_broadcast.py) on a synthetic TV cut; 192
  pass. Proxy what-if `BALL_VISIBLE=0.636` (D12); proxy dead-ball confusion of the reference.

**Learned.** (EXPERIMENTS section 17, D27)
- The alignment is solid: cost p50 0.56 m against rivals at 4.0 m. 108 of 115 stretches are
  confident, with one offset per half.
- Second ball source closed: graded ball visibility goes from 59.3% to 63.6% only, because 53% of
  the tactical gap time is TV cut-away. The optimistic proxy bound is +0.0035.
- The TV is off the pitch on 61% of predicted-D frames and 10% of predicted-live frames. With the
  proxy confusion, that implies P(off | D) about 0.71 and P(off | live) about 0.01.

**Open / blocked.** The independence assumption behind the implied rates is unverified. Where
both feeds see the ball, 12% of frames disagree by more than 5 m (not followed up). The 3 "ahead"
stretches are unexplained.

- Later: miha asked for both tracks. The ball merge is built (`broadcast.fuse_ball`,
  `solution/sources/fused.py`). Its paired proxy test (`tools/broadcast_fill_check.py`) gives
  +0.0038 (+0.0006..+0.0102, 5/5): real, below D4, not submitted (D27 update). The tool gained
  sections G (cut-away timing) and H (merge on BAR-ATH). 15 broadcast tests; 198 pass.

- miha's second run gave sections G and H. Built `solution/cutaway.py` (synthetic cue,
  pessimistic / optimistic, fixed from G before scoring), a decoder term `tv_w` (off by
  default; existing methods byte-identical), methods `touch_c3_seen_w1_tv` / `_tv_opt`, and
  `tools/cutaway_check.py`. 7 tests in `tests/test_cutaway.py`; 205 pass.
- Cut-away cue (EXPERIMENTS section 17): pessimistic (decision) +0.0187 (+0.0125..+0.0373, 4/5),
  0.0013 short of D4. Optimistic +0.0380, ADOPT. DFL not WORSE. The real BAR-ATH cue sits
  between the two variants, and the pessimistic one is noisier in live play than BAR-ATH.

- miha chose the calibrated variant (D31, recorded before calibration). `tools/cutaway_calibrate.py`
  (no score read) fixed spill 0.15 and false 0.02, giving a split of 0.610 / 0.096 against BAR-ATH's
  0.612 / 0.096. `touch_c3_seen_w1_tv_cal`: **+0.0304 (+0.0211..+0.0449, 5/5), ADOPT**,
  reproduced by `make proxy`. DFL +0.0163, not WORSE. `cutaway.cue_for` lets the existing tools
  run TV methods (BAR-ATH: real cue from the alignment cache; proxy: the seed's imitation).
  The BAR-ATH criteria are written into D31 before the check.

- miha ran the D31 BAR-ATH check: **fails** on rules 3, 4 and 5. Tag agreement in gaps fell
  0.734 -> 0.632 and gap home share 0.480 -> 0.411, against the proxy's direction. The dead-ball
  side behaved as on the proxy. Not submitted (D31 update).

**Open / blocked.** Why the TV cue moves the in-gap team towards away on BAR-ATH is
unexplained. The submission is `touch_c3_seen_w1`, unchanged. Nothing is committed.

- miha chose to try the ball merge with the TV cue (D32, hypothesis recorded before the run).
  `fused.merged` lets methods merge in `prepare` (BAR-ATH: real; proxy: the seed's imitation,
  restored frames unstressed). `touch_c3_seen_w1_tv_cal_fused` **+0.0382 (5/5), ADOPT**; DFL
  +0.0256. Merge alone +0.0070, below D4. BAR-ATH criteria are in the D32 update.

- miha ran the D32 BAR-ATH check: `_tv_cal_fused` fails rules 3, 4 and 5, and the hypothesis
  fails (gap home share 0.411 unchanged, tag agreement in gaps 0.632 -> 0.610). **E13 closed**
  (D32 update). `touch_c3_seen_w1` stays submitted; everything built stays registered.

**Learned (overall).** The broadcast mostly fails to see the ball when the tactical ATD does not,
because those moments are TV cut-aways. Its cut-aways are a strong dead-ball cue on the proxy
(+0.030 calibrated). On BAR-ATH, though, the cue moves the in-gap team towards away against the
Smart Tagging phases, and the ball merge does not change that. The proxy has far fewer short
in-gap possessions than BAR-ATH (2.6 against 12.7 per 90), so it could not show this.

**Next.** Back to the PLAN backlog. E13 reopens only on D32's conditions.

---

## 2026-11-05 — miha (Claude Code) — `miha/dev` (personal branch, committed on top of 5338476)

**Goal.** Track 1, dead ball (D15's largest untouched loss): E10 audio check, a tempo stress that
keeps geometry, the learned dead-ball model re-judged under it, decoder work only on a pass.

**Done.**
- Committed last session's D25 work (5338476). Audio: present (FINDINGS point 8); rules allow
  audio and Smart Tagging as inputs (audio by general wording only, PLAN question 9).
- [tools/deadball_check.py](../tools/deadball_check.py): whistle oracle repairs, `--labels`.
- `simulate.retime` + `make proxy TEMPO=0.9`; `deadmodel` v2 feature set, `log_ratio`;
  decoder `deadm_w` / `deadm_where` (off by default, byte-identical); methods
  `touch_c3_seen_w1_dm`, `_dms`; [tools/deadmodel_check.py](../tools/deadmodel_check.py)
  `--tempo`, `--ceiling` (oracles `dmo_*`), `--bar-ath` / `--proxy-side`. 163 tests pass.

**Learned.** (EXPERIMENTS section 16, D26)
- E10 closed: a perfect whistle is worth +0.012 to +0.015. Perfect dead ball is now +0.147.
- D25's SPEED failure was the shrunk pitch: under TEMPO the model clears its bar.
- `_dm` +0.070 and `_dms` +0.036, both ADOPT on keep2 + D12. Both FAIL D26 rule 4 on BAR-ATH:
  seen D falls there (-0.020, -0.021) and rises on the proxy; `_dm` also moved D into gaps.
- `touch_c3_seen_w1` stays submitted; `solution/run.py` is unchanged.

**Open / blocked.** Why the model reads BAR-ATH's seen-ball play as less dead is unexplained
(feature shifts are only candidates). D26 forbids a third read-where variant against its rules.
The rule-4 adaptation was never confirmed by miha (moot: the literal rule failed too). Step 3
(synthetic set-piece tags) not started. PLAN: E10 closed; nothing else in PLAN changed.

**Next.** miha decides: a label-free diagnostic of the seen-frame difference, Step 3's
Step 0, a metric-aware decoder (expected spellF1), or measuring what better tracking is worth
(`--clean`, a 90%-visible simulator variant).

---

## 2026-11-05 — miha (Claude Code) — `miha/dev` (personal branch, committed on top of d867c4b)

**Goal.** Visible-ball invented possessions (section 13's largest loss): Step 0 cause, Step 1
existing knobs on the new reference, Step 3 Step 0 of a learned dead-ball model.

**Done.**
- [tools/gap_check.py](../tools/gap_check.py) `--seen-ceiling` now attributes each invented
  possession: evidence per source, single-source ablation re-decodes, holder start, group oracles.
- `semi_markov` split into `prepare` / `decode` (all registered methods byte-identical); 3
  methods `touch_c3_seen_w1_{hold,k2,hold_k2}` registered.
- [solution/deadmodel.py](../solution/deadmodel.py), [tools/deadmodel_check.py](../tools/deadmodel_check.py).
- 10 tests in [tests/test_seen_attribution.py](../tests/test_seen_attribution.py); 156 pass.

**Learned.** (EXPERIMENTS section 15, D25)
- Wrong-team phantoms (158/seed, 55% of invented error frames) come from the touch holder,
  mostly `touch_takes` hand-overs away from any true change. Ceiling of fixing those: +0.036.
- `touch_takes` False: within noise, WORSE on 4 of 6 runs. keep2 priors: +0.009 to +0.011, 5/5,
  below D4. No ADOPT, so Step 2 did not apply.
- Dead-ball model clears its bar under D12, fails it at edges under SPEED=0.9. Closed.

**Open / blocked.** Nothing ran on `data/`; the submission is unchanged. The byte-identity check
covered two dev matches at seed 0, not BAR-ATH. The new tools' output was not checked by hand.

**Next.** miha picks the lever. PLAN Phase 2 items left: E7 (restart team), E8 (tagging
features). Safety-net freeze is Tue 10 Nov (PLAN Phase 3).

---

## 2026-11-05 — miha (Claude Code) — `miha/dev` (personal branch, committed on top of 2699a7a)

**Goal.** C6 self-training (PLAN C6 / E9): Step 0 ceiling first, Step 1 only if it clears.

**Done.**
- [solution/selftrain.py](../solution/selftrain.py): pick frames (ball seen, holder stable >= 1 s),
  60 s-block cross-fitting, three learners (logistic, small GBM, pooled refit), proxy-only oracle.
- [tools/selftrain_check.py](../tools/selftrain_check.py) `--diag` (frame level) and `--ceiling`
  (registers `c6o_*` oracle methods, then runs the proxy league).
- `semi_markov(frame_fn=...)`, default None, byte-identical. 5 tests in
  [tests/test_selftrain.py](../tests/test_selftrain.py); 146 in total, all pass.

**Learned.** (EXPERIMENTS section 14, D24)
- Even on TRUE labels, adaptation gains under 0.01 over the same reading by the base model
  (gaps +0.002, everywhere +0.008, seen +0.006). The one row at +0.0204 over
  `touch_c3_seen_w1` gets +0.0125 of it from the base model read in gaps (`touch_c3_w1`).
- It does not help in the first 2 s of a gap (D23's question). The learner is not the limit.

**Open / blocked.**
- Step 1 (pseudo-labeller, occlusion test) and Step 2 (BAR-ATH check) were not run, by the
  brief's stop rule. D24 is Claude's reading. +0.0204 is technically over the bar, so miha may
  overrule.
- `LABELS=dfl` and the other stresses were not run for the oracle. The diag table is seed 0 only.
- Nothing ran on `data/`. The submission is unchanged.

**Next.** miha decides the next lever. PLAN C6/E9 is closed at Step 0 (D24). The visible-ball
invented possessions (+0.14 of the +0.19 seen ceiling, section 13) remain the largest loss.

---

## 2026-11-04 — miha (Claude Code) — `miha/dev` (personal branch, committed on top of 237fe79)

**Goal.** C3, the frame model, aimed at who has the ball inside ball gaps; Step 0 ceiling first.
main was not merged in (miha's call).

**Done.**
- [tools/gap_check.py](../tools/gap_check.py): `--labels` (default keep2), `--prior-labels`,
  and `--ceiling` (team and capacity oracles by gap and possession length, plus lag and
  error-run variants). `deadball_check.inputs(labels=)`.
- [solution/framemodel.py](../solution/framemodel.py): P(home | alive) per frame from 35
  players-only, team-swap-symmetric features, sklearn HistGBM (already pinned, so no `make
  lock`), leave-one-match-out on degraded dev matches with keep2 labels.
- Decoder term `frame_w` / `frame_where`: with weight 0 it is identical. 4 methods
  `touch_c3*`. 12 tests added, 137 in total, all pass.

**Learned.** (EXPERIMENTS section 13, D22)
- Step 0 on keep2: capacity +0.053 (DFL +0.051), but nothing if the evidence lags by 3 s.
- The model: 0.873 right on alive gap frames (spread 0.834), median lag 0.3 s.
- **`touch_c3_w1` +0.078 over `touch_gapteam`, ADOPT 5/5.** ADOPT also on DFL and on every
  stress (+0.047 to +0.076). Most of the gain comes from fewer false changes where the ball is
  seen.

**Open / blocked.**
- The BAR-ATH check has not been run (D22 lists its pass criteria). The submission is unchanged.
- Nothing ran on `data/`.

**Next.** miha runs `gap_check.py --decoded --bar-ath --methods touch_gapteam touch_c3
touch_c3_w1`. If it passes D22, switch `solution/run.py` to `touch_c3_w1` and check `make
submission` timing. PLAN C3 v1 is done; C6 (self-training) is the next lever.

**Added later.** miha ran the BAR-ATH check (EXPERIMENTS section 13). Both C3 methods fail D22:
tag agreement in gaps 0.683 / 0.674 against 0.721. With the ball seen it rises (0.813 to 0.868).
Not switched; `touch_gapteam` stays submitted. A proxy-side split also shows the visible-ball
frames hold the larger remaining ceiling (+0.19 against +0.04 in gaps).
Then: `touch_c3_seen_w1` (the model only where the ball is seen) gets ADOPT, +0.0655, and holds
on every stress and on DFL. The visible-ball loss is mostly invented possessions (+0.14 of +0.19).
On the proxy the model is near a coin where a team has under 5 tracked players. Waiting on miha:
`--decoded --bar-ath` for the seen variants, and `--tagdiag --bar-ath`.
Then: miha ran both. The seen variants pass every D22 rule (tag agreement in gaps 0.734). Thin
tracking is ruled out; in the first 15 s of a gap the model agrees with the tags less than spread
does. Weight 1 beats 0.5 directly (+0.0244, ADOPT). **D23: `solution/run.py` now submits
`touch_c3_seen_w1`** (Claude's pick, as miha asked). miha confirmed it on BAR-ATH: two runs are
byte-identical (18 s each). The public window shows no team swap and 16 possessions against 17,
with transitions 11 frames late (EXPERIMENTS section 4).

---

## 2026-11-04 — miha (Claude Code) — `miha/dev` (personal branch, uncommitted on top of 636c4fd)

**Goal.** D20: make the proxy truth follow the task's controlled possession, then re-test what
we adopted on it. main was not merged in (miha's call).

**Done.**
- [solution/relabel.py](../solution/relabel.py) adds `keep2` and `keep2pass`, plus L2 stats.
  [idsse.py](../solution/sources/idsse.py) adds `LABELS`, `load(labels=)` and `completed_passes`.
  `restarts()` now uses a shared `_events()`; its output is identical on all 7 matches.
- `make proxy LABELS=... PRIOR_LABELS=...`. Every cached prior fit takes a label variant, and
  `semi_markov(prior_labels=)` sets it per method.
- [tools/label_check.py](../tools/label_check.py) is Step 0.
- Methods: `touch_markov_hold`, `touch_gapteam_hold`, `touch_gapteam_k2`. 12 tests in
  [tests/test_relabel.py](../tests/test_relabel.py); 125 in total, all pass.

**Learned.** (EXPERIMENTS section 12, D21)
- keep2 removes 21% of DFL possessions (84% of those under 1 s). It gives 275 per 90, 4% under
  2 s, median 6.8 s. A perfect DFL labeller scores 0.757 on it.
- Everything adopted holds. `touch_gapteam` gains +0.032 to +0.039 (ADOPT). `touch_takes` False
  is within noise on controlled truth, against WORSE on DFL.
- `touch_gapteam_k2` gains +0.0189 on the decision run, just under D4. It is better on every
  stress and on DFL. Not adopted.

**Open / blocked.**
- **Added later:** miha agreed to D21. `make proxy` now defaults to `LABELS=keep2
  PRIOR_LABELS=dfl`.
- **Added later:** miha ran the label-free BAR-ATH check (EXPERIMENTS section 12). k2 decodes
  365 per 90 against 393 (the same 0.93 times as on the proxy), with D unchanged. Smart Tagging
  agreement inside gaps falls from 0.721 to 0.676. No exception: `touch_gapteam` stays
  submitted, and k2 stays registered only.
- Nothing ran on `data/`; the submission is unchanged.

**Next.** k2 waits for a method that clears D4 on keep2 without losing
in-gap tag agreement. Then C3 (the frame model) on keep2 labels, aimed at team ownership
inside ball gaps. PLAN unchanged; question 7's "how long counts as control" now has a working answer,
keep2 (D21).

---

## 2026-11-04 — miha (Claude Code) — `miha/dev` (personal branch, uncommitted on top of f81305f)

**Goal.** Step 0 of D18's recommendation: what would a dadvance transition bonus gain, and
what do its false alarms cost?

**Done.**
- [solution/decoder.py](../solution/decoder.py): `switch_bonus` and a `bonus` term in
  `viterbi`. `bonus_w` defaults to 0, which is byte-identical (touch_gapteam's codes hash the
  same). [methods.py](../solution/methods.py): `semi_markov(bonus_fn=...)`, defaulting to
  `dadvance_events`.
- [tools/gap_check.py](../tools/gap_check.py) `--bonus`: all, right-only (oracle-filtered) and
  perfect events, at w 2/4/7 nats. 2 tests added, 113 in total, all pass.

**Learned.** (EXPERIMENTS section 11, D19)
- **The dadvance bonus cannot reach 0.02.** Real events give at most +0.0043 (WORSE at 7 nats).
  Even without false alarms: +0.0096. Its recall (a third of hand-overs) is the limit.
- Perfect events gain +0.024 to +0.040, so the mechanism works; the evidence is what's missing.

**Open / blocked.** No method registered, no proxy-league run (D4). `touch_gapteam` stays
submitted. With D18, short possessions have no route above 0.02 from the evidence so far.

**Next.** miha chooses. Short possessions need an event source that recalls most hand-overs:
score it with `gap_check.py --bonus` first. Otherwise turn to a different loss.

**Added later (label semantics, EXPERIMENTS section 4).** The public window's possessions are
longer than DFL's: 12% under 2 s against 30%, median 7.6 s against 3.7 s, 17 possessions. That is
suggestive, not conclusive. Then miha pointed to CHALLENGE.md ("Possession rules"): possession
changes only on *controlled* possession, and touches, deflections and won headers without control
do not count. DFL flips at the first touch, so the proxy's truth holds possessions the real truth
does not, and it over-rewards splitting (D20). PLAN question 7 is marked answered except for the
dead-ball start. **Next session:** relabel the proxy truth to controlled possession, re-test
`touch_takes` and the adopted methods on it; C3 after that.

---

## 2026-11-04 — miha (Claude Code) — `miha/dev` (personal branch, committed f81305f)

**Goal.** Choose between C2c + C3 for ball gaps (with a transition bonus) and touch/E6 work on
short possessions with the ball seen, by measuring both first. main was not merged in (miha's call).

**Done.**
- [solution/handover.py](../solution/handover.py): 14 players-only moment features (spread,
  advance, reversal, leave, crowd, and past-minus-future differences of spread, advance and line)
  plus their events. It is not read by any method.
- [tools/gap_check.py](../tools/gap_check.py): `--events` (Step A: part (c) for any feature, with
  hits, lag, false alarms, chance, lift and precision; `--noise`, `--speed`, `--rates`, `--bar-ath`)
  and `--seen` (Step B: team, snap and reach oracles against a random subset of the same size).
  11 tests in [tests/test_handover.py](../tests/test_handover.py), 111 in total, all pass.

**Learned.** (EXPERIMENTS section 11, D18)
- **Step A: `dadvance0.5` marks the moment** (all players' run direction changing). It fires
  within +-1 s of 32% of in-gap hand-overs, at lift 4.2 over its false alarms, median lag +0.7 s.
  This holds under NOISE=0.1 and SPEED=0.9. Spread: 5%, lift 2.3. Line, leave and crowd: lift 2 or below.
- **Step B, under 2 s with the ball seen: touch/E6 ceilings are below 0.02.** Snap +0.0185;
  touch reach +0.012, *below* a random subset (+0.054). The engine sees the possessions the decoder
  already gets. The team ceiling is +0.118, and dadvance as evidence reaches +0.042 there.

**Open / blocked.**
- The oracles ignore false alarms: dadvance precision is 0.26, touch claims 0.36. Not measured
  on BAR-ATH beyond rates. miha ran `--events --bar-ath`: dadvance0.5 fires 5.0 per min (proxy
  5.1), 2.0 in gaps (proxy 2.5). No warning sign; expect a somewhat smaller effect.
- No method changed, no proxy-league run; `touch_gapteam` stays submitted. PLAN.md unchanged.

**Next.** Recommendation for miha: a players-only transition bonus from `dadvance0.5`, read on
every frame, not only in gaps (C4's bonus with C2c's feature; C3 not needed for this first step).
Step 0 first: a bonus fed only the right events against all events, to price the 74% false alarms.

---

## 2026-11-04 — miha (Claude Code) — `miha/dev` (personal branch, uncommitted)

**Goal.** E5 part 2: place hand-overs inside ball gaps. Step 0 measures the timing loss under
`touch_gapteam`; Step 1 adds players-only change cues only if timing is worth at least 0.02.
Step 2 (carried over) is a label-free BAR-ATH mode. main was not merged in (miha's call).

**Done.**
- [tools/gap_check.py](../tools/gap_check.py): `--timing` gives (a) the timing oracle against
  the team oracle, split by gap length and by where the loss sits; (b) decoded changes against
  true in-gap hand-overs; (c) the spread cue's flip lead or lag; the duration-prior check.
  `--capacity` decodes with an ideal-sign cue. `--decoded [--bar-ath]` is Step 2: the in-gap team
  share and possessions per 90, plus Smart Tagging team agreement on BAR-ATH.
- [tools/boundaries.py](../tools/boundaries.py): `move_transitions` split out of `snap_to_truth`
  (same behaviour). 7 new tests in [tests/test_gapteam.py](../tests/test_gapteam.py), 100 in
  total, all pass.

**Learned.** (EXPERIMENTS section 10, D17)
- **The timing ceiling is +0.0055** (+0.0025 within +-3 s), well under 0.02, so Step 1 was not
  started. The +0.0865 team ceiling is short possessions inside one gap: +0.074 for those under
  5 s. The decoder makes 10 in-gap hand-overs per 90; truth has 66.
- **The blocker is the evidence.** An ideal-sign cue recovers +0.050 to +0.072. Spread flips
  within +-1 s of 5% of hand-overs and lags by a median 3 s. Minimum durations (1 block) are not
  binding. A short possession costs about 7 nats of prior, so under 2 s even ideal evidence
  finds only 13 to 28%.

**Open / blocked.**
- Step 2 on BAR-ATH run by miha (EXPERIMENTS section 10). The mirror check gives 1.000.
  Smart Tagging agreement inside gaps is 0.596 for `touch_kickoff` and 0.721 for
  `touch_gapteam`, which supports D16. The decoder changes team inside gaps about twice as often
  as on the proxy (19.5 against 10.3 per 90), and it is unknown whether those are real.
- No method changed; `touch_gapteam` stays submitted. PLAN E5: part 2 closed at Step 0;
  PLAN.md unchanged.

**Next.** miha chooses the next session. For C2c + C3, the input is that in-gap gains need
moment cues (about 1 s), not posture. Score any cue with `--timing --capacity` before the
proxy league (D17).

---

## 2026-11-04 — miha (Claude Code) — `miha/dev` (personal branch, uncommitted)

**Goal.** E5 (possession during ball gaps), Steps 0 and 1 only, then report. Step 0 measures what
gaps cost `touch_kickoff`; Step 1 tests player-only team cues in the decoder. Step 2 waits.

**Done.**
- [tools/gap_check.py](../tools/gap_check.py): the oracle team inside gaps (dead ball left
  as predicted), split by gap length and change kind, plus short-possession recall. `--cues` gives
  cue aggregates, and `--cues --bar-ath` gives the label-free version.
- [solution/gapteam.py](../solution/gapteam.py): cues (advance, line, spread, pressure) and the
  per-half attack direction. In the decoder they are the `team_w`, `team_where` and `team_dead`
  terms. [run.py](../solution/run.py) now submits `touch_gapteam` (D16, miha's call after the
  BAR-ATH check). It is the reference from now on.
- [tests/test_gapteam.py](../tests/test_gapteam.py): 16 tests (93 in total, all pass).

**Learned.** (EXPERIMENTS section 9, D16; 17 settings)
- **Ceiling +0.095** under the stress (plain +0.088). Nearly all of it is in gaps holding an
  open-play hand-over, where the team is right on 57% of frames. Restart changes are worth +0.001.
  Possessions under 2 s mostly inside a gap: recall 0.002 (0.59 repaired).
- **Spread (team-width difference) is the cue:** right on 83% of gap frames where it fires.
  Advance, line and pressure are noise. `touch_gapteam` (w 0.5, all frames, "max") gives
  **+0.0225, ADOPT, wins 5/5**. Plain +0.019, SPEED=0.9 +0.021, NOISE=0.1 +0.018, busy
  pre-kickoff +0.022: none WORSE. Weight 1.5 is noise, and 2 is WORSE.
- The gain is restart team and fewer false flips. Hand-overs inside gaps are still not placed,
  and the in-gap ceiling is still +0.089.

**Open / blocked.**
- BAR-ATH cue check run (miha): spread known 0.89 (proxy 0.98), size 0.8 times the proxy's,
  mean favours Barcelona. No warning sign; expect a smaller gain.
- **sha256 check passed** (miha): two `make submission` runs with `touch_gapteam` give
  identical, valid files. D share of graded frames 0.428, inside the IDSSE range (D9). The
  validator counts 462 possessions, against 496 for `touch_kickoff`: fewer, as on the proxy.
- Not done: Step 2 (BAR-ATH decoded team share and possessions inside gaps, Smart Tagging
  agreement). PLAN E5 is in progress; PLAN.md is unchanged.

**Next.** Report to miha. Then Step 2, then hand-over timing inside gaps (what remains of the ceiling).

---

## 2026-11-04 — miha (Claude Code) — `miha/dev` (personal branch, not merged to main)

**Goal.** C2b dead-ball rules (PLAN C2b, E2): measure the dead-ball ceiling first (stop for E5
if under 0.02), then cues the proxy represents fairly, then a label-free BAR-ATH tool.

**Done.**
- [tools/deadball_check.py](../tools/deadball_check.py): Step 0. It matches true D stretches to DFL restarts
  ([idsse.restarts](../solution/sources/idsse.py)), reports errors by type, and scores oracle repairs.
  `--bar-ath` and `--proxy-side` give Step 2's label-free aggregates.
- [solution/deadball.py](../solution/deadball.py) cues (`player`, `ball`, `still`), the decoder's `deadball_w` term
  and `speed_lead_s`, and `simulate.thin_beyond` (`make proxy BEYOND_KEEP= BEYOND_MARGIN=`).
  Registered `touch_deadball`; [run.py](../solution/run.py) still submits `touch_kickoff`.
- [tests/test_deadball.py](../tests/test_deadball.py): 17 tests (77 in total, all pass).

**Learned.** (EXPERIMENTS section 8, D15; 12 settings)
- **Ceiling +0.089** under the stress (plain +0.095): start edges +0.037, false stoppages
  +0.020, end edges +0.018, missed +0.013. Throw-ins are the largest type (+0.028).
- On the missed dead-ball frames players keep moving. What shows the stoppage there is a still
  ball (61%) or one beyond a line, both overstated by DFL's frozen dead ball (13.1% against 2.0%).
- Speed lead: noise. Player beyond a line: +0.005, which survives thinning. With the ball beyond
  1 m, `touch_deadball` gains +0.0084, wins 5/5, below the bar. It is not WORSE on the plain
  proxy (+0.0092) or on the busy pre-kickoff (+0.0073). Even with the unfair still ball, +0.018.

**Open / blocked.**
- BAR-ATH check run (miha): D share 0.418 (in range), player cue 0.7 times the proxy's rate, one
  more throw-in restart in 26 tagged. No warning sign, small effect. D15 stands: not submitted.
- Submitting `touch_deadball` is a team call (D15); D4 says no.

**Next.** E5 (player-only possession model for ball gaps). C2b's fair cues are spent: the rest of
the ceiling needs a ball cue that BAR-ATH really has. PLAN: C2b's rule-based pass is done, below
the bar; PLAN.md itself is unchanged.

---

## 2026-11-04 — miha (Claude Code) — `miha/kickoff-grammar`, merged to main, PR #7

**Goal.** PLAN finding 9 and the C4 grammar: each half is D until a kickoff that the tracking
finds. C2b only if time allowed; it did not.

**Done.**
- [solution/kickoff.py](../solution/kickoff.py): the kickoff detector, registered as
  `touch_kickoff` in [methods.py](../solution/methods.py). It reads players only (D13).
- `simulate.add_prekickoff` (`calm` and `busy` profiles), `make proxy PREKICKOFF=117
  PREKICKOFF_PROFILE=busy`, and [tools/kickoff_check.py](../tools/kickoff_check.py), which has
  `--bar-ath` and `--diagnose` modes.
- [tests/test_kickoff.py](../tests/test_kickoff.py): 14 tests; 60 in total, all pass.

**Learned.** (EXPERIMENTS section 7, D13, D14; 12 settings for the ball rule, 19 for the players rule)
- **sha256 check passed** (miha): two `make submission` runs give identical files.
- **The BAR-ATH ball is never seen resting on the centre mark** (`--diagnose --bar-ath`), so the
  first, ball-based rule found nothing there. The formation does hold: in half 2 its longest hold
  ends at 119.1 s with a speed rise. The rule now reads only that.
- On the proxy it is byte-identical, with 0 false kickoffs in 90 halves. On the synthetic
  pre-kickoff it finds 14 of 15 kicks, a median 1.1 to 1.5 s early. Busy: +0.0015, wins 5/5,
  below the bar. D4's 0.02 is out of reach for a rule that fixes one possession per half.
- `touch_markov` falls into the trap only if players move (busy: D share 0.39); if they stand,
  it already says D (calm: 1.00).

**Open / blocked.**
- **On BAR-ATH the rule finds half 2's kickoff 115.4 s in** (published about 117.4 s, checked
  only, never in code: 2 s early by design), and none in half 1. Before it, `touch_markov`
  labels 46% of frames as a team in 3 false possessions. The trap is real, and the rule removes it.
- **Submitted from now on (D14, miha's call):** a narrow exception to D4 for grammar rules.
  `run.py` uses `touch_kickoff`. Two runs are identical (miha). Against `touch_markov`, D gains
  1,315 frames (H -1,255, A -60) and the validator counts 496 possessions against 501.
- The profiles are invented, but they bracket BAR-ATH on speed and on touch_markov's D share.

**Next.** C2b dead-ball rules, Step 0 first (measure the dead-ball ceiling under the stress); E5 if it is small.

---

## 2026-10-30 to 11-02 — miha (Claude Code) — `miha/e3-touch-engine`, merged into `main` (PR #6)

**Goal.** E3, the touch/control engine (PLAN C2a), as evidence for the decoder; E6 snapping
(C5) only if a boundary report shows it is worth it.

**Done.**
- Step 0: [tools/boundaries.py](../tools/boundaries.py) (`make proxy BOUNDARIES=1`) reports
  transition recall and latency through the official matching, ball seen or not, recall by
  possession length, and an oracle-snap ceiling.
- [solution/touch.py](../solution/touch.py): ball cleaning (jumps > 40 m/s, beyond-line
  frames dropped), touches, control episodes, holder. The decoder reads the holder (`touch_w`),
  registered as `touch_markov`; [run.py](../solution/run.py) now submits it.
- Ball-noise stress `simulate.add_ball_noise` (`BALL_NOISE`, `BALL_JUMPS`), v1 unchanged.
- [tools/touch_check.py](../tools/touch_check.py) for label-free engine aggregates.
- [tests/test_touch.py](../tests/test_touch.py): 15 tests (45 in total, all pass).
- Fixed the previous entry's status line (PR #5).

**Learned.** (EXPERIMENTS sections 5 and 6, D10 to D12; 30 settings tried)
- The loss was missed short possessions and hand-overs, not lag.
- On BAR-ATH the first detector found 2.3 touches per minute against 28.8: **the BAR-ATH
  ball is smoothed by its tracker** (`touch_check --diagnose --bar-ath`, miha). A 0.26 s
  Gaussian on the IDSSE ball reproduces it; that is now the ball stress (D12).
- Under it the clean-ball +0.17 shrinks to +0.024. A detector with a 0.4 s lag and a 3 m
  radius gets **0.3566, +0.070 over `semi_markov`, ADOPT**, and holds at 0.20 to 0.32 s
  smoothing (+0.10 to +0.04).
- A touch still hands over the ball; waiting for control is 0.036 worse (D10's bet).
- E6 gave +0.003 to +0.005 on the clean ball, not adopted; ceiling +0.031 on the smoothed one.

**Open / blocked.**
- **D12 confirmed on BAR-ATH** (miha, 2026-11-02): 22.0 touches and 9.1 controls per
  clean-ball minute, against 21.6 and 9.1 on the stressed proxy (EXPERIMENTS section 6).
- check-loader 100%; submission valid; public window 0.433 / macroF1 0.808, share error 0.006,
  latency +7.8 (section 4, correctness only). 31 possessions against 17 there, while the
  proxy under-segments: the DFL-semantics gap again, not something to tune. The two-run sha256
  check has still not run (shell comments in my commands broke `make`).
- It under-segments on the proxy (360 against 414 per 90). Possessions under 2 s have recall
  0.075; hand-overs with no ball seen 0.05.

**Next.**
- miha: the two-run sha256 check of `make submission`.
- Ask the organisers question 7. Then the second-half pre-kickoff and C2b dead-ball rules.

---

## 2026-10-30 — miha (Claude Code) — `miha/first-submission`, merged into `main` (PR #5, bc7c64c)

**Goal.** The first real submission through the E1 decoder (PLAN.md Phase 1, "first real
submission through solution/run.py").

**Done.**
- [solution/run.py](../solution/run.py): `predict()` runs `semi_markov` on
  `atd.load()` and writes H/A/D, with X exactly outside the graded halves.
  - Priors are fitted at run time on the five IDSSE dev matches (D3).
  - Stops with a message naming `make external` if any dev match is missing
    (`idsse.available`).
- [tests/test_run.py](../tests/test_run.py): one letter per frame, X exactly outside the
  halves, the official format check, and the missing-data message. 2 tests.
- `make proxy` unchanged: 0.2993 (+0.0386, ADOPT).
- miha ran check-loader, submission x2 and eval. Logged in [EXPERIMENTS](EXPERIMENTS.md)
  section 4, as correctness only.

**Learned.**
- **Phase 1 "done when" pipeline items pass:**
  - the loader agrees 100% with the baselines;
  - two runs give the same sha256, in 1.7 s with caches;
  - public window shows no team swap (share error 0.013), no alignment slip (latency +1.3
    frames), macroF1 0.79.
- **24 possessions against 17 on the public window,** where the proxy under-segments. That
  fits DFL labels being choppier, but 17 possessions prove nothing. Do not tune on it.

**Open / blocked.**
- Not verified: the e1 per-class detail (the pager cut it), and run time from a cold cache.
- Priors are not frozen; that is deferred to Phase 4. Until then `run.py` needs `external_data/`.
- Phase 1 still lacks the C2a touch engine and C2b dead-ball rules.

**Next.**
- Commit, open a PR and merge, so `main` submits `semi_markov`.
- Pre-kickoff grammar for the second half (about 2,934 graded frames the proxy cannot test).
- Then C2a touch evidence and C5 boundary snapping, scored against `semi_markov`.

---

## 2026-10-29 — miha (Claude Code) — `miha/e1-semi-markov`, merged into `main` (PR #4, 57e4783)

**Goal.** E1 (PLAN.md C4): replace baseline_smooth50's majority vote with a semi-Markov
decoder. It uses the baseline's evidence plus player speed, scored under simulator v1.

**Done.**
- [solution/decoder.py](../solution/decoder.py): an exact semi-Markov Viterbi over {H, A, D}.
  - Runs at 5 Hz on blocks of 5 frames; numpy only, about 0.7 s per match.
  - Separate costs for switch, dead and restart; a log-normal duration prior and a minimum
    duration (1st percentile) per state class.
  - Priors fitted leave-one-match-out on the clean IDSSE dev labels. Held-out matches never
    enter the fit (asserted).
- Registered as `semi_markov` in [solution/methods.py](../solution/methods.py).
- [tests/test_decoder.py](../tests/test_decoder.py): 10 synthetic tests (26 in total, all
  pass). They check segment recovery, minimum durations, a brute-force exactness check,
  determinism, and NaN players and ghost slots.
- Tuning: 11 settings. Rows are in [EXPERIMENTS](EXPERIMENTS.md) section 2, and the new decision
  is [D8](DECISIONS.md). Also fixed the previous entry's status line (PR #3).
- Speed stress test (EXPERIMENTS section 3):
  - `make proxy NOISE=sd` adds white position noise (`simulate.add_position_noise`; default
    off, v1 unchanged).
  - Two optional speed fixes in the decoder: `speed_lag=12` and `speed_scale="rank"`.
  - [tools/speed_check.py](../tools/speed_check.py) prints speed aggregates for all three
    variants; with `--bar-ath` it is for a human to run. 2 more tests, 28 in total.

**Learned.**
- **ADOPT** at w = 1 with all costs 0: +0.0363 (+0.0257..+0.0440), wins 5/5, macroF1 0.619 to
  0.732. Nonzero costs added at most +0.006, within noise, so none are used. `dead` and
  `restart` act as one knob.
- **The whole gain depends on the speed cue.** Without it the decoder scores 0.2470, WORSE than
  the baseline, because it almost never predicts D.
- It predicts 1,966 possessions against 2,191 true ones, so it slightly under-segments.
- **The registered setting is fragile to position noise.** At 0.1 m of white noise it gains only
  +0.020; at 0.2 m it is WORSE than the baseline. A ±0.5 s speed window or a within-match rank
  holds about +0.036 at every noise level. Neither beats it without noise, so D4 alone does
  not switch.

**Open / blocked.**
- Not run on BAR-ATH. [solution/run.py](../solution/run.py) still submits smooth50. No
  public-window correctness check (L4) yet.
- **Settled by miha's BAR-ATH speed check:** no sign of position noise, but play about 10%
  slower than IDSSE. The decoder now reads speed as a within-match rank ([D9](DECISIONS.md)):
  0.2993 (+0.0386, ADOPT) on the league, unchanged at `SPEED=0.9`, where m/s drops to 0.2841.
  The rank assumes a usual dead-ball share; that trade-off is described in D9.
- Grammar is not implemented yet: the restart rule and second-half pre-kickoff D (PLAN C4).
- Priors are fitted at run time from `external_data/`. Freeze them into a committed file
  before submission, for byte-for-byte reproducibility.

**Next.**
- Review, commit and open a PR (miha to approve).
- Wire `semi_markov` into run.py through the ATD loader, then run `make eval` as a correctness
  check only.
- Then C2a touch evidence and C5 boundary snapping, scored against `semi_markov`.
- PLAN.md: backlog E1 is done in its first form; C4's grammar is still open.

---

## 2026-10-28 (late) — miha (Claude Code) — `miha/simulator-v1`, merged into `main` (PR #3, d74045e)

**Goal.** Simulator v1, calibrated to FINDINGS.md "What it means", points 1 and 4. Re-run the
baselines.

**Done.**
- [solution/simulate.py](../solution/simulate.py): v1 is now the default, and v0 is kept
  unchanged (a test pins it).
  - Ball: v0's stretches, reordered by Metropolis swaps so gaps fall in slow play. The lengths
    are exactly v0's.
  - Players: flicker, multi-player dropout episodes, and stale ghost tracks in spare same-team
    slots.
- [tools/simulator_check.py](../tools/simulator_check.py) (`make sim-check`): prints each
  BAR-ATH target next to the v0 and v1 values.
- `make proxy SIM=v0|v1`, defaulting to v1. Six new tests (16 in total, all pass). TEAM.md and
  CLAUDE.md updated.
- **Bug fix:** the IDSSE loader counted the 3 referees as away players in the DFL-COM-000001
  matches. They are now dropped (cache v2, with a test). The effect on scores is negligible
  (+0.0001).
- Baselines re-run under v1 and v0: [EXPERIMENTS](EXPERIMENTS.md) section 2. New decision:
  [D7](DECISIONS.md).

**Learned.**
- **Targets met on 5 dev matches x 3 seeds:**
  - speed in gaps / ball seen: 1.23 / 2.27 m/s, ratio 0.545 (BAR-ATH 1.17 / 2.16, 0.542);
  - objects per frame: mean 20.7, p5 14;
  - tracklet median 38 s;
  - duplicates 7.9%.
- **Clean IDSSE already has same-team pairs under 0.7 m on about 6.6% of frames**, mostly at set
  pieces. So v1 adds no duplicates on purpose.
- **Best-of-k gap placement distorted the visible-stretch lengths.** Reordering doesn't. A
  per-gap speed score also stalled near 0.70; scoring by frames fixed it.
- **Under v1, the baselines' macroF1 (0.57–0.62) sits near the official full match (0.64–0.68)
  and in the same order.** smooth50 is still the best spellF1 window.
- **Known gap:** the IDSSE ball sits beyond a line during dead ball, so ball-position
  out-of-play cues are easier on the proxy (16% against 4.2%). See D7.

**Open / blocked.**
- Not verified: the player-model choices have no BAR-ATH target. These are where losses
  happen, ID switches, cross-team ghosts and position noise.
- `team frames < 9` is 10.3%, against BAR-ATH's 8.1–9.6%.
- Held-out matches not run (D3). Their caches rebuild on first load.

**Next.**
- Review and commit; PR into `main`.
- E1 (the semi-Markov decoder), scored under v1. PLAN.md Phase 2 item "simulator v1" is done,
  apart from position noise.

---

## 2026-10-28 (evening) — miha (Claude Code) — `miha/eda-findings`, merged into `main` (PR #2)

**Goal.** Turn the exploration results into conclusions, and tidy up before the next task.

**Done.**
- miha ran `notebooks/eda/eda_aggregates.py` on the real data and pasted the output. Claude
  added a "What it means" section on top:
  [notebooks/eda/FINDINGS.md](../notebooks/eda/FINDINGS.md).
- PLAN.md: open question 8 is answered (no hidden channels in the ATD). Question 9 now only asks
  whether audio is permitted. The direction-of-play note is corrected.

**Learned.**
- Ball gaps concentrate in slow play: players average 1.17 m/s in gaps against 2.16 m/s
  otherwise. Most gaps start near a touchline or beyond a line. Simulator v0's random gaps are
  therefore too hard on ball-based methods.
- The ATD ball speed value is unusable, and there are 216 ball teleports.
- Tagging's defending codes are mirror copies of the attacking codes.
- The broadcast sees the ball on 76% of its valid frames.

**Open / blocked.**
- The audio check is deferred. Homebrew cannot install ffmpeg on miha's macOS, and it only
  matters for E10. Use `afinfo` or `mdls` when needed.
- Teammates are not invited yet. Organiser questions are being sent by miha.

**Next.**
- **Simulator v1** first: gaps tied to slow play and near-line positions, messier player
  tracking. Targets are in FINDINGS.md "What it means", points 1 and 4. Re-run the baselines and
  start a new EXPERIMENTS section, because numbers from v0 and v1 are not comparable.
- Then E1, the semi-Markov decoder, tuned on the v1 exam.

---

## 2026-10-28 — miha (Claude Code) — `miha/proxy-league`, merged into `main` (PR #1)

**Addendum, same day.** We moved development to the private repo `mih4god3c/mtah2027` (D6),
because the old fork was public. Everything is committed in four commits on `miha/proxy-league`,
merged by miha as PR #1 without a second reviewer, to unblock work. Teammates, please review it
after the fact and raise anything as a new PR. Team size is three, so use the three-person role split in
PLAN.md section 4. Teammates must be invited and must repoint their remotes (see TEAM.md).

**Goal.** Research and plan the challenge, then build the validation harness the plan starts
with.

**Done.**
- Read the whole repo. Wrote [PLAN.md](../PLAN.md): findings, architecture, a 20-day roadmap, the
  work split, backlog, evaluation strategy, risks, and organiser questions.
- Measured what the public files allow: ball dropouts come in long blocks (median 10.7 s); the
  public window is the best-tracked stretch; 2/3 of dead-ball frames still have a ball; metric
  sensitivities. All in PLAN.md "Findings".
- `notebooks/eda/eda_aggregates.py`: aggregates-only exploration for a human to run on `data/`.
  Not run yet.
- Proxy league v0. These are the pieces:
  - `solution/match.py`: one `Match` format for all matches.
  - `solution/sources/idsse.py` and `solution/sources/atd.py`: the two loaders.
  - `solution/simulate.py`: hides the ball the way our ATD does.
  - `solution/methods.py`: baselines as `Match -> codes`.
  - `tools/proxy_league.py`: the runner.
  - `tools/download_external.py`: pinned, checksummed download.
  - `tools/check_bar_ath_loader.py`: the loader check for a human to run.
  - `tests/`: ten unit tests.
  - Makefile targets `external`, `proxy`, `test` and `check-loader`.
- Two IDSSE matches held out ([D3](DECISIONS.md)).

**Learned.**
- The proxy league picks the official best baseline window (±50); the public window does not
  ([EXPERIMENTS](EXPERIMENTS.md) section 1).
- The simulated dropouts match BAR-ATH closely: 60.0% versus 59.3% visible, and a median gap of
  10.6 s versus 10.7 s.
- DFL possessions are choppier than ours appear (mean 7.7 s, 29% under 2 s). This is a
  label-semantics risk.
- `make check-loader` on the real data: **OK** (run by miha). Our loader reads the ATD exactly
  as `starter/baseline.py` does.

**Open / blocked.**
- Organiser questions (PLAN.md section 9) are not sent.
- Phase 0 human tasks are still open: run the EDA script, watch sampled ball gaps for the video
  gate, and check for an audio track (`ffprobe` is not installed on miha's machine).
- Team roles are not assigned.

**Next.**
- Teammates review PR #1 after the fact and accept the invite to the private repo.
- Add Metrica sample matches to the league, to check label semantics against the same vendor.
- Build E1, the semi-Markov decoder, and score it against `baseline_smooth50`.
