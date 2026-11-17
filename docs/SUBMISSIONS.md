# Submissions: which file is which, and what is in it

Every `*.csv` in `evaluation/results/` is scored by `make eval`, and all but the four organiser
baselines are gitignored, so a file's name says nothing about how it was made. This page maps
each file to its registered method in [solution/methods.py](../solution/methods.py), lists the
parts each method is built from, and gives the lineage of the submission. The method table in
section 4 is generated from the code by `make methods`; re-run it after registering a method and
paste the output here. Public-window scores are correctness checks only (EXPERIMENTS section 4,
RULES.md): 17 possessions choose nothing.

Last updated 2026-11-07.

## 1. The result files on this machine

| file | method | status | written | sha256 | public 5 min macroF1 / spellF1 | proxy spellF1 keep2 + D12 | proxy spellF1 DFL + D12 | proxy spellF1 keep2 plain | parts |
|---|---|---|---|---|---|---|---|---|---|
| `our_team.csv` | `v7_keep2_control` (was `dm_k2ctl`) | **the submission** (`run.py` default since D39) | 2026-11-07 | `e3ba84bf…` | 0.8531 / 0.6286 | 0.5854 | 0.4525 | -- | decoder, touch engine (w1), kickoff grammar, spread cue (w0.5 all max), frame model (w1 seen), dead-ball model (w1 all), keep2 priors, control model (false_drop 0.05) |
| `v7_keep2_control.csv` | `v7_keep2_control` (was `dm_k2ctl`) | D39 check-2 file; identical bytes to `our_team` | 2026-11-07 | `e3ba84bf…` | 0.8531 / 0.6286 | 0.5854 | 0.4525 | -- | decoder, touch engine (w1), kickoff grammar, spread cue (w0.5 all max), frame model (w1 seen), dead-ball model (w1 all), keep2 priors, control model (false_drop 0.05) |
| `v6_deadball_model.csv` | `v6_deadball_model` (was `touch_c3_seen_w1_dm`) | previous submission (D37, 6 to 7 Nov); the fallback | 2026-11-06 | `a4c78a40…` | 0.8474 / 0.6533 | 0.5565 | 0.4601 | 0.6046 | decoder, touch engine (w1), kickoff grammar, spread cue (w0.5 all max), frame model (w1 seen), dead-ball model (w1 all) |
| `v5_frame_model.csv` | `v5_frame_model` (was `touch_c3_seen_w1`) | previous submission (D23, 4 to 6 Nov) | 2026-11-04 | `f7459db0…` | 0.8372 / 0.5818 | 0.4864 | 0.4120 | 0.5346 | decoder, touch engine (w1), kickoff grammar, spread cue (w0.5 all max), frame model (w1 seen) |
| `v5_plus_tv_cue.csv` | `touch_c3_seen_w1_tv_cal` | comparison file, not submitted (D31) | 2026-11-06 | `b2d300a6…` | 0.8266 / 0.5500 | 0.5167 | 0.4283 | -- | decoder, touch engine (w1), kickoff grammar, spread cue (w0.5 all max), frame model (w1 seen), TV cut-away cue (w1 calibrated) |
| `v5_plus_tv_cue_merged_ball.csv` | `touch_c3_seen_w1_tv_cal_fused` | comparison file, not submitted (D32) | 2026-11-06 | `6934a393…` | 0.8266 / 0.5500 | 0.5246 | 0.4376 | -- | decoder, touch engine (w1), kickoff grammar, spread cue (w0.5 all max), frame model (w1 seen), TV cut-away cue (w1 calibrated), merged ball |
| `v2_touch.csv` | `v2_touch` (was `touch_markov`) | submission from 2 to 4 Nov (D10; D12 engine) | 2026-11-02 | `a0eb224c…` | 0.8082 / 0.4333 | 0.3890 | 0.3566 | -- | decoder, touch engine (w1) |
| `v1_decoder.csv` | `semi_markov` | first submission, 30 Oct to 2 Nov (D8) | 2026-10-30 | `95f6366e…` | 0.7893 / 0.3512 | not run | 0.2866 | -- | decoder |
| `baseline_smooth125.csv` | `baseline_smooth125` | organiser baseline, committed | 2026-10-28 | `41704ddc…` | 0.5812 / 0.4242 | -- | -- | -- | nearest player to the ball, majority vote |
| `baseline_smooth50.csv` | `baseline_smooth50` | organiser baseline, the published floor | 2026-10-28 | `6d4e04b8…` | 0.6207 / 0.3256 | 0.2840 | 0.2546 | -- | nearest player to the ball, majority vote |
| `baseline_smooth25.csv` | `baseline_smooth25` | organiser baseline, committed | 2026-10-28 | `c1fd857c…` | 0.6200 / 0.2868 | -- | -- | -- | nearest player to the ball, majority vote |
| `baseline_raw.csv` | `baseline_raw` | organiser baseline, committed | 2026-10-28 | `9471afeb…` | 0.5972 / 0.0378 | -- | -- | -- | nearest player to the ball, majority vote |

Notes.
- `our_team.csv` and `v7_keep2_control.csv` are the same bytes (`e3ba84bf…`, D39), two `make submission`
  runs on 2026-11-07. `v6_deadball_model.csv` keeps the D37 submission's bytes (`a4c78a40…`).
- On the public window v7 against v6: macroF1 0.8474 -> 0.8531, wrong frames 13.7% -> 12.9%, team
  share error 0.022 -> 0.005; spellF1 0.6533 -> 0.6286, one 0.8 s possession (10:13.8) merged into
  its neighbour. Seventeen possessions: one spell moves it by about 0.03. Recorded, never a
  tuning signal (RULES; D39). `v5_plus_tv_cue_merged_ball` and `v5_plus_tv_cue` score the same
  on the window but differ elsewhere.
- `v3_kickoff` (D14) and `v4_spread` (D16) were each the submission for part of 4 Nov; their
  files were overwritten and no copy is kept. `make submission METHOD=<name> TEAM=<file>` writes one.
- `evaluation/scoreboard.csv` is untracked and may be stale; `make eval` rewrites it.
- Public scores are from `make eval` on 2026-11-07 (macroF1 / spellF1 on the public 5 minutes).
- Proxy scores are spellF1 on the five IDSSE Bundesliga dev matches under simulator v1 (the
  "Bundesliga" league, D1, D7): "keep2 + D12" is the decision run (controlled-possession truth,
  D21, with the D12 ball stress), "DFL + D12" the same under DFL's first-touch labels, "keep2
  plain" without the ball stress. Each value is from the EXPERIMENTS table of its decision
  (sections 12, 13, 16, 17); "--" means not recorded on that setting, "not run" means the method
  was never scored as submitted there. The two hold-out matches are untouched until Phase 4 (D3).

## 2. The lineage of the submission (`solution/run.py`'s default)

Each step keeps everything before it and adds one part. The proxy column is the keep2 + D12
decision run (`make proxy`) where it exists; earlier steps were decided on DFL truth (D21 came later).

| since | method | decisions | adds | proxy spellF1 (gain over the step before) |
|---|---|---|---|---|
| 2026-10-30 | `semi_markov` | D8, D9 | the semi-Markov decoder: nearest player, missing ball, speed rank, duration priors, all costs 0 | -- |
| 2026-11-02 | `v2_touch` (was `touch_markov`) | D10 | the touch engine's holder, weight 1 | +0.1727 (sim v1, DFL truth) |
| 2026-11-04 | `v3_kickoff` (was `touch_kickoff`) | D13, D14 | the kickoff grammar (D until the detected kickoff) | +0.0000 plain, +0.0015 busy pre-kickoff; a grammar exception to D4 |
| 2026-11-04 | `v4_spread` (was `touch_gapteam`) | D16 | the spread cue, weight 0.5, no say on dead ball | +0.0225 (D12, DFL); 0.4209 on keep2 + D12 |
| 2026-11-04 | `v5_frame_model` (was `touch_c3_seen_w1`) | D22, D23 | the frame model P(home | alive), weight 1, read where the ball is seen | 0.4864 (+0.0655) |
| 2026-11-06 | `v6_deadball_model` (was `touch_c3_seen_w1_dm`) | D26, D37 | the learned dead-ball model, weight 1, in place of the speed rank | 0.5565 (+0.0702) |
| 2026-11-07 | `v7_keep2_control` (was `dm_k2ctl`) | D21, D35, D38, D39 | two parts measured together (section 25): the decoder's priors fitted on keep2 relabels, and the control model (false_drop 0.05) | 0.5854 (+0.0289) |

## 3. The parts

- **decoder**: semi-Markov decoder over blocks of 5 frames: nearest player, missing ball, speed rank, duration priors (D8, D9)
- **touch**: touch engine holder, solution/touch.py (E3, D10); 'hold' = touch_takes off
- **kickoff**: kickoff grammar: D until the detected kickoff, solution/kickoff.py (D13, D14)
- **spread**: spread cue for the team, solution/gapteam.py (E5, D16); weight and where
- **frame**: frame model P(home | alive), solution/framemodel.py (C3, D22, D23); weight and where
- **deadm**: learned dead-ball model in place of the speed rank, solution/deadmodel.py (D26, D37); where
- **cue**: player or ball beyond a line, solution/deadball.py (C2b, D15)
- **tv**: TV cut-away cue, solution/cutaway.py (E13, D27, D31); variant
- **ball**: ball merged with the broadcast's, solution/sources/fused.py (D32)
- **control**: touches dropped by the control model, solution/controlmodel.py (D35)
- **priors**: label variant the priors are fitted on (DFL first touch unless 'keep2', D21)
- **costs**: decoder transition costs or minimum stoppage length, where not the defaults (D8, D36)

What each part reads: the decoder, the kickoff grammar, the spread cue, the frame model and the
dead-ball model read **players only**; the touch engine reads the **ball** (cleaned, smoothed,
frames beyond a line dropped, D7); the beyond-line cue reads a player or the ball beyond a line;
the TV cue reads the **broadcast alignment cache**; the merged ball reads the broadcast's ball.
None reads the Smart Tagging labels or the public truth.

## 4. Every registered method (generated: `make methods`)

| method | decoder | touch | kickoff | spread | frame | deadm | cue | tv | ball | control | priors | costs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `oracle` | no decoder | | | | | | | | | | | |
| `baseline_raw` | nearest player to the ball, majority vote over +-0 frames | | | | | | | | | | | |
| `baseline_smooth25` | nearest player to the ball, majority vote over +-25 frames | | | | | | | | | | | |
| `baseline_smooth50` | nearest player to the ball, majority vote over +-50 frames | | | | | | | | | | | |
| `baseline_smooth125` | nearest player to the ball, majority vote over +-125 frames | | | | | | | | | | | |
| `v1_decoder` (was `semi_markov`) | yes |  |  |  |  |  |  |  |  |  |  |  |
| `v2_touch` (was `touch_markov`) | yes | w1 |  |  |  |  |  |  |  |  |  |  |
| `v3_kickoff` (was `touch_kickoff`) | yes | w1 | yes |  |  |  |  |  |  |  |  |  |
| `touch_deadball` | yes | w1 | yes |  |  |  | w0.5 player+ball |  |  |  |  |  |
| `v4_spread` (was `touch_gapteam`) | yes | w1 | yes | w0.5 all max |  |  |  |  |  |  |  |  |
| `touch_markov_hold` | yes | w1 hold |  |  |  |  |  |  |  |  |  |  |
| `touch_gapteam_hold` | yes | w1 hold | yes | w0.5 all max |  |  |  |  |  |  |  |  |
| `touch_gapteam_k2` | yes | w1 | yes | w0.5 all max |  |  |  |  |  |  | keep2 |  |
| `touch_c3` | yes | w1 | yes | w0.5 all max | w0.5 all |  |  |  |  |  |  |  |
| `touch_c3_w1` | yes | w1 | yes | w0.5 all max | w1 all |  |  |  |  |  |  |  |
| `touch_c3_gaps` | yes | w1 | yes | w0.5 all max | w0.5 gaps |  |  |  |  |  |  |  |
| `touch_c3_only` | yes | w1 | yes |  | w0.5 all |  |  |  |  |  |  |  |
| `touch_c3_seen` | yes | w1 | yes | w0.5 all max | w0.5 seen |  |  |  |  |  |  |  |
| `v5_frame_model` (was `touch_c3_seen_w1`) | yes | w1 | yes | w0.5 all max | w1 seen |  |  |  |  |  |  |  |
| `touch_c3_seen_w1_hold` | yes | w1 hold | yes | w0.5 all max | w1 seen |  |  |  |  |  |  |  |
| `touch_c3_seen_w1_k2` | yes | w1 | yes | w0.5 all max | w1 seen |  |  |  |  |  | keep2 |  |
| `touch_c3_seen_w1_hold_k2` | yes | w1 hold | yes | w0.5 all max | w1 seen |  |  |  |  |  | keep2 |  |
| `v6_deadball_model` (was `touch_c3_seen_w1_dm`) | yes | w1 | yes | w0.5 all max | w1 seen | w1 all |  |  |  |  |  |  |
| `touch_c3_seen_w1_dms` | yes | w1 | yes | w0.5 all max | w1 seen | w1 seen |  |  |  |  |  |  |
| `touch_c3_seen_w1_tv` | yes | w1 | yes | w0.5 all max | w1 seen |  |  | w1 pessimistic |  |  |  |  |
| `touch_c3_seen_w1_tv_opt` | yes | w1 | yes | w0.5 all max | w1 seen |  |  | w1 optimistic |  |  |  |  |
| `touch_c3_seen_w1_tv_cal` | yes | w1 | yes | w0.5 all max | w1 seen |  |  | w1 calibrated |  |  |  |  |
| `touch_c3_seen_w1_fused` | yes | w1 | yes | w0.5 all max | w1 seen |  |  |  | merged |  |  |  |
| `touch_c3_seen_w1_tv_cal_fused` | yes | w1 | yes | w0.5 all max | w1 seen |  |  | w1 calibrated | merged |  |  |  |
| `touch_c3_seen_w1_ctl` | yes | w1 | yes | w0.5 all max | w1 seen |  |  |  |  | yes |  |  |
| `touch_c3_seen_w1_ctl_p05` | yes | w1 | yes | w0.5 all max | w1 seen |  |  |  |  | {'false_drop': 0.05} |  |  |
| `v6_plus_tv` (was `dm_tv`) | yes | w1 | yes | w0.5 all max | w1 seen | w1 all |  | w1 calibrated |  |  |  |  |
| `v6_plus_ball` (was `dm_ball`) | yes | w1 | yes | w0.5 all max | w1 seen | w1 all |  |  | merged |  |  |  |
| `v6_plus_tv_ball` (was `dm_tvb`) | yes | w1 | yes | w0.5 all max | w1 seen | w1 all |  | w1 calibrated | merged |  |  |  |
| `v6_plus_cue` (was `dm_cue`) | yes | w1 | yes | w0.5 all max | w1 seen | w1 all | w0.5 player+ball |  |  |  |  |  |
| `v6_plus_keep2` (was `dm_k2`) | yes | w1 | yes | w0.5 all max | w1 seen | w1 all |  |  |  |  | keep2 |  |
| `v6_plus_dead4` (was `dm_dead4`) | yes | w1 | yes | w0.5 all max | w1 seen | w1 all |  |  |  |  |  | dead 4 |
| `v6_plus_snap` (was `dm_snap`) | yes | w1 | yes | w0.5 all max | w1 seen | w1 all |  |  |  |  |  |  |
| `v6_plus_control` (was `dm_ctl`) | yes | w1 | yes | w0.5 all max | w1 seen | w1 all |  |  |  | {'false_drop': 0.05} |  |  |
| `v6_plus_all` (was `dm_all`) | yes | w1 | yes | w0.5 all max | w1 seen | w1 all | w0.5 player+ball | w1 calibrated | merged |  | keep2 | dead 4 |
| `v6_plus_stack` (was `dm_stack`) | yes | w1 | yes | w0.5 all max | w1 seen | w1 all |  | w1 calibrated | merged | {'false_drop': 0.05} | keep2 |  |
| `v7_keep2_control` (was `dm_k2ctl`) | yes | w1 | yes | w0.5 all max | w1 seen | w1 all |  |  |  | {'false_drop': 0.05} | keep2 |  |
| `v7_plus_ball` | yes | w1 | yes | w0.5 all max | w1 seen | w1 all |  |  | merged | {'false_drop': 0.05} | keep2 |  |
| `c3s_dead1` | yes | w1 | yes | w0.5 all max | w1 seen |  |  |  |  |  |  | dead 1 |
| `c3s_dead2` | yes | w1 | yes | w0.5 all max | w1 seen |  |  |  |  |  |  | dead 2 |
| `c3s_dead4` | yes | w1 | yes | w0.5 all max | w1 seen |  |  |  |  |  |  | dead 4 |
| `c3s_sw1` | yes | w1 | yes | w0.5 all max | w1 seen |  |  |  |  |  |  | switch 1 |
| `c3s_sw2` | yes | w1 | yes | w0.5 all max | w1 seen |  |  |  |  |  |  | switch 2 |
| `c3s_dead2_sw1` | yes | w1 | yes | w0.5 all max | w1 seen |  |  |  |  |  |  | switch 1, dead 2 |
| `c3s_dead4_sw2` | yes | w1 | yes | w0.5 all max | w1 seen |  |  |  |  |  |  | switch 2, dead 4 |
| `c3s_dmin2` | yes | w1 | yes | w0.5 all max | w1 seen |  |  |  |  |  |  | min stop 2 s |
| `c3s_dmin4` | yes | w1 | yes | w0.5 all max | w1 seen |  |  |  |  |  |  | min stop 4 s |
| `touch_c3_seen_w1_dm_ball` | yes | w1 | yes | w0.5 all max | w1 all | w1 all |  |  |  |  |  |  |
| `dm_bor_0` | yes | w1 | yes | w0.5 all max | w1 all | w1 all |  |  |  |  |  |  |
| `dm_bor_5m` | yes | w1 | yes | w0.5 all max | w1 all | w1 all |  |  |  |  |  |  |
| `dm_bor_10m` | yes | w1 | yes | w0.5 all max | w1 all | w1 all |  |  |  |  |  |  |
| `dm_bor_lag1` | yes | w1 | yes | w0.5 all max | w1 all | w1 all |  |  |  |  |  |  |
| `dm_stack3` | yes | w1 | yes | w0.5 all max | w1 seen | w1 all | w0.5 player |  | merged |  | keep2 |  |
| `dm_stack3_dead4` | yes | w1 | yes | w0.5 all max | w1 seen | w1 all | w0.5 player |  | merged |  | keep2 | dead 4 |

Status of the ones not in the lineage (the decision that closed each):
- `touch_deadball` (C2b beyond-line cues): +0.008, below D4; registered only (D15).
- `touch_markov_hold`, `touch_gapteam_hold`, `touch_c3_seen_w1_hold*` (`touch_takes` off): within noise or WORSE on DFL (D10, D21, D25).
- `touch_gapteam_k2`, `touch_c3_seen_w1_k2` (priors on keep2): +0.0189, below D4 (D21, D25).
- `touch_c3`, `touch_c3_w1`, `touch_c3_gaps`, `touch_c3_only`, `touch_c3_seen` (frame-model variants): superseded by `v5_frame_model` (D22, D23).
- `touch_c3_seen_w1_dms` (dead-ball model where the ball is seen only): the D37 fallback, +0.0362; not needed.
- `touch_c3_seen_w1_tv`, `_tv_opt`, `_tv_cal` (TV cut-away cue): ADOPT on the proxy, BAR-ATH check failed on the in-gap team (D27, D31).
- `touch_c3_seen_w1_fused`, `_tv_cal_fused` (merged ball): the merge did not repair that; E13 closed (D32).
- `touch_c3_seen_w1_ctl`, `_ctl_p05` (control model): below D4 and WORSE on DFL (D35).
- `c3s_*` (decoder costs and minimum stoppage): no ADOPT; costs stay 0 (D36).
- `v6_plus_*` (was `dm_*`; the closed parts stacked on the v6 submission, D38, EXPERIMENTS section 25): no single
  part clears D4; `v6_plus_all` and `v6_plus_stack` are ADOPT on keep2 but read the broadcast (D31's
  BAR-ATH rules apply); `v7_keep2_control` (keep2 priors + control model, no broadcast) is ADOPT on
  keep2 (+0.0289), within noise on DFL, and passes the BAR-ATH rules 1 to 5 with the
  possession-count warning: **the submission since D39** (2026-11-07).
- `v7_plus_ball` (the merged ball alone on v7, section 25 run 4): fails D31's rule 4 on BAR-ATH
  (tag agreement in gaps 0.790 -> 0.771); not submitted.
- `touch_c3_seen_w1_dm_ball`, `dm_bor_0`, `dm_bor_5m`, `dm_bor_10m`, `dm_bor_lag1` (E15, the team in
  gaps read off an inferred ball; denis): the imputer WORSE (-0.0050), even the true ball within
  noise (-0.0009); closed (D41, EXPERIMENTS section 26).
- `dm_stack3`, `dm_stack3_dead4` (REVIEW item 4, keep2 priors + merged ball + player cue on v6;
  denis): `dm_stack3_dead4` ADOPT on keep2 (+0.0206) but fails BAR-ATH rules 1 and 5; not
  submitted, closed (D42, section 27).
- `oracle`: the truth itself, for ceilings on the proxy; never a submission.
