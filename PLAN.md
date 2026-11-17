# PLAN.md: More than a Hack 2027, Phase 1 (ball possession)

Status: research and plan, written 2026-10-28. No solution code yet. Owner of this file: tech lead.
Everything marked **[assumption]** is a guess to be confirmed or replaced by a measurement.
**Progress** is tracked in [docs/WORKLOG.md](docs/WORKLOG.md), results in
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md), lasting choices in [docs/DECISIONS.md](docs/DECISIONS.md).

**Calendar reality.** The deadline is **Tue 17 Nov 2026**. That is **20 days** from today, not a
month. The four phases below are compressed to fit, and we submit on **Mon 16 Nov** to keep a day
of buffer.

**Data rule that shapes how we work.** The terms forbid sending match data or any extract of it to a
hosted LLM. So the AI assistant has not looked at anything under `data/`. Every measured number in
this plan comes from the public ground truth and the committed baseline CSVs, which are public. The
data-dependent exploration is specified in Appendix A as a script that a human runs and that prints
aggregates only.

---

## 1. Executive summary

**The problem.** For each of the 199,319 frames of the tactical video of Barcelona v Athletic we
output `H`, `A` or `D`. The hidden labels are a mechanical copy of Metrica's *corrected* tracking:
its ball-dead flag and its owning-team field. We have to reproduce them from the *uncorrected*
automatic tracking, where the ball is missing on 41% of graded frames, and we are ranked on
`spellF1`, which scores each possession as one interval matched by temporal IoU.

**Our winning angle.** We treat this as event-driven segmentation, not per-frame classification. We
detect the few events that create a boundary (a controlled touch by the other team, the ball going
dead, a restart), then decode a grammar-constrained sequence in which every extra possession has an
explicit cost, tuned directly on `spellF1`. We fill the long ball gaps with a player-only "who has
the ball" model trained on other matches that carry corrected possession labels, degraded to look
like this ATD, then adapted to this match by self-training on its own confident frames. Every
decision is made on a proxy leaderboard of about ten external matches scored with the official
scorer: roughly 2,000+ possessions instead of the 17 in the public window.

---

## Findings that drive the plan

### What is being asked and how it is judged

| | |
|---|---|
| Deliverables | One Google Form: `<team>.csv` (199,319 rows, `frame,label`), a public repo URL, a short technical document `<team>.pdf` or `.md`. One response per team; the last edit before the deadline counts. |
| Ranking | `spellF1` on the private full-match truth (158,857 graded frames). Ties go to `macroF1`, then transition F1, then latency. No jury in Phase 1. |
| Noise | Gaps below ~0.04 `spellF1` are noise on the full match. On the public 5 minutes one possession moves the score by ~0.06. |
| Feedback | None. No live board. The public 5 minutes and `validate_submission.py` are the only signals. |
| Reproducibility | `solution/run.py` must regenerate the submitted file byte for byte. Organisers may re-run the winner. |
| Allowed | Any method, model or external data about other games, LLMs, own tracking or detection. |
| Forbidden | Hand labels on this game for any purpose, including validation. Any human-produced data about this game (Opta, StatsBomb, broadcast event feeds). Tuning on the public 5 minutes. Match data into hosted LLMs. |
| Unstated | Runtime and hardware limits for the re-run. Whether shipped model weights are fine. What the "Innovation Challenge and Final Pitch" in RULES.md is. See section 9. |

### What the hidden labels are

CHALLENGE.md says the truth is derived mechanically from corrected tracking: `D` where it marks the
ball dead, otherwise the team it names as owner. So we are reverse-engineering two channels of a
corrected tracking product. Possession is a maximal run of one team with the ball alive. `H D H` is
two possessions, so every dead ball splits possessions even when the same team restarts.

### Measured from public artifacts

These come from `evaluation/gt/BAR-ATH_possession_gt_public.csv` and
`evaluation/results/baseline_raw.csv`. In the raw baseline, `D` means "ATD has no ball on this
frame", so it is a public view of ball coverage.

1. **Ball dropouts are long blocks, not flicker.** There are 150 ball-missing runs in the graded
   frames. The median run is 10.7 s, p75 21 s, p90 34 s, the longest 221 s. Runs of 10 s or more
   carry 85% of the missing frames. Interpolation cannot fill these; about 40% of graded time needs
   inference from the players alone.
2. **The public window is the best-tracked stretch of the match.** Ball coverage per 5-minute block
   is 0.88 at the start of each half and drops to 0.20 in the worst blocks. Half averages are 0.59.
   Public-window scores will overstate the full-match score, which is a second reason, beyond the
   rules, not to tune on it.
3. **A missing ball does not mean a dead ball.** Only 494 of the 1,469 true dead-ball frames in the
   public window have no ball. Two dead stretches, of 20 s and 15 s, have the ball tracked
   throughout. Dead ball needs its own detector.
4. **Proximity is not ownership.** Where the ball is visible and in play, the nearest player's team
   matches the true owner on 78% of frames. Inside long possessions the raw nearest-player signal
   flips 10 to 24 times. One true 1.3 s Barcelona possession never has a Barcelona player nearest
   the ball. Ownership persists while the ball travels, which suggests last-controlled-touch
   semantics.
5. **True possessions are often short.** In the public window 7 of 17 possessions last under 4 s,
   and the shortest lasts 0.76 s. Of the 20 transitions, 12 are direct hand-overs between teams and
   8 pass through a dead ball.
6. **The metric's sensitivities**, measured by perturbing the public truth itself. This is
   illustrative only, on 17 possessions:

   | perturbation of the true labels | spellF1 |
   |---|---|
   | drop every possession shorter than 2 s / 3 s / 4 s | 0.83 / 0.74 / 0.56 |
   | uniform lag of 8 / 12 / 25 frames | 0.95 / 0.91 / 0.78 |
   | random boundary jitter, sd 12 / 25 / 50 frames | 0.94 / 0.84 / 0.73 |
   | one spurious 1 s team flip inside each possession longer than 20 s | 0.70 |
   | one spurious 1 s dead ball inside each possession longer than 20 s | 0.77 |
   | no dead-ball detection, D becomes the previous team / the restarting team | 0.81 / 0.85 |
   | every dead ball starts 2 s late / ends 2 s late / both | 0.98 / 1.00 / 0.95 |
   | 10 s of `X` inside each possession longer than 20 s | 1.00 (macroF1 0.94) |

7. **The baseline over-segments.** Over the full match `baseline_smooth50` reports 594 possessions
   with a mean of 7.2 s. Extrapolating the public window with a more typical dead-ball share
   suggests roughly 250 to 350 true possessions **[assumption]**.
8. **Smart Tagging under-counts restarts.** It has 59 set pieces, 26 of them throw-ins. A
   professional match typically has well over 100 restarts **[assumption]**, so most dead balls must
   be found from tracking. Throw-ins carry no team tag.
9. **The second-half pre-kickoff is a trap.** The second half is graded from frame 110,808 but
   kicks off at 113,742, so 2,934 graded frames are all `D`. The ball is tracked on 81% of them, and
   `baseline_smooth50` labels 89% of them as a team. That creates one long false possession, which
   can merge into the first real one. Kickoff must be detected by code, not by a constant.

### Design principles that follow

- **Optimise segments, not frames.** The decoder must put an explicit price on each extra
  possession. A majority vote cannot.
- **Precision inside long possessions comes first.** One spurious flip per long possession costs
  about 0.30.
- **Short possessions are about 40% of all possessions.** Where the ball is visible we need real
  touch detection, not smoothing that erases them.
- **Dead ball is its own sub-problem.** It splits possessions, and two thirds of it has a tracked
  ball. What matters is whether a stoppage is there at all, not its exact edges. A false dead ball
  (0.77) is worse than none (0.81), so a stoppage needs two agreeing cues. When unsure, label the
  restarting team (0.85).
- **Add a possession only when it will probably match.** F1 is 2·TP / (predicted + true). An extra
  predicted possession raises it only if its chance of matching exceeds about F1/2, before counting
  any damage to its neighbours. That is the decoder's cost rule.
- **Merges are the costliest error.** Missing the true 0.8 s Barcelona spell inside an Athletic
  possession loses about 2.4 of 3 possession credits. So a strong control event must be able to
  create a short possession against the duration prior.
- **Boundary timing to within about half a second is the target for hand-overs and restarts.**
  Beyond that the IoU thresholds start to bite.
- **`X` only moves the tie-break.** The scorer forward-fills `X`, so it equals "repeat my last
  label" for `spellF1`. It only helps `macroF1`, and only on frames where the repeated label is
  probably wrong. Never write `X` at a transition.

---

## 2. Solution architecture

### Data flow

```
 data/tactical (ATD, Smart Tagging, homography, video)     external matches with corrected labels
            |                                              (IDSSE/DFL, Metrica samples, SkillCorner)
            v                                                          |
 C0  loader -> canonical Match arrays, cached  <---- same schema -----+--> C7 ATD-noise simulator
            |                                                          |    + label converter
            v                                                          v
 C1  normalise: metres, attack direction per half, clean ball and players
            |
            v
 C2  evidence:  a) ball kinematics + touch candidates   b) dead-ball cues
                c) team-shape features (always present) d) Smart Tagging features
            |
            v
 C3  frame model: P(H), P(A), P(D) per frame  <---- C6 self-training on BAR-ATH confident frames
            |
            v
 C4  segment decoder: semi-Markov Viterbi, grammar, per-transition costs, duration priors
            |
            v
 C5  boundary refinement: snap each transition to its touch / out / restart event
            |
            v
 C8  writer + validate_submission  ->  evaluation/results/<team>.csv
```

### Components

| | what it does | why this | rejected alternatives |
|---|---|---|---|
| **C0 Loader** | Parses ATD into `players[T,46,3]`, `team[46]`, `ball[T,...]`, plus tagging and homography, cached as `.npz` in a gitignored `cache/`. Same schema for external matches. Reuses `baseline.team_of_each_track`, `load_smart_tagging.load`, `load_homography.load`, `games.half_video_blocks`. Asserts that H is the first ATD team (Barcelona) and maps the tagging's spelling "Atletic Club" to A. | One schema makes every feature and model run unchanged on BAR-ATH and on external matches. That is what makes a proxy leaderboard possible. | kloppy DataFrames end to end: too slow and memory-heavy for 199k x 46 slots. kloppy stays for external loaders. |
| **C1 Normalise and clean** | Metres from `games.GAMES["BAR-ATH"]["pitch"]`. Attack direction per half detected from team mean x after kickoff, never hard-coded. Mirror so home attacks +x. Remove ball teleports above ~40 m/s, interpolate only gaps of 0.4 s or less, smooth velocity. Merge same-team duplicates closer than ~0.7 m. | Features must mean the same thing in both halves and in every external match. | Fancy ball interpolation: dropouts are 10 s blocks, so it buys nothing. |
| **C2a Ball kinematics** | Touch candidates where ball acceleration or direction changes sharply with a player within ~2 m. Control episodes where the ball stays near one player with matching velocity for k frames. Attributed to that player's team. | Finding 4: ownership changes at controlled touches and persists through ball flight. This mirrors how corrected possession channels behave. | Nearest player: 78% agreement and 10 to 24 flips per long possession. |
| **C2b Dead-ball cues** | Ball beyond the lines, ball stationary for about a second or more, collective slowdown of all players, restart formations (kickoff, corner, goal kick, throw-in), Smart Tagging set-piece windows, second-half pre-kickoff. A stoppage needs at least two agreeing cues. Throw-in team comes from geometry, because the tag says `N/A`. | Finding 3: two thirds of dead-ball frames have a ball, so ball-missing is not a proxy. Laws of the game are stable across matches. | Ball-missing as `D`, which is what the baseline does. Its D-class F1 is 0.32. |
| **C2c Team-shape features** | Per team, at 0.5, 1, 2, 5 and 10 s windows, past and future: centroid, length, width, defensive line, mean speed, velocity towards goal, players in each third, pressing counts, hull area. | Always present, including during the 41% of ball gaps. The task is offline, so future context is allowed. | Raw coordinates into a deep net first: too little labelled data with the right semantics to learn shapes from scratch in 20 days. |
| **C2d Tagging features** | Active code by team, signed: attacking codes imply that team has the ball, "defending in" codes imply the other team does. Time to the nearest set-piece start. | Free, automatic, allowed. Covers phases the tracking misses. | Copying intervals as labels: they are phases, not possessions. |
| **C3 Frame model** | LightGBM multiclass giving P(H, A, D) per frame. Trained on degraded external matches, then adapted with BAR-ATH pseudo-labels. Optional v2 is a small BiGRU or TCN over ±10 s of features. | Fast, deterministic, handles NaN natively, easy ablations, runs on CPU. | End-to-end video models as the core, for the reasons in "Video" below. |
| **C4 Segment decoder** | Semi-Markov Viterbi over {H, A, D}, run at 5 Hz for speed, with boundaries refined at 25 fps in C5. Emissions are calibrated log-probabilities from C3. C2 control events act as transition bonuses at their frames and may create short possessions against the duration prior. Separate costs for H to A, alive to D, and D to team, set from the F1/2 rule. Minimum durations and duration priors are fitted on external labels. Grammar: a dead ball ends only at a restart; the second half is D until a detected kickoff. Costs are tuned on the proxy league against the official `spell_f1_mean`. | The baseline's failure is fragmentation (finding 7). A decoder with a price per segment controls the count-versus-recall trade-off directly. It is exact, deterministic and numpy-only. | Majority vote or median filter: no grammar and no notion of cost. CRF libraries: an extra dependency and more determinism risk for no gain. |
| **C5 Boundary refinement** | For each decoded hand-over or restart, search ±1.5 s for its event: the first controlled touch by the new team, or the restart kick. Snap to it. The start of a dead ball barely matters, so it gets no special effort. | A 12-frame lag costs about 0.09, and jitter with sd 25 frames costs about 0.16 (finding 6). | Leaving boundaries where the decoder puts them: emissions are smooth, so boundaries drift late. |
| **C6 Self-training** | Pseudo-labels on BAR-ATH from frames where the ball is visible, one player clearly controls it for at least 1 s, the decoder margin is high and Smart Tagging agrees. Retrain the team-shape model with them, then apply it to ball gaps. Check it by hiding the ball on confident stretches, with gap lengths drawn from finding 1, and measuring recovery. | RULES.md allows algorithmic pseudo-labels on the provided game. Barcelona and Athletic have their own shapes in and out of possession. | Hand labels: forbidden. |
| **C7 External data and simulator** | Converts external matches to `frame,label` truth files plus canonical inputs. Degrades clean inputs to match BAR-ATH ATD aggregates from Appendix A: ball dropped in long blocks with the measured length distribution and context, track fragmentation, about 20.5 visible players, noise and duplicates. | Gives labelled BAR-ATH-like data to train on, and a validation set about 100 times larger than the public window. | Training on clean external data: the model would lean on a ball that is missing 41% of the time here. |
| **C8 Writer** | `H`/`A`/`D` on every graded frame. `X` on ungraded frames, and only where a module deliberately abstains inside a decoded possession. Written by `baseline.write_submission`, validated by `evaluation/validate_submission.py`, as `solution/run.py` already does. | Keeps the tie-break clean and the file contract safe. | Heavy use of `X`: no benefit on the ranking metric. |

**External data sources**, all public and about other games:

| source | what | labels we get | notes |
|---|---|---|---|
| IDSSE / DFL open data (Bassek et al., Sci. Data 2025) | 7 Bundesliga matches, 25 fps, corrected positions, official events | per-frame `BallPossession` and `BallStatus`, read by kloppy's sportec loader as `ball_owning_team` and `ball_state` | CC-BY 4.0. Closest in kind to our truth. |
| Metrica sample data | 3 anonymised matches, same vendor, same 0-1 coordinates; game 3 is in the same FIFA EPTS format as our ATD | derived by our converter from manual events (set piece, ball out, recovery, ball lost) | Same vendor conventions, but labels are our derivation. |
| SkillCorner open data | broadcast tracking, 10 fps, about 10 matches | per-frame possession group | Noisier and closer to "automatic". Resample to 25 fps. |
| PFF FC World Cup 2022 | 64 matches, broadcast tracking plus events | possession events | Registration needed. Ask organisers first. |

**Video, audio and broadcast (gated, see backlog E10, E12, E13).** A ball detector on the tactical
video could recover part of the 41% gap, but only if the ball is visible to the eye in those
stretches. Humans check that by watching about 20 random gap blocks, which RULES.md allows. A
whistle detector on the audio track is cheap if the file has audio. Broadcast replays, which show up
as invalid homography, mostly happen during stoppages. Aligning broadcast to tactical is possible by
matching the two ATDs' player positions, which share pitch coordinates. End-to-end video action
spotting such as T-DEED from SoccerNet stays out of the core: it is trained on broadcast footage, the
tactical camera is a different domain, it needs heavy compute we do not have, and its outputs do not
map directly onto ownership semantics.

### Tech stack

| layer | choice | why |
|---|---|---|
| Language and env | Python 3.11, the existing `.venv`, `uv`, `make lock` | Already set up and pinned in the repo. |
| Arrays and IO | numpy 2.4, pandas 3.1 (pinned) | Same as the scorer. |
| External loaders | kloppy 3.18 (pinned) | Has sportec, metrica EPTS and skillcorner loaders. |
| Frame model | LightGBM, new pin, run with `deterministic=True`, fixed seed and threads | Speed, NaN handling, determinism. |
| Optional temporal model | PyTorch, CPU inference only | MPS is not deterministic. |
| Optional video and audio | ffmpeg via PyAV, pinned wheel | Only if a gate opens. |
| Hardware | Apple M2 Pro, 16 GB RAM, about 58 GB free disk | Enough for everything except heavy video training. |

**Reproducibility design.** Model weights trained on external data are shipped in the repo, with the
deterministic script that rebuilds them. Self-training on BAR-ATH runs inside `run.py` with fixed
seeds and threads. Probabilities are rounded to 1e-4 before decoding so last-bit float differences
across machines cannot flip a decision. Phase 4 checks that the output's sha256 matches on a clean
macOS clone and in a Linux container. Target runtime is under 20 minutes on CPU **[assumption: no
stated limit]**.

---

## 3. Phased roadmap (20 days)

Today is Wed 28 Oct. A working baseline submission already exists: `make submission` reproduces
`baseline_smooth50`. "End of week 1" means our own method, end to end, by Tue 3 Nov.

### Phase 0: setup and facts (Wed 28 to Thu 29 Oct)

- Everyone runs `make setup` and `make submission TEAM=check`, and confirms `make eval` shows the
  baseline scoreboard.
- A human runs the Appendix A script and pastes the aggregates into `notebooks/eda/FINDINGS.md`.
- Humans watch 20 randomly chosen ball-gap blocks and 10 dead-ball candidates in the tactical video,
  recording only yes/no answers about whether the ball is visible. This feeds the video gate. It
  must not produce labels.
- Download IDSSE (7 matches), Metrica samples (3) and SkillCorner (about 10) with a scripted,
  checksummed `make external`.
- Send the section 9 questions to the organisers.
- Freeze the module interfaces from section 4.

**Deliverable:** FINDINGS.md, external data on disk, interfaces merged as stubs.
**Done when:** every teammate can run the stub pipeline end to end and get a valid file.

### Phase 1: end-to-end v1 and the proxy league (Fri 30 Oct to Tue 3 Nov)

- C0 and C1 for BAR-ATH and all external sources.
- C7 label converters and simulator v0, with ball dropout blocks only.
- Proxy league v0: a script that runs any `predict(match)` on every external match and scores it
  with `possession_metrics.evaluate`, with a bootstrap over possessions.
- Rule-based v1: C2a touch engine, C2b dead-ball rules, C4 decoder with hand-set costs, then costs
  tuned on the proxy league.
- First real submission through `solution/run.py`, validated.

**Deliverable:** `v1.csv` and a proxy-league table for baseline_smooth50 versus v1.
**Done when:** v1 beats the baseline on the proxy league by more than the bootstrap band; the
public-window check shows no alignment or mapping bug; `run.py` finishes in under 15 minutes; and
two runs produce identical sha256.

### Phase 2: learned evidence (Wed 4 to Mon 9 Nov)

- C2c team-shape features and C2d tagging features.
- C3 LightGBM trained leave-one-match-out on degraded external data, with calibration.
- C6 self-training on BAR-ATH plus the synthetic-occlusion test.
- Simulator v1 calibrated to Appendix A, including dropout context and player noise.
- Restart-team classifier for D-to-team transitions.
- Re-tune decoder costs.

**Deliverable:** `v2.csv` and an ablation table where each component is shown on and off with
confidence intervals.
**Done when:** v2 beats v1 on the proxy league beyond noise, and synthetic-occlusion team accuracy
inside gaps is clearly above "carry the last owner forward".

### Phase 3: boundaries, gaps and one stretch bet (Tue 10 to Fri 13 Nov)

- C5 boundary refinement.
- One stretch track chosen at the Phase 0 gate: video ball detector, whistle detector, or broadcast
  alignment. Or none, if no gate opened.
- Optional temporal model (C3 v2) and an ensemble.
- **Safety net on Tue 10 Nov:** freeze and validate a tracking-plus-tagging submission with no
  stretch components, so a broken stretch track can never leave us without a file.
- **Feature freeze Fri 13 Nov, end of day.**

**Deliverable:** release candidate `rc.csv`.
**Done when:** the RC beats v2 on the proxy league and on the two held-out external matches, and
every component in it has an ablation row.

### Phase 4: freeze and ship (Sat 14 to Tue 17 Nov)

- Reproduce from a clean clone on two machines plus a Linux container. Hashes must match.
- Write the technical report from `docs/TECHNICAL_REPORT.md`.
- Final public-window correctness check.
- Make the repo public and check for data with `git ls-files data/`.
- Submit the form on **Mon 16 Nov**.

**Deliverable:** form submitted with CSV, repo URL and PDF.
**Done when:** the TEAM.md submission checklist is fully ticked.

---

## 4. Work split

Written for four people. With five, add R5. With three, R1 also takes R2's dead-ball work and R2's
touch engine moves to R3.

| role | owns | first deliverable | depends on |
|---|---|---|---|
| **R1 Lead and integrator** | `solution/run.py`, C4 decoder, C5 refinement, C8, reproducibility, final submission, code review | decoder on baseline evidence by 31 Oct | C0 stub |
| **R2 Ball and dead ball** | C1 ball cleaning, C2a touch engine, C2b dead-ball cues, restart detection, the video gate | dead-ball rules on proxy league by 2 Nov | C0 |
| **R3 ML and team shape** | C2c features, C3 frame model, C6 self-training, optional temporal model | LightGBM v1 by 6 Nov | C0, C7 |
| **R4 Data and validation** | C0 external loaders, C7 converters and simulator, proxy league, Appendix A, C2d tagging features, technical report editor | proxy league v0 by 1 Nov | none |
| **R5 Video, audio, broadcast** (if five) | the gated stretch tracks, the local possession explorer | gate answers by 30 Oct | Phase 0 viewing |

**Interfaces, frozen on day 2 so nobody blocks anyone:**

- `solution/data.py`: `load_match(name) -> Match` with numpy arrays, the same for BAR-ATH and
  external matches.
- `solution/features/<name>.py`: `compute(match) -> dict[str, np.ndarray of length T]`.
- `solution/model.py`: `predict_proba(match) -> (T, 3)` over H, A, D.
- `solution/decode.py`: `decode(proba, events, params) -> codes`, using the scorer's 0..3 codes.
- `tools/proxy_league.py`: `score(predict_fn) -> table` built on `possession_metrics.evaluate`.

Each owner works against stubs that already satisfy these signatures. Small PRs into `main`, per
TEAM.md. `main` must always run `make submission`.

---

## 5. Experiments backlog

Impact is the expected `spellF1` gain on the proxy league **[assumption]**. Each row is replaced by
a measurement once run. Effort is in person-days.

| # | experiment | impact | effort | gate or note |
|---|---|---|---|---|
| E1 | Semi-Markov decoder over baseline evidence, replacing the majority vote | high, +0.05 to 0.10 | 1 to 2 | first thing to build |
| E2 | Dead-ball detector v1: out of bounds, stationary ball, collective slowdown, set-piece tags, kickoffs | high, +0.05 to 0.10 | 2 to 3 | |
| E3 | Last-touch ownership engine with control hysteresis on visible-ball frames | high, +0.05 to 0.10 | 3 | |
| E4 | Proxy league and ATD simulator | enabler | 3 to 4 | start on day 1; nearly every later row depends on it |
| E5 | Player-only possession model for ball gaps, external plus pseudo-labels | high, +0.05 to 0.15 | 3 to 4 | gaps are about 40% of graded time |
| E6 | Boundary refinement by snapping to events | medium, +0.02 to 0.05 | 1 to 2 | |
| E7 | Restart-team classifier for D to team | medium, +0.02 to 0.04 | 1 to 2 | throw-ins have no team tag |
| E8 | Smart Tagging team-phase features, with tag reliability estimated from agreement with the other channels, since no external match has tags | medium, +0.01 to 0.03 | 1 to 2 | |
| E9 | Self-training rounds with the synthetic-occlusion test | medium, +0.02 to 0.05 | 2 | needs E5 |
| E10 | Whistle detection from the audio track, by its 3 to 4 kHz tone, for fouls and offsides | medium, +0.01 to 0.04 | 1 to 2 | checking for an audio track takes 10 minutes; do it in Phase 0 |
| E11 | Temporal network on per-frame features | medium, +0.02 to 0.05 | 3 to 4 | only after E5 plateaus |
| E12 | Video ball detector inside ATD gaps | uncertain, up to +0.15 | 6 to 10 | only if humans see the ball in most sampled gaps |
| E13 | Broadcast alignment: replays as dead-ball evidence, a second ball source | medium, +0.01 to 0.04 | 5 to 7 | closed (D27, D31, D32): cue +0.030 on the proxy, failed the BAR-ATH check on the in-gap team |
| E14 | Ensemble of frame models and decoder settings | low to medium, +0.01 to 0.02 | 1 | Phase 3 |
| E15 | Set-transformer ball and possessor inference (BallRadar or TranSPORTmer style) | medium to high | 5 to 8 | only if E5 plateaus early |

---

## 6. Evaluation strategy

We may not tune on the public window and may not hand-label this match. So the validation we trust
is built from other matches, and BAR-ATH gets label-free diagnostics only.

**L1. Proxy league, the decision metric.**
- About ten external matches converted to `frame,label` truth files, with `X` outside play.
- Inputs degraded by the simulator with three fixed seeds.
- Scored by the official code, either `possession_metrics.evaluate` or
  `evaluation/run_evaluation.py --gt <file> --results <dir>`, which accepts any truth path.
- Reported per match and as a mean: `spellF1`, `macroF1`, possession count ratio, and an error
  breakdown covering missed short possessions, spurious flips, dead-ball errors and boundary
  latency.
- Anything trained is scored leave-one-match-out.
- **Accept a change only if** the mean gain is at least 0.02 and exceeds the bootstrap band, with
  possessions resampled across all matches, and it wins on at least two thirds of the matches.
- Two external matches stay untouched until Phase 4, to measure how much we overfit the proxy
  league itself.
- The Metrica sample matches are the label-convention set: same vendor, same format. They set the
  control threshold and any global timing lag, and are looked at rarely, so they stay close to
  held out.

**L2. Label-semantics check.** Each external source defines possession its own way. For each one we
compare duration distributions, the share of possessions under 2 s, and the share of direct versus
dead-ball transitions. The public window may be used here only to check that our converters produce
the same *kind* of structure: short possessions exist, and ownership persists through ball flight.
It is never used to pick a parameter. The report says this explicitly.

**L3. BAR-ATH diagnostics, with no labels.**
- Predicted possession count and duration distribution against the external priors.
- Agreement with Smart Tagging: the share of attacking-phase frames where our owner matches the tag
  team, and the share of set-piece windows that contain a predicted restart.
- Synthetic-occlusion recovery on pseudo-labels.
- Stability: a robust configuration does not swing the possession count by 30% under small
  parameter changes.

**L4. Public window, for correctness only.** Run `make eval` at each milestone and log the result.
It catches off-by-one errors, label mapping mistakes and catastrophic regressions. It is never used
to compare two candidates.

---

## 7. Risks and fallbacks

| risk | likelihood | what we do about it | plan B |
|---|---|---|---|
| External label semantics differ from Metrica's, so the proxy optimum is not the real one | high | Use several sources and pick settings that are robust across all of them. Keep decoder parameters in physical units. Ask the organisers how ownership flips. | Prefer rules grounded in the laws of the game over parameters fitted to one source. |
| Ball gaps cannot be inferred from players | medium | Synthetic-occlusion test early in Phase 2. | Precision-first gap policy: keep the last owner unless shape evidence is strong. Open the video gate. |
| A model trained on clean external data is miscalibrated on this noisy ATD | medium | Simulator calibrated to Appendix A, plus self-training on this match. | Fall back to hand-weighted evidence with about six parameters, with decoder costs still tuned on the proxy league. |
| False dead balls split real possessions | medium | Two agreeing cues required. Measured separately on the proxy league. | When unsure, label the restarting team instead of `D`. |
| Output is not byte-identical on the organisers' machine | medium | CPU only, fixed threads, rounded probabilities, pinned wheels, Linux container check. | Remove any non-deterministic component from the final path, even at some score cost. |
| 20 days is not enough | high | Hard gates and feature freeze on 13 Nov. v1 is shippable from 3 Nov. | Ship the best proxy-league version that passed the reproducibility check. |
| A rule interpretation goes against us (pseudo-labels, tagging as validation, published kickoff frame) | low to medium | Ask in writing during Phase 0. Detect kickoffs from tracking, never from FAQ constants. | Drop the disputed component. Each one has an ablation row showing its cost. |
| Accidental tuning on the public window | medium | L4 discipline: log it, never select on it. | Code review rejects PRs justified by public-window gains. |
| Match data leaks to a hosted LLM or into the public repo | low, but disqualifying | `.claude/settings.json` denies, pre-commit hook, no data in issues or PRs. | Rotate the repo history if anything slips. Tell the organisers. |
| External data access or licence problems | low | IDSSE is CC-BY. Metrica samples are public. | Drop PFF and SkillCorner; IDSSE plus Metrica is enough. |
| Smart Tagging errors propagate | medium | Soft features only, never copied into labels. | Ablate it out if the proxy league shows no gain. |
| Compute or memory: video on 16 GB | low | Arrays are about 110 MB. Video work runs in chunks. | Skip the video track. |

---

## 8. Final presentation and demo plan

Phase 1 has no jury and no presentation; the number decides. Two things still matter. The technical
document makes the score credible. RULES.md also mentions an "Innovation Challenge and Final Pitch"
for teams of three or more, which suggests a later stage this repo does not describe. We prepare for
it cheaply.

- **Technical report, 4 to 6 pages.** Approach and why; assets used and dropped; a training-data
  table naming source, labeller and size; the possession rules we implemented, covering loose balls,
  deflections and restarts; the proxy-league ablation table with confidence intervals; limitations;
  reproduction steps.
- **Local possession explorer, never published.** A timeline of predicted possessions, the events
  that justify each boundary, and the Smart Tagging phases. Clicking a boundary opens that video
  frame with the tracking projected through `load_homography.project`. It is our debugging tool and
  doubles as the explainability demo.
- **The three-slide story.** First, why `spellF1` turns this into a segmentation problem, using the
  sensitivity table. Second, our proxy league as a validation set about 100 times larger than the
  public window. Third, inferring possession from team shape while the ball is invisible, with the
  synthetic-occlusion result.
- **We show no match-derived visuals in public** until the organisers confirm what is allowed.

---

## 9. Open questions for the organisers

1. What are the "Innovation Challenge" and "Final Pitch" in RULES.md? Does the Phase 1 rank feed
   them, and what do they require?
2. Are IDSSE/DFL open data, Metrica sample data, SkillCorner open data and the PFF World Cup 2022
   data (which needs registration) acceptable as training and validation data?
3. May we train on algorithmic pseudo-labels from BAR-ATH and validate with synthetic occlusion of
   those pseudo-labels?
4. May we use the automatic Smart Tagging as a validation signal on BAR-ATH?
5. May the pipeline use the published half boundaries and the second-half kickoff frame from the
   FAQ, or must it detect them?
6. What runtime and hardware will the reproduction run use? Is a GPU assumed or forbidden? May we
   ship model weights trained on external data? Is there a repo size limit?
7. ~~In the corrected tracking, does ownership flip at the first touch by the other team or only after
   control? How are blocked passes and deflections treated?~~ **Answered by CHALLENGE.md
   ("Possession rules"):** only on *controlled* possession. A simple touch, a deflection that does
   not intercept, or an aerial duel won without control is not a change, and a loose ball stays
   with the last team. DFL flips at the first touch, so the proxy labels need relabelling (D20).
   Still open: does a dead ball start at the whistle or when the ball crosses the line, and how
   long counts as "control"?
8. ~~Does the ATD metadata define any channel beyond positions and speed?~~ **Answered by our own
   check (notebooks/eda/FINDINGS.md):** no. There are only x, y and speed channels. The ball's
   speed value is missing on 91% of frames and does not track real ball speed, so we compute
   speed from positions. No need to ask.
9. Is audio a permitted input? Check first whether the video has an audio track; that check is
   deferred and only matters for the whistle idea, E10.
10. Are aggregate figures derived from the data, such as histograms, allowed in the public technical
    document?
11. What time of day and time zone is the 17 November deadline?

---

## Decisions needed from the team

1. Team size and an owner for each role in section 4.
2. Whether one person spends about 10 days on a video or audio stretch track, decided at the Phase 0
   gate.
3. Approval to use external data, and who sends the organiser questions. A public GitHub issue helps
   everyone; email to the organisers is private.
4. Whether to build on pseudo-labels and tagging-as-validation before the organisers answer.
5. CPU-only deterministic final pipeline (recommended), or allow GPU or video in the reproducible
   path.

---

## Appendix A: aggregate EDA for a human to run

Script: `notebooks/eda/eda_aggregates.py`. It prints counts, percentiles and histograms only,
never rows or frame-level values. A human decides what to paste into `notebooks/eda/FINDINGS.md`.

- **A1 ATD schema.** Channel names declared in the metadata XML, without values. The number of
  fields in the ball group and what the third one means.
- **A2 Ball.** Visibility per half and per 5-minute block. Gap-length percentiles. Share of visible
  frames outside the pitch lines. Speed percentiles (p50, p95, p99, max). Count of teleports above
  40 m/s. Stationary-ball stretches (speed under 0.5 m/s for at least 1 s): count and length
  percentiles.
- **A3 Players.** Visible count per team per frame (p5, p50, p95). Share of frames with more than 11
  on a team. Slot lifetime percentiles. Same-team duplicates closer than 0.7 m. Speed percentiles.
- **A4 Gap context.** Mean player speed and team spread inside versus outside ball gaps. Zone of the
  last seen ball before each gap (box, wide, middle), as a histogram.
- **A5 Smart Tagging.** Duration percentiles per code. Overlap matrix between codes. Set pieces per
  half and type. `N/A` team counts per code.
- **A6 Video.** `ffprobe` stream list for both videos: codecs, frame counts, frame rate, whether
  audio exists.
- **A7 Broadcast ATD.** Ball visibility, and the count and lengths of valid-homography stretches.
- **A8 Direction of play.** Each team's mean x per half. Result: compare home with away over the
  whole half. The first 30 s do not work, because the second half's first 30 s fall before
  kickoff.

---

## Appendix B: references

- Vidal-Codina et al., *Automatic event detection in football using tracking data*, Sports
  Engineering 2022. Rule-based possession and event detection. https://arxiv.org/abs/2202.00804
- Link and Hoernig, *Individual ball possession in soccer*, PLOS ONE 2017. Control model,
  F = 0.83 for possession. https://pmc.ncbi.nlm.nih.gov/articles/PMC5503225
- Kim et al., *Ball trajectory inference from multi-agent sports contexts*, KDD 2023. Ball and
  possessor from players only. https://arxiv.org/abs/2306.08206
- Capellera et al., *TranSPORTmer*, ACCV 2024. Ball inference and per-frame game-state
  classification, including out of play. https://arxiv.org/abs/2410.17785
- Peral et al., *Multi-modal soccer scene analysis with masked pre-training*, WACV 2026. Ball state
  and possessor from trajectories and crops. https://arxiv.org/abs/2512.19528
- SoccerNet challenge results 2024 and 2025 (ball action spotting, team ball action spotting,
  T-DEED). https://arxiv.org/abs/2409.10587 and https://arxiv.org/abs/2508.19182
- Bassek et al., *An integrated dataset of spatiotemporal and event data in elite soccer*, Scientific
  Data 2025 (IDSSE). https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11787359/
- kloppy sportec loader mapping `BallPossession` and `BallStatus`:
  https://github.com/PySport/kloppy
- Metrica sample data: https://github.com/metrica-sports/sample-data
- SkillCorner open data: https://github.com/SkillCorner/opendata
- PFF FC World Cup 2022 dataset: https://www.blog.fc.pff.com/blog/pff-fc-release-2022-world-cup-data
