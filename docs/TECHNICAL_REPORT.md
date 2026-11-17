# Patrioti — Technical report

More than a Hack 2027, Phase 1: frame-level ball possession on BAR-ATH.
Submission: `Patrioti.csv`, written by `make submission TEAM=Patrioti` (method `v7_keep2_control`).

## Approach

**One decoder, several kinds of evidence.** Every 5-frame block (0.2 s) inside the graded halves
is decoded as Home, Away or Dead by a semi-Markov segment decoder (`solution/decoder.py`). It
chooses the most likely sequence of possessions and stoppages, using duration priors for each
segment type. Each piece of evidence below adds a log-likelihood term per block. All of them
are seeded and deterministic:

| part | what it reads | what it contributes |
|---|---|---|
| touch engine (`solution/touch.py`) | ball and players | who is touching the ball: a player within 3 m of the ball when its velocity changes |
| control model (`solution/controlmodel.py`) | touches | drops touches that are unlikely to change control (deflections, contested touches) |
| frame model (`solution/framemodel.py`) | players only | P(home has the ball \| ball alive), read where the ball is tracked |
| spread cue (`solution/gapteam.py`) | players only | home's outfield width minus away's (the team in possession spreads out), read everywhere |
| dead-ball model (`solution/deadmodel.py`) | players only | P(dead) from how the players move and where they stand around each frame |
| kickoff grammar (`solution/kickoff.py`) | players only | each half is Dead until the kickoff, found from the Law 8 formation hold and the speed rise after it |

The three learned parts are scikit-learn `HistGradientBoostingClassifier` models with fixed
seeds. They are trained at run time on the external matches below, never on the provided game.
Player speeds are read as ranks within the match rather than in m/s: BAR-ATH plays about 10%
slower than our training matches.

**Why this rather than something else.** The organisers' baseline picks the nearest player and
smooths. It fails in two places: where the ball is not tracked (about 40% of graded frames), and
at stoppages, where the nearest player says nothing. A sequence decoder carries possession
through ball gaps and makes stoppages explicit segments. Separate evidence terms let us measure
each cue on its own and keep only the ones that help.

**How we decided what goes in.** You cannot tune on one match with a 5-minute answer key of 17
possessions; on the baselines it picks the wrong smoothing window. So every decision was made
on a **proxy league**:
- **Matches:** five external matches with per-frame possession labels.
- **Degradation:** our simulator (`solution/simulate.py`) makes them look like the BAR-ATH
  tracking. It calibrates where the ball goes missing (in slow play), the ball's smoothing and
  jumps, and how messy the player tracking is, all against label-free aggregates of BAR-ATH.
- **Scoring:** the official `possession_metrics.py`.

A change was adopted only if all three held:
- the pooled spellF1 gain is at least 0.02;
- its paired bootstrap interval excludes 0;
- it wins on at least two thirds of the matches.

It then had to pass checks on BAR-ATH aggregates fixed before they were run (no labels; for
example dead-ball share, agreement with the Smart Tagging team, possessions per 90).
`docs/DECISIONS.md` records every decision and `docs/EXPERIMENTS.md` every result, including the
ideas that failed.

## Assets used

| asset | use |
|---|---|
| tactical ATD (players and ball) | **the only input of the submission** |
| organisers' half spans (`games.py`) | which frames are graded (X outside them) |
| Smart Tagging | label-free sanity checks only (team agreement inside ball gaps, set-piece coverage); not an input |
| public 5-minute ground truth | pipeline correctness only (alignment, file format, no team swap); never used to choose or tune |
| broadcast video, ATD and audio | tried, dropped (below) |
| homography | not used: the ATD is already in pitch metres |

Tried and dropped (all registered in `solution/methods.py`, results in EXPERIMENTS):
- **TV cut-away cue** (the broadcast cuts to replays and close-ups during stoppages): it failed
  its BAR-ATH team checks.
- **Merging the broadcast ball into the tactical ball:** under the bar.
- **Ball imputation inside gaps:** even the true ball adds nothing there.
- **Referee-whistle detection from the audio:** the in-play cue and the end-of-half rule were
  both within noise.
- **Self-training on BAR-ATH**, swapping the frame model's classifier for other learners, and a
  post-hoc spell verifier.

## Training data

| dataset | source | who labelled it | how much | use |
|---|---|---|---|---|
| IDSSE (Bassek et al., *Scientific Data* 2025; DOI 10.6084/m9.figshare.28196177), CC-BY 4.0 | pinned Hugging Face mirror `pysport/idsse-data`, checksummed (`make external`) | DFL / Sportec Solutions: per-frame `BallPossession` and `BallStatus` with official event data, part of the official league data feed | 7 Bundesliga matches. **5 used** for training and proxy scoring (J03WMX, J03WOH, J03WOY, J03WPY, J03WQQ). **2 held out**, never tuned on (J03WN1, J03WR9) | trains the three models and the decoder priors; proxy-league truth |

DFL's possession flips at the first touch by the other team. CHALLENGE.md defines possession as
*controlled* possession, so `solution/relabel.py` relabels the IDSSE labels automatically:
- **Rule (`keep2`):** a team spell shorter than 2 s is a touch, not control, and goes back to
  the previous team. Restarts are exempt.
- **Where the 2 s comes from:** the challenge text, fixed before any measurement and never tuned.
- **What it feeds:** the decoder priors are fitted on these relabels, and the proxy league
  scores on them. Every change also had to be not worse on DFL's original labels.

SkillCorner open data (MIT, 20 A-League matches) was examined once, to compare its
possession-phase labels with ours. It is not used by the submission.

**No labels from the provided game are used** in training or in the submitted method.

## Possession rules implemented

- **Change of possession = the other team gains control.**
  - The touch engine detects touches.
  - The control model removes touches that look like deflections, blocks or contested touches.
  - The keep2 duration priors make spells under about 2 s unlikely.
  - So a loose or contested ball, or an aerial duel won without control, stays with the team
    that last had it.
- **Ball not tracked.** The decoder keeps the current possession through the gap unless the
  player-only evidence (spread cue, dead-ball model) says otherwise. It does not invent a ball
  position.
- **Dead ball.** A stoppage is a Dead segment, decided by the learned dead-ball model and the
  decoder's stoppage priors. The ball's position is not used to call dead ball, because the
  tracked ball is unreliable at the lines. After a stoppage a new possession begins, and the same
  evidence decides which team has it.
- **Kickoffs.** Each half is Dead until the detected kickoff. If no kickoff is found, the half is
  decoded unchanged.
- **X** exactly outside the organisers' graded halves; H, A or D on every graded frame (no
  abstention).

## Results

Proxy league: spellF1 on the five development matches, simulator v1 with the ball stress (noise,
jumps, smoothing).

| method | proxy, keep2 truth | proxy, DFL truth | public 5 min macroF1 / spellF1 |
|---|---|---|---|
| organiser baseline (smooth 50) | 0.284 | 0.255 | 0.621 / 0.326 |
| semi-Markov decoder alone | — | 0.287 | 0.789 / 0.351 |
| + touch engine | — | 0.357 | 0.808 / 0.433 |
| + kickoff grammar, spread cue, frame model | 0.486 | 0.412 | 0.837 / 0.582 |
| + dead-ball model | 0.557 | 0.460 | 0.847 / 0.653 |
| **+ keep2 priors and control model (submitted)** | **0.585** | 0.453 | **0.853** / 0.629 |

- **Last step:** +0.029 on keep2 (interval +0.020 to +0.040, 5 of 5 matches); −0.008 on DFL,
  within noise.
- **Public window:** 17 possessions, so one spell moves spellF1 by about 0.03. Its scores are
  recorded, never used to choose.
- **On BAR-ATH** the submission has 236 possessions, mean 15.0 s. Two runs give byte-identical
  files.
- **Hold-out:** the two held-out IDSSE matches have not been scored with the final method.

## Limitations

- **Domain gap.** The proxy is Bundesliga tracking with simulated degradation, calibrated only to
  BAR-ATH aggregates. The last step cut possessions per 90 on BAR-ATH by a factor of 0.82
  (244 to 201), slightly more than the 0.86 the proxy predicted. That tripped one of our warning
  checks, and we did not resolve it.
- **Label semantics.** We bet on controlled possession. If the truth flips at first touch, the
  previous version (without keep2 priors and the control model) would have been slightly better.
- **Short possessions.** Possessions under about 2 s are mostly merged into their neighbours by
  design. In the public window this costs one 0.8 s possession.
- **Inside ball gaps** the team is read from the players only: 0.79 agreement with the Smart
  Tagging team, against 0.89 where the ball is seen.
- **End of halves.** The graded halves run past the final whistle. In half 1 the method still
  decodes 2 short possessions while the players walk off. A whistle rule to fix this did not
  pass our checks.
- **Ball smoothing.** The touch engine is tuned to the BAR-ATH ball's smoothing, which is matched
  through aggregates only.
- **With more time:** score the hold-out matches, add labelled matches from another league to
  the proxy, and look for evidence about short possessions that does not depend on the ball.

## Reproduction

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/). Dependencies are pinned in
`requirements-lock.txt`.

```bash
make setup                       # virtualenv with the pinned dependencies
make external                    # IDSSE training matches, about 2.5 GB, pinned and checksummed
# unpack the organisers' data bundle into data/ (or set MTAH_DATA_DIR)
make submission TEAM=Patrioti    # writes and validates evaluation/results/Patrioti.csv
```

The three models and the decoder priors are fitted during the run, on the IDSSE development
matches. Two runs give byte-identical output, sha256
`e3ba84bfe9b46eb3be5e959c524e53a4a19395bc441b4ed4718b3859fa0c608f`. `make test` runs the unit
tests (no data needed).
