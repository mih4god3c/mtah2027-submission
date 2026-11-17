# Team shortcuts. Every target is a thin wrapper over a command documented in TEAM.md.
PY   := .venv/bin/python
TEAM ?= our_team
METHODS ?= baseline_smooth50
SIM ?= v1
TRAIN_SIM ?= v1
NOISE ?= 0
SPEED ?= 1
TEMPO ?= 1
BALL_NOISE ?= 0
BALL_JUMPS ?= 0
BALL_SMOOTH ?= 0
BALL_VISIBLE ?= 0
HIDE_KEEPER ?= 1
LOSE_NEAR_GOAL ?= 1
PREKICKOFF ?= 0
PREKICKOFF_PROFILE ?= calm
BEYOND_KEEP ?= 1
BEYOND_MARGIN ?= 0
BOUNDARIES ?=
LABELS ?= keep2
PRIOR_LABELS ?= dfl

.PHONY: setup lock eval bootstrap baseline submission diff frame-report moment clip segments flickers voters restart-check methods stability validate notebook external proxy sim-check track-loss check-loader broadcast-align whistle-broadcast whistle-selfcal halfend-whistle whistle-veto test

setup:              ## venv + pinned deps + git guards + Jupyter kernel (run once per clone)
	uv venv --allow-existing --python 3.11 .venv
	uv pip install --python $(PY) -r requirements-lock.txt
	git config core.hooksPath .githooks
	$(PY) -m nbstripout --install --attributes .gitattributes
	$(PY) -m ipykernel install --user --name mtah2027 --display-name "MTAH 2027 (.venv)"

lock:               ## re-pin after adding a dependency: uv pip install --python .venv/bin/python <pkg>
	{ echo "# Pinned environment for reproducing the submission (Python 3.11). Regenerate with: make lock"; \
	  uv pip freeze --python $(PY); } > requirements-lock.txt

eval:               ## score everything in evaluation/results/ against the public 5 minutes
	$(PY) evaluation/run_evaluation.py

bootstrap:          ## same, with confidence intervals -- use before believing any gap
	$(PY) evaluation/run_evaluation.py --bootstrap 1000

baseline:           ## regenerate the four reference baselines (needs data/)
	$(PY) starter/baseline.py

submission:         ## build + validate evaluation/results/$(TEAM).csv (needs data/) [METHOD=<registered method> writes a comparison file, not the submission]
	$(PY) solution/run.py --team $(TEAM) $(if $(METHOD),--method $(METHOD))

diff:               ## where two result files disagree, as video times: make diff A=our_team B=v5_plus_tv_cue_merged_ball [ARGS="--min-s 3 --limit 40"]
	$(PY) tools/diff_submissions.py evaluation/results/$(A).csv evaluation/results/$(B).csv $(ARGS)

frame-report:       ## frame-by-frame view of result files on the public window (description only): make frame-report FILES="our_team v6_deadball_model" [ARGS="--top 12 --res 0.5"]
	$(PY) tools/frame_report.py $(foreach f,$(FILES),evaluation/results/$(f).csv) $(ARGS)

moment:             ## (needs data/) what the decoder saw at one moment, per second, for a person: make moment FROM=11:50 TO=12:20 [METHOD=v6_deadball_model] [METHODS="touch_c3_seen_w1_dms touch_c3_seen_w1_tv_cal"]
	$(PY) tools/moment_check.py --from $(FROM) --to $(TO) $(if $(METHOD),--method $(METHOD)) $(if $(METHODS),--methods $(METHODS))

clip:               ## (needs data/) frame-accurate clip of a video to watch, data/clips/: make clip [FROM=18685 TO=26184] [SOURCE=tactical] [ARGS="--stamp"] (default: the 5 min after the public truth)
	$(PY) tools/clip_video.py $(if $(FROM),--from $(FROM)) $(if $(TO),--to $(TO)) $(if $(SOURCE),--source $(SOURCE)) $(ARGS)

segments:           ## a result file as possession segments with start/end times (writes evaluation/reports/<FILE>_segments.csv): make segments [FILE=our_team] [ARGS="--all"]
	$(PY) tools/segments.py evaluation/results/$(or $(FILE),our_team).csv $(ARGS)

flickers:           ## how many possessions last under 20/30/40/50 frames in a result file: make flickers [FILE=our_team] [ARGS="--all --thresholds 10 25"]
	$(PY) tools/flickers.py evaluation/results/$(or $(FILE),our_team).csv $(ARGS)

label-share:        ## share of H/A/D/X frames in a result file, per half and for the match: make label-share [FILE=our_team]
	$(PY) tools/label_share.py evaluation/results/$(or $(FILE),our_team).csv

keeper-check:       ## keeper's ball vs goal kick: ball gaps after keeper sightings, keeper model in its zone (ARGS=--bar-ath needs data/)
	$(PY) tools/keeper_check.py $(ARGS)

voters:             ## each part of v7 on its own, scored per frame (proxy by default; ARGS="--bar-ath --frames-csv" needs data/): make voters [ARGS=...]
	$(PY) tools/voters.py $(ARGS)

restart-check:      ## dead ball missed by restart type (goal kick, throw-in, ...) on the IDSSE dev matches, no data/ needed: make restart-check [ARGS="--method v6_deadball_model"]
	$(PY) tools/restart_check.py $(ARGS)

methods:            ## every registered method and what goes into it, as a Markdown table (docs/SUBMISSIONS.md keeps a copy)
	$(PY) tools/methods_overview.py $(ARGS)

stability:          ## L3 stability check (D36) on the proxy: make stability [METHOD=v6_deadball_model] [ARGS="--bar-ath"] (--bar-ath needs data/)
	$(PY) tools/stability_check.py $(if $(METHOD),--method $(METHOD)) $(ARGS)

validate:           ## make validate FILE=path/to.csv
	$(PY) evaluation/validate_submission.py $(FILE)

notebook:
	$(PY) -m notebook starter/

external:           ## download the external validation matches (IDSSE, ~2.5 GB), pinned + checksummed
	$(PY) tools/download_external.py

proxy:              ## score methods on the external matches: make proxy METHODS="baseline_smooth50 new_idea" [SIM=v0] [TRAIN_SIM=v2] [NOISE=0.1] [SPEED=0.9] [TEMPO=0.9] [BALL_NOISE=0.05 BALL_JUMPS=3.4 BALL_SMOOTH=0.26] [BALL_VISIBLE=0.9] [CLEAN_PLAYERS=1] [PREKICKOFF=117 PREKICKOFF_PROFILE=busy] [BEYOND_KEEP=0.82 BEYOND_MARGIN=1] [HIDE_KEEPER=0.5] [LOSE_NEAR_GOAL=0.5] [BOUNDARIES=1] [LABELS=dfl] [PRIOR_LABELS=keep2]
	$(PY) tools/proxy_league.py --methods $(METHODS) --simulator $(SIM) --train-simulator $(TRAIN_SIM) --position-noise $(NOISE) --speed-factor $(SPEED) --tempo $(TEMPO) --ball-noise $(BALL_NOISE) --ball-jumps $(BALL_JUMPS) --ball-smooth $(BALL_SMOOTH) --ball-visible $(BALL_VISIBLE) --prekickoff $(PREKICKOFF) --prekickoff-profile $(PREKICKOFF_PROFILE) --beyond-keep $(BEYOND_KEEP) --beyond-margin $(BEYOND_MARGIN) --hide-keeper $(HIDE_KEEPER) --lose-near-goal $(LOSE_NEAR_GOAL) --labels $(LABELS) --prior-labels $(PRIOR_LABELS) $(if $(BOUNDARIES),--boundaries) $(if $(CLEAN_PLAYERS),--clean-players)

sim-check:          ## does the simulator match the BAR-ATH ATD aggregates? (v0 and v1 side by side)
	$(PY) tools/simulator_check.py

track-loss:         ## where player tracks end (crowding, ball, lines, speed): clean IDSSE vs simulator; ARGS="--bar-ath" needs data/
	$(PY) tools/track_loss_check.py $(ARGS)

whistle-broadcast:  ## (needs data/, then make broadcast-align) whistle Step 0 on both audio tracks (D44): make whistle-broadcast [ARGS="--spot 10"]
	$(PY) tools/whistle_broadcast_check.py $(ARGS)

halfend-whistle:    ## D14 checks for v7_end, the end-of-half rule alone (D48): make halfend-whistle [ARGS="--bar-ath"]
	$(PY) tools/halfend_whistle_check.py $(ARGS)

whistle-selfcal:    ## (needs data/) Step 0 of the self-calibrating whistle detector, aggregates only (D47): make whistle-selfcal [ARGS="--cross"]
	$(PY) tools/whistle_selfcal_check.py $(ARGS)

whistle-veto:       ## Step 0 of the whistle veto on the proxy (no data/): make whistle-veto [ARGS="--labels dfl"]
	$(PY) tools/whistle_veto_check.py $(ARGS)

check-loader:       ## (needs data/) check the BAR-ATH loader reproduces the published baselines
	$(PY) tools/check_bar_ath_loader.py

broadcast-align:    ## (needs data/) E13 Step 0: align the broadcast ATD to tactical, print aggregates [METHOD=none]
	$(PY) tools/broadcast_align.py $(if $(METHOD),--method $(METHOD))

test:               ## unit tests (no data needed): proxy league, simulator, loaders
	$(PY) -m unittest discover -s tests
