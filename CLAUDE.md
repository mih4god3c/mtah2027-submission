# Project notes for Claude Code

Hackathon repo (More than a Hack 2027, Phase 1): label every tactical-video frame of one match
`H`/`A`/`D`/`X`. Ranked on `spellF1`, tie-break `macroF1`. Docs: README → CHALLENGE → DATA →
EVALUATION → SUBMISSION → RULES. `evaluation/possession_metrics.py` is the metric spec.

## Hard rule: never read the match data

The terms forbid sending the videos, datasets or any extract of them to a hosted LLM — that
includes you. Do not open, print, sample, `head`, plot-and-view or summarise anything under
`data/` (or `$MTAH_DATA_DIR`), and do not read notebook outputs or logs that echo data rows.
Work from the documented formats in DATA.md and the loader code instead. When code needs
running against the data, ask the human to run it and report only aggregate numbers they choose
to share. `.claude/settings.json` denies the obvious reads; respect the intent beyond them.

The public ground truth `evaluation/gt/BAR-ATH_possession_gt_public.csv` and the committed
`baseline_*.csv` files are in the public repo and are fine to read.

## Other rules that shape code

- Labels must come from an algorithm. No hand labels, no frame-specific constants for this game.
- Do not tune on the public 5 minutes; it has 17 possessions and picks the wrong answer.
- The submission must be reproducible byte-for-byte from `solution/run.py`: seed randomness,
  pin deps (`make lock` after adding any).

## Session routine (every agent, every session)

1. **Start:** read the top two or three entries of `docs/WORKLOG.md`, then `docs/DECISIONS.md`.
   `PLAN.md` is the overall plan; do not re-litigate a numbered decision without saying so.
2. **Decide with the proxy league,** never the public window: `make proxy METHODS="ref new"`.
   Its truth is controlled possession (`keep2`, D21); also check `LABELS=dfl` is not WORSE.
   Adopt only on an ADOPT verdict (D4). Never pass `--split holdout` except in Phase 4 (D3).
3. **End:** run `/wrap-up` or do what it says. Add a WORKLOG entry, add EXPERIMENTS rows for
   results that informed a choice, and add a DECISIONS entry for anything others must not undo.
   These files are public: aggregates and code paths only, never data rows.

## Layout

- `solution/run.py` — our entry point; `predict()` is where the method goes.
  `docs/SUBMISSIONS.md` says which result file came from which method (`make methods`).
- `starter/` — organiser loaders (`load_atd.py`, `load_smart_tagging.py`, `load_homography.py`)
  and `baseline.py`. Prefer importing from these over re-parsing files.
- `games.py` — frame alignment. ATD `frame_id` = video frame + 1; Smart Tagging seconds × 25.
- `solution/match.py` — the shared `Match` format; `solution/sources/` loads BAR-ATH (`atd.py`)
  and external matches (`idsse.py`); `solution/methods.py` holds methods (`Match -> codes`);
  `solution/touch.py` is the touch/control engine (E3) the `v2_touch` method reads;
  `solution/simulate.py` degrades ball and players like our ATD (v1; v0 kept) and lists the
  known gaps of the proxy league. `solution/relabel.py` turns DFL's first-touch labels into
  controlled possession (`keep2`, D21); `make proxy LABELS=keep2 PRIOR_LABELS=dfl` scores on it,
  `tools/label_check.py` describes the label sets.
- `tools/proxy_league.py` — the decision tool (`BOUNDARIES=1` adds `tools/boundaries.py`'s
  report; `BALL_NOISE`/`BALL_JUMPS`/`BALL_SMOOTH` stress the ball, D12); `tools/simulator_check.py` (`make sim-check`)
  checks the simulator against the BAR-ATH aggregates. `tests/` — `make test`, no data needed.
- External data is in `external_data/` and caches are in `cache/`, both gitignored. Agents may
  read them; they hold public data about other games, not the competition match.
- Run things with `.venv/bin/python`; shortcuts in the `Makefile`.
