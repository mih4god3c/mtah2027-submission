# Team workflow

Our working repo is **private**: `origin` = github.com/mih4god3c/mtah2027, so other teams cannot
see our plan or results while we work (decision D6 in [docs/DECISIONS.md](docs/DECISIONS.md)).
`upstream` = metrica-sports, for organiser fixes. At submission we make a public copy (see the
checklist below). **Deadline: 17 Nov 2026.**

## One-time setup (each of us)

```bash
brew install uv                      # or: curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/mih4god3c/mtah2027.git        # private: accept the invite first
cd mtah2027
git remote add upstream https://github.com/metrica-sports/more-than-a-hack-2027.git
make setup                           # Python 3.11 venv, pinned deps, git guards, Jupyter kernel
make eval                            # should print the 4-baseline scoreboard
make external                        # IDSSE validation matches, ~2.5 GB, into external_data/
```

Already have a clone of the old public fork (DenisNeds/more-than-a-hack-2027)? Repoint it instead:
`git remote set-url origin https://github.com/mih4god3c/mtah2027.git && git fetch origin`.

Then unpack the data bundle (download link from enrolment email) into `data/` — see
[data/README.md](data/README.md). Each of us needs our own copy; **never commit or share it via
the repo**. In VS Code pick the `.venv` interpreter / "MTAH 2027 (.venv)" kernel.

## Daily loop

```bash
git checkout main && git pull
git checkout -b <name>/<idea>        # e.g. miha/dead-ball-from-tagging
# ... work ...
make submission TEAM=<name>_<idea>   # writes + validates evaluation/results/<name>_<idea>.csv
make eval                            # compare against baselines on the public 5 min
git push -u origin HEAD              # open a PR into main, one teammate reviews
```

- Small PRs, merged often. `main` must always run `make submission` cleanly.
- Exploration goes in notebooks under `notebooks/<name>/`; anything we keep moves into
  `solution/` as a function. Notebook outputs are stripped on commit automatically.
- Added a package? `uv pip install --python .venv/bin/python <pkg>` then `make lock` and commit
  `requirements-lock.txt` in the same PR.
- Organiser updates: `git fetch upstream && git merge upstream/main`.

## Deciding whether an idea helps: the proxy league

The public 5 minutes cannot tell two versions apart (17 possessions) and may not be tuned on.
We decide on **other matches with published possession labels** instead, degraded so the ball
goes missing the way it does in our ATD, and scored by the official `possession_metrics` code.

```bash
make external                                   # once: IDSSE, 7 Bundesliga matches, ~2.5 GB, checksummed
make proxy METHODS="baseline_smooth50 my_idea"  # first method is the reference, ~10 s
make sim-check                                  # simulator v0/v1 against the BAR-ATH aggregates
make test                                       # unit tests, no data needed
make check-loader                               # needs data/: our loader == published baseline
```

- New methods go in `solution/methods.py` (`Match -> scorer codes`) and run on BAR-ATH and the
  external matches unchanged. Shared format: `solution/match.py`.
- **Adopt a change only if** the pooled gain is at least 0.02, its paired interval excludes 0,
  and it wins on at least two thirds of the matches. The runner prints the verdict.
- The default is the **5 development matches**. Two (`J03WN1`, `J03WR9`) are **held out until
  Phase 4**: `--split holdout` prints a warning, and nothing may be chosen on them.
- Inputs are degraded by **simulator v1** (`SIM=v0` for the older one; numbers from different
  versions are not comparable). v1 hides the ball mostly in slow play and makes player tracks
  messy: NaN stretches, and extra same-team "ghost" slots, so a degraded match has more columns
  than players. Methods must not assume 22 clean slots.
- Known gaps between this exam and the real one are listed in `solution/simulate.py`.

## Keeping a record (for teammates and for coding agents)

Three files under `docs/`, committed, so the next person or agent starts from what was done:

| file | what goes in | when |
|---|---|---|
| [docs/WORKLOG.md](docs/WORKLOG.md) | one entry per working session: what changed, results, open issues, next step | end of every session |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | one row per proxy-league result worth keeping, with the verdict | whenever a run informs a decision |
| [docs/DECISIONS.md](docs/DECISIONS.md) | numbered decisions with the reason and evidence | when we choose something others must not undo |

In Claude Code, `/wrap-up` drafts all three from the session. `CLAUDE.md` tells every agent to
read the latest entries first. Never paste data rows into these files: they are public.

## Guard rails (these can disqualify us)

| | |
|---|---|
| **No data in git** | `data/` is gitignored and a pre-commit hook blocks data files, submissions and anything > 5 MB. Don't paste data rows into issues/PRs either — the repo is public. |
| **No data to hosted LLMs** | Terms forbid uploading videos/data/extracts to ChatGPT, Claude, Gemini etc. Fine to ask them about code and the documented format; not fine to paste rows, plots or frames. `.claude/settings.json` blocks Claude Code from reading `data/`. |
| **No hand labels** | Watching the video to understand it is fine; turning what we saw into labels, corrections or frame-specific constants is not. |
| **Don't tune on the public 5 min** | 17 possessions — it picks the wrong window even for the baseline. Use it to check correctness; decide with `make proxy`. |
| **Reproducible** | `python solution/run.py` must regenerate the exact submitted file. Seed everything. |

## Who does what

| | owner | notes |
|---|---|---|
| Possession / team-in-control model | | |
| Dead-ball detection (Smart Tagging set pieces, ball out of bounds) | | |
| Ball-gap handling & smoothing / state model | | |
| Technical report ([docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md)) | | |
| Final submission: CSV + repo URL + report via Google Form | | one response per team |

## Submission checklist

- [ ] `make submission TEAM=<team_name>` on a clean clone of `main` reproduces the file
- [ ] `make validate FILE=evaluation/results/<team_name>.csv` passes
- [ ] report exported as `<team_name>.pdf` (or `.md`)
- [ ] the submitted repo is public and contains no data (`git ls-files data/` shows only the
      README). Either make `mih4god3c/mtah2027` public or push `main` to a public repo; check
      that nothing in `docs/` or the history contains data rows before doing either
- [ ] uploaded through the form before 17 Nov 2026
