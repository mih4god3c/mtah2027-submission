---
description: Record this session in docs/WORKLOG.md, docs/EXPERIMENTS.md and docs/DECISIONS.md
---

Wrap up this working session so the next teammate or agent can pick it up cold.

1. Run `git status --short`, `git log --oneline -5` and `git branch --show-current`. Review what
   this session changed and concluded.
2. **docs/WORKLOG.md**: add a new entry at the TOP, below the `---` line, using the template in
   that file. Cover the date, who and which agent, the branch and whether it is committed, the
   goal, what was done, what was learned, what is open or blocked, and the next step.
   - Keep it to about 25 lines.
   - Link to files rather than restating them.
   - Be honest about what was not verified.
3. **docs/EXPERIMENTS.md**: add one row for each proxy-league result from this session that
   informed a choice, copied exactly from the runner output.
   - Rows go in the current section's table.
   - Start a new section if the setup changed: matches, simulator version, seeds or labels.
4. **docs/DECISIONS.md**: add a numbered entry for each choice others must not silently undo,
   with the reason and the evidence. Never edit or renumber old entries; supersede them.
5. **PLAN.md**: if a roadmap item was finished or a plan changed, say so in one line in the
   WORKLOG entry. Edit PLAN.md itself only if the plan really changed.
6. Check that nothing you wrote contains data rows, frame-level values, or anything read from
   `data/`. These files are committed to a public repo.
7. Show the user the WORKLOG entry. Do not commit unless the user asks.

$ARGUMENTS
