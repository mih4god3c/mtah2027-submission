"""Every registered method and what goes into it, as a Markdown table. No data needed.

    .venv/bin/python tools/methods_overview.py            # all registered methods
    .venv/bin/python tools/methods_overview.py --only v6_deadball_model v5_frame_model
    make methods

Read from solution/methods.py itself (each method's decoder settings and `config`), so the
table cannot drift from the code. docs/SUBMISSIONS.md keeps a copy next to the result files
and the submission's lineage; regenerate it with `make methods` after registering a method.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution import decoder  # noqa: E402
from solution.methods import ALIASES, FORMER, METHODS  # noqa: E402

# (column, what it means)
COLUMNS = (
    ("decoder", "semi-Markov decoder over blocks of 5 frames: nearest player, missing ball, "
                "speed rank, duration priors (D8, D9)"),
    ("touch", "touch engine holder, solution/touch.py (E3, D10); 'hold' = touch_takes off"),
    ("kickoff", "kickoff grammar: D until the detected kickoff, solution/kickoff.py (D13, D14)"),
    ("spread", "spread cue for the team, solution/gapteam.py (E5, D16); weight and where"),
    ("frame", "frame model P(home | alive), solution/framemodel.py (C3, D22, D23); weight and where"),
    ("deadm", "learned dead-ball model in place of the speed rank, solution/deadmodel.py "
              "(D26, D37); where"),
    ("cue", "player or ball beyond a line, solution/deadball.py (C2b, D15)"),
    ("tv", "TV cut-away cue, solution/cutaway.py (E13, D27, D31); variant"),
    ("ball", "ball merged with the broadcast's, solution/sources/fused.py (D32)"),
    ("control", "touches dropped by the control model, solution/controlmodel.py (D35)"),
    ("priors", "label variant the priors are fitted on (DFL first touch unless 'keep2', D21)"),
    ("costs", "decoder transition costs or minimum stoppage length, where not the defaults (D8, D36)"),
)


def describe(fn):
    """{column: text} for one registered method, '' where a part is absent."""
    settings = getattr(fn, "settings", None)
    if settings is None:
        return None
    config = fn.config
    row = {name: "" for name, _ in COLUMNS}
    row["decoder"] = "yes"
    if settings["touch_w"] > 0:
        row["touch"] = "w{:g}".format(settings["touch_w"])
        if config["touch_params"].get("touch_takes") is False:
            row["touch"] += " hold"
    if config["kickoff_params"] is not None:
        row["kickoff"] = "yes"
    if settings["team_w"] > 0:
        row["spread"] = "w{:g} {}".format(settings["team_w"], settings["team_where"])
        if settings["team_dead"] != "fit":
            row["spread"] += " " + settings["team_dead"]
    if settings["frame_w"] > 0:
        row["frame"] = "w{:g} {}".format(settings["frame_w"], settings["frame_where"])
    if settings["deadm_w"] > 0:
        row["deadm"] = "w{:g} {}".format(settings["deadm_w"], settings["deadm_where"])
        if config["deadm_version"] != "v1":
            row["deadm"] += " " + config["deadm_version"]
    if settings["deadball_w"] > 0:
        row["cue"] = "w{:g} {}".format(settings["deadball_w"],
                                       "+".join(config["deadball_params"].get("cues", ("player",))))
    if settings.get("tv_w", 0.0) > 0:
        row["tv"] = "w{:g} {}".format(settings["tv_w"], config["tv_variant"])
    if config["merge_ball"]:
        row["ball"] = "merged"
    if config["control"]:
        row["control"] = "yes" if config["control"] is True else str(config["control"])
    if config["prior_labels"]:
        row["priors"] = config["prior_labels"]
    costs = []
    for key in ("switch", "dead", "restart", "no_ball"):
        if settings[key] != decoder.DEFAULTS[key]:
            costs.append("{} {:g}".format(key, settings[key]))
    if settings.get("dead_min_s") is not None:
        costs.append("min stop {:g} s".format(settings["dead_min_s"]))
    row["costs"] = ", ".join(costs)
    if settings["bonus_w"] > 0:
        row["costs"] = (row["costs"] + "; " if row["costs"] else "") + "bonus w{:g}".format(
            settings["bonus_w"])
    return row


def table(names):
    lines = ["| method | " + " | ".join(name for name, _ in COLUMNS) + " |",
             "|---" * (len(COLUMNS) + 1) + "|"]
    for name in names:
        row = describe(METHODS[name])
        if name in FORMER:
            label = "`{}` (was `{}`)".format(name, FORMER[name])
        else:
            label = "`{}`".format(name)
        if row is None:
            if name.startswith("baseline_"):
                text = "nearest player to the ball, majority vote over +-{} frames".format(
                    name.replace("baseline_smooth", "").replace("baseline_raw", "0"))
            else:
                text = "no decoder"
            lines.append("| {} | {} |".format(label, text) + " |" * (len(COLUMNS) - 1))
            continue
        lines.append("| {} | ".format(label) + " | ".join(row[c] for c, _ in COLUMNS) + " |")
    return lines


def legend():
    return ["- **{}**: {}".format(name, text) for name, text in COLUMNS]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="*", default=None, metavar="NAME",
                        help="methods to list (default: every registered one)")
    parser.add_argument("--no-legend", action="store_true")
    args = parser.parse_args(argv)
    names = ([ALIASES.get(n, n) for n in args.only] if args.only
             else [n for n in METHODS if n not in ALIASES])
    unknown = [n for n in names if n not in METHODS]
    if unknown:
        raise SystemExit("unknown methods: {}".format(unknown))
    print("\n".join(table(names)))
    if not args.no_legend:
        print()
        print("\n".join(legend()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
