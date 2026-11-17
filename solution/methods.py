"""Methods the proxy league can score: each takes a Match and returns scorer codes, length T.

Every method must work on BAR-ATH and on external matches alike, which is why they take a Match
rather than file paths. Register new ones in METHODS; `tools/proxy_league.py --methods name`
then scores them.
"""

import os
import sys

import numpy as np

from solution import (deadball, deadmodel, decoder, framemodel, gapteam, handover, kickoff,
                      touch)
from solution.match import AWAY, HOME, ROOT, pm

sys.path.insert(0, os.path.join(ROOT, "starter"))
import baseline  # noqa: E402  (smooth: the published majority vote)


def nearest_player_codes(match):
    """Team of the tracked player nearest the ball; -1 where there is no ball or no player.

    The same rule as starter/baseline.py's nearest_player_labels, vectorised over a Match, so
    it runs on external matches too. Distances are in metres on both axes.
    """
    dx = match.player_x - match.ball_x[:, None]
    dy = match.player_y - match.ball_y[:, None]
    distance = dx * dx + dy * dy
    distance = np.where(np.isfinite(distance), distance, np.inf)
    nearest = np.argmin(distance, axis=1)
    found = np.isfinite(distance[np.arange(match.n_frames), nearest])
    return np.where(found, match.team[nearest], -1).astype(np.int8)


def to_scorer_codes(letters, match):
    """'H'/'A'/'D' letters -> scorer codes, with X outside the halves."""
    codes = pm.label_codes(letters)
    codes[~match.in_play_mask()] = pm.X
    return codes


def baseline_window(window):
    """starter/baseline.py's algorithm: nearest player, then a centred majority vote."""
    def predict(match):
        raw = nearest_player_codes(match)
        assert HOME == baseline.HOME and AWAY == baseline.AWAY
        return to_scorer_codes(baseline.smooth(raw, window), match)
    predict.__name__ = "baseline_smooth{}".format(window) if window else "baseline_raw"
    return predict


def dadvance_events(match):
    """solution/handover.py's dadvance0.5 events: all players' run direction changing (D18)."""
    return handover.events(match, "dadvance0.5")


def semi_markov(name="v1_decoder", touch_params=None, snap=None, kickoff_params=None,
                deadball_params=None, team_params=None, bonus_fn=dadvance_events,
                prior_labels=None, frame_params=None, frame_fn=None, deadm_version="v1",
                deadm_fn=None, tv_variant="pessimistic", merge_ball=False, control=False,
                nearest=True, keeper_params=None, whistle_variant="prior",
                whistle_end=False, whistle_end_guard=False, **params):
    """solution/decoder.py over the baseline's evidence plus player speed (E1).

    Priors come from the IDSSE dev matches other than this one (leave-one-match-out); on
    BAR-ATH from all five. `params` override decoder.DEFAULTS. With touch_w > 0 the decoder
    also reads solution/touch.py's holder (E3); `touch_params` override touch.PARAMS. `snap`,
    a dict of touch.snap_transitions keywords, moves decoded changes onto touches (E6).
    `kickoff_params`, a dict overriding kickoff.PARAMS ({} for the defaults), turns on the
    kickoff grammar: each half is D until its detected kickoff and decoded from there on.
    `deadball_params` override deadball.PARAMS for the cue that deadball_w > 0 reads (C2b).
    `team_params` override gapteam.PARAMS for the team cues that team_w > 0 reads (E5).
    With bonus_w > 0, `bonus_fn(match)` gives (frames, teams) of hand-over events, and an
    H <-> A switch into a team near one of its events gains bonus_w (decoder.switch_bonus).
    `prior_labels` fits every prior on that IDSSE label variant (solution/relabel.py, D20)
    instead of the proxy league's current one (idsse.LABELS). With frame_w > 0 the decoder
    reads solution/framemodel.py's per-frame P(home | alive) (C3); `frame_params` override
    framemodel.PARAMS. `frame_fn(match)`, if given, supplies those per-frame log-odds instead
    (C6, solution/selftrain.py). With deadm_w > 0 the decoder reads solution/deadmodel.py's
    per-frame log P(x | dead) / P(x | alive), model `deadm_version`, in place of the speed
    likelihood; `deadm_fn(match)`, if given, supplies it instead (oracles, section 16).
    With tv_w > 0 the decoder reads the TV cut-away cue (solution/cutaway.cue_for: attached,
    BAR-ATH's real one, or the proxy imitation of variant `tv_variant`), through a table fitted
    on synthetic cues of that variant on the other IDSSE dev matches (E13, D31). With
    `merge_ball` the input's ball is first merged with the broadcast's (solution/sources/
    fused.merged: the real one on BAR-ATH, its imitation on a degraded proxy match; D32).
    With `control` (True, or a dict overriding controlmodel.PARAMS) the holder is re-swept
    without the touches solution/controlmodel.py scores as contradicted (#3, EXPERIMENTS
    section 21); priors are unchanged. With `nearest` False the decoder gets no nearest player
    on any frame (an ablation; section 31). With keeper_w > 0 the decoder reads
    solution/keepermodel.py's per-state log-likelihoods in the keeper zone; `keeper_params`
    override keepermodel.PARAMS. With whistle_w > 0 the decoder reads solution/whistlecue.py's
    per-state log-likelihoods (the referee's whistle: BAR-ATH's from the tactical audio, the
    proxy's imitated, variant `whistle_variant`); with `whistle_end` the decoded states then get
    whistlecue.end_of_half on the same whistles, with its tracking guard when
    `whistle_end_guard` (D48).
    """
    unknown = set(params) - set(decoder.DEFAULTS)
    unknown |= set(touch_params or {}) - set(touch.PARAMS)
    unknown |= set(kickoff_params or {}) - set(kickoff.PARAMS)
    unknown |= set(deadball_params or {}) - set(deadball.PARAMS)
    unknown |= set(team_params or {}) - set(gapteam.PARAMS)
    unknown |= set(frame_params or {}) - set(framemodel.PARAMS)
    if unknown:
        raise ValueError("unknown decoder or touch parameters: {}".format(sorted(unknown)))
    settings = dict(decoder.DEFAULTS, **params)
    touch_items = tuple(sorted((touch_params or {}).items()))
    touch_settings = dict(touch.PARAMS, **dict(touch_items))
    cue_items = tuple(sorted((deadball_params or {}).items()))
    team_items = tuple(sorted((team_params or {}).items()))
    frame_settings = dict(framemodel.PARAMS, **(frame_params or {}))

    def prepare(match):
        """Everything the decoder reads for this match: priors and per-frame inputs."""
        if merge_ball:
            from solution.sources import fused
            match = fused.merged(match)
        lag, scale = settings["speed_lag"], settings["speed_scale"]
        lead = settings["speed_lead_s"]
        priors = decoder.idsse_priors(match.name, lag, scale, lead, prior_labels)
        holder = touches = None
        if settings["touch_w"] > 0 or snap:
            touches = touch.detect(match, touch_settings)
        if settings["touch_w"] > 0:
            priors = dict(priors, holder_agree=touch.idsse_holder_agree(match.name, touch_items,
                                                                          prior_labels))
            holder = touches.holder
            if control:
                from solution import controlmodel
                control_settings = dict(controlmodel.PARAMS,
                                        **(control if isinstance(control, dict) else {}))
                holder = touch.sweep_holder(match.n_frames, touches.touch_frame,
                                            touches.touch_team, touches.control, touches.usable,
                                            match.halves, match.fps, touch_settings,
                                            controlmodel.keep_mask(match, touches,
                                                                   control_settings))
        cue = None
        if settings["deadball_w"] > 0:
            priors = dict(priors, cue_log=deadball.idsse_log(match.name, cue_items,
                                                             prior_labels))
            cue = deadball.cues(match, dict(deadball.PARAMS, **dict(cue_items)))
        team = None
        if settings["team_w"] > 0:
            team_settings = dict(gapteam.PARAMS, **dict(team_items))
            priors = dict(priors, team_log=gapteam.idsse_log(match.name, team_items,
                                                             prior_labels),
                          team_names=team_settings["cues"])
            team = gapteam.cues(match, team_settings)
        frame = None
        if settings["frame_w"] > 0:
            frame = (framemodel.log_odds(match, frame_settings) if frame_fn is None
                     else frame_fn(match))
        deadm = None
        if settings["deadm_w"] > 0:
            deadm = (deadmodel.log_ratio(match, version=deadm_version) if deadm_fn is None
                     else deadm_fn(match))
        keeper_ll = None
        if settings["keeper_w"] > 0:
            from solution import keepermodel
            keeper_ll = keepermodel.log_lik(match, dict(keepermodel.PARAMS,
                                                        **(keeper_params or {})))
        whistle_ll = whistle_onsets = None
        if settings["whistle_w"] > 0 or whistle_end:
            from solution import whistlecue
            whistle_onsets = whistlecue.onsets_for(match, whistle_variant)
        if settings["whistle_w"] > 0:
            whistle_ll = whistlecue.log_lik(match, whistle_variant)
        tv = None
        if settings["tv_w"] > 0:
            from solution import cutaway
            tv = cutaway.cue_for(match, tv_variant)
            priors = dict(priors, tv_log=cutaway.idsse_table(match.name, tv_variant))
        bonus_frames = bonus_teams = None
        if settings["bonus_w"] > 0:
            bonus_frames, bonus_teams = bonus_fn(match)
        kicks = [None] * len(match.halves)
        if kickoff_params is not None:
            kicks = kickoff.detect(match, dict(kickoff.PARAMS, **kickoff_params))
        spans = []
        for (start, stop), kick in zip(match.halves, kicks):
            if kick is not None:
                # D before the kickoff. Decode from the block that holds it, on the half's own
                # block grid, so the blocks after it are the ones the decoder saw without the rule.
                start += (kick - start) // decoder.BLOCK * decoder.BLOCK
            spans.append((start, stop))
        return {"priors": priors, "touches": touches, "holder": holder, "cue": cue,
                "team": team, "frame": frame, "deadm": deadm, "tv": tv, "keeper": keeper_ll,
                "whistle": whistle_ll, "whistle_onsets": whistle_onsets,
                "bonus": (bonus_frames, bonus_teams),
                "nearest": (nearest_player_codes(match) if nearest
                            else np.full(match.n_frames, -1, dtype=np.int8)),
                "speed": decoder.speed_feature(match, lag, scale, lead),
                "seen": match.ball_seen, "spans": spans}

    def half_inputs(inp, start, stop):
        """decoder.decode_half's per-frame arguments for frames [start, stop)."""
        def cut(values):
            return None if values is None else values[start:stop]
        return (cut(inp["nearest"]), cut(inp["seen"]), cut(inp["speed"]), inp["priors"],
                cut(inp["holder"]), cut(inp["cue"]), cut(inp["team"]), cut(inp["frame"]),
                cut(inp.get("deadm")))

    def decode(match, inp, overrides=None):
        """Decoder states 0/1/2 per frame from prepare()'s inputs; `overrides` change decoder
        settings (for ablations: touch_w, frame_w, team_w at 0)."""
        params = dict(settings, **(overrides or {}))
        states = np.full(match.n_frames, decoder.D, dtype=np.int8)
        bonus_frames, bonus_teams = inp["bonus"]
        for start, stop in inp["spans"]:
            events = None
            if bonus_frames is not None:
                pick = (bonus_frames >= start) & (bonus_frames < stop)
                events = (bonus_frames[pick] - start, bonus_teams[pick])
            (nearest, seen, speed, priors, holder, cue, team, frame,
             deadm) = half_inputs(inp, start, stop)
            tv = None if inp.get("tv") is None else inp["tv"][start:stop]
            kp = None if inp.get("keeper") is None else inp["keeper"][start:stop]
            wl = None if inp.get("whistle") is None else inp["whistle"][start:stop]
            states[start:stop] = decoder.decode_half(nearest, seen, speed, priors, params,
                                                     match.fps, holder, cue, team, events, frame,
                                                     deadm, tv, kp, wl)
        if snap:
            states = touch.snap_transitions(states, inp["touches"], match.halves, match.fps,
                                            **snap)
        if whistle_end:
            from solution import whistlecue
            states = whistlecue.end_of_half(states, match, inp["whistle_onsets"], decoder.D,
                                            guard=whistle_end_guard)
        return states

    def sample(match, inp, n_samples, seed=0, temperature=1.0, overrides=None):
        """(n_samples, T) decoder states drawn from the decoder's posterior (#6, MBR Step 0):
        decode() with forward filtering, backward sampling in place of Viterbi. Frames outside
        the decoded spans are D, as in decode(); `snap` is applied to every sample."""
        params = dict(settings, **(overrides or {}))
        rng = np.random.RandomState(seed)
        states = np.full((n_samples, match.n_frames), decoder.D, dtype=np.int8)
        bonus_frames, bonus_teams = inp["bonus"]
        for start, stop in inp["spans"]:
            events = None
            if bonus_frames is not None:
                pick = (bonus_frames >= start) & (bonus_frames < stop)
                events = (bonus_frames[pick] - start, bonus_teams[pick])
            (nearest, seen, speed, priors, holder, cue, team, frame,
             deadm) = half_inputs(inp, start, stop)
            tv = None if inp.get("tv") is None else inp["tv"][start:stop]
            kp = None if inp.get("keeper") is None else inp["keeper"][start:stop]
            wl = None if inp.get("whistle") is None else inp["whistle"][start:stop]
            model = decoder.half_model(nearest, seen, speed, priors, params, match.fps, holder,
                                       cue, team, events, frame, deadm, tv, kp, wl)
            states[:, start:stop] = decoder.sample_half(model, stop - start, n_samples, rng,
                                                        temperature)
        if snap:
            for i in range(n_samples):
                states[i] = touch.snap_transitions(states[i], inp["touches"], match.halves,
                                                   match.fps, **snap)
        return states

    def predict(match):
        states = decode(match, prepare(match))
        return to_scorer_codes(np.array(["H", "A", "D"])[states], match)
    predict.__name__ = name
    # For diagnostics (tools/gap_check.py --seen-ceiling, tools/mbr_check.py): the inputs, a
    # re-decode with other settings, posterior samples, and the settings themselves.
    predict.prepare, predict.decode, predict.half_inputs = prepare, decode, half_inputs
    predict.sample = sample
    predict.settings = settings
    # The non-decoder choices, for tools/methods_overview.py: what else the method is made of.
    predict.config = {"touch_params": dict(touch_items), "snap": snap or {},
                      "kickoff_params": kickoff_params, "deadball_params": dict(cue_items),
                      "team_params": dict(team_items), "prior_labels": prior_labels,
                      "frame_params": dict(frame_params or {}), "deadm_version": deadm_version,
                      "tv_variant": tv_variant, "merge_ball": merge_ball, "control": control,
                      "whistle_variant": whistle_variant, "whistle_end": whistle_end,
                      "whistle_end_guard": whistle_end_guard,
                      "bonus_fn": getattr(bonus_fn, "__name__", None)}
    return predict


def oracle(match):
    """The truth itself. A sanity row: it must score exactly 1.0."""
    return match.labels.copy()


METHODS = {
    "oracle": oracle,
    "baseline_raw": baseline_window(0),
    "baseline_smooth25": baseline_window(25),
    "baseline_smooth50": baseline_window(50),
    "baseline_smooth125": baseline_window(125),
    # E1. Costs tried on the proxy league did not beat all-zero costs beyond noise, so the
    # defaults stand: w = 1, costs 0, speed as a within-match rank (D8, D9).
    "v1_decoder": semi_markov(),
    # E3: the decoder also reads the touch engine's holder (solution/touch.py), weight 1, the
    # engine at its physical defaults. 15 settings tried; EXPERIMENTS section 5, D10.
    "v2_touch": semi_markov("v2_touch", touch_w=1.0),
    # PLAN finding 9 and C4's grammar: v2_touch, D until each half's detected kickoff
    # (solution/kickoff.py). The proxy league needs PREKICKOFF to see it.
    "v3_kickoff": semi_markov("v3_kickoff", touch_w=1.0, kickoff_params={}),
    # C2b: v3_kickoff plus solution/deadball.py's cues, a player or the ball beyond a line,
    # at weight 0.5. Registered, not submitted: +0.008 under the D12 stress, below D4's bar
    # (EXPERIMENTS section 8). tools/deadball_check.py --bar-ath compares it on BAR-ATH.
    "touch_deadball": semi_markov("touch_deadball", touch_w=1.0, kickoff_params={},
                                  deadball_w=0.5, deadball_params={"cues": ("player", "ball")}),
    # E5: v3_kickoff plus solution/gapteam.py's spread cue (home's outfield width minus
    # away's), read on every frame, weight 0.5, with no say on dead ball ("max"). 17 settings
    # tried; weight 1 gains about as much, 1.5 and 2 merge real possessions (EXPERIMENTS 9).
    # Submitted (D16) after the BAR-ATH cue check.
    "v4_spread": semi_markov("v4_spread", touch_w=1.0, kickoff_params={},
                                 team_w=0.5, team_where="all", team_dead="max",
                                 team_params={"cues": ("spread",)}),
    # D20 re-tests of D10's label-semantics choice: the touch engine without touch_takes, so an
    # opposing touch only ends the hold and the other team takes over at its control episode.
    # Registered for the controlled-possession proxy (LABELS=keep2), not submitted.
    "touch_markov_hold": semi_markov("touch_markov_hold", touch_w=1.0,
                                     touch_params={"touch_takes": False}),
    "touch_gapteam_hold": semi_markov("touch_gapteam_hold", touch_w=1.0, kickoff_params={},
                                      team_w=0.5, team_where="all", team_dead="max",
                                      team_params={"cues": ("spread",)},
                                      touch_params={"touch_takes": False}),
    # D20: v4_spread with every prior (durations, agreements, cue tables) fitted on the
    # controlled-possession labels (relabel keep2) instead of DFL's first-touch ones.
    "touch_gapteam_k2": semi_markov("touch_gapteam_k2", touch_w=1.0, kickoff_params={},
                                    team_w=0.5, team_where="all", team_dead="max",
                                    team_params={"cues": ("spread",)}, prior_labels="keep2"),
}
# C3: v4_spread plus solution/framemodel.py's per-frame P(home | alive), a gradient-boosted
# model of players-only features trained leave-one-match-out on degraded dev matches with keep2
# labels, read with no say on dead ball. Weight 1 or below (D16). Registered for the proxy
# league; touch_c3_only replaces spread by the model (which already reads spread).
_GAPTEAM = dict(touch_w=1.0, kickoff_params={}, team_w=0.5, team_where="all", team_dead="max",
                team_params={"cues": ("spread",)})
METHODS.update({
    "touch_c3": semi_markov("touch_c3", frame_w=0.5, **_GAPTEAM),
    "touch_c3_w1": semi_markov("touch_c3_w1", frame_w=1.0, **_GAPTEAM),
    "touch_c3_gaps": semi_markov("touch_c3_gaps", frame_w=0.5, frame_where="gaps", **_GAPTEAM),
    "touch_c3_only": semi_markov("touch_c3_only", touch_w=1.0, kickoff_params={}, frame_w=0.5),
    # After the BAR-ATH check (D22 update): the model read only where the ball is seen, where the
    # Smart Tagging agreement rose; gaps stay with spread as in v4_spread.
    "touch_c3_seen": semi_markov("touch_c3_seen", frame_w=0.5, frame_where="seen", **_GAPTEAM),
    "v5_frame_model": semi_markov("v5_frame_model", frame_w=1.0, frame_where="seen",
                                    **_GAPTEAM),
})
# Visible-ball Step 1 (2026-11-05): existing knobs re-tested on top of the submitted reference
# (D23), after gap_check.py --seen-ceiling put most possessions invented for the wrong team on
# the holder's touch hand-overs. Settings fixed once: touch_takes False (D10, D21), priors fitted
# on keep2 (D21's touch_gapteam_k2), and both. Registered for the proxy league only.
METHODS.update({
    "touch_c3_seen_w1_hold": semi_markov("touch_c3_seen_w1_hold", frame_w=1.0,
                                         frame_where="seen", touch_params={"touch_takes": False},
                                         **_GAPTEAM),
    "touch_c3_seen_w1_k2": semi_markov("touch_c3_seen_w1_k2", frame_w=1.0, frame_where="seen",
                                       prior_labels="keep2", **_GAPTEAM),
    "touch_c3_seen_w1_hold_k2": semi_markov("touch_c3_seen_w1_hold_k2", frame_w=1.0,
                                            frame_where="seen", prior_labels="keep2",
                                            touch_params={"touch_takes": False}, **_GAPTEAM),
})
# Track 1 (2026-11-05, EXPERIMENTS section 16): the learned dead-ball model (solution/deadmodel.py,
# v1 = D25's) in place of the speed likelihood, at weight 1, on top of the submitted reference.
# Settings fixed once before its first run. It cleared deadmodel_check's bar under D12 and TEMPO.
METHODS.update({
    "v6_deadball_model": semi_markov("v6_deadball_model", frame_w=1.0, frame_where="seen",
                                       deadm_w=1.0, **_GAPTEAM),
    # After the BAR-ATH check (D26 update): _dm moved dead ball into ball gaps there, against
    # the proxy. The model read only where the ball is seen; gaps keep the speed likelihood.
    "touch_c3_seen_w1_dms": semi_markov("touch_c3_seen_w1_dms", frame_w=1.0,
                                        frame_where="seen", deadm_w=1.0, deadm_where="seen",
                                        **_GAPTEAM),
})

# E13 (2026-11-06, D27, EXPERIMENTS section 17): the TV cut-away cue added to the submitted
# reference at weight 1, read from match.notes["tv"]. Its table comes from synthetic cues
# (solution/cutaway.py) of the decision ("pessimistic") or the reported ("optimistic") variant.
# Settings fixed once before the first run. Registered for tools/cutaway_check.py, not submitted.
METHODS.update({
    "touch_c3_seen_w1_tv": semi_markov("touch_c3_seen_w1_tv", frame_w=1.0, frame_where="seen",
                                       tv_w=1.0, tv_variant="pessimistic", **_GAPTEAM),
    "touch_c3_seen_w1_tv_opt": semi_markov("touch_c3_seen_w1_tv_opt", frame_w=1.0,
                                           frame_where="seen", tv_w=1.0,
                                           tv_variant="optimistic", **_GAPTEAM),
    # D31: the decision variant, calibrated to BAR-ATH's real split (tools/cutaway_calibrate.py)
    "touch_c3_seen_w1_tv_cal": semi_markov("touch_c3_seen_w1_tv_cal", frame_w=1.0,
                                           frame_where="seen", tv_w=1.0,
                                           tv_variant="calibrated", **_GAPTEAM),
})
# D32 (2026-11-06): the ball merge (solution/sources/fused.merged) under the reference alone, and
# under the calibrated TV method, after the TV method's BAR-ATH check failed on the in-gap team.
# Settings fixed once before the first run. Registered, not submitted.
METHODS.update({
    "touch_c3_seen_w1_fused": semi_markov("touch_c3_seen_w1_fused", frame_w=1.0,
                                          frame_where="seen", merge_ball=True, **_GAPTEAM),
    "touch_c3_seen_w1_tv_cal_fused": semi_markov("touch_c3_seen_w1_tv_cal_fused", frame_w=1.0,
                                                 frame_where="seen", tv_w=1.0,
                                                 tv_variant="calibrated", merge_ball=True,
                                                 **_GAPTEAM),
})
# #3 control model (2026-11-06, EXPERIMENTS section 21 Step 1): the submitted reference with the
# touches solution/controlmodel.py scores as contradicted left out of the holder. Settings fixed
# once before the first run. Registered for the proxy league, not submitted.
METHODS.update({
    "touch_c3_seen_w1_ctl": semi_markov("touch_c3_seen_w1_ctl", frame_w=1.0, frame_where="seen",
                                        control=True, **_GAPTEAM),
    # The one variant the Step 1 diagnostic allowed (its false drops sit on real hand-overs):
    # the threshold at a 5% false-drop target. Fixed before its first run.
    "touch_c3_seen_w1_ctl_p05": semi_markov("touch_c3_seen_w1_ctl_p05", frame_w=1.0,
                                            frame_where="seen", control={"false_drop": 0.05},
                                            **_GAPTEAM),
})
# Stacking (2026-11-07, EXPERIMENTS section 25, D38): every part that was "better" or ADOPT on
# the proxy but is not in the submission, put on top of the submission `v6_deadball_model`,
# one at a time and all together. Settings are each part's own registered ones; fixed before the
# first run. `v6_plus_stack` (the second run) is composed by section 25's rule from the singles.
# Registered for the proxy league; none is submitted without D31's BAR-ATH rules.
_DM = dict(frame_w=1.0, frame_where="seen", deadm_w=1.0, **_GAPTEAM)
METHODS.update({
    "v6_plus_tv": semi_markov("v6_plus_tv", tv_w=1.0, tv_variant="calibrated", **_DM),
    "v6_plus_ball": semi_markov("v6_plus_ball", merge_ball=True, **_DM),
    "v6_plus_tv_ball": semi_markov("v6_plus_tv_ball", tv_w=1.0, tv_variant="calibrated", merge_ball=True, **_DM),
    "v6_plus_cue": semi_markov("v6_plus_cue", deadball_w=0.5,
                          deadball_params={"cues": ("player", "ball")}, **_DM),
    "v6_plus_keep2": semi_markov("v6_plus_keep2", prior_labels="keep2", **_DM),
    "v6_plus_dead4": semi_markov("v6_plus_dead4", dead=4.0, **_DM),
    "v6_plus_snap": semi_markov("v6_plus_snap", snap={"window_s": 1.5, "pick": "first"}, **_DM),
    "v6_plus_control": semi_markov("v6_plus_control", control={"false_drop": 0.05}, **_DM),
    "v6_plus_all": semi_markov("v6_plus_all", tv_w=1.0, tv_variant="calibrated", merge_ball=True,
                          deadball_w=0.5, deadball_params={"cues": ("player", "ball")},
                          prior_labels="keep2", dead=4.0,
                          snap={"window_s": 1.5, "pick": "first"}, **_DM),
    # Run 2, composed by section 25's rule from run 1 under D12: the singles better on keep2 with
    # the interval excluding 0 and not WORSE on DFL were the merged ball, the TV cue with it, the
    # keep2 priors and the control model.
    "v6_plus_stack": semi_markov("v6_plus_stack", tv_w=1.0, tv_variant="calibrated", merge_ball=True,
                            prior_labels="keep2", control={"false_drop": 0.05}, **_DM),
    # Run 3, registered before it ran: the same stack without the two broadcast parts, so no
    # alignment cache and no D31 in-gap question. keep2 priors + control model only.
    "v7_keep2_control": semi_markov("v7_keep2_control", prior_labels="keep2", control={"false_drop": 0.05},
                            **_DM),
    # Run 4 (section 25), registered before it ran: the merged ball alone on the v7 submission,
    # no TV cue. D27's ride-along clause decides, with D31's BAR-ATH rules.
    "v7_plus_ball": semi_markov("v7_plus_ball", merge_ball=True, prior_labels="keep2",
                                control={"false_drop": 0.05}, **_DM),
})
# The referee's whistle (denis, 2026-11-12; reopens D26 and D45 at his call; solution/whistlecue.py):
# v7 plus the whistle cue at weight 1, with and without D33's end-of-half rule on the same
# whistles, the cue with the pessimistic detector (reported), and the control: the same table
# with no whistles (only the ball's place), which the whistle must beat to count. Settings fixed
# before the first run; decided under D4 on keep2 + D12, DFL not WORSE. Since D47 (2026-11-16)
# BAR-ATH's whistles come from the self-calibrating detector and the proxy's decision variant is
# "prior", set a priori.
METHODS.update({
    "v7_whistle": semi_markov("v7_whistle", prior_labels="keep2", control={"false_drop": 0.05},
                              whistle_w=1.0, **_DM),
    "v7_whistle_end": semi_markov("v7_whistle_end", prior_labels="keep2",
                                  control={"false_drop": 0.05}, whistle_w=1.0, whistle_end=True,
                                  **_DM),
    "v7_whistle_pess": semi_markov("v7_whistle_pess", prior_labels="keep2",
                                   control={"false_drop": 0.05}, whistle_w=1.0,
                                   whistle_variant="pessimistic", **_DM),
    "v7_place": semi_markov("v7_place", prior_labels="keep2", control={"false_drop": 0.05},
                            whistle_w=1.0, whistle_variant="none", **_DM),
    # D48 (D14 route): v7 plus only the end-of-half rule on the detected whistles, guarded by the
    # players' speed after the pick; no in-play cue (whistle_w 0, so the decode is v7's and only
    # the end of each half can change). The proxy's whistles: the noisiest plausible ("matched").
    "v7_end": semi_markov("v7_end", prior_labels="keep2", control={"false_drop": 0.05},
                          whistle_end=True, whistle_end_guard=True, whistle_variant="matched",
                          **_DM),
    # D47 update: the proxy with the false-whistle rate matched to BAR-ATH's Step 0 upper bound
    "v7_whistle_matched": semi_markov("v7_whistle_matched", prior_labels="keep2",
                                      control={"false_drop": 0.05}, whistle_w=1.0,
                                      whistle_variant="matched", **_DM),
})
# Dead-ball weight (denis, 2026-11-10; EXPERIMENTS section 31). D26 never tried deadm_w above 1.
# On v7 the dead-ball model alone leans dead at the 12:01 goal kick but loses the D-vs-alive sum;
# about 2 flips it. Two settings, fixed before the first run; decided under D4 on keep2 + D12,
# DFL not WORSE. Everything else is v7_keep2_control.
METHODS.update({
    "v7_deadm15": semi_markov("v7_deadm15", prior_labels="keep2", control={"false_drop": 0.05},
                              **{**_DM, "deadm_w": 1.5}),
    "v7_deadm2": semi_markov("v7_deadm2", prior_labels="keep2", control={"false_drop": 0.05},
                             **{**_DM, "deadm_w": 2.0}),
})
# The keeper's ball against the goal kick (denis, 2026-11-11; solution/keepermodel.py): the keeper
# model's per-state log-likelihoods in the keeper zone, at weight 1, on top of the submission.
# Fixed before the first run. Decided under D4 on keep2 + D12, DFL not WORSE, and also scored
# with the keeper's ball hidden (HIDE_KEEPER=0.5 and 0), the BAR-ATH failure denis reported.
METHODS.update({
    "v7_keeper": semi_markov("v7_keeper", prior_labels="keep2", control={"false_drop": 0.05},
                             keeper_w=1.0, **_DM),
    # Registered 2026-11-11 after denis's BAR-ATH gap check (keeper zone seen 0.30 against the
    # proxy's 0.42; gaps ending at the keeper 12.9% against 2.6%) and before any league result:
    # the same model trained with the ball also lost behind the goal (train_lose_keep 0.5).
    "v7_keeper2": semi_markov("v7_keeper2", prior_labels="keep2", control={"false_drop": 0.05},
                              keeper_w=1.0, keeper_params={"train_hide_keep": 1.0,
                                                           "train_lose_keep": 0.5}, **_DM),
})
# Without the nearest-player voter (denis, 2026-11-10; EXPERIMENTS section 31). `no_nearest`
# gives the decoder no nearest player on any frame, so its H/A term and its share of the D vs
# alive sum both go. `nearest_fb` keeps it only where the touch engine names no holder
# (decoder.DEFAULTS nearest_fallback). Fixed before the first run; D4 on keep2 + D12, DFL not WORSE.
METHODS.update({
    "v7_no_nearest": semi_markov("v7_no_nearest", prior_labels="keep2",
                                 control={"false_drop": 0.05}, nearest=False, **_DM),
    "v7_nearest_fb": semi_markov("v7_nearest_fb", prior_labels="keep2",
                                 control={"false_drop": 0.05}, nearest_fallback=True, **_DM),
})
# Decoder cost re-tune (2026-11-06, EXPERIMENTS section 22). The transition costs were set to 0
# on semi_markov with speed-only evidence and DFL truth (D8) and never revisited on the current
# evidence or on keep2 truth. Nine settings on top of the submitted reference, fixed before the
# first run; `c3s_` stands for v5_frame_model plus. Registered for the proxy league only.
_SEEN = dict(frame_w=1.0, frame_where="seen", **_GAPTEAM)
METHODS.update({
    "c3s_dead1": semi_markov("c3s_dead1", dead=1.0, **_SEEN),
    "c3s_dead2": semi_markov("c3s_dead2", dead=2.0, **_SEEN),
    "c3s_dead4": semi_markov("c3s_dead4", dead=4.0, **_SEEN),
    "c3s_sw1": semi_markov("c3s_sw1", switch=1.0, **_SEEN),
    "c3s_sw2": semi_markov("c3s_sw2", switch=2.0, **_SEEN),
    "c3s_dead2_sw1": semi_markov("c3s_dead2_sw1", dead=2.0, switch=1.0, **_SEEN),
    "c3s_dead4_sw2": semi_markov("c3s_dead4_sw2", dead=4.0, switch=2.0, **_SEEN),
    "c3s_dmin2": semi_markov("c3s_dmin2", dead_min_s=2.0, **_SEEN),
    "c3s_dmin4": semi_markov("c3s_dmin4", dead_min_s=4.0, **_SEEN),
})


# E15 ball inference (denis, 2026-11-07; solution/ballmodel.py). The submitted reference
# (touch_c3_seen_w1_dm) with the C3 frame model kept on seen frames and, on gap frames, the team
# nearer the INFERRED ball, through the same frame term (frame_w 1, frame_where "all"; the fn
# leaves frames it has no say on NaN). `_ball` uses the learned imputer; the `_bor_*` rows are
# Step 0 oracles: the true ball plus noise / lag, IDSSE only. Settings fixed before the first run.
def ball_frame_fn(mode="model", noise_m=0.0, lag_s=0.0):
    def fn(match):
        from solution import ballmodel
        seen = framemodel.log_odds(match, framemodel.PARAMS)
        seen = np.where(match.ball_seen, seen, np.nan)
        gaps = ballmodel.log_odds(match, mode=mode, noise_m=noise_m, lag_s=lag_s)
        return np.where(np.isfinite(gaps), gaps, seen)
    return fn


_DM_ALL = dict(frame_w=1.0, frame_where="all", deadm_w=1.0, **_GAPTEAM)
METHODS.update({
    "touch_c3_seen_w1_dm_ball": semi_markov("touch_c3_seen_w1_dm_ball",
                                            frame_fn=ball_frame_fn(), **_DM_ALL),
    "dm_bor_0": semi_markov("dm_bor_0", frame_fn=ball_frame_fn("oracle", 0.0, 0.0), **_DM_ALL),
    "dm_bor_5m": semi_markov("dm_bor_5m", frame_fn=ball_frame_fn("oracle", 5.0, 0.0),
                             **_DM_ALL),
    "dm_bor_10m": semi_markov("dm_bor_10m", frame_fn=ball_frame_fn("oracle", 10.0, 0.0),
                              **_DM_ALL),
    "dm_bor_lag1": semi_markov("dm_bor_lag1", frame_fn=ball_frame_fn("oracle", 0.0, 1.0),
                               **_DM_ALL),
})


# REVIEW.md item 4 (denis, 2026-11-07): the three sub-bar 5/5 gains stacked on the submitted
# reference touch_c3_seen_w1_dm: priors fitted on keep2 labels (D25), the broadcast ball merge
# (D27, D32) and the player-beyond-a-line dead-ball cue at weight 0.5 (D15; player only, so no
# ball-beyond thinning applies). `_dead4` adds the `dead` 4 cost, the one try D36 allows inside
# this stack. Both fixed before the first run; decided under D4 on keep2 + D12, DFL not WORSE.
_STACK = dict(frame_w=1.0, frame_where="seen", deadm_w=1.0, prior_labels="keep2",
              merge_ball=True, deadball_w=0.5, deadball_params={"cues": ("player",)},
              **_GAPTEAM)
METHODS.update({
    "dm_stack3": semi_markov("dm_stack3", **_STACK),
    "dm_stack3_dead4": semi_markov("dm_stack3_dead4", dead=4.0, **_STACK),
})


# Names (D40, 2026-11-07): the submissions are `v1_decoder` .. `v7_keep2_control` in order, and
# `v6_plus_<part>` is a part stacked on the v6 submission (EXPERIMENTS section 25). Every former
# name stays registered as an alias of the same function, so EXPERIMENTS, DECISIONS, cached proxy
# results and old commands still resolve; tools print the new name.
ALIASES = {
    "semi_markov": "v1_decoder",
    "touch_markov": "v2_touch",
    "touch_kickoff": "v3_kickoff",
    "touch_gapteam": "v4_spread",
    "touch_c3_seen_w1": "v5_frame_model",
    "touch_c3_seen_w1_dm": "v6_deadball_model",
    "dm_tv": "v6_plus_tv",
    "dm_ball": "v6_plus_ball",
    "dm_tvb": "v6_plus_tv_ball",
    "dm_cue": "v6_plus_cue",
    "dm_k2": "v6_plus_keep2",
    "dm_dead4": "v6_plus_dead4",
    "dm_snap": "v6_plus_snap",
    "dm_ctl": "v6_plus_control",
    "dm_all": "v6_plus_all",
    "dm_stack": "v6_plus_stack",
    "dm_k2ctl": "v7_keep2_control",
}
FORMER = {new: old for old, new in ALIASES.items()}
for _old, _new in ALIASES.items():
    METHODS[_old] = METHODS[_new]


def canonical(name):
    """The current name of a registered method (an alias resolves to it)."""
    return ALIASES.get(name, name)
