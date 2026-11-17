# Experiments log

One row per proxy-league result worth keeping. Newest at the bottom of each table. Copy the
numbers from `make proxy` output; they are aggregates of public external data, so they are safe
to commit. **Never put numbers computed from the BAR-ATH truth here except the public-window
correctness check, and never choose anything on that.**

How to read a row:
- **spellF1** is pooled over every true possession of the split, mean of the simulator seeds.
- **interval** is the 5-95% bootstrap over true possessions (first seed).
- **gain** is against the reference named in the section, with a paired bootstrap interval.
- **verdict** is the runner's: ADOPT needs gain >= 0.02, an interval above 0, and wins on at
  least two thirds of the matches.

## Setup the numbers depend on

| item | value |
|---|---|
| matches | IDSSE dev split: J03WMX, J03WOH, J03WOY, J03WPY, J03WQQ (2,191 true possessions). Held out: J03WN1, J03WR9 |
| degradation | simulator v0: ball hidden by timing only, lengths drawn from BAR-ATH `baseline_raw.csv` |
| seeds | 0, 1, 2 |
| labels | D where DFL `BallStatus == 0`, else `BallPossession` team |

If any of these change, start a new section: numbers from different setups are not comparable.

## Section 1: baselines, simulator v0 (2026-10-28)

Reference: `baseline_smooth50`. Code at branch `miha/proxy-league`, on top of `1d86a13`.

| date | method | spellF1 | interval | macroF1 | possessions | gain vs ref | verdict | note |
|---|---|---|---|---|---|---|---|---|
| 2026-10-28 | oracle, clean input, all 7 | 1.0000 | 1.0000..1.0000 | 1.0000 | 3061 | | | sanity: the scorer path is exact |
| 2026-10-28 | baseline_smooth50 | 0.2114 | 0.1995..0.2237 | 0.4534 | 2055 | ref | | official full-match 0.2433 |
| 2026-10-28 | baseline_smooth25 | 0.2018 | 0.1896..0.2117 | 0.4485 | 2535 | -0.0097 (-0.0175..-0.0035) | WORSE | official 0.1996 |
| 2026-10-28 | baseline_smooth125 | 0.1732 | 0.1662..0.1910 | 0.4389 | 1433 | -0.0382 (-0.0425..-0.0227) | WORSE | official 0.2242 |
| 2026-10-28 | baseline_raw | 0.0773 | 0.0726..0.0821 | 0.4289 | 6952 | -0.1341 (-0.1450..-0.1237) | WORSE | official 0.0486 |

Takeaway: the proxy league picks the same best window as the official full-match scoring, where
the public 5 minutes pick ±125. It ranks ±25 above ±125, the reverse of the official order, so it
is a guide rather than a copy.

## Section 2: baselines, simulator v1 (2026-10-28)

Setup changes from section 1, so these numbers are not comparable with it:
- **Degradation:** simulator v1 (`solution/simulate.py`, `V1` parameters). The ball keeps v0's
  stretch lengths, reordered so gaps fall in slow play. Players get flicker, multi-player
  dropout episodes and stale ghost slots.
- **Loader:** the IDSSE loader no longer counts referees as away players (cache v2). This
  mattered in J03WMX and, presumably, J03WN1.
- **Unchanged:** the matches, seeds 0 1 2, and the labels.

Reference: `baseline_smooth50`. Code at branch `miha/simulator-v1`, on top of `a7373b8`.

| date | method | spellF1 | interval | macroF1 | possessions | gain vs ref | verdict | note |
|---|---|---|---|---|---|---|---|---|
| 2026-10-28 | baseline_smooth50 | 0.2607 | 0.2470..0.2705 | 0.6191 | 2354 | ref | | official full-match 0.2433 / macroF1 0.6724 |
| 2026-10-28 | baseline_smooth25 | 0.2449 | 0.2356..0.2578 | 0.6234 | 2991 | -0.0157 (-0.0185..-0.0049) | WORSE | official 0.1996 / 0.6752 |
| 2026-10-28 | baseline_smooth125 | 0.2017 | 0.1832..0.2092 | 0.5695 | 1672 | -0.0590 (-0.0722..-0.0517) | WORSE | official 0.2242 / 0.6356 |
| 2026-10-28 | baseline_raw | 0.0848 | 0.0811..0.0896 | 0.6075 | 8487 | -0.1759 (-0.1849..-0.1623) | WORSE | official 0.0486 / 0.6532 |
| 2026-10-28 | baseline_smooth50, `SIM=v0` | 0.2115 | 0.1994..0.2235 | 0.4535 | 2051 | ref (v0) | | v0 after the referee fix; section 1 had 0.2114 |
| 2026-10-28 | baseline_smooth25, `SIM=v0` | 0.2022 | 0.1897..0.2119 | 0.4486 | 2527 | -0.0093 (-0.0169..-0.0035) | WORSE | |
| 2026-10-28 | baseline_smooth125, `SIM=v0` | 0.1738 | 0.1661..0.1914 | 0.4389 | 1427 | -0.0377 (-0.0421..-0.0223) | WORSE | |
| 2026-10-28 | baseline_raw, `SIM=v0` | 0.0773 | 0.0726..0.0819 | 0.4288 | 6939 | -0.1342 (-0.1449..-0.1232) | WORSE | |
| 2026-10-29 | semi_markov w=0.1, costs 0 | 0.0957 | 0.0785..0.1016 | 0.6993 | 349 | -0.1649 (-0.1822..-0.1540) | WORSE | E1 tuning A; branch `miha/e1-semi-markov` on `d74045e` |
| 2026-10-29 | semi_markov w=0.3, costs 0 | 0.2367 | 0.2068..0.2397 | 0.7451 | 818 | -0.0240 (-0.0485..-0.0214) | WORSE | E1 tuning A |
| 2026-10-29 | semi_markov w=1, costs 0 | 0.2969 | 0.2794..0.3067 | 0.7322 | 1966 | +0.0363 (+0.0257..+0.0440) | ADOPT | E1 tuning A; wins 5/5 |
| 2026-10-29 | semi_markov w=3, costs 0 | 0.1729 | 0.1645..0.1794 | 0.7062 | 4534 | -0.0878 (-0.0965..-0.0755) | WORSE | E1 tuning A, added because w=1 was the grid edge |
| 2026-10-29 | semi_markov w=1, switch=2 | 0.3010 | 0.2833..0.3137 | 0.7345 | 1531 | +0.0404 (+0.0301..+0.0500) | ADOPT | E1 tuning B; vs w=1 costs 0: +0.0041 (-0.0016..+0.0119), within noise |
| 2026-10-29 | semi_markov w=1, switch=5 | 0.2826 | 0.2603..0.2924 | 0.7363 | 1236 | +0.0220 (+0.0067..+0.0286) | ADOPT | E1 tuning B |
| 2026-10-29 | semi_markov w=1, dead=2 | 0.2992 | 0.2795..0.3064 | 0.7347 | 1930 | +0.0386 (+0.0257..+0.0443) | ADOPT | E1 tuning B |
| 2026-10-29 | semi_markov w=1, dead=5 | 0.3007 | 0.2813..0.3083 | 0.7356 | 1894 | +0.0400 (+0.0272..+0.0461) | ADOPT | E1 tuning B; vs w=1 costs 0: +0.0037 (-0.0014..+0.0047), within noise |
| 2026-10-29 | semi_markov w=1, restart=2 | 0.2993 | 0.2795..0.3064 | 0.7348 | 1930 | +0.0386 (+0.0257..+0.0443) | ADOPT | E1 tuning B; same as dead=2 |
| 2026-10-29 | semi_markov w=1, restart=5 | 0.3008 | 0.2813..0.3084 | 0.7356 | 1894 | +0.0401 (+0.0273..+0.0462) | ADOPT | E1 tuning B; same as dead=5 |
| 2026-10-29 | semi_markov w=1, switch=2, dead=5 | 0.3027 | 0.2837..0.3143 | 0.7385 | 1458 | | | E1 tuning C, run against w=1 costs 0 only: +0.0058 (-0.0015..+0.0132), wins 3/5, within noise |
| 2026-10-29 | **semi_markov** (registered: w=1, costs 0) | 0.2969 | 0.2794..0.3067 | 0.7322 | 1966 | +0.0363 (+0.0257..+0.0440) | **ADOPT** | final `make proxy METHODS="baseline_smooth50 semi_markov"`; wins 5/5. D8 |
| 2026-10-29 | semi_markov, speed cue off (diagnostic) | 0.2470 | 0.2319..0.2596 | 0.3859 | 1624 | -0.0136 (-0.0224..-0.0020) | WORSE | ablation, not a tuning setting: without speed, D is hardly predicted |
| 2026-10-29 | semi_markov, speed_lag=12 | 0.2965 | 0.2776..0.3045 | 0.7288 | 1976 | +0.0358 (+0.0241..+0.0422) | ADOPT | speed fix 1 (±0.5 s window); same as semi_markov here, see section 3 |
| 2026-10-29 | semi_markov, speed_scale=rank | 0.2993 | 0.2823..0.3093 | 0.7312 | 1956 | +0.0386 (+0.0289..+0.0471) | ADOPT | speed fix 2 (quantile within the match); see section 3 |
| 2026-10-29 | semi_markov, speed_lag=12 + rank | 0.2960 | 0.2782..0.3046 | 0.7252 | 1976 | +0.0353 (+0.0248..+0.0423) | ADOPT | both fixes; see section 3 |

Simulator calibration from `make sim-check` (5 dev matches x 3 seeds), with selected rows:

| statistic | BAR-ATH | v0 | v1 |
|---|---|---|---|
| ball visible share | 0.593 | 0.597 | 0.599 |
| gap length p50 / p75 (s) | 10.72 / 21.06 | 10.64 / 21.12 | 10.60 / 20.88 |
| visible stretch p50 / p75 (s) | 13.80 / 30.32 | 13.80 / 29.56 | 13.72 / 29.55 |
| player speed in gaps / ball seen (m/s) | 1.17 / 2.16 | 1.88 / 1.83 | 1.23 / 2.27 |
| speed ratio | 0.542 | 1.027 | 0.545 |
| gaps after ball wide or beyond a line | 0.540 | 0.591 | 0.651 |
| seen frames with the ball beyond a line | 0.042 | 0.262 | 0.164 |
| objects per frame, mean (p5 / p25 / p50) | 20.49 (14 / 21 / 22) | 21.93 (21 / 22 / 22) | 20.69 (14 / 21 / 22) |
| team frames with > 11 / < 9 | 0.064 / 0.088 | 0 / 0 | 0.059 / 0.103 |
| tracklet life p10 / p50 / p90 (s) | 2.6 / 35.5 / 284.6 | 986 / 2765 / 2937 | 1.4 / 38.3 / 297.7 |
| frames with a same-team duplicate | 0.068 | 0.066 | 0.079 |

Takeaways:
- **The referee fix barely matters:** smooth50 under v0 moved by 0.0001.
- **v1 brings macroF1 close to the official full-match level,** 0.62 against 0.67, where v0
  gave 0.45. Its macroF1 order is exactly the official one (smooth25 > smooth50 > raw >
  smooth125).
- **smooth50 is still the best spellF1 window.** As under v0, smooth25 ranks above smooth125,
  the reverse of the official order.
- **Seed spread rose:** smooth50 now varies by 0.0089 across seeds, against 0.0011 under v0.
- **These comparisons are sanity checks only.** The simulator was calibrated to the EDA
  aggregates, never to these scores.

E1, the semi-Markov decoder (2026-10-29), [solution/decoder.py](../solution/decoder.py):
- **11 settings tried** on the 5 dev matches: 4 evidence weights, 6 one-cost-at-a-time, 1
  combination. The ablation is extra and was not a candidate. The 3 speed variants were added
  afterwards for robustness (section 3), which makes 14 settings scored in all.
- **The decoder is adopted** at w = 1 with all costs 0 (+0.036, wins 5/5). No cost setting beat
  that beyond noise, so none is used.
- **`dead` and `restart` act as one knob.** Every interior dead-ball spell pays both, so only
  their sum matters.
- **The gain depends entirely on the speed cue.** Without it the decoder is worse than the
  baseline. Speed priors come from clean IDSSE, so how well the cue transfers to BAR-ATH is
  the main open risk.

## Section 3: stress test, simulator v1 plus white position noise (2026-10-29)

Setup: section 2, plus `make proxy NOISE=<sd>`. That adds independent Gaussian noise of `sd` metres
to every player coordinate on every frame, after degrading (`simulate.add_position_noise`, its
own random stream). This is **not a simulator version**: BAR-ATH's position noise is unmeasured.
These rows test robustness and pick nothing by themselves. Reference: `baseline_smooth50` at the
same noise. Code at branch `miha/e1-semi-markov`.

| date | noise sd (m) | method | spellF1 | interval | macroF1 | possessions | gain vs ref | verdict |
|---|---|---|---|---|---|---|---|---|
| 2026-10-29 | 0.05 | baseline_smooth50 | 0.2607 | 0.2456..0.2698 | 0.6191 | 2356 | ref | |
| 2026-10-29 | 0.05 | semi_markov | 0.3032 | 0.2871..0.3144 | 0.7287 | 1930 | +0.0425 (+0.0339..+0.0521) | ADOPT |
| 2026-10-29 | 0.1 | baseline_smooth50 | 0.2602 | 0.2453..0.2696 | 0.6191 | 2374 | ref | |
| 2026-10-29 | 0.1 | semi_markov | 0.2798 | 0.2660..0.2933 | 0.6293 | 1835 | +0.0196 (+0.0141..+0.0309) | better, below the adoption bar |
| 2026-10-29 | 0.1 | semi_markov, speed_lag=12 | 0.2964 | 0.2819..0.3087 | 0.7276 | 1959 | +0.0361 (+0.0288..+0.0474) | ADOPT |
| 2026-10-29 | 0.1 | semi_markov, speed_scale=rank | 0.2991 | 0.2837..0.3107 | 0.7329 | 1953 | +0.0389 (+0.0302..+0.0487) | ADOPT |
| 2026-10-29 | 0.1 | semi_markov, speed_lag=12 + rank | 0.2956 | 0.2809..0.3075 | 0.7246 | 1960 | +0.0354 (+0.0282..+0.0461) | ADOPT |
| 2026-10-29 | 0.2 | baseline_smooth50 | 0.2583 | 0.2432..0.2669 | 0.6187 | 2398 | ref | |
| 2026-10-29 | 0.2 | semi_markov | 0.2444 | 0.2299..0.2581 | 0.3686 | 1582 | -0.0139 (-0.0205..-0.0005) | WORSE |
| 2026-10-29 | 0.2 | semi_markov, speed_lag=12 | 0.2956 | 0.2803..0.3070 | 0.7251 | 1919 | +0.0373 (+0.0298..+0.0482) | ADOPT |
| 2026-10-29 | 0.2 | semi_markov, speed_scale=rank | 0.2961 | 0.2789..0.3065 | 0.7303 | 1919 | +0.0378 (+0.0287..+0.0475) | ADOPT |
| 2026-10-29 | 0.2 | semi_markov, speed_lag=12 + rank | 0.2953 | 0.2793..0.3062 | 0.7250 | 1930 | +0.0370 (+0.0292..+0.0473) | ADOPT |

From `tools/speed_check.py`, all 5 dev matches x 3 seeds:

| input | speed p10 / p50 / p90 (m/s) | missing / seen speed ratio | decoded dead share (truth 0.441) |
|---|---|---|---|
| degraded v1, 2-frame lag | 0.69 / 1.64 / 3.24 | 0.544 | 0.432 |
| + noise 0.05 | 0.92 / 1.74 / 3.29 | 0.596 | 0.382 |
| + noise 0.1 | 1.32 / 2.00 / 3.43 | 0.677 | 0.209 |
| + noise 0.2 | 2.21 / 2.82 / 3.97 | 0.810 | 0.000 |
| + noise 0.1, 12-frame lag | 0.72 / 1.62 / 3.21 | 0.551 | 0.428 |
| + noise 0.1, rank | (as the 2-frame lag) | 0.677 | 0.439 |

Takeaways:
- **The registered `semi_markov` (2-frame lag) is fragile.** White noise inflates every speed,
  so play stops looking slow. At 0.1 m it decodes half the dead ball; at 0.2 m it decodes none
  and loses to the baseline.
- **Both fixes remove the fragility at no cost without noise.** At any noise level they stay
  within 0.001 to 0.005 of their noise-free score.
- **Neither fix beats `semi_markov` on the noise-free league** (+0.0024 at best, within noise).
  So under D4 alone neither would be adopted. Whether to switch waits on the BAR-ATH speed
  check (WORKLOG).
- **White noise is the worst case for a 2-frame lag.** Real tracking noise is likely smoother,
  so 0.1 to 0.2 m white is a stress level, not an estimate.

### Tempo stress and the BAR-ATH speed check (2026-10-29)

`make proxy SPEED=0.9` shrinks every position towards the pitch centre by 0.9
(`simulate.scale_speeds`). Every speed drops by 10%, and who is nearest the ball does not change.
Rows in the same format as above (column 2 is the speed factor):

| date | speed factor | method | spellF1 | interval | macroF1 | possessions | gain vs ref | verdict |
|---|---|---|---|---|---|---|---|---|
| 2026-10-29 | 0.9 | baseline_smooth50 | 0.2607 | 0.2470..0.2705 | 0.6191 | 2354 | ref | |
| 2026-10-29 | 0.9 | semi_markov, speed in m/s (D8) | 0.2841 | 0.2686..0.2945 | 0.7233 | 1967 | +0.0234 (+0.0137..+0.0328) | ADOPT |
| 2026-10-29 | 0.9 | semi_markov, speed_lag=12 | 0.2810 | 0.2639..0.2910 | 0.7185 | 1966 | +0.0203 (+0.0097..+0.0284) | ADOPT |
| 2026-10-29 | 0.9 | semi_markov, speed_scale=rank | 0.2993 | 0.2823..0.3093 | 0.7312 | 1956 | +0.0386 (+0.0289..+0.0471) | ADOPT |
| 2026-10-29 | 1 | **semi_markov, registered (rank, D9)** | 0.2993 | 0.2823..0.3093 | 0.7312 | 1956 | +0.0386 (+0.0289..+0.0471) | **ADOPT** |

BAR-ATH, `tools/speed_check.py --bar-ath`, run by miha. Label-free aggregates of the ATD; nothing
here is scored against BAR-ATH truth.

| | speed p10 / p50 / p90 (m/s) | missing / seen ratio | decoded dead share | possessions per 90 min |
|---|---|---|---|---|
| BAR-ATH, 2-frame lag | 0.68 / 1.48 / 3.01 | 0.544 | 0.500 (m/s) | 439 |
| BAR-ATH, 12-frame lag | 0.66 / 1.44 / 2.97 | 0.542 | 0.509 (m/s) | 439 |
| BAR-ATH, rank | (as the 2-frame lag) | 0.544 | 0.429 (rank) | 425 |
| IDSSE degraded v1, 2-frame lag | 0.69 / 1.64 / 3.24 | 0.544 | 0.432 (truth 0.441) | 372 (truth 414) |

Takeaways:
- **BAR-ATH shows no sign of position noise.** Its 2-frame and 12-frame deciles differ by
  0.02 to 0.04 m/s, the same as clean IDSSE. Its missing-to-seen ratio, 0.544, equals the
  degraded IDSSE's, where noise would push it up (0.68 at 0.1 m).
- **BAR-ATH plays about 10% slower** at the median and upper deciles (0.90 to 0.93 times IDSSE).
  In m/s the decoder then calls half of the match dead ball, against the 0.43 it decodes on
  IDSSE.
- **A 10% slowdown costs the m/s decoder 0.013 spellF1** on the proxy league; the ±0.5 s window
  loses 0.016. The rank is unchanged by construction.
- **The rank's own risk is a match with an unusual amount of dead ball.** On the plain league it
  lost narrowly on the two dev matches with the most and least dead ball (J03WOH 0.476: -0.002;
  J03WQQ 0.407: -0.006) and won on the other three.

## Section 4: public-window correctness checks (L4), never for choosing (2026-10-30)

`make eval` on `gt/BAR-ATH_possession_gt_public.csv`: 7,500 graded frames, 17 possessions (mean
14.2 s). **Correctness only.** It catches alignment, team-mapping and file-format bugs. With 17
possessions its ranking is noise, and RULES.md forbids tuning on it. Run by miha; the submission
is `make submission TEAM=e1_semi_markov` from branch `miha/first-submission`.

| date | submission | macroF1 | spellF1 | poss | dur (s) | transF1 | lat_f | shareE | frag |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-30 | e1_semi_markov | 0.7893 | 0.3512 | 24 | 10.2 | 0.2857 | +1.3 | 0.013 | 1.43 |
| 2026-10-30 | e3_touch_markov, section 5 engine | 0.8039 | 0.4600 | 23 | 10.8 | -- | -1.6 | 0.036 | -- |
| 2026-11-02 | e3_touch_markov, D12 engine | 0.8082 | 0.4333 | 31 | 8.0 | 0.3636 | +7.8 | 0.006 | 1.71 |
| 2026-11-04 | c3_seen_w1 (`touch_c3_seen_w1`, D23) | 0.8372 | 0.5818 | 16 | 15.2 | 0.3415 | +11.1 | 0.006 | 1.05 |
| 2026-11-06 | our_team_tv (`touch_c3_seen_w1_tv_cal_fused`, comparison file) | 0.8266 | 0.5500 | 15 | 16.4 | 0.2564 | +13.0 | 0.004 | 0.95 |
| 2026-11-06 | our_team_dm (`touch_c3_seen_w1_dm`, D37 check 2) | 0.8474 | 0.6533 | 13 | 19.8 | 0.4000 | +5.4 | 0.022 | 0.76 |
| 2026-10-30 | baseline_smooth50 | 0.6207 | 0.3256 | 26 | 10.8 | 0.0816 | +4.0 | 0.048 | 1.43 |
| 2026-10-30 | baseline_smooth125 | 0.5812 | 0.4242 | 16 | 18.5 | 0.1667 | -7.7 | 0.052 | 0.81 |
| 2026-10-30 | baseline_smooth25 | 0.6200 | 0.2868 | 36 | 7.6 | 0.2034 | +8.2 | 0.056 | 1.90 |
| 2026-10-30 | baseline_raw | 0.5972 | 0.0378 | 131 | 2.0 | 0.1410 | -3.1 | 0.046 | 6.52 |

**c3_seen_w1 (2026-11-04), run by miha.** These are correctness checks, not a ranking:
- two `solution/run.py` runs gave the same sha256 (`f7459db0…bcf8358d`), about 18 s each, as
  the model now trains at start-up;
- no team swap (share error 0.006, team right on 0.922 of both-alive frames);
- 16 possessions against 17. Every earlier method over-split (24 to 31);
- transitions a median 11 frames late (e1 +1.3, touch_markov +7.8). That is the decoder, not
  an alignment slip, which would shift every method alike. It is noted, not tuned on;
- full match: dead ball 0.44 of graded frames (inside D9's IDSSE 0.41 to 0.48), 332 possessions
  by the validator's count.
The scoreboard's `our_team` row (0.8130 / 0.4800, 28 possessions) is an earlier submission file
whose method was not recorded.

**our_team_dm (2026-11-06), D37's check 2, run by Claude on the file miha wrote.** Correctness
criteria fixed in D37 before the file existed: share error 0.022 (under 0.03), 0 abstentions on
graded frames, and no stoppage inside the window's five long possessions. All three pass. What
changed against the submission, described only: the two false stoppages (12425-12485 and
14450-14515) are gone; the 5.5 s true stoppage at 12851-12989 is now missed, so possession #5
runs into #6; the spurious possessions fall from 5 to 2; the same 6 short possessions are
missed; restarts are decoded a median 5 frames late against 11. The window spellF1 (0.6533,
5-95% 0.45..0.83) is read and recorded, never compared to choose (D37). The TV comparison file is
on the board for the same reason.

Pipeline checks from the same run: `make check-loader` agrees with all four published baselines on
100.000% of graded frames. Two `make submission` runs gave the same sha256 (`95f6366e…69fa12`)
in 1.7 s each, using the existing caches.

What it says about bugs:
- **No team swap.** macroF1 is the highest on the board and the possession-share error is 0.013;
  a swap would collapse both.
- **No alignment slip.** Transition latency is +1.3 frames.
- **Dead ball not grossly off.** macroF1 0.79 means D's F1 is at least 0.37, against 0.15 for
  smooth125. The e1 detail block was cut off by the pager, so the per-class figures are
  unchecked.
- **More possessions than the truth here (24 against 17), where the proxy league under-segments.**
  That fits the known gap that DFL labels are choppier than BAR-ATH's (simulate.py docstring),
  but 17 possessions cannot show it. It is a question for the label-semantics check (L2),
  not something to tune.

Label-length comparison (2026-11-04, after EXPERIMENTS section 11). The public truth's own
possessions are set against the clean DFL labels of the 5 IDSSE dev matches. It is a
description of the labels, never for choosing:

| labels | possessions per 90 | under 1 s | under 2 s | median s | mean s | direct H<->A per alive min |
|---|---|---|---|---|---|---|
| BAR-ATH public window (17 possessions) | 306 | 0.06 | 0.12 (2 of 17) | 7.6 | 14.2 | 2.98 |
| IDSSE dev, DFL (2,191) | 414 | 0.12 | 0.30 | 3.7 | 7.3 | 6.31 |

- **The count alone is not unusual.** Of DFL's 92 five-minute windows, 20% hold 17
  possessions or fewer (p10 16, p50 22.5).
- **The short share is lower.** If 30% were short, 2 or fewer of 17 would come up about 8% of
  the time: suggestive, not conclusive. Hand-overs per alive minute are half DFL's.
- **Why it matters:** if BAR-ATH's truth splits less finely than DFL's, the short possessions
  that sections 9 to 11 chased are worth less on the real match than on the proxy. Chasing
  them raises the possession count, and on BAR-ATH the count may already be too high:
  `touch_gapteam` decodes 393 per 90 (section 10) against 306 here. CHALLENGE.md
  ("Possession rules") explains it: possession changes only on control, while DFL flips at the
  first touch (D20). Still never tuned on.


**c3_seen_w1, segment-level diagnostic (2026-11-06, review session).** Correctness and description
only ([REVIEW.md](../REVIEW.md) section 4); no proxy-league run this session. `run_evaluation.py
--bootstrap 1000`: spellF1 0.5818 (5-95% 0.4000..0.7515), per IoU 0.667 / 0.667 / 0.667 / 0.545 /
0.364. At IoU 0.5: 11 recovered, 6 missed (5 of them 3.6 s or shorter, all with the ball seen; the
sixth merged because its short neighbour was missed), 5 spurious (2 fragments cut by false
stoppages of 2.4 and 2.6 s, 1 missed stoppage start, 1 phantom hand-over, 1 real 0.8 s possession
found 0.6 s late). Ball seen on 0.882 of the window; true D share on seen frames 0.147 (predicted
0.143), on missing frames 0.558 (0.525); 4 true stoppages against 6 predicted. Matched hand-over
latency median +20 frames, restarts +13, all positive. Oracle repairs on the window (each reads
the truth; 17 possessions): remove false stoppages +0.063; perfect dead ball +0.076; true labels
on the spurious spells +0.063; snap matched hand-overs within 1.5 s +0.085; write in the missed
possessions under 4 s +0.166, all of them +0.229. Never for choosing.

**our_team_dm, frame-level diagnostic (2026-11-07).** `make frame-report FILES="our_team
c3_seen_w1"` on the submitted file (`touch_c3_seen_w1_dm`, sha `a4c78a40…`), with the previous
submission beside it. Frame level, graded window only, description only; the window's 21
segments decide nothing.

| file | macroF1 | acc | F1 H | F1 A | F1 D | wrong frames | runs | late | early | missed | spurious | lat med (f) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| our_team (`_dm`) | 0.8474 | 0.863 | 0.870 | 0.894 | 0.778 | 1028 (13.7%) | 19 | 310 | 79 | 329 | 310 | +5 |
| c3_seen_w1 | 0.8372 | 0.850 | 0.854 | 0.893 | 0.765 | 1124 (15.0%) | 22 | 423 | 0 | 191 | 510 | +11 |
| baseline_smooth50 | 0.6207 | 0.713 | 0.778 | 0.762 | 0.323 | 2151 (28.7%) | 28 | 127 | 301 | 608 | 727 | +4 |

The kinds (tool docstring): *late* and *early* are the frames between a true boundary and the
file's; *missed* is a true segment with no right frame; *spurious* a file segment with no right
frame. For the submission:
- Per class, H precision 0.83 and D recall 0.67 are the weak figures: 27% of true dead-ball
  frames carry H. By label pair, D>H is 39% of the wrong frames, A>H 25%, H>A 20%, D>A 8%,
  H>D 7%. The team call where both sides say alive is right on 0.92; alive/dead F1 0.955.
- Two stoppages make every D>H frame: the 5.5 s one at 08:34 is missed outright (the file goes
  A>H 2 s early and bridges it), and the 14.5 s one at 12:01 starts 10.5 s late because an
  invented H segment (12.4 s, the window's one spurious segment) bridges the true A>D. The
  previous submission starts the same stoppage 10.3 s late with the same invented segment, so
  this is not the dead-ball model's doing.
- Four true segments are never shown: A 2.8 s at 07:42 (the file is 5 s late on the A>H before
  it and skips it), D 5.5 s at 08:34, H 1.3 s at 10:30, H 3.5 s at 11:31 (the file goes A>D
  2.2 s early). The previous submission misses three.
- Boundaries: 12 of 20 true transitions have a file transition within 5 s, 9 of them late,
  median +18 frames; 7 are within the scorer's 25 frames, 3 within 12. Repairing late and early
  frames alone would give +0.053 macroF1, the missed segments +0.051, the spurious one +0.055.
- Forgiving a margin around every true boundary: macroF1 0.874 at +-12 frames, 0.897 at +-25,
  0.929 at +-50, 0.940 at +-75. Outside the 3 s margin 237 wrong frames remain, from two
  places only: the 5 s late A>H at 07:34 and the 10.5 s late start of the 12:01 stoppage.
- Runs: median 33 frames, longest 263; 91% of the wrong frames sit in runs over 1 s, 74% over
  2 s. There is no flicker: the long true possessions (41.6, 28.6, 42.4, 35.7 s) are right on
  every interior frame, and the previous submission's two false stoppages inside them (D37) are
  gone.
Against the previous submission the frame view agrees with the spell view: fewer wrong frames
(-96), no spurious stoppages, fewer late frames, one more missed short segment (the 3.5 s H at
11:31, which the dead-ball model now ends 2.2 s early into the stoppage).

## Section 5: E3 touch engine and E6 snapping, simulator v1 (2026-10-30)

Setup: section 2 (simulator v1, IDSSE dev, seeds 0 1 2, same labels). Reference: `semi_markov`
(D8, D9), except the E6 table, whose reference is `touch_markov`. Code at branch
`miha/e3-touch-engine`, on top of `bc7c64c`. Variants were registered ad hoc through
`semi_markov(name, touch_params=..., snap=..., touch_w=...)`; only `touch_markov` is in
`METHODS`. **Settings tried: 19** (6 decoder, 5 engine, 4 around `touch_takes`, 4 snapping).
Engine defaults are `touch.PARAMS` at the time of the run, with `touch_takes` False in stages
A and B (then the default) and True from stage C on.

**Step 0, boundary report** (`make proxy METHODS="baseline_smooth50 semi_markov" BOUNDARIES=1`),
pooled per simulator pass:

| method | hand-over recall ±25 / ±75 | hand-over latency mean, sd (frames) | ball seen at true hand-overs | recall at IoU 0.5: <2 s / 2-5 s / 5-15 s / ≥15 s | count ratio | oracle snap ±37: hand-overs / all |
|---|---|---|---|---|---|---|
| semi_markov | 0.228 / 0.471 | -5.0, 13.0 | 0.79 | 0.027 / 0.198 / 0.479 / 0.631 | 0.893 | 0.3440 / 0.3544 (from 0.2993) |
| touch_markov | 0.424 / 0.570 | -1.4, 7.6 | 0.79 | 0.156 / 0.361 / 0.662 / 0.747 | 0.824 | 0.4951 / 0.5050 (from 0.4720) |

Engine and decoder grid (`make proxy`, plain v1):

| date | method | spellF1 | interval | macroF1 | possessions | gain vs ref | verdict | note |
|---|---|---|---|---|---|---|---|---|
| 2026-10-30 | semi_markov | 0.2993 | 0.2823..0.3093 | 0.7312 | 1956 | ref | | |
| 2026-10-30 | touch_w=0.5, hysteresis | 0.3478 | 0.3281..0.3587 | 0.7660 | 1563 | +0.0485 (+0.0381..+0.0553) | ADOPT | stage A |
| 2026-10-30 | touch_w=1, hysteresis | 0.3492 | 0.3277..0.3586 | 0.7683 | 1558 | +0.0499 (+0.0377..+0.0557) | ADOPT | stage A |
| 2026-10-30 | touch_w=2, hysteresis | 0.3532 | 0.3352..0.3667 | 0.7712 | 1550 | +0.0539 (+0.0448..+0.0641) | ADOPT | stage A |
| 2026-10-30 | touch_w=0.5, nearest_fallback | 0.3404 | 0.3201..0.3515 | 0.7621 | 1573 | +0.0410 (+0.0310..+0.0484) | ADOPT | stage A |
| 2026-10-30 | touch_w=1, nearest_fallback | 0.3467 | 0.3249..0.3562 | 0.7661 | 1562 | +0.0474 (+0.0348..+0.0526) | ADOPT | stage A |
| 2026-10-30 | touch_w=2, nearest_fallback | 0.3520 | 0.3318..0.3633 | 0.7705 | 1551 | +0.0527 (+0.0412..+0.0608) | ADOPT | stage A |
| 2026-10-30 | w=1, hysteresis, touch_dv=2 | 0.3513 | 0.3303..0.3620 | 0.7679 | 1568 | +0.0519 (+0.0405..+0.0581) | ADOPT | stage B |
| 2026-10-30 | w=1, hysteresis, touch_dv=4.5 | 0.3476 | 0.3255..0.3561 | 0.7671 | 1551 | +0.0483 (+0.0354..+0.0534) | ADOPT | stage B |
| 2026-10-30 | w=1, hysteresis, control_s=0.24 | 0.3572 | 0.3379..0.3688 | 0.7698 | 1566 | +0.0579 (+0.0470..+0.0650) | ADOPT | stage B |
| 2026-10-30 | w=1, hysteresis, control_s=0.8 | 0.3336 | 0.3110..0.3416 | 0.7620 | 1592 | +0.0343 (+0.0220..+0.0383) | ADOPT | stage B |
| 2026-10-30 | **touch_markov** (w=1, touch_takes) | 0.4720 | 0.4558..0.4883 | 0.7893 | 1806 | +0.1727 (+0.1627..+0.1887) | **ADOPT** | registered (D10) |
| 2026-10-30 | touch_takes, touch_w=2 | 0.4834 | 0.4695..0.5022 | 0.7943 | 1820 | +0.1841 (+0.1767..+0.2032) | ADOPT | +0.011 over touch_markov: below the 0.02 bar for a change |
| 2026-10-30 | touch_takes, control_s=0.24 | 0.4740 | 0.4590..0.4913 | 0.7894 | 1815 | +0.1747 (+0.1664..+0.1920) | ADOPT | stage C |
| 2026-10-30 | touch_takes, touch_dv=2 | 0.4780 | 0.4618..0.4935 | 0.7901 | 1856 | +0.1787 (+0.1684..+0.1944) | ADOPT | stage C |
| 2026-10-30 | touch_takes, touch_dv=4.5 | 0.4591 | 0.4402..0.4740 | 0.7867 | 1753 | +0.1598 (+0.1474..+0.1737) | ADOPT | stage C |

Robustness of `touch_markov` against `semi_markov` at the same stress. Ball noise is
`simulate.add_ball_noise`, opt-in and not a simulator version. sd 0.05 m with 3.4 jumps per
visible minute matches the BAR-ATH ball-speed median and jump rate in FINDINGS A2 (degraded
IDSSE p50 4.94 against 5.08, p90 15.15 against 14.31); the harsh setting doubles both.

| date | stress | semi_markov | touch_markov | interval | macroF1 | possessions | gain | verdict |
|---|---|---|---|---|---|---|---|---|
| 2026-10-30 | `BALL_NOISE=0.05 BALL_JUMPS=3.4` | 0.2988 | 0.4698 | 0.4534..0.4858 | 0.7886 | 1815 | +0.1710 (+0.1611..+0.1872) | ADOPT |
| 2026-10-30 | `BALL_NOISE=0.1 BALL_JUMPS=6.8` | 0.2985 | 0.4686 | 0.4528..0.4848 | 0.7875 | 1836 | +0.1701 (+0.1603..+0.1860) | ADOPT |
| 2026-10-30 | `NOISE=0.1` (players) | 0.2991 | 0.4760 | 0.4595..0.4942 | 0.7936 | 1790 | +0.1768 (+0.1664..+0.1927) | ADOPT |
| 2026-10-30 | `SPEED=0.9` | 0.2993 | 0.4678 | 0.4490..0.4811 | 0.7866 | 1809 | +0.1685 (+0.1564..+0.1820) | ADOPT |

E6 boundary snapping (`touch.snap_transitions`, ±1.5 s), reference `touch_markov`:

| date | method | spellF1 | interval | macroF1 | possessions | gain vs ref | verdict |
|---|---|---|---|---|---|---|---|
| 2026-10-30 | snap hand-overs, first touch | 0.4746 | 0.4578..0.4903 | 0.7896 | 1806 | +0.0025 (+0.0000..+0.0042) | within noise |
| 2026-10-30 | snap hand-overs, nearest touch | 0.4746 | 0.4582..0.4905 | 0.7896 | 1806 | +0.0026 (+0.0005..+0.0044) | better, below the adoption bar |
| 2026-10-30 | snap hand-overs + restarts, first | 0.4774 | 0.4606..0.4939 | 0.7896 | 1806 | +0.0054 (+0.0029..+0.0086) | better, below the adoption bar |
| 2026-10-30 | snap hand-overs + restarts, nearest | 0.4757 | 0.4591..0.4918 | 0.7880 | 1806 | +0.0037 (+0.0011..+0.0067) | better, below the adoption bar |

Takeaways:
- **The loss was missed short possessions and missed hand-overs, not lag.** Before E3, 60% of
  true possessions last under 5 s and 11% of them were matched; matched hand-overs were only
  5 frames early on average.
- **Hysteresis is the wrong semantics for DFL labels.** It gains +0.05; handing the ball over at
  the other team's first touch gains +0.17. This is a label-semantics result as much as a
  method result (D10).
- **The engine is insensitive to its thresholds** once a touch hands over (0.459 to 0.478 over
  stage C) and to ball noise (-0.002 and -0.003 at the BAR-ATH-like and harsh levels).
- **E6 is not worth it after E3.** The touch engine already places hand-overs (sd 7.6 frames);
  perfect snapping would add +0.023, the real one +0.003 to +0.005.
- `tools/touch_check.py` (IDSSE dev, degraded v1): 28.8 touches and 13.5 control episodes per
  clean-ball minute; a team holds 0.439 of graded frames; `touch_markov` decodes 341
  possessions per 90 min against the truth's 414 (`semi_markov` 370). The same figures under
  both ball-noise levels move by at most 2.2 touches per minute and 6 possessions per 90.

BAR-ATH, `tools/touch_check.py --bar-ath`, run by miha. Label-free aggregates of the ATD, next to
the same tool on IDSSE dev degraded v1; nothing here is scored against BAR-ATH truth.

| | touches / clean-ball min | controls / clean-ball min (median s) | cleaning keeps | held share (home) | holder changes / 90 | semi_markov poss / 90 | touch_markov poss / 90 |
|---|---|---|---|---|---|---|---|
| IDSSE degraded v1 | 28.8 | 13.5 (0.88) | 0.836 | 0.439 (0.505) | 251 | 370 | 341 |
| BAR-ATH | **2.3** | 9.7 (0.96) | 0.956 | 0.403 (0.621) | 167 | 425 | 336 |

**The touch candidates almost never fire on BAR-ATH: 2.3 per minute against 28.8.** Ball speed
matches (p50 / p90 / p99 5.08 / 14.31 / 24.60 m/s; 3.44 jumps per visible minute, the stress's
target). So the +0.17 of `touch_markov`, which needs touches to hand over the ball (D10), very
likely does not transfer: on BAR-ATH the holder comes from control episodes alone, closer to the
hysteresis variant (+0.03 to +0.06 on the proxy). The ball-noise stress did not catch this,
because white jitter adds touch candidates rather than removing them.
`tools/touch_check.py --diagnose` separates a time-smoothed ball from velocity changes far from
tracked players. On IDSSE dev degraded v1: dv >= 3 m/s peaks at lag 0.12 / 0.24 / 0.40 s are
33.4 / 35.9 / 40.2 per clean minute, 0.868 of them with a player within 2 m at lag 0.12 s;
lag-1 autocorrelation of frame acceleration -0.10.

## Section 6: E3 on a smoothed ball, simulator v1 plus the calibrated ball stress (2026-10-30)

Setup: section 2, plus `make proxy BALL_NOISE=0.05 BALL_JUMPS=3.4 BALL_SMOOTH=0.26` unless a row
says otherwise. Numbers are not comparable with section 5. Code at branch
`miha/e3-touch-engine`. **Settings tried here: 11** (30 for E3 in total), plus the stress
calibration below.

**Why.** `tools/touch_check.py --diagnose --bar-ath` (run by miha) shows the BAR-ATH ball is
smoothed by its tracker:

| | lag-1 autocorr. of acceleration | dv p90 / p99 at lag 0.12 s | at 0.24 s | at 0.40 s | dv >= 3 peaks / min at 0.12 / 0.24 / 0.40 s | player within 2 m at peaks, 0.40 s | touches / min (old detector) | controls / min (old) |
|---|---|---|---|---|---|---|---|---|
| IDSSE degraded v1 | -0.10 | 2.93 / 10.75 | 5.38 / 15.50 | 8.00 / 17.76 | 33.4 / 35.9 / 40.2 | 0.723 | 28.8 | 13.5 |
| BAR-ATH | +0.99 | 2.16 / 4.14 | 4.19 / 7.65 | 6.61 / 11.76 | 6.6 / 20.6 / 27.9 | 0.609 | 2.3 | 9.7 |
| IDSSE v1 + stress, smoothing 0.24 s | +0.52 | 2.14 / 4.32 | 4.07 / 7.94 | 6.27 / 11.71 | 10.3 / 21.6 / 27.8 | 0.662 | 3.5 | 10.7 |
| IDSSE v1 + stress, smoothing 0.28 s | +0.50 | 1.96 / 3.81 | 3.75 / 7.06 | 5.84 / 10.65 | 7.2 / 19.1 / 25.9 | 0.581 | 1.5 | 10.2 |

Smoothing sd was calibrated on the dv percentiles over 0.08 to 0.40 s (seed 0; jitter 0, 0.05,
0.1 m made no difference). The touch and control rates were not fitted and land either side of
BAR-ATH, which checks the calibration independently. The acceleration autocorrelation is not
matched (+0.5 against +0.99); DFL positions are rounded, which limits it.

Detector grid under the stress (reference: `touch_markov` with section 5's engine, lag 0.12 s,
radius 2 m):

| date | method | spellF1 | interval | macroF1 | possessions | gain vs ref | verdict |
|---|---|---|---|---|---|---|---|
| 2026-10-30 | semi_markov | 0.2866 | 0.2685..0.2953 | 0.7253 | 2006 | -0.0240 (the ref gains +0.0240, +0.0122..+0.0325, ADOPT) | |
| 2026-10-30 | touch_markov, section 5 engine | 0.3105 | 0.2895..0.3193 | 0.7491 | 1653 | ref | |
| 2026-10-30 | hysteresis, section 5 engine | 0.3106 | 0.2905..0.3201 | 0.7516 | 1604 | +0.0001 (-0.0045..+0.0064) | within noise |
| 2026-10-30 | lag 0.24, dv 2 | 0.3428 | 0.3267..0.3569 | 0.7559 | 1820 | +0.0323 (+0.0279..+0.0469) | ADOPT |
| 2026-10-30 | lag 0.4, dv 3, window 0.4 | 0.3379 | 0.3179..0.3483 | 0.7551 | 1778 | +0.0273 (+0.0193..+0.0377) | ADOPT |
| 2026-10-30 | lag 0.4, dv 2, window 0.4 | 0.3466 | 0.3259..0.3561 | 0.7567 | 1835 | +0.0360 (+0.0260..+0.0456) | ADOPT |
| 2026-10-30 | **lag 0.4, dv 3, window 0.4, radius 3 m** | 0.3566 | 0.3409..0.3714 | 0.7592 | 1906 | +0.0461 (+0.0410..+0.0626) | **ADOPT**, registered (D12) |
| 2026-10-30 | lag 0.4, dv 3, window 0.4, engine smoothing 0.04 s | 0.3423 | 0.3244..0.3544 | 0.7556 | 1780 | +0.0318 (+0.0247..+0.0441) | ADOPT |
| 2026-10-30 | lag 0.4, dv 2, window 0.4, radius 3 m | 0.3619 | 0.3463..0.3763 | 0.7593 | 2006 | +0.0514 (+0.0460..+0.0682) | ADOPT; +0.005 over registered, noise |
| 2026-10-30 | lag 0.4, dv 3, window 0.4, radius 4 m | 0.3603 | 0.3459..0.3759 | 0.7585 | 1956 | +0.0497 (+0.0447..+0.0677) | ADOPT; plateau |
| 2026-10-30 | registered, control radius 2.5 m | 0.3554 | 0.3398..0.3697 | 0.7583 | 1931 | +0.0449 (+0.0395..+0.0614) | ADOPT; plateau |
| 2026-10-30 | registered, hysteresis | 0.3211 | 0.3037..0.3332 | 0.7525 | 1642 | -0.0355 (-0.0473..-0.0268) | WORSE |

Registered `touch_markov` (D12 engine) against `semi_markov` at the same stress:

| date | stress | semi_markov | touch_markov | interval | macroF1 | possessions | gain | verdict |
|---|---|---|---|---|---|---|---|---|
| 2026-10-30 | ball 0.05 m, 3.4 jumps, smoothing 0.26 s (reference) | 0.2866 | 0.3566 | 0.3409..0.3714 | 0.7592 | 1906 | +0.0700 (+0.0617..+0.0849) | ADOPT |
| 2026-10-30 | ... smoothing 0.20 s | 0.2933 | 0.3958 | 0.3821..0.4141 | 0.7693 | 1862 | +0.1025 (+0.0958..+0.1208) | ADOPT |
| 2026-10-30 | ... smoothing 0.32 s | 0.2836 | 0.3260 | 0.3091..0.3370 | 0.7479 | 1974 | +0.0424 (+0.0307..+0.0529) | ADOPT |
| 2026-10-30 | ball 0.1 m, 6.8 jumps, smoothing 0.26 s | 0.2866 | 0.3562 | 0.3401..0.3702 | 0.7588 | 1906 | +0.0696 (+0.0611..+0.0832) | ADOPT |
| 2026-10-30 | reference + `NOISE=0.1` | 0.2877 | 0.3548 | 0.3402..0.3704 | 0.7603 | 1902 | +0.0671 (+0.0573..+0.0810) | ADOPT |
| 2026-10-30 | reference + `SPEED=0.9` | 0.2865 | 0.3591 | 0.3427..0.3726 | 0.7573 | 1930 | +0.0725 (+0.0639..+0.0861) | ADOPT |
| 2026-10-30 | none (plain v1, clean ball) | 0.2993 | 0.4498 | 0.4331..0.4660 | 0.7825 | 1919 | +0.1505 (+0.1410..+0.1659) | ADOPT |

Takeaways:
- **Section 5's +0.17 was mostly a clean-ball artifact.** On the calibrated ball the section 5
  engine gains only +0.024 over `semi_markov`.
- **A detector built for a smoothed ball recovers +0.070**: a 0.4 s lag and a 3 m radius,
  because the smoothed ball sits a little off the feet. Threshold and radius are on a plateau
  (0.355 to 0.362).
- **The size of the gain depends on the real smoothing:** +0.10 at 0.20 s, +0.04 at 0.32 s. It
  is ADOPT throughout.
- **A touch still hands over the ball:** waiting for control loses 0.036 (D10's bet stands).
- Boundary report under the stress: hand-over latency sd 10.6 frames; E6 ceiling +0.031; recall
  of possessions under 2 s 0.075.
- `tools/touch_check.py` with the D12 engine, IDSSE v1 + stress: 21.6 touches and 9.1 controls
  per clean-ball minute, 360 possessions per 90 (truth 414). BAR-ATH should look similar.

BAR-ATH check of the D12 engine, `tools/touch_check.py --bar-ath`, run by miha on 2026-11-02:

| | touches / clean-ball min | controls / clean-ball min (median s) | held share (home) | holder changes / 90 | semi_markov poss / 90 | touch_markov poss / 90 |
|---|---|---|---|---|---|---|
| IDSSE v1 + D12 ball stress | 21.6 | 9.1 (1.08) | 0.434 (0.507) | 262 | 379 | 360 |
| BAR-ATH | **22.0** | **9.1** (1.00) | 0.500 (0.606) | 341 | 425 | 426 |

**The calibration holds:** the event rates on BAR-ATH match the stressed proxy, though neither
was fitted. On BAR-ATH a team holds the ball on more of the match, and the holder changes 30%
more often. `touch_markov` decodes as many possessions there as `semi_markov`, where it decodes
5% fewer on the proxy.

## Section 7: kickoff grammar, simulator v1 + D12 ball stress + synthetic pre-kickoff (2026-11-04)

Setup changes from section 6: with `PREKICKOFF=117`, `simulate.add_prekickoff` inserts 117 s of
synthetic dead ball in front of each second half before degrading. Inside that stretch the ball
is hidden with v0's BAR-ATH stretches instead of v1's (v1 hid ~74% of it; BAR-ATH sees 81%).
Two invented profiles: `calm` (players settle early and stand, the ball rests on the mark) and
`busy` (players mill about at 1 to 2 m/s and pass until the last seconds). Neither is
calibrated to BAR-ATH. Reference `touch_markov`; all rows under
`BALL_NOISE=0.05 BALL_JUMPS=3.4 BALL_SMOOTH=0.26`.

Step 0: IDSSE positions start at the kick in every dev half (ball alive within 0 to 48 frames),
so there is nothing before kickoff to test on. At the kick the ball is 0.2 to 10 m from the mark,
and 0.64 to 0.91 of each team stands strictly in its own half (they line up on the line).

`touch_kickoff` (`solution/kickoff.py`, `kickoff.PARAMS`) against `touch_markov`:

| date | pre-kickoff | touch_markov | touch_kickoff | interval | macroF1 | possessions | gain | verdict |
|---|---|---|---|---|---|---|---|---|
| 2026-11-04 | none (plain) | 0.3566 | 0.3566 | 0.3409..0.3714 | 0.7592 | 1906 | +0.0000 (+0.0000..+0.0000) | within noise; identical, never fires |
| 2026-11-04 | 117 s calm, decode from the kick | 0.3587 | 0.3586 | 0.3397..0.3700 | 0.7608 | 1886 | -0.0002 (-0.0021..-0.0000) | WORSE: block grid shifted |
| 2026-11-04 | 117 s busy, decode from the kick | 0.3567 | 0.3575 | 0.3385..0.3687 | 0.7575 | 1906 | +0.0008 (-0.0005..+0.0016) | within noise |
| 2026-11-04 | **117 s calm, decode on the half's block grid** | 0.3587 | 0.3588 | 0.3406..0.3711 | 0.7607 | 1887 | +0.0001 (-0.0004..+0.0002) | within noise |
| 2026-11-04 | **117 s busy, decode on the half's block grid** | 0.3567 | 0.3579 | 0.3391..0.3697 | 0.7575 | 1907 | +0.0011 (+0.0009..+0.0015) | better, below the adoption bar; wins 4/5 |

Detection, `tools/kickoff_check.py` (5 dev matches x 3 seeds, same stress):

| pre-kickoff | false kickoffs | kickoff-like moments (no guards) | found | kick error s, median (range) | touch_markov on the true pre-kickoff |
|---|---|---|---|---|---|
| none (plain) | 0/30 halves | 0 | -- | -- | -- |
| 117 s calm | 0/30 | 10 | 10/15 | -0.16 (-14.60..+0.04) | D share 1.000, 0 possessions; speed 0.43 m/s |
| 117 s busy | 0/30 | 9 | 9/15 | -0.16 (-5.92..+0.04) | D share 0.387, about 1 possession; speed 1.50 m/s |

Grid, busy profile: `spot_m` 1.5/3.0 x `own_share` 0.7/0.9 x `quiet_ms` 1.2/2.0. All 8 gave
9/15 found, the same errors and 0 false kickoffs. `spot_m` 3.0 shows one kickoff-like moment on
the plain halves (none at 1.5 or 2.0). Registered values are mid-grid and round. Settings tried
in total: 8 grid settings plus 4 detector revisions (gap tolerance, sightings for leave/away,
`gap_s` 20 to 60 s, the `still_m` refinement), each driven by a miss on the calm synthetic.

Takeaways:
- **The rule is safe:** it is byte-identical on the plain proxy, and it never fired in 90 halves
  without a pre-kickoff. Kicks are placed early (safe), at worst +0.04 s late.
- **D4 cannot be met by this rule.** It fixes about one possession per half that has a
  pre-kickoff, out of about 2,191, so even perfect detection is worth a few thousandths of
  spellF1 here. On BAR-ATH (about 300 possessions, one long false possession that can also
  swallow the first real one) it is worth more, but still well under 0.02.
- **Whether touch_markov falls into the trap depends on how players behave:** none when they
  stand, 61% of frames as a team when they mill about. BAR-ATH decides which; see
  `kickoff_check --bar-ath`.
- Misses are halves where the ball is never seen resting on the mark (v0 stretches hide it), plus
  J03WOH, whose own second half opens with a misplaced ball and 2 s of dead ball.

BAR-ATH, `tools/kickoff_check.py --bar-ath`, run by miha on 2026-11-04 (label-free counts):

| half | kickoff found | candidates | set | formation | leave | away | quiet | rise | before |
|---|---|---|---|---|---|---|---|---|---|
| 1 | none in 300 s | 3 | 0 | 0 | 2 | 2 | 0 | 2 | 0 |
| 2 | none in 300 s | 1 | 0 | 0 | 1 | 1 | 1 | 0 | 1 |

**The rule does nothing on BAR-ATH yet.** No candidate is ever seen resting on the mark with
both teams in formation. The synthetic halves pass both cues, so BAR-ATH differs from them in
the ball near the mark, the tracked players, or the formation geometry. The next step is
`--diagnose --bar-ath`, which prints the same window aggregates for both.

`--diagnose --bar-ath` (miha, 2026-11-04), first 300 s of each half, against the synthetic
true pre-kickoff (calm / busy, means over 15 halves):

| | ball seen | sightings within 2 / 5 m of the mark | ball-to-mark p1 | longest run of sightings within 5 m | both teams >= 6 tracked | teams on opposite sides | formation holds | speed p50 |
|---|---|---|---|---|---|---|---|---|
| BAR-ATH half 1 | 0.882 | 0.005 / 0.011 | 4.5 m | 1.9 s | 1.000 | 0.162 | 0.004 | 2.02 |
| BAR-ATH half 2 | 0.878 | 0.000 / 0.009 | 5.1 m | 1.2 s | 0.890 | 0.511 | 0.101 | 1.65 |
| synthetic calm, pre-kickoff | 0.492 | 0.216 / 0.235 | 0.8 m | 13.5 s | 0.952 | 0.940 | 0.578 | 0.43 |
| synthetic busy, pre-kickoff | 0.492 | 0.121 / 0.143 | 1.4 m | 7.7 s | 0.952 | 0.952 | 0.474 | 1.50 |

**The BAR-ATH ball is never seen resting on the centre mark.** Not one sighting comes within
2 m in half 2's first 300 s, though the kickoff is in it and the ball is seen on 88% of frames.
Coordinates are normalised to the pitch, and only 4.2% of the ball is beyond a line (FINDINGS),
so the centre is not misplaced. This fits FINDINGS' "almost never still" ball: the tracker does
not report a resting ball where it rests. The ball cue of `kickoff.py` cannot work on BAR-ATH.
The formation does hold for about 30 s of half 2's window and about 1 s of half 1's. Half 2 has
extra tracks before kickoff (p95 15 and 12 per team).

### Players-only kickoff rule (2026-11-04)

The ball cue was dropped (BAR-ATH has no ball on the mark). `kickoff.py` now reads a formation
hold of at least `hold_s`, quiet play, a speed rise and the `before` guard, and places the kick
`back_s` before play speeds up. Same setup and stress as above.

`--diagnose --bar-ath`, formation-hold line (miha, 2026-11-04): in half 2 the longest hold is
23.5 s and ends at 119.1 s from the first graded frame (2 holds >= 2 s). Speed is 1.27 m/s in
it, 1.46 m/s over its last 5 s and 2.30 m/s over the next 5 s; the ball is seen on 0.532 of it, a
median 24.1 m from the mark. In half 1 the longest hold is the opening 1.3 s (speed 1.37 to
3.61 m/s), which fits a half that starts at its kick.

| date | pre-kickoff | touch_markov | touch_kickoff | interval | macroF1 | possessions | gain | verdict |
|---|---|---|---|---|---|---|---|---|
| 2026-11-04 | none (plain), `hold_s` 5, onset in last 3 s | 0.3566 | 0.3566 | 0.3409..0.3714 | 0.7592 | 1906 | +0.0000 (+0.0000..+0.0000) | within noise; identical |
| 2026-11-04 | 117 s calm, same | 0.3587 | 0.3591 | 0.3402..0.3710 | 0.7611 | 1887 | +0.0004 (-0.0010..+0.0003) | within noise |
| 2026-11-04 | 117 s busy, same | 0.3567 | 0.3583 | 0.3393..0.3700 | 0.7610 | 1903 | +0.0015 (+0.0008..+0.0021) | better, below the adoption bar; kicks late |
| 2026-11-04 | **none (plain), registered: `hold_s` 8, `back_s` 2** | 0.3566 | 0.3566 | 0.3409..0.3714 | 0.7592 | 1906 | +0.0000 (+0.0000..+0.0000) | within noise; identical |
| 2026-11-04 | **117 s calm, registered** | 0.3587 | 0.3589 | 0.3401..0.3708 | 0.7609 | 1887 | +0.0002 (-0.0011..+0.0003) | within noise |
| 2026-11-04 | **117 s busy, registered** | 0.3567 | 0.3582 | 0.3392..0.3699 | 0.7610 | 1903 | +0.0015 (+0.0006..+0.0021) | better, below the adoption bar; wins 5/5 |

Detection, registered setting (`tools/kickoff_check.py`, 5 dev matches x 3 seeds):

| pre-kickoff | false kickoffs | kickoff-like holds (no guard) | found | kick error s, median (range) |
|---|---|---|---|---|
| none (plain) | 0/30 halves | 35 | -- | -- |
| 117 s calm | 0/30 | 49 | 14/15 | -1.54 (-2.16..+2.44) |
| 117 s busy | 0/30 | 49 | 14/15 | -1.08 (-1.68..+2.92) |

- **The first players-only version placed kicks late** (busy median +0.92 s, up to +4.9 s): the
  speed onset trails the kick by 0.3 to 1.8 s, as IDSSE shows (1.4 to 2.5 m/s in the 2 s after a
  kick). The +2.4 to +2.9 s tail of the registered setting is the J03WOH half, whose own labels
  stay D for 2 s after the kick.
- **Formation holds that pass quiet and rise are common in open play** (35 on the plain halves).
  Only the `before` guard keeps false kickoffs at 0; a first grid at `hold_s` 3 and
  `own_share` 0.7 let 2 through. Hence `hold_s` 8.
- Grids, busy profile: `own_share` 0.7/0.9 x `hold_s` 3/8 x `quiet_ms` 1.5/2.5 (first version),
  then `own_share` 0.7/0.9 x `hold_s` 5/12 x `back_s` 1/3 (registered). The second grid had 0
  false kickoffs everywhere and found 12 to 14 of 15. `back_s` moves the error median from
  -0.04 s (1 s) to -2.1 s (3 s); 2 s is between. Settings tried for the players-only rule: 16
  grid + 3 revisions.

BAR-ATH, `tools/kickoff_check.py --bar-ath` with the registered players-only rule (miha,
2026-11-04), label-free:

| half | kickoff | candidates (quiet / rise / before) | before it: touch_markov D share | possessions | ball seen | median collective speed |
|---|---|---|---|---|---|---|
| 1 | none in 300 s | 0 | -- | -- | -- | -- |
| 2 | 115.4 s from the first graded frame | 1 (1 / 1 / 1) | 0.537 | 3 | 0.804 | 1.16 m/s |

- **Found where it should be, on the early side.** The published kickoff (PLAN finding 9, used
  here only as a check, never in code) is about 117.4 s in, so the rule is 2 s early, as
  `back_s` intends. Half 1 starts at its kick, so none there is correct.
- **The trap is real for `touch_markov`:** about 46% of the pre-kickoff stretch is labelled a
  team, in 3 false possessions. The rule turns all of it into D.
- **The synthetic profiles bracket BAR-ATH** on both measures: speed 1.16 m/s against 0.43
  (calm) and 1.50 (busy), and D share 0.54 against 1.00 and 0.39. So the proxy's +0.0002 to
  +0.0015 is a fair guide to the size of the effect. The ball-seen share matches finding 9 (81%).

## Section 8: C2b dead-ball cues, simulator v1 + D12 ball stress (2026-11-04)

Setup: section 6 (5 dev matches x 3 seeds, simulator v1, `BALL_NOISE=0.05 BALL_JUMPS=3.4
BALL_SMOOTH=0.26`) unless a row says otherwise. Reference `touch_kickoff` (D14). New here:
`simulate.thin_beyond` (`BEYOND_KEEP`, `BEYOND_MARGIN`) hides each stretch of ball beyond a
line with probability 1 - keep, so a ball-beyond cue gets BAR-ATH's rate (D7). Code at branch
`miha/dev`. **Settings tried: 12** (4 speed leads, 3 player-cue weights, 3 ball-cue
variants under 3 thinning conditions, 2 still-ball probes).

Step 0, `tools/deadball_check.py` (true D stretches matched to DFL restart events, ±6 s, from
`idsse.restarts`; oracle repairs from the truth that never fix possession):

| ball | as is | perfect D | start edges | end edges | false stoppages | missed | throw-in | free kick | goal kick | corner | goal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| D12 stress | 0.3566 | +0.0891 | +0.0373 | +0.0184 | +0.0197 | +0.0127 | +0.0281 | +0.0163 | +0.0135 | +0.0054 | +0.0016 |
| plain | 0.4498 | +0.0947 | +0.0399 | +0.0178 | +0.0201 | +0.0139 | +0.0298 | +0.0178 | +0.0150 | +0.0064 | +0.0016 |

Under the stress: 514 true stretches per seed, 10.7% missed (67% of those under 2 s, 50% of
2-5 s); found ones start a median +0.88 s late (p90 +8.8 s) and end +0.72 s late (p90 +4.0 s);
123 false stoppages per seed (median 4.4 s); D frames recall 0.820, precision 0.875.
**The ceiling is far above 0.02, so C2b went ahead.**

Where cues fire, share of frames (seed 0, stressed proxy, `touch_kickoff`):

| cue | true D, decoded D | true D, decoded alive | true alive, decoded D | true alive, decoded alive | all graded |
|---|---|---|---|---|---|
| ball still (< 0.5 m/s over 1 s) | 0.213 | 0.608 | 0.016 | 0.010 | 0.131 |
| ball beyond a line by > 0.5 m | 0.089 | 0.244 | 0.001 | 0.001 | 0.052 |
| player beyond a touchline by > 0.5 m, < 2 m/s | 0.155 | 0.101 | 0.018 | 0.007 | 0.069 |
| player beyond a goal line by > 0.5 m, < 2 m/s | 0.073 | 0.075 | 0.004 | 0.003 | 0.034 |
| over half the players under 1 m/s | 0.611 | 0.007 | 0.343 | 0.009 | 0.244 |

Proxy against BAR-ATH (FINDINGS A2): still ball 13.1% of graded frames against 2.0%; ball beyond
a line 16.2% of visible frames against 4.2%, beyond by > 1 m 3.3% against 2.7%.

Proxy league against `touch_kickoff` (0.3566 unless the row's stress changes it):

| date | method | stress | spellF1 | interval | macroF1 | possessions | gain | verdict |
|---|---|---|---|---|---|---|---|---|
| 2026-11-04 | speed lead 0.4 s | D12 | 0.3568 | 0.3417..0.3712 | 0.7623 | 1908 | +0.0003 (-0.0024..+0.0033) | within noise |
| 2026-11-04 | speed lead 0.8 s | D12 | 0.3583 | 0.3451..0.3745 | 0.7637 | 1903 | +0.0018 (-0.0003..+0.0074) | within noise |
| 2026-11-04 | speed lead 1.2 s | D12 | 0.3575 | 0.3422..0.3721 | 0.7627 | 1895 | +0.0009 (-0.0036..+0.0054) | within noise |
| 2026-11-04 | speed lead 1.6 s | D12 | 0.3551 | 0.3394..0.3687 | 0.7615 | 1886 | -0.0015 (-0.0076..+0.0029) | within noise |
| 2026-11-04 | player cue, w 0.5 | D12 | 0.3618 | 0.3461..0.3763 | 0.7663 | 1916 | +0.0052 (+0.0022..+0.0084) | better, below the adoption bar |
| 2026-11-04 | player cue, w 1 | D12 | 0.3612 | 0.3460..0.3762 | 0.7677 | 1926 | +0.0046 (+0.0005..+0.0094) | better, below the adoption bar |
| 2026-11-04 | player cue, w 2 | D12 | 0.3583 | 0.3427..0.3723 | 0.7626 | 1943 | +0.0017 (-0.0052..+0.0072) | within noise |
| 2026-11-04 | **player + ball > 1 m, w 0.5 (`touch_deadball`)** | D12 | 0.3650 | 0.3483..0.3783 | 0.7719 | 1900 | +0.0084 (+0.0034..+0.0111) | better, below the adoption bar; wins 5/5 |
| 2026-11-04 | player + ball, w 1 | D12 | 0.3651 | 0.3495..0.3795 | 0.7716 | 1904 | +0.0085 (+0.0035..+0.0134) | better, below the adoption bar |
| 2026-11-04 | ball > 1 m, w 0.5 | D12 | 0.3600 | 0.3436..0.3739 | 0.7645 | 1893 | +0.0034 (+0.0006..+0.0048) | better, below the adoption bar |
| 2026-11-04 | player cue, w 0.5 | D12 + thin 0.82 beyond 1 m (ref 0.3568) | 0.3621 | 0.3465..0.3766 | 0.7665 | 1912 | +0.0053 (+0.0022..+0.0083) | better, below the adoption bar |
| 2026-11-04 | player + ball, w 0.5 | D12 + thin 0.82 beyond 1 m | 0.3649 | 0.3480..0.3780 | 0.7712 | 1900 | +0.0081 (+0.0027..+0.0104) | better, below the adoption bar |
| 2026-11-04 | player + ball, w 1 | D12 + thin 0.82 beyond 1 m | 0.3645 | 0.3491..0.3790 | 0.7707 | 1905 | +0.0077 (+0.0025..+0.0124) | better, below the adoption bar |
| 2026-11-04 | ball > 1 m, w 0.5 | D12 + thin 0.82 beyond 1 m | 0.3599 | 0.3435..0.3737 | 0.7640 | 1893 | +0.0031 (+0.0000..+0.0042) | better, below the adoption bar |
| 2026-11-04 | player cue, w 0.5 | D12 + thin 0.26 beyond 0 m (ref 0.3599) | 0.3648 | 0.3478..0.3789 | 0.7703 | 1848 | +0.0049 (+0.0014..+0.0078) | better, below the adoption bar |
| 2026-11-04 | player + ball, w 0.5 | D12 + thin 0.26 beyond 0 m | 0.3656 | 0.3470..0.3781 | 0.7706 | 1843 | +0.0057 (+0.0002..+0.0077) | better, below the adoption bar |
| 2026-11-04 | player + ball, w 1 | D12 + thin 0.26 beyond 0 m | 0.3633 | 0.3476..0.3778 | 0.7696 | 1850 | +0.0034 (-0.0011..+0.0089) | within noise |
| 2026-11-04 | ball > 1 m, w 0.5 | D12 + thin 0.26 beyond 0 m | 0.3598 | 0.3422..0.3730 | 0.7634 | 1834 | -0.0001 (-0.0031..+0.0007) | within noise |
| 2026-11-04 | still ball, w 0.5 (upper bound, unfair) | D12 | 0.3681 | 0.3580..0.3892 | 0.7533 | 1904 | +0.0115 (+0.0092..+0.0254) | better, below the adoption bar |
| 2026-11-04 | player + ball + still, w 0.5 (upper bound, unfair) | D12 | 0.3747 | 0.3648..0.3960 | 0.7649 | 1909 | +0.0182 (+0.0157..+0.0320) | better, below the adoption bar |

`touch_deadball` robustness against `touch_kickoff`:

| date | stress | touch_kickoff | touch_deadball | interval | macroF1 | possessions | gain | verdict |
|---|---|---|---|---|---|---|---|---|
| 2026-11-04 | none (plain v1) | 0.4498 | 0.4590 | 0.4394..0.4732 | 0.7958 | 1902 | +0.0092 (+0.0031..+0.0100) | better, below the adoption bar; wins 5/5 |
| 2026-11-04 | D12 + 117 s busy pre-kickoff | 0.3582 | 0.3656 | 0.3462..0.3781 | 0.7726 | 1893 | +0.0073 (+0.0037..+0.0110) | better, below the adoption bar; wins 5/5 |

Step 0 rerun on `touch_deadball` (D12): missed 0.090 (from 0.107), false stoppages 106 per seed
(from 123), D recall 0.838, precision 0.894; perfect-D ceiling still +0.080. Two runs with the
priors refitted give identical codes.

`tools/deadball_check.py --proxy-side` (label-free aggregates, the comparison for `--bar-ath`):
D share 0.413 for both methods (truth 0.441); 111.6 and 110.8 stoppages per 90 (truth 97.1);
player cue on 0.098 of graded frames; ball beyond by > 1 m on 0.033 of visible frames. A
decoded restart within ±2 s of a DFL restart event: 0.37 (`touch_kickoff`), 0.39
(`touch_deadball`), 0.65 for the truth itself, since the event feed's clock is off by up to
about 2 s.

BAR-ATH, `tools/deadball_check.py --bar-ath --methods touch_kickoff touch_deadball`, run by miha
on 2026-11-04 (label-free aggregates):

| | touch_kickoff | touch_deadball | stressed proxy (kickoff / deadball) |
|---|---|---|---|
| D share of graded frames | 0.419 | 0.418 | 0.413 / 0.413 (truth 0.441) |
| stoppages per 90, median s | 119.8, 13.6 | 119.0, 13.1 | 111.6, 13.0 / 110.8, 13.6 |
| frames turned to D / away from D | -- | 0.0132 / 0.0133 | 0.0156 / 0.0153 |
| Smart Tagging set pieces with a decoded restart inside (±2 s) | 0.90 of 58 | 0.91 of 58 | -- |
| ... throw-ins / goal kicks / corners | 0.88 / 0.80 / 1.00 | 0.92 / 0.80 / 1.00 | -- |

Cue rates on BAR-ATH: player beyond a line on 0.0685 of graded frames (proxy 0.098); ball beyond
a line by > 1 m on 0.0268 of visible frames (proxy 0.033), beyond at all 0.0419 (proxy 0.162).
- **No warning sign.** The D share sits inside the IDSSE range (D9 holds). The player cue fires at
  0.7 times the proxy's rate, so linesmen are not tracked as players, and the gain there is
  likely a little under the proxy's. The ball rates reproduce FINDINGS A2 (2.7%, 4.2%), which
  confirms both the tool and `thin_beyond`'s calibration.
- **The change is small and balanced:** about 1.3% of frames each way, D share unchanged, one
  more throw-in window with a restart. D15 stands: registered, not submitted.

Takeaways:
- **The dead-ball ceiling is large (+0.089) but the fair cues reach only +0.005 to +0.008.**
  Where the decoder misses dead ball, players are moving; what shows the stoppage there is a
  still or out-of-play ball, and DFL's frozen dead ball overstates both.
- **Reading speed earlier does not help:** the edge errors are not a fixed reaction lag.
- **A player standing beyond a line is a fair cue and small:** +0.005, holding under every
  thinning. The ball beyond 1 m adds +0.003, about nothing once thinned to BAR-ATH's 4.2%.
- **Even the unfair still ball, with everything else, stays under 0.02** (+0.018).

---

## Section 9: E5 team cues for ball gaps, simulator v1 + D12 ball stress (2026-11-04)

Setup: section 8 (5 dev matches x 3 seeds, simulator v1, `BALL_NOISE=0.05 BALL_JUMPS=3.4
BALL_SMOOTH=0.26`) unless a row says otherwise. Reference `touch_kickoff` (D14). New here:
`tools/gap_check.py` (Step 0 ceilings, `--cues` aggregates) and `solution/gapteam.py` (player-only
team cues, decoder term `team_w`). Code at branch `miha/dev`. **Settings tried: 17**
(each of 4 cues alone, spread at 4 weights x 2 places x 2 dead-ball modes, width against
gyration, 2 combinations x 2 places).

Step 0, `tools/gap_check.py`: a ball gap is a run of frames without a seen ball (40.1% of graded
frames, 684 per seed, median 10.6 s). The oracle gives the true team on gap frames where truth
and `touch_kickoff` are both alive, and leaves dead ball as predicted. Pooled spellF1 gain:

| ball | as is | every gap | <2 s | 2-10 s | 10-30 s | >30 s | team changes | no change | change at a restart | change in play |
|---|---|---|---|---|---|---|---|---|---|---|
| D12 stress | 0.3566 | +0.0949 | -0.0010 | +0.0206 | +0.0262 | +0.0554 | +0.0950 | +0.0012 | +0.0011 | +0.0947 |
| plain | 0.4498 | +0.0879 | -0.0009 | +0.0181 | +0.0248 | +0.0512 | +0.0885 | +0.0003 | -- | -- |

Under the stress, only 28% of gap frames are truly alive (simulator v1 puts gaps in slow play).
`touch_kickoff`'s team accuracy on both-alive gap frames: 0.664 overall, 0.871 in gaps with no
change, 0.566 in gaps that hold an open-play hand-over. Possessions under 2 s: recall 0.075 as is,
0.252 with the repair. Of 654 per seed, 186 lie mostly inside a gap: recall 0.002 there, 0.590
repaired. **The ceiling is far above 0.02, so E5 went ahead.**

Cue aggregates on frames without a seen ball (`gap_check.py --cues`, 3 seeds; identical on the
plain proxy, since the stress touches only the ball). "sign right" and LLR are on true-alive
frames, with the tables fitted leave-one-match-out:

| cue | known | fires | abs p50 | abs p90 | mean | sign right | LLR per frame |
|---|---|---|---|---|---|---|---|
| advance (m/s, 2 s) | 0.997 | 0.663 | 0.37 | 1.25 | -0.02 | 0.590 | 0.103 |
| line (m) | 0.977 | 0.923 | 35.53 | 70.77 | +3.00 | 0.470 | 0.030 |
| spread, width (m) | 0.977 | 0.906 | 12.15 | 24.97 | +0.58 | 0.829 | 1.278 |
| pressure (count, 5 m, 2 s) | 0.106 | 0.052 | 0.00 | 1.00 | +0.03 | 0.692 | 0.224 |

Proxy league against `touch_kickoff` (0.3566), D12 stress. "gaps" means the cue is read only on
frames without a seen ball, "all" on every frame. "fit" means dead ball uses its own fitted
likelihood; "max" means dead ball takes the better team's, so the cue has no say on dead ball:

| date | method | spellF1 | interval | macroF1 | possessions | gain | verdict |
|---|---|---|---|---|---|---|---|
| 2026-11-04 | advance, w 1, gaps, fit | 0.3572 | 0.3406..0.3690 | 0.7499 | 2063 | +0.0006 (-0.0069..+0.0042) | within noise, wins 3/5 |
| 2026-11-04 | spread, w 0.5, gaps, fit | 0.3711 | 0.3548..0.3833 | 0.7736 | 1966 | +0.0145 (+0.0078..+0.0181) | better, below the adoption bar, 5/5 |
| 2026-11-04 | spread, w 1, gaps, fit | 0.3726 | 0.3569..0.3858 | 0.7717 | 1974 | +0.0160 (+0.0095..+0.0214) | better, below the adoption bar, 5/5 |
| 2026-11-04 | line, w 1, gaps, fit | 0.3571 | 0.3407..0.3704 | 0.7562 | 1948 | +0.0005 (-0.0061..+0.0051) | within noise, 3/5 |
| 2026-11-04 | pressure, w 1, gaps, fit | 0.3575 | 0.3414..0.3714 | 0.7595 | 1925 | +0.0009 (-0.0019..+0.0027) | within noise, 4/5 |
| 2026-11-04 | spread, w 2, gaps, fit | 0.3679 | 0.3527..0.3822 | 0.7597 | 2009 | +0.0113 (+0.0053..+0.0179) | better, below the adoption bar, 4/5 |
| 2026-11-04 | spread gyration, w 1, gaps, fit | 0.3672 | 0.3532..0.3819 | 0.7697 | 1973 | +0.0106 (+0.0054..+0.0166) | better, below the adoption bar, 5/5 |
| 2026-11-04 | spread, w 1, gaps, max | 0.3695 | 0.3541..0.3821 | 0.7728 | 1993 | +0.0129 (+0.0075..+0.0172) | better, below the adoption bar, 5/5 |
| 2026-11-04 | spread, w 1, all, fit | 0.3794 | 0.3632..0.3947 | 0.7770 | 1663 | +0.0228 (+0.0120..+0.0332) | ADOPT, 5/5 |
| 2026-11-04 | spread + advance, w 1, gaps, fit | 0.3703 | 0.3547..0.3831 | 0.7684 | 2029 | +0.0138 (+0.0073..+0.0186) | better, below the adoption bar, 5/5 |
| 2026-11-04 | all four, w 0.5, gaps, fit | 0.3738 | 0.3576..0.3857 | 0.7729 | 1991 | +0.0172 (+0.0098..+0.0215) | better, below the adoption bar, 5/5 |
| 2026-11-04 | spread, w 1, all, max | 0.3816 | 0.3654..0.3959 | 0.7808 | 1700 | +0.0250 (+0.0151..+0.0340) | ADOPT, 5/5 |
| 2026-11-04 | spread, w 0.5, all, fit | 0.3803 | 0.3614..0.3914 | 0.7797 | 1826 | +0.0237 (+0.0121..+0.0281) | ADOPT, 5/5 |
| 2026-11-04 | spread, w 2, all, fit | 0.3402 | 0.3224..0.3553 | 0.7524 | 1452 | -0.0164 (-0.0307..-0.0046) | WORSE, 1/5 |
| 2026-11-04 | all four, w 0.5, all, fit | 0.3781 | 0.3572..0.3872 | 0.7780 | 1807 | +0.0215 (+0.0063..+0.0250) | ADOPT, 5/5 |
| 2026-11-04 | **spread, w 0.5, all, max (`touch_gapteam`)** | 0.3791 | 0.3627..0.3930 | 0.7796 | 1837 | +0.0225 (+0.0145..+0.0290) | **ADOPT, 5/5** |
| 2026-11-04 | spread, w 1.5, all, max | 0.3655 | 0.3473..0.3792 | 0.7785 | 1591 | +0.0089 (-0.0037..+0.0177) | within noise, 4/5 |

`touch_gapteam` robustness against `touch_kickoff` (`make proxy METHODS="touch_kickoff
touch_gapteam" ...`; the SPEED and NOISE runs are on plain v1, with no ball stress):

| date | stress | touch_kickoff | touch_gapteam | interval | macroF1 | possessions | gain | verdict |
|---|---|---|---|---|---|---|---|---|
| 2026-11-04 | D12 (the decision run) | 0.3566 | 0.3791 | 0.3627..0.3930 | 0.7796 | 1837 | +0.0225 (+0.0145..+0.0290) | ADOPT, 5/5 |
| 2026-11-04 | none (plain v1) | 0.4498 | 0.4689 | 0.4502..0.4828 | 0.8020 | 1824 | +0.0191 (+0.0084..+0.0249) | better, below the adoption bar, 5/5 |
| 2026-11-04 | SPEED=0.9 | 0.4421 | 0.4630 | 0.4462..0.4786 | 0.7980 | 1869 | +0.0208 (+0.0110..+0.0274) | ADOPT, 5/5 |
| 2026-11-04 | NOISE=0.1 | 0.4538 | 0.4713 | 0.4486..0.4807 | 0.8041 | 1824 | +0.0175 (+0.0016..+0.0189) | better, below the adoption bar, 5/5 |
| 2026-11-04 | D12 + 117 s busy pre-kickoff | 0.3582 | 0.3797 | 0.3598..0.3906 | 0.7816 | 1833 | +0.0215 (+0.0136..+0.0275) | ADOPT, 5/5 |

BAR-ATH, `tools/gap_check.py --cues --bar-ath`, run by miha on 2026-11-04 (label-free, frames
without a seen ball):

| cue | known | fires | abs p50 | abs p90 | mean |
|---|---|---|---|---|---|
| advance | 1.000 | 0.650 | 0.34 | 1.28 | -0.09 |
| line | 0.892 | 0.816 | 36.60 | 72.24 | +10.49 |
| spread | 0.892 | 0.780 | 9.83 | 22.44 | +1.09 |
| pressure | 0.104 | 0.065 | 1.00 | 2.00 | -0.27 |

- **Widths are not inflated:** spread's size is about 0.8 times the proxy's (SPEED=0.9 shrinks it
  by 0.9 and still gets ADOPT). Neither ghosts nor tracked linesmen show.
- **The sign is plausible:** home (Barcelona) is the wider team on average, and its line is higher.
- **Known is 0.89 against 0.98.** On 11% of BAR-ATH gap frames a team has fewer than 5 tracked
  players, which simulator v1 does not reproduce (its losses ignore gaps). The cue is neutral on
  those frames. Expect a somewhat smaller gain than on the proxy.

Where the gain comes from: the boundary report (`BOUNDARIES=1`) and `gap_check.py`, run on the
w 1, all, max setting:
- Restart recall within ±25 frames rises from 0.320 to 0.413, and spurious hand-overs fall from
  848 to 648 per pass.
- Hand-overs with the ball missing barely move (recall 0.053 to 0.068), and neither do
  possessions under 2 s (0.075 to 0.069).
- Team accuracy on both-alive gap frames rises from 0.664 to 0.798. In gaps with an open-play
  hand-over it goes from 0.566 to 0.723, and in gaps with a change at a restart from 0.709 to
  0.936. Yet the in-gap ceiling only drops from +0.095 to +0.089.
- Two runs with the priors refitted give identical codes.

Takeaways:
- **Spread, the team-width difference, is the one strong cue.** Advance, line height and
  pressure are noise alone and add nothing beside spread.
- **Reading it on every frame beats reading it only in gaps** (+0.023 against +0.013). Much of
  the gain is outside gaps: fewer false flips and the right team after restarts.
- **The weight has a cliff above 1.** At 1.5 the gain is noise, and at 2 it is WORSE, because the
  decoder merges real possessions (1,591 and 1,452 predicted against 2,191 true).
- **The cue does not place hand-overs inside gaps.** The team is now right on most gap frames,
  but the in-gap ceiling is almost untouched, so the remaining loss is timing: where inside the
  gap the ball changes hands.

## Section 10: E5 part 2, hand-over timing inside ball gaps, simulator v1 + D12 ball stress (2026-11-04)

Setup: section 9 (5 dev matches x 3 seeds, simulator v1, `BALL_NOISE=0.05 BALL_JUMPS=3.4
BALL_SMOOTH=0.26`). Reference `touch_gapteam` (D16, submitted), spellF1 0.3791 (plain 0.4689).
New: `tools/gap_check.py --timing [--capacity]` (Step 0) and `--decoded [--bar-ath]` (Step 2).
Code at branch `miha/dev`. **No proxy-league runs and no settings tried.** Step 0's
timing ceiling is under 0.02, so Step 1 (new cues) was not started (D17).

Step 0 (a), pooled spellF1 gain over `touch_gapteam` as decoded. "timing" moves each decoded
H<->A change inside a gap onto the true open-play hand-over it matches (same destination,
one-to-one), with teams as decoded. "team" is the section 9 oracle (true team on both-alive gap
frames):

| ball | gaps | timing +-1.5 s | timing +-3 s | timing same gap | team |
|---|---|---|---|---|---|
| D12 stress | all | +0.0007 | +0.0025 | +0.0055 | +0.0865 |
| D12 stress | 2-10 s | +0.0003 | +0.0004 | +0.0009 | +0.0189 |
| D12 stress | 10-30 s | +0.0001 | +0.0004 | +0.0013 | +0.0244 |
| D12 stress | >30 s | +0.0003 | +0.0017 | +0.0033 | +0.0492 |
| plain | all | +0.0007 | +0.0029 | +0.0057 | +0.0818 |

Where the team ceiling sits (team oracle on part of the gap frames, D12 stress; plain in
brackets): within +-3 s of a true in-gap hand-over +0.0769 (+0.0715), farther from one +0.0046
(+0.0037), frames of true possessions mostly in a gap and under 2 s +0.0421 (+0.0397), under 5 s
+0.0740 (+0.0709).

Capacity oracle: `touch_gapteam` with its spread cue replaced, on alive gap frames, by the true
team's sign at a fixed size. It shows what the decoder makes of ideal evidence:

| ball | cue size, team_w | gain | recall in-gap poss <2 s | 2-5 s | predicted / true poss |
|---|---|---|---|---|---|
| D12 stress | 12 m (p50), 0.5 | +0.0506 | 0.134 | 0.416 | 0.881 |
| D12 stress | 25 m (p90), 0.5 | +0.0671 | 0.269 | 0.485 | 0.905 |
| D12 stress | 12 m, 1 | +0.0723 | 0.278 | 0.500 | 0.843 |
| plain | 12 m, 0.5 | +0.0498 | 0.131 | 0.446 | 0.875 |
| plain | 25 m, 0.5 | +0.0663 | 0.267 | 0.513 | 0.898 |
| plain | 12 m, 1 | +0.0565 | 0.278 | 0.521 | 0.840 |

Step 0 (b) and (c), D12 stress: per true open-play hand-over inside a gap (347.7 per seed),
whether the decoder makes a change to the new team, and when the spread cue (1 s centred mean,
signed towards the new team) turns positive within +-10 s. "new" is the length of the
possession the hand-over starts:

| hand-overs | n/seed | decoded +-1 s | +-3 s | none in gap +-3 s | cue flips | lag s p10/p50/p90 | flip +-1 s | flip +-3 s | cue new at +2 s |
|---|---|---|---|---|---|---|---|---|---|
| all | 347.7 | 0.065 | 0.169 | 0.327 | 0.438 | -4.3/+3.0/+8.2 | 0.047 | 0.159 | 0.381 |
| gap 2-10 s | 77.0 | 0.074 | 0.234 | 0.437 | 0.519 | -4.0/+3.1/+7.7 | 0.056 | 0.195 | 0.394 |
| gap 10-30 s | 101.0 | 0.086 | 0.185 | 0.479 | 0.333 | -5.6/+3.4/+7.6 | 0.033 | 0.116 | 0.350 |
| gap >30 s | 167.3 | 0.044 | 0.125 | 0.181 | 0.460 | -4.5/+3.0/+9.0 | 0.048 | 0.163 | 0.390 |
| new <2 s | 150.3 | 0.040 | 0.129 | 0.399 | 0.368 | -4.9/+3.8/+9.2 | 0.031 | 0.098 | 0.310 |
| new 2-5 s | 107.7 | 0.059 | 0.139 | 0.328 | 0.440 | -5.8/+3.2/+8.8 | 0.031 | 0.149 | 0.334 |
| new >5 s | 89.7 | 0.115 | 0.271 | 0.204 | 0.554 | -2.3/+2.8/+5.5 | 0.093 | 0.275 | 0.554 |

Latency of matched decoded changes within +-3 s: p50 +1.1 s. The nearest decoded change to the
new team anywhere in the gap: p10/p50/p90 -11.9/+1.6/+29.4 s. A 4 s mean of the cue is no
better than 1 s (flips +-1 s 0.036, never 0.609).

Duration priors (possession, leave-one-match-out): the minimum is 0.12 to 0.16 s (1 block), and
only 1.0% of true possessions mostly in a gap are shorter. log P(d) is flat from 0.6 to 2 s
(about -3.3 to -3.55, near the mode). A short possession inside a long one costs 6.4 to 8.1 nats
of prior: two extra segments.

Step 2 on the stressed proxy (`gap_check.py --decoded`, label-free aggregates). Truth is the
target:

| method | gap H/(H+A) | gap D | seen D | poss/90 | in-gap poss/90 | in-gap <2 s/90 | in-gap H<->A /90 |
|---|---|---|---|---|---|---|---|
| touch_kickoff | 0.629 | 0.746 | 0.189 | 360.2 | 39.9 | 1.4 | 2.8 |
| touch_gapteam | 0.528 | 0.751 | 0.205 | 347.3 | 46.1 | 2.8 | 10.3 |
| truth | 0.547 | 0.716 | 0.256 | 414.1 | 92.2 | 35.2 | 65.7 |

BAR-ATH, `tools/gap_check.py --decoded --bar-ath`, run by miha on 2026-11-04. These are
label-free aggregates. "tag" is agreement with the team of the Smart Tagging attacking phases,
on decoded team frames. The phase tags give 140 attacking and 140 defending tags, 4 without a
team. They cover 61,216 frames, the defending mirror agrees 1.000, and home holds 0.699 of the
tagged frames:

| method | gap H/(H+A) | gap D | seen D | poss/90 | in-gap poss/90 | in-gap <2 s/90 | in-gap H<->A /90 | tag gap | tag seen |
|---|---|---|---|---|---|---|---|---|---|
| touch_kickoff | 0.534 | 0.768 | 0.178 | 421.5 | 53.5 | 11.9 | 6.8 | 0.596 | 0.786 |
| touch_gapteam | 0.486 | 0.776 | 0.189 | 392.6 | 58.6 | 13.6 | 19.5 | 0.721 | 0.813 |

- **The spread cue helps on BAR-ATH too.** Tag agreement inside gaps rises from 0.596 to 0.721,
  much like the proxy's team accuracy on gap frames (0.664 to 0.797). With the ball seen it
  rises only from 0.786 to 0.813. The tags are coarse automatic phases, so about 0.8 is their
  own ceiling.
- **The decoder changes team inside gaps more often on BAR-ATH than on the proxy:** 19.5
  against 10.3 hand-overs per 90, and 13.6 against 2.8 short possessions per 90. Both are still
  far below the proxy truth (65.7 and 35.2). Without labels it cannot be told whether these are
  real hand-overs or false flips. The tag agreement does not suggest harm. D17's conclusion
  stands: the decoder still makes far fewer in-gap hand-overs than truth has.
- `touch_gapteam` gives home a smaller share of gap team frames (0.486 against 0.534), while the
  tags give home 0.699 of tagged frames over the whole match. Tag agreement still rises, so the
  shift lies on frames the tags do not cover or does not hurt on the ones they do. The tags'
  home share inside gaps alone was not printed.

Takeaways:
- **What remains of the in-gap ceiling is not timing.** Moving existing changes is worth at most
  +0.006. The loss is possessions that start and end inside one gap (+0.074 under 5 s). The
  decoder makes 10 in-gap hand-overs per 90 where truth has 66.
- **The evidence is the blocker, not the decoder.** With an ideal-sign cue of typical size, the
  decoder recovers +0.050 to +0.072 of +0.0865, and minimum durations are not binding. The real
  spread cue is posture: it flips within +-1 s of only 5% of hand-overs and lags by a median
  3 s. The prior's split cost (about 7 nats) still caps the shortest: even ideal evidence finds
  only 13 to 28% of in-gap possessions under 2 s.

## Section 11: Step A (players-only moment features) and Step B (short possessions, ball seen), simulator v1 + D12 ball stress (2026-11-04)

Setup: section 10 (5 dev matches x 3 seeds, simulator v1, `BALL_NOISE=0.05 BALL_JUMPS=3.4
BALL_SMOOTH=0.26`). Reference `touch_gapteam` (D16), spellF1 0.3791 (plain 0.4689). New:
`solution/handover.py` (features, events), `tools/gap_check.py --events` and `--seen`. Code at
branch `miha/dev`, uncommitted. **No proxy-league runs.** Diagnostics only: 14 features, each
under the stress, plain, NOISE=0.1 and SPEED=0.9.

Step A, `gap_check.py --events`. Rows are against true open-play hand-overs inside gaps (347.7
per seed; 150.3 start a possession under 2 s). "cross" features fire at zero crossings; "peak"
features fire at maxima of |value| above the match's 0.9 quantile, at most one per 1 s.
- "hit" means an event towards the new team within +-1 or +-3 s.
- "FA" is events per minute of alive in-gap play more than 3 s from any true transition.
- "chance" is the hit share of events at the FA rate placed at random (signed: half per
  direction). lift = hit / chance.
- "prec" is the share of events on alive gap frames within +-1 s of a hand-over to their team.

| feature | kind | ev/min | hit 1 s | hit 3 s | lag s p10/p50/p90 | FA/min | lift 1 s | lift 3 s | <2 s hit 1 s | prec 1 s |
|---|---|---|---|---|---|---|---|---|---|---|
| spread (1 s mean) | cross | 2.3 | 0.048 | 0.160 | -4.5/+3.0/+8.2 | 1.3 | 2.27 | 2.61 | 0.033 | 0.140 |
| advance (1 s) | cross | 7.3 | 0.261 | 0.562 | -5.1/+0.6/+4.3 | 9.2 | 1.80 | 1.52 | 0.202 | 0.154 |
| reversal (both teams) | cross | 3.6 | 0.159 | 0.396 | -6.0/+1.1/+6.3 | 5.3 | 1.85 | 1.69 | 0.106 | 0.150 |
| leave (last-seen ball) | cross | 14.7 | 0.143 | 0.337 | -5.9/+0.0/+6.7 | 6.4 | 1.39 | 1.22 | 0.118 | 0.098 |
| crowd (unsigned) | peak | 4.7 | 0.066 | 0.232 | -6.6/+0.8/+7.1 | 1.1 | 1.87 | 2.30 | 0.062 | 0.214 |
| dspread 0.5 s | peak | 5.2 | 0.126 | 0.356 | -5.4/+0.9/+6.4 | 4.8 | 1.62 | 1.67 | 0.109 | 0.126 |
| dspread 1 s | peak | 4.0 | 0.094 | 0.288 | -5.9/+1.2/+6.4 | 3.6 | 1.58 | 1.74 | 0.082 | 0.122 |
| dspread 2 s | peak | 2.7 | 0.061 | 0.206 | -5.6/+1.6/+6.5 | 2.4 | 1.52 | 1.80 | 0.049 | 0.122 |
| **dadvance 0.5 s** | peak | 5.1 | **0.324** | 0.542 | -5.2/**+0.7**/+4.2 | 4.7 | **4.23** | 2.58 | 0.282 | 0.262 |
| dadvance 1 s | peak | 4.1 | 0.262 | 0.459 | -5.6/+0.7/+4.8 | 3.7 | 4.30 | 2.71 | 0.220 | 0.257 |
| dadvance 2 s | peak | 3.0 | 0.176 | 0.338 | -6.6/+0.8/+5.5 | 2.4 | 4.49 | 3.03 | 0.111 | 0.261 |
| dline 0.5 s | peak | 3.9 | 0.048 | 0.151 | -7.2/+0.8/+7.1 | 3.1 | 0.94 | 1.05 | 0.049 | 0.083 |
| dline 1 s | peak | 2.7 | 0.029 | 0.094 | -7.0/+2.1/+7.2 | 1.9 | 0.92 | 1.04 | 0.040 | 0.075 |
| dline 2 s | peak | 1.8 | 0.012 | 0.046 | -7.3/+2.8/+7.7 | 0.9 | 0.85 | 1.07 | 0.020 | 0.057 |

- **The spread row reproduces section 10's part (c)**: 0.048 / 0.160 / +3.0 s, against
  0.047 / 0.159 / +3.0 s. Only "leave" and "crowd" read the ball. Plain matches the stress
  except for those two.
- **dadvance holds up under the stresses:**
  - NOISE=0.1 (`--noise 0.1`): hit 1 s 0.342, lift 4.01, FA 5.2/min. Plain "advance" collapses
    there (35 ev/min, lift 0.82).
  - SPEED=0.9 (`--speed 0.9`): unchanged (0.324, lift 4.23), because the thresholds are quantiles.
- dadvance's label-free rates on the proxy (`--rates`): 5.12 per minute of play, 2.52 per gap
  minute. These are the numbers to compare against `--events --bar-ath`.

Step B, `gap_check.py --methods touch_gapteam --seen`, D12 stress (plain in brackets for the
"<2 s" row). The pool is true possessions under 5 s lying mostly where the ball is seen. Oracle
gains in pooled spellF1:
- "team": the true team on the group's both-alive frames.
- "snap": decoded changes matching the group's true starts and ends within +-1.5 s, moved
  onto them.
- "reach": team, only on possessions whose start the evidence sees (an event by their team
  within +-1 s).
- "random": team on as many possessions drawn at random.
"seen" is the share of starts the evidence sees.

| length | n/seed | recall | team acc | team | snap | touch: seen | touch: reach | random | dadvance: seen | dadvance: reach | touch or dadvance: reach |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <1 s | 188.3 | 0.030 | 0.413 | +0.0498 | +0.0067 | 0.419 | +0.0067 | +0.0207 | 0.255 | +0.0134 | +0.0181 |
| 1-2 s | 279.7 | 0.150 | 0.501 | +0.0716 | +0.0139 | 0.433 | +0.0055 | +0.0298 | 0.356 | +0.0289 | +0.0311 |
| <2 s | 468.0 | 0.102 (0.190) | 0.484 | +0.1176 (+0.0987) | +0.0185 (+0.0194) | 0.427 (0.563) | +0.0121 (+0.0186) | +0.0536 | 0.316 | +0.0418 (+0.0321) | +0.0484 (+0.0410) |
| 2-5 s | 504.7 | 0.282 | 0.638 | +0.1038 | +0.0200 | 0.491 | +0.0136 | +0.0514 | 0.384 | +0.0494 | +0.0567 |
| <5 s | 972.7 | 0.195 | 0.602 | +0.2175 | +0.0321 | 0.460 | +0.0262 | +0.1056 | 0.351 | +0.0893 | +0.1040 |

The "random" column is for touch's seen share. For dadvance's share it is +0.0388 under 2 s.

- What the touch engine sees at the starts of the <2 s pool (419.7 per seed start at a
  hand-over):
  - the ball is usable at the start for 0.986;
  - a touch by the new team within +-0.5 s for 0.350 (plain 0.562), within +-1 s for 0.423
    (plain 0.594);
  - an engine claim (holder change to the new team) within +-1 s for 0.359, median lag -0.1 s;
  - their team holds 0.445 of their frames.
- Engine claims on usable-ball alive frames: 8.3 per minute; 0.360 lie within +-1 s of a true
  change to their team (plain 0.478).
- Step A analogue, in-gap short possessions (186.0 per seed under 2 s, recall 0.018), with
  dadvance as the evidence:

| length | team | snap | dadvance: seen | dadvance: reach | random |
|---|---|---|---|---|---|
| <2 s | +0.0426 | +0.0013 | 0.262 | +0.0196 | +0.0118 |
| <5 s | +0.0771 | +0.0032 | 0.283 | +0.0369 | +0.0201 |

Takeaways:
- **One players-only feature marks the moment of a hand-over:** dadvance, everyone's run
  velocity along x' changing. Lift 4.2 at +-1 s, against 2.3 for spread and 2 or below for
  the rest. It lags by a median 0.7 s, against spread's 3 s. Its precision is low: 0.26 of
  its events on alive gap frames sit at a hand-over.
- **Touch and E6 ceilings on the <2 s ball-seen pool are under 0.02:** snap +0.0185, touch
  reach +0.012. Touch reach is below random because the engine sees the possessions the
  decoder already finds through the holder term. The large team ceiling (+0.118) needs
  evidence the touch engine does not have.
- **dadvance reaches more of the ball-seen pool than the touch engine:** +0.042 against
  +0.012 under 2 s, from fewer sightings (0.32 against 0.43). Together they reach +0.048, and
  +0.104 under 5 s. Every reach number ignores false alarms, so it is an upper bound.

BAR-ATH, `tools/gap_check.py --events --bar-ath`, run by miha on 2026-11-04. These are
label-free event rates, compared with the proxy's `--events --rates` (plain; the D12 stress
changes only leave and crowd):

| feature | proxy ev/min all | BAR-ATH ev/min all | proxy ev/min gaps | BAR-ATH ev/min gaps |
|---|---|---|---|---|
| spread | 2.30 | 2.53 | 1.73 | 2.67 |
| advance | 7.33 | 8.27 | 8.22 | 8.54 |
| reversal | 3.57 | 3.28 | 2.43 | 1.21 |
| leave | 15.08 | 16.95 | 3.93 | 4.57 |
| crowd | 4.93 | 4.71 | 0.63 | 0.60 |
| dspread0.5 | 5.20 | 5.53 | 3.80 | 4.57 |
| dspread1 | 3.97 | 4.15 | 2.80 | 3.39 |
| dspread2 | 2.74 | 2.90 | 1.92 | 2.44 |
| **dadvance0.5** | 5.12 | 5.01 | 2.52 | 2.02 |
| dadvance1 | 4.13 | 4.29 | 1.88 | 1.74 |
| dadvance2 | 3.02 | 3.00 | 1.21 | 0.90 |
| dline0.5 | 3.88 | 2.87 | 2.38 | 1.79 |
| dline1 | 2.74 | 2.14 | 1.28 | 1.11 |
| dline2 | 1.80 | 1.51 | 0.54 | 0.56 |

- **dadvance0.5 shows no warning sign.** The overall rate matches (5.01 against 5.12), but
  that is mostly by construction: the threshold is each match's own 0.9 quantile. The
  informative number is the in-gap rate: 2.02 against 2.52, about 0.8 times. That is the same
  factor as spread's size in section 9, and there is no burst from noise or ghosts.
- Spread crosses zero more often inside BAR-ATH gaps (2.67 against 1.73). That is consistent
  with fewer tracked players there (spread known 0.89 against 0.98, section 9). Reversal
  fires half as often in gaps (1.21 against 2.43): both teams agree on a run direction less
  often.
- Rates cannot show whether BAR-ATH events land on hand-overs; there are no labels.

Transition bonus Step 0, `tools/gap_check.py --bonus`, D12 stress, 3 seeds. These are
diagnostic decodes, not proxy-league runs. The method is `touch_gapteam` plus a bonus of w nats
on an H <-> A switch into a team at the blocks within 0.4 s of its events
(`decoder.switch_bonus`), on every frame. The event sources:
- "all" is every dadvance0.5 event;
- "right" keeps only those within 1 s of a true hand-over to their team, an oracle filter:
  no false alarms;
- "perfect" is every true hand-over.
The shift moves the events earlier by dadvance's median lag. Recall is of true possessions
under 2 s, all and mostly in a gap. "ratio" is predicted over true possessions.
**Settings tried: 15.**

| bonus | spellF1 | gain | recall <2 s | in gap <2 s | ratio |
|---|---|---|---|---|---|
| touch_gapteam | 0.3791 | -- | 0.078 | 0.018 | 0.839 |
| all, w 2, shift 0 | 0.3804 | +0.0013 | 0.081 | 0.022 | 0.852 |
| all, w 4, shift 0 | 0.3799 | +0.0008 | 0.087 | 0.032 | 0.882 |
| all, w 7, shift 0 | 0.3573 | -0.0217 | 0.094 | 0.047 | 1.053 |
| all, w 2, shift -0.7 s | 0.3803 | +0.0012 | 0.082 | 0.020 | 0.852 |
| all, w 4, shift -0.7 s | 0.3834 | +0.0043 | 0.088 | 0.029 | 0.876 |
| all, w 7, shift -0.7 s | 0.3717 | -0.0074 | 0.110 | 0.065 | 1.018 |
| right, w 2, shift 0 | 0.3817 | +0.0026 | 0.080 | 0.022 | 0.843 |
| right, w 4, shift 0 | 0.3842 | +0.0051 | 0.086 | 0.032 | 0.852 |
| right, w 7, shift 0 | 0.3858 | +0.0067 | 0.095 | 0.048 | 0.888 |
| right, w 2, shift -0.7 s | 0.3811 | +0.0020 | 0.081 | 0.020 | 0.843 |
| right, w 4, shift -0.7 s | 0.3849 | +0.0058 | 0.086 | 0.027 | 0.852 |
| right, w 7, shift -0.7 s | 0.3887 | +0.0096 | 0.100 | 0.050 | 0.891 |
| perfect, w 2 | 0.3875 | +0.0084 | 0.094 | 0.029 | 0.853 |
| perfect, w 4 | 0.4034 | +0.0243 | 0.134 | 0.091 | 0.901 |
| perfect, w 7 | 0.4193 | +0.0403 | 0.223 | 0.237 | 1.189 |

- **The dadvance bonus cannot reach D4's bar.** Even with every false alarm removed, it gains
  at most +0.0096. Its recall (about a third of hand-overs within 1 s) is the limit, not its
  precision.
- **The real events gain at most +0.0043**, and at 7 nats the false alarms make it WORSE
  (-0.022, ratio 1.05).
- **The bonus mechanism itself works:** perfect events gain +0.024 at 4 nats and +0.040 at 7.
  A bonus pays only with an event source that recalls most hand-overs.
- With `bonus_w` 0 the decoder is byte-identical: `touch_gapteam`'s codes on J03WMX seed 0
  hash the same before and after the change.

## Section 12: controlled-possession truth (D20), simulator v1 + D12 ball stress (2026-11-04)

Setup: section 11 (5 dev matches x 3 seeds, simulator v1, `BALL_NOISE=0.05 BALL_JUMPS=3.4
BALL_SMOOTH=0.26`) except the **truth labels**. New: `solution/relabel.py`, `idsse.LABELS` and
`load(labels=)`, `idsse.completed_passes`, `tools/label_check.py`, and `make proxy LABELS=...
PRIOR_LABELS=...` (`--labels`, `--prior-labels`). Code at branch `miha/dev`. **Numbers
on different labels are not comparable**: keep2 has 1,455 true possessions where DFL has 2,191.

The two variants were fixed from CHALLENGE.md's wording before any measurement. Neither was
chosen or tuned on the public window:
- **keep2:** a team run under 2 s ("keeps the ball for a few seconds", the low end of "a few")
  is a touch, not control. It goes to the team that had the ball before it, and short runs are
  absorbed shortest first. So a single touch inside the other team's spell does not delay that
  spell's start. The first team run of a half, or after a stoppage, is never absorbed. D stays
  as DFL marks it.
- **keep2pass:** keep2, but a run that holds a completed pass by its team (DFL `Play`
  `successfullyCompleted`, +-1 s for the manual event clock; 697 per match) is control at any
  length. "More than two consecutive passes" adds nothing beyond 2 s, so one completed pass is
  the event-feed reading of control.

Step 0, `tools/label_check.py --public` (clean labels). keep1 and keep3 are sensitivity rows,
described here, not candidates:

| labels | possessions | per 90 | under 1 s | under 2 s | median s | mean s | direct H<->A per alive min |
|---|---|---|---|---|---|---|---|
| IDSSE dev, DFL | 2191 | 414 | 0.122 | 0.298 | 3.7 | 7.3 | 6.31 |
| keep1 | 1843 | 348 | 0.004 | 0.182 | 4.8 | 8.7 | 5.00 |
| **keep2** | 1455 | 275 | 0.005 | 0.039 | 6.8 | 11.0 | 3.54 |
| **keep2pass** | 1473 | 278 | 0.007 | 0.045 | 6.7 | 10.8 | 3.61 |
| keep3 | 1212 | 229 | 0.005 | 0.034 | 8.8 | 13.2 | 2.63 |
| BAR-ATH public window (17) | 17 | 306 | 0.059 | 0.118 | 7.6 | 14.2 | 2.98 |
| public window put through keep2 | 13 | 234 | 0 | 0 | 8.1 | 18.6 | 1.99 |
| Metrica Sample Game 3, event-derived (2026-11-06, see below) | 317 | 298 | 0.038 | 0.174 | 7.0 | 12.6 | 3.64 |

**Metrica sample data, checked 2026-11-06 (review session).** The same-vendor check PLAN section 2
asked for. `metrica-sports/sample-data` (commit `e706dd5`, public, other games): Sample Game 3 is
the EPTS match (25 fps, 143,761 frames, `Sample_Game_3_tracking.txt` 53 MB, `_metadata.xml`,
`_events.json` 3,620 manual events). **Its tracking carries x and y only, for players and ball: no
ball-status and no owning-team channel.** So no public Metrica file shows the corrected tracking's
"dead" and "owning" conventions; that question stays open (PLAN question 7). The row above is a
possession timeline derived from the manual events by a fixed rule (scratch script, not committed):
dead from a BALL OUT, a FAULT RECEIVED or a GOAL until the next SET PIECE, otherwise the team of
the last pass, carry, recovery, shot or set piece. Described only, never chosen on (L2):
- the structure sits next to the public window (298 against 306 per 90, median 7.0 against
  7.6 s, 3.6 against 3.0 direct hand-overs per alive minute) and between keep2 and DFL on the
  share under 2 s (0.17 against 0.04 and 0.30). Metrica's own annotators keep short possessions:
  29% of RECOVERY events are followed by a BALL LOST of the same team within 2 s;
- the BALL OUT event sits at the end of the preceding pass or carry (p50 0.08 s): the annotation
  places a ball out of play where it crosses the line, not at a whistle. Whether the corrected
  tracking's dead flag does the same is not shown by this file;
- its dead share (0.30) and stoppages per 90 (69, median 13.7 s) under-count: fouls carry only 12
  FAULT RECEIVED events against 28 free kicks, and injuries or substitutions have no event.
One match of one competition; the tracking file itself was not downloaded.

What keep2 does to the DFL possessions, by DFL length (keep2pass in brackets). "absorbed": its
frames go to the other team. "grown": it now spans a touch by the other team.

| DFL length | n | kept | absorbed | grown | removed share |
|---|---|---|---|---|---|
| <1 s | 267 | 7 (10) | 224 (218) | 36 (39) | 0.839 (0.816) |
| 1-2 s | 387 | 47 (55) | 233 (230) | 107 (102) | 0.602 (0.594) |
| 2-5 s | 664 | 419 | 0 | 245 | 0 |
| 5-10 s | 396 | 267 | 0 | 129 | 0 |
| >10 s | 477 | 334 | 0 | 143 | 0 |
| all | 2191 | 1074 (1096) | 457 (448) | 660 (647) | 0.209 (0.204) |

- keep2 changes 1.6% of graded frames, but 21% of DFL possessions disappear, and 30% of the rest
  grow. Clean DFL labels scored against keep2 truth give **spellF1 0.757**: the label semantics
  alone are worth a quarter of the metric. keep2pass against keep2: 0.989.
- **Structure (PLAN L2, not for choosing).** The public window sits between the two label sets.
  Its share under 2 s (0.12) is between keep2's (0.04) and DFL's (0.30). Its direct hand-overs
  (2.98 per alive minute) are nearer keep2's (3.54) than DFL's (6.31). keep2 is not exact for the
  window: it would remove its 2 short in-play possessions and merge 17 into 13. So the real
  truth keeps some in-play possessions under 2 s, and keep2 is the stricter bracket, DFL the
  looser.
- The decoded/true count fits keep2 too. On the stressed proxy, `touch_gapteam` decodes 347 per
  90: 0.84 times DFL's truth, 1.26 times keep2's. On BAR-ATH it decodes 393 against the
  window's 306, 1.28 times (section 10; 17 possessions).

Step 2, re-tests under the D12 stress. Rows give each comparison's gain over its first method.
- "priors keep2": every prior is fitted on the same labels as the truth (`LABELS=keep2`).
- "priors DFL": the methods as submitted (`LABELS=keep2 PRIOR_LABELS=dfl`).
- The kickoff rows add `PREKICKOFF=117 PREKICKOFF_PROFILE=busy`, as in section 7.
- keep2pass was run with "priors keep2pass" and "priors DFL" (gapteam_k2 only).

| comparison | DFL truth (old) | keep2, priors keep2 | keep2, priors DFL | keep2pass, priors keep2pass |
|---|---|---|---|---|
| touch_markov (spellF1) | 0.3566 | 0.4005 | 0.3890 | 0.3954 |
| touch_markov_hold - touch_markov (D10 touch_takes False) | -0.0355 (-0.047..-0.027), 0/5, WORSE | +0.0055 (-0.004..+0.018), 2/5, noise | -0.0009 (-0.011..+0.011), 2/5, noise | +0.0068 (-0.002..+0.020), 2/5, noise |
| semi_markov - touch_markov | -0.0700, WORSE | -0.0476, WORSE | -- | -0.0447, WORSE |
| touch_kickoff - touch_markov (busy pre-kickoff) | +0.0015 (+0.0006..+0.0021), 5/5, better | +0.0023 (+0.0023..+0.0031), 5/5, better | +0.0019 (+0.0009..+0.0027), 5/5, better | +0.0022, 5/5, better |
| touch_gapteam - touch_kickoff | +0.0225 (+0.015..+0.029), 5/5, **ADOPT** | +0.0393 (+0.031..+0.048), 5/5, **ADOPT** | +0.0319 (+0.027..+0.042), 5/5, **ADOPT** | +0.0391 (+0.032..+0.049), 5/5, **ADOPT** |
| touch_gapteam_hold - touch_gapteam | -0.0362 (-0.044..-0.021), 0/5, WORSE | +0.0026 (-0.003..+0.022), 2/5, noise | +0.0119 (+0.006..+0.029), 4/5, better | +0.0047 (-0.003..+0.022), 2/5, noise |
| baseline_smooth50 (spellF1) | 0.2546 | 0.2840 | 0.2840 | 0.2841 |
| touch_gapteam (spellF1, predicted poss) | 0.3791, 1837 | 0.4398, 1713 | 0.4209, 1837 | 0.4345, 1731 |

**Settings tried: 1 new method** (`touch_gapteam_k2`: `touch_gapteam` with every prior fitted on
keep2 labels, `prior_labels="keep2"`). `touch_gapteam_hold` and `touch_markov_hold` are D10's
re-tests, not candidates. `touch_gapteam_k2` against `touch_gapteam` as submitted (DFL priors):

| truth | stress | touch_gapteam | touch_gapteam_k2 | interval | macroF1 | possessions | gain | verdict |
|---|---|---|---|---|---|---|---|---|
| keep2 | D12 (the decision run) | 0.4209 | 0.4398 | 0.4194..0.4518 | 0.7835 | 1713 | +0.0189 (+0.0144..+0.0212) | better, below the adoption bar, 5/5 |
| keep2pass | D12 | 0.4196 | 0.4379 | 0.4163..0.4499 | 0.7833 | 1713 | +0.0183 (+0.0136..+0.0203) | better, below the adoption bar, 5/5 |
| DFL (old) | D12 | 0.3791 | 0.3855 | 0.3675..0.3984 | 0.7798 | 1713 | +0.0064 (+0.0010..+0.0094) | better, below the adoption bar, 5/5 |
| keep2 | none (plain v1) | 0.4942 | 0.5134 | 0.4946..0.5272 | 0.8025 | 1705 | +0.0192 (+0.0172..+0.0238) | better, below the adoption bar, 5/5 |
| keep2 | SPEED=0.9 | 0.4828 | 0.5052 | 0.4858..0.5184 | 0.7984 | 1732 | +0.0223 (+0.0197..+0.0264) | ADOPT, 5/5 |
| keep2 | NOISE=0.1 | 0.4956 | 0.5130 | 0.4953..0.5278 | 0.8044 | 1707 | +0.0174 (+0.0161..+0.0228) | better, below the adoption bar, 5/5 |
| keep2 | D12 + 117 s busy pre-kickoff | 0.4228 | 0.4413 | 0.4257..0.4577 | 0.7860 | 1718 | +0.0185 (+0.0173..+0.0239) | better, below the adoption bar, 5/5 |

Takeaways:
- **What we adopted still holds on controlled-possession truth.** The touch engine still beats
  `semi_markov` by about 0.045. The kickoff grammar stays better at 5/5. The spread cue's gain
  grows, from +0.0225 to +0.032 (submitted priors) or +0.039 (keep2 priors), ADOPT on every
  label set.
- **D10's `touch_takes` no longer separates.** On DFL truth, waiting for control is WORSE by
  0.036. On controlled truth it is within noise: -0.001 to +0.007, 2/5. With `touch_gapteam` it is +0.012 (4/5) under
  submitted priors, but WORSE on DFL. `touch_takes = True` stays: nothing gets ADOPT. D10's
  "risk taken on purpose" turned out harmless.
- **Fitting the priors on controlled labels is the one lever that shows**: +0.017 to +0.022
  on every stress, 5/5, and not WORSE (in fact better) on DFL truth. It decodes fewer
  possessions (1,713 against 1,837). It misses D4's bar on the decision run (+0.0189), so it
  is **not adopted**.

BAR-ATH, `tools/gap_check.py --decoded --bar-ath --methods touch_gapteam touch_gapteam_k2`, run
by miha on 2026-11-04. These are label-free aggregates, with columns as in section 10:

| method | gap H/(H+A) | gap D | seen D | poss/90 | in-gap poss/90 | in-gap <2 s/90 | in-gap H<->A /90 | tag gap | tag seen |
|---|---|---|---|---|---|---|---|---|---|
| touch_gapteam | 0.486 | 0.776 | 0.189 | 392.6 | 58.6 | 13.6 | 19.5 | 0.721 | 0.813 |
| touch_gapteam_k2 | 0.502 | 0.776 | 0.192 | 364.6 | 59.5 | 12.7 | 20.4 | 0.676 | 0.823 |

- **The count falls as on the proxy:** 393 to 365 per 90, 0.93 times, against the proxy's 347 to
  324 (also 0.93). It is nearer the window's 306, but still above it.
- **Dead ball does not move** (gap D 0.776, seen D 0.19). The change is only in team ownership.
- **Warning sign:** agreement with the Smart Tagging team inside gaps falls from 0.721 to
  0.676; with the ball seen it rises from 0.813 to 0.823. The tags are coarse, but the drop is
  on the frames where team choice is hardest. No case for an exception to D4:
  `touch_gapteam` stays submitted.

## Section 13: C3 frame model, keep2 truth, simulator v1 + D12 ball stress (2026-11-04)

Setup: section 12 (5 dev matches x 3 seeds, simulator v1, `BALL_NOISE=0.05 BALL_JUMPS=3.4
BALL_SMOOTH=0.26`), truth `LABELS=keep2`, priors `PRIOR_LABELS=dfl` (D21) unless a row says
otherwise. Reference `touch_gapteam` (D16, submitted). New: `tools/gap_check.py --ceiling`,
`--labels`, `--prior-labels` (truth now keep2 by default), `solution/framemodel.py`, decoder
term `frame_w` / `frame_where`. Code at branch `miha/dev`. **Settings tried: 4**
(weight 0.5 and 1 on every frame, 0.5 in gaps only, 0.5 replacing spread). The model's own
parameters (`framemodel.PARAMS`, its feature list, `TRAIN_SEED`) were set once and not tuned.

Step 0, `gap_check.py --ceiling` (DFL truth in brackets; the DFL column reproduces section 10
exactly). Team oracle: the true team on both-alive gap frames. Capacity: `touch_gapteam` with
spread replaced on alive gap frames by the true team's sign at 12 m (spread's p50), `team_w`
0.5 (D17). The split rows act only on that split's gap frames:

| split | n/seed keep2 (DFL) | team oracle keep2 (DFL) | capacity keep2 (DFL) |
|---|---|---|---|
| all gaps | 683.7 | +0.0451 (+0.0865) | **+0.0530** (+0.0506) |
| gap <2 s | 83.0 | -0.0021 (-0.0012) | +0.0003 (+0.0002) |
| gap 2-10 s | 252.3 | +0.0085 (+0.0189) | +0.0092 (+0.0084) |
| gap 10-30 s | 261.7 | +0.0128 (+0.0244) | +0.0153 (+0.0150) |
| gap >30 s | 86.7 | +0.0279 (+0.0492) | +0.0288 (+0.0276) |
| poss <2 s touching a gap | 25.7 (204.0) | +0.0005 (+0.0421) | +0.0003 (+0.0063) |
| poss 2-5 s | 168.3 (224.7) | +0.0299 (+0.0323) | +0.0265 (+0.0264) |
| poss 5-15 s | 244.3 (230.7) | +0.0142 (+0.0110) | +0.0174 (+0.0126) |
| poss >15 s | 188.3 (169.0) | +0.0007 (+0.0009) | +0.0036 (+0.0022) |

Capacity settings on every gap (keep2; DFL in brackets). Predicted / true possessions: 1.263
for `touch_gapteam` (0.839):

| setting | gain | ratio |
|---|---|---|
| ideal 12 m, w 0.5 | +0.0530 (+0.0506) | 1.320 |
| ideal 25 m, w 0.5 | +0.0542 (+0.0671) | 1.329 |
| ideal 12 m, w 1 | +0.0691 (+0.0723) | 1.234 |
| lag 1 s, 12 m, w 0.5 | +0.0350 (+0.0283) | 1.315 |
| lag 3 s, 12 m, w 0.5 | +0.0041 (+0.0037) | 1.304 |
| 20% wrong in 2 s runs, 12 m, w 0.5 | -0.0016 (+0.0096) | 1.427 |

- On keep2 the team ceiling halves (+0.045 against +0.087), because keep2 has few in-gap
  possessions under 2 s (26 against 204 per seed). The capacity holds (+0.053), mostly in
  possessions of 2 to 15 s and gaps over 10 s. **Step 0 clears 0.02, so C3 was built.**
- Evidence that lags by 3 s (spread's lag) is worth nothing, and 20% wrong in runs is worth
  nothing: the model has to be timely and accurate.

The model (`solution/framemodel.py`): P(home | alive) per frame from 35 players-only features
(spread, gyration, line, advance, reversal, every d-feature of `handover.py`, spread context
means, past and future 3 s means of spread and advance, field tilt, per-team depth, front,
width, velocity, speed and count). Team-swap symmetric by construction. sklearn
`HistGradientBoostingClassifier` (pinned 1.9.1), trained leave-one-match-out on dev matches
degraded with seed 101 and labelled keep2, every 5th alive frame plus its team-swapped copy.
Diagnostics, seed 0, D12 stress, keep2:
- alive gap frames: accuracy 0.873 against spread's sign 0.834; log-loss 0.285;
- after a true in-gap hand-over (150) it favours the new team within 10 s for 0.880, median
  lag 0.3 s (p90 2.8 s), within 1 s for 0.633, 3 s 0.807. Spread: a median 3 s (section 10);
- leakage check, J03WMX: alive-frame accuracy 0.868 when left out, 0.885 when in the training
  set. Two processes give byte-identical codes.

Proxy league against `touch_gapteam`, D12 stress (the decision run, keep2):

| date | method | spellF1 | interval | macroF1 | possessions | gain | verdict |
|---|---|---|---|---|---|---|---|
| 2026-11-04 | touch_gapteam | 0.4209 | 0.4022..0.4334 | 0.7831 | 1837 | -- | -- |
| 2026-11-04 | touch_c3 (w 0.5, all) | 0.4731 | 0.4503..0.4848 | 0.7982 | 1633 | +0.0522 (+0.0432..+0.0566) | ADOPT, 5/5 |
| 2026-11-04 | **touch_c3_w1 (w 1, all)** | 0.4989 | 0.4791..0.5148 | 0.8051 | 1528 | **+0.0780 (+0.0704..+0.0884)** | **ADOPT, 5/5** |
| 2026-11-04 | touch_c3_gaps (w 0.5, gaps) | 0.4310 | 0.4123..0.4429 | 0.7870 | 1862 | +0.0101 (+0.0066..+0.0144) | better, below the adoption bar, 5/5 |
| 2026-11-04 | touch_c3_only (w 0.5, no spread) | 0.4598 | 0.4405..0.4735 | 0.7950 | 1775 | +0.0390 (+0.0325..+0.0470) | ADOPT, 5/5 |

Robustness (gain over `touch_gapteam` on the same run; the SPEED and NOISE runs are plain v1):

| date | truth, stress | touch_gapteam | touch_c3 | touch_c3_w1 |
|---|---|---|---|---|
| 2026-11-04 | DFL, D12 | 0.3791 | 0.4119, +0.0328 (+0.0279..+0.0424), 5/5, ADOPT | 0.4260, +0.0469 (+0.0375..+0.0568), 5/5, ADOPT |
| 2026-11-04 | keep2, plain v1 | 0.4942 | 0.5319, +0.0377 (+0.0299..+0.0449), 5/5, ADOPT | 0.5450, +0.0508 (+0.0417..+0.0624), 5/5, ADOPT |
| 2026-11-04 | keep2, SPEED=0.9 | 0.4828 | 0.5249, +0.0421 (+0.0330..+0.0475), 5/5, ADOPT | 0.5421, +0.0593 (+0.0498..+0.0694), 5/5, ADOPT |
| 2026-11-04 | keep2, NOISE=0.1 | 0.4956 | 0.5359, +0.0403 (+0.0350..+0.0500), 5/5, ADOPT | 0.5464, +0.0508 (+0.0429..+0.0633), 5/5, ADOPT |
| 2026-11-04 | keep2, D12 + 117 s busy pre-kickoff | 0.4228 | 0.4737, +0.0510 (+0.0409..+0.0549), 5/5, ADOPT | 0.4988, +0.0760 (+0.0681..+0.0880), 5/5, ADOPT |

(DFL run, other rows: touch_c3_gaps +0.0104, better, below the bar; touch_c3_only +0.0306,
ADOPT.)

Step 2 on the stressed proxy, `gap_check.py --decoded` (label-free aggregates, keep2 truth as
the target; columns as section 10):

| method | gap H/(H+A) | gap D | seen D | poss/90 | in-gap poss/90 | in-gap <2 s/90 | in-gap H<->A /90 |
|---|---|---|---|---|---|---|---|
| touch_gapteam | 0.528 | 0.751 | 0.205 | 347.3 | 46.1 | 2.8 | 10.3 |
| touch_c3 | 0.521 | 0.755 | 0.216 | 308.7 | 47.1 | 2.5 | 14.2 |
| touch_c3_w1 | 0.522 | 0.756 | 0.224 | 288.7 | 47.6 | 3.2 | 16.9 |
| truth (keep2) | 0.541 | 0.716 | 0.256 | 275.0 | 56.4 | 4.0 | 30.3 |

Takeaways:
- **The gain comes from reading the model on every frame**, not from gaps (gaps only:
  +0.010). It removes false possession changes where the ball is seen: predicted possessions
  1837 to 1528, against 1455 true. In gaps it adds hand-overs (10 to 17 per 90; truth 30).
- **It is not a keep2 artefact**: ADOPT on DFL truth too, and on every stress, at both
  weights. Weight 1 beats 0.5 on every run, by +0.010 to +0.026.
- Dead ball barely moves (gap D 0.751 to 0.756, seen D 0.205 to 0.224), as intended: dead ball takes
  the better team's likelihood, so the model only chooses between the teams.
- **Not yet checked on BAR-ATH.** The model reads absolute positions and speeds learned on
  IDSSE; SPEED=0.9 and NOISE=0.1 hold, but BAR-ATH's missing players inside gaps (spread known
  0.89 against 0.98, section 9) are not simulated.

BAR-ATH, `tools/gap_check.py --decoded --bar-ath --methods touch_gapteam touch_c3 touch_c3_w1`,
run by miha on 2026-11-04. These are label-free aggregates, with columns as in section 10 (tags:
140 attacking, 140 defending, 4 without a team, 61,216 frames, mirror agree 1.000, home share
0.699):

| method | gap H/(H+A) | gap D | seen D | poss/90 | in-gap poss/90 | in-gap <2 s/90 | in-gap H<->A /90 | tag gap | tag seen |
|---|---|---|---|---|---|---|---|---|---|
| touch_gapteam | 0.486 | 0.776 | 0.189 | 392.6 | 58.6 | 13.6 | 19.5 | 0.721 | 0.813 |
| touch_c3 | 0.481 | 0.784 | 0.202 | 309.3 | 57.8 | 13.6 | 19.5 | 0.683 | 0.857 |
| touch_c3_w1 | 0.476 | 0.783 | 0.208 | 283.8 | 57.8 | 14.4 | 18.7 | 0.674 | 0.868 |

- **Both fail D22's pre-registered check:** tag agreement inside gaps falls from 0.721 to 0.683
  and 0.674, as it did for `touch_gapteam_k2` (0.676). Every other rule passes. Seen D for
  w 1 is at the 0.02 edge.
- **With the ball seen, tag agreement rises clearly** (0.813 to 0.857 and 0.868). That agrees
  with the proxy, where most of the gain is on seen frames.
- **Inside gaps BAR-ATH disagrees with the proxy**, where team accuracy on both-alive gap frames
  rose from 0.814 to 0.867 (keep2, D12). Candidate causes, not yet told apart:
  - thin player tracking in BAR-ATH gaps (a team under 5 players on 11% of gap frames, which
    the simulator does not reproduce); the model reads counts, widths and depths directly;
  - the tags being slow phases against a fast model;
  - proxy gaps being mostly dead ball (0.716) where BAR-ATH's are not.
- Possessions per 90 fall further than the proxy factor: 0.79 and 0.72 against 0.89 and 0.83.
  `touch_c3`'s 309 is near the window's 306. That is described only, never chosen on (L2).

Proxy-side split by ball seen (scratch diagnostic, keep2, D12, 3 seeds): on both-alive frames
`touch_gapteam` gets the team right on 0.852 (seen) and 0.814 (hidden), `touch_c3_w1` on 0.893
and 0.867. The true-team oracle on seen frames is worth +0.266 and +0.188, against +0.045 and
+0.038 in gaps. Recall of true possessions lying mostly where the ball is seen: 0.528 and 0.557
(1,156 per seed); mostly in a gap: 0.258 and 0.334 (299 per seed).

After the BAR-ATH check, the model read only where the ball is seen (`frame_where` "seen"), gaps
left to spread. **Settings tried: 2 more (6 in all).** These were chosen after seeing the BAR-ATH
result (tag agreement rose on seen frames and fell in gaps); D22's rules are unchanged.

| date | truth, stress | touch_gapteam | touch_c3_seen (w 0.5) | touch_c3_seen_w1 (w 1) |
|---|---|---|---|---|
| 2026-11-04 | **keep2, D12 (the decision run)** | 0.4209 | 0.4619 (0.4397..0.4735), 1614 poss, +0.0411 (+0.0335..+0.0452), 5/5, ADOPT | 0.4864 (0.4665..0.5024), 1491 poss, **+0.0655 (+0.0584..+0.0757), 5/5, ADOPT** |
| 2026-11-04 | DFL, D12 | 0.3791 | 0.4008, +0.0217 (+0.0168..+0.0289), 5/5, ADOPT | 0.4120, +0.0329 (+0.0238..+0.0408), 5/5, ADOPT |
| 2026-11-04 | keep2, plain v1 | 0.4942 | 0.5232, +0.0290 (+0.0221..+0.0356), 5/5, ADOPT | 0.5346, +0.0404 (+0.0322..+0.0501), 5/5, ADOPT |
| 2026-11-04 | keep2, SPEED=0.9 | 0.4828 | 0.5162, +0.0334 (+0.0261..+0.0380), 5/5, ADOPT | 0.5320, +0.0491 (+0.0414..+0.0580), 5/5, ADOPT |
| 2026-11-04 | keep2, NOISE=0.1 | 0.4956 | 0.5274, +0.0318 (+0.0280..+0.0410), 5/5, ADOPT | 0.5364, +0.0408 (+0.0331..+0.0514), 4/5, ADOPT |
| 2026-11-04 | keep2, D12 + 117 s busy pre-kickoff | 0.4228 | 0.4629, +0.0401 (+0.0325..+0.0446), 5/5, ADOPT | 0.4860, +0.0633 (+0.0555..+0.0735), 5/5, ADOPT |

Visible-ball Step 0, `gap_check.py --seen-ceiling` (keep2, D12). Team errors on seen frames
where truth and prediction are both alive, by kind. Each oracle gives the true team on those
frames only:

| method | boundary (+-1.5 s): share, oracle | invented possession: share, oracle | inside a matched one: share, oracle | invented, whole spell | all seen | true D, pred team | true team, pred D |
|---|---|---|---|---|---|---|---|
| touch_gapteam | 0.054, +0.0316 | 0.044, +0.1062 | 0.006, +0.0102 | +0.2187 | +0.2656 | 0.090 | 0.039 |
| touch_c3_w1 | 0.047, +0.0339 | 0.022, +0.0524 | 0.006, +0.0070 | +0.1403 | +0.1877 | 0.084 | 0.051 |
| touch_c3_seen_w1 | 0.047, +0.0375 | 0.022, +0.0528 | 0.005, +0.0073 | +0.1422 | +0.1911 | 0.084 | 0.051 |

- **Invented possessions are most of the visible-ball loss**: decoded possessions that match no
  true one (+0.14 of +0.19). C3 halves their frames (0.044 to 0.022). Timing is small
  (+0.034), as D18 found.
- **Dead-ball errors are 13.5% of seen frames** and lie outside the team oracle (C2b, D15).

Tag diagnostic, proxy side (`gap_check.py --tagdiag`, keep2, D12): agreement with the true team
on alive gap frames (share of the split's frames read):

| split | share | model | spread | touch_gapteam | touch_c3 | touch_c3_w1 |
|---|---|---|---|---|---|---|
| all gap frames | 1.000 | 0.866 (1.00) | 0.826 (0.98) | 0.814 (0.71) | 0.857 (0.70) | 0.867 (0.70) |
| full tracking | 0.977 | 0.872 | 0.826 | 0.821 | 0.863 | 0.873 |
| thin tracking (a team under 5) | 0.023 | 0.564 (0.88) | -- (0.00) | 0.467 | 0.597 | 0.577 |
| ball missing <2 s | 0.137 | 0.831 | 0.759 | 0.815 | 0.835 | 0.842 |
| 2-5 s | 0.175 | 0.851 | 0.795 | 0.795 | 0.849 | 0.851 |
| 5-15 s | 0.274 | 0.887 | 0.865 | 0.847 | 0.884 | 0.893 |
| >15 s | 0.406 | 0.868 | 0.833 | 0.795 | 0.850 | 0.864 |

- With thin tracking the model is near a coin (0.564). The proxy has 2.3% such gap frames;
  BAR-ATH about 11% (section 9). The BAR-ATH run (`--tagdiag --bar-ath`) shows whether that
  explains the in-gap tag drop.

Weight 1 against 0.5, directly (keep2, D12):

| date | method | spellF1 | gain over touch_c3_seen | verdict |
|---|---|---|---|---|
| 2026-11-04 | touch_c3_seen_w1 | 0.4864 | +0.0244 (+0.0218..+0.0334) | ADOPT, 5/5 |

BAR-ATH, run by miha on 2026-11-04 (label-free). `gap_check.py --decoded --bar-ath --methods
touch_gapteam touch_c3_seen touch_c3_seen_w1`:

| method | gap H/(H+A) | gap D | seen D | poss/90 | in-gap poss/90 | in-gap <2 s/90 | in-gap H<->A /90 | tag gap | tag seen |
|---|---|---|---|---|---|---|---|---|---|
| touch_gapteam | 0.486 | 0.776 | 0.189 | 392.6 | 58.6 | 13.6 | 19.5 | 0.721 | 0.813 |
| touch_c3_seen | 0.482 | 0.777 | 0.201 | 311.9 | 59.5 | 14.4 | 19.5 | 0.736 | 0.857 |
| touch_c3_seen_w1 | 0.480 | 0.777 | 0.208 | 282.1 | 56.9 | 12.7 | 17.0 | 0.734 | 0.867 |

- **Both pass every D22 rule.** Tag agreement now rises in gaps too (0.721 to 0.734/0.736).
  Seen D for w 1 moves +0.019, at the 0.02 limit. Possessions per 90 fall further than the
  proxy factor (0.72 against 0.81 for w 1), inside the 250 to 393 band.

`gap_check.py --tagdiag --bar-ath`, agreement with the Smart Tagging team on gap frames (tags
cover them with home 0.612):

| split | share | tag H | model | spread | touch_gapteam | touch_c3 | touch_c3_w1 |
|---|---|---|---|---|---|---|---|
| all gap frames | 1.000 | 0.612 | 0.711 | 0.731 | 0.721 (0.45) | 0.683 (0.44) | 0.674 (0.44) |
| full tracking | 0.999 | 0.613 | 0.711 | 0.731 | 0.724 | 0.685 | 0.676 |
| thin tracking | 0.001 | -- | -- | -- | -- | -- | -- |
| ball missing <2 s | 0.203 | 0.618 | 0.697 | 0.781 | 0.738 | 0.702 | 0.707 |
| 2-5 s | 0.234 | 0.597 | 0.670 | 0.760 | 0.762 | 0.713 | 0.700 |
| 5-15 s | 0.352 | 0.653 | 0.606 | 0.611 | 0.728 | 0.670 | 0.611 |
| >15 s | 0.210 | 0.555 | 0.948 | 0.854 | 0.497 (0.18) | 0.561 (0.18) | 0.709 (0.18) |

- **Thin tracking is ruled out**: 0.1% of tagged gap frames (the thin row is too small to read).
- In the first 15 s of a gap the model agrees with the tags less than spread does (0.70 against
  0.78 under 2 s, 0.67 against 0.76 at 2 to 5 s). On the proxy it agreed more (0.83 against
  0.76). After 15 s it agrees far more (0.95 against 0.85). Either the tags lag fast changes or
  the model shifts on BAR-ATH early in gaps; label-free data cannot tell which. The model stays
  out of gaps.

## Section 14: C6 Step 0, the in-match adaptation ceiling, keep2 truth, simulator v1 + D12 ball stress (2026-11-05)

Setup: section 13 (5 dev matches x 3 seeds, D12 stress, `LABELS=keep2 PRIOR_LABELS=dfl`).
Reference `touch_c3_seen_w1` (D23, submitted). New: [solution/selftrain.py](../solution/selftrain.py)
(selection, cross-fitting, learners), [tools/selftrain_check.py](../tools/selftrain_check.py)
(`--diag`, `--ceiling`), and `semi_markov(frame_fn=...)` (default None, byte-identical).
Code at branch `miha/dev`. **Settings tried: 5 frame-level oracles, 6 league
oracle rows.** `selftrain.PARAMS` (learners, 60 s blocks, 5 folds, `min_hold_s` 1) were set once.

The oracle: a match-specific model g([base log-odds, features]), not symmetrised, fitted on
the match's own TRUE labels at frames a pseudo-labeller could pick. "hold" = ball seen, in play,
the touch engine's holder unchanged for >= 1 s (0.889 of seen alive frames, 0.396 of graded
frames); "seen" = every seen alive frame. It is cross-fitted: frames fall into 60 s blocks, block b
in fold b mod 5, and each fold is predicted by a g fitted on the other folds. So a gap is never
filled from labels of its own block. Learners: `logit` (L2 logistic), `gbm` (small HistGBM),
`pooled` (the frame model refitted on its training matches plus the match's picked frames,
weighted equally in total).

Frame level, `selftrain_check.py --diag --seeds 0` (sign accuracy / mean log-loss against the
true team, alive frames):

| split | frames | base | gbm hold | gbm seen | logit hold | logit seen | pooled hold |
|---|---|---|---|---|---|---|---|
| gap, all | 81503 | 0.873 / 0.285 | 0.865 / 0.308 | 0.875 / 0.296 | 0.884 / 0.287 | 0.888 / 0.281 | 0.877 / 0.286 |
| gap, missing <2 s | 10472 | 0.802 / 0.428 | 0.780 / 0.472 | 0.796 / 0.471 | 0.800 / 0.456 | 0.806 / 0.453 | 0.795 / 0.456 |
| gap, missing 2-5 s | 12826 | 0.823 / 0.360 | 0.822 / 0.379 | 0.827 / 0.369 | 0.830 / 0.365 | 0.838 / 0.362 | 0.838 / 0.355 |
| gap, missing 5-15 s | 21003 | 0.894 / 0.265 | 0.888 / 0.286 | 0.893 / 0.275 | 0.906 / 0.255 | 0.909 / 0.247 | 0.896 / 0.259 |
| gap, missing >15 s | 36679 | 0.897 / 0.232 | 0.890 / 0.252 | 0.903 / 0.236 | 0.913 / 0.232 | 0.916 / 0.226 | 0.902 / 0.232 |
| seen, not picked | 35414 | 0.826 / 0.401 | 0.814 / 0.430 | 0.818 / 0.419 | 0.829 / 0.414 | 0.823 / 0.394 | 0.829 / 0.408 |
| seen, picked | 282495 | 0.881 / 0.280 | 0.877 / 0.299 | 0.877 / 0.298 | 0.885 / 0.286 | 0.885 / 0.283 | 0.885 / 0.280 |

Proxy league, `selftrain_check.py --ceiling` (D12, keep2). Readings: `_gaps` = adapted model in
gaps, base where seen; `_all` = adapted everywhere; `_seen` = adapted where seen, nothing in gaps:

| date | method | spellF1 | interval | macroF1 | possessions | gain over touch_c3_seen_w1 | verdict |
|---|---|---|---|---|---|---|---|
| 2026-11-05 | touch_c3_seen_w1 | 0.4864 | 0.4665..0.5024 | 0.8012 | 1491 | -- | -- |
| 2026-11-05 | touch_c3_w1 (base model, also in gaps) | 0.4989 | 0.4791..0.5148 | 0.8051 | 1528 | +0.0125 (+0.0077..+0.0176) | better, below the adoption bar, 5/5 |
| 2026-11-05 | c6o_logit_hold_gaps | 0.5010 | 0.4825..0.5174 | 0.8059 | 1531 | +0.0147 (+0.0092..+0.0208) | better, below the adoption bar, 5/5 |
| 2026-11-05 | c6o_logit_hold_all | 0.5067 | 0.4899..0.5247 | 0.8062 | 1508 | +0.0204 (+0.0140..+0.0303) | ADOPT, 5/5 |
| 2026-11-05 | c6o_logit_hold_seen | 0.4922 | 0.4730..0.5081 | 0.8014 | 1474 | +0.0059 (-0.0001..+0.0116) | within noise, 4/5 |
| 2026-11-05 | c6o_gbm_hold_gaps | 0.4997 | 0.4804..0.5150 | 0.8051 | 1541 | +0.0133 (+0.0078..+0.0187) | better, below the adoption bar, 5/5 |
| 2026-11-05 | c6o_logit_seen_gaps | 0.5013 | 0.4831..0.5183 | 0.8061 | 1532 | +0.0149 (+0.0102..+0.0220) | better, below the adoption bar, 5/5 |
| 2026-11-05 | c6o_logit_seen_all | 0.5059 | 0.4915..0.5268 | 0.8067 | 1510 | +0.0196 (+0.0162..+0.0325) | better, below the adoption bar, 5/5 |

Takeaways:
- **Adaptation itself is worth under 0.01, even with true labels.** Compare each reading with
  the same reading done by the base model: in gaps +0.0022 (0.5010 against `touch_c3_w1`'s
  0.4989), everywhere +0.0078, seen only +0.0059. The one ADOPT row (+0.0204) gets +0.0125 of
  it from reading the unadapted model inside gaps, which D23 forbids until explained.
- **It does not touch the D23 question.** In the first 2 s of a gap no learner beats the base
  (0.780 to 0.806 against 0.802), and at 2 to 5 s it gains at most +0.015.
- The learner is not the limit: the pooled refit is no better than the logistic residual. A
  wider selection (every seen frame, true labels) adds nothing over "hold".
- A real pseudo-labeller can only do worse than this oracle. **Step 0 does not clear; Steps 1
  and 2 were not run** (D24).

## Section 15: visible-ball invented possessions and a learned dead-ball model, keep2 truth, simulator v1 + D12 ball stress (2026-11-05)

Setup: section 13 (5 dev matches x 3 seeds, D12 stress, `LABELS=keep2 PRIOR_LABELS=dfl`).
Reference `touch_c3_seen_w1` (D23, submitted). New: `gap_check.py --seen-ceiling` attribution
(`invented_records`, `evidence_llr`, `holder_start`, group oracles), `semi_markov`'s
`prepare` / `decode` split (all 14 registered decoder methods byte-identical on two dev matches,
seed 0), [solution/deadmodel.py](../solution/deadmodel.py),
[tools/deadmodel_check.py](../tools/deadmodel_check.py). Code at branch `miha/dev`.
**Settings tried: 3 league settings** (`touch_takes` False, keep2-fitted priors, both), **1
dead-ball model** (`deadmodel.PARAMS` and its feature list set once, not tuned).

Step 0, `gap_check.py --seen-ceiling --methods touch_c3_seen_w1` (reproduces section 13:
spellF1 0.4864, invented share 0.022, "invented, whole" +0.1422). Decoded possessions that match
no true one at IoU 0.5 and lie mostly on seen frames, grouped by what truth holds under them:

| truth under it | n/seed | share of invented error frames | p50 s | under 5 s | start / end within 1.5 s of a true change | a true change inside |
|---|---|---|---|---|---|---|
| other team (phantom hand-over) | 157.7 | 0.549 | 2.6 | 0.795 | 0.353 / 0.442 | 0.178 |
| own team (wrong extent) | 307.0 | 0.306 | 5.6 | 0.457 | 0.452 / 0.443 | 0.237 |
| dead ball | 141.3 | 0.083 | 7.2 | 0.377 | 0.509 / 0.177 | 0.292 |
| mixed | 16.0 | 0.061 | 11.7 | 0.042 | 0.500 / 0.396 | 0.958 |

What put the phantom hand-overs there (other team; "dominant": largest evidence towards the
spell's team over its first 1 s / all of it; "removed": gone when only that source is switched
off and the match re-decoded):

| source | dominant, start | dominant, held | removed | median nats towards it |
|---|---|---|---|---|
| nearest player | 0.087 | 0.053 | 0.207 | +7.26 |
| touch holder | 0.740 | 0.797 | 0.486 | +22.40 |
| spread | 0.038 | 0.015 | 0.021 | +0.00 |
| frame model | 0.129 | 0.135 | 0.025 | -0.69 |

The holder at their start: touch hand-over (`touch_takes`) 0.535, touch from none 0.133, control
0.116, held before 0.044, none 0.171. Total margin p10 / p50 / p90: 8.5 / 26.4 / 123.7 nats. The
"none" category is empty by construction: H and A share the duration prior and every transition
costs 0, so a decoded team spell always has net evidence for its team.

Oracles (true team on the seen frames of the group's invented possessions), against
`touch_c3_seen_w1`: all invented +0.1422; phantom hand-overs +0.0719; phantoms the holder started
with a touch hand-over +0.0364. With `touch_takes` False: phantoms 157.7 to 99.0 per seed, now
led by the nearest player (dominant at the start 0.418), boundary oracle +0.0375 to +0.0521
(real hand-overs come late).

Proxy league, the decision run (keep2, D12), against `touch_c3_seen_w1`:

| date | method | spellF1 | interval | macroF1 | possessions | gain | verdict |
|---|---|---|---|---|---|---|---|
| 2026-11-05 | touch_c3_seen_w1 | 0.4864 | 0.4665..0.5024 | 0.8012 | 1491 | -- | -- |
| 2026-11-05 | touch_c3_seen_w1_hold (touch_takes False) | 0.4861 | 0.4718..0.5104 | 0.8017 | 1317 | -0.0002 (-0.0056..+0.0175) | within noise, 2/5 |
| 2026-11-05 | touch_c3_seen_w1_k2 (priors on keep2) | 0.4964 | 0.4742..0.5101 | 0.8014 | 1430 | +0.0101 (+0.0040..+0.0114) | better, below the adoption bar, 5/5 |
| 2026-11-05 | touch_c3_seen_w1_hold_k2 (both) | 0.4909 | 0.4769..0.5160 | 0.8014 | 1286 | +0.0046 (-0.0008..+0.0230) | within noise, 2/5 |

Robustness (gain over `touch_c3_seen_w1` on the same run; SPEED and NOISE runs are plain v1):

| date | truth, stress | touch_c3_seen_w1 | _hold | _k2 | _hold_k2 |
|---|---|---|---|---|---|
| 2026-11-05 | DFL, D12 | 0.4120 | 0.3821, -0.0299 (-0.0366..-0.0160), 0/5, WORSE | 0.4133, +0.0014 (-0.0027..+0.0048), 2/5, within noise | 0.3828, -0.0291 (-0.0376..-0.0171), 0/5, WORSE |
| 2026-11-05 | keep2, plain v1 | 0.5346 | 0.5022, -0.0324 (-0.0403..-0.0159), 0/5, WORSE | 0.5437, +0.0091 (+0.0074..+0.0137), 5/5, better, below the bar | 0.5061, -0.0285 (-0.0343..-0.0088), 0/5, WORSE |
| 2026-11-05 | keep2, SPEED=0.9 | 0.5320 | 0.5018, -0.0302 (-0.0404..-0.0154), 0/5, WORSE | 0.5411, +0.0092 (+0.0044..+0.0123), 5/5, better, below the bar | 0.5069, -0.0250 (-0.0341..-0.0092), 0/5, WORSE |
| 2026-11-05 | keep2, NOISE=0.1 | 0.5364 | 0.5004, -0.0360 (-0.0449..-0.0201), 0/5, WORSE | 0.5464, +0.0100 (+0.0054..+0.0123), 5/5, better, below the bar | 0.5045, -0.0319 (-0.0418..-0.0162), 0/5, WORSE |
| 2026-11-05 | keep2, D12 + 117 s busy pre-kickoff | 0.4860 | 0.4875, +0.0014 (-0.0018..+0.0197), 3/5, within noise | 0.4968, +0.0108 (+0.0042..+0.0108), 5/5, better, below the bar | 0.4933, +0.0073 (+0.0036..+0.0261), 3/5, better, below the bar |

Step 3, Step 0: `deadmodel_check.py` (frame level, no decoding). A players-only P(dead): speed
rank and its past / future means over 1, 3 and 8 s, speed dispersion, ranked slow share, max and
p25 speed, gyration, |tilt|, tracked count, nearest player to a touchline and a goal line, box
count, a player beyond a line (past / future 3 s). HistGBM as C3, leave-one-match-out, trained on
TRAIN_SEED. AUC of dead against alive; "start" / "end" = within 2 s of a true change into / out
of D. The bar, fixed before the first run: AUC gain >= 0.03 over the speed rank and wins >= 4/5
at both starts and ends.

| stress | region | model | speed rank | gain | wins | speed rank, 1 s mean | 3 s mean |
|---|---|---|---|---|---|---|---|
| D12 | start | 0.7823 | 0.7412 | +0.0412 | 5/5 | 0.7390 | 0.7149 |
| D12 | end | 0.8449 | 0.7926 | +0.0523 | 5/5 | 0.7971 | 0.7942 |
| D12 | interior | 0.9880 | 0.9461 | +0.0419 | 5/5 | 0.9495 | 0.9567 |
| D12, SPEED=0.9 | start | 0.7318 | 0.7412 | -0.0093 | 1/5 | 0.7390 | 0.7149 |
| D12, SPEED=0.9 | end | 0.7735 | 0.7926 | -0.0191 | 0/5 | 0.7971 | 0.7942 |
| D12, SPEED=0.9 | interior | 0.9776 | 0.9461 | +0.0315 | 5/5 | 0.9495 | 0.9567 |

Edge timing at a 45% dead call rate, D12 (1,524 true stoppages >= 2 s): p50 |lag| at starts
0.76 s against the speed rank's 0.96 s, at ends 0.52 s against 0.88 s. Under SPEED=0.9: 0.92 and
0.68 s.

Takeaways:
- **The phantom hand-overs are the touch engine's.** It is dominant over 0.80 of them, starts
  0.54 with a `touch_takes` hand-over, and removing it alone removes 0.49. Most sit away from any
  true change: touches keep2 does not count as control.
- **`touch_takes` False does not fix it.** It cuts the phantoms by about a third but times real
  hand-overs late, and the nearest player invents its own. It is within noise on the decision run
  and WORSE on four of six runs.
- **keep2 priors gain +0.009 to +0.011 on every keep2 run** (5/5), never 0.02; DFL is flat.
- **The ceiling for a selective fix of touch hand-overs is +0.036** with perfect knowledge. A real
  filter would get a fraction of it, so nothing new was built.
- **The dead-ball model passes the bar on D12 and fails it under SPEED=0.9**, where the speed rank
  is unchanged by design. Its edge gain rests on position features that a shrunk pitch moves. Only
  the interior gain (false stoppages) holds. No decoder work was done.

---

## Section 16: dead ball (Track 1): E10 whistle ceiling, a tempo stress that keeps geometry, dead-ball model v2, keep2 truth, simulator v1 + D12 ball stress (2026-11-05)

Setup: section 15 (5 dev matches x 3 seeds, D12 stress, `LABELS=keep2 PRIOR_LABELS=dfl`).
Reference `touch_c3_seen_w1` (D23, submitted). Code at branch `miha/dev`.
**Settings tried: 1 model setting** (dead-ball model v2) **and 2 decoder settings**
(`touch_c3_seen_w1_dm`, `touch_c3_seen_w1_dms`, the second picked after seeing the first's
failed BAR-ATH check), plus oracle repairs and 2 capacity oracles that tune nothing.

**The tempo stress, decided before anything was scored with it.** `simulate.retime(match, F)`
replays each clean match F times as fast before `degrade`: new frame k shows the old match at
time k * F, positions interpolated, labels and halves those of old frame floor(k * F). Every
speed is x F, where anyone stands is unchanged, and truth is retimed with it (the match is 1 / F
as long). `make proxy TEMPO=0.9`, `deadmodel_check.py --tempo 0.9`, `deadball_check.inputs(...,
tempo)`. **From here on, `TEMPO=0.9`, not `SPEED=0.9`, is the D9 tempo check for any method that
reads positions** (line distances, box counts, spread, gyration). `simulate.scale_speeds` reaches
x0.9 by shrinking the pitch towards its centre, which also moves every player 10% away from the
lines; a position feature then fails for a reason BAR-ATH does not have. `SPEED=0.9` stays
available and is still reported, so tables can be compared with sections 3 to 15; for methods
that read speed only, the two should agree. What `TEMPO` changes besides speed: possessions and
stoppages last 1 / F as long, while the decoder's duration priors are fitted at IDSSE tempo. That
is part of slower play too, but it makes `TEMPO` a little harsher than a pure speed change on
anything that reads durations.

**Dead-ball model v2, fixed before its first run.** `deadmodel.DROPPED["v2"]`: the D25 model
(`v1`, unchanged, no new setting) without the columns read from absolute positions in metres
(`gyration`, `tilt_abs`, `touchline_min`, `goalline_min`, `in_box`) and the tracked count. It
keeps the speed rank and its past / future means and changes, the ranked slow share, max and p25
speed, speed CV and its past / future means, and the beyond-line windows. Same `PARAMS`, same
training (leave-one-match-out on the dev matches, `TRAIN_SEED`). The bar is `deadmodel_check`'s,
unchanged from D25: AUC gain >= 0.03 over the speed rank and wins >= 4/5, at both starts and
ends, **under D12 and under D12 + `TEMPO=0.9`**. `SPEED=0.9` is reported, not decisive.

Step 0 of E10, the whistle ceiling (audio exists: AAC stereo 48 kHz over the full 7,972.7 s, by
miha's `afinfo`; the rules read as allowing it, PLAN question 9). `deadball_check.py` repairs
from the truth only the stoppages a referee whistles: restarts `free_kick` (fouls, offsides),
`penalty`, `goal`, `referee_ball`. "start" moves their start edges, "+missed" also adds their
missed stretches whole (generous: a whistle does not say when play resumes), "all" repairs those
types entirely (upper bound, including end edges the whistle does not mark). Against
`touch_c3_seen_w1`, D12 stress:

| truth | as is | perfect D | start edges | end edges | false stoppages | missed | whistle start | whistle start+missed | whistle all | throw-in | free kick | goal kick |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| keep2 | 0.4864 | +0.1469 | +0.0430 | +0.0376 | +0.0462 | +0.0174 | +0.0117 | +0.0150 | +0.0253 | +0.0448 | +0.0214 | +0.0200 |
| DFL | 0.4120 | +0.1036 | +0.0354 | +0.0294 | +0.0224 | +0.0165 | +0.0105 | +0.0133 | +0.0210 | +0.0357 | +0.0179 | +0.0167 |

514 true stretches per seed (free kicks 115, goals 12, penalties 2, dropped balls 7); 9.2%
missed; 183 false stoppages per seed (median 3.8 s), up from `touch_kickoff`'s 123 in section 8;
D recall 0.843, precision 0.854.
- **E10 closes at Step 0.** A perfect whistle is worth +0.012 (start edges) to +0.015 (with
  missed stretches) on keep2, under D4's 0.02 before any detector error. Its whistled types are a
  minority of stoppages, and their starts are already the better-placed ones (free-kick start
  error p50 +0.28 s).
- **Dead ball is a bigger lever on the current reference than D15 measured:** perfect D +0.147
  on keep2 (+0.104 on DFL, against D15's +0.089 on `touch_kickoff`). False stoppages grew to the
  largest single part (+0.046), then start edges (+0.043) and end edges (+0.038). Throw-ins are
  the largest type (+0.045), and no whistle marks them.

Step 0 of the learned dead-ball model, on the new bar (`deadmodel_check.py`, frame level, keep2;
gain = AUC over the speed rank, wins out of 5; the D12 rows of v1 reproduce section 15 exactly):

| stress | v1 start | v1 end | v1 interior | v2 start | v2 end | v2 interior | verdict |
|---|---|---|---|---|---|---|---|
| D12 | +0.0412, 5/5 | +0.0523, 5/5 | +0.0419 | +0.0299, 5/5 | +0.0213, 4/5 | +0.0390 | v1 clears, v2 does not |
| D12 + TEMPO=0.9 (decisive) | +0.0448, 5/5 | +0.0561, 5/5 | +0.0423 | +0.0325, 5/5 | +0.0254, 5/5 | +0.0394 | v1 clears, v2 does not |
| D12 + SPEED=0.9 (reference) | -0.0093, 1/5 | -0.0191, 0/5 | +0.0315 | +0.0022, 3/5 | -0.0006, 4/5 | +0.0334 | neither |

Edge timing at a 45% dead call rate, p50 |lag| start / end: D12 v1 0.76 / 0.52 s, v2 0.78 / 0.60 s,
speed rank 0.96 / 0.88 s; TEMPO v1 0.76 / 0.56 s, v2 0.80 / 0.68 s, speed rank 0.96 / 0.92 s.
- **D25's failure was the stress, not the model.** Under the tempo stress that keeps geometry,
  v1's edge gains are as large as under D12. Even v2 collapses under SPEED=0.9, and the only
  position columns left in v2 are the beyond-line windows, which a pitch shrunk 10% towards the
  centre switches off. TEMPO moves the slow-share threshold just as SPEED does, so that is not it.
- **Absolute positions carry part of the edge signal** (v1 over v2: +0.011 at starts, +0.031 at
  ends). v2 was the fallback; it is not needed.

**The decoder term, fixed before its first proxy-league run.** `decoder.DEFAULTS["deadm_w"]`
(0 = off, byte-identical): `deadmodel.log_ratio`, the model's log-odds minus its training prior's
(so the duration priors are not counted twice), block-averaged and added to D at weight 1. On
blocks where it is read, the speed likelihood is switched off (weight 1 - share read): the model
reads the speed rank itself, and keeping both would count speed twice. Method
`touch_c3_seen_w1_dm` = `touch_c3_seen_w1` + `deadm_w` 1, model v1. **1 decoder setting.**
Capacity oracles first (`deadmodel_check.py --ceiling`, proxy only, they read the truth):
`dmo_sign` gives each frame the model's own |ratio| with the truth's sign; `dmo_edges` uses the
model's ratio but the truth's sign within 2 s of a true change into or out of D. The ceiling
above both is the perfect-dead-ball repair, +0.147 (keep2, above).

Capacity oracles (keep2, D12), against `touch_c3_seen_w1` (0.4864):

| date | method | spellF1 | interval | macroF1 | possessions | gain | verdict |
|---|---|---|---|---|---|---|---|
| 2026-11-05 | dmo_sign (oracle: model size, true sign) | 0.6329 | 0.6120..0.6484 | 0.9079 | 1388 | +0.1465 (+0.1315..+0.1594) | 5/5 (oracle) |
| 2026-11-05 | dmo_edges (oracle: true sign within 2 s of D changes) | 0.5905 | 0.5738..0.6084 | 0.8834 | 1496 | +0.1041 (+0.0939..+0.1190) | 5/5 (oracle) |

The term can carry the whole perfect-dead-ball ceiling (+0.147) at the model's own strength.

Proxy league, `touch_c3_seen_w1_dm` against `touch_c3_seen_w1` (the decision run first; SPEED and
NOISE runs are plain v1, as in section 15):

| date | truth, stress | touch_c3_seen_w1 | touch_c3_seen_w1_dm | interval | macroF1 | possessions (ref -> dm) | gain | verdict |
|---|---|---|---|---|---|---|---|---|
| 2026-11-05 | **keep2, D12 (decision)** | 0.4864 | 0.5565 | 0.5359..0.5731 | 0.8573 | 1491 -> 1386 | +0.0702 (+0.0572..+0.0812) | **ADOPT**, 5/5 |
| 2026-11-05 | DFL, D12 | 0.4120 | 0.4601 | 0.4413..0.4760 | 0.8517 | 1491 -> 1386 | +0.0482 (+0.0391..+0.0585) | ADOPT, 5/5 |
| 2026-11-05 | keep2, D12 + TEMPO=0.9 | 0.4952 | 0.5652 | 0.5373..0.5724 | 0.8590 | 1543 -> 1436 | +0.0699 (+0.0577..+0.0778) | ADOPT, 5/5 |
| 2026-11-05 | keep2, plain v1 | 0.5346 | 0.6046 | 0.5854..0.6222 | 0.8675 | 1538 -> 1432 | +0.0700 (+0.0610..+0.0852) | ADOPT, 5/5 |
| 2026-11-05 | keep2, SPEED=0.9 | 0.5320 | 0.5658 | 0.5433..0.5794 | 0.8382 | 1578 -> 1501 | +0.0338 (+0.0229..+0.0438) | ADOPT, 5/5 |
| 2026-11-05 | keep2, NOISE=0.1 | 0.5364 | 0.6079 | 0.5871..0.6237 | 0.8654 | 1542 -> 1416 | +0.0715 (+0.0602..+0.0826) | ADOPT, 5/5 |
| 2026-11-05 | keep2, D12 + 117 s busy pre-kickoff | 0.4860 | 0.5554 | 0.5420..0.5780 | 0.8585 | 1494 -> 1400 | +0.0694 (+0.0628..+0.0847) | ADOPT, 5/5 |

True possessions: 1,455 (keep2), 2,191 (DFL). The reference reproduces section 15 on every
shared run. The smallest gain is under SPEED=0.9, the stress that switches off the beyond-line
windows; it still clears D4.

BAR-ATH check, D26's rules fixed before it ran; run by miha on 2026-11-05 (label-free aggregates;
proxy values keep2 + D12 in brackets):

| | touch_c3_seen_w1 | touch_c3_seen_w1_dm | D26 rule |
|---|---|---|---|
| D share of graded frames | 0.440 | 0.444 (0.435 -> 0.443) | 1: pass |
| stoppages per 90, median s | 140.2, 9.4 | 96.9, 18.1 (126.3, 11.4 -> 99.7, 17.4; truth 97.1, 19.0) | -- |
| frames turned to D / away from D | -- | 0.0472 / 0.0428 (0.0554 / 0.0478) | -- |
| set pieces with a decoded restart inside (±2 s) | 0.93 of 58 | 0.93 of 58 (throw-ins 24 -> 25, goal kicks 9 -> 8) | 2: pass (equal) |
| tag agreement, gaps / seen | 0.734 / 0.867 | 0.771 / 0.869 | 3: pass |
| gap D / seen D | 0.777 / 0.208 | 0.818 / 0.188 (proxy 0.751 -> 0.732 / 0.223 -> 0.249) | **4: fail** (both changes opposite to the proxy's; D22's literal rule fails too, gap D +0.041) |
| possessions per 90 | 282.1 | 243.9, x0.865 (x0.93) | 5: pass, near the 240 warning |
| home share of gap team frames | 0.480 | 0.473 | 6: pass |
| model: lean dead / Spearman with slowness / call agreement | -- | 0.441 / 0.856 / 0.873 (0.444 / 0.843 / 0.872) | 7: pass |

Model feature medians on BAR-ATH (proxy mean, range over 15 match-seeds): tracked 22.0 (21.8),
gyration 21.19 m (20.44, 20.11..20.95), touchline 3.97 m (3.27, 2.71..3.88), goal line 5.36 m
(6.37, 6.08..7.08), in box 3.87 (3.37, 2.49..3.73), beyond 0.068 (0.099, expected about 0.07),
speed CV 0.594 (0.547, 0.539..0.554). Warnings only (rule 7), all modestly outside the range.
- **The check fails (rule 4): `touch_c3_seen_w1` stays submitted (D26).** Every other rule passes.
- **The model does on BAR-ATH what it does on the proxy to stoppages:** fewer (140 to 97 per 90)
  and longer (9.4 to 18.1 s), close to the proxy's truth. The total D share does not move.
- **It moves dead ball the other way between gaps and seen frames.** On the proxy it moved D out of
  gaps (towards truth 0.716) and onto seen frames (towards 0.256). On BAR-ATH it moves D into
  gaps and off seen frames. Consistent with section 13's candidate cause, "proxy gaps being mostly
  dead ball where BAR-ATH's are not" (D7, D8: the simulator ties gaps to slow play), but the
  check cannot tell. The rise in gap tag agreement is partly selection: frames turned to D leave it.

**Seen-only variant, chosen by miha after the failed check, fixed before its first run.**
`touch_c3_seen_w1_dms`: `deadm_where` "seen", so the model is read only on frames with a seen
ball, and gap frames keep the speed likelihood as in the reference (as D23 did for the frame
model). **1 more decoder setting (2 in this section), picked after seeing D26's failed check.**
Its BAR-ATH check uses D26's rules unchanged, with one clarification written now, before any
of its numbers exist: in rule 4, where the proxy's change of gap D or seen D is smaller than
0.005, the BAR-ATH change must be within +-0.02 (D22's literal rule), since a sign is undefined.

Proxy league, `touch_c3_seen_w1_dms` against `touch_c3_seen_w1` (decision run first; SPEED and
NOISE runs plain v1):

| date | truth, stress | touch_c3_seen_w1 | _dms | interval | macroF1 | possessions (ref -> dms) | gain | verdict |
|---|---|---|---|---|---|---|---|---|
| 2026-11-05 | **keep2, D12 (decision)** | 0.4864 | 0.5225 | 0.5034..0.5391 | 0.8354 | 1491 -> 1449 | +0.0362 (+0.0272..+0.0457) | **ADOPT**, 5/5 |
| 2026-11-05 | DFL, D12 | 0.4120 | 0.4371 | 0.4199..0.4530 | 0.8299 | 1491 -> 1449 | +0.0251 (+0.0199..+0.0344) | ADOPT, 5/5 |
| 2026-11-05 | keep2, D12 + TEMPO=0.9 | 0.4952 | 0.5334 | 0.5066..0.5415 | 0.8367 | 1543 -> 1506 | +0.0382 (+0.0294..+0.0457) | ADOPT, 5/5 |
| 2026-11-05 | keep2, plain v1 | 0.5346 | 0.5697 | 0.5539..0.5893 | 0.8456 | 1538 -> 1493 | +0.0351 (+0.0306..+0.0492) | ADOPT, 5/5 |
| 2026-11-05 | keep2, SPEED=0.9 | 0.5320 | 0.5475 | 0.5310..0.5657 | 0.8275 | 1578 -> 1553 | +0.0156 (+0.0117..+0.0286) | better, below the adoption bar, 4/5 |
| 2026-11-05 | keep2, NOISE=0.1 | 0.5364 | 0.5733 | 0.5525..0.5883 | 0.8451 | 1542 -> 1484 | +0.0370 (+0.0275..+0.0456) | ADOPT, 5/5 |
| 2026-11-05 | keep2, D12 + 117 s busy pre-kickoff | 0.4860 | 0.5227 | 0.5123..0.5465 | 0.8368 | 1494 -> 1465 | +0.0366 (+0.0348..+0.0521) | ADOPT, 5/5 |

In the decision run `touch_c3_seen_w1_dm` scored 0.5565 (+0.0702): about half of its gain came
from reading the model inside gaps. Proxy-side (keep2, D12): D share 0.435 -> 0.451 (truth 0.441);
stoppages per 90 126.3 -> 111.8, median 11.4 -> 15.4 s (truth 97.1, 19.0); set-piece windows with
a restart 0.36 -> 0.43 (throw-ins 0.29 -> 0.41); gap D 0.751 -> 0.754; seen D 0.223 -> 0.248
(truth 0.256); possessions per 90 281.7 -> 273.9 (truth 275.0); gap home share 0.526 -> 0.526.
- **ADOPT on the decision run and on every stress but SPEED=0.9** (+0.016, 4/5), the shrunk-pitch
  stress that D26 made reported, not deciding, for position features.
- **The model now leaves gap dead ball alone (+0.003) and adds D where the ball is seen**, the
  change BAR-ATH's rule 4 can check.

BAR-ATH check of `touch_c3_seen_w1_dms`, D26 Update 2's rules fixed before it ran; run by miha on
2026-11-05 (label-free; proxy keep2 + D12 in brackets):

| | touch_c3_seen_w1 | touch_c3_seen_w1_dms | rule |
|---|---|---|---|
| D share | 0.440 | 0.426 (0.435 -> 0.451) | 1: pass |
| stoppages per 90, median s | 140.2, 9.4 | 117.3, 14.6 (126.3, 11.4 -> 111.8, 15.4) | -- |
| frames turned to D / away from D | -- | 0.0170 / 0.0302 (0.0407 / 0.0247) | -- |
| set pieces with a decoded restart inside | 0.93 of 58 | 0.95 of 58 (throw-ins 24 -> 26 of 26, goal kicks 9 -> 8) | 2: pass |
| tag agreement gaps / seen | 0.734 / 0.867 | 0.761 / 0.868 | 3: pass |
| gap D | 0.777 | 0.775 (proxy +0.003) | 4: pass (within +-0.02) |
| seen D | 0.208 | 0.187 (proxy +0.025) | **4: fail** (needs 0..+0.050; D22's literal rule fails too, -0.021) |
| possessions per 90 | 282.1 | 267.7, x0.949 (x0.972) | 5: pass |
| gap home share | 0.480 | 0.500 | 6: pass |

- **The check fails (rule 4, seen D): `touch_c3_seen_w1` stays submitted.**
- **Both variants move seen D the same way on BAR-ATH (-0.020, -0.021) and the other way on the
  proxy (+0.026, +0.025).** On BAR-ATH's seen frames the model calls less dead ball than the speed
  rank; it turns more frames away from D (0.030) than to D (0.017), the reverse of the proxy. It
  is systematic, not a gap effect. Whether BAR-ATH or the proxy is right there cannot be told
  without labels. The model's feature shifts (speed CV 0.594 against the proxy's 0.539..0.554, more
  players in the box, closer to the goal lines) are candidate causes, not yet checked.

## Section 17: E13 Step 0, broadcast alignment: second ball source closed, cut-aways as a dead-ball cue (2026-11-06)

Setup: `solution/broadcast.py` aligns the broadcast ATD to the tactical timeline from the two ATDs
alone. Ball-position votes give candidate offsets, and a one-way player-distance cost refines them
to the frame. A stretch is confident against rival offsets. A live chain of non-falling offsets
covers the live broadcast, and confident stretches below it are replays. Parameters were set once,
before the first run, and are not tuned. `tools/broadcast_align.py` (`make broadcast-align`),
run by miha on BAR-ATH on 2026-11-06. Label-free aggregates only.
**Settings tried: 1 alignment setting; 1 proxy what-if (no method change).**

Alignment (BAR-ATH):
- Both metadata files state 25.0 fps. Orientation: unmirrored, 10/10 confident; the other three
  mirrorings got 0/10.
- 115 valid stretches of at least 2 s hold 99.8% of the valid frames. 108 are confident (96.6% of
  stretch frames, 24 of them via neighbours' offsets). Cost p50 0.56 m (p90 0.86), best rival p50
  4.04 m, random offsets p50 4.66 m: the offsets are trustworthy.
- 98 live stretches (4,079 s), 7 replays (65 s; most replays are close-ups with no homography),
  3 "ahead" stretches (31 s, unexplained), 7 unaligned (146 s).
- One live offset per half. In the first half it wanders by up to 6 frames across three segments;
  the second half is 22,906 frames (916 s) further on, because the broadcast drops most of
  halftime. Drift inside stretches of at least 20 s: mostly 0 to 1 frame, 2 stretches over 5.
  Per-stretch offsets absorb it.
- Team codes agree on 95.5% of 38,057 paired players: same home/away order in both files.

What the TV shows at the moment of each graded tactical frame: live pitch 63.4%, cut away (no
homography) 28.3%, replay 0.9%, on pitch but not aligned 2.1%, not covered 5.2%.

**Ball: the broadcast sees the ball when the tactical ATD does.**
- Graded frames: tactical 59.3%, union with the broadcast 63.6% (both halves alike). Adding
  replays changes nothing.
- Where the tactical ball is missing, the TV at that moment: shows the pitch and the ball 10.5%,
  the pitch without the ball 23.3%, is cut away 52.9%, replay 1.6%, other 3.4%, not covered 8.2%.
- By gap length (share of gap frames, share filled): 3 to 10 s 14.6%, 32.2% filled; 10 to 30 s
  41.4%, 7.8%; 30 s and longer 43.8%, 5.9%. FINDINGS point 6's 76% was the share of on-pitch
  frames, and the long gaps are mostly cut-aways.
- Where both see it (73,592 frames), ball distance is p50 0.81 m, p90 6.27 m, p99 26.1 m, with
  12.4% over 5 m. One feed's ball is far off often. Airborne-ball parallax between two viewpoints
  or false detections are candidates, not checked.

Proxy what-if, optimistic bound (`make proxy METHODS=touch_c3_seen_w1` + D12 + `BALL_VISIBLE=0.636`;
`reveal_gaps` reveals whole gaps at random, while the real fill is partial and mostly short gaps):

| date | truth, stress | touch_c3_seen_w1 | with ball visible 0.636 | interval | macroF1 | possessions | gain | verdict |
|---|---|---|---|---|---|---|---|---|
| 2026-11-06 | keep2, D12 | 0.4864 | 0.4899 | 0.4718..0.5076 | 0.8029 | 1491 -> 1496 | +0.0035 (cross-run, not paired) | below PLAN's 0.02 stop rule |

**Cut-aways as a dead-ball cue.** Graded, covered frames, split by the submitted method's
prediction and by Smart Tagging:

| | frames | cut | replay | off pitch |
|---|---|---|---|---|
| predicted D (`touch_c3_seen_w1`) | 61,989 | 59.1% | 2.1% | 61.2% |
| predicted H or A | 88,586 | 9.4% | 0.2% | 9.6% |
| inside SET PIECES tags | 43,840 | 20.7% | 0.0% | 20.7% |
| outside SET PIECES tags | 106,735 | 33.7% | 1.3% | 35.0% |

- Off-pitch lift on predicted D: 6.4x. On set-piece tags it is 0.59x: the TV returns to the pitch
  for the restart, and the tag windows include the play after it.
- **Implied rates against truth.** The proxy confusion of `touch_c3_seen_w1` (keep2, D12, 5 dev x
  3 seeds) is P(true D | predicted D) = 0.854 (0.79 to 0.91 per run) and P(true D | predicted
  live) = 0.122 (0.09 to 0.15). Solving the two rows above for P(off pitch | true D) and
  P(off pitch | true live) gives about 0.71 and 0.01; over the per-run range, 0.68 to 0.77 and 0 to
  0.03. This assumes cut-aways are independent of our errors given the truth, which is unverified.
  If it holds, about 9 in 10 cut-away frames inside predicted live play are dead ball we miss
  (about 8,500 frames). On-pitch frames inside predicted D are a weak live signal (about 37% live).

**Ball merge, paired (D27 reopened by miha, 2026-11-06).** `solution/broadcast.fuse_ball` fills
the broadcast ball into tactical gaps only. It never overwrites a tactical ball, keeps a ball
within 40 m/s + 3 m of the tactical ball on each side, and keeps filled runs of at least 5 frames.
Proxy imitation: `tools/broadcast_fill_check.py` gives each simulated gap its clean ball back as one
chunk at the gap's start or end, at BAR-ATH's fill share per gap length (0.573 / 0.322 / 0.078 /
0.059). This runs before the D12 stress, and the comparison is paired on the same degraded inputs.
**Settings tried: 1** (profile and placement fixed from section E before the run).

| date | truth, stress | touch_c3_seen_w1 | merged input | ball seen | gain (paired) | wins | verdict |
|---|---|---|---|---|---|---|---|
| 2026-11-06 | keep2, D12 | 0.4864 | 0.4902 | 0.599 -> 0.642 | +0.0038 (+0.0006..+0.0102) | 5/5 | better, below the adoption bar |

- The base reproduces the decision reference (0.4864) exactly, and the visibility rise matches
  BAR-ATH's (59.3% -> 63.6%).
- The gain is real (the interval excludes 0, 5/5) but a fifth of D4's 0.02.

**Cut-away cue, settings fixed before its first proxy run (2026-11-06).** BAR-ATH section G
(miha's second run): 128 off-pitch runs on graded frames (108.8 per 90), length p10 3.3 / p50 8.5 /
p90 31.5 s, 77.9% of off-pitch time in runs of at least 10 s. Against predicted D runs of 0-5 /
5-15 / 15-30 / 30+ s: any cut-away 17.1 / 52.0 / 83.8 / 100%, mean off share 11.1 / 40.3 / 55.7 /
75.9%. The TV cuts away p50 0 to 1.2 s after the run starts and is back p50 0 to 4 s before it
ends. Inside predicted live: 43 off-pitch runs next to a predicted D run (232 s) and 13 isolated
ones (81 s). Around set-piece tags, corners, goal kicks, free kicks and kickoffs after goals are
mostly off pitch before the tag; throw-ins only 21-36%. Section H (merge on BAR-ATH): 6,596
frames filled (196 rejected by the speed gate, 61 as short runs), ball seen 59.3% -> 63.4%. The
method relabels 0.9% of graded frames (14.8% of the filled ones). D share 0.440 -> 0.439,
possessions per 90 282.1 -> 286.4.

From these, `solution/cutaway.py` imitates the cue on the proxy from the truth:
- P(any cut-away) by true stoppage length 0.171 / 0.520 / 0.838 / 1.0. When cut away, one chunk
  covers 0.649 / 0.775 / 0.665 / 0.759 of the stoppage (mean off share / hit rate), and the
  uncovered rest is split with a front share drawn from U(0, 0.5).
- **Pessimistic (decides):** with p 0.494 (43 / 87 runs with a cut) the cut runs past the restart
  for U(2.1, 8.7) s, and 2.29% of live frames (81 s / 3,543 s predicted live) get other
  cut-aways of U(2.39, 11.13) s. **Optimistic (reported):** no spill, 1% false.
- Not imitated: BAR-ATH's 5.2% of frames with no broadcast mapped.

Methods `touch_c3_seen_w1_tv` (pessimistic table) and `_tv_opt`: the reference plus a decoder
term `tv_w` 1 (block off-share likelihood, `priors["tv_log"]` fitted leave-one-match-out on
synthetic cues of the other dev matches, seeds 0-2). `tools/cutaway_check.py`, paired on the
same D12 inputs. Settings tried: 1 weight x 2 variants. The pessimistic one decides, and
`LABELS=dfl` must not be WORSE.

Results (`tools/cutaway_check.py`, 5 dev x seeds 0-2, D12; paired against `touch_c3_seen_w1`):

| date | truth | variant | method | spellF1 | gain (paired) | wins | verdict |
|---|---|---|---|---|---|---|---|
| 2026-11-06 | **keep2 (decision)** | pessimistic | touch_c3_seen_w1_tv | 0.5051 | +0.0187 (+0.0125..+0.0373) | 4/5 | better, below the adoption bar |
| 2026-11-06 | keep2 | optimistic | touch_c3_seen_w1_tv_opt | 0.5244 | +0.0380 (+0.0325..+0.0540) | 5/5 | ADOPT |
| 2026-11-06 | DFL | pessimistic | touch_c3_seen_w1_tv | 0.4189 | +0.0070 (+0.0048..+0.0250) | 3/5 | better, below the adoption bar |
| 2026-11-06 | DFL | optimistic | touch_c3_seen_w1_tv_opt | 0.4354 | +0.0235 (+0.0171..+0.0361) | 5/5 | ADOPT |

Reference 0.4864 (keep2) and 0.4120 (DFL), as in section 16. The synthetic cue, against the truth
and against the reference's own prediction (BAR-ATH's real cue against our prediction: 0.612 on
predicted D, 0.096 on predicted live):

| variant | off on true D | off on true live | off on predicted D | off on predicted live | table P(off \| alive), P(off \| dead) |
|---|---|---|---|---|---|
| pessimistic | 0.738 | 0.077 | 0.678 | 0.130 | 0.0774, 0.7375 |
| optimistic | 0.651 | 0.009 | 0.577 | 0.072 | 0.0086, 0.6512 |

- **The decision variant misses D4 by 0.0013** (+0.0187 against 0.02; interval excludes 0, 4/5).
  Under the rule fixed before the run, the cue is not adopted.
- **The real cue lies between the two variants.** On predicted D, BAR-ATH's 0.612 is between
  0.577 and 0.678. On predicted live, its 0.096 is between 0.072 and 0.130. The pessimistic variant
  has more cut-aways in live play than BAR-ATH shows, which makes it harder on the cue than the
  real broadcast. Its assumption that every off-pitch run next to a predicted stoppage is TV lag
  is too strong.
- DFL truth is not WORSE in either variant.
- The keep2 and DFL cue rates are identical: relabelling changes H/A, not D runs, so the
  synthetic cue is the same.

**D31: the calibrated variant (decides).** `tools/cutaway_calibrate.py` (reads no score). Its grid is
spill_p 0 to 0.5 in steps of 0.05 and false_share 0 to 0.025 in steps of 0.0025. It chose spill_p
0.15 and false_share 0.0200, whose cue splits 0.610 / 0.096 against the reference's prediction
(BAR-ATH 0.612 / 0.096). Against the truth: off pitch on true D 0.677, true live 0.037 (table
0.6772, 0.0373). One scored run:

| date | truth | method | spellF1 | gain (paired) | wins | verdict |
|---|---|---|---|---|---|---|
| 2026-11-06 | **keep2 (decision)** | touch_c3_seen_w1_tv_cal | 0.5167 | +0.0304 (+0.0211..+0.0449) | 5/5 | **ADOPT** |
| 2026-11-06 | DFL | touch_c3_seen_w1_tv_cal | 0.4283 | +0.0163 (+0.0076..+0.0271) | 5/5 | better, below the adoption bar (not WORSE) |

- **ADOPT under D4, on the variant D31 fixed before the run.** The gain sits between the
  pessimistic (+0.0187) and optimistic (+0.0380) runs, as the realism check predicted.
- The calibration's assumption is that our method errs alike on the proxy and on BAR-ATH. Not
  submitted until the BAR-ATH check (D31) passes.

BAR-ATH check of `touch_c3_seen_w1_tv_cal`, D31's rules fixed before it ran; run by miha on
2026-11-06 (label-free; proxy keep2 + D12 in brackets):

| | touch_c3_seen_w1 | touch_c3_seen_w1_tv_cal | rule |
|---|---|---|---|
| D share | 0.440 | 0.435 (0.435 -> 0.419) | 1: pass |
| stoppages per 90, median s | 140.2, 9.4 | 108.8, 12.4 (126.3, 11.4 -> 98.0, 17.2) | 2: pass |
| frames turned to D / away from D | -- | 0.0472 / 0.0521 (0.0458 / 0.0622) | -- |
| set pieces with a decoded restart inside | 0.93 of 58 | 0.91 of 58 (throw-ins 24 -> 23 of 26) | **3: fail** (one set piece) |
| tag agreement gaps / seen | 0.734 / 0.867 | 0.632 / 0.870 | **4: fail** (gaps -0.102) |
| gap D | 0.777 | 0.781 (0.751 -> 0.711) | **5: fail** (+0.004, needs -0.080..0) |
| seen D | 0.208 | 0.197 (0.223 -> 0.223) | 5: pass (within 0.02) |
| possessions per 90 | 282.1 | 261.7, x0.928 (x0.951) | 6: no warning |
| gap home share | 0.480 | 0.411 (0.526 -> 0.536) | **7: warning** (-0.069) |
| gap possessions per 90, under 2 s | 56.9, 12.7 | 48.4, 6.8 (42.5, 2.6 -> 40.6, 3.0) | -- |

- **The check fails (rules 3, 4 and 5): `touch_c3_seen_w1` stays submitted.**
- Rules 3 and 5 fail narrowly. Rule 4 and warning 7 do not: inside ball gaps the TV method moves
  team frames towards away (home share -0.069) and loses 0.102 of tag agreement. On the proxy it
  moved the gap home share +0.010, towards the truth. The dead-ball side behaves as on the proxy
  (fewer, longer stoppages, D share in range). Why the gap team moves on BAR-ATH is unexplained.
  One candidate: the cue removes about half the short in-gap possessions (12.7 -> 6.8 per 90),
  far more than BAR-ATH-like ones exist on the proxy (2.6), and the Smart Tagging phases are
  70% home.

**D32: the ball merge under the TV method (2026-11-06).** `make proxy` (D12) and the proxy-side
aggregates (`deadball_check.py --proxy-side`, `gap_check.py --decoded`). Methods merge inside
`prepare` (`fused.merged`). On the proxy, the restored frames skip the D12 stress (the input
already carries it), so the merge reads higher than `broadcast_fill_check`'s +0.0038, which
restored before the stress.

| date | truth | method | spellF1 | gain (paired) | wins | verdict |
|---|---|---|---|---|---|---|
| 2026-11-06 | **keep2 (decision)** | touch_c3_seen_w1_tv_cal_fused | 0.5246 | +0.0382 (+0.0311..+0.0560) | 5/5 | **ADOPT** |
| 2026-11-06 | keep2 | touch_c3_seen_w1_fused | 0.4934 | +0.0070 (+0.0046..+0.0137) | 5/5 | better, below the adoption bar |
| 2026-11-06 | DFL | touch_c3_seen_w1_tv_cal_fused | 0.4376 | +0.0256 (+0.0180..+0.0391) | 5/5 | ADOPT (not WORSE) |
| 2026-11-06 | DFL | touch_c3_seen_w1_fused | 0.4194 | +0.0074 (+0.0040..+0.0123) | 5/5 | better, below the adoption bar |

Proxy-side, `touch_c3_seen_w1` -> `_tv_cal_fused` (truth): D share 0.435 -> 0.418 (0.441). Stoppages
per 90 126.3 -> 98.3 (97.1), median 11.4 -> 17.0 s (19.0). Frames to D / from D 0.0459 / 0.0628.
Set pieces with a restart 0.36 -> 0.39. Gap home share 0.526 -> 0.533 (0.541). Gap D 0.751 -> 0.710
(0.716). Seen D 0.223 -> 0.223 (0.256). Possessions per 90 281.7 -> 271.9 (275.0). Gap possessions
per 90 42.5 -> 43.3; under 2 s 2.6 -> 4.0.

BAR-ATH check of `touch_c3_seen_w1_tv_cal_fused` (D32 rules fixed before it ran; run by miha on
2026-11-06; label-free; `_tv_cal` from the same run):

| | touch_c3_seen_w1 | _tv_cal | _tv_cal_fused | rule |
|---|---|---|---|---|
| D share | 0.440 | 0.435 | 0.434 | 1: pass |
| stoppages per 90, median s | 140.2, 9.4 | 108.8, 12.4 | 108.8, 12.5 | 2: pass |
| frames to D / from D | -- | 0.0472 / 0.0521 | 0.0474 / 0.0533 | -- |
| set pieces with a decoded restart | 0.93 of 58 | 0.91 | 0.91 | **3: fail** |
| tag agreement gaps / seen | 0.734 / 0.867 | 0.632 / 0.870 | 0.610 / 0.872 | **4: fail** |
| gap D / seen D | 0.777 / 0.208 | 0.781 / 0.197 | 0.778 / 0.197 | **5: fail** (gap +0.001) |
| possessions per 90 | 282.1 | 261.7 | 266.0 (x0.943) | 6: no warning |
| gap home share | 0.480 | 0.411 | 0.411 | **7: warning** |
| gap possessions per 90, under 2 s, changes | 56.9, 12.7, 17.0 | 48.4, 6.8, 17.0 | 52.7, 9.3, 20.4 | -- |

- **D32's hypothesis fails.** Against `_tv_cal`, the merge leaves the gap home share unchanged
  (0.411) and lowers tag agreement in gaps (0.632 -> 0.610). It adds in-gap possessions and
  hand-overs (48.4 -> 52.7 per 90, changes 17.0 -> 20.4) without moving the team. The in-gap team
  shift comes from the TV cue's re-segmentation, not from missing ball evidence.
- **E13 stops (D32).** `touch_c3_seen_w1` stays submitted.

## Section 18: #4 W0 (what better ball tracking is worth) and #6 Step 0 (decoding for spellF1), keep2 truth, simulator v1 + D12 ball stress (2026-11-06)

Setup as section 16: IDSSE dev, seeds 0 1 2, priors on DFL labels, reference `touch_c3_seen_w1`
(0.4864). Code at branch `miha/dev`, uncommitted. **Settings tried: W0 5 ball variants (what-ifs
about a hypothetical tracker, not adoption candidates); #6 2 temperatures x 6 variants, none
registered.**

#4 W0. The same method under different simulated balls: D12 noise 0.05 m and 3.4 jumps per minute
throughout; only smoothing and visible share change. `BALL_VISIBLE` (`simulate.reveal_gaps`)
reveals whole gaps at random until the share is reached, so the remaining gaps keep v1's
placement and lengths. Means reproduce with `make proxy`; the paired intervals (seed 0, as in
`proxy_league`) came from an uncommitted scratch driver that mirrors it and reproduced its
baseline exactly.

| date | ball (keep2) | seen | spellF1 | interval | macroF1 | possessions | gain vs D12 | paired interval | wins |
|---|---|---|---|---|---|---|---|---|---|
| 2026-11-06 | D12 (smoothing 0.26 s), reference | 0.599 | 0.4864 | 0.4665..0.5024 | 0.8012 | 1491 | -- | -- | -- |
| 2026-11-06 | (a) smoothing 0 | 0.599 | 0.5323 | 0.5092..0.5429 | 0.8116 | 1534 | +0.0460 | +0.0303..+0.0518 | 5/5 |
| 2026-11-06 | (b) visible 0.76 | 0.763 | 0.4971 | 0.4791..0.5150 | 0.8039 | 1515 | +0.0107 | +0.0060..+0.0180 | 5/5 |
| 2026-11-06 | (b) visible 0.90 | 0.902 | 0.5086 | 0.4921..0.5268 | 0.8077 | 1541 | +0.0222 | +0.0166..+0.0322 | 5/5 |
| 2026-11-06 | (c) smoothing 0 + visible 0.90 | 0.902 | 0.5566 | 0.5310..0.5654 | 0.8194 | 1597 | +0.0703 | +0.0497..+0.0762 | 5/5 |
| 2026-11-06 | clean ball (`--clean`) | 1.000 | 0.5904 | 0.5730..0.6084 | 0.8341 | 1595 | +0.1041 | +0.0906..+0.1211 | 5/5 |
| 2026-11-06 | DFL truth: D12 reference | 0.599 | 0.4120 | 0.3925..0.4267 | 0.7957 | 1491 | -- | -- | -- |
| 2026-11-06 | DFL truth: (a) smoothing 0 | 0.599 | 0.4728 | 0.4513..0.4873 | 0.8086 | 1534 | +0.0609 | +0.0485..+0.0690 | 5/5 |
| 2026-11-06 | DFL truth: (c) smoothing 0 + visible 0.90 | 0.902 | 0.5062 | 0.4825..0.5190 | 0.8175 | 1597 | +0.0942 | +0.0772..+0.1023 | 5/5 |

- **A sharper ball is most of the value; coverage adds little** (+0.011 at 0.76, +0.022 at 0.90).
  The two roughly add.
- The touch engine was not retuned for a sharp ball (D12 tuned it for a smoothed one), so (a) may
  understate. Revealed gaps show the clean IDSSE ball, which in dead ball sits still near a line,
  so (b) may flatter. The clean row also removes jitter and jumps: an upper reference.
- Everything rests on D12's 0.26 s smoothing, fitted from aggregates, not measured directly.

#6 Step 0, `tools/mbr_check.py --samples 100 --candidates 20 --temperatures 1 2` (and
`--labels dfl`). Samples come from the decoder's own posterior (forward filtering, backward
sampling); temperature divides every decoder score and leaves Viterbi unchanged. "support" = the
share of samples holding a same-team spell over a Viterbi possession at IoU >= 0.5; "merge s<x"
absorbs the possessions below x into a neighbour; "merge oracle" absorbs every Viterbi possession
that matches no true one at IoU >= 0.5 (reads the truth). Paired intervals are seed 0.

| date | truth, T | variant | spellF1 | possessions | gain vs Viterbi | paired interval | wins |
|---|---|---|---|---|---|---|---|
| 2026-11-06 | keep2, 1 | viterbi (`touch_c3_seen_w1`) | 0.4864 | 1491 | -- | -- | -- |
| 2026-11-06 | keep2, 1 | one sample | 0.4564 | 1759 | -0.0299 | -0.0435..-0.0299 | 0/5 |
| 2026-11-06 | keep2, 1 | marginal | 0.4663 | 1685 | -0.0201 | -0.0290..-0.0190 | 0/5 |
| 2026-11-06 | keep2, 1 | mbr | 0.4841 | 1511 | -0.0022 | +0.0000..+0.0000 | 0/5 |
| 2026-11-06 | keep2, 1 | oracle best sample | 0.4737 | 1683 | -0.0127 | -0.0197..-0.0072 | 0/5 |
| 2026-11-06 | keep2, 1 | merge s<0.3 | 0.4850 | 1482 | -0.0014 | -0.0030..+0.0004 | 1/5 |
| 2026-11-06 | keep2, 1 | merge s<0.5 | 0.4837 | 1471 | -0.0026 | -0.0051..+0.0007 | 0/5 |
| 2026-11-06 | keep2, 1 | merge s<0.7 | 0.4811 | 1443 | -0.0052 | -0.0063..+0.0019 | 0/5 |
| 2026-11-06 | keep2, 1 | merge oracle | 0.4647 | 988 | -0.0217 | -0.0381..-0.0142 | 0/5 |
| 2026-11-06 | keep2, 2 | mbr | 0.4864 | 1491 | +0.0000 | +0.0000..+0.0000 | 0/5 |
| 2026-11-06 | keep2, 2 | merge s<0.3 / 0.5 / 0.7 | 0.4807 / 0.4687 / 0.4563 | 1456 / 1397 / 1278 | -0.0056 / -0.0176 / -0.0300 | 0.5 and 0.7: WORSE | 1/5, 0/5, 0/5 |
| 2026-11-06 | DFL, 1 | viterbi | 0.4120 | 1491 | -- | -- | -- |
| 2026-11-06 | DFL, 1 | mbr | 0.4121 | 1511 | +0.0002 | +0.0000..+0.0000 | 1/5 |
| 2026-11-06 | DFL, 1 | oracle best sample | 0.4123 | 1715 | +0.0003 | -0.0076..+0.0043 | 2/5 |
| 2026-11-06 | DFL, 1 | merge s<0.3 / 0.5 / 0.7 | 0.4108 / 0.4094 / 0.4047 | 1482 / 1471 / 1443 | -0.0011 / -0.0025 / -0.0072 | 0.7: WORSE | 1/5, 0/5, 0/5 |
| 2026-11-06 | DFL, 1 | merge oracle | 0.3650 | 974 | -0.0470 | -0.0596..-0.0384 | 0/5 |

- **MBR keeps Viterbi** (14/15 match-seeds at T=1, 15/15 at T=2): samples hold more possessions
  than Viterbi (350 against 298 per match at T=1; truth 291), so they never argue for fewer splits.
- **Even the oracle merge loses.** About a third of Viterbi's possessions match no true one, but
  they are misplaced or mis-bounded possessions, not removable extras: absorbing them breaks their
  neighbours' matches.
- **The decoder is confidently wrong.** At T=1 its own spellF1 estimate is 0.879 against 0.486;
  91% of its possessions have support >= 0.9, and half of those match a true possession (0.2 to
  0.3 below 0.9). Per frame, all samples agree on 84% of graded frames, where Viterbi is right on
  0.863; elsewhere about 0.5. Support is informative but cannot fix errors the decoder is sure of.

## Section 19: #2 adversarial validation: are BAR-ATH's frames different from the proxy's, and does that explain D26's seen-D sign? (2026-11-06)

Setup: no proxy-league run. [tools/adversarial_check.py](../tools/adversarial_check.py): a
classifier of "BAR-ATH or proxy" per frame, label-free on BAR-ATH. Proxy = the 5 dev matches,
simulator v1 + D12, seed 0, every 5th graded frame; BAR-ATH every 5th graded frame. Feature sets:
`deadmodel` (exactly `deadmodel.features`) and `context` (those plus ball and visibility). Learner:
deadmodel's HistGBM at half the rounds, twice the rate; 5 group folds of 60 s blocks; weighted
out-of-fold AUC. Baseline: each dev match against the other four. Reading rules R1, R2, R2b, R4
are in the tool's docstring. R2 was amended and R2b and R4 added after the first `--sanity` run
(simulator v0 as target) and before any BAR-ATH run: one IDSSE match is told from the others at
AUC 0.87 to 0.93, which hid a real speed-rank shift in the group test. Sanity (v0 against v1):
seen AUC 0.985; "tracked" is a DRIVER, and the speed-rank shift shows per feature (AUC 0.374).

BAR-ATH against the proxy (miha's run, 892 s):

| frames | features | AUC | odd-one-out max | R1 | R2 drivers (excess drop) | R2b shifts (own AUC) |
|---|---|---|---|---|---|---|
| seen | deadmodel | 0.973 | 0.930 | above, +0.043 | speed spread +0.109, lines and box +0.036 | speed_cv 0.627, cv_past3 / cv_future3 0.643, goalline_min 0.413, in_box 0.630, rank_change8 0.533 |
| gaps | deadmodel | 0.908 | 0.879 | within | shape +0.050, lines and box +0.065 | tracked 0.336, touchline_min 0.577, gyration 0.610, rank_change8 0.457 |
| seen | context | 0.970 | 0.896 | above, +0.074 | speed spread +0.107, lines and box +0.045 | as seen/deadmodel, plus ball_still 0.415, ball_nearest_m 0.453 |
| gaps | context | 0.894 | 0.829 | above, +0.065 | shape +0.048, lines and box +0.062 | as gaps/deadmodel |

Medians, BAR-ATH against proxy. Seen: speed_rank 0.663 / 0.660, slow_share_rank 0.343 / 0.350,
speed_cv 0.568 / 0.506, goalline_min 5.6 / 7.1 m, in_box 2 / 1, run_s 49.1 / 49.1 s. Gaps: tracked
21 / 22, touchline_min 4.5 / 2.2 m, run_s 28.9 / 29.2 s. Model mean P(dead) on seen frames:
BAR-ATH 0.210, proxy 0.260 (proxy truth D 0.249). R4 fails in all four (effective rows 0.04 to
0.06): the covariate-shift reweighting is not read.

Two proxy-only follow-ups (scratch scripts, not committed; 5 dev matches, seed 0, D12):

| test | seen speed_cv p50 | seen model P(dead) |
|---|---|---|
| proxy as is | 0.506 | 0.260 |
| + player position noise 0.03 / 0.05 / 0.10 m | 0.501 / 0.499 / 0.497 | 0.261 / 0.261 / 0.264 |
| speed spread features +0.062 (BAR-ATH's median gap) | | 0.251 |
| goalline_min -1.5 m, in_box +1 | | 0.282 |
| both | | 0.273 |
| *BAR-ATH* | *0.568* | *0.210* |

- **The seen-ball frames differ, but not where the simulator acts.** Speed composition, visibility
  stretches and the ball beyond a line match. The drivers are speed spread and players near the
  goal line, and neither is a simulator setting: position noise moves speed_cv the other way.
- **The differences do not explain D26's seen-D sign.** Shifting the proxy by BAR-ATH's median
  differences moves the model by -0.009 (spread) and +0.022 (lines), against BAR-ATH's -0.050.
  What remains is that P(dead | features) differs, or BAR-ATH has less seen dead ball; neither
  can be told without labels.
- **PLAN #1's "more gaps in dead ball" is not supported:** seen-frame speed ranks match, so v1's
  gaps-in-slow-play calibration holds.
- **One simulator gap is real, in gaps only:** BAR-ATH loses a player during ball gaps and keeps
  players away from the touchlines; v1 drops players independently of position and of gaps.
  Noted for #1, not built.
- The frozen DFL ball shows (ball_still), as simulate.py already lists; the dead-ball model does
  not read the ball.


## Section 20: end of half: the walk-off after the whistle is graded D; players-only cues measured, whistle Step 0 (2026-11-06)

**Failure.** miha noticed possessions predicted after the half-time whistle while players leave.
`tools/halfend_check.py --diagnose --bar-ath` (miha's run; 10 s bins counted back from each graded
block's last frame):
- Half 1: the whistle is about 80 s before the block end. From that bin, players inside the
  lines (median) go 18, 11, 4, 4, 4, 2, 1, 1. The reference labels about 12 s of it H/A and starts
  6 possessions. The 70 s before the last live bin were a stoppage the reference already decodes D.
- Half 2: inside 22 until about 40 s before the end, then 15-17. The reference labels about 9 s
  H/A there. The full-time whistle is not yet located.
- Size: about 8 extra possessions and about 500 frames, roughly +0.005 spellF1 if removed.

**Players-only cues on IDSSE dev, v1 + D12** (scratch scripts; seeds 0-1 unless stated):
| cue | live play / plain halves | why it fails as an end-of-half trigger |
|---|---|---|
| mean speed above 2.5 m/s for 2 s ("burst") | gaps between bursts up to 234 s; half ends 8-150 s after the last burst, truth D 0.40-0.96 | slow live play looks the same |
| fastest player above 5 m/s for 1 s ("sprint") | gaps in fully live play p99 23.8 s, max 43.0 s; half ends 4-89 s after the last sprint | sprint-free tails exist at plain half ends; on BAR-ATH 1-4 tracked people still "sprint" (jitter) |
| tracked players (1 s mean) <= 6 | raw IDSSE never below 21; v1 has such runs in 85% of halves (seeds 0-5), one 85 s long at a half end after 45:00 | v1's dropout episodes (calibrated to BAR-ATH) mimic the collapse |
| players beyond the lines by 1 m, >= 3 | longest run per half p90 28.2 s, max 34.9 s (seeds 0-5) | celebrations and substitutions |
| inside players <= median - 3 | final runs up to 48.5 s on plain halves | dropout episodes |

**Whistle detector** (`solution/whistle.py`, 6 tests in `tests/test_whistle.py`): per video frame,
5-bin-smoothed band spectrum (2-5 kHz) in dB, minus a fitted line; tonality = peak above median,
rise = peak above its 10 s rolling median. STRICT tonality >= 15 dB, rise >= 8 dB, >= 0.15 s;
LOOSE 10 / 4 / 0.10 s. On synthetic crowd noise with a +15 dB roar and a low-pitched voice: every
blast found within 2 frames, lengths within 0.12 s, a 0.25 s-apart pair kept as two, no events
from noise, roar or voice under either setting, the same through an AAC round trip. Limit: a blast
11 dB under the roar is missed by STRICT, and LOOSE finds only its later part. Pass criteria for the
BAR-ATH run: D33.

---

**Step 0 result (miha's BAR-ATH run, D33 update).** Features on all 199,319 frames. STRICT 252
events in half 1 (4.76/min, length p50 0.28 s, max 1.64 s) and 220 in half 2 (4.16/min); LOOSE
950 and 985. The half-time whistle: two STRICT blasts at 3.72-3.78 kHz. Kickoff whistles: STRICT at 3.9 kHz, 1.4-1.5 s before each kick.
Candidate pick: half 1, 22.0 s before the end, on crowd whistling at 2.9 kHz; half 2, 0.4 s before
the end. Fails D33 criterion 1. STRICT events at 3.6 kHz or more: about 0.6/min in half 1, and
none in half 1 after the half-time whistle.


## Section 21: #3 control model, Step 0: what a perfect control judgement on the touch engine's events is worth, keep2 truth, simulator v1 + D12 ball stress (2026-11-06)

**Why now.** The ball lag check on BAR-ATH (`tools/ball_lag_check.py --bar-ath`, miha's run,
aggregates only) finds no lag (best shift -0.143 s, as clean IDSSE's -0.142 s; past-only
smoothing would give +0.067 s) and a centred blur (kick rise late/early 1.02; past-only 1.96)
heavier than D12's: kick rise 10-90% p50 0.692 s (p10 0.548, p90 0.858) against 0.535 s at
D12's 0.26 s and 0.760 s at 0.40 s, about 0.36 s by interpolation. #4's cheap fix (a time
shift) is therefore out. D25 asks for evidence before a selective fix of the touch engine's
hand-overs is built; this section is that Step 0.

**Question.** If the touch engine knew, for each of its events (touches and control episodes),
whether the event's team really has the ball after it, how much would `touch_c3_seen_w1` gain?
A control model can only recover part of that.

**Method, fixed before the first run.** `tools/control_check.py`. Reference `touch_c3_seen_w1`
(D23, submitted), all settings as registered, priors unchanged (`holder_agree` stays fitted on the
unfiltered engine, which under-weights a cleaner holder: conservative). Per event at frame f
(a touch, or a control episode's start) by team k, the truth over [f, f + 1.0 s) (TOUCH_TOL_S)
decides: majority k is "agreed", majority the other team is "contradicted", anything else
(mostly D) is "neutral" and never touched. The holder is re-swept (`touch.sweep_holder` with an
event mask; no mask reproduces the reference exactly, checked) with some events removed, and the
match re-decoded. Rows:
1. as is (the re-sweep without a mask: must equal the reference);
2. perfect: every contradicted event removed;
3. perfect, touches only (control episodes kept);
4. imitations of a real model, by seeded random choice per event: recall r of contradicted events
   removed and a share f of agreed events removed by mistake, (r, f) = (0.7, 0.1), (0.5, 0.1),
   (0.7, 0.2).
Pooled spellF1 over the 5 dev matches x seeds 0-2, as the proxy league pools it. Event counts per
seed and the share of contradicted events are reported. **Settings tried: 0 tuned**; the window
and the (r, f) grid are fixed here.

**Pass bar, fixed before the first run.** Decision run keep2, v1 + D12 (smoothing 0.26 s). Step 1
(build a learned control score, then decide it with `make proxy` under D4) goes ahead only if both:
- (i) row 2 gains at least +0.04 (twice D4's bar: a real model recovers a fraction);
- (ii) the (0.7, 0.1) imitation gains at least +0.02 (D4's bar at a plausible operating point).
Reported, not deciding: the same rows with smoothing 0.40 s (the BAR-ATH blur reading above),
and on DFL truth. If (ii) passes at 0.26 s but falls under +0.01 at 0.40 s, that is recorded as a
warning and Step 1 must decide under both. If the bar fails, #3 closes at Step 0 and nothing is
built.

**Result** (`tools/control_check.py`, 5 dev matches x seeds 0-2; gains against `touch_c3_seen_w1`
re-decoded with the unchanged holder, which reproduces it exactly, asserted):

| truth, smoothing | as is | perfect | touches only | r 0.7 f 0.1 | r 0.5 f 0.1 | r 0.7 f 0.2 | touches / seed (contradicted) | control episodes / seed (contradicted) |
|---|---|---|---|---|---|---|---|---|
| **keep2, 0.26 s (decision)** | 0.4864 | **+0.0750** | +0.0623 | **+0.0415** | +0.0261 | +0.0300 | 5158 (0.137) | 2160 (0.066) |
| keep2, 0.40 s | 0.4596 | +0.0746 | +0.0650 | +0.0457 | +0.0243 | +0.0354 | 3435 (0.205) | 1864 (0.077) |
| DFL, 0.26 s | 0.4120 | +0.0377 | +0.0329 | +0.0180 | +0.0093 | +0.0080 | 5158 (0.137) | 2160 (0.059) |
| DFL, 0.40 s | 0.3842 | +0.0324 | +0.0288 | +0.0192 | +0.0108 | +0.0111 | 3435 (0.208) | 1864 (0.071) |

- **The bar passes:** (i) perfect +0.0750 >= 0.04; (ii) (0.7, 0.1) +0.0415 >= 0.02. It holds at
  0.40 s (+0.0457), so no warning. **Step 1 goes ahead.**
- Touches carry most of it (+0.062 of +0.075); control episodes are rarely contradicted.
- False drops are costly: going from f 0.1 to 0.2 at r 0.7 loses about 0.012 (keep2). A real
  model should be run at high precision on agreed events.
- On DFL truth the gains halve but stay positive (DFL counts first touches as changes, so fewer
  touches are contradicted in its sense). Reported only; the oracle is defined per truth.
- Heavier blur costs the reference itself -0.027 (0.4864 to 0.4596) and cuts the touches found by a
  third, with a larger contradicted share (0.137 to 0.205). If BAR-ATH's ball is near 0.36 s (the
  lag check above), D12's proxy is somewhat optimistic for every touch-reading method.
- Caveat: the imitation drops events at random. A real model's errors cluster on the hard events,
  so its gain at the same (r, f) can be lower.

### Step 1: a learned control score on the touch engine's touches (design and bars fixed before the first run, 2026-11-06)

**Model** (`solution/controlmodel.py`). One row per touch of `touch.detect` (control episodes are
left alone: they carry +0.013 of the +0.075 and are rarely contradicted). Target: the Step 0
verdict, contradicted (1) against agreed (0), neutral touches not trained on. Features are
relative to the touching team k ("own") and the other ("opp"), so they need no team swap, and
read no absolute position (D22 / D26's transfer risk):
- at the touch: velocity change, ball speed before and after, distance from the ball to the
  nearest own and nearest opp player, own and opp players within 3 m, and who held the ball just
  before by the engine (own, opp, nobody);
- timing: seconds since the previous and until the next touch by own and by opp, and until the
  next control episode start by own and by opp (capped at 5 s);
- after the touch: share of usable-ball frames whose nearest player is own, and the mean of
  (opp distance - own distance) to the ball, over [-1, 0), [0, 1), [1, 2), [2, 4) s; ball
  travel over the next 1 s; usable-ball share over the next 2 s.
Learner: HistGradientBoostingClassifier as C3 (max_iter 200, learning_rate 0.05, max_leaf_nodes
15, min_samples_leaf 50, l2 1.0, no early stopping, random_state 0). Training: the other IDSSE dev
matches (never the hold-out, D3), degraded by v1 with `framemodel.TRAIN_SEED` and D12's ball
noise and jumps, at both smoothings 0.26 s and 0.40 s (the lag check put BAR-ATH between them), on
keep2 labels. On BAR-ATH all five train it.

**Drop rule.** A touch is dropped (touch.sweep_holder's keep mask) when its P(contradicted) is
above a threshold set inside the training matches: their out-of-fold predictions (inner
leave-one-match-out) on agreed touches, quantile 0.90, i.e. a 10% false-drop target. Nothing is
set on the evaluated match. Method `touch_c3_seen_w1_ctl`: `touch_c3_seen_w1` with the dropped
touches, every other setting and prior unchanged (`holder_agree` stays fitted on the unfiltered
engine, as in Step 0). **Settings tried: 1** (this one).

**Bars, in order; stop at the first that fails.**
1. Touch level (`tools/control_check.py --model`), eval seeds 0-2, held-out match: recall of
   contradicted touches at least 0.50 and agreed touches dropped at most 0.12, at both 0.26 s and
   0.40 s smoothing. AUC reported.
2. Proxy league, D4: `make proxy METHODS="touch_c3_seen_w1 touch_c3_seen_w1_ctl"
   BALL_NOISE=0.05 BALL_JUMPS=3.4 BALL_SMOOTH=0.26` ADOPT on keep2; `LABELS=dfl` not WORSE;
   `BALL_SMOOTH=0.40` reported, and a WORSE there is a warning recorded with the result.
3. Only after ADOPT: a label-free BAR-ATH check, criteria written before it runs; the submission
   changes only on miha's decision after it.

**Step 1 result (2026-11-06).** Bar 1 passes, bar 2 fails; per the rules above, #3 Step 1 stops
here and `touch_c3_seen_w1_ctl` stays registered, not submitted.

Bar 1, `tools/control_check.py --model` (held-out match, seeds 0-2; thresholds 0.216..0.255):

| smoothing | touches | contradicted recall | agreed dropped | neutral dropped | AUC |
|---|---|---|---|---|---|
| 0.26 s | 15475 (2127 contradicted) | **0.790** | **0.108** | 0.523 | 0.929 |
| 0.40 s | 10306 (2111 contradicted) | **0.799** | **0.106** | 0.512 | 0.937 |

Bar 2, `make proxy METHODS="touch_c3_seen_w1 touch_c3_seen_w1_ctl" BALL_NOISE=0.05
BALL_JUMPS=3.4 BALL_SMOOTH=<s> [LABELS=dfl]`:

| truth, smoothing | touch_c3_seen_w1 | touch_c3_seen_w1_ctl | gain (interval), wins | verdict | possessions (truth) |
|---|---|---|---|---|---|
| **keep2, 0.26 s (decision)** | 0.4864 | 0.4947 | +0.0083 (-0.0027..+0.0187), 3/5 | **within noise** | 1491 -> 1292 (1455) |
| DFL, 0.26 s | 0.4120 | 0.3879 | -0.0240 (-0.0346..-0.0150), 0/5 | **WORSE** | 1491 -> 1292 (2191) |
| keep2, 0.40 s (reported) | 0.4596 | 0.4797 | +0.0201 (+0.0133..+0.0332), 4/5 | ADOPT | 1499 -> 1300 (1455) |

- **The model beats the imitation's operating point (0.79 / 0.11 against 0.70 / 0.10) yet gains
  a fifth of its +0.042.** Step 0's caveat came true: random drops are not the model's drops.
  Two differences, neither measured yet: the model drops half the neutral touches (the
  imitation dropped none), and its false drops can sit on the real hand-overs that matter
  (possessions fall to 1292, under the truth's 1455, while the reference had 1491).
- DFL is WORSE, as expected for a model trained to remove touches DFL counts as changes, but
  CLAUDE.md's routine requires it not to be.
- Under the heavier blur the same model is ADOPT (+0.020). D12 (0.26 s) decides; the lag check's
  0.36 s reading was taken before this result, but switching the decision stress now would be
  chosen after seeing it. Recorded only.
- **Settings tried: 1.** No threshold or feature change was tried after the result.

### Step 1 diagnostic: where the imitation's gain went (rows and reading rule fixed before the run, 2026-11-06)

`tools/control_check.py --diagnose`, keep2 truth, D12 (0.26 s) deciding, 0.40 s reported; 5 dev
matches x seeds 0-2. Every row re-sweeps the holder of `touch_c3_seen_w1` and re-decodes, as
Step 0; verdicts are Step 0's (keep2). Control episodes are never dropped.
- A: as is.
- B: the model's drops (must reproduce `touch_c3_seen_w1_ctl`, 0.4947 on the decision run).
- C: the model's drops, but neutral touches kept.
- D: the model's drops on contradicted touches only (its false drops and neutral drops undone).
- E: D plus random agreed drops at the model's own false-drop rate (per match and seed).
- F: random drops at the model's own rates on contradicted and agreed touches (Step 0's
  imitation at the model's operating point).
Reading: C - B is the cost of the neutral drops; D - C the cost of the model's false drops; E
against C whether its false drops hurt more than random ones would (clustered on the hand-overs
that matter); D against F whether its true drops are worth less than random ones.

**Rule, fixed now.** At most one variant, for the larger cause, and only if that cause is worth
at least +0.02 on the decision run:
- neutral drops (C - B >= 0.02): `touch_c3_seen_w1_ctl_live`, which drops a touch only where the
  reference's own decode (without the control model) is alive on most of [f, f + 1 s): a
  label-free stand-in for "not neutral";
- false drops (D - C >= 0.02 and E - C >= 0.01): `touch_c3_seen_w1_ctl_p05`, the same model with
  its threshold at the agreed touches' 0.95 quantile (5% false-drop target).
The variant then faces Step 1's bar 2 unchanged (D4 ADOPT on keep2, D12; DFL not WORSE; 0.40 s
reported). DFL was -0.024 for `_ctl`, and keep2 and DFL disagree on exactly these touches, so a
DFL failure is likely and will not be argued away. If neither cause reaches +0.02, or D is
already under +0.02 over A (the model's true drops alone are not worth D4's bar), #3 closes.
**Settings tried so far: 1; the variant, if any, is the 2nd and last.**

**Diagnostic result** (`tools/control_check.py --diagnose`; row B reproduces `_ctl`'s 0.4947):

| row | keep2, 0.26 s (decision) | possessions | keep2, 0.40 s | possessions |
|---|---|---|---|---|
| A as is | 0.4864 | 1491 | 0.4596 | 1499 |
| B model | +0.0083 | 1292 | +0.0201 | 1300 |
| C model, neutral kept | +0.0077 | 1298 | +0.0198 | 1303 |
| D model, contradicted only | **+0.0558** | 1366 | +0.0540 | 1352 |
| E D + random false drops | +0.0504 | 1361 | +0.0494 | 1350 |
| F random at model rates | +0.0445 | 1381 | +0.0438 | 1369 |

Per seed at 0.26 s: 5158 touches (709 contradicted, 4407 agreed, 43 neutral); the model drops 1060
= 560 contradicted + 477 agreed + 22 neutral.
- **Neutral drops cost nothing** (C - B = -0.0006): there are only 43 neutral touches per seed.
- **The model's false drops are the whole loss** (D - C = +0.048), and **they are clustered**:
  the same number of random false drops costs 0.005 (D - E), the model's cost 0.048 (E - C =
  +0.043). They sit on the real hand-overs that matter, and possessions fall below the truth's
  count.
- **Its true drops are worth more than random ones** (D +0.056 against a random-recall row,
  F with E's false drops, about 0.006 lower). The score finds the costly phantoms.
- By the rule fixed above (D - C >= 0.02 and E - C >= 0.01), the one variant is
  `touch_c3_seen_w1_ctl_p05`: threshold at the agreed touches' 0.95 quantile. Settings tried: 2.

**Variant result** (`touch_c3_seen_w1_ctl_p05`, the proxy league's table truncates its name to
`touch_c3_seen_w1_ctl`; same commands as bar 2):

| truth, smoothing | touch_c3_seen_w1 | _ctl_p05 | gain (interval), wins | verdict | possessions (truth) |
|---|---|---|---|---|---|
| **keep2, 0.26 s (decision)** | 0.4864 | 0.5054 | +0.0191 (+0.0115..+0.0295), 5/5 | **better, below the adoption bar** | 1491 -> 1323 (1455) |
| DFL, 0.26 s | 0.4120 | 0.3986 | -0.0133 (-0.0189..-0.0017), 0/5 | **WORSE** | 1491 -> 1323 (2191) |
| keep2, 0.40 s (reported) | 0.4596 | 0.4868 | +0.0272 (+0.0225..+0.0387), 5/5 | ADOPT | 1499 -> 1328 (1455) |

- **Bar 2 fails twice:** 0.0009 short of D4's 0.02 on the decision run (every match better), and
  WORSE on DFL. Fewer false drops more than doubled the gain (+0.008 to +0.019), as the
  diagnostic predicted; DFL's loss halved but stayed.
- **#3 closes here** by the rule fixed before the diagnostic: two settings tried, the second the
  last. `_ctl` and `_ctl_p05` stay registered, not submitted. No third threshold.
- Under the heavier blur the variant is ADOPT (+0.027, 5/5). Whether D12 should move to the
  blur the lag check measured is a separate question for miha, decided on its own evidence and
  for every method, not to rescue this one.

## Section 22: decoder cost re-tune on the current evidence and keep2 truth (2026-11-06)

Setup as section 21: IDSSE dev, seeds 0 1 2, simulator v1, the D12 ball stress
(`BALL_NOISE=0.05 BALL_JUMPS=3.4 BALL_SMOOTH=0.26`), truth `LABELS=keep2`, priors
`PRIOR_LABELS=dfl`. Reference `touch_c3_seen_w1` (D23, submitted), 0.4864. Code at branch
`miha/dev`. **Why.** D8 set every transition cost to 0 on `semi_markov` (speed-only evidence,
DFL truth, 2026-10-29) and nothing re-tested them on the current evidence or on controlled
possession, where false stoppages are the largest single loss (+0.046 of the +0.147 perfect
dead ball, section 16; two of 2.5 s on the public window, REVIEW.md section 4). New: the decoder
setting `dead_min_s` (a minimum stoppage length where above the fitted 1st percentile; None is
byte-identical).

**Grid, fixed before the first run. Settings tried: 9**, each the reference plus:
`dead` 1, 2, 4 (alive to D; D8: `dead` and `restart` act as one knob, so `restart` stays 0);
`switch` 1, 2 (H to A); `dead` 2 + `switch` 1; `dead` 4 + `switch` 2; `dead_min_s` 2 and 4.
Registered as `c3s_dead1 c3s_dead2 c3s_dead4 c3s_sw1 c3s_sw2 c3s_dead2_sw1 c3s_dead4_sw2
c3s_dmin2 c3s_dmin4`. **Decision:** D4 on the keep2 run; the best ADOPT, if any, must not be
WORSE on `LABELS=dfl`; the BAR-ATH check is `deadball_check.py --bar-ath` and
`gap_check.py --decoded --bar-ath` with D26's rules 1 to 3 and 5 to 6, criteria copied from the
proxy-side numbers of the same run, before it runs. No second grid.

**Result** (`make proxy METHODS="touch_c3_seen_w1 c3s_..." BALL_NOISE=0.05 BALL_JUMPS=3.4
BALL_SMOOTH=0.26`, then the same with `LABELS=dfl`; 485 s and 446 s). Gains are paired against
the reference on the first seed:

| date | method | keep2 spellF1 | poss | gain (interval), wins | verdict | DFL spellF1 | DFL gain (interval), wins | DFL verdict |
|---|---|---|---|---|---|---|---|---|
| 2026-11-06 | touch_c3_seen_w1 | 0.4864 | 1491 | -- | ref | 0.4120 | -- | ref |
| 2026-11-06 | c3s_dead1 | 0.4872 | 1475 | +0.0009 (-0.0029..+0.0023), 4/5 | within noise | 0.4120 | +0.0000 (-0.0032..+0.0010), 3/5 | within noise |
| 2026-11-06 | c3s_dead2 | 0.4904 | 1461 | +0.0040 (+0.0014..+0.0083), 5/5 | better, below the bar | 0.4133 | +0.0013 (+0.0004..+0.0061), 3/5 | better, below the bar |
| 2026-11-06 | c3s_dead4 | 0.4929 | 1431 | +0.0066 (+0.0053..+0.0151), 5/5 | better, below the bar | 0.4138 | +0.0018 (+0.0023..+0.0101), 3/5 | better, below the bar |
| 2026-11-06 | c3s_sw1 | 0.4919 | 1431 | +0.0055 (+0.0009..+0.0072), 3/5 | better, below the bar | 0.4119 | -0.0001 (-0.0034..+0.0026), 2/5 | within noise |
| 2026-11-06 | c3s_sw2 | 0.4952 | 1387 | +0.0089 (+0.0002..+0.0089), 4/5 | better, below the bar | 0.4089 | -0.0031 (-0.0065..+0.0014), 1/5 | within noise |
| 2026-11-06 | c3s_dead2_sw1 | 0.4955 | 1401 | +0.0091 (+0.0037..+0.0129), 4/5 | better, below the bar | 0.4135 | +0.0016 (-0.0013..+0.0068), 2/5 | within noise |
| 2026-11-06 | **c3s_dead4_sw2** | 0.5029 | 1323 | **+0.0165 (+0.0083..+0.0211), 5/5** | better, below the bar | 0.4106 | -0.0014 (-0.0028..+0.0085), 2/5 | within noise |
| 2026-11-06 | c3s_dmin2 | 0.4867 | 1490 | +0.0003 (-0.0004..+0.0000), 1/5 | within noise | 0.4119 | -0.0001, 1/5 | within noise |
| 2026-11-06 | c3s_dmin4 | 0.4905 | 1456 | +0.0042 (+0.0038..+0.0124), 4/5 | better, below the bar | 0.4110 | -0.0010 (-0.0023..+0.0055), 1/5 | within noise |

True possessions: 1,455 (keep2), 2,191 (DFL). Takeaways:
- **No setting clears D4; the grid is closed** (no second grid, as fixed above). The best,
  `dead` 4 + `switch` 2, is 0.0035 short on keep2 and its DFL gain is -0.0014: it predicts 1,323
  possessions against keep2's 1,455 and merges real ones, which the stricter truth cancels.
- **The `switch` cost's keep2 gain is a merge effect, not a fix.** Alone it is +0.006 to +0.009
  on keep2 and -0.0001 to -0.003 on DFL. A `switch` cost does not belong in the submission.
- **The `dead` cost is the one gain with the same sign on both truths**: +0.0066 (5/5) and
  +0.0018 at 4 nats, with 60 fewer possessions. Small, consistent, under the bar on its own; a
  candidate for a stack with the other sub-bar 5/5 gains (REVIEW.md item 4), not for a grid.
- **A minimum stoppage length does nothing**: the false stoppages are not the short ones.
  `dead_min_s` stays None (byte-identical).
- D8's zero costs stand on the current evidence and controlled truth; the question the plan left
  open is answered. Settings tried: 9, as fixed.

## Section 23: L3 stability check of the submitted configuration (2026-11-06)

PLAN section 6, L3: "a robust configuration does not swing the possession count by 30% under
small parameter changes". Never run until now. [tools/stability_check.py](../tools/stability_check.py)
moves one setting of `touch_c3_seen_w1` at a time by a fixed factor and re-decodes: the four
evidence weights at 0.8 and 1.2 times, the fitted median possession and stoppage lengths at 0.9
and 1.1 times, the touch engine's `touch_dv` and `touch_m` at 0.8 and 1.2 times. Setup: IDSSE dev,
seed 0 only, simulator v1 + D12 ball stress, keep2 truth. The list is fixed in the tool; the
best-scoring row is not a candidate (nothing here is a grid). `--bar-ath` prints the counts
label-free for miha.

| perturbation | possessions | ratio | stoppages | median s | D share | home | spellF1 | change |
|---|---|---|---|---|---|---|---|---|
| reference | 1506 | 1.000 | 692 | 11.0 | 0.439 | 0.507 | 0.4852 | -- |
| w x0.8 | 1407 | 0.934 | 630 | 13.1 | 0.435 | 0.506 | 0.4922 | +0.0070 |
| w x1.2 | 1559 | 1.035 | 722 | 10.2 | 0.440 | 0.507 | 0.4754 | -0.0099 |
| touch_w x0.8 | 1457 | 0.967 | 692 | 11.1 | 0.441 | 0.507 | 0.4839 | -0.0013 |
| touch_w x1.2 | 1541 | 1.023 | 688 | 10.9 | 0.437 | 0.507 | 0.4794 | -0.0058 |
| team_w x0.8 | 1502 | 0.997 | 686 | 11.1 | 0.438 | 0.507 | 0.4851 | -0.0002 |
| team_w x1.2 | 1504 | 0.999 | 697 | 10.9 | 0.440 | 0.508 | 0.4829 | -0.0024 |
| frame_w x0.8 | 1539 | 1.022 | 680 | 11.2 | 0.437 | 0.508 | 0.4778 | -0.0075 |
| frame_w x1.2 (above D16's limit, probe only) | 1484 | 0.985 | 698 | 10.8 | 0.441 | 0.508 | 0.4898 | +0.0046 |
| possession length x0.9 / x1.1 | 1509 / 1504 | 1.002 / 0.999 | 692 / 692 | 11.0 / 11.0 | 0.439 | 0.507 | 0.4846 / 0.4858 | -0.0006 / +0.0006 |
| stoppage length x0.9 / x1.1 | 1507 / 1502 | 1.001 / 0.997 | 693 / 685 | 11.0 / 11.2 | 0.439 | 0.507 | 0.4853 / 0.4843 | +0.0001 / -0.0010 |
| touch_dv x0.8 / x1.2 | 1527 / 1491 | 1.014 / 0.990 | 689 / 691 | 11.2 / 11.0 | 0.439 | 0.506 / 0.507 | 0.4840 / 0.4830 | -0.0013 / -0.0023 |
| touch_m x0.8 / x1.2 | 1468 / 1524 | 0.975 / 1.012 | 685 / 693 | 11.1 / 10.9 | 0.438 / 0.439 | 0.507 | 0.4801 / 0.4841 | -0.0052 / -0.0012 |

- **Plateau.** The worst possession-count change over the +-20% weight rows is 6.6% (`w` x0.8),
  far under L3's 30%; every spellF1 change is within +-0.01, the noise level of a single seed.
- The one direction with any slope is the overall evidence weight `w`: lower is a little better
  here (+0.007 at x0.8), with fewer and longer stoppages. That is a one-seed reading on a
  perturbation list, not a grid; D8's `w` = 1 stands unless a proper D4 run says otherwise.
- The duration priors do nothing within +-10%: the decoder is driven by the evidence.
- **BAR-ATH** (`tools/stability_check.py --bar-ath`, miha, 2026-11-06, label-free, whole match):
  reference 332 possessions, 165 stoppages of median 9.4 s, D share 0.440, home 0.610. The
  +-20% weight rows give 306 to 348 possessions (ratios 0.922 to 1.048), 146 to 173 stoppages,
  D share 0.438 to 0.443, home share 0.608 to 0.616; the prior and touch-engine rows move the
  count by at most 2.7%. **Worst change 7.8% (`w` x0.8): plateau**, as on the proxy (6.6%), and
  in the same direction (a lower `w` gives fewer, longer stoppages: 146 of median 10.8 s).
- **`touch_c3_seen_w1_dm` on BAR-ATH** (miha, 2026-11-06, D37 check 1): reference 287
  possessions, 114 stoppages of median 18.1 s, D share 0.444, home 0.623; the +-20% weight rows
  give 269 to 303 possessions (0.937 to 1.056), 107 to 117 stoppages, D share 0.443 to 0.445.
  **Worst change 6.3%: plateau.**
- **`touch_c3_seen_w1_dm` on the proxy** (seed 0, D12, keep2): reference 1,399 possessions, 531
  stoppages of median 17.1 s, D share 0.444, spellF1 0.5542; the +-20% weight rows give 1,342 to
  1,462 possessions (0.959 to 1.045) and spellF1 changes within -0.010..+0.007. **Worst change
  4.5%: plateau.** D37's check 1 passes on both sides.

## Section 24: E8 Step 0, where the Smart Tagging set-piece windows sit against the restart (2026-11-06)

PLAN E8 was never built. The only safe part of it is the SET PIECES windows as anchors for the
stoppage before each tagged restart (REVIEW.md item 6; team phases stay a validation signal).
Before a decoder term can be designed, the tag start's position relative to the restart kick has
to be measured, and BAR-ATH has no labels, so [tools/tag_anchor_check.py](../tools/tag_anchor_check.py)
asks two label-free clocks: the submitted method's decoded restarts (D to a team) and the touch
engine's first ball event after the tag opens. `--proxy-side` gives the reference picture with
DFL restart events standing in for tag starts (a "tag" that opens exactly at the kick, with the
event clock's own error of about a second). IDSSE dev, seed 0, D12 ball, keep2 truth. Columns:
share of tags with a decoded restart within +-2 s of the tag start and within -5..+10 s
("reach"); the nearest decoded restart's offset from the tag start, median (p10..p90), seconds;
decoded D share in the 10 s before the start, the 5 s after, and the rest of a 35 s window; the
touch engine's first event from 3 s before the start.

| codes | type | n | +-2 s | reach | restart offset s | D before | D after | D rest | first touch s |
|---|---|---|---|---|---|---|---|---|---|
| truth | throw-in | 225 | 0.70 | 1.00 | +1.2 (-1.3..+2.4) | 0.80 | 0.25 | 0.31 | +3.5 (+0.3..+18.5) |
| truth | goal kick | 86 | 0.69 | 1.00 | +1.0 (-2.2..+2.4) | 0.90 | 0.22 | 0.29 | +4.6 (+0.9..+19.4) |
| truth | corner | 44 | 0.55 | 1.00 | +1.9 (+0.5..+3.7) | 0.99 | 0.42 | 0.42 | +5.3 (+2.9..+53.2) |
| truth | free kick | 119 | 0.71 | 0.96 | +1.0 (-2.0..+2.3) | 0.88 | 0.20 | 0.31 | +4.3 (-0.1..+37.5) |
| truth | all | 510 | 0.65 | 0.97 | +1.2 (-1.4..+2.6) | 0.85 | 0.27 | 0.32 | +4.2 (+0.4..+27.7) |
| touch_c3_seen_w1 | throw-in | 225 | 0.29 | 0.81 | +2.6 (-0.0..+5.7) | 0.71 | 0.49 | 0.31 | as truth |
| touch_c3_seen_w1 | goal kick | 86 | 0.31 | 0.78 | +1.9 (-2.1..+5.9) | 0.62 | 0.32 | 0.28 | |
| touch_c3_seen_w1 | corner | 44 | 0.45 | 0.98 | +2.2 (+0.8..+4.7) | 0.97 | 0.50 | 0.32 | |
| touch_c3_seen_w1 | free kick | 119 | 0.59 | 0.91 | +1.2 (-1.2..+2.8) | 0.79 | 0.25 | 0.30 | |
| touch_c3_seen_w1 | all | 510 | 0.36 | 0.83 | +2.0 (-0.8..+5.1) | 0.74 | 0.41 | 0.30 | |

- The DFL event clock runs about a second ahead of the truth's restart (truth offset +1.2 s). Our
  decoded restarts come a further 0.8 s later (+2.0 s), spread over -0.8..+5.1 s: the late
  restarts the public window also shows (REVIEW.md section 4.2).
- The touch engine's first event after a restart comes a median 4 s later: the restart kick
  itself is mostly not a detected touch on the blurred ball, so the touch clock cannot place the
  kick; the decoded restart is the usable clock.

**Reading rule for the BAR-ATH run, fixed before it runs** (`tools/tag_anchor_check.py --bar-ath`,
miha; label-free). Compare BAR-ATH's `touch_c3_seen_w1` rows with the proxy's above, per type:
1. offsets alike (median within +-1.5 s of the proxy's, p10..p90 no wider than -3..+8 s) and
   D before the start at least 0.6: the tag opens at the kick, as the proxy imitation assumes.
   Step 1 may be built: a synthetic tag on the proxy that opens at the DFL restart with BAR-ATH's
   measured jitter, and a decoder term that raises D in the seconds before a tagged start and
   adds a restart bonus at it, decided under D4 with DFL not WORSE;
2. offsets clearly positive beyond that (the tag opens before the kick, during the stoppage) or
   negative (after it): Step 1 may still be built, with the synthetic tag shifted by the measured
   median, but its proxy gain is to be read as optimistic by the spread;
3. reach under 0.6 for throw-ins, or D before the start under 0.4: the tag start is not an
   anchor for the stoppage, and E8 closes at Step 0.
The restart-team column (tag Team against the decoded team after the restart) is reported for the
types that carry a team; throw-ins carry none, so a restart-team term stays out of scope (E7).

**BAR-ATH result** (`tools/tag_anchor_check.py --bar-ath`, miha, 2026-11-06; 58 SET PIECES tags
inside the halves; `touch_c3_seen_w1`):

| type | n | +-2 s | reach | restart offset s | D before | D after | D rest | first touch s | team |
|---|---|---|---|---|---|---|---|---|---|
| throw-in | 26 | 0.04 | 0.65 | +6.1 (+3.6..+7.9) | 0.52 | 0.77 | 0.25 | +7.0 (-2.4..+13.3) | -- |
| goal kick | 10 | 0.30 | 0.80 | +3.9 (-0.2..+7.8) | 0.45 | 0.51 | 0.31 | +1.4 (-0.9..+9.9) | 0.88 |
| corner | 12 | 0.08 | 1.00 | +4.0 (+2.1..+6.6) | 1.00 | 0.75 | 0.18 | +7.9 (+5.6..+12.0) | 0.92 |
| free kick | 5 | 0.00 | 0.40 | +2.0 (-1.4..+5.4) | 0.87 | 0.90 | 0.29 | +7.9 (+1.8..+51.3) | 1.00 |
| goal | 4 | 0.00 | 0.75 | +4.2 (+4.1..+4.3) | 1.00 | 0.88 | -- | +4.3 (+3.7..+10.7) | -- |
| kickoff | 1 | 0.00 | 1.00 | +3.1 | 1.00 | 0.62 | -- | +3.1 | -- |
| all | 58 | 0.09 | 0.74 | +5.1 (+0.5..+7.5) | 0.68 | 0.74 | 0.25 | +6.9 (-1.0..+12.7) | 0.91 |

- **Case 2 of the reading rule.** The decoded restart comes a median 5.1 s after the tag start
  (proxy, tag at the kick: 2.0 s), the spread +0.5..+7.5 s is inside the rule's -3..+8 s, D
  before the start is 0.68 (at least 0.6), reach 0.74 and 0.65 for throw-ins (at least 0.6).
  The decoder keeps D on 0.74 of the 5 s after the tag opens, against 0.41 on the proxy, and
  the touch engine's first event comes 2.7 s later than on the proxy. All three say the same:
  **the tag opens about 3 s before the restart kick, inside the stoppage**, not at the kick.
- The tag start is therefore a usable anchor for "a restart follows within a few seconds" and
  for "the last seconds of a stoppage", not for the stoppage's start, which is where most of the
  start-edge loss (+0.043) lies. What it can fix: late or missed restarts (end edges, +0.038 on
  the proxy) and the 9 of 26 tagged throw-ins with no decoded restart within -5..+10 s, which are
  either stoppages the decoder played through or restarts it placed more than 10 s late.
- Where the tag carries a team, the decoded restart team agrees on 0.91: a restart-team term has
  nothing to add (E7 stays closed).
- **Step 1 may be built under case 2**: a synthetic tag on the proxy opening a median 3 s before
  the DFL restart with the spread above, by type at BAR-ATH's coverage (26 of the match's
  throw-ins, all corners, about half the goal kicks, few free kicks), and a decoder term that
  raises D in the seconds before and around the tag start and makes a restart likely within a
  few seconds after it; D4 on keep2, DFL not WORSE, then D26-style BAR-ATH rules fixed before the
  check. Expected gain, honestly: +0.01 to +0.02, mostly restart edges, below D4 on its own
  unless it stacks with a dead-ball change. Decision to build it is miha's.

## Section 25: stacking the closed parts on the submission, keep2 truth, simulator v1 + D12 ball stress (2026-11-07)

**Why.** miha (2026-11-07): the parts were each tested alone on top of the reference of their day,
never on top of each other or of `touch_c3_seen_w1_dm`; small gains could add up. Registered in
[solution/methods.py](../solution/methods.py) as the `dm_*` family, **settings fixed before the
first run**, each part at its own registered setting:

| method | on top of `touch_c3_seen_w1_dm` | alone it was (vs the reference of its day) |
|---|---|---|
| `dm_tv` | TV cut-away cue, calibrated, w1 (D31) | +0.0304 ADOPT, BAR-ATH check failed |
| `dm_ball` | merged broadcast ball (D32) | reported alongside `_tv_cal_fused` |
| `dm_tvb` | both (D32) | +0.0382 ADOPT, BAR-ATH check failed |
| `dm_cue` | player + ball beyond a line, w0.5 (D15) | +0.008, below D4 |
| `dm_k2` | every prior fitted on keep2 (D21, D25) | +0.0189, below D4 |
| `dm_dead4` | decoder cost `dead` 4 (D36) | +0.007 keep2 / +0.002 DFL, below D4 |
| `dm_snap` | E6 snapping, hand-overs onto the first touch within 1.5 s (section 5) | +0.0025, within noise |
| `dm_ctl` | control model at the 5% false-drop threshold (D35) | below D4, WORSE on DFL |
| `dm_all` | all of the above except the control model | -- |

Not stacked: `touch_takes` off (WORSE on DFL, D10, D21), the dadvance bonus (D19), self-training
(D24), `_dms` (a variant of the same model, D37's fallback).

**Rule, fixed before the run.** Reference `touch_c3_seen_w1_dm`; `make proxy` on keep2 + D12
(decision) and on `LABELS=dfl`. Run 1: the nine methods above. Run 2, one method: `dm_stack` = the
singles that are *better* on keep2 (gain > 0 with the paired interval excluding 0) and not WORSE
on DFL, all together; if that set is empty or equals `dm_all`'s, there is no run 2. ADOPT under
D4 (gain >= 0.02, interval excludes 0, wins >= 2/3 of the matches, DFL not WORSE) names a
candidate; a candidate that reads the broadcast is still subject to D31's BAR-ATH rules 1 to 5
before `run.py` changes, and nothing here changes D32's reading of those rules. Merged-ball
methods restore clean frames on the proxy (fused.merged docstring: slightly optimistic).

**Run 0 (2026-11-07, plain v1 by mistake, reported only).** The first run went out without the
D12 ball stress (`make proxy` defaults), so it is the "keep2, plain v1" setting of sections 13
and 16 (reference 0.6046, DFL 0.5273), not the decision setting. Kept because it is a clean
second setting; the decision run follows. Pooled spellF1 and paired gain over the reference:

| method | keep2 plain | gain | wins | verdict | DFL plain | gain | wins | verdict |
|---|---|---|---|---|---|---|---|---|
| `touch_c3_seen_w1_dm` | 0.6046 | -- | -- | reference | 0.5273 | -- | -- | reference |
| `dm_tv` | 0.6082 | +0.0035 (-0.0010..+0.0157) | 3/5 | noise | 0.5284 | +0.0011 | 3/5 | noise |
| `dm_ball` | 0.6085 | +0.0039 (+0.0016..+0.0103) | 5/5 | better, below bar | 0.5351 | +0.0078 (+0.0046..+0.0122) | 5/5 | better, below bar |
| `dm_tvb` | 0.6134 | +0.0088 (+0.0022..+0.0214) | 5/5 | better, below bar | 0.5369 | +0.0095 (-0.0025..+0.0146) | 5/5 | noise |
| `dm_cue` | 0.6053 | +0.0007 | 2/5 | noise | 0.5276 | +0.0003 | 2/5 | noise |
| `dm_k2` | 0.6204 | +0.0158 (+0.0098..+0.0174) | 5/5 | better, below bar | 0.5212 | -0.0061 (-0.0078..+0.0013) | 1/5 | noise |
| `dm_dead4` | 0.6064 | +0.0018 | 4/5 | noise | 0.5277 | +0.0003 | 3/5 | noise |
| `dm_snap` | 0.6090 | +0.0044 (+0.0018..+0.0077) | 5/5 | better, below bar | 0.5326 | +0.0053 (+0.0054..+0.0113) | 5/5 | better, below bar |
| `dm_ctl` | 0.5927 | -0.0120 (-0.0205..+0.0016) | 0/5 | noise | 0.4645 | -0.0629 | 0/5 | WORSE |
| `dm_all` | 0.6339 | **+0.0293 (+0.0218..+0.0418)** | 5/5 | **ADOPT** | 0.5355 | +0.0082 (-0.0001..+0.0190) | 3/5 | noise, not WORSE |

Reading, on this setting only: the stack of everything adds up to an ADOPT on keep2 where no
single part reaches the bar; on DFL it is not WORSE. The keep2 priors carry the most alone
(+0.016) and lose on DFL, as expected when the priors match the truth's labels. The control
model is the one part that hurts. Possession counts: `dm_all` 1379 against the reference's 1432
(truth 1455). Run time 1019 s per truth.

**Run 1 (2026-11-07, the decision setting: keep2 + D12, and DFL + D12).** `make proxy
METHODS="touch_c3_seen_w1_dm dm_tv dm_ball dm_tvb dm_cue dm_k2 dm_dead4 dm_snap dm_ctl dm_all"
BALL_NOISE=0.05 BALL_JUMPS=3.4 BALL_SMOOTH=0.26` and the same with `LABELS=dfl`. Reference
0.5565 (keep2) and 0.4601 (DFL), as in section 16. Pooled spellF1, paired gain, wins, verdict;
possessions predicted against 1455 true (keep2):

| method | poss | keep2 + D12 | gain | wins | verdict | DFL + D12 | gain | wins | verdict |
|---|---|---|---|---|---|---|---|---|---|
| `touch_c3_seen_w1_dm` | 1386 | 0.5565 | -- | -- | reference | 0.4601 | -- | -- | reference |
| `dm_tv` | 1388 | 0.5600 | +0.0035 (-0.0022..+0.0153) | 3/5 | noise | 0.4630 | +0.0029 | 4/5 | noise |
| `dm_ball` | 1407 | 0.5620 | +0.0055 (+0.0041..+0.0128) | 5/5 | better, below bar | 0.4669 | +0.0068 (+0.0044..+0.0122) | 5/5 | better, below bar |
| `dm_tvb` | 1415 | 0.5667 | +0.0102 (+0.0044..+0.0235) | 4/5 | better, below bar | 0.4710 | +0.0109 (+0.0018..+0.0195) | 5/5 | better, below bar |
| `dm_cue` | 1395 | 0.5583 | +0.0018 (-0.0002..+0.0077) | 4/5 | noise | 0.4612 | +0.0011 | 4/5 | noise |
| `dm_k2` | 1313 | 0.5706 | +0.0141 (+0.0095..+0.0172) | 5/5 | better, below bar | 0.4640 | +0.0039 (+0.0015..+0.0090) | 5/5 | better, below bar |
| `dm_dead4` | 1365 | 0.5569 | +0.0004 | 3/5 | noise | 0.4592 | -0.0010 | 3/5 | noise |
| `dm_snap` | 1386 | 0.5525 | -0.0040 (-0.0083..-0.0008) | 1/5 | WORSE | 0.4578 | -0.0023 | 2/5 | noise |
| `dm_ctl` | 1215 | 0.5831 | +0.0265 (+0.0174..+0.0373) | 5/5 | ADOPT | 0.4536 | -0.0065 (-0.0119..+0.0056) | 0/5 | noise |
| `dm_all` | 1329 | 0.5782 | **+0.0216 (+0.0132..+0.0346)** | 5/5 | **ADOPT** | 0.4707 | +0.0106 (+0.0021..+0.0209) | 4/5 | better, not WORSE |

Reading:
- **`dm_all` is an ADOPT candidate under D4**: +0.0216 on keep2, 5/5, and better on DFL. No single
  part reaches the bar; four of them (merged ball, TV cue with it, keep2 priors, control model)
  are better with the interval excluding 0, and they add up.
- **The control model is the largest single gain on keep2 and the only part that loses every
  match on DFL** (0/5, -0.0065, within noise by the interval). It predicts 1215 possessions, 17%
  under the reference's and 16% under the truth's: it gains by merging, which the first-touch
  labels punish. Its D35 verdict (on top of `touch_c3_seen_w1`: below D4, WORSE on DFL) softens
  on top of the dead-ball model but the warning sign is the same.
- Snapping, which helped on the clean ball (run 0), is WORSE under the ball stress; the dead
  cost and the beyond-line cue do nothing; the TV cue alone is noise once the dead-ball model is
  in, and most of `dm_tvb`'s gain is the merged ball's.
- Run time 1056 s and 1037 s.

**Run 2, composed by the rule above.** Singles better on keep2 (interval excluding 0) and not
WORSE on DFL: `dm_ball`, `dm_tvb`, `dm_k2`, `dm_ctl`. `dm_stack` = `touch_c3_seen_w1_dm` + TV cue
(calibrated) + merged ball + keep2 priors + control model (false_drop 0.05); registered, one run
on both truths under D12.

| method | poss | keep2 + D12 | gain | wins | verdict | DFL + D12 | gain | wins | verdict |
|---|---|---|---|---|---|---|---|---|---|
| `dm_stack` | 1198 | 0.5937 | **+0.0372 (+0.0283..+0.0538)** | 5/5 | **ADOPT** | 0.4592 | -0.0009 (-0.0096..+0.0132) | 3/5 | noise, not WORSE |

Run time 338 s and 334 s. **Outcome of section 25: two ADOPT candidates on the proxy, neither
submitted.**
- `dm_all` (no control model): +0.0216 keep2, +0.0106 DFL, 1329 possessions (0.91 of the truth's).
- `dm_stack` (with it): +0.0372 keep2, -0.0009 DFL, 1198 possessions (0.82 of the truth's). The
  control model is the whole difference: +0.016 on keep2, -0.012 on DFL, 131 fewer possessions.
- Both read the broadcast (TV cue and merged ball), so D31's BAR-ATH rules 1 to 5 apply before
  `run.py` can switch, and run.py must then build the alignment cache itself (Phase 4,
  byte-identical rule). The TV parts failed rule 4 (in-gap team) on their own; whether the
  dead-ball model and keep2 priors change that is for the check to show, not to assume.
- Not run: a broadcast-free stack (keep2 priors + control model) that would avoid both the
  alignment dependency and the in-gap question. Its two parts are +0.0141 and +0.0265 alone.

**Run 3, registered before it ran (2026-11-07).** `dm_k2ctl` = `touch_c3_seen_w1_dm` + keep2
priors + control model (false_drop 0.05): `dm_stack` without the TV cue and the merged ball, so it
reads no broadcast and needs no alignment cache. Same reference, same D12 setting, both truths,
D4 unchanged (gain >= 0.02, interval excluding 0, wins >= 2/3, DFL not WORSE). One run. If it
is ADOPT it becomes the submission candidate ahead of the two broadcast stacks, because it has
no D31 gate and no reproducibility work; if not, the broadcast stacks stay the only candidates
and D31's check decides. Warning 6's possession count is read alongside: `dm_ctl` alone kept
0.835 of the truth's possessions.

Run 3, keep2 + D12 (382 s):

| method | poss | spellF1 | 5-95% | macroF1 | gain | wins | verdict |
|---|---|---|---|---|---|---|---|
| `touch_c3_seen_w1_dm` | 1386 | 0.5565 | 0.5359..0.5731 | 0.8573 | | | reference |
| `dm_k2ctl` | 1185 | 0.5854 | 0.5632..0.6033 | 0.8619 | **+0.0289 (+0.0203..+0.0397)** | 5/5 | **ADOPT** |

Per match it wins all five (+0.007, +0.022, +0.030, +0.061, +0.026). It keeps 0.855 of the
reference's possessions on the proxy (0.814 of the truth's).

BAR-ATH, label-free (miha ran `tools/deadball_check.py --bar-ath` and `tools/gap_check.py
--decoded --bar-ath` with both methods; D31's rules read as a sanity check, since no part of
this candidate reads the broadcast):

| | `touch_c3_seen_w1_dm` | `dm_k2ctl` | rule |
|---|---|---|---|
| D share | 0.444 | 0.443 | 1: in 0.41..0.48, pass |
| stoppages per 90, median s | 96.9, 18.1 | 95.2, 18.5 | 2: fewer and longer, pass |
| set pieces with a decoded restart | 0.93 of 58 | 0.93 of 58 | 3: at least as many, pass |
| frames turned to D, away from D | | 0.0028, 0.0040 | |
| tag agreement in gaps, ball seen | 0.771, 0.869 | 0.790, 0.890 | 4: not below, pass (both rise) |
| gap D, seen D | 0.818, 0.188 | 0.817, 0.186 | 5: unchanged, pass |
| possessions per 90 | 243.9 | 200.6 | 6: x0.822, **below 0.85, warning** (proxy x0.855) |
| gap home share | 0.473 | 0.481 | 7: +0.008, within 0.03 |
| gap possessions per 90, under 2 s, team changes in gaps | 33.1, 2.5, 13.6 | 27.2, 1.7, 9.3 | |

Reading: the candidate leaves the dead-ball side alone and removes possessions, more on
BAR-ATH (18%) than on the proxy (14.5%); the proxy's known gap in short in-gap possessions
(2.6 against 12.7 per 90, D32) is the likely reason. The possessions it removes agree with the
Smart Tagging team less than the ones it keeps: tag agreement rises both in gaps and with the
ball seen. That is label-free evidence that the merged-away spells were flickers, not a proof
that the organisers' labels omit them.

Run 3, DFL + D12 (352 s):

| method | poss | spellF1 | 5-95% | macroF1 | gain | wins | verdict |
|---|---|---|---|---|---|---|---|
| `touch_c3_seen_w1_dm` | 1386 | 0.4601 | 0.4413..0.4760 | 0.8517 | | | reference |
| `dm_k2ctl` | 1185 | 0.4525 | 0.4350..0.4720 | 0.8546 | -0.0076 (-0.0141..+0.0039) | 1/5 | within noise, not WORSE |

**Outcome of run 3: `dm_k2ctl` passes D4** (ADOPT on keep2 + D12, not WORSE on DFL + D12) and
the BAR-ATH label-free rules 1 to 5, with warning 6 (possessions x0.822). It reads no broadcast.
Under the pre-registered reading it is the submission candidate ahead of `dm_all` and
`dm_stack`. The DFL side is the same trade as `dm_ctl` alone: a small loss on first-touch labels
on four of five matches, inside the interval. miha decided: switched, D39 (2026-11-07).

**The two broadcast stacks on BAR-ATH (2026-11-07, after D39).** miha asked why the broadcast
stacks should not be tried; the answer was to run D31's label-free check on them, which had not
been done. Reference `v7_keep2_control` (the submission); candidates `v6_plus_stack` (= v7 + TV
cue + merged ball) and `v6_plus_all` (= v6 + TV cue + merged ball + keep2 priors + beyond-line
cue + dead cost 4 + snapping; no control model). miha ran both D31 commands and shared the
aggregates:

| | `v7_keep2_control` | `v6_plus_stack` | `v6_plus_all` | rule |
|---|---|---|---|---|
| D share | 0.443 | 0.440 | 0.446 | 1: in 0.41..0.48, both pass |
| stoppages per 90, median s | 95.2, 18.5 | 86.7, 19.5 | 86.7, 19.8 | 2: fewer and longer, both pass |
| set pieces with a decoded restart | 0.93 of 58 | **0.90 of 58** (throw-ins 0.96 -> 0.85) | 0.97 of 58 | 3: stack **fails**, all passes |
| tag agreement in gaps | 0.790 | **0.692** | **0.667** | 4: **both fail** |
| tag agreement, ball seen | 0.890 | 0.899 | 0.878 | 4: not below 0.80, both pass |
| gap D, seen D | 0.817, 0.186 | 0.817, 0.182 | 0.820, 0.190 | 5: within bounds, both pass |
| possessions per 90 | 200.6 | 195.5 | 232.9 | 6: all rises against v7 (no control model), warning |
| gap home share | 0.481 | 0.424 | 0.431 | 7: -0.057 and -0.050, **warning fires** for both |
| gap possessions under 2 s, team changes in gaps, per 90 | 1.7, 9.3 | 0.8, 10.2 | 4.2, 14.4 | |

Reading: the same failure as `touch_c3_seen_w1_tv_cal` and `_tv_cal_fused` (D31, D32), larger.
Inside ball gaps both stacks agree with the Smart Tagging attacking team about 0.10 to 0.12
less often than the submission and move about 0.05 of the gap team frames from home to away,
with the ball seen unchanged. Neither the dead-ball model, the keep2 priors nor the control
model changes what the TV cue does in gaps. `v6_plus_stack` also loses throw-in restarts.
Outcome: not submitted; E13 stays closed (D32); the stacks stay registered. The proxy's ADOPT
for them stands as a proxy result only: the proxy has 2.6 short in-gap possessions per 90
against BAR-ATH's 12.7 and imitated cut-aways, so it cannot see this.

**Run 4, registered before it ran (2026-11-07): the merged ball alone on v7.** miha asked
whether more visible ball should not help; the merge alone was never measured on top of an
adopted change. `v7_plus_ball` = `v7_keep2_control` + `merge_ball` (`fused.merged`: the real
broadcast ball in tactical gaps on BAR-ATH, `imitate_fill` on the proxy), no TV cue. Same
reference, D12, both truths, one run. The reading rule is D27's ride-along clause, fixed on
2026-11-06, not a new bar: the merge may ride along with an adopted change only if, measured
with it, it is *better* on keep2 + D12 (paired interval excluding 0, wins at least 3/5), not
WORSE on DFL + D12, and the BAR-ATH check shows no harm, read as D31's rules 1 to 5 against
`v7_keep2_control` (rule 4, tag agreement in gaps not below v7's 0.790 and seen not below 0.80,
is the one the broadcast parts have failed so far), with warnings 6 and 7. Expected size: about
+0.005, the merge's gain on v5 (section 17). Even on a pass it is submitted only if
`solution/run.py` builds the alignment cache itself before the Phase 4 freeze, byte-identical;
otherwise it stays a comparison file. miha runs the two BAR-ATH commands in parallel.

Run 4, BAR-ATH (miha's run of the two D31 commands, before the proxy result was in):

| | `v7_keep2_control` | `v7_plus_ball` | rule |
|---|---|---|---|
| D share | 0.443 | 0.441 | 1: pass |
| stoppages per 90, median s | 95.2, 18.5 | 95.2, 18.4 | 2: unchanged, no harm |
| set pieces with a decoded restart | 0.93 of 58 | 0.93 of 58 | 3: pass |
| frames turned to D, away from D | | 0.0008, 0.0023 | |
| tag agreement in gaps | 0.790 | **0.771** | 4: **fail** (-0.019; seen 0.890 unchanged) |
| gap D, seen D | 0.817, 0.186 | 0.815, 0.185 | 5: pass |
| possessions per 90 | 200.6 | 202.3 | 6: a rise, warning |
| gap home share | 0.481 | 0.480 | 7: pass |
| gap possessions per 90, under 2 s, team changes in gaps | 27.2, 1.7, 9.3 | 29.7, 3.4, 11.9 | |

Reading: the merge alone does what it did under the TV cue (section 17, D32): it adds short
in-gap possessions and hand-overs (under 2 s per 90 doubled, 1.7 -> 3.4; team changes in gaps
9.3 -> 11.9) and those agree with the Smart Tagging team less than the decoder's own carry
through the gap, so tag agreement in gaps falls. The harm is a fifth of the TV cue's (-0.019
against -0.10) but rule 4 is "not below", and the ride-along clause needs the BAR-ATH check to
show no harm. **Refused on BAR-ATH whatever the proxy says**; the proxy result is recorded below
for the record.

Run 4, proxy (379 s and 376 s):

| method | truth + D12 | poss | spellF1 | 5-95% | macroF1 | gain over v7 | wins | verdict |
|---|---|---|---|---|---|---|---|---|
| `v7_plus_ball` | keep2 | 1198 | 0.5895 | 0.5709..0.6121 | 0.8635 | +0.0041 (+0.0043..+0.0126) | 5/5 | better, below the adoption bar |
| `v7_plus_ball` | DFL | 1198 | 0.4578 | 0.4413..0.4783 | 0.8562 | +0.0053 (+0.0037..+0.0107) | 5/5 | better, below the adoption bar |

The proxy side of D27's clause would have passed (better on keep2, not WORSE on DFL), at the
size section 17 predicted; the BAR-ATH side failed. **Outcome of run 4: not submitted.** With
this the broadcast is measured four ways on BAR-ATH (TV cue, TV cue + merge, the two stacks,
the merge alone) and every one lowers the in-gap tag agreement. Section 25 is closed.

---

## Section 26: E15 ball inference in gaps, Step 0 and v1 (2026-11-07)

*Section 25 on `denis/ball-impute` (PR #9); renumbered at the merge into `miha/dev` on 2026-11-07 because `miha/dev` had taken section 25 (and D38 to D40; his D38 and D39 are D41 and D42 here).*

Setup: `make proxy` defaults (5 dev matches x 3 seeds, simulator v1, **no D12 ball stress**,
`LABELS=keep2 PRIOR_LABELS=dfl`). Reference `touch_c3_seen_w1_dm` (D37, submitted). New:
[solution/ballmodel.py](../solution/ballmodel.py) (imputer, team reader, oracle),
`methods.ball_frame_fn`, [tests/test_ballmodel.py](../tests/test_ballmodel.py). Code at branch
`denis/ball-impute`. **Settings tried: 5** (the imputer, and 4 oracles: true ball, + 5 m noise,
+ 10 m noise (registered, not run), 1 s late), all fixed before the first run.

The method: inside each gap, a HistGBM pair predicts the ball's offset from the straight line
between the last and next seen ball, from those endpoints, the time into and left in the gap,
and player summaries (centroids, spread, densest player, fastest player, player nearest the
line, mean velocity). Trained on degraded dev matches (`framemodel.TRAIN_SEED`) against the true
ball, leave-one-match-out. z = nearest away minus nearest home distance to the inferred ball is
read through a symmetric binned table P(home | z), fitted on cross-fitted imputations. The
decoder reads it through the frame term on gap frames (C3 frame model on seen frames, as now).

Frame level, J03WMX seed 0 (63,833 gap frames, imputer trained on the other four):

| ball in gaps | median error | p90 | 2-10 s gaps | 10-30 s | > 30 s |
|---|---|---|---|---|---|
| imputed | 7.2 m | 24.4 m | 3.4 m | 5.5 m | 10.4 m |
| straight line between the ends | 8.7 m | 42.2 m | 2.8 m | 6.5 m | 15.7 m |

Team right on alive gap frames (same match): imputed ball 0.551; true ball 0.785, + 2 m 0.713,
+ 5 m 0.632, + 10 m 0.613, 1 s late 0.639, 3 s late 0.645; **C3 frame model 0.853**; with the
ball seen, the nearest team is right on 0.740.

Proxy league against `touch_c3_seen_w1_dm` (0.6046, 1,432 possessions; 787 s):

| date | method | spellF1 | interval | macroF1 | possessions | gain | verdict |
|---|---|---|---|---|---|---|---|
| 2026-11-07 | `touch_c3_seen_w1_dm_ball` (imputer) | 0.5996 | 0.5808..0.6178 | 0.8658 | 1444 | -0.0050 (-0.0069..-0.0016) | WORSE, 0/5 |
| 2026-11-07 | `dm_bor_0` (true ball, oracle) | 0.6037 | 0.5899..0.6251 | 0.8687 | 1474 | -0.0009 (-0.0018..+0.0090) | within noise, 2/5 |
| 2026-11-07 | `dm_bor_5m` (true + 5 m) | 0.6033 | 0.5858..0.6226 | 0.8681 | 1444 | -0.0014 (-0.0022..+0.0034) | within noise, 2/5 |
| 2026-11-07 | `dm_bor_lag1` (true, 1 s late) | 0.5993 | 0.5851..0.6213 | 0.8672 | 1468 | -0.0054 (-0.0048..+0.0029) | within noise, 1/5 |

Takeaways:
- **Even the true ball, read as "which team is nearer", adds nothing** on top of the frame model
  and spread. No better imputer (a GNN or set transformer included) can beat its own oracle.
- Through the whole pipeline (touches included) more ball is worth +0.011 to +0.022 at best
  (D29). Both routes put ball inference in gaps under the bar.
- The imputer only beats the straight line in gaps over 10 s, and stays far from the 2 m an
  oracle needs to read the team on even 71% of frames.
- Run without the D12 stress: it changes the seen ball, not the gaps the oracle fills, so the
  oracle's verdict should hold; the imputer's endpoints would only get noisier.

---

## Section 27: REVIEW item 4, the sub-bar gains stacked on the reference (2026-11-07)

*Section 26 on `denis/ball-impute` (PR #9); renumbered at the merge into `miha/dev` on 2026-11-07 because `miha/dev` had taken section 25 (and D38 to D40; his D38 and D39 are D41 and D42 here).*

Setup: the decision run (5 dev matches x 3 seeds, simulator v1, D12 ball stress
`BALL_NOISE=0.05 BALL_JUMPS=3.4 BALL_SMOOTH=0.26`, `LABELS=keep2 PRIOR_LABELS=dfl`), and the
same on `LABELS=dfl`. Reference `touch_c3_seen_w1_dm` (D37; reproduces 0.5565). Code at branch
`denis/ball-impute`. **Settings tried: 2**, both fixed before the first run: `dm_stack3`
(keep2 priors D25, ball merge D27/D32, player-beyond-a-line cue at 0.5, player only, D15) and
`dm_stack3_dead4` (the same plus the `dead` 4 cost, the one try D36 allows inside this stack).
The merge's proxy imitation restores clean frames under the D12 stress, so it is slightly
optimistic (fused.merged docstring).

| date | truth | method | spellF1 | interval | macroF1 | possessions | gain | verdict |
|---|---|---|---|---|---|---|---|---|
| 2026-11-07 | keep2, D12 | touch_c3_seen_w1_dm | 0.5565 | 0.5359..0.5731 | 0.8573 | 1386 | -- | -- |
| 2026-11-07 | keep2, D12 | dm_stack3 | 0.5761 | 0.5580..0.5964 | 0.8584 | 1343 | +0.0195 (+0.0174..+0.0294) | better, below the adoption bar, 5/5 |
| 2026-11-07 | keep2, D12 | **dm_stack3_dead4** | 0.5771 | 0.5604..0.5988 | 0.8587 | 1326 | **+0.0206 (+0.0192..+0.0316)** | **ADOPT, 5/5** |
| 2026-11-07 | DFL, D12 | touch_c3_seen_w1_dm | 0.4601 | 0.4413..0.4760 | 0.8517 | 1386 | -- | -- |
| 2026-11-07 | DFL, D12 | dm_stack3 | 0.4696 | 0.4534..0.4891 | 0.8525 | 1343 | +0.0095 (+0.0069..+0.0183) | better, below the adoption bar, 5/5 |
| 2026-11-07 | DFL, D12 | dm_stack3_dead4 | 0.4699 | 0.4548..0.4905 | 0.8529 | 1326 | +0.0097 (+0.0083..+0.0197) | better, below the adoption bar, 5/5 (not WORSE) |

Takeaways:
- **The stack adds up** roughly to the sum of its parts (+0.009 to +0.011, +0.004 to +0.007,
  +0.005: about +0.02). `dead` 4 adds +0.001 on both truths, as in D36.
- **It is a narrow ADOPT**: +0.0206 against the 0.02 bar, the better of two pre-fixed variants.
  Its interval is well clear of 0 and it wins on every match under both truths.
- Possessions fall 1386 to 1326 (factor 0.957; truth 1455): the stack is more reluctant still.

BAR-ATH, D42's checks, run by denis on 2026-11-07 (label-free aggregates). `tools/gap_check.py
--decoded --bar-ath` (Smart Tagging: 140 attack tags, 140 defend, 4 without a team, 61,216
frames, mirror agree 1.000, home share 0.699):

| method | gap H/(H+A) | gap D | seen D | poss/90 | in-gap poss/90 | in-gap <2 s/90 | in-gap H<->A /90 | tag gap | tag seen |
|---|---|---|---|---|---|---|---|---|---|
| touch_c3_seen_w1_dm | 0.473 | 0.818 | 0.188 | 243.9 | 33.1 | 2.5 | 13.6 | 0.771 | 0.869 |
| dm_stack3_dead4 | 0.502 | 0.805 | 0.180 | 233.7 | 35.7 | 4.2 | 16.1 | 0.768 | 0.872 |

`tools/deadball_check.py --bar-ath`:

| method | D share | stoppages /90 | median s | to D | from D | set pieces with a restart (+-2 s) | throw-ins |
|---|---|---|---|---|---|---|---|
| touch_c3_seen_w1_dm | 0.444 | 96.9 | 18.1 | -- | -- | 0.93 of 58 | 0.96 of 26 |
| dm_stack3_dead4 | 0.435 | 88.4 | 20.8 | 0.0028 | 0.0121 | 0.90 of 58 | 0.88 of 26 |

Corners, free kicks, goals, goal kicks and the kickoff are unchanged. Cue rates as section 8.

- **Fails D42 rules 1 (tag gap 0.768 against 0.771) and 5 (set pieces 0.90 against 0.93).**
  Rules 2 to 4 pass (D share 0.435; possession factor 0.958 against the proxy's 0.957, both
  methods under 250 per 90; gap home share +0.029).
- Both failures point at `dead` 4: it moves frames out of D (0.0121) and drops 8.5 stoppages
  per 90, and the short ones it absorbs include two throw-in restarts.

---

## Section 28: Step 0, what a perfect player tracker is worth (2026-11-07)

*Section 27 on `denis/ball-impute` (PR #9); renumbered at the merge into `miha/dev` on 2026-11-07 because `miha/dev` had taken section 25 (and D38 to D40; his D38 and D39 are D41 and D42 here).*

Setup: the decision run (keep2, D12, 5 dev matches x 3 seeds) with `CLEAN_PLAYERS=1`
(`simulate.degrade(players=False)`, new): the ball is hidden and stressed exactly as usual (its
gaps come from their own random stream, tested), the players stay as in the clean match. The
frame and dead-ball models are still trained on degraded players. **Settings tried: 1.**

| players | method | spellF1 | interval | macroF1 | possessions | seed spread |
|---|---|---|---|---|---|---|
| degraded (section 27) | touch_c3_seen_w1_dm | 0.5565 | 0.5359..0.5731 | 0.8573 | 1386 | 0.0206 |
| **clean** | touch_c3_seen_w1_dm | **0.5770** | 0.5604..0.5959 | 0.8691 | 1375 | 0.0024 |

Per match +0.010 to +0.032 (J03WMX 0.5677, J03WOH 0.5616, J03WOY 0.5622, J03WPY 0.6128,
J03WQQ 0.5830). Two separate runs, so no paired interval.

- **A perfect player tracker is worth about +0.02**, at D4's bar: the size of the stack (D42)
  and under half of a sharper ball (+0.046, D29).
- The simulated player losses are most of the proxy's seed-to-seed spread (0.021 to 0.002).
- It may overstate BAR-ATH: its tagged live gap frames have a team under 5 on 0.1% (section 13)
  against the proxy's 2.3%. Models retrained on clean players could gain a little more.
- Nothing is built: no player re-tracking is possible before the Phase 4 freeze.

## Section 30: Step 0, what SkillCorner's open-data possession labels look like (2026-11-09)

Setup: no proxy-league run. [tools/skillcorner_check.py](../tools/skillcorner_check.py)
`--idsse --public` (16 s). Source: github.com/SkillCorner/opendata (MIT) at commit `4340d27`:
**20** A-League 2024/25 matches in `matches.json` (the README still says 10), broadcast tracking
at 10 fps. Only `matches.json`, `{id}_match.json` and `{id}_phases_of_play.csv` are read. Labels:
H or A where a phase of play covers a frame (`team_in_possession_id`), D elsewhere inside the
match periods (SkillCorner defines phases only with the ball in play), X outside. A phase covers
[frame_start, frame_end): on 6,754 of 8,794 joins a phase's end is the next one's start; the
first reading (inclusive end) overlapped every one of them by one frame. Numbered 30 because
`denis/spell-verify` (not merged yet) holds section 29. **Settings tried: 0.** Description only.

How the phases join (within a period): 6,754 tile, 34 leave 0.1 to 0.9 s, 2,006 leave 1 s or
more (median 20.8 s, the stoppages). 4,747 neighbours are the same team: 3,737 tile (a phase-type
change inside one possession), 1,010 have a gap (a stoppage, the same team restarts). 140 phase
frames fall outside the match periods.

| labels | possessions | per 90 | < 1 s | < 2 s | median s | mean s | H<->A per alive min |
|---|---|---|---|---|---|---|---|
| SkillCorner phases | 5106 | 233 | 0.015 | 0.135 | 7.3 | 12.4 | 2.88 |
| SkillCorner put through keep2 | 4585 | 209 | 0.005 | 0.066 | 8.7 | 13.8 | 2.38 |
| IDSSE dev, DFL | 2191 | 414 | 0.122 | 0.298 | 3.7 | 7.3 | 6.31 |
| IDSSE dev, keep2 (the proxy's truth, D21) | 1455 | 275 | 0.005 | 0.039 | 6.8 | 11.0 | 3.54 |
| public window (structure only, L2) | 17 | 306 | 0.059 | 0.118 | 7.6 | 14.2 | 2.98 |

| labels | stoppages | per 90 | median s | D share |
|---|---|---|---|---|
| SkillCorner phases | 2084 | 95 | 20.1 | 0.466 |
| IDSSE dev, DFL and keep2 | 514 | 97 | 19.0 | 0.441 |
| public window (structure only) | 4 | 72 | 16.4 | 0.196 |

keep2 changes 0.35% of SkillCorner's graded frames.

- **SkillCorner's possessions look like controlled possession, not first touches.** Share under
  2 s 0.135 and median 7.3 s, against the public window's 0.118 and 7.6 s and DFL's 0.298 and
  3.7 s; keep2 barely moves them. Its stoppages match IDSSE's (95 per 90, median 20 s, D share
  0.47 against 0.44).
- It has fewer possessions per 90 than the window or keep2 (233 against 275 and 306): the 2.88
  direct changes per alive minute sit between keep2's 3.54 and the window's 2.98.
- Caveats: phases are SkillCorner's own (model-derived) segmentation, not a stated possession
  rule; the tracking is broadcast (off-screen players extrapolated, 10 fps), unlike both IDSSE
  and BAR-ATH, so it would need its own degradation check before entering the proxy league. The
  per-frame `possession` field of the tracking (Git LFS, about 90 MB a match) is not read here.
- What it could be: a second, independent controlled-possession label source with four times
  IDSSE's dev matches. Not built: a loader and a proxy-league entry are a setup change after the
  Phase 4 freeze, and miha's call.

## Section 31: whistle with both audio tracks, synthetic Step 0 of the pairing (2026-11-10)

**Question.** Does pairing the tactical and broadcast audio keep the referee's whistle and drop
crowd whistling (tactical only) and commentary (broadcast only)? Synthetic audio only
(`tests/test_whistle_broadcast.py`, 9 tests); nothing ran on BAR-ATH. Pass criteria for the
BAR-ATH run: D44.

**Scenes.** 120 s. Tactical: D33's synthetic crowd, 18 referee blasts at 3.8 kHz and 18 crowd
whistles at 2.3-2.9 kHz, all heard 0.2 s late (sound to the stand). Broadcast: half the crowd
bed, a close-miked commentator (8 s phrases; gliding f0, harmonics to 6 kHz with formant bumps
at 2.7 and 3.5 kHz, sibilants at 4-8 kHz), the same blasts; it starts 12 s into live time and
has a 1 s unaligned gap.

**One feed, a sweep of voices** (3 spectral tilts x 3 f0 ranges x 3 levels, 30 s scenes, whistle
amplitude 0.04): soft or dark voices leave STRICT clean; loud or bright ones (tilt -6 or -9
dB/oct, f0 180-320 Hz) miss blasts and add up to 14 false STRICT events in 30 s. Tactical alone
fires on the crowd whistles, as on BAR-ATH (D33).

**Pairing** (`PAIR_TOL_S` 0.24 s, `PAIR_HZ` 250 Hz after 150 Hz lost pairs to the warble: the
per-event pitch median moves by up to 170 Hz between feeds; `feed_lag` reuses the pairing at a
1 s tolerance, after STRICT-only pairs were too few to estimate a lag):

| broadcast voice | seeds | referee blasts paired (of 18) | false pairs | chance pairs per shift | lag (true 5 frames) |
|---|---|---|---|---|---|
| 4 moderate voices | 2 each | 18 | 0 to 1 | 0 to 2 | 5 to 6 |
| harsh (f0 180-320 Hz, tilt -9, level 0.3) | 2 | 7 | 5 to 6 | 2 to 6 | 5 |

- Where the broadcast hears the referee at all, pairing keeps every blast and drops the crowd.
- **Limit:** a loud, bright voice whose formant sits on the crowd's pitch pairs with the crowd's
  whistling. The shifted chance pairing then runs at a third of the real rate or more, which is
  why the BAR-ATH report prints the chance rate beside every paired rate (D44 criterion 4).
- How loud BAR-ATH's commentary is against the whistle is unknown; the BAR-ATH run answers it.

---

## Section 32: keeper's ball against goal kick, and team shape at stoppages, keep2 truth, simulator v1 + D12 ball stress (2026-11-11)

**Question (denis, after watching the whole BAR-ATH match against `voters_blocks_v7_keep2_control`).**
The keeper holding the ball (alive, his team) is decoded D; a goal kick (dead from the ball going
out) is first decoded as a team. Celebrations, the drinks break and the walk-offs are decoded as
possession while the teams stand off the middle of the pitch. Branch `denis/keeper-ball`.

**Dead-ball ceilings against the submission** (`tools/deadball_check.py --methods
v7_keep2_control`; section 8 measured them against `v3_kickoff`): perfect D +0.0816; edges
+0.0551 (start +0.0284, end +0.0313); missed +0.0140; false +0.0129. By type: throw-in +0.0288,
free kick +0.0224, goal kick +0.0114, corner +0.0044, goal +0.0019. 33% of stoppages end more
than 1 s off.

**Team shape ("belt"), Step 0** (`tools/stoppage_shape_check.py`). Per frame, swap-invariant:
largest |team centre y - W/2| (`lat_max`), |all players' centre y - W/2|, smaller team gyration,
share beyond a touchline; within-match rank held 3 s. At rank 0.99 they flag 0.06-0.36% of live
frames and 1.5-2.0% of D frames, but only 0.1-0.8% of the D frames the submission misses:
those are stoppage edges with an ordinary shape. As forced-D rules (12 settings: 4 features x
rank 0.95/0.98/0.99): best +0.0006 (`gyr_min` 0.99), most negative, down to -0.0161. **Closed on
the proxy.** Walk-offs and drinks breaks are not on IDSSE (D33), so a belt guard could only come
through D14 (long sustain, BAR-ATH check fixed before the run).

**Keeper holds on IDSSE.** DFL's keeper events (BallClaiming BallHeld, saves held; BallClaimed is
any player) are alive in the truth: 50 of 52 H/A over the next 3 s. Keeper on the ball (clean
ball within 2 m of the keeper in his own area, >= 1 s): 38.8 stretches per match, all 66 DFL keeper
events within 3 s of one. The submission decodes 18.1% of those frames D; 10% of holds mostly D.
Oracles: goal kicks repaired +0.0114, false D touching holds released +0.0085, both +0.0195.

**Is the keeper's ball lost on BAR-ATH?** (`make keeper-check`, label-free; denis ran `--bar-ath`.)

| | BAR-ATH | proxy | keeper's ball hidden 0.5 / all | lost near goal 0.5 (`LOSE_NEAR_GOAL`) | lost near goal, all |
|---|---|---|---|---|---|
| ball seen in the keeper zone | 0.299 | 0.419 | 0.365 / 0.315 | 0.357 | 0.304 |
| ball seen outside it | 0.650 | 0.638 | 0.637 / 0.636 | 0.624 | 0.613 |
| gaps ending at a keeper (share) | 0.129 | 0.026 | 0.020 / 0.014 | 0.134 | 0.222 |
| their median length | 16.2 s (n 19) | 6.5 s | 5.8 / 2.4 s | about 1.4 s | about 1.3 s |

BAR-ATH loses the ball in the keeper zone more than the proxy does even with every keeper hold
hidden, and one gap in eight ends with the ball at a keeper, long: more like a goal kick (out
behind the goal, ball reappears when the keeper places it) than a catch. DFL keeps tracking the
ball behind the line, so the proxy never does. New stress `keeper.lose_ball_near_goal` (out over
a goal line: hidden until a keeper has it; keeper's ball: hidden but its last 1 s). Calibration
tried (label-free): `hide_keeper_ball` 1/0.5/0, `thin_beyond` 0.26/0 with it (overshoots outside:
0.57), `lose_ball_near_goal` 1/0.5/0.25/0 and out x hold 0/0.5 x 1/0.5/0. No setting matches all
of BAR-ATH (the gap lengths stay short); 0.5 matches the gap share, 0 the zone's seen share, so
the two bracket it.

**Keeper model** (`solution/keepermodel.py`): in the keeper zone (last seen ball within 20 m of
a goal line, and unseen, near the keeper or in the goal area; 15.6% of graded frames, 71% D), 25
features from the zone team's side, 3 classes (alive own, alive other, dead), GBM
leave-one-match-out. Decoder term `keeper_w` (block mean of log P(state | x) - log prior).
Frame level, argmax in the zone (seeds 0-2, D12): keeper on the ball 0.925 / 0.903 / 0.873 with
the keeper's ball kept 1 / 0.5 / 0; goal-kick frames 0.986-0.987 ("always D": 0 and 1).
Two settings, each fixed before its first league result: `v7_keeper` (trained with the keeper's
ball hidden at 0.5) and `v7_keeper2` (registered after the BAR-ATH gap check: trained with
`lose_ball_near_goal` 0.5 instead). Both `keeper_w` 1.

Proxy league against `v7_keep2_control` (5 dev x 3 seeds, D12; gain on the first seed's paired
bootstrap):

| date | method | stress | spellF1 | interval | macroF1 | possessions | gain | verdict |
|---|---|---|---|---|---|---|---|---|
| 2026-11-11 | v7_keep2_control | keep2 + D12 | 0.5854 | 0.5632..0.6033 | 0.8619 | 1185 | -- | reference |
| 2026-11-11 | v7_keeper | keep2 + D12 | 0.5963 | 0.5751..0.6162 | 0.8693 | 1188 | +0.0108 (+0.0066..+0.0169) | better, below the adoption bar; 5/5 |
| 2026-11-11 | v7_keeper2 | keep2 + D12 | 0.5971 | 0.5770..0.6177 | 0.8692 | 1187 | +0.0117 (+0.0083..+0.0190) | better, below the adoption bar; 5/5 |
| 2026-11-11 | v7_keeper | DFL + D12 (ref 0.4525) | 0.4617 | 0.4428..0.4798 | 0.8621 | 1188 | +0.0093 (+0.0034..+0.0114) | better; 5/5 |
| 2026-11-11 | v7_keeper2 | DFL + D12 | 0.4620 | 0.4434..0.4800 | 0.8619 | 1187 | +0.0095 (+0.0034..+0.0123) | better; 5/5 |
| 2026-11-11 | v7_keeper | + HIDE_KEEPER 0.5 (ref 0.5819) | 0.5945 | 0.5729..0.6140 | 0.8685 | 1192 | +0.0126 (+0.0082..+0.0182) | better; 5/5 |
| 2026-11-11 | v7_keeper | + HIDE_KEEPER 0 (ref 0.5792) | 0.5925 | 0.5714..0.6125 | 0.8675 | 1195 | +0.0132 (+0.0095..+0.0196) | better; 5/5 |
| 2026-11-11 | v7_keeper | + LOSE_NEAR_GOAL 0.5 (ref 0.5853) | 0.5942 | 0.5734..0.6142 | 0.8680 | 1187 | +0.0089 (+0.0049..+0.0151) | better; 5/5 |
| 2026-11-11 | v7_keeper2 | + LOSE_NEAR_GOAL 0.5 | 0.5948 | 0.5745..0.6153 | 0.8680 | 1186 | +0.0096 (+0.0061..+0.0161) | better; 5/5 |
| 2026-11-11 | v7_keeper | + LOSE_NEAR_GOAL 0 (ref 0.5816) | 0.5901 | 0.5687..0.6101 | 0.8664 | 1186 | +0.0085 (+0.0045..+0.0145) | better; 5/5 |
| 2026-11-11 | v7_keeper2 | + LOSE_NEAR_GOAL 0 | 0.5923 | 0.5717..0.6123 | 0.8670 | 1183 | +0.0107 (+0.0067..+0.0175) | better; 5/5 |

- Every match gains in every run. Under D4 (+0.02) neither is adopted; no DECISIONS entry yet.
- `v7_keeper2` is at least as good as `v7_keeper` everywhere and holds its gain best under the
  two stresses that bracket BAR-ATH (+0.0096, +0.0107).
- **Where the gain comes from** (`tools/stoppage_shape_check.py --methods v7_keep2_control
  v7_keeper2`, same setup): keeper-on-ball frames decoded D 18.1% -> 9.3% (holds mostly D 10.0%
  -> 4.3%; right team 79.2% -> 88.6%); goal-kick stoppages covered by D 0.854 -> 0.912, invented
  possessions per stoppage 0.15 -> 0.09, lead-in on the restarting team 0.31 -> 0.12; after goals
  covered 0.909 -> 0.975, invented 0.64 -> 0.14 (the ball in or behind the goal is in the zone);
  throw-ins, free kicks and corners within 0.01. Oracle "goal kick + hold" on top of v7_keeper2
  +0.0127 (on v7: +0.0195).
- Not yet run: TEMPO=0.9 (the model reads keeper speed and opponents' velocities in m/s, D26),
  a weight check, and label-free BAR-ATH aggregates (D share, possessions per 90, Smart Tagging
  goal-kick restarts) before any submission talk.

---

## Section 33: the two-tone whistle rule on BAR-ATH, Step 0 (2026-11-12)

**Question.** Does `whistle.two_tone_events` (D44 update; settings `TT_*` fixed before the scan)
find the referee's whistle in each recording? denis ran `tools/whistle_two_tone_check.py`;
aggregates only.

| | broadcast | tactical |
|---|---|---|
| minutes | 109.7 | 132.9 |
| whistles (per min) | 79 (0.72) | 78 (0.59) |
| candidates (per min) | 484 (4.41) | 851 (6.40) |
| median low / high pitch, spacing | 3781 / 4234 Hz, 453 Hz | 3781 / 4234 Hz, 453 Hz |
| length p50 / p90 / max | 0.44 / 0.90 / 1.68 s | 0.62 / 1.18 / 1.80 s |
| rejected: one band / not together / spacing / unsteady | 278 / 99 / 16 / 12 | 574 / 181 / 9 / 9 |
| `--spot 10` spot check, referee | 5 of 8 | 7 of 8 |

- Tactical per half: 0.60 and 0.74 whistles/min; none in the half-time break's 10-minute bin.
- End picks (tactical): half 1 starts 78 s before the half's last graded frame, half 2 1.2 s
  before. Both kickoff whistles found (calibration, not evidence).
- Cross-feed (`--cross`): 70 of 79 broadcast whistles map to the tactical timeline; 65 of 78
  tactical whistles fall where the broadcast airs. 46 pair within 0.24 s at a lag of -2
  frames (71% of the aired tactical whistles, 66% of the mapped broadcast ones); at +-10 and
  +-20 s: 0, 1, 2, 0 pairs (real / chance 61). By half: 22 and 21 pairs.
- Many whistles sit at the bottom of both pitch bands (low tone near 3.70 kHz, high near
  4.18 kHz), so the bands likely clip some real whistles; not tuned on this scan.

**Verdict (D44).** Fails as written: broadcast spot check 0.62, under 0.8. The tactical scan passes. The broadcast mix is not free of fan whistling.

---

## Section 34: whistle veto, Step 0 on the proxy (2026-11-12)

**Question.** Does releasing decoded stoppages that sit away from the lines and have no whistle
near them gain on the current reference? (D45.) `tools/whistle_veto_check.py`, IDSSE dev x seeds
0-4, simulator v1 + D12, keep2 truth, `v7_keep2_control`. Whistle = start of each true D stretch
whose restart is free kick, penalty, goal or dropped ball (perfect), then thinned (`miss`) or
padded with fan whistles; a stretch is covered by a whistle in [start - 5 s, stop]; its place is
the median seen ball inside, else the last seen ball up to 10 s before.

Per seed: 518 decoded stoppages, 48 false (touch no true D), 136 whistled true stoppages
(0.29/min), 45 with no ball place. Share at least 10 m from every line: false stoppages 0.59,
stoppages touching true D 0.16. As is 0.5848; false-stoppage oracle +0.0127.

| margin | miss | fans/min | gain | wins | vetoed/seed | share false | released live s/seed | released dead s/seed |
|---|---|---|---|---|---|---|---|---|
| 0 m | 0 | 0 | -0.0424 | 0/5 | 154.4 | 0.27 | 507 | 2281 |
| 5 m | 0 | 0 | -0.0070 | 0/5 | 75.2 | 0.49 | 338 | 818 |
| 10 m | 0 | 0 | -0.0035 | 0/5 | 49.0 | 0.49 | 219 | 508 |
| 15 m | 0 | 0 | -0.0008 | 1/5 | 28.6 | 0.48 | 113 | 257 |
| 20 m | 0 | 0 | -0.0020 | 0/5 | 16.4 | 0.39 | 61 | 148 |
| 10 m | 0.1 | 0 | -0.0072 | 0/5 | 54.0 | 0.45 | 225 | 701 |
| 10 m | 0.2 | 0 | -0.0098 | 0/5 | 58.6 | 0.41 | 242 | 903 |
| 10 m | 0.3 | 0 | -0.0123 | 0/5 | 63.0 | 0.38 | 248 | 1007 |
| 10 m | 0.5 | 0 | -0.0205 | 0/5 | 72.6 | 0.33 | 273 | 1476 |
| 10 m | 0 | 0.5 | -0.0017 | 1/5 | 41.8 | 0.53 | 168 | 338 |
| 10 m | 0.3 | 0.5 | -0.0082 | 0/5 | 51.6 | 0.42 | 193 | 633 |

**Verdict.** Closed (D45): the ceiling (+0.0127) is under D4's 0.02 and the perfect-whistle veto
loses everywhere. Likely reasons, not checked one by one: pieces of a split stoppage after the
first have no whistle within 5 s; the last seen ball before a gap is often mid-pitch for a ball
that went out; restarts not matched to an event count as unwhistled.

---

## Section 35: the whistle as a decoder cue: proxy, BAR-ATH label-free (2026-11-12)

**What.** `solution/whistlecue.py` (D46): per frame, time since the last whistle (under 3 s, 3-15,
15-45, 45 s or none) crossed with the ball's place (at least 10 m from every line, nearer,
unknown), read by the decoder at `whistle_w` 1 through a table of log P(category | alive / dead)
fitted on the proxy with imitated whistles. BAR-ATH's whistles: the two-tone detector on the
tactical audio. Methods: `v7_whistle`, `v7_whistle_end` (plus D33's end-of-half rule, used only
within 120 s of the end), `v7_whistle_pess` (miss 0.5, 0.2 false/min), `v7_place` (the control:
the same table with no whistles). Settings fixed before the first run.

Fitted table, log P(dead) / P(alive): whistle under 3 s +1.0 to +2.2; 3-15 s +0.6 to +1.8;
45 s or none in mid-pitch -2.2, near a line +0.3.

**Proxy** (IDSSE dev x seeds 0-2, v1 + D12; gain over `v7_keep2_control`, paired interval, wins):

| method | keep2 spellF1 | gain | interval | wins | DFL gain | interval | wins |
|---|---|---|---|---|---|---|---|
| v7_keep2_control | 0.5854 | | | | (0.4525) | | |
| v7_whistle | 0.5935 | +0.0081 | +0.0031..+0.0184 | 4/5 | +0.0071 | +0.0010..+0.0128 | 5/5 |
| v7_whistle_end | 0.5933 | +0.0079 | +0.0031..+0.0184 | 4/5 | +0.0069 | +0.0010..+0.0128 | 5/5 |
| v7_place | 0.5840 | -0.0014 | -0.0074..+0.0067 | 2/5 | -0.0007 | -0.0053..+0.0071 | 1/5 |
| v7_whistle_pess | 0.5874 | +0.0020 | -0.0050..+0.0091 | 3/5 | | | |

D4: better, below the adoption bar (0.02), on both truths. The gain is the whistles' (the
place-only control is noise) and shrinks with a worse detector. The end rule cannot show on the
proxy (halves end at the whistle).

**BAR-ATH label-free** (`deadball_check.py --bar-ath`, `gap_check.py --decoded --bar-ath`;
v7_keep2_control -> v7_whistle_end): D share 0.443 -> 0.458 (IDSSE range 0.41-0.48); stoppages
per 90 95.2 -> 85.0, median 18.5 -> 20.7 s; frames to D 0.023, from D 0.008; possessions per 90
200.6 -> 191.2; in gaps: possessions per 90 27.2 -> 16.1, gap D 0.817 -> 0.843, seen D 0.186 ->
0.194, home share 0.481 -> 0.513; Smart Tagging team agreement in gaps 0.790 -> 0.775, seen
0.890 -> 0.892; set pieces with a decoded restart within +-2 s 54 -> 53 of 58 (one corner).


---

## Section 36: the self-calibrating whistle detector, before BAR-ATH (2026-11-16)

**What.** D47: `solution/whistle.py` finds the whistle by code alone. It takes kickoffs from the
tracking (`kickoff_anchors`), the pitch signature repeated before them against chance windows
(`calibrate`), then detects in those bands (`whistle_events`). Nothing on BAR-ATH has run yet.

**Synthetic audio** (`tests/test_whistle_selfcal.py`; 10 min of crowd, a made-up referee at
3150 + 3620 Hz or a pea whistle at 2850 Hz, 3 kickoffs, 9 other whistles, 8 sliding fan whistles
including some right before kickoffs, a steady stadium horn away from kickoffs):
- the two-tone referee is calibrated as a pair within 40 Hz of both tones, at all 3 anchors (2
  seeds); the pea whistle as a single tone within 40 Hz; leaving out any one anchor gives the
  same tones;
- no whistle at the kickoffs (only fans there): no calibration; fewer than 2 anchors: none;
- detection finds all 12 blasts within 2 frames and nothing else (no fan, no horn).
Settings found on these: calibration counts a tone only if it holds 0.3 s (a fan sliding through
the same cell before two kickoffs was taken otherwise), and the event steadiness is 50 Hz (a
glide across the ±150 Hz band has a std of 87 Hz and passed 100 Hz).

**Anchors on IDSSE** (dev, simulator v1, seeds 0-2): 35 anchors, all within 10 s of a DFL kickoff
or goal restart, 0.91 within 3 s, offset p10 / p50 / p90 -0.3 / +1.1 / +2.1 s. IDSSE halves start
at the kick, so these are the restarts after goals (12 anchors on seed 0 against 13 goals).

**Proxy with the a-priori variant** ("prior": miss 0.3, 0.1 false/min, jitter 0.3 s; D12 stress,
IDSSE dev x 3 seeds):

| method | keep2 spellF1 | gain | interval | wins | DFL gain | interval | wins |
|---|---|---|---|---|---|---|---|
| v7_keep2_control | 0.5854 | | | | (0.4525) | | |
| v7_whistle | 0.5930 | +0.0075 | +0.0029..+0.0178 | 5/5 | +0.0069 | +0.0010..+0.0128 | 5/5 |
| v7_whistle_end | 0.5927 | +0.0073 | +0.0029..+0.0178 | 4/5 | | | |

D4: better, below the adoption bar, as with the old variant (section 35).

**Step 0 on BAR-ATH** (`make whistle-selfcal`, run by denis; all of its output is aggregates):

| | half 1 | half 2 |
|---|---|---|
| kickoff anchors from the tracking | 1 | 2 |
| whistles (per min) | 38 (0.72) | 51 (0.96) |
| grouped onsets | 35 | 48 |
| onsets with a decoded stoppage start in [-2, +6] s | 0.40 | 0.40 |
| the same, onsets shifted +-30 / +-60 s (chance) | 0.18 | 0.14 |

Calibration: a pair of tones, 3828 + 4266 Hz, at 3 of 3 anchors, in 0.04 of 23 chance windows
(ratio 23); leaving out any one anchor gives the same. Verdict: all four criteria pass (D47).
Note: 0.60 of onsets (about 0.4-0.55/min) have no decoded stoppage near them. That is an upper
bound on false whistles; restart whistles, repeat blasts and late decoded stoppages also land
there. It is above both proxy variants' false rates, hence `v7_whistle_matched` (D47 update).

**Proxy matched to that bound** (`v7_whistle_matched`: miss 0.3, false 0.5/min, jitter 0.3 s;
same setup as above):

| method | keep2 spellF1 | gain | interval | wins | DFL gain | interval | wins |
|---|---|---|---|---|---|---|---|
| v7_whistle_matched | 0.5871 | +0.0017 | -0.0040..+0.0110 | 3/5 | +0.0024 | -0.0023..+0.0094 | 4/5 |

Within noise on both truths. The in-play cue's value falls from +0.0075 (0.1 false/min) to about
nothing at the upper-bound noise.

---

## Section 37: the end-of-half rule alone (`v7_end`), D14 checks on the proxy (2026-11-17)

`tools/halfend_whistle_check.py` (D48; criteria fixed before the run). IDSSE dev x seeds 0-4, v1 +
D12, keep2; the rule on `v7_keep2_control`'s decode (checked equal to `v7_end` itself); whistles
imitated per variant. Gain = mean over seeds of pooled spellF1 against v7.

| stress | whistles | guard | gain | wins | halves fired | false fires |
|---|---|---|---|---|---|---|
| plain | prior | on | -0.0003 | 1/5 | 18 | 6 |
| plain | matched | on | -0.0004 | 0/5 | 23 | 7 |
| plain | pessimistic | on | -0.0003 | 1/5 | 20 | 10 |
| plain | matched | off | -0.0005 | 0/5 | 38 | 10 |
| walk-off 30 s | prior / matched / pessimistic | on | -0.0003 / -0.0000 / +0.0000 | 0 / 2 / 2 of 5 | 39 / 41 / 32 | 6 / 4 / 4 |
| walk-off 90 s | prior / matched / pessimistic | on | +0.0000 each | 1 / 1 / 0 of 5 | 34 / 38 / 32 | 1 / 0 / 0 |

P1 pass, P2 fail, P3 fail: not adopted (D48). Diagnostic (seeds 0-1, "matched", guard on):
- False fires: picks 14-62 s before the true end of play, losing 4-26 s of live play. A missed
  final whistle leaves a late whistle as the last group.
- v7 labels 0.0 of the synthetic walk-off's frames H/A, and 0.057 of the truly-D frames in the
  last 30 s of plain halves. The stress does not reproduce a failure v7 has.

**BAR-ATH, label-free** (`--bar-ath`, denis): half 1, pick 78.4 s before the end, collective speed
after it 1.06 m/s, players inside 21 -> 4 (median, 30 s before / after), v7 starts 2 possessions
after it (H/A share 0.19). Half 2: pick 1.3 s before the end, 0.93 m/s, v7 starts none (share 0.00).
B1 pass, B2 fail (half 2 fires and removes nothing). The remaining failure is 2 possessions,
about +0.003 at most.
