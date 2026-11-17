# EDA findings: BAR-ATH aggregates

Output of `notebooks/eda/eda_aggregates.py`, run by miha on 2026-10-28. It contains aggregates
only, no rows. The repo is private for now. Before it goes public, confirm with the organisers
that aggregate figures may be published (PLAN.md, open question 10).

## What it means

1. **The ball mostly goes missing when play is slow.** Mean player speed is 1.17 m/s inside ball
   gaps and 2.16 m/s when the ball is visible. Team spread is smaller in gaps too. Before 81 of
   the 150 gaps, the ball was last seen in a wide channel or beyond a line. So many gaps are
   probably stoppages such as throw-ins and balls out of play, not live play.
   - Collective player speed is a strong dead-ball cue, and it is always available.
   - Simulator v0 hides the ball at random times. That under-rates methods that rely on the
     ball. **Simulator v1 should hide it mainly in slow, near-line moments**, calibrated to this
     speed ratio of about 0.54.
2. **The ATD has no ball-state or owning-team channel**, only x, y and speed. The ball's speed
   value is missing on 91% of graded frames and has a correlation of 0.06 with computed speed:
   ignore it and compute speed from positions. Player speeds are in m/s, although the channel
   is named `speed_kph`.
3. **The ball needs cleaning.** It jumps faster than 40 m/s 216 times. It sits beyond a line on
   4.2% of visible frames, which is an out-of-play cue. It is rarely still while tracked: only
   42 stretches of 1 s or more, so the "stationary ball" restart cue is weak.
4. **Player tracking targets for simulator v1**:
   - 20.5 objects per frame on average (p5 14, p50 22);
   - a team shows more than 11 players on 5 to 8% of frames, and fewer than 9 on 8 to 10%;
   - tracklet lifetime median 35 s (p10 2.6 s);
   - same-team duplicates closer than 0.7 m on 6.8% of frames.
5. **Smart Tagging defending codes mirror the attacking codes exactly.** Each pair has the same
   intervals with the team swapped: BUILD UP and DEFENDING IN ATTACKING THIRD, PROGRESSION and
   DEFENDING IN MIDDLE THIRD, CREATING CHANCES and DEFENDING IN DEFENSIVE THIRD, ATTACKING
   TRANSITION and DEFENSIVE TRANSITION.
   - This confirms the sign rule: a defending tag means the other team has the ball.
   - The tag set is about half as informative as it looks.
   - Tags cover 82% of graded frames.
   - Windows are coarse: CREATING CHANCES lasts exactly 30 s, set pieces about 35 s.
   - 32 of 59 set pieces have no team, not just the 26 throw-ins.
6. **The broadcast sees the ball much more often:** on 76% of frames with a valid homography,
   against 59% for the tactical ATD. If the two feeds can be aligned, the broadcast could fill
   many gaps, so backlog E13 is worth more than PLAN.md estimated.
7. **Direction of play:** compare the home and away mean x over a whole half. The teams swap
   order at half time. The first 30 s of a half do not work, because the second half's first
   30 s fall before kickoff.
8. **Audio track: present** (checked 2026-11-05 with `afinfo`): AAC, stereo, 48 kHz, over the
   whole video. It only mattered for the whistle idea (E10), which closed at Step 0: a perfect
   whistle is worth +0.012 to +0.015 (docs/EXPERIMENTS.md section 16, D26).

## Raw output

```
AGGREGATES ONLY. Review before sharing; never paste rows, frames or plots of the data.
parsing tactical ATD (46 slots) ...
  done in 3 s

==============================================================================
A1  ATD schema (names only, no values)
==============================================================================
XML element names and counts:
  BallChannelRef                           3
  Channel                                  3
  Channels                                 2
  Competition                              1
  DataFormatSpecification                  1
  DataFormatSpecifications                 1
  Device                                   1
  Devices                                  1
  FieldSize                                1
  FileDate                                 1
  FileName                                 1
  FrameRate                                1
  GlobalConfig                             1
  Height                                   1
  LocalTeamScore                           1
  MatchParameters                          1
  Metadata                                 1
  Name                                     252
  Player                                   46
  PlayerChannel                            138
  PlayerChannelRef                         138
  PlayerChannels                           1
  Players                                  1
  ProviderGlobalParameters                 1
  ProviderName                             1
  ProviderParameter                        198
  ProviderPlayerParameters                 46
  ProviderSessionParameters                1
  ProviderTeamsParameters                  2
  Score                                    1
  Sensor                                   2
  Sensors                                  1
  Session                                  1
  SessionType                              1
  Sessions                                 1
  ShirtNumber                              46
  SplitRegister                            48
  Start                                    1
  StringRegister                           1
  Team                                     2
  Teams                                    1
  TrackingType                             1
  Unit                                     3
  Value                                    198
  VisitingTeamScore                        1
  Width                                    1
  main                                     1

channel identifiers (digits shown as #):
  BallChannelRef.channelId=s                                   x1
  BallChannelRef.channelId=x                                   x1
  BallChannelRef.channelId=y                                   x1
  Channel.Name=position_x                                      x1
  Channel.Name=position_y                                      x1
  Channel.Name=speed_kph                                       x1
  Channel.id=s                                                 x1
  Channel.id=x                                                 x1
  Channel.id=y                                                 x1
  PlayerChannel.channelId=s                                    x46
  PlayerChannel.channelId=x                                    x46
  PlayerChannel.channelId=y                                    x46
  PlayerChannel.id=player#_speed                               x46
  PlayerChannel.id=player#_x                                   x46
  PlayerChannel.id=player#_y                                   x46
  PlayerChannel.playerId=#                                     x138
  PlayerChannelRef.playerChannelId=player#_speed               x46
  PlayerChannelRef.playerChannelId=player#_x                   x46
  PlayerChannelRef.playerChannelId=player#_y                   x46

track slots in metadata: 46  (home 24, away 22)
parse: records 199319

ball group has 3 comma-separated fields
  field 1: NaN on 40.7% of graded frames; n=94214  p1=0.01  p50=0.50  p99=0.95
  field 2: NaN on 40.7% of graded frames; n=94214  p1=0.02  p50=0.54  p99=1.07
  field 3: NaN on 91.2% of graded frames; n=13953  p1=0.03  p50=1.13  p99=5.26
  (fields 1-2 are x, y in 0-1. A third field near 0 with a long upper tail looks like
   height; one tracking ball speed looks like m/s. Compare with section A2 speeds.)

==============================================================================
A2  ball
==============================================================================
ball visible on 56.9% of all frames, 59.3% of graded frames
  half 1: 59.3% visible; per 5-min block: 0.88 0.83 0.76 0.72 0.56 0.65 0.37 0.57 0.29 0.54 0.20
  half 2: 59.3% visible; per 5-min block: 0.88 0.67 0.74 0.45 0.74 0.61 0.20 0.49 0.48 0.60 0.72

ball-missing runs inside graded halves: 150
  length (s): n=150  p10=0.47  p25=6.09  p50=10.72  p75=21.06  p90=33.81  p99=104.75  max=221.5
  runs >=  1 s:  133  carrying 99.9% of missing frames
  runs >=  5 s:  132  carrying 99.7% of missing frames
  runs >= 10 s:   79  carrying 85.2% of missing frames
  runs >= 30 s:   19  carrying 43.8% of missing frames

visible graded frames with the ball beyond a line: 4.2%  (beyond by > 1 m: 2.7%)
ball speed from consecutive positions (m/s): n=94066  p50=5.08  p90=14.31  p95=17.35  p99=24.60  max=109.6
  jumps above 40 m/s (tracking teleports): 216
stationary-ball stretches (< 0.5 m/s for >= 1 s): 42
  length (s): n=42  p10=1.16  p50=1.98  p90=7.18  covering 2.0% of graded frames
  correlation of ball field 3 with computed speed: 0.06

==============================================================================
A3  players
==============================================================================
tracked objects per graded frame: mean 20.49; n=158857  p5=14.00  p25=21.00  p50=22.00  p75=22.00  p95=22.00
  home: mean 10.20; n=158857  p5=6.00  p25=10.00  p50=11.00  p75=11.00  p95=12.00; frames with > 11: 8.1%; with < 9: 9.6%
  away: mean 10.29; n=158857  p5=7.00  p25=10.00  p50=11.00  p75=11.00  p95=11.00; frames with > 11: 4.7%; with < 9: 8.1%

tracklets (contiguous runs of one slot) inside graded halves: 1219
  lifetime (s): n=1219  p10=2.60  p25=7.92  p50=35.52  p75=145.32  p90=284.63
same-team pairs closer than 0.7 m: 6.8% of graded frames have one; 0.077 pairs per frame
player speed field (m/s): n=2995575  p50=1.41  p90=3.85  p99=6.44  max=9.4

==============================================================================
A4  context of ball gaps
==============================================================================
mean player speed (m/s): inside gaps 1.17, ball visible 2.16
home spread, mean distance to centroid (m): inside gaps 16.32, ball visible 18.19
away spread, mean distance to centroid (m): inside gaps 15.41, ball visible 16.93

where the ball was last seen before each gap:
  wide channel     58 gaps   median length 9.5 s
  penalty box      34 gaps   median length 13.9 s
  middle           34 gaps   median length 8.5 s
  beyond a line    23 gaps   median length 12.4 s
  unknown           1 gaps   median length 0.1 s
distance between last-seen and first-seen ball across a gap (m): n=147  p10=2.71  p50=10.20  p90=35.52

==============================================================================
A5  Smart Tagging
==============================================================================
code                                n  duration (s)                               Team tag counts
ATTACKING TRANSITION               24  n=24  p10=10.66  p50=15.44  p90=20.46      Atletic Club 11, Barcelona 13
BALL IN FINAL THIRD                93  n=93  p10=22.40  p50=27.00  p90=39.46      Atletic Club 39, Barcelona 54
BALL IN THE BOX                    72  n=72  p10=15.60  p50=17.44  p90=21.06      Atletic Club 30, Barcelona 42
BUILD UP                           22  n=22  p10=13.72  p50=20.52  p90=25.16      Atletic Club 7, Barcelona 15
CREATING CHANCES                   30  n=30  p10=30.00  p50=30.00  p90=30.00      Atletic Club 9, Barcelona 19, N/A 2
DEFENDING IN ATTACKING THIRD       22  n=22  p10=13.72  p50=20.52  p90=25.16      Atletic Club 15, Barcelona 7
DEFENDING IN DEFENSIVE THIRD       30  n=30  p10=30.00  p50=30.00  p90=30.00      Atletic Club 19, Barcelona 9, N/A 2
DEFENDING IN MIDDLE THIRD          66  n=66  p10=11.12  p50=17.64  p90=27.74      Atletic Club 47, Barcelona 19
DEFENSIVE TRANSITION               24  n=24  p10=10.66  p50=15.44  p90=20.46      Atletic Club 13, Barcelona 11
GOALS                               4  n=4  p10=94.12  p50=169.26  p90=232.55     Atletic Club 2, Barcelona 2
LONG BALL                          29  n=29  p10=9.02  p50=12.40  p90=18.18       Atletic Club 20, Barcelona 9
PLAYERS IN THE BOX                 97  n=97  p10=18.51  p50=27.68  p90=62.45      Atletic Club 46, Barcelona 51
PROGRESSION                        66  n=66  p10=11.12  p50=17.64  p90=27.74      Atletic Club 19, Barcelona 47
SET PIECES                         59  n=59  p10=29.00  p50=35.00  p90=35.00      Atletic Club 11, Barcelona 16, N/A 32

set pieces by type and half:
  Corner Kick              1st Half   6
  Corner Kick              2nd Half   6
  Free Kick                1st Half   4
  Free Kick                2nd Half   1
  Goal Kick                1st Half   4
  Goal Kick                2nd Half   6
  Kick Off (After Goal)    1st Half   3
  Kick Off (After Goal)    2nd Half   1
  Kick Off (Start)         1st Half   1
  Kick Off (Start)         2nd Half   1
  Throw-in                 1st Half   9
  Throw-in                 2nd Half   17

graded frames covered by any tag: 82.3%
overlap matrix: share of the ROW code's frames also covered by the COLUMN code
            AT  BIFT  BITB    BU    CC  DIAT  DIDT  DIMT    DT     G    LB  PITB     P    SP
  AT      1.00  0.66  0.44  0.04  0.17  0.04  0.17  0.16  1.00  0.00  0.07  0.69  0.16  0.23
  BIFT    0.10  1.00  0.45  0.06  0.31  0.06  0.31  0.22  0.10  0.05  0.04  0.82  0.22  0.36
  BITB    0.13  0.93  1.00  0.08  0.41  0.08  0.41  0.20  0.13  0.06  0.05  0.99  0.20  0.39
  BU      0.03  0.35  0.21  1.00  0.06  1.00  0.06  0.08  0.03  0.00  0.09  0.49  0.08  0.43
  CC      0.07  0.85  0.55  0.03  1.00  0.03  1.00  0.21  0.07  0.07  0.03  0.90  0.21  0.42
  DIAT    0.03  0.35  0.21  1.00  0.06  1.00  0.06  0.08  0.03  0.00  0.09  0.49  0.08  0.43
  DIDT    0.07  0.85  0.55  0.03  1.00  0.03  1.00  0.21  0.07  0.07  0.03  0.90  0.21  0.42
  DIMT    0.05  0.44  0.19  0.03  0.15  0.03  0.15  1.00  0.05  0.03  0.01  0.42  1.00  0.25
  DT      1.00  0.66  0.44  0.04  0.17  0.04  0.17  0.16  1.00  0.00  0.07  0.69  0.16  0.23
  G       0.00  0.19  0.11  0.00  0.09  0.00  0.09  0.05  0.00  1.00  0.00  0.25  0.05  0.21
  LB      0.06  0.24  0.16  0.10  0.06  0.10  0.06  0.03  0.06  0.00  1.00  0.35  0.03  0.27
  PITB    0.08  0.62  0.36  0.07  0.25  0.07  0.25  0.16  0.08  0.05  0.04  1.00  0.16  0.27
  P       0.05  0.44  0.19  0.03  0.15  0.03  0.15  1.00  0.05  0.03  0.01  0.42  1.00  0.25
  SP      0.05  0.51  0.26  0.11  0.21  0.11  0.21  0.17  0.05  0.08  0.06  0.49  0.17  1.00
  key: AT=ATTACKING TRANSITION, BIFT=BALL IN FINAL THIRD, BITB=BALL IN THE BOX, BU=BUILD UP, CC=CREATING CHANCES, DIAT=DEFENDING IN ATTACKING THIRD, DIDT=DEFENDING IN DEFENSIVE THIRD, DIMT=DEFENDING IN MIDDLE THIRD, DT=DEFENSIVE TRANSITION, G=GOALS, LB=LONG BALL, PITB=PLAYERS IN THE BOX, P=PROGRESSION, SP=SET PIECES

==============================================================================
A6  video stream metadata (ffprobe, nothing decoded)
==============================================================================
ffprobe not found; install ffmpeg (brew install ffmpeg) and re-run this section

==============================================================================
A7  broadcast recording
==============================================================================
parsing broadcast ATD, ball only ...
  records 164488; ball visible on 50.5% of frames
  valid homography stretches: 122; length (s): n=122  p10=2.97  p50=22.92  p90=82.10  p99=160.70
  invalid homography stretches: 123; length (s): n=123  p10=4.76  p50=11.16  p90=39.10  p99=79.93
  ball visible where homography valid: 76.4%; where invalid: 0.8%

==============================================================================
A8  direction of play (mean normalised x, 0-1)
==============================================================================
  half 1: home first 30 s 0.265, whole half 0.466;  away first 30 s 0.326, whole half 0.583
  half 2: home first 30 s 0.496, whole half 0.530;  away first 30 s 0.664, whole half 0.415
  (a team well below 0.5 at the start of a half defends the x = 0 goal)
```
