# Findings: why the RL controller outperforms Tree Method

Every number below comes from a file in `results/` (named under each section) and can be
regenerated with the scripts in `scripts/` (see `README.md`). All runs are the scenario of
experiment `exp_20260105_090955` on the reference environment (TAU cluster, SUMO 1.25.0).

Terms:
- **Tree Method (TM)**: the Tree Method controller, 90 s cycle.
- **RL**: PPO checkpoint 1,945,600 steps, 90 s cycle.
- **RL fixed plan**: each junction's most common RL durations (from t = 270 s), played
  every cycle without the model (`results/plans/rl_modal.csv`).
- **SUMO seed**: SUMO's random seed (driver behaviour noise). Network, demand, routes and
  departure times are identical for every seed. The default seed is 23423.

## 1. What the two controllers decide

Both controllers decide the same quantity. Every 90 s, for each of the 34 junctions, they
set the green duration of each phase. Phases run in a fixed order, each lasts at least 10 s,
and the durations sum to 90 s. The 28 four-phase junctions use `partial_opposites`:
NS straight+right, NS left+U-turn, EW straight+right, EW left+U-turn. The 6 boundary
junctions (A0, A2, A4, A5, F0, F5) have 3 phases. Opposing directions always share a phase.

- **Tree Method** moves each phase by at most a cost-dependent step per cycle.
- **RL** outputs 136 values (34 × 4) in [-10, 10], converted by softmax to proportions of
  the 50 s left after the 10 s minima.

## 2. Headline result and its stability

Default SUMO seed (`runs/tm_ref`, `runs/rl_ref`):

| | Throughput (veh/h) | Average trip duration (s) |
|---|---:|---:|
| Tree Method | 8194 | 727.0 |
| RL | 8660 | 577.5 |
| Difference | +5.7% | −20.6% |

Over 10 further SUMO seeds (`results/seeds/seed_summary.txt`):

| | Avg duration: mean (sd, range) | Throughput: mean (sd) |
|---|---|---|
| Tree Method | 775.4 s (54.7, 687.4–839.0) | 7865 (382) |
| RL | 637.8 s (30.9, 581.3–683.8) | 8378 (137) |
| RL fixed plan | 626.8 s (18.7, 592.8–663.1) | 8418 (91) |

Paired over the 10 seeds (95% CI, t distribution):

| Comparison | Average trip duration | Throughput |
|---|---|---|
| RL vs Tree Method | −137.6 s (−17.8%; CI 88.3 to 187.0 s); shorter on 10/10 seeds | +513 veh/h (CI 217 to 809); higher on 9/10 |
| RL fixed plan vs Tree Method | −148.6 s (−19.2%; CI 117.4 to 179.8 s); shorter on 10/10 | +553 veh/h (CI 307 to 798); higher on 10/10 |
| RL vs RL fixed plan | +10.9 s (CI −14.8 to +36.6); no significant difference | −39 veh/h (CI −154 to +76) |

- The default seed is RL's best of 11 seeds by average duration, and Tree Method's 3rd best.
  So the single-run headline overstates the typical gap slightly (−20.6% vs −17.8% on average).
- Tree Method has the largest spread across seeds: sd 54.7 s, vs 30.9 s for RL and 18.7 s
  for the fixed plan.

## 3. The gain is real speed-up, not a different set of finishing vehicles

SUMO's average trip duration covers only vehicles that arrived. On the default seed, 16,132
vehicles arrived in both runs (`results/step0/step0_output.txt`). The 149.4 s gap splits exactly:

| Term | Seconds | Share |
|---|---:|---:|
| Speed-up of the same vehicles (matched) | 136.0 | 91% |
| Composition (which vehicles arrived) | 13.4 | 9% |

For the matched vehicles, TM vs RL:
- **Waiting time:** 465.6 → 371.4 s (−94.2 s).
- **Time loss:** 577.9 → 452.0 s.
- **Route length:** 1613 → 1478 m (−8.4%), with fewer reroutes (4.7 → 3.6).
- **Who gains:** 67.2% of matched vehicles are faster under RL and 32.3% slower. The winners
  gain 3.66 M vehicle-seconds; the losers lose 1.47 M.
- **When:** the gain grows with congestion, from 21 s for departures in the first 15 min
  to 212–239 s for departures between 3600 and 5400 s.

## 4. What RL actually does: a fixed plan per junction

From the signal switches actually displayed (`results/phase2a/phase2a_output.txt`, default seed):

| Per junction over 81 cycles | Tree Method | RL |
|---|---:|---:|
| Share of phase greens unchanged from the previous cycle | 35.1% | 85.7% |
| Mean cycle-to-cycle change of a phase's green | 2.0 s | 0.3 s |

After the first 2–3 cycles, each RL junction holds an almost constant plan
(`results/plans/rl_modal.csv`, `rl_modal_share.csv`):
- **18 junctions: one phase dominant (≥ 40 s), the others at or near the 10 s minimum.**
  - NS straight+right dominant (45–60 s) at B5, C2, C3, C4, C5, D2, E3, F2, F3.
  - EW straight+right dominant (51–60 s) at D5, E4, E5.
  - The rest are the 6 three-phase boundary junctions. For two of them (A5, F0) RL's four
    durations are an equal split (24/22/22/22), but the controller adds the 4th duration to
    phase 2 on three-phase junctions, so the displayed plan has one 44 s phase. Counted on
    RL's outputs, 16 junctions have one dominant phase (section 5).
- **15 junctions: a near-equal split (24/22/22/22).**
- **1 junction** is close to equal (D1: 21/21/27/21).

Overall RL gives NS 55.9% of green time (TM 51.1%). It puts a phase at the 10 s minimum far
more often (22–26% of cycles vs TM's 5–12% for the three smaller phase types) and gives NS
straight+right ≥ 60 s in 18.7% of cycles (TM 0.3%).

The two controllers differ most in the first 15 min (20.6 s of green reallocated per junction
and cycle), and 15.6–17.2 s afterwards. The divergence does not grow with congestion.

## 5. Why RL does not react to traffic

All 136 policy outputs were logged at each of the 82 decisions
(`runs/rl_ref/outputs/rl_actions.csv.gz`, `results/phase2a/rl_outputs_output.txt`):

- **Most outputs are below the lower bound of the action space (−10), and SB3 clips them to −10.**
  - Median 115 of 136 per decision; 105–117 from t = 270 s. No output ever exceeds +10.
  - Raw outputs reach −190.5 (median −15.1).
  - 52 outputs are below −10 at every decision; the closest they come to the bound has a median of −11.9.
- **The outputs barely move with the state.** From t = 270 s the median standard deviation of
  an output over the run is 0.58, and only 13 of 136 outputs ever cross the bound.
- **19 outputs are never clipped. 16 of them are the dominant phase of each of the 16
  junctions whose plan has one dominant phase** (e.g. phase 0 at B5, C2, C3, C4, C5, D2, E3,
  F2, F3). The other 3 are C2 phase 1, A4 phase 1 and D1 phase 2. The plan follows from the clipping:
  - all four outputs of a junction at −10 → equal split
  - one output above −10 → that phase takes nearly all the spare time

Mechanism, from the saved model:
- The policy is a Gaussian with clipping, not squashing (`squash_output = False`).
- Its exploration std is 1.19–1.32 per output.
- Clipped outputs sit a median of 5 std below −10, so during training the sampled actions of
  those outputs were almost always clipped. The environment responded identically, and
  training received no signal to bring them back.

**Not verified:** behavioural-cloning targets. The pretraining encoded missing phases as
log(1e-8) ≈ −18.4, below the bound, which may have pushed outputs past −10 from the start.
This has not been checked.

## 6. The plan alone reproduces RL's performance

Plans built from RL's logged outputs, played through the RL controller in place of the
model's durations (the model still runs; default seed):

| Run | Decisions differing from RL (of 2788) | Throughput | Avg duration |
|---|---:|---:|---:|
| `rl_replay`: RL's exact durations | 0 | 8660 | 577.5 |
| `rl_fixed`: modal plan every cycle | 459 | 8402 | 629.6 |
| `rl_startup`: RL's first 3 cycles, then the modal plan | 391 | 8380 | 632.3 |
| `rl_switching`: RL's durations at the 10 plan-switching junctions, modal elsewhere | 57 | 8445 | 619.1 |
| `rl_startup_switching`: both of the above | 19 | 8317 | 642.1 |

- **The replay is identical to RL** in every trip, signal switch, lane record and route, which
  validates the mechanism.
- **Every plan-based run lies between 619 and 642 s** on the default seed, all far below
  Tree Method's 727.0 s.
- **RL's own 577.5 s cannot be attributed to any decisions.** The run differing from RL in
  only 19 decisions (57 green-seconds over the whole run) is 64.6 s slower. Over 10 seeds the
  fixed plan and RL do not differ (section 2), so this extra is within the outcome's
  sensitivity to small changes.

**Conclusion:** the policy's advantage over Tree Method is its fixed, junction-specific timing
plan, taken as a whole.

## 7. Where the time is saved (descriptive)

Over 11 seeds (01–10 and default), TM vs RL fixed plan. RL gives nearly identical numbers.

**Congestion** (`results/phase2b/phase2b_output.txt`, lane data, all vehicles):

| Mean over 11 runs | Tree Method | RL | RL fixed plan |
|---|---:|---:|---:|
| Waiting (veh-h) | 2878 | 2366 | 2362 |
| Time loss (veh-h) | 3483 | 2831 | 2828 |
| Blocked tail lane-intervals* | 2013 | 1651 | 1638 |

\*A tail lane interval (90 s) is blocked when occupancy ≥ 50%: the queue fills most of the
road back towards the upstream junction.

- Waiting is lower on 11/11 runs for RL and for the fixed plan.
- Waiting is equal in the first 15 min (101 / 100 / 99 veh-h). Under Tree Method it then grows
  faster: 495 vs 378 / 382 veh-h in the last full 15 min.
- RL and the fixed plan have *more* blocked tails than TM in the first 45 min (0–2700 s), and
  fewer from 2700 s.

**Attribution by junction** (`results/phase2d/phase2d_output.txt`): each matched vehicle's trip
time splits exactly into the time on each junction's approach (tail and head edges). Matched
vehicles save 534 veh-h in total (126 s each; positive on 11/11 seeds). Per junction approach:
- **Gains are in the centre and east.** 23 of 34 junctions save time in ≥ 10 of 11 seeds.
  - The top eight (D4, C3, D3, C2, D2, E4, E3, C1) carry 354.6 veh-h, 56% of the junction gains.
- **Losses are in the west.**
  - B4 (−43.5 veh-h), B2 (−20.5), B5 (−19.5), A4 (−18.9), A2 (−9.8) and A1 (−5.7) lose time;
    each saves time in at most 3 of 11 seeds (B4, B5, A4 in none). Together with C5 (−2.3,
    mixed) the losses are 120.2 veh-h.
  - A2, A4, B2 and B4 neighbour the two removed junctions (A3, B3).
- **The NS-dominant decision acts where expected.** At the 9 NS straight+right-dominant
  junctions, 191.6 of their 208.3 veh-h saved (92%) is on NS approaches.
- **Equal-split junctions matter as much.** The 16 equal-split junctions carry 42.6% of the net
  saving (e.g. D4, D3, C1, E1). There RL's plan is a steady equal split.
  - **Not determined:** whether their gain comes from not adapting or from less congestion
    arriving from neighbours. Isolating it would need junction-group swap runs, which were
    deliberately not done: a few chosen groups cannot represent all combinations.

## 8. Limitations and notes

1. **One scenario.** Network, demand and seeds are fixed; this is the paper's scope. The SUMO
   seeds vary only driver behaviour.
2. **Simulator version matters.** The same code and model on SUMO 1.22.0 give different
   numbers: TM 8507 / 708.7 s, RL 8504 / 618.4 s. The pair "TM 8507 / 708.7 vs RL 8660 / 577.5"
   that appears in `docs/RL_DISCUSSION.md` §5.5 mixes SUMO 1.22 (TM) and SUMO 1.25 (RL). The
   numbers above are all SUMO 1.25.
3. **High sensitivity.** 19 decisions changed by 2–3 s moved the average duration by 64.6 s
   (section 6). Single-run differences below roughly this size should not be interpreted.
4. **Section 7 is descriptive.** Only the plan as a whole was tested causally (section 6).
5. **Three-phase junctions (code finding, not quantified).** At inference RL's 4th duration is
   added to phase 2; the training environment instead drops it.
6. **Average trip duration counts only arrived vehicles.** On the default seed 2,837 (TM) and
   2,165 (RL) vehicles are still in the network at 7300 s, and 2,548 / 2,274 were never
   inserted. Section 3 bounds the effect of this (9% of the gap).
7. **Teleports.** 270 (TM) and 257 (RL) on the default seed. Time of a trip that ends by
   teleport lies on no edge. Section 7 keeps it as a separate bucket, and among matched
   vehicles it is ≈ 0 veh-h.

**Context (different environment, SUMO 1.22, December 2025).** On the same scenario and
traffic seeds, SUMO's default equal-split static program gave 5628 veh/h / 881.7 s, against
Tree Method's 8507 / 708.7 s (`../evaluation/comparative_analysis/`). A generic fixed plan is
therefore much worse than Tree Method; the plan RL found is not.
