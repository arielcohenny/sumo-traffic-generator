# Paper outline

Working outline for the paper on experiment `exp_20260105_090955` and the decision
comparison in this folder.

## Thesis

We trained a reinforcement learning agent to control one junction set and it beat the
adaptive baseline (Tree Method) on that junction set's scenario: +5.7% throughput and
−20.6% average trip duration on the reference seed, shorter trips on 10 of 10 further
SUMO seeds (−17.8% on average).

Framing notes:
- The single scenario is a compute constraint, not the subject. A deployment would train
  the same junction set across demand levels, times of day and many seeds. State this once
  in the setup and once in future work; do not hedge every claim with it.
- The fixed plan the agent converged on is a **product of the training**, not an argument
  that the model was unnecessary. Nobody had that plan before training, and a generic
  equal-split plan performs far worse than Tree Method (5628 veh/h / 881.7 s vs
  8507 / 708.7 on SUMO 1.22).
- Audience: transportation, technical paper.

## Source rules

Every claim comes from `docs/RL_DISCUSSION.md` or from data in this folder
(`FINDINGS.md`, `results/`, `runs/`). No generic background filler, no related work, no
compute-cost discussion, no fixed-time or actuated baselines.

**All performance numbers come from `FINDINGS.md`, not from RL_DISCUSSION.md §5.5.** That
section's pair (TM 8507 / 708.7 vs RL 8660 / 577.5) mixes SUMO 1.22 with SUMO 1.25 and is
invalid. Reference environment: TAU cluster, SUMO 1.25.0.

## Sections

### 1. Introduction and problem

Source: RL_DISCUSSION.md §1.

Network-wide signal control, the coordination problem, and what an RL controller would
have to do. State the question the paper answers: can an agent trained for a given
junction set outperform an adaptive controller on that junction set?

### 2. Formulation choices

Source: RL_DISCUSSION.md §2 (training scope, agent architecture, input model, reward
design, action representation, time resolution, exploration and constraints) and the §2
summary. Condense to the choices made and why; keep the alternatives brief.

### 3. Environment design

Source: RL_DISCUSSION.md §3. State space, action space, reward system, and the alignment
with Tree Method's traffic metrics.

### 4. Training framework

Source: RL_DISCUSSION.md §4. PPO, configuration strategy, multi-objective reward.

### 5. Implementation

Source: RL_DISCUSSION.md §5.1–5.5. Technology stack; how the state, action, reward and
training configuration evolved, including pretraining by behavioural cloning on Tree
Method demonstrations (§5.5 Phase 3), the stability fixes (Phase 4), learning-rate
stabilisation (Phase 5), the throughput coefficient (Phases 6–7) and the final run
(Phase 8).

### 6. Experimental setup

Sources: RL_DISCUSSION.md §5.8, `rl/experiments/exp_20260105_090955/config.yaml` and
`notes.md`, FINDINGS.md §1.

- Network: 6x6 grid, 2 junctions removed (34 signalised), realistic lanes, block 280 m,
  `partial_opposites` phasing, network seed 24208.
- Demand: 22,000 passenger vehicles, inner routes, uniform departures, start 08:00,
  7300 s, traffic seeds 72632 / 27031.
- Control: both controllers set the green duration of each phase of each junction per 90 s
  cycle, fixed phase order, 10 s minimum. 28 four-phase and 6 three-phase junctions.
- Model: PPO checkpoint 1,945,600, selected by throughput and average duration over 474
  evaluated checkpoints (`rl/experiments/exp_20260105_090955/compare_checkpoints_result.csv`).
- Reference environment: TAU, SUMO 1.25.0. Reference SUMO seed 23423; 10 further seeds
  vary driver behaviour only.

### 7. Results

Source: FINDINGS.md §2, §3, §7.

1. **Headline** (FINDINGS §2): TM 8194 / 727.0 s vs RL 8660 / 577.5 s → +5.7%, −20.6%.
2. **Stability** (FINDINGS §2): over 10 further seeds, RL shorter on 10/10, −137.6 s
   (−17.8%, CI 88.3–187.0); throughput +513 veh/h (CI 217–809), higher on 9/10. Note in
   one sentence that the reference seed is RL's best of 11, alongside the 10-seed average.
   Tree Method has the widest spread across seeds (sd 54.7 s vs 30.9 s).
3. **The gain is real speed-up** (FINDINGS §3): 91% of the 149.4 s gap is the same 16,132
   vehicles travelling faster; waiting 465.6 → 371.4 s; 67% of matched vehicles faster.
   The gain grows with congestion (21 s early, 212–239 s for departures 3600–5400 s).
4. **Where the time is saved** (FINDINGS §7, descriptive): waiting 2878 → 2366 veh-h,
   lower on 11/11 runs; gains in the centre and east (top eight junctions carry 56%),
   losses in the west near the removed junctions; at the 9 NS-dominant junctions 92% of
   the saving is on NS approaches.
5. **Training progress** (figure): throughput and average duration against training steps
   from the checkpoint CSV, with the Tree Method baseline as a horizontal line.

### 8. What the agent learned

Source: FINDINGS.md §4, §5, §6 and RL_DISCUSSION.md §6.

1. **A near-constant plan per junction**: a phase's green is unchanged from the previous
   cycle 85.7% of the time, against 35.1% for Tree Method. 18 junctions have one dominant
   phase (45–60 s), 15 a near-equal split (24/22/22/22). RL gives NS 55.9% of green time
   (TM 51.1%).
2. **Mechanism**: a median of 115 of the 136 policy outputs fall below the −10 action bound
   and are clipped; they barely vary with the state (median std 0.58). The 19 never-clipped
   outputs are the dominant phases. From the saved model: Gaussian policy with clipping and
   no squashing, exploration std 1.19–1.32, clipped outputs a median of 5 std below the
   bound, so training received no signal to bring them back. Flag the behavioural-cloning
   target (log(1e-8) ≈ −18.4 for missing phases) as an unverified hypothesis.
3. **The plan carries the gain**: extracted from the agent's own outputs and played every
   cycle, it gives 8402 / 629.6 s on the reference seed, and over 10 seeds is shorter than
   Tree Method on 10/10 (−148.6 s, CI 117.4–179.8) and not significantly different from the
   agent itself (+10.9 s, CI −14.8 to +36.6). Exact replay of the agent's durations
   reproduces its run exactly, which validates the mechanism.
4. **Reading**: the training produced a junction-specific timing plan that beats an adaptive
   controller — timings that can be read, checked and deployed directly. Non-adaptivity is
   an artefact of the action bound, stated plainly, not a design goal.

### 9. Limitations

Source: FINDINGS.md §8.

One scenario (compute constraint; SUMO seeds vary driver behaviour only); simulator
version affects absolute numbers, so all results are SUMO 1.25.0; high sensitivity (19
decisions changed by 2–3 s moved average duration by 64.6 s), so small single-run
differences are not interpretable; §7 attribution is descriptive, only the plan as a whole
was tested causally; average duration counts arrived vehicles only, bounded by §3; the
three-phase junction discrepancy between training and inference; teleports.

### 10. Conclusion and future work

The agent trained for this junction set beat the adaptive baseline. With more compute, the
same junction set would be trained across demand levels, departure patterns and many seeds.
Removing the clipping constraint so the policy can react to traffic is the obvious next
step, and its value is untested here.

## Open items

- Figures: training progress (checkpoint CSV); per-seed duration comparison (FINDINGS §2);
  possibly the junction plan map (FINDINGS §4) and the per-junction attribution map
  (FINDINGS §7).
- Venue not chosen.
- `docs/RL_DISCUSSION.pdf` has not been regenerated since the Chapter 6 update.
