# RL vs Tree Method: decision comparison (paper experiment)

Self-contained experiment folder for the paper on RL traffic signal control
(experiment `exp_20260105_090955`). It explains why the trained RL controller
outperforms Tree Method on one fixed scenario, by comparing the decisions the
two controllers make.

This folder is independent of the main application in the repository root:
it runs its own frozen copy of the app (`app/src`), so changes to the main app
do not affect these results, and nothing here changes the main app.

## Scenario

| Parameter | Value |
|-----------|-------|
| Network | 6x6 grid, 2 junctions removed (34 signalized), block 280 m, realistic lanes, `partial_opposites` |
| Demand | 22,000 passenger vehicles, inner routes, uniform departures, `realtime 100` routing |
| Duration | 7,300 s, start 08:00 |
| Seeds | network 24208, private traffic 72632, public traffic 27031 |
| Controllers | Tree Method (90 s cycle) vs RL checkpoint 1,945,600 steps (90 s cycle) |

Both controllers decide the same quantity: the green duration of each of the
junction's phases (fixed order, min 10 s, sum 90 s) for every 90 s cycle.

## Reference environment

Results are reproducible **only** with SUMO 1.25.0. The same code and model on
SUMO 1.22.0 give different numbers (Tree Method 8507 veh/h / 708.7 s,
RL 8504 veh/h / 618.4 s).

- Python 3.10
- SUMO 1.25.0 (`eclipse-sumo==1.25.0`)
- Python packages: `requirements.txt`

The reference runs were made on the TAU `power` cluster.

## Expected results

| Controller | Throughput (veh/h) | Average trip duration (s) |
|------------|-------------------:|--------------------------:|
| Tree Method | 8194 | 727.0 |
| RL checkpoint 1945600 | 8660 | 577.5 |

Runs are deterministic: repeating a run gives identical numbers.

## Layout

```
paper_rl_vs_tm/
├── README.md
├── requirements.txt
├── app/src/        frozen copy of the app (see "App copy" below)
├── model/
│   ├── rl_traffic_model_1945600_steps.zip   the RL model used in the paper
│   ├── config.yaml                          experiment configuration
│   └── compare_checkpoints_result.csv       all checkpoints on this scenario (selection of 1945600)
├── scripts/
│   ├── run.sh               run one simulation (tm | rl), then pack its outputs
│   ├── pack_outputs.sh      compress a run's SUMO outputs into outputs/*.xml.gz
│   ├── run_seed_batch.sh    SUMO-seed stability runs (tm, rl, fixed plan) for given seeds
│   ├── verify.sh            check a run reproduces the expected numbers
│   ├── step0_decompose.py   split the average-duration gap into speed-up vs composition
│   ├── phase2a_decisions.py analysis A: green per phase per cycle, TM vs RL
│   ├── phase2a_rl_outputs.py analysis A: the RL policy's 136 outputs over the run
│   └── make_rl_plans.py     timing plans from the RL run's outputs (exact replay, fixed modal plan)
└── results/
    ├── runs/<run_name>/
    │   ├── run.log          console log, starting with the versions used
    │   ├── outputs/         compressed SUMO outputs (kept in git, ~4.5 MB per run)
    │   └── workspace/       all files of the run (not kept in git; deleted after packing
    │                        unless KEEP_WORKSPACE=1; regenerated exactly by rerunning)
    ├── step0/               gap decomposition output
    ├── phase2a/             analysis A output
    └── plans/               timing plans (rl_replay.csv, rl_modal.csv)
```

## How to reproduce

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r paper_rl_vs_tm/requirements.txt

paper_rl_vs_tm/scripts/run.sh tm tm_ref
paper_rl_vs_tm/scripts/run.sh rl rl_ref
paper_rl_vs_tm/scripts/verify.sh tm tm_ref
paper_rl_vs_tm/scripts/verify.sh rl rl_ref

python paper_rl_vs_tm/scripts/step0_decompose.py \
  --run-a paper_rl_vs_tm/results/runs/tm_ref --label-a TM \
  --run-b paper_rl_vs_tm/results/runs/rl_ref --label-b RL \
  --out-dir paper_rl_vs_tm/results/step0
```

### Fixed-plan test

`make_rl_plans.py` converts the RL run's logged outputs into the durations the RL controller
applied (cross-checked against the durations printed in its `run.log`):
`rl_replay.csv` (every decision) and `rl_modal.csv` (each junction's most common durations
from t = 270 s, used at every decision). They are played through the RL controller:

```bash
paper_rl_vs_tm/scripts/run.sh rl rl_replay results/plans/rl_replay.csv   # must equal rl_ref exactly
paper_rl_vs_tm/scripts/verify.sh rl rl_replay
paper_rl_vs_tm/scripts/run.sh rl rl_fixed results/plans/rl_modal.csv     # the fixed-plan test
```

### SUMO-seed stability runs

The outcome is very sensitive to small decision changes, so each controller is also run
with different SUMO random seeds (driver behaviour noise only; network, demand, routes and
departure times are unchanged):

```bash
paper_rl_vs_tm/scripts/run_seed_batch.sh 1 2   # tm_s01, rl_s01, fixed_s01, tm_s02, ... one after another
```

Seeds 1-10 were run as five such batches (1 2, 3 4, 5 6, 7 8, 9 10). A run with
`SUMO_SEED=23423` (SUMO's default) reproduces `rl_ref` exactly.

`run.sh` stops if Python imports `src` from anywhere other than `app/src`, and
records Python, SUMO and package versions plus the model md5 at the top of `run.log`.
When the simulation ends it runs `pack_outputs.sh`, which writes the files below to
`results/runs/<run_name>/outputs/` as `.xml.gz`. The analysis scripts read only those.

## App copy

`app/src` is a copy of `src/` at commit `45762cc`. Every change made to it for
this experiment is marked with a `PAPER_RL_VS_TM` comment:

| File | Change |
|------|--------|
| `sumo_integration/sumo_utils.py` | adds SUMO-native logging outputs (below) to the generated `grid.sumocfg` |
| `rl/controller.py` | writes all 136 policy outputs per decision to `rl_actions.csv` (RL runs only); with `--rl-plan-file`, replaces the model's durations by a timing plan's (the model still runs and its outputs are logged) |
| `args/parser.py` | adds the `--rl-plan-file` and `--sumo-seed` arguments |
| `orchestration/simulator.py` | passes `--sumo-seed` to SUMO as `--seed` (not passed by default: SUMO's default seed 23423) |

The logging outputs are passive and do not change the simulation (runs with
logging reproduce the numbers above exactly):

| File | Content |
|------|---------|
| `tls_switches.xml` | every signal state switch of every traffic light (what was actually displayed) |
| `lanedata.xml` | per lane, per 90 s interval: waiting time, time loss, occupancy, speed, vehicles entered/left |
| `vehroutes.xml` | final route of every arrived vehicle, with the exit time of each edge |
| `tripinfo.xml`, `summary.xml`, `sumo_statistics.xml` | standard outputs (unchanged) |
| `grid.net.xml` | the network (lane → junction / phase mapping for the analysis) |
| `rl_actions.csv` | RL runs only: per decision (every 90 s), the 136 policy outputs before (`raw_<tls>_<k>`) and after (`clipped_<tls>_<k>`) clipping to the action bounds [-10, 10] |
