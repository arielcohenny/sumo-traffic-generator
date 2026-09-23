"""
Build timing plans from the RL run's logged policy outputs, for the fixed-plan test.

Reads outputs/rl_actions.csv.gz of an RL run and converts the clipped outputs of every
decision into the phase durations the RL controller applied, using the app copy's own
functions (softmax, proportions_to_durations). Writes:

    rl_replay.csv   time, tls, d0..d3   RL's durations at every decision (exact replay)
    rl_modal.csv    tls, d0..d3         each junction's most common durations from decision
                                        --settle on, used at every decision (the fixed plan)
    rl_modal_share.csv                  tls, share of decisions (from --settle on) using the modal plan

Cross-check: the durations are compared with the "Example: junction X -> durations [...]"
lines the RL controller printed in run.log (first 24 junctions of every decision).

Usage (from repo root):
    python paper_rl_vs_tm/scripts/make_rl_plans.py \
        --run paper_rl_vs_tm/results/runs/rl_ref --out-dir paper_rl_vs_tm/results/plans
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

APP = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP))
from src.rl.utils import softmax, proportions_to_durations  # noqa: E402

CYCLE_S = 90
MIN_PHASE_S = 10
N_OUT = 4


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--settle", type=int, default=3,
                        help="first decision index used for the modal plan (default 3: t >= 270)")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    a = pd.read_csv(args.run / "outputs" / "rl_actions.csv.gz")
    tls_ids = list(dict.fromkeys(c[len("clipped_"):].rsplit("_", 1)[0] for c in a if c.startswith("clipped_")))

    rows = []
    for _, r in a.iterrows():
        for tl in tls_ids:
            values = [r[f"clipped_{tl}_{k}"] for k in range(N_OUT)]
            d = proportions_to_durations(softmax(values), CYCLE_S, MIN_PHASE_S)
            rows.append({"time": int(r["time"]), "tls": tl, **{f"d{k}": int(d[k]) for k in range(N_OUT)}})
    replay = pd.DataFrame(rows)
    replay.to_csv(args.out_dir / "rl_replay.csv", index=False)

    # Cross-check against the durations the controller printed during the run
    printed = re.findall(r"Example: junction (\w+) -> durations \[([\d, ]+)\]", (args.run / "run.log").read_text())
    per_decision = len(printed) // len(a)
    logged = pd.DataFrame([(t, j, *map(int, d.split(", "))) for t, (j, d) in
                           zip([int(x) for x in a.time for _ in range(per_decision)], printed)],
                          columns=["time", "tls", "d0", "d1", "d2", "d3"])
    merged = logged.merge(replay, on=["time", "tls"], suffixes=("_log", ""))
    mismatches = sum((merged[f"d{k}_log"] != merged[f"d{k}"]).sum() for k in range(N_OUT))
    print(f"cross-check with run.log: {len(merged)} (time, junction) pairs compared, {mismatches} mismatching values")

    settled = replay[replay.time >= int(a.time.iloc[args.settle])]
    modal_rows, share_rows = [], []
    for tl, g in settled.groupby("tls", sort=False):
        counts = Counter(tuple(x) for x in g[["d0", "d1", "d2", "d3"]].values.tolist())
        plan, n = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        modal_rows.append({"tls": tl, **{f"d{k}": plan[k] for k in range(N_OUT)}})
        share_rows.append({"tls": tl, "modal_share": round(n / len(g), 3), "distinct_plans": len(counts)})
    pd.DataFrame(modal_rows).to_csv(args.out_dir / "rl_modal.csv", index=False)
    share = pd.DataFrame(share_rows)
    share.to_csv(args.out_dir / "rl_modal_share.csv", index=False)

    # Start-up then fixed: RL's own durations before the settle time, the modal plan after it
    settle_time = int(a.time.iloc[args.settle])
    modal = pd.DataFrame(modal_rows)
    startup = replay[replay.time < settle_time]
    after = pd.DataFrame([{"time": t, **m} for t in replay.time.unique() if t >= settle_time
                          for m in modal_rows])
    startup_then_modal = pd.concat([startup, after[startup.columns]]).sort_values(["time", "tls"], kind="stable")
    startup_then_modal.to_csv(args.out_dir / "rl_startup_then_modal.csv", index=False)
    print(f"rl_startup_then_modal.csv: {len(startup_then_modal)} rows (RL durations for t < {settle_time}, "
          f"modal plan from t >= {settle_time})")

    print(f"rl_replay.csv: {len(replay)} rows ({len(a)} decisions x {len(tls_ids)} junctions)")
    print(f"rl_modal.csv: {len(modal_rows)} junctions; modal plan share of decisions from t >= "
          f"{int(a.time.iloc[args.settle])}: median {share.modal_share.median():.2f}, min {share.modal_share.min():.2f}")
    print(pd.DataFrame(modal_rows).merge(share, on="tls").to_string(index=False))


if __name__ == "__main__":
    main()
