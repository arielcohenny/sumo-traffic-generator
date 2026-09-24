"""
Phase 2 / analysis B: where and when congestion builds, Tree Method vs RL vs RL's fixed plan.

Reads outputs/lanedata.xml.gz (per lane, per 90 s interval; covers all vehicles, including
those still in the network at the end) of every run of the three controllers over the SUMO
seeds (tm_sNN / rl_sNN / fixed_sNN for NN = 01..10, plus the default-seed runs tm_ref / rl_ref
/ rl_fixed), and reports per run and averaged over runs:

    - network waiting time (vehicle-hours halted) and time loss, over time
    - waiting time per junction approach (all lanes of all edges leading to the junction)
    - blocked tails: tail-edge lane intervals with occupancy >= --blocked-occupancy (%), i.e. the
      queue fills most of the road back towards the upstream junction
    - the same per group of junctions by RL's plan type (from results/plans/rl_modal.csv)

Usage (from repo root):
    python paper_rl_vs_tm/scripts/phase2b_congestion.py \
        --results paper_rl_vs_tm/results --out-dir paper_rl_vs_tm/results/phase2b
"""

import argparse
import gzip
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from netmap import load_edges

CONTROLLERS = {"tm": "Tree Method", "rl": "RL", "fixed": "RL fixed plan"}
DEFAULT_RUNS = {"tm": "tm_ref", "rl": "rl_ref", "fixed": "rl_fixed"}
PERIOD_S = 900


def runs_of(runs_dir: Path, c: str) -> dict:
    """{seed label: run dir} for controller c: SUMO seeds 01..10 and the default seed."""
    runs = {d.name[-2:]: d for d in sorted(runs_dir.glob(f"{c}_s[0-9][0-9]"))}
    runs["default"] = runs_dir / DEFAULT_RUNS[c]
    return runs


def load_lanedata(run_dir: Path) -> pd.DataFrame:
    rows = []
    with gzip.open(run_dir / "outputs" / "lanedata.xml.gz") as f:
        begin = None
        for event, elem in ET.iterparse(f, events=("start", "end")):
            if event == "start" and elem.tag == "interval":
                begin = float(elem.get("begin"))
            elif event == "end" and elem.tag == "lane":
                rows.append((begin, elem.get("id"), float(elem.get("waitingTime", 0)), float(elem.get("timeLoss", 0)),
                             float(elem.get("occupancy", 0))))
                elem.clear()
            elif event == "end" and elem.tag == "interval":
                elem.clear()
    df = pd.DataFrame(rows, columns=["begin", "lane", "waiting_s", "timeloss_s", "occupancy"])
    df["edge"] = df["lane"].str.rsplit("_", n=1).str[0]
    return df


def plan_types(results: Path) -> pd.Series:
    """RL plan type per junction: 3-phase boundary, dominant <phase label> (a phase >= 40 s), or equal split."""
    modal = pd.read_csv(results / "plans" / "rl_modal.csv").set_index("tls")
    labels = pd.read_csv(results / "phase2a" / "phase_labels.csv")
    n_phases = labels.groupby("tls").n_phases.first()
    out = {}
    for tls, d in modal.iterrows():
        durations = d[["d0", "d1", "d2", "d3"]].tolist()
        if n_phases[tls] == 3:
            out[tls] = "boundary (3-phase)"
        elif max(durations) >= 40:
            k = durations.index(max(durations))
            out[tls] = "dominant " + labels[(labels.tls == tls) & (labels.phase == k)].label.iloc[0]
        else:
            out[tls] = "equal split"
    return pd.Series(out, name="rl_plan_type")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--blocked-occupancy", type=float, default=50.0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = []

    def say(text=""):
        print(text)
        out.append(text)

    runs_dir = args.results / "runs"
    edges = load_edges(runs_dir / "tm_ref").set_index("edge")
    ptype = plan_types(args.results)

    per_junction, per_period, per_run = [], [], []
    for c in CONTROLLERS:
        for seed, run in runs_of(runs_dir, c).items():
            ld = load_lanedata(run).join(edges[["approach_junction", "part"]], on="edge")
            ld["blocked"] = (ld.part == "tail") & (ld.occupancy >= args.blocked_occupancy)
            j = ld.groupby("approach_junction").agg(waiting_s=("waiting_s", "sum"), timeloss_s=("timeloss_s", "sum"),
                                                    blocked=("blocked", "sum")).reset_index()
            per_junction.append(j.assign(controller=c, seed=seed))
            p = ld.assign(period=(ld.begin // PERIOD_S * PERIOD_S).astype(int)).groupby("period").agg(
                waiting_s=("waiting_s", "sum"), blocked=("blocked", "sum")).reset_index()
            per_period.append(p.assign(controller=c, seed=seed))
            per_run.append({"controller": c, "seed": seed, "waiting_vh": ld.waiting_s.sum() / 3600,
                            "timeloss_vh": ld.timeloss_s.sum() / 3600, "blocked": int(ld.blocked.sum())})
            print(f"  read {run.name}")
    pj = pd.concat(per_junction)
    pj.to_csv(args.out_dir / "congestion_per_junction_run.csv", index=False)
    pp = pd.concat(per_period)
    pr = pd.DataFrame(per_run)
    pr.to_csv(args.out_dir / "congestion_per_run.csv", index=False)
    n = pr.seed.nunique()

    say(f"Runs per controller: {n} (SUMO seeds 01-10 and the default seed). Blocked tail = tail-edge lane "
        f"interval (90 s) with occupancy >= {args.blocked_occupancy:.0f}%.")

    say("\n=== B1. Network totals (mean over runs; per-run consistency vs Tree Method) ===")
    wide = pr.pivot(index="seed", columns="controller")
    for metric, label in [("waiting_vh", "waiting (veh-h)"), ("timeloss_vh", "time loss (veh-h)"), ("blocked", "blocked tail lane-intervals")]:
        parts = [f"{CONTROLLERS[c]} {wide[metric][c].mean():.0f}" for c in CONTROLLERS]
        cons = [f"{CONTROLLERS[c]} < Tree Method in {(wide[metric][c] < wide[metric]['tm']).sum()}/{n}" for c in ("rl", "fixed")]
        say(f"{label:28s}: " + ", ".join(parts) + " | " + ", ".join(cons))

    say(f"\n=== B2. Network waiting (veh-h) per {PERIOD_S // 60} min, mean over runs ===")
    t = pp.groupby(["period", "controller"]).waiting_s.sum().div(3600).div(n).unstack()[list(CONTROLLERS)].round(0)
    t.columns = [CONTROLLERS[c] for c in t.columns]
    b = pp.groupby(["period", "controller"]).blocked.sum().div(n).unstack()[list(CONTROLLERS)].round(0)
    b.columns = [f"blocked {CONTROLLERS[c]}" for c in b.columns]
    say(pd.concat([t, b], axis=1).to_string())

    say("\n=== B3. Per junction: waiting (veh-h, mean over runs), difference to Tree Method, consistency ===")
    jw = pj.pivot_table(index=["approach_junction", "seed"], columns="controller", values="waiting_s").div(3600)
    jb = pj.pivot_table(index=["approach_junction", "seed"], columns="controller", values="blocked")
    rows = []
    for jn, g in jw.groupby(level=0):
        gb = jb.loc[jn]
        rows.append({"junction": jn, "rl_plan_type": ptype[jn],
                     "TM": g["tm"].mean(), "RL": g["rl"].mean(), "fixed": g["fixed"].mean(),
                     "TM-fixed": (g["tm"] - g["fixed"]).mean(), "fixed<TM runs": int((g["fixed"] < g["tm"]).sum()),
                     "TM-RL": (g["tm"] - g["rl"]).mean(), "RL<TM runs": int((g["rl"] < g["tm"]).sum()),
                     "blocked TM": gb["tm"].mean(), "blocked fixed": gb["fixed"].mean()})
    jt = pd.DataFrame(rows).sort_values("TM-fixed", ascending=False)
    jt.to_csv(args.out_dir / "congestion_per_junction.csv", index=False)
    say(jt.round(1).to_string(index=False))

    say("\n=== B4. By RL plan type: waiting (veh-h, mean over runs, summed over the group's junctions) ===")
    grp = jt.groupby("rl_plan_type").agg(junctions=("junction", "count"), TM=("TM", "sum"), RL=("RL", "sum"),
                                          fixed=("fixed", "sum"), **{"TM-fixed": ("TM-fixed", "sum"), "TM-RL": ("TM-RL", "sum")})
    grp["share of TM-fixed %"] = 100 * grp["TM-fixed"] / grp["TM-fixed"].sum()
    say(grp.round(1).to_string())

    (args.out_dir / "phase2b_output.txt").write_text("\n".join(out) + "\n")
    print(f"\nWritten to {args.out_dir}")


if __name__ == "__main__":
    main()
