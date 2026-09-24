"""
SUMO-seed stability: Tree Method vs RL vs RL's fixed plan over SUMO seeds.

Reads outputs/sumo_statistics.xml.gz of the runs tm_sNN, rl_sNN, fixed_sNN (and, for
reference, the default-seed runs tm_ref, rl_ref, rl_fixed). Throughput and average
duration are computed exactly as the app does (src/utils/statistics.py):
    throughput   = (loaded - running - waiting) / (performance duration / 3600)
    avg duration = vehicleTripStatistics@duration
and checked against the numbers printed in each run.log.

Usage (from repo root):
    python paper_rl_vs_tm/scripts/seed_summary.py \
        --runs paper_rl_vs_tm/results/runs --out-dir paper_rl_vs_tm/results/seeds
"""

import argparse
import gzip
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

CONTROLLERS = {"tm": "Tree Method", "rl": "RL", "fixed": "RL fixed plan"}
DEFAULT_RUNS = {"tm": "tm_ref", "rl": "rl_ref", "fixed": "rl_fixed"}  # SUMO default seed 23423
T_975 = {9: 2.262}  # two-sided 95% t quantile for n - 1 = 9 degrees of freedom


def run_metrics(run_dir: Path) -> dict:
    with gzip.open(run_dir / "outputs" / "sumo_statistics.xml.gz") as f:
        root = ET.parse(f).getroot()
    veh = root.find("vehicles")
    arrived = int(veh.get("loaded")) - int(veh.get("running")) - int(veh.get("waiting"))
    hours = float(root.find("performance").get("duration")) / 3600
    m = {"throughput": arrived / hours, "avg_duration": float(root.find("vehicleTripStatistics").get("duration"))}

    log = (run_dir / "run.log").read_text()
    logged_thr = float(re.search(r"Throughput: ([0-9.]+) veh/h", log).group(1))
    logged_dur = float(re.search(r"Average duration: ([0-9.]+)s", log).group(1))
    if round(m["throughput"]) != logged_thr or round(m["avg_duration"], 1) != logged_dur:
        raise ValueError(f"{run_dir.name}: computed {m} differs from run.log ({logged_thr}, {logged_dur})")
    return m


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = []

    def say(text=""):
        print(text)
        out.append(text)

    rows = []
    for c in CONTROLLERS:
        for d in sorted(args.runs.glob(f"{c}_s[0-9][0-9]")):
            rows.append({"controller": c, "seed": int(d.name[-2:]), **run_metrics(d)})
        rows.append({"controller": c, "seed": 23423, **run_metrics(args.runs / DEFAULT_RUNS[c])})
    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "seed_runs.csv", index=False)

    seeds = df[df.seed != 23423]
    n = seeds.seed.nunique()
    say(f"Runs: {n} SUMO seeds ({', '.join(str(s) for s in sorted(seeds.seed.unique()))}) per controller; all metrics match run.log")

    say("\n=== Per controller over the seeds (default seed 23423 shown separately) ===")
    for c, name in CONTROLLERS.items():
        s = seeds[seeds.controller == c]
        dflt = df[(df.controller == c) & (df.seed == 23423)].iloc[0]
        rank = int((df[df.controller == c].avg_duration < dflt.avg_duration).sum()) + 1
        say(f"{name:14s} avg duration mean {s.avg_duration.mean():6.1f} s (sd {s.avg_duration.std():5.1f}, "
            f"range {s.avg_duration.min():.1f}-{s.avg_duration.max():.1f}) | throughput mean "
            f"{s.throughput.mean():6.0f} (sd {s.throughput.std():4.0f}) | default seed: {dflt.throughput:.0f} / "
            f"{dflt.avg_duration:.1f} s (rank {rank} of {n + 1} by duration, 1 = best)")

    say(f"\n=== Paired differences over the {n} seeds (95% CI from the t distribution) ===")
    wide_d = seeds.pivot(index="seed", columns="controller", values="avg_duration")
    wide_t = seeds.pivot(index="seed", columns="controller", values="throughput")
    for a, b in [("tm", "rl"), ("tm", "fixed"), ("rl", "fixed")]:
        dd = wide_d[a] - wide_d[b]
        dt = wide_t[b] - wide_t[a]
        ci_d = T_975[n - 1] * dd.std() / math.sqrt(n)
        ci_t = T_975[n - 1] * dt.std() / math.sqrt(n)
        say(f"{CONTROLLERS[a]} - {CONTROLLERS[b]}: avg duration {dd.mean():+.1f} s (95% CI {dd.mean() - ci_d:+.1f} to "
            f"{dd.mean() + ci_d:+.1f}; {100 * dd.mean() / wide_d[a].mean():.1f}% of {CONTROLLERS[a]}), "
            f"{CONTROLLERS[b]} shorter on {(dd > 0).sum()}/{n} seeds | throughput {CONTROLLERS[b]} - {CONTROLLERS[a]}: "
            f"{dt.mean():+.0f} veh/h (95% CI {dt.mean() - ci_t:+.0f} to {dt.mean() + ci_t:+.0f}), higher on {(dt > 0).sum()}/{n}")

    say("\n=== Per seed: throughput / avg duration ===")
    table = seeds.assign(v=seeds.throughput.round().astype(int).astype(str) + " / " + seeds.avg_duration.round(1).astype(str)
                         ).pivot(index="seed", columns="controller", values="v")[list(CONTROLLERS)]
    table.columns = [CONTROLLERS[c] for c in table.columns]
    say(table.to_string())

    (args.out_dir / "seed_summary.txt").write_text("\n".join(out) + "\n")
    print(f"\nWritten to {args.out_dir}")


if __name__ == "__main__":
    main()
