"""
Phase 2 / analysis D: which junctions carry the time vehicles save (or lose) under RL's plan.

For each SUMO seed (01..10 and the default seed) it compares Tree Method with RL's fixed plan
and with RL, using only vehicles that arrived in both runs of the pair. From
outputs/vehroutes.xml.gz (route with the exit time of every edge) each vehicle's trip time
splits exactly into the time spent on each edge; every edge is assigned to the junction it
leads to (tail and head edges of a road, see netmap.py) and to the travel axis (NS/EW) of the
road; the last edge of a trip (the vehicle leaves the network on it, before its junction) is
assigned to "trip end"; time of a trip that ends by teleport and is on no edge is assigned to
"teleport". So, per vehicle,

    duration = sum over junction approaches of time on them + time on the last edge (+ teleport)

and the difference in total duration between two runs splits exactly by junction and axis.

Usage (from repo root):
    python paper_rl_vs_tm/scripts/phase2d_attribution.py \
        --results paper_rl_vs_tm/results --out-dir paper_rl_vs_tm/results/phase2d
"""

import argparse
import gzip
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import pandas as pd

from netmap import load_edges
from phase2b_congestion import plan_types, runs_of, CONTROLLERS

TRIP_END = "trip end"
TELEPORT = "teleport (trip ended by teleport)"
PAIRS = [("tm", "fixed"), ("tm", "rl")]


def load_vehicle_times(run_dir: Path, edge_junction: dict, edge_axis: dict) -> dict:
    """{vehicle id: {(junction, axis): seconds}} for every arrived vehicle; checks the split is exact."""
    out = {}
    with gzip.open(run_dir / "outputs" / "vehroutes.xml.gz") as f:
        for _, elem in ET.iterparse(f, events=("end",)):
            if elem.tag != "vehicle":
                continue
            # The driven route is the one with exit times; rerouted vehicles also list replaced routes
            driven = [r for r in elem.iter("route") if r.get("exitTimes") is not None]
            if len(driven) != 1:
                raise ValueError(f"{run_dir.name} {elem.get('id')}: {len(driven)} routes with exit times")
            route = driven[0]
            edges = route.get("edges").split()
            exits = [float(x) for x in route.get("exitTimes").split()]
            depart, arrival = float(elem.get("depart")), float(elem.get("arrival"))
            times = defaultdict(float)
            prev = depart
            for i, (e, t) in enumerate(zip(edges, exits)):
                key = (TRIP_END, "-") if i == len(edges) - 1 else (edge_junction[e], edge_axis[e])
                times[key] += t - prev
                prev = t
            # A trip that ends by teleport (vaporized) has time not spent on any edge; keep it as its own bucket
            residual = (arrival - depart) - sum(times.values())
            if abs(residual) > 1e-6:
                times[(TELEPORT, "-")] += residual
            out[elem.get("id")] = dict(times)
            elem.clear()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = []

    def say(text=""):
        print(text)
        out.append(text)

    runs_dir = args.results / "runs"
    edges = load_edges(runs_dir / "tm_ref")
    edge_junction = dict(zip(edges.edge, edges.approach_junction))
    edge_axis = dict(zip(edges.edge, edges.axis))
    ptype = plan_types(args.results)

    runs = {c: runs_of(runs_dir, c) for c in CONTROLLERS}
    seeds = list(runs["tm"])
    rows, summary = [], []
    for seed in seeds:
        vt = {c: load_vehicle_times(runs[c][seed], edge_junction, edge_axis) for c in CONTROLLERS}
        print(f"  read seed {seed}")
        for a, b in PAIRS:
            matched = set(vt[a]) & set(vt[b])
            diff = defaultdict(float)
            for v in matched:
                for k, s in vt[a][v].items():
                    diff[k] += s
                for k, s in vt[b][v].items():
                    diff[k] -= s
            total = sum(diff.values())
            direct = sum(sum(vt[a][v].values()) - sum(vt[b][v].values()) for v in matched)
            assert abs(total - direct) < 1e-3, "junction split does not add up to the duration difference"
            summary.append({"pair": f"{a}-{b}", "seed": seed, "matched": len(matched),
                            "total_vh": total / 3600, "per_vehicle_s": total / len(matched)})
            for (j, axis), s in diff.items():
                rows.append({"pair": f"{a}-{b}", "seed": seed, "junction": j, "axis": axis, "delta_vh": s / 3600})
    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "attribution_per_junction_axis_seed.csv", index=False)
    sm = pd.DataFrame(summary)
    n = len(seeds)

    say(f"Seeds: {n} (01-10 and default). Matched vehicles = arrived in both runs of the pair. "
        f"delta = time in the first run - time in the second (positive = saved by the second).")
    say("\n=== D1. Total time saved by matched vehicles (veh-h), per pair ===")
    for p, g in sm.groupby("pair", sort=False):
        say(f"{p}: mean {g.total_vh.mean():.0f} veh-h ({g.per_vehicle_s.mean():.1f} s per matched vehicle, "
            f"{g.matched.mean():.0f} matched vehicles), positive in {(g.total_vh > 0).sum()}/{n} seeds")

    for p in [f"{a}-{b}" for a, b in PAIRS]:
        d = df[df.pair == p]
        pj = d.groupby(["junction", "seed"]).delta_vh.sum().unstack(fill_value=0)
        pa = d.groupby(["junction", "axis", "seed"]).delta_vh.sum().unstack(fill_value=0).mean(axis=1).unstack(fill_value=0)
        t = pd.DataFrame({"rl_plan_type": [ptype.get(j, "") for j in pj.index], "saved_vh": pj.mean(axis=1),
                          "positive_seeds": (pj > 0).sum(axis=1), "saved_NS_vh": pa.get("NS", 0), "saved_EW_vh": pa.get("EW", 0)},
                         index=pj.index)
        t["share_%"] = 100 * t.saved_vh / t.saved_vh.sum()
        t = t.sort_values("saved_vh", ascending=False)
        t.to_csv(args.out_dir / f"attribution_per_junction_{p}.csv")
        say(f"\n=== D2 ({p}). Time saved per junction approach (veh-h, mean over seeds), split by approach axis ===")
        say(t.round(1).to_string())

        g = t[~t.index.isin([TRIP_END, TELEPORT])].groupby("rl_plan_type").agg(junctions=("saved_vh", "size"), saved_vh=("saved_vh", "sum"),
                                                                saved_NS_vh=("saved_NS_vh", "sum"), saved_EW_vh=("saved_EW_vh", "sum"))
        g["share_%"] = 100 * g.saved_vh / t.saved_vh.sum()
        say(f"\n=== D3 ({p}). By RL plan type ===")
        say(g.round(1).to_string())

    (args.out_dir / "phase2d_output.txt").write_text("\n".join(out) + "\n")
    print(f"\nWritten to {args.out_dir}")


if __name__ == "__main__":
    main()
