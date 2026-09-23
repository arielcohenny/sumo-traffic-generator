"""
Step 0: decompose the average-trip-duration gap between two runs of the same scenario.

Reads only SUMO outputs (tripinfo.xml, summary.xml, sumo_statistics.xml) from two
run directories, e.g. tm_a (Tree Method) and rl_a (RL checkpoint 1945600).

SUMO's "Average duration" is the mean over ARRIVED vehicles only. Both runs use the
same demand (same vehicle IDs, OD pairs, scheduled departures), so the gap splits
exactly into:

    mean_A(all A arrivals) - mean_B(all B arrivals)
        = [mean_A(M) - mean_B(M)]                                   speed-up on matched vehicles
        + [(mean_A(all) - mean_A(M)) - (mean_B(all) - mean_B(M))]   composition effect

where M = vehicles that arrived in both runs.

Usage (from repo root):
    python evaluation/decision_comparison/step0_decompose.py \
        --run-a evaluation/decision_comparison/tm_a --label-a TM \
        --run-b evaluation/decision_comparison/rl_a --label-b RL
"""

import argparse
import csv
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

TRIP_FIELDS = ["depart", "departDelay", "arrival", "duration", "routeLength",
               "waitingTime", "timeLoss", "rerouteNo"]
BIN_SECONDS = 900
END_TIME = 7300


def load_tripinfo(run_dir: Path) -> dict:
    """Return {vehicle_id: {field: float}} for every arrived vehicle."""
    trips = {}
    for _, elem in ET.iterparse(run_dir / "workspace" / "tripinfo.xml", events=("end",)):
        if elem.tag == "tripinfo":
            rec = {f: float(elem.get(f)) for f in TRIP_FIELDS}
            # Scheduled departure is identical across runs; actual depart includes insertion delay
            rec["scheduled"] = rec["depart"] - rec["departDelay"]
            trips[elem.get("id")] = rec
            elem.clear()
    return trips


def load_summary(run_dir: Path) -> dict:
    """Return {time: {running, waiting, arrived}} from summary.xml."""
    steps = {}
    for _, elem in ET.iterparse(run_dir / "workspace" / "summary.xml", events=("end",)):
        if elem.tag == "step":
            steps[int(float(elem.get("time")))] = {
                k: int(elem.get(k)) for k in ("running", "waiting", "arrived")}
            elem.clear()
    return steps


def load_statistics(run_dir: Path) -> dict:
    root = ET.parse(run_dir / "workspace" / "sumo_statistics.xml").getroot()
    veh = root.find("vehicles")
    tel = root.find("teleports")
    saf = root.find("safety")
    trip = root.find("vehicleTripStatistics")
    return {
        "loaded": int(veh.get("loaded")),
        "inserted": int(veh.get("inserted")),
        "running_at_end": int(veh.get("running")),
        "never_inserted": int(veh.get("waiting")),
        "arrived": int(trip.get("count")),
        "avg_duration": float(trip.get("duration")),
        "avg_departDelay": float(trip.get("departDelay")),
        "teleports": int(tel.get("total")),
        "teleports_jam": int(tel.get("jam")),
        "teleports_yield": int(tel.get("yield")),
        "teleports_wrongLane": int(tel.get("wrongLane")),
        "collisions": int(saf.get("collisions")),
    }


def mean_of(trips: dict, ids, field: str) -> float:
    return float(np.mean([trips[i][field] for i in ids])) if ids else float("nan")


def print_table(title: str, header: list, rows: list) -> None:
    print(f"\n=== {title} ===")
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    fmt = "  ".join(f"{{:>{w}}}" for w in widths)
    print(fmt.format(*header))
    for r in rows:
        print(fmt.format(*r))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-a", required=True, type=Path)
    parser.add_argument("--run-b", required=True, type=Path)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "step0_output")
    args = parser.parse_args()
    la, lb = args.label_a, args.label_b

    A, B = load_tripinfo(args.run_a), load_tripinfo(args.run_b)
    stats_a, stats_b = load_statistics(args.run_a), load_statistics(args.run_b)
    both = sorted(set(A) & set(B))
    only_a = sorted(set(A) - set(B))
    only_b = sorted(set(B) - set(A))

    # --- Run-level statistics ---
    print_table("Run-level statistics (sumo_statistics.xml)", ["metric", la, lb],
                [[k, stats_a[k], stats_b[k]] for k in stats_a])

    # --- Decomposition ---
    mean_a_all, mean_b_all = mean_of(A, A.keys(), "duration"), mean_of(B, B.keys(), "duration")
    mean_a_m, mean_b_m = mean_of(A, both, "duration"), mean_of(B, both, "duration")
    gap = mean_a_all - mean_b_all
    speedup = mean_a_m - mean_b_m
    composition = (mean_a_all - mean_a_m) - (mean_b_all - mean_b_m)

    def share(x):
        return f"{100 * x / gap:.1f}%" if gap != 0 else "n/a"

    print_table(f"Average-duration gap decomposition ({la} - {lb}, seconds)", ["term", "value", "share"], [
        [f"mean duration {la} (all arrivals)", f"{mean_a_all:.1f}", ""],
        [f"mean duration {lb} (all arrivals)", f"{mean_b_all:.1f}", ""],
        ["total gap", f"{gap:.1f}", "100%"],
        ["speed-up on matched vehicles", f"{speedup:.1f}", share(speedup)],
        ["composition effect", f"{composition:.1f}", share(composition)],
        ["check: speed-up + composition - gap", f"{speedup + composition - gap:.2e}", ""],
    ])

    # --- Arrival sets ---
    set_rows = []
    for name, ids, src in [("arrived in both", both, A), (f"only {la}", only_a, A), (f"only {lb}", only_b, B)]:
        set_rows.append([name, len(ids),
                         f"{mean_of(src, ids, 'scheduled'):.0f}",
                         f"{mean_of(src, ids, 'routeLength'):.0f}",
                         f"{mean_of(src, ids, 'duration'):.1f}"])
    print_table("Arrival sets (depart/route/duration taken from the run where the vehicle arrived)",
                ["set", "count", "mean sched. depart (s)", "mean routeLength (m)", "mean duration (s)"], set_rows)

    # --- Matched vehicles: per-field means ---
    fields = ["duration", "waitingTime", "timeLoss", "routeLength", "departDelay", "rerouteNo"]
    print_table(f"Matched vehicles (n={len(both)}): means", ["field", la, lb, f"{la} - {lb}"],
                [[f, f"{mean_of(A, both, f):.1f}", f"{mean_of(B, both, f):.1f}",
                  f"{mean_of(A, both, f) - mean_of(B, both, f):.1f}"] for f in fields])

    # --- Matched vehicles: distribution of per-vehicle delta (positive = faster under B) ---
    delta = np.array([A[i]["duration"] - B[i]["duration"] for i in both])
    pct = np.percentile(delta, [5, 25, 50, 75, 95])
    print_table(f"Per-vehicle duration delta ({la} - {lb}; positive = faster under {lb})", ["stat", "value"], [
        [f"faster under {lb} (delta >= 1s)", f"{np.mean(delta >= 1) * 100:.1f}%"],
        [f"slower under {lb} (delta <= -1s)", f"{np.mean(delta <= -1) * 100:.1f}%"],
        ["within 1s", f"{np.mean(np.abs(delta) < 1) * 100:.1f}%"],
        ["p5 / p25 / p50 / p75 / p95", " / ".join(f"{p:.0f}" for p in pct)],
        ["sum of positive deltas (s)", f"{delta[delta > 0].sum():.0f}"],
        ["sum of negative deltas (s)", f"{delta[delta < 0].sum():.0f}"],
    ])

    # --- Matched vehicles by scheduled-departure bin ---
    bin_rows = []
    for lo in range(0, END_TIME, BIN_SECONDS):
        ids = [i for i in both if lo <= A[i]["scheduled"] < lo + BIN_SECONDS]
        n_a = sum(1 for i in A if lo <= A[i]["scheduled"] < lo + BIN_SECONDS)
        n_b = sum(1 for i in B if lo <= B[i]["scheduled"] < lo + BIN_SECONDS)
        d = np.array([A[i]["duration"] - B[i]["duration"] for i in ids]) if ids else np.array([np.nan])
        bin_rows.append([f"{lo}-{min(lo + BIN_SECONDS, END_TIME)}", n_a, n_b, len(ids),
                         f"{mean_of(A, ids, 'duration'):.1f}", f"{mean_of(B, ids, 'duration'):.1f}",
                         f"{np.nanmean(d):.1f}", f"{d.sum() if ids else 0:.0f}"])
    print_table("By scheduled departure bin",
                ["bin (s)", f"arrived {la}", f"arrived {lb}", "matched",
                 f"matched dur {la}", f"matched dur {lb}", "mean delta", "total delta (s)"], bin_rows)

    # --- Network state over time (summary.xml) ---
    sa, sb = load_summary(args.run_a), load_summary(args.run_b)
    ts = list(range(BIN_SECONDS, END_TIME, BIN_SECONDS)) + [max(sa)]
    print_table("Network state over time (summary.xml)",
                ["time", f"running {la}", f"running {lb}", f"waiting-to-insert {la}",
                 f"waiting-to-insert {lb}", f"arrived {la}", f"arrived {lb}"],
                [[t, sa[t]["running"], sb[t]["running"], sa[t]["waiting"], sb[t]["waiting"],
                  sa[t]["arrived"], sb[t]["arrived"]] for t in ts if t in sa and t in sb])

    # --- Per-vehicle CSV for later steps ---
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / f"matched_{la}_vs_{lb}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "scheduled"] + [f"{x}_{la}" for x in fields] + [f"{x}_{lb}" for x in fields])
        for i in both:
            w.writerow([i, A[i]["scheduled"]] + [A[i][x] for x in fields] + [B[i][x] for x in fields])
    print(f"\nPer-vehicle matched data written to {out_csv}")


if __name__ == "__main__":
    main()
