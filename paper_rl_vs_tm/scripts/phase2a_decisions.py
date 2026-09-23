"""
Phase 2 / analysis A: how the decisions of the two controllers differ.

Reads, per run, the signal switches actually displayed (outputs/tls_switches.xml.gz)
and the network (outputs/grid.net.xml.gz). For every traffic light, cycle (90 s) and
phase it computes the green seconds the phase received, labels each phase by the
movements it serves (axis NS/EW, movement type SR = straight+right, LU = left+U-turn),
and compares the two runs.

Usage (from repo root):
    python paper_rl_vs_tm/scripts/phase2a_decisions.py \
        --run-a paper_rl_vs_tm/results/runs/tm_ref --label-a TM \
        --run-b paper_rl_vs_tm/results/runs/rl_ref --label-b RL \
        --out-dir paper_rl_vs_tm/results/phase2a

Outputs (in --out-dir):
    phase_labels.csv        tls, phase, label, green links (from the network)
    decisions_long.csv      run, tls, cycle, phase, label, green_s
    junction_divergence.csv per traffic light: mean green reallocated per cycle between runs
    phase2a_output.txt      the printed tables
"""

import argparse
import gzip
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

CYCLE_S = 90
END_TIME = 7300
N_FULL_CYCLES = END_TIME // CYCLE_S  # 81; the partial last interval (7290-7300) is excluded
MIN_GREEN_S = 10
PERIOD_S = 900


def output_file(run_dir: Path, name: str):
    return gzip.open(run_dir / "outputs" / f"{name}.xml.gz", "rb")


# ---------------------------------------------------------------- network / phase labels

def compass(dx: float, dy: float) -> str:
    """Travel direction of a vector (SUMO y axis points north)."""
    if abs(dx) >= abs(dy):
        return "E" if dx > 0 else "W"
    return "N" if dy > 0 else "S"


def load_phase_labels(run_dir: Path) -> pd.DataFrame:
    """Label every phase of every traffic light by the movements that are green in it."""
    root = ET.parse(output_file(run_dir, "grid.net")).getroot()
    node_xy = {j.get("id"): (float(j.get("x")), float(j.get("y"))) for j in root.iter("junction")}
    edge_nodes = {e.get("id"): (e.get("from"), e.get("to"))
                  for e in root.iter("edge") if e.get("function") != "internal"}

    # linkIndex -> (approach travel direction, movement type) per traffic light
    links = {}
    for c in root.iter("connection"):
        tl = c.get("tl")
        if tl is None:
            continue
        frm, to = edge_nodes[c.get("from")], edge_nodes[c.get("to")]
        (fx, fy), (tx, ty) = node_xy[frm[0]], node_xy[frm[1]]
        approach = compass(tx - fx, ty - fy)
        uturn = to[1] == frm[0]
        move = "U" if uturn else {"s": "S", "r": "R", "l": "L", "t": "U"}[c.get("dir")]
        links.setdefault(tl, {})[int(c.get("linkIndex"))] = (approach, move)

    rows = []
    for tl in root.iter("tlLogic"):
        tid = tl.get("id")
        for p, phase in enumerate(tl.iter("phase")):
            green = sorted({links[tid][i] for i, ch in enumerate(phase.get("state"))
                            if ch in "Gg" and i in links[tid]})
            axes = {"NS" if a in "NS" else "EW" for a, _ in green}
            types = {"SR" if m in "SR" else "LU" for _, m in green}
            label = f"{'+'.join(sorted(axes))}-{'+'.join(sorted(types))}" if green else "none"
            rows.append({"tls": tid, "phase": p, "n_phases": sum(1 for _ in tl.iter("phase")),
                         "label": label, "green": " ".join(a + m for a, m in green)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- decisions from switches

def load_green_per_cycle(run_dir: Path, run_label: str) -> pd.DataFrame:
    """Green seconds per (tls, cycle, phase), from the switch log; intervals split at cycle bounds."""
    switches = {}
    for _, elem in ET.iterparse(output_file(run_dir, "tls_switches"), events=("end",)):
        if elem.tag == "tlsState":
            switches.setdefault(elem.get("id"), []).append((float(elem.get("time")), int(elem.get("phase"))))
            elem.clear()

    acc = {}
    for tid, sw in switches.items():
        sw.sort()
        for (t0, phase), (t1, _) in zip(sw, sw[1:] + [(float(END_TIME), None)]):
            t = t0
            while t < t1:
                cycle = int(t // CYCLE_S)
                seg_end = min(t1, (cycle + 1) * CYCLE_S)
                if cycle < N_FULL_CYCLES:
                    key = (tid, cycle, phase)
                    acc[key] = acc.get(key, 0.0) + (seg_end - t)
                t = seg_end
    df = pd.DataFrame([(run_label, k[0], k[1], k[2], v) for k, v in acc.items()],
                      columns=["run", "tls", "cycle", "phase", "green_s"])
    return df


# ---------------------------------------------------------------- reporting

def print_table(title: str, df: pd.DataFrame, out: list) -> None:
    text = f"\n=== {title} ===\n{df.to_string(index=False)}"
    print(text)
    out.append(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-a", required=True, type=Path)
    parser.add_argument("--run-b", required=True, type=Path)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    la, lb = args.label_a, args.label_b
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = []

    labels = load_phase_labels(args.run_a)
    labels.to_csv(args.out_dir / "phase_labels.csv", index=False)
    four = labels[labels.n_phases == 4]
    print_table("Phase labels of 4-phase junctions (count of junctions per phase index/label)",
                four.groupby(["phase", "label"]).size().reset_index(name="junctions"), out)
    print_table("3-phase junctions (phase labels)",
                labels[labels.n_phases != 4][["tls", "phase", "label", "green"]], out)

    dec = pd.concat([load_green_per_cycle(args.run_a, la), load_green_per_cycle(args.run_b, lb)])
    dec = dec.merge(labels[["tls", "phase", "label", "n_phases"]], on=["tls", "phase"])
    dec.to_csv(args.out_dir / "decisions_long.csv", index=False)

    # Sanity: each (run, tls, cycle) must sum to one cycle; every phase present in every cycle
    sums = dec.groupby(["run", "tls", "cycle"]).green_s.sum()
    counts = dec.groupby(["run", "tls", "cycle"]).phase.nunique()
    n_phases = dec.groupby(["run", "tls", "cycle"]).n_phases.first()
    print_table("Sanity checks", pd.DataFrame([
        {"check": "cycles per tls and run", "value": dec.groupby(["run", "tls"]).cycle.nunique().unique().tolist()},
        {"check": "cycle green sum (min, max)", "value": [sums.min(), sums.max()]},
        {"check": "(run,tls,cycle) with a phase missing", "value": int((counts < n_phases).sum())},
    ]), out)

    d4 = dec[dec.n_phases == 4]

    # A1: green per phase label
    a1 = d4.groupby(["label", "run"]).green_s.agg(
        mean="mean", median="median",
        at_min=lambda s: (s <= MIN_GREEN_S).mean() * 100,
        ge_40=lambda s: (s >= 40).mean() * 100,
        ge_60=lambda s: (s >= 60).mean() * 100).round(1).reset_index()
    a1.columns = ["label", "run", "mean green (s)", "median (s)", "% cycles at 10s min", "% cycles >= 40s", "% cycles >= 60s"]
    print_table("A1. Green per phase type, 4-phase junctions (all cycles)", a1, out)

    # A1b: axis and movement-type split of the cycle
    axis = d4.assign(axis=d4.label.str[:2], mtype=d4.label.str[-2:])
    a1b = pd.concat([
        axis.groupby(["run", "axis"]).green_s.sum().groupby(level=0).transform(lambda s: s / s.sum() * 100).rename("share %").reset_index().rename(columns={"axis": "group"}),
        axis.groupby(["run", "mtype"]).green_s.sum().groupby(level=0).transform(lambda s: s / s.sum() * 100).rename("share %").reset_index().rename(columns={"mtype": "group"}),
    ]).round(1)
    print_table("A1b. Share of green time by axis and by movement type (4-phase junctions)", a1b, out)

    # A2: cycle-to-cycle change of each phase's green
    d4s = d4.sort_values(["run", "tls", "phase", "cycle"])
    d4s["change"] = d4s.groupby(["run", "tls", "phase"]).green_s.diff().abs()
    a2 = d4s.groupby("run").change.agg(
        mean="mean", median="median", p95=lambda s: s.quantile(0.95), max="max",
        zero=lambda s: (s == 0).mean() * 100, ge_20=lambda s: (s >= 20).mean() * 100).round(1).reset_index()
    a2.columns = ["run", "mean |change| (s)", "median", "p95", "max", "% unchanged", "% change >= 20s"]
    print_table("A2. Cycle-to-cycle change of a phase's green (4-phase junctions)", a2, out)

    # A3: divergence between runs, per junction: green reallocated per cycle = sum_p |a - b| / 2
    wide = dec.pivot_table(index=["tls", "cycle", "phase", "label", "n_phases"], columns="run", values="green_s").reset_index()
    wide["absdiff"] = (wide[lb] - wide[la]).abs()
    per_cycle = wide.groupby(["tls", "cycle"]).absdiff.sum().div(2).rename("realloc_s").reset_index()
    per_junction = per_cycle.groupby("tls").realloc_s.mean().round(1).rename("mean green reallocated per cycle (s)")
    more = wide.assign(diff=wide[lb] - wide[la]).groupby(["tls", "label"])["diff"].mean().round(1).unstack()
    jd = pd.concat([per_junction, more], axis=1).reset_index().rename(columns={"index": "tls"})
    jd = jd.sort_values("mean green reallocated per cycle (s)", ascending=False)
    jd.to_csv(args.out_dir / "junction_divergence.csv", index=False)
    print_table(f"A3. Per junction: mean green reallocated per cycle, and mean ({lb} - {la}) green per phase label (s)", jd, out)

    # A3 on the grid (rows y=5..0, columns A..F)
    grid = {}
    for tid, v in per_junction.items():
        grid[(int(tid[1:]), tid[0])] = v
    cols = sorted({c for _, c in grid})
    lines = ["row " + "".join(f"{c:>7}" for c in cols)]
    for r in sorted({r for r, _ in grid}, reverse=True):
        lines.append(f"{r:>3} " + "".join(f"{grid[(r, c)]:>7.1f}" if (r, c) in grid else f"{'-':>7}" for c in cols))
    text = "\n=== A3 map: mean green reallocated per cycle (s); '-' = removed junction ===\n" + "\n".join(lines)
    print(text)
    out.append(text)

    # A4: divergence over time
    per_cycle["period"] = (per_cycle.cycle * CYCLE_S // PERIOD_S) * PERIOD_S
    a4 = per_cycle.groupby("period").realloc_s.mean().round(1).reset_index()
    a4["period"] = a4.period.map(lambda p: f"{p}-{min(p + PERIOD_S, N_FULL_CYCLES * CYCLE_S)}")
    a4.columns = ["period (s)", "mean green reallocated per cycle (s), all junctions"]
    print_table("A4. Divergence between runs over time", a4, out)

    (args.out_dir / "phase2a_output.txt").write_text("\n".join(out) + "\n")
    print(f"\nWritten to {args.out_dir}")


if __name__ == "__main__":
    main()
