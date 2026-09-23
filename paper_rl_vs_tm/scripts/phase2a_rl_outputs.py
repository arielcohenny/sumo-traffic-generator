"""
Phase 2 / analysis A (part 2): the RL policy's 136 outputs over the run.

Reads outputs/rl_actions.csv.gz of an RL run (one row per decision, every 90 s; raw policy
outputs and the clipped values actually used, bounds [-10, 10]) and reports how many
outputs are clipped at the lower bound, how far below it they are, and how much the
outputs change over the run.

Usage (from repo root):
    python paper_rl_vs_tm/scripts/phase2a_rl_outputs.py \
        --run paper_rl_vs_tm/results/runs/rl_ref --out-dir paper_rl_vs_tm/results/phase2a
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LOW, HIGH = -10.0, 10.0
SETTLE_DECISIONS = 3  # decisions at t = 0, 90, 180 are excluded from the "settled" statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    a = pd.read_csv(args.run / "outputs" / "rl_actions.csv.gz")
    raw = a[[c for c in a if c.startswith("raw_")]]
    raw.columns = [c[len("raw_"):] for c in raw.columns]
    clipped = a[[c for c in a if c.startswith("clipped_")]]
    out = []

    def say(text=""):
        print(text)
        out.append(text)

    say(f"decisions: {len(a)} (t = {int(a.time.iloc[0])} .. {int(a.time.iloc[-1])}), outputs per decision: {raw.shape[1]}")
    say(f"check clip(raw) == clipped action used: "
        f"{np.array_equal(np.clip(raw.values, LOW, HIGH).astype(np.float32), clipped.values.astype(np.float32))}")

    below = (raw < LOW).sum(axis=1)
    say("\n=== Outputs below the lower bound (-10) per decision ===")
    say(f"min {below.min()}, median {int(below.median())}, max {below.max()} of {raw.shape[1]}; "
        f"above the upper bound (+10) in any decision: {int((raw > HIGH).values.sum())}")
    ts = pd.Series(below.values, index=a.time.astype(int))
    say(ts.iloc[[0, 1, 2, 3, 4, 5, 10, 20, 40, 60, len(ts) - 1]].to_string())

    say("\n=== Raw output values (all decisions) ===")
    say("min %.1f, p5 %.1f, median %.1f, p95 %.1f, max %.1f" % tuple(np.percentile(raw.values, [0, 5, 50, 95, 100])))

    frac = (raw < LOW).mean()
    say("\n=== Per output: share of decisions clipped at -10 ===")
    say(f"always clipped: {(frac == 1).sum()}, never clipped: {(frac == 0).sum()}, "
        f"sometimes: {((frac > 0) & (frac < 1)).sum()}")
    always = frac[frac == 1].index
    say(f"always-clipped outputs, highest value ever reached: median {raw[always].max().median():.1f}, "
        f"max {raw[always].max().max():.1f}")

    settled = raw.iloc[SETTLE_DECISIONS:]
    changes = ((settled < LOW).any() & (settled >= LOW).any()).sum()
    say(f"\n=== From decision {SETTLE_DECISIONS} on (t >= {int(a.time.iloc[SETTLE_DECISIONS])}) ===")
    say(f"clipped outputs per decision: min {(settled < LOW).sum(axis=1).min()}, max {(settled < LOW).sum(axis=1).max()}")
    say(f"outputs whose clipped/unclipped status ever changes: {changes}")
    say(f"std of each output over time: median {settled.std().median():.2f}, "
        f"p90 {settled.std().quantile(0.9):.2f}, max {settled.std().max():.2f}")

    never = frac[frac == 0].index
    say("\n=== Outputs never clipped (tls_phase: median raw value, std over time from decision "
        f"{SETTLE_DECISIONS} on) ===")
    for n in never:
        say(f"{n}: median {raw[n].median():.1f}, std {settled[n].std():.2f}")

    per_output = pd.DataFrame({"output": raw.columns, "share_clipped": frac.values.round(3),
                               "min": raw.min().values.round(2), "median": raw.median().values.round(2),
                               "max": raw.max().values.round(2), "std_settled": settled.std().values.round(3)})
    per_output.to_csv(args.out_dir / "rl_outputs_per_output.csv", index=False)
    (args.out_dir / "rl_outputs_output.txt").write_text("\n".join(out) + "\n")
    print(f"\nWritten to {args.out_dir}")


if __name__ == "__main__":
    main()
