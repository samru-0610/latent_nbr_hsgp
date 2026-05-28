"""
Runs exp1_script.py N times as a subprocess, each with a different seed,
then aggregates the silhouette scores to report mean ± std.

Usage:
    python repeat_exp_script.py                      # default: 5 runs
    python repeat_exp_script.py --n_runs 10
    python repeat_exp_script.py --n_runs 10 --base_seed 42
    python repeat_exp_script.py --script path/to/exp1_script.py
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--n_runs",    type=int, default=5,             help="Number of repeated runs")
parser.add_argument("--base_seed", type=int, default=0,             help="Seeds will be base_seed, base_seed+1, …")
parser.add_argument("--script",    type=str, default="exp1_script.py", help="Path to the experiment script")
parser.add_argument("--outdir",    type=str, default="exp_1_repeated",  help="Root output directory")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

all_scores = {}   # {run_idx: scores_df}

for run_idx in range(args.n_runs):
    seed    = args.base_seed + run_idx
    run_dir = os.path.join(args.outdir, f"run_{run_idx:03d}_seed{seed}")

    print(f"\n{'='*60}")
    print(f"  RUN {run_idx + 1}/{args.n_runs}   seed={seed}   outdir={run_dir}")
    print(f"{'='*60}\n")

    result = subprocess.run(
        [sys.executable, args.script],
        env={
            **os.environ,
            "EXP_SEED":   str(seed),
            "EXP_OUTDIR": run_dir,
        },
    )

    if result.returncode != 0:
        print(f"[WARNING] Run {run_idx} exited with code {result.returncode}")

    scores_csv = os.path.join(run_dir, "all_scores.csv")
    if os.path.exists(scores_csv):
        df = pd.read_csv(scores_csv)
        all_scores[run_idx] = df
    else:
        print(f"[WARNING] {scores_csv} not found — run may have failed.")


# ── Aggregate ─────────────────────────────────────────────────────────────────

if not all_scores:
    print("No results collected. Exiting.")
    sys.exit(1)

combined = pd.concat(all_scores.values(), keys=all_scores.keys())
combined.index.names = ["run_idx", "row"]

agg = (
    combined
    .groupby("setting")["silhouette_score"]
    .agg(["mean", "std", "min", "max", "count"])
    .reset_index()
    .rename(columns={"count": "n_runs"})
)

print("\n" + "="*60)
print("  AGGREGATED SILHOUETTE SCORES")
print("="*60)
print(agg.to_string(index=False))

agg.to_csv(os.path.join(args.outdir, "aggregated_silhouette.csv"), index=False)
combined.reset_index().to_csv(os.path.join(args.outdir, "all_runs_silhouette.csv"), index=False)


# ── Plot ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 4))
names = agg["setting"].tolist()
x     = np.arange(len(names))

ax.bar(x, agg["mean"], yerr=agg["std"], capsize=6,
       color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"],
       alpha=0.85, error_kw=dict(lw=1.5))

# Overlay individual run dots
for i, name in enumerate(names):
    vals = combined.xs(name, level="setting")["silhouette_score"].values
    ax.scatter([i] * len(vals), vals, color="black", s=25, zorder=5, alpha=0.6)

ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylabel("Silhouette Score")
ax.set_title(f"Silhouette Score: mean ± std  (n={args.n_runs} runs)")
ax.set_ylim(bottom=0)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(args.outdir, "silhouette_errorbars.png"), dpi=150)
plt.close(fig)

print(f"\nDone. Results saved to: {args.outdir}")

