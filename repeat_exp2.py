import argparse
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Repeat exp2_bnn_arch_search.py over multiple seeds and save each run in its own folder."
    )

    # repeat wrapper arguments
    parser.add_argument("--n_runs", type=int, default=5,
                        help="Number of repeated runs")
    parser.add_argument("--base_seed", type=int, default=0,
                        help="Seeds will be base_seed, base_seed+1, ...")
    parser.add_argument("--script", type=str, default="exp_2_only1.py",
                        help="Path to exp2_bnn_arch_search.py")

    # exp2_bnn_arch_search.py arguments
    parser.add_argument("--data", type=str, required=True,
                        help="Input CSV passed to exp2_bnn_arch_search.py")
    parser.add_argument("--outdir", type=str, default="exp_2_repeated",
                        help="Root output directory for repeated runs")
    parser.add_argument("--n_steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--m1", type=int, default=14)
    parser.add_argument("--m2", type=int, default=14)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--wf_values", type=float, nargs="+", default=[0.3])
    parser.add_argument("--bias", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    failed_runs = []

    for run_idx in range(args.n_runs):
        seed = args.base_seed + run_idx
        run_dir = os.path.join(args.outdir, f"run_{run_idx:03d}_seed{seed}")
        os.makedirs(run_dir, exist_ok=True)

        cmd = [
            sys.executable, args.script,
            "--data", args.data,
            "--outdir", run_dir,
            "--n_steps", str(args.n_steps),
            "--lr", str(args.lr),
            "--seed", str(seed),
            "--m1", str(args.m1),
            "--m2", str(args.m2),
            "--num_samples", str(args.num_samples),
            "--wf_values", *[str(wf) for wf in args.wf_values],
        ]

        if args.bias:
            cmd.append("--bias")

        print("\n" + "=" * 70)
        print(f"RUN {run_idx + 1}/{args.n_runs} | seed={seed} | outdir={run_dir}")
        print("=" * 70)
        print("Command:", " ".join(cmd), "\n")

        result = subprocess.run(cmd)

        if result.returncode != 0:
            failed_runs.append((run_idx, seed, result.returncode))
            print(f"[WARNING] run_idx={run_idx}, seed={seed} exited with code {result.returncode}")
        else:
            print(f"[OK] run_idx={run_idx}, seed={seed} finished successfully")

    print("\nRepeated exp2 runs finished.")
    print(f"Root output directory: {args.outdir}")

    if failed_runs:
        print("\nFailed runs:")
        for run_idx, seed, code in failed_runs:
            print(f"  run_idx={run_idx}, seed={seed}, returncode={code}")
    else:
        print("All runs finished successfully.")


if __name__ == "__main__":
    main()
