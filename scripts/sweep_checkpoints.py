"""Evaluate every checkpoint of a training run under one protocol and print a table.

Example:
  python scripts/sweep_checkpoints.py --run_dir runs/see2act_bin-picking/<ts> --n_episodes 50 \
      --protocol single --out_dir results
  python scripts/sweep_checkpoints.py --run_dir ... --protocol nr10 --checkpoints epoch_8000.pth,epoch_10000.pth
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROTOCOLS = {
    "single": [],
    "nr10": ["--n_rollout", "10", "--select_by", "final_action_variance"],
    "nr10_camvar": ["--n_rollout", "10", "--select_by", "camera_variance"],
    "nr10_entropy": ["--n_rollout", "10", "--select_by", "action_entropy"],
    "nr10_action_var": ["--n_rollout", "10", "--select_by", "action_variance"],
    "steps20": ["--num_steps", "20"],
    "steps50": ["--num_steps", "50"],
}


def epoch_of(path):
    m = re.search(r"epoch_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def main(args):
    ckpts = sorted(glob.glob(os.path.join(args.run_dir, "checkpoints", "*.pth")), key=epoch_of)
    if args.checkpoints:
        wanted = set(args.checkpoints.split(","))
        ckpts = [c for c in ckpts if os.path.basename(c) in wanted]
    extra = PROTOCOLS[args.protocol] + (args.extra.split() if args.extra else [])
    rows = []
    for ck in ckpts:
        tag = os.path.basename(ck).replace(".pth", "")
        run_name = f"{args.prefix}{args.protocol}_{tag}"
        cmd = [sys.executable, os.path.join(HERE, "evaluate.py"), "--checkpoint", ck, "--run_name", run_name,
               "--out_dir", args.out_dir, "--n_episodes", str(args.n_episodes), "--renderer_workers", str(args.renderer_workers)] + extra
        if args.task:
            cmd += ["--task", args.task]
        print("[sweep] " + " ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=False)
        task = args.task
        if task is None:
            import torch
            task = torch.load(ck, map_location="cpu").get("task")
        summary_path = os.path.join(args.out_dir, task, run_name, "summary.json")
        if os.path.exists(summary_path):
            s = json.load(open(summary_path))
            rows.append((tag, s["n"], s["success_rate"], s["success_ci95"], s["failure_breakdown"]))
    print("\n| checkpoint | n | success | 95% CI | failures |\n|---|---|---|---|---|")
    for tag, n, sr, ci, fb in rows:
        print(f"| {tag} | {n} | {100 * sr:.0f}% | [{100 * ci[0]:.0f}, {100 * ci[1]:.0f}] | {fb} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--task", default=None)
    ap.add_argument("--protocol", default="single", choices=sorted(PROTOCOLS))
    ap.add_argument("--checkpoints", default=None, help="comma-separated checkpoint file names (default: all)")
    ap.add_argument("--n_episodes", type=int, default=50)
    ap.add_argument("--renderer_workers", type=int, default=1)
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--extra", default="", help="extra arguments passed to evaluate.py")
    ap.add_argument("--dry_run", action="store_true")
    main(ap.parse_args())
