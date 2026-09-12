"""Evaluate a See2Act checkpoint on a task with fixed per-episode seeds.

Examples:
  python scripts/evaluate.py --checkpoint runs/see2act_bin-picking/<ts>/checkpoints/best_valid_epoch_9919.pth \
      --n_episodes 50 --run_name single
  python scripts/evaluate.py --checkpoint ... --n_rollout 10 --select_by camera_variance --run_name nr10

Writes <out_dir>/<task>/<run_name>/{episode_<seed>.json, summary.json}. Existing episodes are reused.
Episodes are seeded (scene, initial action noise and refinement noise all derive from the episode seed) and CUDA
kernels run deterministically, so repeated evaluations of a checkpoint give identical results on the same GPU.
"""
import argparse
import json
import multiprocessing as mp
import os
import signal
import sys
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # needed for deterministic CUDA matmuls (set before torch)
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from see2act import evaluation as ev  # noqa: E402
from see2act.see2act import See2Act  # noqa: E402
from see2act.renderer import Renderer  # noqa: E402


class EpisodeTimeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise EpisodeTimeout()


def set_deterministic():
    """Bit-reproducible GPU kernels: with the per-episode seeds this makes an evaluation repeatable."""
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def main(args):
    if not args.nondeterministic:
        set_deterministic()
    policy = See2Act.load(args.checkpoint, device="cuda", overrides=args.set)
    task = args.task or policy.checkpoint.get("task") or policy.config.data.get("task")
    assert task, "task unknown: pass --task"
    out_dir = os.path.join(args.out_dir, task, args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    settings = dict(vars(args), task=task, config=policy.config.to_dict())
    with open(os.path.join(out_dir, "settings.json"), "w") as f:
        json.dump(settings, f, indent=2)
    print(f"[eval] task={task} checkpoint={args.checkpoint}\n[eval] n_rollout={args.n_rollout} select_by={args.select_by} "
          f"steps={args.num_steps or policy.config.diffusion.num_refinement_steps} init_view={args.init_view or policy.schedule.init_view}", flush=True)

    mp.set_start_method("spawn", force=True)
    renderer = Renderer(task, num_workers=args.renderer_workers).start()
    env = ev.make_env(task)
    seeds = [args.seed_start + i * args.seed_stride for i in range(args.n_episodes)]
    if args.episode_timeout > 0:
        signal.signal(signal.SIGALRM, _on_alarm)
    episodes = []
    try:
        for i, seed in enumerate(seeds):
            path = os.path.join(out_dir, f"episode_{seed}.json")
            if os.path.exists(path) and not args.overwrite:
                episodes.append(json.load(open(path)))
                continue
            t0 = time.time()
            try:
                if args.episode_timeout > 0:
                    signal.alarm(int(args.episode_timeout))
                res = ev.run_episode(env, policy, renderer, seed, init_view=args.init_view, n_rollout=args.n_rollout,
                                     select_by=args.select_by, num_steps=args.num_steps, keep_trace=not args.no_trace)
                signal.alarm(0)
            except EpisodeTimeout:
                signal.alarm(0)
                print(f"[eval] episode {seed} exceeded {args.episode_timeout}s: counted as a failure", flush=True)
                res = dict(seed=int(seed), success=False, failure_category="timeout")
            except Exception as e:
                signal.alarm(0)
                traceback.print_exc()
                res = dict(seed=int(seed), success=False, failure_category="exception", error=str(e))
            res["wall_clock_s"] = time.time() - t0
            with open(path, "w") as f:
                json.dump(res, f)
            episodes.append(res)
            n_ok = sum(e["success"] for e in episodes)
            print(f"[eval] {task}/{args.run_name} episode {i + 1}/{len(seeds)} seed={seed} success={res['success']} "
                  f"({res['failure_category']}, {res['wall_clock_s']:.1f}s)  running rate {n_ok}/{len(episodes)}", flush=True)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        summary = ev.summarize(episodes)
        summary.update(dict(task=task, run_name=args.run_name, checkpoint=os.path.abspath(args.checkpoint),
                            n_rollout=args.n_rollout, select_by=args.select_by, num_steps=args.num_steps,
                            init_view=args.init_view))
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2), flush=True)
        renderer.stop()
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task", default=None, help="defaults to the task stored in the checkpoint")
    ap.add_argument("--run_name", default="default")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--n_episodes", type=int, default=50)
    ap.add_argument("--seed_start", type=int, default=200001,
                    help="first episode seed; the default scenes are the ones reported in the README (checkpoints were "
                         "selected on seeds 100001, 100003, ...)")
    ap.add_argument("--seed_stride", type=int, default=2)
    ap.add_argument("--n_rollout", type=int, default=None, help="refinement runs per keyframe (default: config)")
    ap.add_argument("--select_by", default=None, choices=[None, "none", "camera_variance", "final_camera_variance", "final_action_variance",
                             "action_variance", "action_entropy"])
    ap.add_argument("--num_steps", type=int, default=None, help="refinement steps T (default: config)")
    ap.add_argument("--init_view", default=None, help="initial view C_T (default: config)")
    ap.add_argument("--renderer_workers", type=int, default=1, help="render worker processes; refinement runs are sequential, so one worker is enough and keeps results bit-reproducible")
    ap.add_argument("--episode_timeout", type=float, default=600.0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no_trace", action="store_true", help="do not store camera/action trajectories")
    ap.add_argument("--nondeterministic", action="store_true", help="allow non-deterministic (faster) CUDA kernels")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override for inference")
    main(ap.parse_args())
