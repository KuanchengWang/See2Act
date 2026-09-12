"""Collect scripted keyframe demonstrations for a task and write them to an hdf5 file.

Example:
  python scripts/collect_demos.py --task bin-picking --n_train 100 --n_valid 20 --out data/bin-picking.hdf5

Train episodes use even seeds and validation episodes odd seeds (as in Ravens). Only successful
demonstrations are kept; the seed of every stored demonstration is saved with it.
"""
import argparse
import json
import os
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from see2act.sim import ASSETS_ROOT, cameras, tasks  # noqa: E402
from see2act.sim.environment import Environment, vector_from_action  # noqa: E402
from see2act.sim.scene import encode_scene  # noqa: E402
from see2act.utils import set_seed  # noqa: E402


def collect_split(env, task, n_episodes, first_seed, split, f, demo_counter):
    seed = first_seed
    n_saved = 0
    names = []
    while n_saved < n_episodes:
        seed += 2
        set_seed(seed)
        env.set_task(task)
        obs = env.reset()
        scene = task.scene
        action = task.oracle_action(env)
        if action is None:
            continue
        _, reward, done, _ = env.step(action)
        if reward < 0.99:
            print(f"[{split}] seed {seed}: demonstration failed (reward {reward:.2f}), skipped", flush=True)
            continue
        name = f"demo_{demo_counter[0]}"
        demo_counter[0] += 1
        g = f["data"].create_group(name)
        g.attrs["seed"] = seed
        g.attrs["reward"] = float(reward)
        g.attrs["num_samples"] = 1
        g.create_dataset("actions", data=vector_from_action(action)[None].astype(np.float64))
        g.create_dataset("scene", data=encode_scene(scene))
        for cam_name, color in zip(cameras.AgentCameras.NAMES, obs["color"]):
            g.create_dataset(f"obs/rgb_{cam_name}", data=np.asarray(color, dtype=np.uint8)[None], compression="gzip")
        names.append(name)
        n_saved += 1
        print(f"[{split}] {n_saved}/{n_episodes} seed={seed} objects={len(scene)}", flush=True)
    return names


def main(args):
    task = tasks.make(args.task)
    env = Environment(assets_root=ASSETS_ROOT, task=task, disp=args.disp, use_egl=not args.no_egl)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t0 = time.time()
    with h5py.File(args.out, "w") as f:
        f.create_group("data")
        f["data"].attrs["task"] = args.task
        f["data"].attrs["env_args"] = json.dumps(dict(task=args.task, hz=480, assets="see2act.sim.assets",
                                                      cameras=list(cameras.AgentCameras.NAMES)))
        counter = [0]
        train = collect_split(env, task, args.n_train, args.train_seed_start, "train", f, counter)
        valid = collect_split(env, task, args.n_valid, args.valid_seed_start, "valid", f, counter)
        f.create_dataset("mask/train", data=np.array(train, dtype="S"))
        f.create_dataset("mask/valid", data=np.array(valid, dtype="S"))
        f["data"].attrs["total"] = counter[0]
    env.close()
    print(f"wrote {counter[0]} demonstrations to {args.out} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(tasks.names))
    ap.add_argument("--out", required=True, help="output hdf5 path")
    ap.add_argument("--n_train", type=int, default=100)
    ap.add_argument("--n_valid", type=int, default=20)
    ap.add_argument("--train_seed_start", type=int, default=-2, help="train seeds are start+2, start+4, ...")
    ap.add_argument("--valid_seed_start", type=int, default=-1, help="validation seeds are start+2, start+4, ...")
    ap.add_argument("--no_egl", action="store_true", help="use the CPU renderer instead of EGL")
    ap.add_argument("--disp", action="store_true")
    main(ap.parse_args())
