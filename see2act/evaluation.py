"""Evaluation of a keyframe policy in the simulator: fixed per-episode seeds, per-episode records, summaries."""
import time

import numpy as np
import pybullet as p
import torch

from see2act.sim import ASSETS_ROOT, tasks
from see2act.sim.environment import Environment, action_from_vector, vector_from_action
from see2act.sim.scene import encode_scene, find_object
from see2act.utils import set_seed, wilson_ci


class Trace:
    """Per-episode record of the active viewpoint inference (camera trajectories, action estimates, timing)."""

    def __init__(self):
        self.keyframes = []
        self._t = {}
        self.timing = {"render": 0.0, "forward": 0.0}
        self.n_renders = 0

    def tic(self, key):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t[key] = time.time()

    def toc(self, key):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.timing[key] += time.time() - self._t[key]
        if key == "render":
            self.n_renders += 1

    def log_keyframe(self, name, runs, best):
        self.keyframes.append(dict(
            name=name, selected_run=int(best),
            runs=[dict(score=float(r["score"]), final=[float(x) for x in r["action"]],
                       cam_traj=np.asarray(r["cam_traj"]).round(6).tolist(),
                       act_traj=np.asarray(r["history"]).round(6).tolist()) for r in runs]))

    def to_dict(self):
        return dict(keyframes=self.keyframes, n_renders=int(self.n_renders),
                    t_render=float(self.timing["render"]), t_forward=float(self.timing["forward"]))


def make_env(task_name, use_egl=True, disp=False):
    return Environment(assets_root=ASSETS_ROOT, task=tasks.make(task_name), disp=disp, use_egl=use_egl)


def categorize_failure(prim_rec, success):
    if success:
        return "success"
    if prim_rec is None:
        return "unknown"
    if prim_rec.get("timeout") and not prim_rec.get("pick_contact"):
        return "pick_timeout"
    if not prim_rec.get("pick_success"):
        return "pick_miss"
    if prim_rec.get("timeout"):
        return "place_timeout"
    return "place_miss"


def _body_near(env, pos, tol=0.03):
    best, best_d = None, tol
    for oid in env.obj_ids.get("rigid", []):
        bp = np.array(p.getBasePositionAndOrientation(oid)[0])
        d = np.linalg.norm(bp[:2] - np.asarray(pos)[:2])
        if d < best_d:
            best, best_d = oid, d
    return best


def run_episode(env, policy, renderer, seed, init_view=None, n_rollout=None, select_by=None, num_steps=None,
                keep_trace=True):
    """Sample the scene of `seed`, infer the keyframe action, execute it, and score the outcome."""
    set_seed(seed)
    env.reset()                                    # scene sampled from the seed
    scene = env.task.scene
    env.reset_to_scene(scene)                      # same scene with the robot loaded
    task = env.task
    oracle = task.oracle_action(env)
    oracle14 = vector_from_action(oracle) if oracle is not None else None
    red = find_object(scene, "red_block")
    bowl = find_object(scene, "green_bowl")
    red_id = _body_near(env, red.position) if red is not None else None

    trace = Trace() if keep_trace else None
    t0 = time.time()
    action14 = policy.act(encode_scene(scene), renderer, init_view=init_view, n_rollout=n_rollout,
                          select_by=select_by, num_steps=num_steps, trace=trace)
    t_policy = time.time() - t0

    t1 = time.time()
    if np.all(np.isfinite(action14)):
        _, reward, done, info = env.step(action_from_vector(action14))
        prim_rec = task.primitive.last
    else:
        reward, prim_rec = 0.0, dict(timeout=True, pick_contact=False, pick_success=False)
    t_exec = time.time() - t1
    success = bool(task.done())
    final_block = list(p.getBasePositionAndOrientation(red_id)[0]) if red_id is not None else None

    result = dict(seed=int(seed), success=success, reward=float(reward),
                  failure_category=categorize_failure(prim_rec, success),
                  action14=[float(x) for x in action14],
                  oracle14=[float(x) for x in oracle14] if oracle14 is not None else None,
                  pick_error=float(np.linalg.norm(oracle14[:3] - action14[:3])) if oracle14 is not None else None,
                  place_error=float(np.linalg.norm(oracle14[7:10] - action14[7:10])) if oracle14 is not None else None,
                  t_policy_s=t_policy, t_exec_s=t_exec, primitive=prim_rec,
                  red_block_pos=list(red.position) if red is not None else None,
                  green_bowl_pos=list(bowl.position) if bowl is not None else None, final_block_pos=final_block)
    if trace is not None:
        result.update(trace.to_dict())
    return result


def summarize(episodes):
    n = len(episodes)
    k = sum(1 for e in episodes if e["success"])
    cats = {}
    for e in episodes:
        cats[e["failure_category"]] = cats.get(e["failure_category"], 0) + 1

    def mean(key):
        vals = [e[key] for e in episodes if e.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    return dict(n=n, successes=k, success_rate=(k / n if n else None), success_ci95=wilson_ci(k, n),
                failure_breakdown=cats, pick_error_m=mean("pick_error"), place_error_m=mean("place_error"),
                t_policy_s=mean("t_policy_s"), t_exec_s=mean("t_exec_s"), renders_per_episode=mean("n_renders"))
