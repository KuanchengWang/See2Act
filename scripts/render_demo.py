"""Replay a collected demonstration: rebuild its scene, execute the expert's keyframe action, and record a video.

  python scripts/render_demo.py --dataset data/bin-picking.hdf5 --demo demo_0 --out media/demo_bin-picking.mp4

Also writes the reference images stored with the demonstration (<out>_front.png, <out>_top.png).
"""
import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from see2act.data import dataset_task  # noqa: E402
from see2act.sim import ASSETS_ROOT, cameras, tasks  # noqa: E402
from see2act.sim.environment import Environment, action_from_vector  # noqa: E402
from see2act.sim.scene import decode_scene  # noqa: E402
from see2act.utils import FrameRecorder, write_video  # noqa: E402


def main(args):
    import cv2
    task_name = dataset_task(args.dataset)
    with h5py.File(args.dataset, "r") as f:
        g = f[f"data/{args.demo}"]
        action = np.asarray(g["actions"][0]); scene = decode_scene(g["scene"][()])
        seed = int(g.attrs["seed"]); imgs = {k: np.asarray(g[f"obs/{k}"][0]) for k in g["obs"]}
    task = tasks.make(task_name)
    env = Environment(assets_root=ASSETS_ROOT, task=task, use_egl=not args.no_egl)
    try:
        env.reset_to_scene(scene)
        ac = cameras.AgentCameras
        pos, rot = (ac.top_down_position, ac.top_down_rotation) if args.camera == "top" else (ac.front_position, ac.front_rotation)
        f = cameras.INTRINSICS[0] * args.size / cameras.IMAGE_SIZE[0]          # same field of view at any resolution
        cam = cameras.camera_config(pos, rot, image_size=(args.size, args.size),
                                    intrinsics=(f, 0, args.size / 2, 0, f, args.size / 2, 0, 0, 1))
        rec = FrameRecorder(env, cam, every=args.frame_every)
        rec.capture()
        env.step_hook = rec
        _, reward, done, _ = env.step(action_from_vector(action))
        env.step_hook = None
        rec.capture()
        base = os.path.splitext(args.out)[0]
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        write_video(rec.frames, args.out, fps=20)
        write_video(rec.frames, base + ".gif", fps=20, gif_every=args.gif_every,
                    gif_size=(args.gif_size, args.gif_size) if args.gif_size else None)
        for k, im in imgs.items():
            cv2.imwrite(f"{base}_{k}.png", cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        print(f"{task_name} {args.demo} (seed {seed}): reward {reward}, success {task.done()}; "
              f"wrote {args.out}, {base}.gif and {', '.join(f'{base}_{k}.png' for k in imgs)} ({len(rec.frames)} frames)")
    finally:
        env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--demo", default="demo_0")
    ap.add_argument("--out", required=True, help="output .mp4 path")
    ap.add_argument("--camera", default="front", choices=list(cameras.AgentCameras.NAMES))
    ap.add_argument("--frame_every", type=int, default=8, help="record one frame every N simulation steps")
    ap.add_argument("--size", type=int, default=320, help="render resolution (square)")
    ap.add_argument("--gif_every", type=int, default=2, help="keep every N-th recorded frame in the GIF")
    ap.add_argument("--gif_size", type=int, default=None, help="GIF resolution (default: the render resolution)")
    ap.add_argument("--no_egl", action="store_true")
    main(ap.parse_args())
