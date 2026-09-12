"""Render the views the policy chose during one evaluation episode into an image grid (and optionally a GIF).

  python scripts/visualize_episode.py --checkpoint <ckpt> --seed 100001 --out viz/bin-picking_100001.png
"""
import argparse
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from see2act import evaluation as ev  # noqa: E402
from see2act.see2act import See2Act  # noqa: E402
from see2act.renderer import Renderer  # noqa: E402
from see2act.sim import cameras  # noqa: E402
from see2act.sim.scene import encode_scene  # noqa: E402
from see2act.utils import FrameRecorder, write_video  # noqa: E402


def main(args):
    import cv2
    policy = See2Act.load(args.checkpoint, device="cuda")
    task = args.task or policy.checkpoint.get("task")
    mp.set_start_method("spawn", force=True)
    renderer = Renderer(task, num_workers=1).start()
    env = ev.make_env(task)
    try:
        recorder = None
        if args.video:
            cam = cameras.AgentCameras.CONFIG[cameras.AgentCameras.NAMES.index(args.execution_camera)]
            recorder = FrameRecorder(env, cam, every=args.frame_every)
            env.step_hook = recorder
        res = ev.run_episode(env, policy, renderer, args.seed, n_rollout=args.n_rollout, select_by=args.select_by,
                             num_steps=args.num_steps)
        env.step_hook = None
        scene = encode_scene(env.task.scene)
        print(f"seed {args.seed}: success={res['success']} ({res['failure_category']}), pick error {res['pick_error']:.3f} m, "
              f"place error {res['place_error']:.3f} m")
        rows = []
        frames = []
        for kf in res["keyframes"]:
            run = kf["runs"][kf["selected_run"]]
            views = []
            for pose in run["cam_traj"]:
                color, _, _ = renderer.render_raw(scene, np.asarray(pose))
                views.append(color)
                frames.append(color)
            step = max(1, len(views) // args.columns)
            picked = views[::step][:args.columns]
            rows.append(np.concatenate([cv2.resize(v, (args.size, args.size)) for v in picked], axis=1))
        grid = np.concatenate(rows, axis=0)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        cv2.imwrite(args.out, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"wrote {args.out} (rows: {', '.join(k['name'] for k in res['keyframes'])}; views from t=T to t=0 left to right)")
        if args.gif:
            write_video(frames, args.gif, fps=6, gif_every=1)
            print(f"wrote {args.gif} (the policy's views, pick then place)")
        if args.video:
            # the policy's views (each held for a few frames) followed by the robot executing the action
            import cv2
            h, w = recorder.frames[0].shape[:2] if recorder.frames else (320, 320)
            view_frames = [cv2.resize(f, (w, h)) for f in frames for _ in range(3)]
            write_video(view_frames + recorder.frames, args.video, fps=20)
            print(f"wrote {args.video} ({len(frames)} policy views, then {len(recorder.frames)} execution frames "
                  f"from the {args.execution_camera} camera)")
            gif = os.path.splitext(args.video)[0] + "_execution.gif"
            write_video(recorder.frames, gif, fps=20, gif_every=3, gif_size=(240, 240))
            print(f"wrote {gif}")
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        renderer.stop()
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task", default=None)
    ap.add_argument("--seed", type=int, default=200001)
    ap.add_argument("--out", default="viz/episode.png")
    ap.add_argument("--gif", default=None)
    ap.add_argument("--columns", type=int, default=10)
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--n_rollout", type=int, default=None)
    ap.add_argument("--select_by", default=None)
    ap.add_argument("--num_steps", type=int, default=None)
    ap.add_argument("--video", default=None, help=".mp4: the policy's views followed by the robot executing the action")
    ap.add_argument("--execution_camera", default="front", choices=list(cameras.AgentCameras.NAMES))
    ap.add_argument("--frame_every", type=int, default=8, help="record one execution frame every N simulation steps")
    main(ap.parse_args())
