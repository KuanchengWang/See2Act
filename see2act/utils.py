"""Small helpers: seeding, logging, timing, statistics."""
import json
import os
import random
import sys
import time

import numpy as np
import torch


def set_seed(seed):
    seed = int(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


class Tee:
    """Duplicate stdout/stderr into a file."""

    def __init__(self, path):
        self.terminal = sys.stdout
        self.file = open(path, "a")

    def write(self, msg):
        self.terminal.write(msg)
        self.file.write(msg)
        self.file.flush()

    def flush(self):
        self.terminal.flush()
        self.file.flush()


class MetricsLogger:
    """JSON-lines metrics file plus optional TensorBoard."""

    def __init__(self, log_dir, tensorboard=True):
        os.makedirs(log_dir, exist_ok=True)
        self.jsonl = open(os.path.join(log_dir, "metrics.jsonl"), "a")
        self.writer = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(os.path.join(log_dir, "tb"))
            except Exception as e:  # pragma: no cover
                print(f"[logger] tensorboard disabled: {e}")

    def log(self, epoch, metrics, prefix=""):
        rec = {"epoch": int(epoch)}
        rec.update({f"{prefix}{k}": float(v) for k, v in metrics.items()})
        self.jsonl.write(json.dumps(rec) + "\n")
        self.jsonl.flush()
        if self.writer is not None:
            for k, v in metrics.items():
                self.writer.add_scalar(f"{prefix}{k}", float(v), epoch)

    def close(self):
        self.jsonl.close()
        if self.writer is not None:
            self.writer.close()


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def elapsed(self):
        return time.time() - self.t0


def wilson_ci(k, n, z=1.96):
    """95% Wilson score interval of a success rate."""
    if n == 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (float(center - half), float(center + half))


class FrameRecorder:
    """Collects frames from a camera every `every` simulation steps (used as Environment.step_hook)."""

    def __init__(self, env, camera_config, every=8):
        self.env, self.camera_config, self.every = env, camera_config, int(every)
        self.frames, self._n = [], 0

    def __call__(self):
        self._n += 1
        if self._n % self.every == 0:
            self.capture()

    def capture(self):
        color, _, _ = self.env.render_camera(self.camera_config)
        self.frames.append(np.asarray(color, dtype=np.uint8))


def write_video(frames, path, fps=20, gif_every=2, gif_size=None):
    """Write frames (list of HxWx3 uint8 RGB) to an .mp4 (OpenCV) and/or .gif (Pillow) depending on the extension."""
    import cv2
    frames = [np.asarray(f, dtype=np.uint8) for f in frames]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if path.endswith(".gif"):
        from PIL import Image
        ims = [Image.fromarray(f) for f in frames[::max(1, gif_every)]]
        if gif_size:
            ims = [im.resize(gif_size) for im in ims]
        ims[0].save(path, save_all=True, append_images=ims[1:], duration=int(1000 * gif_every / fps), loop=0)
    else:
        h, w = frames[0].shape[:2]
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in frames:
            out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        out.release()
    return path
