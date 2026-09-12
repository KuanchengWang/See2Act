"""Configuration: nested dict with attribute access, JSON files and `key.path=value` overrides."""
import copy
import json

import numpy as np

DEFAULTS = {
    "experiment": {
        "name": "see2act",              # run name (a timestamped sub-directory is created under output_dir)
        "output_dir": "runs",
        "seed": 1,
        "validate": True,               # evaluate the loss on the validation split every epoch
        "save_every_n_epochs": 2000,    # periodic checkpoints
        "save_best_validation": True,   # keep the checkpoint with the lowest validation loss
        "save_optimizer": True,         # store the optimizer state in periodic checkpoints (3x larger, resumable)
        "log_tensorboard": True,
    },
    "data": {
        "path": None,                   # hdf5 written by scripts/collect_demos.py
        "train_split": "train",
        "valid_split": "valid",
    },
    "train": {
        "batch_size": 50,
        "num_epochs": 30000,            # one epoch = one pass over the demonstrations
        "learning_rate": 1e-4,
        # The shared image encoder receives the gradient step of all three optimizers of the original
        # implementation (one per network); a single optimizer reproduces this with a 3x learning rate.
        "encoder_lr_multiplier": 3.0,
        "num_noisy_samples": 1,         # noisy action samples per demonstration and step
        "renderer_workers": 14,
        "early_stopping_patience": 0,   # epochs without validation improvement before stopping (0 = off)
    },
    "model": {
        "image_size": [320, 320],
        "feature_dim": 100,             # image feature size after the encoder MLP
        "encoder_mlp_dims": [1024, 1024],
        "denoiser_mlp_dims": [512, 512, 512, 512],
        "timestep_encoding_dim": 64,
        "text_embedding_dim": 512,      # fixed text embeddings of the keyframe names (pick / place)
    },
    "diffusion": {
        "beta_start": 1e-4,             # variance schedule beta(t) = beta_start + (beta_end - beta_start) * t
        "beta_end": 0.02,
        "num_refinement_steps": 50,     # T: denoising / viewpoint refinement steps at test time (inference only)
    },
    "camera": {
        "init_view": "front",           # C_T: front | top | back | left | right
        "offset": 0.6,                  # distance d of the terminal view C_0 from the target along the approach axis
        "jitter": True,                 # train-time pose jitter, scaled by t
        "jitter_position": 0.05,        # m
        "jitter_rotation_deg": 5.0,
    },
    "inference": {
        "n_rollout": 1,                 # independent refinement runs per keyframe
        # which run to keep (lower score wins): none | camera_variance (variance of the camera-pose changes along
        # the run) | final_camera_variance / final_action_variance (same, over the last `final_window` steps) |
        # action_variance | action_entropy (spread of the per-step action estimates)
        "select_by": "none",
        "final_window": 5,
    },
    "action": {
        # world-frame action bounds used for normalization: x, y, z (m) and three orientation angles (rad)
        "position_bounds": [[0.0, 1.0], [-0.5, 0.5], [0.0, 1.0]],
        "orientation_bounds": [[-np.pi, np.pi]] * 3,
    },
}


class Config(dict):
    """dict with attribute access (`cfg.train.batch_size`)."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key, value):
        self[key] = value

    @classmethod
    def wrap(cls, d):
        return cls({k: (cls.wrap(v) if isinstance(v, dict) else v) for k, v in d.items()})

    def to_dict(self):
        return {k: (v.to_dict() if isinstance(v, Config) else v) for k, v in self.items()}

    def get_path(self, dotted):
        node = self
        for k in dotted.split("."):
            node = node[k]
        return node

    def set_path(self, dotted, value):
        parts = dotted.split(".")
        node = self
        for k in parts[:-1]:
            if k not in node:
                node[k] = Config()
            node = node[k]
        node[parts[-1]] = Config.wrap(value) if isinstance(value, dict) else value

    def dumps(self):
        return json.dumps(self.to_dict(), indent=2)


def _deep_update(base, other):
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = Config.wrap(v) if isinstance(v, dict) else v
    return base


def parse_value(text):
    """'20' -> 20, 'true' -> True, '[1,2]' -> [1, 2], 'front' -> 'front'."""
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return text


def default_config():
    return Config.wrap(copy.deepcopy(DEFAULTS))


def load_config(path=None, overrides=(), base=None):
    """Defaults, updated by the JSON file at `path` (if any), then by `key.path=value` overrides."""
    cfg = default_config() if base is None else Config.wrap(copy.deepcopy(base))
    if path is not None:
        with open(path) as f:
            _deep_update(cfg, json.load(f))
    for item in overrides or ():
        key, value = item.split("=", 1)
        cfg.set_path(key.strip(), parse_value(value.strip()))
    return cfg
