"""Keyframe demonstration dataset (hdf5 written by scripts/collect_demos.py).

Layout:
  data.attrs["task"], data.attrs["env_args"] (json)
  data/demo_i/actions       (1, 14)  pick position, pick quaternion, place position, place quaternion
  data/demo_i/scene         (8 * n_objects,) encoded scene, see see2act.sim.scene
  data/demo_i/obs/rgb_<cam> (1, H, W, 3) uint8 reference images (not used by the policy)
  data/demo_i.attrs["seed"], ["reward"]
  mask/train, mask/valid    demo names of each split
"""
import json

import h5py
import numpy as np
import torch


class KeyframeDataset(torch.utils.data.Dataset):
    """All demonstrations of a split, held in memory. Items: {'actions': (14,), 'scene': (8k,)}."""

    def __init__(self, hdf5_path, split="train"):
        self.path = hdf5_path
        with h5py.File(hdf5_path, "r") as f:
            self.task = f["data"].attrs.get("task", None)
            if isinstance(self.task, bytes):
                self.task = self.task.decode()
            names = [n.decode() if isinstance(n, bytes) else str(n) for n in f[f"mask/{split}"][()]]
            names = sorted(names, key=lambda s: int(s.split("_")[-1]))
            self.demos = names
            self.actions = np.stack([f[f"data/{n}/actions"][0] for n in names]).astype(np.float32)
            self.scenes = np.stack([f[f"data/{n}/scene"][()] for n in names]).astype(np.float64)
            self.seeds = [int(f[f"data/{n}"].attrs["seed"]) for n in names]

    def __len__(self):
        return len(self.demos)

    def __getitem__(self, i):
        return {"actions": self.actions[i], "scene": self.scenes[i]}

    def __repr__(self):
        return f"KeyframeDataset(path={self.path}, task={self.task}, n_demos={len(self)})"


def collate(items):
    return {"actions": torch.as_tensor(np.stack([it["actions"] for it in items])),
            "scene": np.stack([it["scene"] for it in items])}


def make_loader(dataset, batch_size, shuffle, drop_last, seed=None):
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                                       collate_fn=collate, num_workers=0, generator=generator)


def dataset_task(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        task = f["data"].attrs.get("task", None)
        if task is None:
            task = json.loads(f["data"].attrs["env_args"])["task"]
    return task.decode() if isinstance(task, bytes) else str(task)
