"""Train See2Act on a keyframe demonstration dataset.

Example:
  python scripts/train.py --config configs/see2act.json --dataset data/bin-picking.hdf5 --name see2act_bin-picking

Writes runs/<name>/<timestamp>/{config.json, log.txt, metrics.jsonl, tb/, checkpoints/}.
"""
import argparse
import datetime
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from see2act.config import load_config  # noqa: E402
from see2act.data import KeyframeDataset, dataset_task, make_loader  # noqa: E402
from see2act.see2act import See2Act  # noqa: E402
from see2act.renderer import Renderer  # noqa: E402
from see2act.utils import MetricsLogger, Tee, set_seed  # noqa: E402


def run_epoch(policy, loader, renderer, train):
    logs = []
    for batch in loader:
        logs.append(policy.train_step(batch, renderer) if train else policy.validation_step(batch, renderer))
    return {k: float(np.mean([l[k] for l in logs])) for k in logs[0]}


def main(args):
    cfg = load_config(args.config, args.set)
    if args.dataset:
        cfg.data.path = args.dataset
    if args.name:
        cfg.experiment.name = args.name
    if args.output_dir:
        cfg.experiment.output_dir = args.output_dir
    if args.num_epochs:
        cfg.train.num_epochs = args.num_epochs
    if args.renderer_workers:
        cfg.train.renderer_workers = args.renderer_workers
    assert cfg.data.path, "--dataset (or data.path in the config) is required"
    task = dataset_task(cfg.data.path)
    cfg.data.task = task

    stamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir = os.path.join(os.path.expanduser(cfg.experiment.output_dir), cfg.experiment.name, stamp)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    sys.stdout = sys.stderr = Tee(os.path.join(run_dir, "log.txt"))
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        f.write(cfg.dumps())
    print(f"run directory: {run_dir}\nconfig:\n{cfg.dumps()}", flush=True)

    set_seed(cfg.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = See2Act(cfg, device)
    policy.create_optimizer()
    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        policy.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt and not args.reset_optimizer:
            policy.optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1 if args.start_epoch is None else args.start_epoch
        print(f"resumed {args.resume} at epoch {start_epoch}", flush=True)
    n_params = sum(p.numel() for p in policy.nets.parameters())
    print(f"task: {task}  device: {device}  parameters: {n_params / 1e6:.1f}M", flush=True)

    train_set = KeyframeDataset(cfg.data.path, cfg.data.train_split)
    train_loader = make_loader(train_set, cfg.train.batch_size, shuffle=True, drop_last=True)
    valid_loader = None
    if cfg.experiment.validate:
        valid_set = KeyframeDataset(cfg.data.path, cfg.data.valid_split)
        valid_loader = make_loader(valid_set, min(cfg.train.batch_size, len(valid_set)), shuffle=False, drop_last=False)
        print(f"{train_set}\n{valid_set}", flush=True)

    logger = MetricsLogger(run_dir, tensorboard=cfg.experiment.log_tensorboard)
    best_valid, best_epoch, best_path = None, start_epoch, None
    renderer = Renderer(task, num_workers=cfg.train.renderer_workers).start()
    t_start = time.time()
    try:
        for epoch in range(start_epoch, int(cfg.train.num_epochs) + 1):
            policy.train_mode()
            t0 = time.time()
            train_log = run_epoch(policy, train_loader, renderer, train=True)
            train_log["epoch_time_s"] = time.time() - t0
            logger.log(epoch, train_log, prefix="train/")
            msg = f"epoch {epoch} train loss {train_log['loss']:.5f} ({train_log['epoch_time_s']:.1f}s)"

            save_periodic = cfg.experiment.save_every_n_epochs and epoch % int(cfg.experiment.save_every_n_epochs) == 0
            if valid_loader is not None:
                policy.eval_mode()
                valid_log = run_epoch(policy, valid_loader, renderer, train=False)
                logger.log(epoch, valid_log, prefix="valid/")
                msg += f" valid loss {valid_log['loss']:.5f}"
                if best_valid is None or valid_log["loss"] <= best_valid:
                    best_valid, best_epoch = valid_log["loss"], epoch
                    if cfg.experiment.save_best_validation:
                        if best_path and os.path.exists(best_path):
                            os.remove(best_path)
                        best_path = os.path.join(ckpt_dir, f"best_valid_epoch_{epoch}.pth")
                        policy.save(best_path, epoch=epoch, valid_loss=best_valid, task=task)
                        msg += " *"
            if save_periodic:
                policy.save(os.path.join(ckpt_dir, f"epoch_{epoch}.pth"), epoch=epoch, task=task,
                            save_optimizer=bool(cfg.experiment.get("save_optimizer", True)))
            if epoch % 10 == 0 or epoch == start_epoch or save_periodic:
                print(msg + f"  [elapsed {(time.time() - t_start) / 3600:.2f} h]", flush=True)
            patience = int(cfg.train.early_stopping_patience or 0)
            if patience > 0 and best_valid is not None and (epoch - best_epoch) >= patience:
                print(f"early stopping at epoch {epoch}: no validation improvement since epoch {best_epoch}", flush=True)
                break
        policy.save(os.path.join(ckpt_dir, "last.pth"), epoch=epoch, task=task, save_optimizer=True)  # resumable
        print(f"training finished; best validation loss {best_valid} at epoch {best_epoch}", flush=True)
    finally:
        logger.close()
        renderer.stop()
        sys.stdout.flush()
        os._exit(0)  # render workers can keep the interpreter alive at shutdown


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="JSON config (defaults from see2act/config.py)")
    ap.add_argument("--dataset", type=str, default=None)
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--num_epochs", type=int, default=None)
    ap.add_argument("--renderer_workers", type=int, default=None)
    ap.add_argument("--resume", type=str, default=None, help="checkpoint to continue from")
    ap.add_argument("--start_epoch", type=int, default=None)
    ap.add_argument("--reset_optimizer", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override, e.g. train.num_epochs=100")
    main(ap.parse_args())
