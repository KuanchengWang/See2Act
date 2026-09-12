"""The See2Act policy: training loss (Alg. 1) and active viewpoint inference (Alg. 2)."""
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from see2act import camera as cam
from see2act import geometry
from see2act.config import load_config
from see2act.constants import KEYFRAMES, load_text_embeddings
from see2act.models import ImageEncoder, NoisePredictor, sinusoidal_encoding

ACTION_DIM = 6          # x, y, z, rx, ry, rz
KEYFRAME_DIM = 14       # pick pose7 + place pose7
SELECTION_METHODS = ("none", "camera_variance", "final_camera_variance", "final_action_variance", "action_variance",
                     "action_entropy")


class See2Act:
    """Keyframe policy: for each keyframe (pick, place) an action is inferred by T refinement steps, each
    conditioned on a view rendered from a camera pose computed from the current action estimate."""

    def __init__(self, config, device):
        self.config = config
        self.device = torch.device(device)
        m = config.model
        self.image_shape = [3, int(m.image_size[0]), int(m.image_size[1])]
        self.action_bounds = np.array(list(config.action.position_bounds) + list(config.action.orientation_bounds),
                                      dtype=np.float32)
        assert self.action_bounds.shape == (ACTION_DIM, 2)
        self.schedule = cam.CameraSchedule.from_config(config.camera)
        self.text_embeddings = {k: torch.as_tensor(v, device=self.device) for k, v in load_text_embeddings().items()}

        self.nets = nn.ModuleDict()
        self.nets["encoder"] = ImageEncoder(self.image_shape, m.feature_dim, m.encoder_mlp_dims, m.text_embedding_dim)
        for kf in KEYFRAMES:
            self.nets[f"{kf}_denoiser"] = NoisePredictor(m.feature_dim, ACTION_DIM, m.timestep_encoding_dim,
                                                         m.denoiser_mlp_dims)
        self.nets = self.nets.float().to(self.device)
        self.optimizer = None

    # ------------------------------------------------------------------ optimizer / state
    def create_optimizer(self):
        lr = float(self.config.train.learning_rate)
        groups = [{"params": self.nets["encoder"].parameters(), "lr": lr * float(self.config.train.encoder_lr_multiplier)}]
        for kf in KEYFRAMES:
            groups.append({"params": self.nets[f"{kf}_denoiser"].parameters(), "lr": lr})
        self.optimizer = torch.optim.Adam(groups)
        return self.optimizer

    def train_mode(self):
        self.nets.train()

    def eval_mode(self):
        self.nets.eval()

    def state_dict(self):
        return self.nets.state_dict()

    def load_state_dict(self, state):
        self.nets.load_state_dict(state)

    def save(self, path, **extra):
        save_optimizer = extra.pop("save_optimizer", False)
        payload = dict(model=self.nets.state_dict(), config=self.config.to_dict(), version=1, **extra)
        if self.optimizer is not None and save_optimizer:
            payload["optimizer"] = self.optimizer.state_dict()
        torch.save(payload, path)

    @classmethod
    def load(cls, path, device="cuda", overrides=()):
        """Rebuild a policy from a checkpoint; `overrides` (key=value) adjust inference settings."""
        ckpt = torch.load(path, map_location="cpu")
        config = load_config(None, overrides, base=ckpt["config"])
        policy = cls(config, device)
        policy.load_state_dict(ckpt["model"])
        policy.eval_mode()
        policy.checkpoint = {k: v for k, v in ckpt.items() if k not in ("model", "optimizer")}
        return policy

    # ------------------------------------------------------------------ diffusion helpers
    def beta(self, t):
        d = self.config.diffusion
        return d.beta_start + (d.beta_end - d.beta_start) * t

    def timesteps(self, num_steps=None):
        """Refinement schedule t = 1, 1 - 1/T, ..., 1/T (t = 1 is pure noise / the initial view)."""
        n = int(num_steps or self.config.diffusion.num_refinement_steps)
        return np.linspace(0, 1, n + 1)[1:][::-1]

    def _encode_t(self, t):
        return sinusoidal_encoding(t, self.config.model.timestep_encoding_dim)

    def _features(self, keyframe, images):
        cond = self.text_embeddings[keyframe].unsqueeze(0).expand(images.shape[0], -1)
        return self.nets["encoder"](images, cond, keyframe)

    def normalize(self, actions):
        return geometry.normalize(actions, self.action_bounds)

    def unnormalize(self, actions):
        return geometry.unnormalize(actions, self.action_bounds)

    @staticmethod
    def keyframe_targets(actions14, keyframe):
        """(B, 14) keyframe vectors -> world-frame 6-d targets (B, 6) of one keyframe."""
        off = 0 if keyframe == "pick" else 7
        pos = actions14[:, off:off + 3]
        quats = actions14[:, off + 3:off + 7].detach().cpu().numpy()
        eul = torch.as_tensor(np.stack([geometry.action_quat_to_euler(q) for q in quats]), device=actions14.device)
        return torch.cat([pos, eul], -1)

    # ------------------------------------------------------------------ training
    def compute_loss(self, batch, renderer, train=True, rng=np.random):
        """Denoising loss of Alg. 1 for a batch {'actions': (B, 14) tensor, 'scene': (B, 8k) array}."""
        actions = batch["actions"].to(self.device).float()
        scenes = np.asarray(batch["scene"])
        b = actions.shape[0]
        k = int(self.config.train.num_noisy_samples)
        losses = OrderedDict()
        total = torch.zeros((), device=self.device)
        for kf in KEYFRAMES:
            target = self.keyframe_targets(actions, kf)                           # (B, 6) world frame
            # random numbers are drawn on the CPU so that seeded runs are reproducible across devices
            t = torch.rand([b, k, 1]).to(self.device)                               # diffusion time
            noise = torch.randn([b, k, ACTION_DIM]).to(self.device)
            beta = self.beta(t)
            # camera pose of every sample from its ground-truth keyframe and the sampled time, then render
            tgt = target.detach().cpu().numpy()
            init_pose = cam.init_view_pose(self.schedule.init_view)
            poses = [cam.pose7(*cam.view_at(self.schedule, tgt[i, :3], tgt[i, 3:], float(t[i, 0, 0]), init_pose,
                                             train=train, rng=rng)) for i in range(b)]
            assert k == 1, "num_noisy_samples > 1 needs one view per sample"
            poses = np.asarray(poses)
            images = torch.from_numpy(renderer.render(scenes, poses)).to(self.device)
            cam_pos, cam_eul = poses[:, :3], np.stack([geometry.quat_to_euler(q) for q in poses[:, 3:7]])
            # express the target in the camera frame and noise it there
            target_cam = torch.from_numpy(geometry.world_to_camera(tgt, cam_pos, cam_eul)).to(self.device)
            clean = self.normalize(target_cam)
            noisy = clean.unsqueeze(1) + beta.sqrt() * noise
            pred = self.nets[f"{kf}_denoiser"](self._features(kf, images), noisy, self._encode_t(t))
            loss = nn.functional.mse_loss(pred, noise)
            losses[f"{kf}_loss"] = loss
            losses[f"{kf}_position_loss"] = nn.functional.mse_loss(pred[..., :3], noise[..., :3])
            losses[f"{kf}_orientation_loss"] = nn.functional.mse_loss(pred[..., 3:], noise[..., 3:])
            total = total + loss
        losses["loss"] = total
        return losses

    def train_step(self, batch, renderer):
        assert self.optimizer is not None, "call create_optimizer() first"
        self.optimizer.zero_grad()
        losses = self.compute_loss(batch, renderer, train=True)
        losses["loss"].backward()
        info = {k: float(v.item()) for k, v in losses.items()}
        for name in self.nets:
            g = 0.0
            for p_ in self.nets[name].parameters():
                if p_.grad is not None:
                    g += p_.grad.data.norm(2).pow(2).item()
            info[f"{name}_grad_norm"] = g
        self.optimizer.step()
        return info

    @torch.no_grad()
    def validation_step(self, batch, renderer):
        losses = self.compute_loss(batch, renderer, train=True)
        return {k: float(v.item()) for k, v in losses.items()}

    # ------------------------------------------------------------------ inference (active viewpoint inference)
    @torch.no_grad()
    def denoise_step(self, keyframe, act_world, images, cam_pos, cam_eul, t, add_noise):
        """One refinement step on normalized world-frame estimates act_world (B, 6) given the views rendered
        from the cameras at (cam_pos, cam_eul). Returns the refined normalized world-frame estimates."""
        b = act_world.shape[0]
        t_tensor = t * torch.ones([b, 1], device=self.device)
        beta = self.beta(t_tensor)
        world = self.unnormalize(act_world).detach().cpu().numpy()
        act = self.normalize(torch.from_numpy(geometry.world_to_camera(world, cam_pos, cam_eul)).to(self.device))
        if add_noise:
            act = act + beta.sqrt() * torch.randn(act.shape).to(self.device)
        pred = self.nets[f"{keyframe}_denoiser"](self._features(keyframe, images), act.unsqueeze(1),
                                                 self._encode_t(t_tensor).unsqueeze(1))[:, 0]
        act = act - beta.sqrt() * pred
        local = self.unnormalize(act).detach().cpu().numpy()
        return self.normalize(torch.from_numpy(geometry.camera_to_world(local, cam_pos, cam_eul)).to(self.device))

    @torch.no_grad()
    def refine(self, keyframe, scene, renderer, init_pose, num_steps=None, trace=None):
        """One run of Alg. 2 for a keyframe. Returns (world action (6,), action history (T, 6), camera
        trajectory (T, 7))."""
        act = torch.randn([1, ACTION_DIM]).to(self.device)
        history, cam_traj = [], []
        for i, t in enumerate(self.timesteps(num_steps)):
            world = self.unnormalize(act)[0].detach().cpu().numpy()
            pos, quat = cam.view_at(self.schedule, world[:3], world[3:], t, init_pose, train=False)
            pose = cam.pose7(pos, quat)
            if trace is not None:
                trace.tic("render")
            images = torch.from_numpy(renderer.render(scene, pose[None])).to(self.device)
            if trace is not None:
                trace.toc("render")
                trace.tic("forward")
            act = self.denoise_step(keyframe, act, images, pose[None, :3], geometry.quat_to_euler(quat)[None],
                                    float(t), add_noise=(i != 0))
            if trace is not None:
                trace.toc("forward")
            history.append(self.unnormalize(act)[0].detach().cpu().numpy())
            cam_traj.append(pose)
        return history[-1], np.asarray(history), np.asarray(cam_traj)

    @torch.no_grad()
    def infer_keyframe(self, keyframe, scene, renderer, init_pose, n_rollout=None, select_by=None, num_steps=None,
                       trace=None):
        """Run `n_rollout` refinements and keep the one with the lowest `select_by` score."""
        n_rollout = int(n_rollout or self.config.inference.n_rollout)
        select_by = select_by or self.config.inference.select_by
        assert select_by in SELECTION_METHODS, select_by
        runs = []
        for r in range(n_rollout):
            action, history, cam_traj = self.refine(keyframe, scene, renderer, init_pose, num_steps, trace)
            score = 0.0 if n_rollout == 1 else cam.run_score(select_by, history, cam_traj,
                                                             int(self.config.inference.get("final_window", 5)))
            runs.append(dict(action=action, history=history, cam_traj=cam_traj, score=float(score)))
        best = int(np.argmin([run["score"] for run in runs]))
        if trace is not None:
            trace.log_keyframe(keyframe, runs, best)
        return runs[best]["action"], runs

    @torch.no_grad()
    def act(self, scene, renderer, init_view=None, n_rollout=None, select_by=None, num_steps=None, trace=None):
        """Infer the full keyframe action (14,) for a scene: [pick pos, pick quat, place pos, place quat]."""
        self.eval_mode()
        init_pose = cam.init_view_pose(init_view or self.schedule.init_view)
        out = np.zeros(KEYFRAME_DIM, dtype=np.float64)
        for kf, off in zip(KEYFRAMES, (0, 7)):
            action, _ = self.infer_keyframe(kf, scene, renderer, init_pose, n_rollout, select_by, num_steps, trace)
            out[off:off + 3] = action[:3]
            out[off + 3:off + 7] = geometry.action_euler_to_quat(action[3:])
        return out
