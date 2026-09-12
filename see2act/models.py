"""Networks: FiLM-conditioned ResNet-18 image encoder and the MLP noise predictor."""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as vision_models

from see2act.constants import KEYFRAMES


class MLP(nn.Module):
    """Linear-ReLU stack. With `residual=True` the output is body(x) + relu(skip(x)) (a residual MLP)."""

    def __init__(self, input_dim, output_dim, hidden_dims=(), output_activation=None, residual=False):
        super().__init__()
        layers, dim = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(dim, h), nn.ReLU()]
            dim = h
        layers.append(nn.Linear(dim, output_dim))
        self.body = nn.Sequential(*layers)
        self.skip = nn.Sequential(nn.Linear(input_dim, output_dim), nn.ReLU()) if residual else None
        self.output_activation = output_activation() if output_activation is not None else None

    def forward(self, x):
        out = self.body(x)
        if self.skip is not None:
            out = out + self.skip(x)
        if self.output_activation is not None:
            out = self.output_activation(out)
        return out


class FiLM(nn.Module):
    """Feature-wise linear modulation of a conv feature map by a conditioning vector."""

    def __init__(self, cond_dim, channels):
        super().__init__()
        self.proj = nn.Linear(cond_dim, channels * 2)

    def forward(self, x, cond):
        b, c = x.shape[:2]
        beta, gamma = torch.split(self.proj(cond).reshape(b, c * 2, 1, 1), [c, c], 1)
        return torch.relu((1 + gamma) * x + beta)


class KeyframeBatchNorm2d(nn.Module):
    """BatchNorm2d with one set of running statistics per keyframe.

    The encoder is conditioned on the keyframe (pick / place) through FiLM and every training batch contains
    the views of a single keyframe, so the feature statistics normalized by BatchNorm in training mode are
    keyframe-specific. A single shared running estimate (the mixture of both) then does not match either
    keyframe at inference, which can degrade eval-mode predictions badly. Keeping running statistics per
    keyframe makes inference consistent with training; the training computation itself is unchanged.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, keyframes=KEYFRAMES):
        super().__init__()
        self.num_features, self.eps, self.momentum, self.keyframes = num_features, eps, momentum, tuple(keyframes)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        for kf in self.keyframes:
            self.register_buffer(f"running_mean_{kf}", torch.zeros(num_features))
            self.register_buffer(f"running_var_{kf}", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        self.active = self.keyframes[0]

    @classmethod
    def from_batchnorm(cls, bn):
        m = cls(bn.num_features, bn.eps, bn.momentum)
        with torch.no_grad():
            m.weight.copy_(bn.weight)
            m.bias.copy_(bn.bias)
            for kf in m.keyframes:
                getattr(m, f"running_mean_{kf}").copy_(bn.running_mean)
                getattr(m, f"running_var_{kf}").copy_(bn.running_var)
            m.num_batches_tracked.copy_(bn.num_batches_tracked)
        return m

    def forward(self, x):
        factor = 0.0
        if self.training:
            self.num_batches_tracked += 1
            # momentum=None -> cumulative average (as in torch.nn.BatchNorm2d)
            factor = 1.0 / float(self.num_batches_tracked) if self.momentum is None else self.momentum
        return F.batch_norm(x, getattr(self, f"running_mean_{self.active}"), getattr(self, f"running_var_{self.active}"),
                            self.weight, self.bias, self.training, factor, self.eps)

    def extra_repr(self):
        return f"{self.num_features}, eps={self.eps}, momentum={self.momentum}, keyframes={self.keyframes}"


def _replace_batchnorms(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, KeyframeBatchNorm2d.from_batchnorm(child))
        else:
            _replace_batchnorms(child)


class FiLMResNet18(nn.Module):
    """ResNet-18 trunk (no pooling / fc) with a FiLM layer after each of the eight basic blocks and
    per-keyframe BatchNorm statistics."""

    def __init__(self, cond_dim=512, input_channels=3):
        super().__init__()
        net = vision_models.resnet18(weights=None)
        if input_channels != 3:
            net.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        stem, blocks = [], []
        for layer in net.children():
            if isinstance(layer, nn.Sequential):
                blocks.extend(list(layer))
            elif len(blocks) == 0:
                stem.append(layer)
        self.stem = nn.Sequential(*stem)
        self.blocks = nn.ModuleList(blocks)
        film, channels = [], input_channels
        with torch.no_grad():
            channels = self.stem(torch.rand(1, input_channels, 3, 3)).shape[1]
            for block in self.blocks:
                channels = block(torch.rand(1, channels, 3, 3)).shape[1]
                film.append(FiLM(cond_dim, channels))
        self.film = nn.ModuleList(film)
        self.output_channels = channels
        _replace_batchnorms(self)
        self.norms = [m for m in self.modules() if isinstance(m, KeyframeBatchNorm2d)]

    def set_keyframe(self, keyframe):
        for m in self.norms:
            m.active = keyframe

    def output_shape(self, input_shape):
        c, h, w = input_shape
        return [self.output_channels, int(math.ceil(h / 32.0)), int(math.ceil(w / 32.0))]

    def forward(self, images, cond, keyframe=KEYFRAMES[0]):
        self.set_keyframe(keyframe)
        x = self.stem(images)
        for block, film in zip(self.blocks, self.film):
            x = film(block(x), cond)
        return x


class ImageEncoder(nn.Module):
    """FiLM ResNet-18 -> flatten -> MLP -> `feature_dim` features (ReLU)."""

    def __init__(self, image_shape, feature_dim=100, mlp_dims=(1024, 1024), cond_dim=512):
        super().__init__()
        self.backbone = FiLMResNet18(cond_dim=cond_dim, input_channels=image_shape[0])
        self.conv_feature_dim = int(np.prod(self.backbone.output_shape(list(image_shape))))
        self.mlp = MLP(self.conv_feature_dim, feature_dim, mlp_dims, output_activation=nn.ReLU)
        self.feature_dim = feature_dim

    def forward(self, images, cond, keyframe=KEYFRAMES[0]):
        x = self.backbone(images, cond, keyframe)
        return self.mlp(x.flatten(1))


class NoisePredictor(nn.Module):
    """epsilon_psi(image features, noisy action, timestep encoding) -> predicted noise (action_dim)."""

    def __init__(self, feature_dim, action_dim, timestep_dim, hidden_dims=(512, 512, 512, 512)):
        super().__init__()
        self.mlp = MLP(feature_dim + action_dim + timestep_dim, action_dim, hidden_dims, residual=True)

    def forward(self, features, noisy_actions, timestep_enc):
        """features (B, F); noisy_actions (B, K, A); timestep_enc (B, K, D) -> (B, K, A)."""
        feats = features.unsqueeze(1).expand(-1, noisy_actions.shape[1], -1)
        return self.mlp(torch.cat([feats, noisy_actions, timestep_enc], -1))


def sinusoidal_encoding(values, dim):
    """Sinusoidal positional encoding of scalars: (..., 1) -> (..., dim)."""
    device = values.device
    div_term = torch.exp(torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim))
    pe = torch.zeros(size=list(values.shape) + [dim], device=device)
    pe[..., 0::2] = torch.sin(values.unsqueeze(-1) * div_term)
    pe[..., 1::2] = torch.cos(values.unsqueeze(-1) * div_term)
    return pe.reshape(list(pe.shape[:-2]) + [-1])
