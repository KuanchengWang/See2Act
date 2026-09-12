# Copyright 2023 The Ravens Authors. Licensed under the Apache License, Version 2.0.
# Trimmed and adapted for See2Act; see NOTICE for details.
"""Geometry and heightmap utilities (subset of ravens.utils.utils)."""
import numpy as np
import pybullet as p
from transforms3d import euler


# ----------------------------------------------------------------------------- heightmaps
def get_heightmap(points, colors, bounds, pixel_size):
    """Top-down orthographic heightmap from a point cloud (HxWx3 points, HxWxC colors)."""
    width = int(np.round((bounds[0, 1] - bounds[0, 0]) / pixel_size))
    height = int(np.round((bounds[1, 1] - bounds[1, 0]) / pixel_size))
    heightmap = np.zeros((height, width), dtype=np.float32)
    colormap = np.zeros((height, width, colors.shape[-1]), dtype=np.uint8)

    ix = (points[..., 0] >= bounds[0, 0]) & (points[..., 0] < bounds[0, 1])
    iy = (points[..., 1] >= bounds[1, 0]) & (points[..., 1] < bounds[1, 1])
    iz = (points[..., 2] >= bounds[2, 0]) & (points[..., 2] < bounds[2, 1])
    valid = ix & iy & iz
    points = points[valid]
    colors = colors[valid]

    iz = np.argsort(points[:, -1])
    points, colors = points[iz], colors[iz]
    px = np.int32(np.floor((points[:, 0] - bounds[0, 0]) / pixel_size))
    py = np.int32(np.floor((points[:, 1] - bounds[1, 0]) / pixel_size))
    px = np.clip(px, 0, width - 1)
    py = np.clip(py, 0, height - 1)
    heightmap[py, px] = points[:, 2] - bounds[2, 0]
    for c in range(colors.shape[-1]):
        colormap[py, px, c] = colors[:, c]
    return heightmap, colormap


def get_pointcloud(depth, intrinsics):
    """Point cloud (HxWx3, camera frame) from a perspective depth image and a 3x3 intrinsics matrix."""
    height, width = depth.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
    py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
    return np.float32([px, py, depth]).transpose(1, 2, 0)


def transform_pointcloud(points, transform):
    """Apply a 4x4 rigid transform to an HxWx3 point cloud (in place)."""
    padding = ((0, 0), (0, 0), (0, 1))
    homogen_points = np.pad(points.copy(), padding, "constant", constant_values=1)
    for i in range(3):
        points[..., i] = np.sum(transform[i, :] * homogen_points, axis=-1)
    return points


def reconstruct_heightmaps(color, depth, configs, bounds, pixel_size):
    """Reconstruct top-down heightmaps from one or more RGB-D views."""
    heightmaps, colormaps = [], []
    for color, depth, config in zip(color, depth, configs):
        intrinsics = np.array(config["intrinsics"]).reshape(3, 3)
        xyz = get_pointcloud(depth, intrinsics)
        position = np.array(config["position"]).reshape(3, 1)
        rotation = p.getMatrixFromQuaternion(config["rotation"])
        rotation = np.array(rotation).reshape(3, 3)
        transform = np.eye(4)
        transform[:3, :] = np.hstack((rotation, position))
        xyz = transform_pointcloud(xyz, transform)
        heightmap, colormap = get_heightmap(xyz, color, bounds, pixel_size)
        heightmaps.append(heightmap)
        colormaps.append(colormap)
    return heightmaps, colormaps


def pix_to_xyz(pixel, height, bounds, pixel_size, skip_height=False):
    """Heightmap pixel -> 3D position."""
    u, v = pixel
    x = bounds[0, 0] + v * pixel_size
    y = bounds[1, 0] + u * pixel_size
    z = 0.0 if skip_height else bounds[2, 0] + height[u, v]
    return (x, y, z)


def xyz_to_pix(position, bounds, pixel_size):
    """3D position -> heightmap pixel."""
    u = int(np.round((position[1] - bounds[1, 0]) / pixel_size))
    v = int(np.round((position[0] - bounds[0, 0]) / pixel_size))
    return (u, v)


def sample_distribution(prob, n_samples=1):
    """Sample pixel coordinates from an unnormalized 2D distribution."""
    flat_prob = prob.flatten() / np.sum(prob)
    rand_ind = np.random.choice(np.arange(len(flat_prob)), n_samples, p=flat_prob, replace=False)
    rand_ind_coords = np.array(np.unravel_index(rand_ind, prob.shape)).T
    return np.int32(rand_ind_coords.squeeze())


# ----------------------------------------------------------------------------- rigid transforms
def invert(pose):
    return p.invertTransform(pose[0], pose[1])


def multiply(pose0, pose1):
    return p.multiplyTransforms(pose0[0], pose0[1], pose1[0], pose1[1])


def apply(pose, position):
    position = np.float32(position)
    position_shape = position.shape
    position = np.float32(position).reshape(3, -1)
    rotation = np.float32(p.getMatrixFromQuaternion(pose[1])).reshape(3, 3)
    translation = np.float32(pose[0]).reshape(3, 1)
    position = rotation @ position + translation
    return tuple(position.reshape(position_shape))


def eulerXYZ_to_quatXYZW(rotation):
    """Ravens' euler (x, y, z) -> quaternion (x, y, z, w) conversion (static z-x-y axes)."""
    euler_zxy = (rotation[2], rotation[0], rotation[1])
    quaternion_wxyz = euler.euler2quat(*euler_zxy, axes="szxy")
    q = quaternion_wxyz
    return (q[1], q[2], q[3], q[0])


def quatXYZW_to_eulerXYZ(quaternion_xyzw):
    """Inverse of eulerXYZ_to_quatXYZW."""
    q = quaternion_xyzw
    quaternion_wxyz = np.array([q[3], q[0], q[1], q[2]])
    euler_zxy = euler.quat2euler(quaternion_wxyz, axes="szxy")
    return (euler_zxy[1], euler_zxy[2], euler_zxy[0])


def legacy_quat_mult(q1, q2):
    """Quaternion product as used by the original task code to compose box/block orientations.

    NOTE: it reads its (x, y, z, w) inputs as if they were (w, x, y, z). The result is not the
    rotational composition of q1 and q2, but the published scene distributions (and hence the
    datasets and success rates of the paper) depend on it, so it is kept verbatim.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return tuple([-x1 * x2 - y1 * y2 - z1 * z2 + w1 * w2,
                  x1 * w2 + y1 * z2 - z1 * y2 + w1 * x2,
                  -x1 * z2 + y1 * w2 + z1 * x2 + w1 * y2,
                  x1 * y2 - y1 * x2 + z1 * w2 + w1 * z2])


# ----------------------------------------------------------------------------- colors
COLORS = {
    "blue": [78.0 / 255.0, 121.0 / 255.0, 167.0 / 255.0],
    "red": [255.0 / 255.0, 87.0 / 255.0, 89.0 / 255.0],
    "green": [89.0 / 255.0, 169.0 / 255.0, 79.0 / 255.0],
    "orange": [242.0 / 255.0, 142.0 / 255.0, 43.0 / 255.0],
    "yellow": [237.0 / 255.0, 201.0 / 255.0, 72.0 / 255.0],
    "purple": [176.0 / 255.0, 122.0 / 255.0, 161.0 / 255.0],
    "pink": [255.0 / 255.0, 157.0 / 255.0, 167.0 / 255.0],
    "cyan": [118.0 / 255.0, 183.0 / 255.0, 178.0 / 255.0],
    "brown": [156.0 / 255.0, 117.0 / 255.0, 95.0 / 255.0],
    "gray": [186.0 / 255.0, 176.0 / 255.0, 172.0 / 255.0],
}


def color_name(rgb):
    """Inverse lookup in COLORS."""
    for name, value in COLORS.items():
        if value == rgb:
            return name
    raise KeyError(rgb)
