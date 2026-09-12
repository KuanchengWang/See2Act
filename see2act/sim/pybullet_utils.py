"""PyBullet helpers."""


def load_urdf(pybullet_client, file_path, *args, **kwargs):
    """Load a URDF, returning None (instead of raising) when PyBullet fails."""
    try:
        return pybullet_client.loadURDF(file_path, *args, **kwargs)
    except pybullet_client.error:
        return None
