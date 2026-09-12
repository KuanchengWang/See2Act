"""See2Act: Learning to See While Learning to Act (Wang et al., 2026).

Diffusion policy whose action denoising is coupled with viewpoint refinement: at every denoising step the
camera is moved along a schedule computed from the current action estimate, and the next denoising step
conditions on the newly rendered view.
"""
__version__ = "1.0.0"

from see2act.see2act import See2Act  # noqa: E402,F401
