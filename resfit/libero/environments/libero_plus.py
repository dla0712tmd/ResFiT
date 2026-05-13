from __future__ import annotations

import logging
import os

import gymnasium as gym
import numpy as np
import torch

from resfit.libero.environments.libero import (
    LIBERO_STATE_KEYS,
    LIBERO_CONTROL_FREQ,
    SUITE_HORIZONS,
    LiberoGymWrapper,
    VectorizedLiberoEnvWrapper,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LIBERO_PLUS_TASK_SUITES = [
    "libero_spatial",   # LIBERO-Spatial
    "libero_object",    # LIBERO-Object
    "libero_goal",      # LIBERO-Goal
    "libero_10",        # LIBERO-Long
    "libero_90",        # LIBERO-90
]

# Mapping: LIBERO-plus robosuite camera name
# Sylvest/libero_plus_lerobot uses "front" and "wrist" instead of the standard
# LIBERO "image" and "wrist_image" key suffixes.
LIBERO_PLUS_CAMERA_TO_OBS_KEY: dict[str, str] = {
    "agentview": "front",
    "robot0_eye_in_hand": "wrist",
}

# Reverse mapping: dataset obs key suffix → LIBERO camera name (used for rendering)
LIBERO_PLUS_OBS_KEY_TO_CAMERA: dict[str, str] = {v: k for k, v in LIBERO_PLUS_CAMERA_TO_OBS_KEY.items()}

# Cameras to request from LIBERO (in order)
LIBERO_PLUS_CAMERAS: list[str] = list(LIBERO_PLUS_CAMERA_TO_OBS_KEY.keys())


# ---------------------------------------------------------------------------
# Single-environment wrapper
# ---------------------------------------------------------------------------


class LiberoPlusGymWrapper(LiberoGymWrapper):
    """LIBERO-plus variant of LiberoGymWrapper.

    Identical to the standard wrapper except for the camera observation key
    names, which match the ``Sylvest/libero_plus_lerobot`` dataset format:

    - ``observation.images.front``  (agentview;         standard LIBERO → ``image``)
    - ``observation.images.wrist``  (robot0_eye_in_hand; standard LIBERO → ``wrist_image``)

    State space (8-dim) and action space (7-dim) are unchanged.
    """

    def __init__(
        self,
        task_suite_name: str,
        task_id: int,
        camera_size: int = 256,
        render_size: int | tuple[int, int] | None = None,
        render_gpu_device_id: int = 0,
        env_id: int = 0,
    ):
        if task_suite_name not in LIBERO_PLUS_TASK_SUITES:
            raise ValueError(
                f"Unknown LIBERO-plus task suite: '{task_suite_name}'. "
                f"Available suites: {LIBERO_PLUS_TASK_SUITES}"
            )
        # Delegate to parent. Because _process_obs is overridden here, the
        # observation space built during _setup_spaces() will already use the
        # LIBERO-plus key names (Python MRO ensures the subclass method is called).
        super().__init__(
            task_suite_name=task_suite_name,
            task_id=task_id,
            camera_size=camera_size,
            render_size=render_size,
            render_gpu_device_id=render_gpu_device_id,
            env_id=env_id,
        )
        # Fix the image_obs_keys set by the parent (it used standard LIBERO names).
        self.image_obs_keys = [
            f"observation.images.{LIBERO_PLUS_CAMERA_TO_OBS_KEY[c]}"
            for c in self.camera_names
        ]

    # ------------------------------------------------------------------
    # Observation processing — LIBERO-plus key names
    # ------------------------------------------------------------------

    def _process_obs(self, obs: dict) -> dict:
        processed: dict[str, np.ndarray] = {}

        # Proprioceptive state: (8,) — unchanged from standard LIBERO.
        state_parts = []
        for key, n_dims in LIBERO_STATE_KEYS:
            if key in obs:
                v = obs[key]
                if v.ndim == 0:
                    v = np.array([v])
                state_parts.append(v[:n_dims] if n_dims is not None else v)
            else:
                logger.debug(f"Expected state key '{key}' not found in obs")

        if state_parts:
            processed["observation.state"] = np.concatenate(state_parts).astype(np.float32)

        # Camera images with LIBERO-plus specific obs-key suffixes.
        for cam in self.camera_names:
            robosuite_key = f"{cam}_image"
            if robosuite_key in obs:
                img = obs[robosuite_key].astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))          # (H,W,C) → (C,H,W)
                obs_key = LIBERO_PLUS_CAMERA_TO_OBS_KEY[cam]  # "front" | "wrist"
                processed[f"observation.images.{obs_key}"] = img

        self._last_obs = processed

        if not hasattr(self, "_logged_obs_keys"):
            logger.debug(f"Observation keys: {list(processed.keys())}")
            self._logged_obs_keys = True

        return processed

    # ------------------------------------------------------------------
    # Rendering — resolve video key via LIBERO-plus camera map
    # ------------------------------------------------------------------

    def render(self) -> np.ndarray:
        camera_name = "agentview"  # sensible default
        if getattr(self, "video_key", None):
            key = self.video_key
            suffix = key.split(".")[-1] if "." in key else key
            camera_name = LIBERO_PLUS_OBS_KEY_TO_CAMERA.get(suffix, suffix)

        frame = self.env.sim.render(
            camera_name=camera_name,
            height=self.render_size[0],
            width=self.render_size[1],
        )[::-1]  # MuJoCo images are bottom-up; flip to top-down

        return frame


# ---------------------------------------------------------------------------
# Vectorised environment helpers
# ---------------------------------------------------------------------------


def make_libero_plus_env(
    task_suite_name: str,
    task_id: int,
    camera_size: int = 256,
    render_size: int | tuple[int, int] | None = None,
    render_gpu_device_id: int = 0,
    env_id: int = 0,
):
    """Return a no-argument factory that creates one :class:`LiberoPlusGymWrapper`."""

    def _make() -> LiberoPlusGymWrapper:
        return LiberoPlusGymWrapper(
            task_suite_name=task_suite_name,
            task_id=task_id,
            camera_size=camera_size,
            render_size=render_size,
            render_gpu_device_id=render_gpu_device_id,
            env_id=env_id,
        )

    return _make


def create_vectorized_libero_plus_env(
    task_suite_name: str,
    task_id: int,
    num_envs: int,
    device: str = "cpu",
    camera_size: int = 256,
    render_size: int | tuple[int, int] | None = None,
    debug: bool = False,
    video_key: str = "observation.images.front",
) -> VectorizedLiberoEnvWrapper:
    """Create a vectorised LIBERO-plus environment.

    Args:
        task_suite_name: One of the LIBERO-plus task suite names
            (``libero_spatial``, ``libero_object``, ``libero_goal``, ``libero_10``).
        task_id: Zero-indexed task ID within the suite (0–9).
        num_envs: Number of parallel environment instances.
        device: PyTorch device string for tensor conversion (e.g. ``"cuda"``).
        camera_size: Pixel resolution (height = width) for policy cameras.
        render_size: Resolution for video recording frames.  ``None`` uses
            (240, 320).  An integer expands to (render_size, render_size).
        debug: If ``True`` uses ``SyncVectorEnv`` (sequential, easier to debug).
            Otherwise uses ``AsyncVectorEnv`` (parallel, faster).
        video_key: Observation key whose camera is used for video recording.
            Defaults to ``"observation.images.front"`` (the LIBERO-plus agentview).

    Returns:
        A :class:`VectorizedLiberoEnvWrapper` wrapping the vector env.
    """
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None:
        visible_ids = [int(x) for x in cuda_visible.split(",") if x.strip()]
    else:
        visible_ids = list(range(max(torch.cuda.device_count(), 1)))

    num_gpus = len(visible_ids) if visible_ids else 1

    env_fns = [
        make_libero_plus_env(
            task_suite_name=task_suite_name,
            task_id=task_id,
            camera_size=camera_size,
            render_size=render_size,
            render_gpu_device_id=visible_ids[env_id % num_gpus] if visible_ids else 0,
            env_id=env_id,
        )
        for env_id in range(num_envs)
    ]

    if debug:
        logger.debug("Debug mode: using gymnasium.vector.SyncVectorEnv")
        vec_env = gym.vector.SyncVectorEnv(
            env_fns,
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
        )
    else:
        logger.debug("Production mode: using gymnasium.vector.AsyncVectorEnv")
        vec_env = gym.vector.AsyncVectorEnv(
            env_fns,
            shared_memory=True,
            copy=True,
            context="spawn",
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
        )

    vec_env.call("set_wrapper_attr", "video_key", video_key)

    wrapped = VectorizedLiberoEnvWrapper(vec_env, video_key, device)

    wrapped.env_name = f"{task_suite_name}/{task_id}"
    wrapped.camera_size = camera_size
    wrapped.render_size = render_size

    logger.info(
        f"Created {num_envs} vectorized LIBERO-plus envs "
        f"[{task_suite_name}/task_{task_id}] | camera_size={camera_size}"
    )
    return wrapped
