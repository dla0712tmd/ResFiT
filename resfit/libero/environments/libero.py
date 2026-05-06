from __future__ import annotations

import logging
import os

import gymnasium as gym
import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LIBERO_TASK_SUITES = [
    "libero_spatial",   # LIBERO-Spatial  (10 tasks, short-horizon)
    "libero_object",    # LIBERO-Object   (10 tasks, short-horizon)
    "libero_goal",      # LIBERO-Goal     (10 tasks, short-horizon)
    "libero_10",        # LIBERO-Long     (10 tasks, long-horizon)  ← was 'libero_long'
    "libero_90",        # LIBERO-90       (90 tasks)
    "libero_100",       # LIBERO-100      (100 tasks)
]

# Mapping: LIBERO robosuite camera name → LeRobot dataset observation key suffix
# (matches lerobot/libero_*_image datasets on HuggingFace Hub)
LIBERO_CAMERA_TO_OBS_KEY: dict[str, str] = {
    "agentview": "image",
    "robot0_eye_in_hand": "wrist_image",
}

# Reverse mapping: dataset obs key suffix → LIBERO camera name (used for rendering)
LIBERO_OBS_KEY_TO_CAMERA: dict[str, str] = {v: k for k, v in LIBERO_CAMERA_TO_OBS_KEY.items()}

# Cameras to request from LIBERO (in order)
LIBERO_CAMERAS: list[str] = list(LIBERO_CAMERA_TO_OBS_KEY.keys())

# Low-dimensional state keys extracted from robosuite obs dict.
# Produces shape (8,) = eef_pos(3) + eef_quat(4) + gripper[:1](1),
# matching the 'observation.state' in lerobot/libero_*_image datasets.
LIBERO_STATE_KEYS = [
    ("robot0_eef_pos", None),       # (3,) → all dims
    ("robot0_eef_quat", None),      # (4,) → all dims
    ("robot0_gripper_qpos", 1),     # (2,) → first dim only → (1,)
]

# Episode horizon per suite at 10 Hz (dataset FPS)
SUITE_HORIZONS: dict[str, int] = {
    "libero_spatial": 300,
    "libero_object": 300,
    "libero_goal": 300,
    "libero_10": 600,    # LIBERO-Long tasks
    "libero_90": 600,
    "libero_100": 600,
}

# Control frequency that matches lerobot/libero_*_image datasets (10 Hz)
LIBERO_CONTROL_FREQ = 10


# ---------------------------------------------------------------------------
# Single-environment wrapper
# ---------------------------------------------------------------------------


class LiberoGymWrapper:
    """Gymnasium-compatible wrapper around a LIBERO ``OffScreenRenderEnv``.

    Observation format mirrors the ``lerobot/libero_*_image`` datasets on HF Hub:
    - ``observation.state``                    float32 (8,)
        = [eef_pos(3), eef_quat(4), gripper(1)]
    - ``observation.images.image``             float32 (3, H, W)  in [0, 1]
        (from agentview camera)
    - ``observation.images.wrist_image``       float32 (3, H, W)  in [0, 1]
        (from robot0_eye_in_hand camera)

    Action space: 7-dim delta EE pose + gripper (x, y, z, roll, pitch, yaw, gripper).
    Control frequency: 10 Hz (matches the dataset FPS).
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
        if task_suite_name not in LIBERO_TASK_SUITES:
            raise ValueError(
                f"Unknown LIBERO task suite: '{task_suite_name}'. "
                f"Available suites: {LIBERO_TASK_SUITES}"
            )

        self.task_suite_name = task_suite_name
        self.task_id = task_id
        self.camera_size = camera_size
        self.render_gpu_device_id = render_gpu_device_id
        self.env_id = env_id

        if isinstance(render_size, int):
            self.render_size = (render_size, render_size)
        elif render_size is None:
            self.render_size = (240, 320)
        else:
            self.render_size = tuple(render_size)

        horizon = SUITE_HORIZONS.get(task_suite_name, 600)

        # Gymnasium-required metadata
        self.metadata = {
            "render_modes": ["rgb_array"],
            "render_fps": LIBERO_CONTROL_FREQ,
            "horizon": horizon,
        }
        self.spec = None
        self.render_mode = "rgb_array"
        self.video_key = None
        self.episode_steps = 0

        # ------------------------------------------------------------------
        # Load the LIBERO task
        # ------------------------------------------------------------------
        from libero.libero import benchmark as libero_benchmark  # noqa: PLC0415

        benchmark_dict = libero_benchmark.get_benchmark_dict()
        if task_suite_name not in benchmark_dict:
            raise ValueError(
                f"LIBERO benchmark '{task_suite_name}' not found. "
                f"Available: {list(benchmark_dict.keys())}"
            )
        task_suite = benchmark_dict[task_suite_name]()
        tasks = task_suite.tasks

        if task_id >= len(tasks):
            raise ValueError(
                f"Task ID {task_id} is out of range for suite '{task_suite_name}' "
                f"which has {len(tasks)} task(s) (0–{len(tasks) - 1})."
            )

        self.task = tasks[task_id]
        self.task_name: str = self.task.name
        self.task_language: str = getattr(self.task, "language", "")

        # Build the full BDDL file path:
        #   <bddl_root>/<problem_folder>/<bddl_file>
        # e.g. ~/.libero/bddl_files/libero_spatial/pick_up_...bddl
        from libero.libero import get_libero_path  # noqa: PLC0415
        bddl_root = get_libero_path("bddl_files")
        self.task_bddl_file: str = os.path.join(
            bddl_root,
            self.task.problem_folder,
            self.task.bddl_file,
        )

        if env_id == 0:
            logger.info(f"LIBERO task [{task_suite_name}/{task_id}]: {self.task_name}")
            if self.task_language:
                logger.info(f"  Language instruction: {self.task_language}")
            logger.debug(f"  BDDL file: {self.task_bddl_file}")

        # Camera names used to query LIBERO, and their output obs key suffixes
        self.camera_names: list[str] = list(LIBERO_CAMERAS)
        self.image_obs_keys: list[str] = [
            f"observation.images.{LIBERO_CAMERA_TO_OBS_KEY[c]}" for c in self.camera_names
        ]

        # MuJoCo renderer device
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(render_gpu_device_id)

        # ------------------------------------------------------------------
        # Create LIBERO environment
        # ------------------------------------------------------------------
        from libero.libero.envs import OffScreenRenderEnv  # noqa: PLC0415

        env_args = {
            "bddl_file_name": self.task_bddl_file,
            "camera_heights": camera_size,
            "camera_widths": camera_size,
            "camera_names": self.camera_names,
            "has_offscreen_renderer": True,
            "use_camera_obs": True,
            "control_freq": LIBERO_CONTROL_FREQ,  # 10 Hz to match dataset FPS
            "render_gpu_device_id": render_gpu_device_id,
            "ignore_done": False,
            "horizon": horizon,
        }

        self.env = OffScreenRenderEnv(**env_args)

        self._setup_spaces()

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def _setup_spaces(self):
        low, high = self.env.env.action_spec
        self.action_space = gym.spaces.Box(
            low=low.astype(np.float32), high=high.astype(np.float32), dtype=np.float32
        )

        # Sample one observation to infer shapes
        sample_obs_raw = self.env.reset()
        sample_obs = self._process_obs(sample_obs_raw)

        obs_spaces = {}
        for key, value in sample_obs.items():
            obs_spaces[key] = gym.spaces.Box(
                low=-np.inf if "state" in key else 0.0,
                high=np.inf if "state" in key else 1.0,
                shape=value.shape,
                dtype=value.dtype,
            )
        self.observation_space = gym.spaces.Dict(obs_spaces)

    def seed(self, seed=None):
        return [seed]

    def reset(self, *, seed=None, options=None):
        obs = self.env.reset()
        processed_obs = self._process_obs(obs)
        self._last_obs = processed_obs
        self.episode_steps = 0
        return processed_obs, {}

    def step(self, action):
        if hasattr(action, "cpu"):
            action = action.cpu().numpy()
        if action.ndim > 1:
            action = action[0]

        # LIBERO (Robosuite) returns 4 values; convert to Gymnasium's 5-value API.
        obs, reward, done, info = self.env.step(action)
        self.episode_steps += 1

        processed_obs = self._process_obs(obs)

        reward_scalar = float(reward)
        # LIBERO overrides `done` with _check_success() only — so done=True means
        # success, not horizon expiry. Check robosuite's internal done flag separately.
        success = bool(done) or reward_scalar == 1.0
        horizon_done = bool(self.env.env.done)  # robosuite's timestep >= horizon flag
        terminated_scalar = success
        truncated_scalar = horizon_done and not success

        if terminated_scalar or truncated_scalar:
            info = {
                **info,
                "success": success,
                "episode_steps": self.episode_steps,
            }
            self.episode_steps = 0

        return processed_obs, reward_scalar, terminated_scalar, truncated_scalar, info

    # ------------------------------------------------------------------
    # Observation processing
    # ------------------------------------------------------------------

    def _process_obs(self, obs: dict) -> dict:
        processed: dict[str, np.ndarray] = {}

        # Proprioceptive state: (8,) = eef_pos(3) + eef_quat(4) + gripper[:1](1)
        # Matches 'observation.state' shape in lerobot/libero_*_image datasets.
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

        # Camera images:  (H, W, C) uint8  →  (C, H, W) float32 in [0, 1]
        # Camera names are mapped to dataset observation key suffixes.
        for cam in self.camera_names:
            robosuite_key = f"{cam}_image"
            if robosuite_key in obs:
                img = obs[robosuite_key].astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))
                obs_key = LIBERO_CAMERA_TO_OBS_KEY[cam]  # e.g. agentview → image
                processed[f"observation.images.{obs_key}"] = img

        self._last_obs = processed

        if not hasattr(self, "_logged_obs_keys"):
            logger.debug(f"Observation keys: {list(processed.keys())}")
            self._logged_obs_keys = True

        return processed

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> np.ndarray:
        """Return an RGB frame (H, W, 3) uint8 suitable for video recording.

        The ``video_key`` (e.g. ``"observation.images.image"``) is resolved to
        the corresponding LIBERO camera name via ``LIBERO_OBS_KEY_TO_CAMERA``.
        Falls back to ``agentview`` for unknown keys.
        """
        camera_name = "agentview"  # default
        if getattr(self, "video_key", None):
            key = self.video_key
            # Extract the last component: "observation.images.image" → "image"
            suffix = key.split(".")[-1] if "." in key else key
            # Map dataset obs suffix back to a LIBERO camera name
            camera_name = LIBERO_OBS_KEY_TO_CAMERA.get(suffix, suffix)

        frame = self.env.sim.render(
            camera_name=camera_name,
            height=self.render_size[0],
            width=self.render_size[1],
        )[::-1]  # MuJoCo images are bottom-up; flip to top-down

        return frame

    # ------------------------------------------------------------------
    # Gymnasium vectorized-env compatibility helpers
    # ------------------------------------------------------------------

    def set_video_key(self, video_key: str):
        self.video_key = video_key
        logging.getLogger("gymnasium.vector").setLevel(logging.ERROR)

    def close(self):
        self.env.close()

    @property
    def unwrapped(self):
        return self

    def get_wrapper_attr(self, name: str):
        """Walk-through for Gymnasium's AsyncVectorEnv worker protocol."""
        if hasattr(self, name):
            return getattr(self, name)
        raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}")

    def set_wrapper_attr(self, name: str, value):
        setattr(self, name, value)


# ---------------------------------------------------------------------------
# Vectorised environment helpers
# ---------------------------------------------------------------------------


def make_libero_env(
    task_suite_name: str,
    task_id: int,
    camera_size: int = 84,
    render_size: int | tuple[int, int] | None = None,
    render_gpu_device_id: int = 0,
    env_id: int = 0,
):
    """Return a factory callable (no-argument) that creates one LiberoGymWrapper."""

    def _make() -> LiberoGymWrapper:
        return LiberoGymWrapper(
            task_suite_name=task_suite_name,
            task_id=task_id,
            camera_size=camera_size,
            render_size=render_size,
            render_gpu_device_id=render_gpu_device_id,
            env_id=env_id,
        )

    return _make


class VectorizedLiberoEnvWrapper:
    """Thin wrapper around a Gymnasium vector env that adds rendering and
    converts observations to PyTorch tensors on the requested device."""

    def __init__(
        self,
        vec_env: gym.vector.SyncVectorEnv | gym.vector.AsyncVectorEnv,
        video_key: str,
        device: str = "cpu",
    ):
        self.vec_env = vec_env
        self.video_key = video_key
        self._last_obs = None
        self.device = device

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        obs, info = self.vec_env.reset(**kwargs)
        self._last_obs = obs
        obs = self._to_torch(obs)
        return obs, info

    def step(self, actions):
        obs, rewards, terminated, truncated, info = self.vec_env.step(actions)
        self._last_obs = obs

        obs = self._to_torch(obs)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        terminated = torch.tensor(terminated, device=self.device, dtype=torch.bool)
        truncated = torch.tensor(truncated, device=self.device, dtype=torch.bool)

        return obs, rewards, terminated, truncated, info

    def render(self) -> np.ndarray:
        """Return stacked RGB frames of shape (num_envs, H, W, 3) uint8."""
        frames = self.vec_env.render()
        if frames is None:
            raise RuntimeError("No frames returned from the vectorized environment")
        return frames

    def close(self):
        return self.vec_env.close()

    # ------------------------------------------------------------------
    # Properties / delegation
    # ------------------------------------------------------------------

    @property
    def num_envs(self) -> int:
        return self.vec_env.num_envs

    @property
    def fps(self) -> int:
        return self.vec_env.metadata["render_fps"]

    def __getattr__(self, name: str):
        return getattr(self.vec_env, name)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_torch(self, obs):
        if isinstance(obs, dict):
            return {
                k: torch.from_numpy(v).to(self.device) if isinstance(v, np.ndarray) else v
                for k, v in obs.items()
            }
        if isinstance(obs, np.ndarray):
            return torch.from_numpy(obs).to(self.device)
        return obs


def create_vectorized_libero_env(
    task_suite_name: str,
    task_id: int,
    num_envs: int,
    device: str = "cpu",
    camera_size: int = 84,
    render_size: int | tuple[int, int] | None = None,
    debug: bool = False,
    video_key: str = "observation.images.agentview",
) -> VectorizedLiberoEnvWrapper:
    """Create a vectorised LIBERO environment.

    Args:
        task_suite_name: One of the LIBERO task suite names
            (``libero_spatial``, ``libero_object``, ``libero_goal``,
            ``libero_long``, ``libero_10``).
        task_id: Zero-indexed task ID within the suite (0–9 for most suites).
        num_envs: Number of parallel environment instances.
        device: PyTorch device string for tensor conversion (e.g. ``"cuda"``).
        camera_size: Pixel resolution (height = width) for policy cameras.
        render_size: Resolution for video recording frames. ``None`` uses
            (240, 320). An integer uses (render_size, render_size).
        debug: If ``True``, uses ``SyncVectorEnv`` (sequential, easier to
            debug). Otherwise uses ``AsyncVectorEnv`` (parallel, faster).
        video_key: Observation key whose camera is used for video recording.

    Returns:
        A :class:`VectorizedLiberoEnvWrapper` wrapping the vector env.
    """
    # Determine per-environment GPU assignment
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None:
        visible_ids = [int(x) for x in cuda_visible.split(",") if x.strip()]
    else:
        visible_ids = list(range(max(torch.cuda.device_count(), 1)))

    num_gpus = len(visible_ids) if visible_ids else 1

    env_fns = [
        make_libero_env(
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

    # Propagate the video camera key to every worker
    vec_env.call("set_wrapper_attr", "video_key", video_key)

    wrapped = VectorizedLiberoEnvWrapper(vec_env, video_key, device)

    # Attach metadata for later retrieval
    wrapped.env_name = f"{task_suite_name}/{task_id}"
    wrapped.camera_size = camera_size
    wrapped.render_size = render_size

    logger.info(
        f"Created {num_envs} vectorized LIBERO envs "
        f"[{task_suite_name}/task_{task_id}] | camera_size={camera_size}"
    )
    return wrapped
