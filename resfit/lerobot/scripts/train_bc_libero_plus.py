#!/usr/bin/env python
"""Train a LeRobot policy on the LIBERO-plus dataset (lerobot/libero_plus).

LIBERO-plus is a robustness benchmark that extends the four standard LIBERO
task suites (spatial, object, goal, long) with ~10 000 perturbation variants
covering object layout, camera viewpoints, robot initial states, language
instructions, lighting, textures, and sensor noise.
(HuggingFace PR: huggingface/lerobot#3313)

Key differences from train_bc_libero.py
========================================
- Dataset:  ``lerobot/libero_plus`` (v3.0 format, auto-downloaded by lerobot)
- Camera observation keys: ``observation.images.front`` / ``observation.images.wrist``
  (standard LIBERO uses ``observation.images.image`` / ``observation.images.wrist_image``)
- State and action spaces are identical to standard LIBERO (8-dim state, 7-dim action).
- Supported eval suites: libero_spatial, libero_object, libero_goal, libero_10, libero_90.

Assets pre-requisite
====================
The LIBERO-plus simulation requires additional objects, textures, and init-states
that are not part of the standard LIBERO package.  Run setup_libero_plus.sh once
to install them before running evaluations.  Training on the dataset alone does
not require the assets.

Example
=======
python -m resfit.lerobot.scripts.train_bc_libero_plus \\
    --dataset lerobot/libero_plus \\
    --policy act \\
    --wandb_enable --wandb_project libero-plus-runs
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import multiprocessing as mp
import os
import re
import shutil
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import torch
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.transforms import ImageTransforms, ImageTransformsConfig
from lerobot.datasets.utils import cycle
from lerobot.utils.random_utils import set_seed
from PIL import Image, ImageDraw, ImageFont
from termcolor import colored

import wandb
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.policies.factory import make_policy, make_policy_config
from resfit.libero.environments.libero_plus import (
    LIBERO_PLUS_TASK_SUITES,
    VectorizedLiberoEnvWrapper,
    create_multi_task_libero_plus_env,
    create_vectorized_libero_plus_env,
)
from resfit.lerobot.policies.vec_env_policy import VecEnvPolicy
from resfit.lerobot.utils.load_policy import load_checkpoint, save_checkpoint

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

_CACHE_ROOT = Path(os.environ.get("CACHE_DIR", ".")).expanduser().resolve()


parser = argparse.ArgumentParser(
    description="Offline BC training on LIBERO-plus (Sylvest/libero_plus_lerobot)"
)

# Required args
parser.add_argument(
    "--dataset",
    type=str,
    default="lerobot/libero_plus",
    help="HF Hub dataset repo-id or local path (default: lerobot/libero_plus)",
)
parser.add_argument(
    "--policy",
    type=str,
    default="diffusion",
    choices=[
        "diffusion",
        "act",
        "latent_act",
        "pi0",
        "pi0fast",
        "tdmpc",
        "vqbet",
        "smolvla",
    ],
    help="Which policy architecture to train",
)

# Training hyper-parameters
parser.add_argument("--steps", type=int, default=100_000, help="Total optimisation steps")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--grad_clip_norm", type=float, default=10.0)
parser.add_argument("--num_workers", type=int, default=4)

# Reproducibility / device
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

# Logging & checkpoints
parser.add_argument("--output_dir", type=str, default="outputs/train_libero_plus")
parser.add_argument("--log_freq", type=int, default=100)
parser.add_argument("--save_freq", type=int, default=10_000)

# Weights & Biases
parser.add_argument("--wandb_enable", action="store_true")
parser.add_argument("--wandb_project", type=str, default=None)
parser.add_argument("--wandb_entity", type=str, default=None)

# Resume
parser.add_argument("--resume_ckpt", type=str, default=None)
parser.add_argument("--resume_run_id", type=str, default=None)

# ------------------------------------------------------------------
# Evaluation rollouts
# ------------------------------------------------------------------
parser.add_argument(
    "--rollout_freq",
    type=int,
    default=None,
    help="Frequency (in steps) at which to run evaluation rollouts. Disabled when not set.",
)
parser.add_argument(
    "--eval_suite",
    type=str,
    default=None,
    choices=LIBERO_PLUS_TASK_SUITES,
    help="LIBERO-plus task suite used for evaluation rollouts.",
)
parser.add_argument("--eval_num_envs", type=int, default=8)
parser.add_argument("--eval_camera_size", type=int, default=256)
parser.add_argument("--eval_render_size", type=int, default=None)
parser.add_argument(
    "--eval_video_key",
    type=str,
    default="observation.images.front",
    help=(
        "Observation key for the camera used for video recording. "
        "LIBERO-plus datasets use 'observation.images.front' (agentview) "
        "or 'observation.images.wrist' (wrist camera)."
    ),
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="Use SyncVectorEnv instead of AsyncVectorEnv (easier to debug).",
)

# -------------------------------------------------
# Policy configuration overrides
# -------------------------------------------------
parser.add_argument(
    "--policy_kwargs",
    type=str,
    default=None,
    help=(
        "Overrides for the policy configuration. Accepts either: "
        '1) A JSON dictionary string, e.g. \'{"dim_model": 1024, "chunk_size": 100}\', or '
        "2) A compact 'key=value' list separated by commas or spaces."
    ),
)

# Camera selection
parser.add_argument(
    "--policy_cameras",
    type=str,
    nargs="*",
    default=None,
    help=(
        "List of camera names to use for the policy. "
        "LIBERO-plus cameras are: front, wrist. "
        "Example: --policy_cameras front wrist"
    ),
)

# Proprioceptive observations
parser.add_argument(
    "--disable_proprioceptive_obs",
    action="store_true",
    help="Disable observation.state during training (vision-only).",
)

args_cli = parser.parse_args()

# ---------------------------------------------------------------------------
# Setup logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _get_suite_num_tasks(suite_name: str) -> int:
    from libero.libero import benchmark as libero_benchmark
    benchmark_dict = libero_benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[suite_name]()
    return len(task_suite.tasks)


def _load_task_info(suite_name: str) -> tuple[dict[int, str], dict[int, str]]:
    """Return (task_id -> category, task_id -> language_instruction) for a suite.

    Matches benchmark task ordering (0-indexed) to the task_classification.json
    entries by task name.
    """
    from pathlib import Path as _Path
    import libero.libero.benchmark as _bench_pkg
    from libero.libero import benchmark as libero_benchmark

    json_path = _Path(_bench_pkg.__file__).parent / "task_classification.json"
    with open(json_path) as f:
        classification = json.load(f)

    suite_entries = classification.get(suite_name, [])
    if not suite_entries:
        raise ValueError(f"No classification data for suite '{suite_name}' in task_classification.json")
    name_to_category: dict[str, str] = {e["name"]: e["category"] for e in suite_entries}

    benchmark_dict = libero_benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[suite_name]()
    tasks = task_suite.tasks

    task_id_to_category: dict[int, str] = {}
    task_id_to_language: dict[int, str] = {}
    for task_id, task in enumerate(tasks):
        cat = name_to_category.get(task.name)
        if cat is None:
            logger.warning(f"Task '{task.name}' not found in classification JSON; labelled 'Unknown'")
            cat = "Unknown"
        task_id_to_category[task_id] = cat
        task_id_to_language[task_id] = getattr(task, "language", "")

    return task_id_to_category, task_id_to_language


def _run_eval(
    *,
    policy: VecEnvPolicy,
    eval_suite: str,
    num_envs: int,
    device_str: str,
    camera_size: int,
    render_size: int | None,
    video_key: str,
    save_dir: Path,
    step: int,
    run_start_time: str,
    debug: bool,
    smolvla_tokenizer=None,
) -> tuple[dict[str, float], float, Path | None]:
    """Evaluate 1 episode per task, all tasks, results by perturbation category.

    Batches ``num_envs`` tasks at a time (each sub-env runs a *different* task).
    Returns (category_success_rates, overall_success_rate, video_path).
    """
    task_category_map, task_language_map = _load_task_info(eval_suite)
    num_tasks = len(task_category_map)

    logger.info(colored(
        f"Paper eval [{eval_suite}]: {num_tasks} tasks × 1 episode, {num_envs} tasks in parallel",
        "cyan",
    ))

    policy_was_training = policy.training
    policy.eval()
    device = torch.device(device_str)

    task_success: dict[int, bool] = {}
    video_path: Path | None = None
    video_writer = None
    start_time = time.perf_counter()

    all_task_ids = list(range(num_tasks))
    num_batches = (num_tasks + num_envs - 1) // num_envs

    for batch_idx, batch_start in enumerate(range(0, num_tasks, num_envs)):
        batch_task_ids = all_task_ids[batch_start : batch_start + num_envs]
        actual_n = len(batch_task_ids)

        env = create_multi_task_libero_plus_env(
            task_suite_name=eval_suite,
            task_ids=batch_task_ids,
            device=device_str,
            camera_size=camera_size,
            render_size=render_size,
            video_key=video_key,
            debug=debug,
        )

        # Record a single video from the very first batch (first sub-env only)
        if batch_idx == 0:
            try:
                now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                parent_dir = save_dir / f"eval_{eval_suite}" / run_start_time
                parent_dir.mkdir(parents=True, exist_ok=True)
                video_path = parent_dir / f"eval_step_{step}_{now}.mp4"
                video_writer = imageio.get_writer(video_path.as_posix(), fps=20)
            except Exception as e:
                logger.warning(f"Could not open video writer: {e}")
                video_writer = None

        obs, _ = env.reset()
        done_flags = torch.zeros(actual_n, dtype=torch.bool, device=device)
        success_flags = torch.zeros(actual_n, dtype=torch.bool, device=device)
        policy.reset()

        while not done_flags.all():
            with torch.inference_mode():
                if smolvla_tokenizer is not None:
                    task_strs = [task_language_map.get(tid, "") + "\n" for tid in batch_task_ids]
                    tokenized = smolvla_tokenizer(
                        task_strs,
                        return_tensors="pt",
                        padding="longest",
                        truncation=True,
                        max_length=48,
                    )
                    obs[OBS_LANGUAGE_TOKENS] = tokenized["input_ids"].to(device)
                    obs[OBS_LANGUAGE_ATTENTION_MASK] = tokenized["attention_mask"].bool().to(device)
                action = policy.select_action(obs)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated
            newly_done = done & ~done_flags

            if video_writer is not None and batch_idx == 0:
                try:
                    frames = env.render()
                    video_writer.append_data(frames[0])
                except Exception:
                    pass

            if newly_done.any():
                success_flags |= newly_done & (reward >= 1.0)
                done_flags |= newly_done
                policy.reset(env_ids=torch.where(newly_done)[0])

        if video_writer is not None and batch_idx == 0:
            video_writer.close()
            video_writer = None

        env.close()

        for i, task_id in enumerate(batch_task_ids):
            task_success[task_id] = bool(success_flags[i].item())

        if (batch_idx + 1) % max(1, num_batches // 10) == 0 or batch_idx == num_batches - 1:
            elapsed = time.perf_counter() - start_time
            done_tasks = min(batch_start + num_envs, num_tasks)
            cur_rate = sum(task_success.values()) / len(task_success)
            logger.info(
                f"  [{eval_suite}] {done_tasks}/{num_tasks} tasks "
                f"| running SR: {cur_rate * 100:.1f}% | {elapsed:.0f}s"
            )

    # Aggregate by perturbation category
    category_successes: dict[str, list[bool]] = {}
    for task_id, success in task_success.items():
        cat = task_category_map.get(task_id, "Unknown")
        category_successes.setdefault(cat, []).append(success)

    category_rates = {
        cat: sum(vals) / len(vals) for cat, vals in sorted(category_successes.items())
    }
    overall_rate = sum(task_success.values()) / len(task_success) if task_success else 0.0

    total_elapsed = time.perf_counter() - start_time
    logger.info(colored(
        f"Eval done: {overall_rate * 100:.1f}% overall | "
        f"{num_tasks} tasks in {total_elapsed:.1f}s",
        "cyan",
    ))
    for cat, rate in category_rates.items():
        logger.info(f"  {cat}: {rate * 100:.1f}%")

    if policy_was_training:
        policy.train()

    return category_rates, overall_rate, video_path


def _annotate_frame(
    frame: np.ndarray,
    env_idx: int,
    episode_num: int,
    total_episodes: int,
    episode_step: int,
    is_success: bool,
    font=None,
) -> np.ndarray:
    pil_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_img)

    episode_text = f"Env {env_idx + 1} | Episode {episode_num}/{total_episodes}"
    step_text = f"Step {episode_step}"
    status_text = "SUCCESS" if is_success else "FAIL"
    status_color = (0, 255, 0) if is_success else (255, 0, 0)

    y_offset = 10
    draw.text((10, y_offset), episode_text, fill=(255, 255, 255), font=font)
    y_offset += 15
    draw.text((10, y_offset), step_text, fill=(255, 255, 255), font=font)
    y_offset += 15
    draw.text((10, y_offset), status_text, fill=status_color, font=font)

    return np.array(pil_img)


def _run_rollouts(
    *,
    policy: VecEnvPolicy,
    env: VectorizedLiberoEnvWrapper,
    save_dir: Path,
    step: int,
    num_episodes: int,
    run_start_time: str,
    eval_suite: str,
    eval_task_id: int,
    smolvla_tokenizer=None,
    smolvla_lang_instruction: str | None = None,
):
    """Run *num_episodes* episodes and return (success_rate, video_path, fps)."""

    save_dir.mkdir(parents=True, exist_ok=True)

    policy_was_training = policy.training
    policy.eval()

    num_parallel_envs = env.num_envs
    env_label = getattr(env, "env_name", f"{eval_suite}/task_{eval_task_id}")

    successes = 0
    done_episodes = 0
    total_steps = 0

    start_time = time.perf_counter()

    logger.info(f"Running rollouts: {num_episodes} episodes, {num_parallel_envs} envs | {env_label}")

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
    except Exception:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parent_dir = save_dir / f"eval_{eval_suite}_task{eval_task_id}" / run_start_time
    parent_dir.mkdir(parents=True, exist_ok=True)
    video_path = parent_dir / f"eval_step_{step}_{now}.mp4"

    video_writer = imageio.get_writer(video_path.as_posix(), fps=20)

    obs, _ = env.reset()
    episode_frames = [[] for _ in range(num_parallel_envs)]
    episode_steps = [0] * num_parallel_envs

    while done_episodes < num_episodes:
        with torch.inference_mode():
            if smolvla_tokenizer is not None and smolvla_lang_instruction is not None:
                task_strs = [smolvla_lang_instruction + "\n"] * env.num_envs
                tokenized = smolvla_tokenizer(
                    task_strs,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=48,
                )
                obs[OBS_LANGUAGE_TOKENS] = tokenized["input_ids"].to(env.device)
                obs[OBS_LANGUAGE_ATTENTION_MASK] = tokenized["attention_mask"].bool().to(env.device)
            action = policy.select_action(obs)

        obs, reward, terminated, truncated, info = env.step(action)

        frames = env.render()
        for env_idx in range(num_parallel_envs):
            episode_frames[env_idx].append(frames[env_idx])
            episode_steps[env_idx] += 1

        total_steps += num_parallel_envs

        done = terminated | truncated

        if any(done):
            terminated_envs = torch.where(done)[0]
            success_envs = torch.where(reward == 1.0)[0]

            policy.reset(env_ids=terminated_envs)

            for env_idx in terminated_envs:
                env_idx_int = env_idx.item()
                is_success = env_idx in success_envs
                done_episodes += 1
                successes += int(is_success)

                for step_idx, frame in enumerate(episode_frames[env_idx_int]):
                    annotated_frame = _annotate_frame(
                        frame=frame,
                        env_idx=env_idx_int,
                        episode_num=done_episodes,
                        total_episodes=num_episodes,
                        episode_step=step_idx + 1,
                        is_success=is_success,
                        font=font,
                    )
                    video_writer.append_data(annotated_frame)

                episode_frames[env_idx_int] = []
                episode_steps[env_idx_int] = 0

        if total_steps % 1_000 == 0:
            logger.info(
                f"Total steps: {total_steps}, done episodes: {done_episodes}, successes: {successes}, "
                f"FPS={total_steps / (time.perf_counter() - start_time):.1f}"
            )

    video_writer.close()

    success_rate = successes / done_episodes if done_episodes > 0 else 0.0

    if policy_was_training:
        policy.train()

    total_elapsed_time = time.perf_counter() - start_time
    final_fps = total_steps / total_elapsed_time if total_elapsed_time > 0 else 0.0
    episodes_per_sec = done_episodes / total_elapsed_time if total_elapsed_time > 0 else 0.0

    logger.info(f"Evaluation: {done_episodes} episodes, {successes} successes ({success_rate * 100:.1f}%)")
    logger.info(f"Performance: {total_steps} steps in {total_elapsed_time:.1f}s | FPS: {final_fps:.1f}")
    logger.info(f"Video saved: {video_path}")

    return success_rate, video_path, final_fps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cfg: argparse.Namespace):
    device = torch.device(cfg.device)
    logger.info(colored(f"Using device: {device}", "green"))

    run_start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_cache_dir = _CACHE_ROOT / f"libero_plus_bc_{timestamp}_{Path(cfg.dataset).name}_{cfg.policy}"
    run_cache_dir.mkdir(parents=True, exist_ok=True)

    if cfg.seed is not None:
        set_seed(cfg.seed)
        logger.info(colored(f"Random seed set to {cfg.seed}", "yellow"))

    # ------------------------------------------------------------------
    # Dataset metadata
    # ------------------------------------------------------------------
    logger.info("Fetching dataset metadata…")
    ds_meta = LeRobotDatasetMetadata(cfg.dataset)

    # ------------------------------------------------------------------
    # Policy config
    # ------------------------------------------------------------------
    def _infer_type(val: str):
        if val.lower() in {"true", "false"}:
            return val.lower() == "true"
        try:
            if val.isdigit() or (val.startswith("-") and val[1:].isdigit()):
                return int(val)
            return float(val)
        except ValueError:
            return val

    if cfg.policy_kwargs is not None:
        policy_kwargs: dict = {}
        try:
            policy_kwargs = json.loads(cfg.policy_kwargs)
            if not isinstance(policy_kwargs, dict):
                raise TypeError
        except Exception:
            text = cfg.policy_kwargs.strip()
            tokens = re.split(r"[ ,]+", text)
            for token in filter(None, tokens):
                if "=" not in token:
                    raise ValueError(f"Cannot parse --policy_kwargs token '{token}'. Expected 'key=value'.")
                k, v = token.split("=", 1)
                policy_kwargs[k] = _infer_type(v)
    else:
        policy_kwargs = {}

    policy_cfg = make_policy_config(cfg.policy, **policy_kwargs)

    policy_cfg.chunk_size = 10
    policy_cfg.n_action_steps = 10

    if isinstance(cfg.device, str):
        policy_cfg.device = cfg.device.split(":", 1)[0]
    else:
        policy_cfg.device = cfg.device

    # Camera filtering
    if cfg.policy_cameras is not None:
        logger.info(f"Filtering dataset cameras to: {cfg.policy_cameras}")

        filtered_features = {
            key: feat for key, feat in ds_meta.features.items()
            if not key.startswith("observation.images.")
        }

        available_cameras = []
        for key, feat in ds_meta.features.items():
            if key.startswith("observation.images."):
                cam_name = key.replace("observation.images.", "")
                available_cameras.append(cam_name)
                if cam_name in cfg.policy_cameras:
                    filtered_features[key] = feat

        missing = [c for c in cfg.policy_cameras if c not in available_cameras]
        if missing:
            raise ValueError(
                f"Requested cameras not found in dataset: {missing}. "
                f"Available: {available_cameras}"
            )

        logger.info(f"Available cameras: {available_cameras}")
        logger.info(f"Selected cameras: {cfg.policy_cameras}")
        ds_meta.info["features"] = filtered_features

    # Proprioceptive filtering
    if cfg.disable_proprioceptive_obs:
        logger.info("Removing observation.state from dataset features")
        filtered_features = {k: v for k, v in ds_meta.features.items() if k != "observation.state"}
        remaining_obs = [k for k in filtered_features if k.startswith("observation")]
        if not remaining_obs:
            raise ValueError("No observation features remain after removing observation.state.")
        logger.info(f"Remaining observation keys: {remaining_obs}")
        ds_meta.info["features"] = filtered_features

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)
    logger.info("Building LeRobotDataset with inferred delta-timestamps…")

    image_transforms_config = ImageTransformsConfig(enable=True)
    image_transforms = ImageTransforms(image_transforms_config)

    dataset = LeRobotDataset(
        cfg.dataset,
        delta_timestamps=delta_timestamps,
        download_videos=True,
        image_transforms=image_transforms,
    )

    # ------------------------------------------------------------------
    # Dataloader
    # ------------------------------------------------------------------
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type != "cpu",
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
    dl_iter = cycle(dataloader)

    # ------------------------------------------------------------------
    # Policy + optimiser
    # ------------------------------------------------------------------
    policy = VecEnvPolicy(make_policy(policy_cfg, ds_meta=ds_meta))
    policy.train()

    smolvla_tokenizer = None
    smolvla_task_lookup = None
    if cfg.policy == "smolvla":
        from transformers import AutoTokenizer
        smolvla_tokenizer = AutoTokenizer.from_pretrained(policy_cfg.vlm_model_name)
        smolvla_task_lookup = {i: task_str for i, task_str in enumerate(ds_meta.tasks.index)}

    print(policy_cfg)

    lr_default = getattr(policy_cfg, "optimizer_lr", 1e-4)
    wd_default = getattr(policy_cfg, "optimizer_weight_decay", 0.0)
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=lr_default, weight_decay=wd_default)

    # ------------------------------------------------------------------
    # WandB
    # ------------------------------------------------------------------
    if cfg.wandb_enable:
        if cfg.wandb_project is None:
            raise ValueError("--wandb_project is required when --wandb_enable is set")

        wandb_run_id = cfg.resume_run_id if cfg.resume_run_id else None

        extra_cfg: dict[str, Any] = {}
        extra_cfg.update(vars(cfg))

        try:
            extra_cfg["policy_config"] = asdict(policy_cfg) if is_dataclass(policy_cfg) else policy_cfg.__dict__
        except Exception:
            extra_cfg["policy_config"] = str(policy_cfg)

        extra_cfg["dataset_meta"] = {
            "repo_id": ds_meta.repo_id,
            "fps": ds_meta.fps,
            "robot_type": ds_meta.robot_type,
            "total_episodes": ds_meta.total_episodes,
            "total_frames": ds_meta.total_frames,
            "feature_keys": list(ds_meta.features.keys()),
        }
        extra_cfg["delta_timestamps"] = delta_timestamps
        extra_cfg["image_transforms"] = asdict(image_transforms_config)

        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            config=extra_cfg,
            name=f"{cfg.policy}_{Path(cfg.dataset).name}",
            id=wandb_run_id,
            resume="must" if wandb_run_id else None,
        )
        logger.info(colored("W&B logging enabled", "blue"))

    # ------------------------------------------------------------------
    # Resume from checkpoint
    # ------------------------------------------------------------------
    start_step = 0
    if cfg.resume_run_id is not None:
        logger.info(colored(f"Resuming from WandB run {cfg.resume_run_id}", "cyan"))
        api = wandb.Api()
        artifact_path = (
            f"{(cfg.wandb_entity + '/' if cfg.wandb_entity else '')}"
            f"{cfg.wandb_project}/run_{cfg.resume_run_id}_latest:latest"
        )
        artifact = api.artifact(artifact_path)
        artifact_dir = Path(artifact.download())
        start_step, policy, optimizer = load_checkpoint(artifact_dir, policy, optimizer)
        policy.to(device)
    elif cfg.resume_ckpt is not None:
        logger.info(colored(f"Resuming from local checkpoint {cfg.resume_ckpt}", "cyan"))
        start_step, policy, optimizer = load_checkpoint(Path(cfg.resume_ckpt), policy, optimizer)
        policy.to(device)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    output_dir = run_cache_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    step = start_step
    best_success_rate = 0.0

    if cfg.rollout_freq and cfg.eval_suite:
        eval_num_tasks = _get_suite_num_tasks(cfg.eval_suite)
        logger.info(colored(
            f"Eval enabled: suite='{cfg.eval_suite}', "
            f"{eval_num_tasks} tasks × 1 episode, {cfg.eval_num_envs} tasks in parallel, "
            f"every {cfg.rollout_freq} steps",
            "cyan",
        ))

    while step < cfg.steps:
        iter_start_t = time.perf_counter()

        data_t0 = time.perf_counter()
        batch: dict[str, Any] = next(dl_iter)

        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                batch[key] = val.to(device, non_blocking=True)

        if smolvla_tokenizer is not None:
            task_indices = batch["task_index"].squeeze(-1).tolist()
            task_strs = [smolvla_task_lookup[int(i)] + "\n" for i in task_indices]
            tokenized = smolvla_tokenizer(
                task_strs,
                return_tensors="pt",
                padding=policy_cfg.pad_language_to,
                truncation=True,
                max_length=policy_cfg.tokenizer_max_length,
            )
            batch[OBS_LANGUAGE_TOKENS] = tokenized["input_ids"].to(device)
            batch[OBS_LANGUAGE_ATTENTION_MASK] = tokenized["attention_mask"].bool().to(device)

        data_load_ms = (time.perf_counter() - data_t0) * 1000

        update_t0 = time.perf_counter()

        loss, _ = policy.forward(batch)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        update_ms = (time.perf_counter() - update_t0) * 1_000
        iter_ms = (time.perf_counter() - iter_start_t) * 1_000
        loss_val = loss.item()

        if step % cfg.log_freq == 0:
            msg = (
                f"[step {step:>6d}/{cfg.steps}]"
                f" loss: {loss_val:.4f}"
                f" | data: {data_load_ms:.1f} ms"
                f" | update: {update_ms:.1f} ms"
                f" | iter: {iter_ms:.1f} ms"
            )
            logger.info(msg)
            if cfg.wandb_enable:
                wandb.log(
                    {
                        "train/loss": loss_val,
                        "time/data_load_ms": data_load_ms,
                        "time/update_ms": update_ms,
                        "time/iter_ms": iter_ms,
                    },
                    step=step,
                )

        if (step % cfg.save_freq == 0 and step != start_step) or step + 1 == cfg.steps:
            model_dir = output_dir / f"policy_step_{step}"
            model_dir.mkdir(parents=True, exist_ok=True)
            policy.save_pretrained(model_dir / "policy")

            latest_dir = output_dir / "latest"
            if latest_dir.exists():
                shutil.rmtree(latest_dir)
            save_checkpoint(latest_dir, step, policy, optimizer)

            logger.info(
                colored(
                    f"Checkpoint saved (model-only @ {model_dir}, full state @ {latest_dir})",
                    "magenta",
                )
            )

            if cfg.wandb_enable:
                art_model = wandb.Artifact(name=f"run_{wandb.run.id}_model_step_{step}", type="model")
                art_model.add_dir(str(model_dir))
                wandb.log_artifact(art_model)

                art_latest = wandb.Artifact(name=f"run_{wandb.run.id}_latest", type="model")
                art_latest.add_dir(str(latest_dir))
                wandb.log_artifact(art_latest, aliases=["latest"])

        step += 1

        # ------------------------------------------------------------------
        # Rollout evaluation
        # ------------------------------------------------------------------
        if (
            cfg.rollout_freq is not None
            and cfg.eval_suite is not None
            and step % cfg.rollout_freq == 0
            and step != start_step
        ):
            rollout_t0 = time.perf_counter()
            device_str = "cpu" if cfg.device == "cpu" else "cuda"

            category_rates, mean_success_rate, last_video_path = _run_eval(
                policy=policy,
                eval_suite=cfg.eval_suite,
                num_envs=cfg.eval_num_envs,
                device_str=device_str,
                camera_size=cfg.eval_camera_size,
                render_size=cfg.eval_render_size,
                video_key=cfg.eval_video_key,
                save_dir=output_dir,
                step=step,
                run_start_time=run_start_time,
                debug=cfg.debug,
                smolvla_tokenizer=smolvla_tokenizer,
            )

            rollout_ms = (time.perf_counter() - rollout_t0) * 1_000

            if cfg.wandb_enable:
                log_dict = {
                    f"eval/category/{cat}": rate
                    for cat, rate in category_rates.items()
                }
                log_dict["eval/mean_success_rate"] = mean_success_rate
                log_dict["time/rollout_ms"] = rollout_ms
                wandb.log(log_dict, step=step)
                if last_video_path is not None and last_video_path.exists():
                    wandb.log(
                        {"eval/rollout_video": wandb.Video(str(last_video_path), format="mp4", fps=20)},
                        step=step,
                    )

            success_rate = mean_success_rate

            if success_rate > best_success_rate:
                best_success_rate = success_rate
                logger.info(
                    colored(
                        f"New best mean success-rate ({best_success_rate * 100:.1f}%)! "
                        f"Saving checkpoint at step {step}",
                        "magenta",
                    )
                )

                best_model_dir = output_dir / f"best_step_{step}"
                best_model_dir.mkdir(parents=True, exist_ok=True)
                policy.save_pretrained(best_model_dir / "policy")

                best_dir = output_dir / "best"
                if best_dir.exists():
                    shutil.rmtree(best_dir)
                save_checkpoint(best_dir, step, policy, optimizer)

                if cfg.wandb_enable:
                    art_best = wandb.Artifact(name=f"run_{wandb.run.id}_best", type="model")
                    art_best.add_dir(str(best_dir))
                    wandb.log_artifact(art_best, aliases=["best", "latest"])

    logger.info(colored("Training finished!", "green", attrs=["bold"]))
    if cfg.wandb_enable:
        wandb.finish()

    if run_cache_dir.exists():
        logger.info(f"Cleaning up run directory: {run_cache_dir}")
        shutil.rmtree(run_cache_dir)


if __name__ == "__main__":
    """
    Example commands
    ================

    Basic training on LIBERO-plus (ACT policy):

        python -m resfit.lerobot.scripts.train_bc_libero_plus \\
            --dataset lerobot/libero_plus \\
            --policy act \\
            --batch_size 64 --num_workers 8 \\
            --wandb_project libero-plus-test \\
            --wandb_enable

    Training + evaluation on libero_spatial (SmolVLA):

        python -m resfit.lerobot.scripts.train_bc_libero_plus \\
            --dataset lerobot/libero_plus \\
            --policy smolvla \\
            --batch_size 32 --num_workers 8 \\
            --wandb_project libero-plus-test \\
            --rollout_freq 5000 --eval_suite libero_spatial \\
            --eval_num_envs 8 \\
            --eval_camera_size 256 --eval_render_size 256 \\
            --eval_video_key observation.images.front \\
            --wandb_enable

    Vision-only training (no proprioceptive state):

        python -m resfit.lerobot.scripts.train_bc_libero_plus \\
            --dataset lerobot/libero_plus \\
            --policy diffusion \\
            --disable_proprioceptive_obs \\
            --wandb_enable --wandb_project libero-plus-test

    Front-camera only:

        python -m resfit.lerobot.scripts.train_bc_libero_plus \\
            --dataset lerobot/libero_plus \\
            --policy act \\
            --policy_cameras front \\
            --wandb_enable --wandb_project libero-plus-test

    Debug mode (SyncVectorEnv for easier debugging):

        python -m resfit.lerobot.scripts.train_bc_libero_plus \\
            --dataset lerobot/libero_plus \\
            --policy act \\
            --rollout_freq 1000 --eval_suite libero_spatial \\
            --eval_num_envs 2 \\
            --debug

    Observation key differences vs. standard LIBERO
    ================================================
    Standard LIBERO (lerobot/libero_*_image):
        - observation.images.image      (agentview)
        - observation.images.wrist_image (wrist)

    LIBERO-plus (Sylvest/libero_plus_lerobot):
        - observation.images.front      (agentview)
        - observation.images.wrist      (wrist)

    The environment wrapper (LiberoPlusGymWrapper) automatically maps the
    agentview/robot0_eye_in_hand camera outputs to the correct key names so
    that policy observations match the dataset format.
    """
    main(args_cli)
