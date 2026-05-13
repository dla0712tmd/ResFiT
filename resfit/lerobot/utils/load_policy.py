# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import json
from pathlib import Path

import torch
import wandb

from resfit.lerobot.policies.vec_env_policy import VecEnvPolicy


def download_policy_from_wandb(
    run_id: str,
    *,
    step: str | None = None,
    artifact_version: str = "latest",
) -> tuple[Path, str]:
    """Download a policy checkpoint logged on W&B and return its folder.

    The policy is expected to have been created with the training utilities in
    `train_hf.py` and therefore to contain a `config.json` in the root of the
    downloaded artifact.
    """
    api = wandb.Api()
    project, id_ = run_id.split("/")

    if step is None or str(step).lower() == "latest":
        artifact_name = f"run_{id_}_latest:{artifact_version}"
        checkpoint_step = "latest"
    elif str(step).lower() == "best":
        artifact_name = f"run_{id_}_best:{artifact_version}"
        checkpoint_step = "best"
    else:
        artifact_name = f"run_{id_}_model_step_{step}:{artifact_version}"
        checkpoint_step = str(step)

    artifact_path = f"{project}/{artifact_name}"
    artifact = api.artifact(artifact_path)

    art_dir = Path(artifact.download())
    policy_dir = art_dir / "policy"  # The artifact root already contains the policy files.

    if not (policy_dir / "config.json").exists():
        raise FileNotFoundError(f"Policy directory not found inside downloaded artifact: {policy_dir}")

    return policy_dir, checkpoint_step


def load_policy(policy_dir: Path) -> VecEnvPolicy:
    """Load a policy checkpoint and wrap it with VecEnvPolicy for vectorized-env support."""
    return VecEnvPolicy.from_pretrained(policy_dir)


def save_checkpoint(ckpt_dir: Path, step: int, policy, optimizer) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Save model weights + config (delegates to the underlying upstream policy)
    policy.save_pretrained(ckpt_dir / "policy")
    # Save optimizer & misc state
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
        },
        ckpt_dir / "trainer_state.pt",
    )


def load_checkpoint(ckpt_dir: Path, policy, optimizer):
    state_pth = ckpt_dir / "trainer_state.pt"
    if not state_pth.exists():
        raise FileNotFoundError(state_pth)
    state = torch.load(state_pth, map_location="cpu")
    policy_loaded = VecEnvPolicy.from_pretrained(ckpt_dir / "policy")
    optimizer.load_state_dict(state["optimizer"])
    return state["step"], policy_loaded, optimizer
