from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import torch
from torch import Tensor, nn


class VecEnvPolicy(nn.Module):
    """Thin wrapper adding per-environment reset/action-queue to any upstream lerobot policy.

    The upstream lerobot policies assume a single environment. This wrapper adds:
    - reset(env_ids=...)  — selective reset of individual environments
    - select_action(batch) — per-environment action-queue management for batch_size > 1

    Works with any policy that exposes predict_action_chunk(batch) -> Tensor (ACT,
    Diffusion, SmolVLA in lerobot ≥ v0.5.1).

    Usage::

        policy = VecEnvPolicy(upstream_policy)
        # or
        policy = VecEnvPolicy.from_pretrained(path)
    """

    def __init__(self, policy: nn.Module) -> None:
        super().__init__()
        self.policy = policy

    # ------------------------------------------------------------------
    # Config / attribute pass-through
    # ------------------------------------------------------------------

    @property
    def config(self):
        return self.policy.config

    # ------------------------------------------------------------------
    # Vectorized-environment support
    # ------------------------------------------------------------------

    def reset(self, env_ids: list[int] | None = None) -> None:
        """Reset per-environment action caches.

        Args:
            env_ids: Indices of environments to reset. Resets all when None.
        """
        temporal_coeff = getattr(self.config, "temporal_ensemble_coeff", None)
        if temporal_coeff is not None:
            if not getattr(self, "_temporal_ensemblers", None):
                return
            targets = range(len(self._temporal_ensemblers)) if env_ids is None else env_ids
            for i in targets:
                if i < len(self._temporal_ensemblers):
                    self._temporal_ensemblers[i].reset()
        else:
            if not getattr(self, "_action_queues", None):
                return
            targets = range(len(self._action_queues)) if env_ids is None else env_ids
            for i in targets:
                if i < len(self._action_queues):
                    self._action_queues[i].clear()

    def _ensure_action_queues(self, batch_size: int) -> None:
        n_steps = self.config.n_action_steps
        if not hasattr(self, "_action_queues"):
            self._action_queues = []
        while len(self._action_queues) < batch_size:
            self._action_queues.append(deque(maxlen=n_steps))
        if len(self._action_queues) > batch_size:
            self._action_queues = self._action_queues[:batch_size]

    def _ensure_temporal_ensemblers(self, batch_size: int) -> None:
        from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

        coeff = self.config.temporal_ensemble_coeff
        chunk = self.config.chunk_size
        if not hasattr(self, "_temporal_ensemblers"):
            self._temporal_ensemblers = []
        while len(self._temporal_ensemblers) < batch_size:
            self._temporal_ensemblers.append(ACTTemporalEnsembler(coeff, chunk))
        if len(self._temporal_ensemblers) > batch_size:
            self._temporal_ensemblers = self._temporal_ensemblers[:batch_size]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Return one action per environment, managing per-env action queues."""
        self.policy.eval()
        batch_size = next(v.shape[0] for v in batch.values() if isinstance(v, Tensor))

        temporal_coeff = getattr(self.config, "temporal_ensemble_coeff", None)
        if temporal_coeff is not None:
            self._ensure_temporal_ensemblers(batch_size)
            chunk = self.policy.predict_action_chunk(batch)  # (B, chunk_size, act_dim)
            return torch.cat(
                [self._temporal_ensemblers[i].update(chunk[i : i + 1]) for i in range(batch_size)],
                dim=0,
            )

        self._ensure_action_queues(batch_size)
        envs_needing_chunk = [i for i, q in enumerate(self._action_queues) if len(q) == 0]
        if envs_needing_chunk:
            chunk = self.policy.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            for i in envs_needing_chunk:
                self._action_queues[i].extend(chunk[i].unbind(0))
        return torch.stack([q.popleft() for q in self._action_queues], dim=0)

    # ------------------------------------------------------------------
    # Delegation to the wrapped policy
    # ------------------------------------------------------------------

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        return self.policy(batch)

    def get_optim_params(self):
        return self.policy.get_optim_params()

    def save_pretrained(self, path: str | Path) -> None:
        self.policy.save_pretrained(path)

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, path: str | Path) -> VecEnvPolicy:
        """Load any upstream lerobot policy from a checkpoint directory and wrap it.

        Policy type is inferred from config.json (``type`` field, or ``use_vae``
        field as a fallback for ACT checkpoints saved without a type field).
        """
        from lerobot.policies.factory import get_policy_class

        policy_dir = Path(path)
        with (policy_dir / "config.json").open() as f:
            cfg_dict = json.load(f)

        policy_type = str(cfg_dict.get("type", "")).lower()
        if not policy_type:
            # Fallback: ACT configs contain use_vae; diffusion configs contain noise_scheduler_type
            if "use_vae" in cfg_dict:
                policy_type = "act"
            elif "noise_scheduler_type" in cfg_dict:
                policy_type = "diffusion"
            else:
                raise ValueError(
                    f"Cannot infer policy type from config.json in {policy_dir}. "
                    "Add a 'type' field or ensure ACT/Diffusion-specific fields are present."
                )

        policy_cls = get_policy_class(policy_type)
        policy = policy_cls.from_pretrained(str(policy_dir))
        return cls(policy)
