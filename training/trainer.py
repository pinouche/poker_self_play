"""Gradient updates for the shared policy/Q network.

    L_total = q_weight * L_Q + policy_weight * L_policy + entropy_weight * L_entropy

* ``L_Q`` is a Huber regression of Q(s, a_taken) onto the lambda-return target
  computed along the acting seat's own decision chain.
* ``L_policy`` is the cross-entropy of the current policy against the improved
  policy derived from the current Q-values, renormalised over legal actions.
* ``L_entropy`` is *negative* mean entropy, so a positive coefficient rewards
  exploration.

Only the taken action contributes to the Q loss (counterfactual targets for the
untaken actions are not available in model-free self-play).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from config import Config, reward_scale
from model.network import PokerNet, masked_log_softmax

from .policy_improvement import improved_policy_torch, masked_entropy, masked_kl
from .replay_buffer import Batch, ReplayBuffer

_LOG_EPS = 1e-8


class Trainer:
    def __init__(
        self,
        cfg: Config,
        network: PokerNet,
        device: str = "cpu",
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        self.cfg = cfg
        self.network = network
        self.device = torch.device(device)
        self.network.to(self.device)
        self.q_scale = reward_scale(cfg.env)
        self.optimizer = optimizer or self._build_optimizer(cfg, network)
        self.updates = 0

    @staticmethod
    def _build_optimizer(cfg, network) -> torch.optim.Optimizer:
        kind = cfg.train.optimizer.lower()
        params = dict(
            lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay
        )
        if kind == "adamw":
            return torch.optim.AdamW(network.parameters(), **params)
        if kind == "adam":
            return torch.optim.Adam(network.parameters(), **params)
        raise ValueError(f"unknown optimizer {cfg.train.optimizer!r}")

    # --- helpers -----------------------------------------------------------
    def _to_tensor(self, array: np.ndarray, dtype=torch.float32) -> torch.Tensor:
        return torch.as_tensor(array, dtype=dtype, device=self.device)

    def _reference_log_policy(
        self, log_pi: torch.Tensor, old_policy: torch.Tensor, legal_mask: torch.Tensor
    ) -> torch.Tensor:
        """Log of the reference policy for the improvement operator."""
        mode = self.cfg.train.reference_policy
        if mode == "current":
            return log_pi.detach()
        if mode == "behavior":
            masked = old_policy * (legal_mask > 0)
            masked = masked / masked.sum(dim=-1, keepdim=True).clamp_min(_LOG_EPS)
            return torch.log(masked.clamp_min(_LOG_EPS))
        raise ValueError(f"unknown reference_policy: {mode}")

    # --- single update -----------------------------------------------------
    def train_step(self, batch: Batch) -> Dict[str, float]:
        cfg = self.cfg.train
        self.network.train()

        observations = self._to_tensor(batch.observations)
        legal_mask = self._to_tensor(batch.legal_action_masks)
        actions = self._to_tensor(batch.actions, dtype=torch.int64)
        q_targets = self._to_tensor(batch.q_targets)
        old_policy = self._to_tensor(batch.old_policies)

        policy_logits, q_values = self.network(observations)
        log_pi = masked_log_softmax(policy_logits, legal_mask)

        # --- Q loss on the taken action only ------------------------------
        # Both sides are divided by the reward scale so that `huber_delta`,
        # `q_weight` and `policy_weight` mean the same thing in every reward
        # mode.  Unnormalised, bb-scale targets (~+-50) sit far outside the
        # Huber knee and the Q term swamps the policy term by ~50x.
        q_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        q_loss = F.huber_loss(
            q_taken / self.q_scale, q_targets / self.q_scale, delta=cfg.huber_delta
        )

        # --- improved policy target ---------------------------------------
        with torch.no_grad():
            reference_log_policy = self._reference_log_policy(log_pi, old_policy, legal_mask)
            target_policy = improved_policy_torch(
                q_values.detach(),
                reference_log_policy,
                legal_mask,
                cfg.alpha,
                cfg.beta,
                q_scale=self.q_scale,
            )
        policy_loss = -(target_policy * log_pi.clamp_min(-30.0)).sum(dim=-1).mean()

        # --- entropy bonus -------------------------------------------------
        pi = torch.exp(log_pi) * (legal_mask > 0)
        entropy = masked_entropy(pi, legal_mask).mean()
        entropy_loss = -entropy

        total_loss = (
            cfg.q_weight * q_loss
            + cfg.policy_weight * policy_loss
            + cfg.entropy_weight * entropy_loss
        )

        if not torch.isfinite(total_loss):
            raise FloatingPointError("non-finite loss; aborting update")

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), cfg.grad_clip)
        self.optimizer.step()
        self.updates += 1

        with torch.no_grad():
            kl = masked_kl(target_policy, pi, legal_mask).mean()

        return {
            "loss": float(total_loss.item()),
            "q_loss": float(q_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "entropy": float(entropy.item()),
            "kl_target_vs_policy": float(kl.item()),
            "q_taken_mean": float(q_taken.mean().item()),
            "q_target_mean": float(q_targets.mean().item()),
            "grad_norm": float(grad_norm),
        }

    # --- many updates ------------------------------------------------------
    def train_iteration(
        self,
        buffer: ReplayBuffer,
        num_updates: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, float]:
        cfg = self.cfg.train
        num_updates = num_updates if num_updates is not None else cfg.updates_per_iteration
        if not buffer.is_ready(min(cfg.min_buffer_before_training, buffer.capacity)):
            return {}

        accumulated: Dict[str, float] = {}
        for _ in range(num_updates):
            batch = buffer.sample(cfg.batch_size, rng)
            metrics = self.train_step(batch)
            for key, value in metrics.items():
                accumulated[key] = accumulated.get(key, 0.0) + value

        if not accumulated:
            return {}
        return {key: value / num_updates for key, value in accumulated.items()}

    # --- state -------------------------------------------------------------
    def state_dict(self) -> dict:
        return {"optimizer": self.optimizer.state_dict(), "updates": self.updates}

    def load_state_dict(self, payload: dict) -> None:
        self.optimizer.load_state_dict(payload["optimizer"])
        self.updates = payload.get("updates", 0)
