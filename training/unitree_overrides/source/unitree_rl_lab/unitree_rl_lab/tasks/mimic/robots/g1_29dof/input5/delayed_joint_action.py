from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass
from isaaclab.utils.buffers import DelayBuffer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class DelayedJointPositionAction(JointPositionAction):
    """Joint-position target with one randomized policy-step delay per environment."""

    cfg: DelayedJointPositionActionCfg

    def __init__(self, cfg: DelayedJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        if cfg.min_delay < 0 or cfg.max_delay < cfg.min_delay:
            raise ValueError(f"Invalid action delay range: [{cfg.min_delay}, {cfg.max_delay}]")

        self._delay_buffer = DelayBuffer(cfg.max_delay, self.num_envs, self.device)
        self._randomize_delay(None)

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        self._processed_actions = self._delay_buffer.compute(self._processed_actions)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        self._delay_buffer.reset(env_ids)
        self._randomize_delay(env_ids)

    def _randomize_delay(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            batch_ids = torch.arange(self.num_envs, device=self.device)
        else:
            batch_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        time_lags = torch.randint(
            low=self.cfg.min_delay,
            high=self.cfg.max_delay + 1,
            size=(batch_ids.numel(),),
            device=self.device,
        )
        self._delay_buffer.set_time_lag(time_lags, batch_ids)


@configclass
class DelayedJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for randomized action-target delay in policy steps."""

    class_type: type[ActionTerm] = DelayedJointPositionAction
    min_delay: int = 0
    max_delay: int = 1
