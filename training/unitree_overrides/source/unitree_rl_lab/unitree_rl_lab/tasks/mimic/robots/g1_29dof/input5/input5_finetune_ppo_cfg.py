from isaaclab.utils import configclass

from unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class Input5FineTunePPORunnerCfg(BasePPORunnerCfg):
    """Low-exploration PPO settings for post-imitation robustness fine-tuning."""

    def __post_init__(self):
        self.experiment_name = "unitree_g1_29dof_mimic_input5"
        self.algorithm.entropy_coef = 0.001
        self.algorithm.learning_rate = 1.0e-4
