"""RL environment, observation encoding, and (upcoming) policy training."""

from starship.train.encoding import OBS_DIM, ObsConfig
from starship.train.env import RocketEnv, StepResult

__all__ = ["OBS_DIM", "ObsConfig", "RocketEnv", "StepResult"]
