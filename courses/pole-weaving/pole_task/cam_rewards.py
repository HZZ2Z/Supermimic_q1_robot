"""One source reward contract for collection and differentiable imagination."""

from dataclasses import asdict, dataclass, fields
import math
import torch
from supermimic.reward_tuning import tuned_reward
from .centroidal import CentroidalMomentum


@dataclass(frozen=True)
class CAMConfig:
    reward_schema_version: int = 1
    group: str = "B0"
    enabled: bool = False
    target_mode: str = "none"
    target_gradient: str = "full"
    momentum_scale: float = 1.0
    tracking_sigma: float = 1.0
    tracking_weight: float = 0.0
    damping_weight: float = 0.0
    activation: str = "first_warmup"

    def __post_init__(self):
        expected = {"B0": (False, "none"), "B_ref": (True, "reference"), "B_task": (True, "task")}
        if self.group not in expected or (self.enabled, self.target_mode) != expected[self.group]:
            raise ValueError("CAM group/enabled/target_mode disagree")
        if type(self.enabled) is not bool or type(self.reward_schema_version) is not int or self.reward_schema_version != 1:
            raise ValueError("Invalid CAM schema or enabled flag")
        if self.target_gradient != "full" or self.activation != "first_warmup":
            raise ValueError("CAM requires full gradients from first_warmup")
        for key in ("momentum_scale", "tracking_sigma", "tracking_weight", "damping_weight"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Invalid CAM {key}")
        if self.momentum_scale <= 0 or self.tracking_sigma <= 0 or self.tracking_weight < 0 or self.damping_weight != 0:
            raise ValueError("CAM scales must be positive; weights nonnegative; damping disabled")
        if not self.enabled and self.tracking_weight != 0:
            raise ValueError("B0 must disable CAM weight")

    @classmethod
    def parse(cls, values=None):
        if values is None:
            return cls()
        if not isinstance(values, dict) or set(values) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown CAM configuration fields")
        return cls(**values)

    def to_dict(self):
        return asdict(self)


class CAMReward:
    def __init__(self, robot, config=None):
        self.capture_terms = False
        self.config = config if isinstance(config, CAMConfig) else CAMConfig.parse(config)
        self.momentum = CentroidalMomentum(robot).to(robot.default_q.device) if self.config.enabled and self.config.tracking_weight else None

    def terms(self, robot, next_state, next_reference, context):
        cfg = self.config
        if self.momentum is None:
            raise ValueError("CAM terms requested for disabled reward")
        actual = self.momentum(robot, next_state, context)
        if cfg.target_mode == "reference":
            target_state = next_reference[..., :57]
        else:
            # Desired twist is world-frame; current configuration remains in the graph.
            velocity = torch.cat((next_reference[..., 29:31], torch.zeros_like(next_reference[..., 31:34]), next_reference[..., 34:35], torch.zeros_like(next_state[..., 35:57])), -1)
            target_state = torch.cat((next_state[..., :29], velocity), -1)
        target = self.momentum(robot, target_state, context)
        error = (target[..., 2] - actual[..., 2]) / (cfg.momentum_scale + target[..., 2].abs())
        return actual, target, error, torch.exp(-error.square() / cfg.tracking_sigma)

    def __call__(self, robot, state, next_state, target, action, dt, *, tuning, context, target_heads=None):
        reward = tuned_reward(robot, state, next_state, target, action, dt, tuning=tuning, context=context, target_heads=target_heads)
        if self.momentum is None:
            return reward
        addition = dt * self.config.tracking_weight * self.terms(robot, next_state, target, context)[3]
        if self.capture_terms:
            self.last_terms = (action, reward, addition)
        return reward + addition
