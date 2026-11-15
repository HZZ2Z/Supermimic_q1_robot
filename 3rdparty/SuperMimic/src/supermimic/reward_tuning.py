"""Opt-in Q1 rewards shared by training, plus a fixed-state CPU comparison.

Disabled tuning calls the original reward unchanged. Termination, observations,
physics and motion data are untouched.
"""

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from supermimic.data import Q1Motions
from supermimic.robot import Q1Robot, rotation_error


@dataclass(frozen=True)
class RewardTuningCfg:
  """Small auxiliary penalties; all weights default to zero.

  Arm terms average over Q1's eight shoulder/elbow joints. The joint term
  bypasses the saturated body-orientation Gaussian. The PD-target term is an
  explicit action prior and must use the simulator's own action scale/zero.
  """

  arm_joint_weight: float = 0.0
  arm_pd_target_weight: float = 0.0
  arm_error_std: float = 0.8
  action_scale: float = 0.5
  body_orientation_std: float | None = None
  foot_slip_weight: float = 0.0
  foot_landing_weight: float = 0.0
  foot_slip_speed_std: float = 0.25
  foot_landing_speed_std: float = 0.5
  foot_contact_height: float = 0.01
  foot_landing_height: float = 0.04

  def __post_init__(self) -> None:
    for weight in (
      self.arm_joint_weight, self.arm_pd_target_weight,
      self.foot_slip_weight, self.foot_landing_weight,
    ):
      if not math.isfinite(weight) or weight < 0:
        raise ValueError("Auxiliary reward weights must be finite and nonnegative")
    for scale in (
      self.arm_error_std, self.action_scale, self.foot_slip_speed_std,
      self.foot_landing_speed_std, self.foot_contact_height, self.foot_landing_height,
    ):
      if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Auxiliary reward scales must be finite and positive")
    if self.body_orientation_std is not None and (
      not math.isfinite(self.body_orientation_std) or self.body_orientation_std <= 0
    ):
      raise ValueError("Body orientation std must be finite and positive")

  @property
  def enabled(self) -> bool:
    return (
      self.arm_joint_weight > 0
      or self.arm_pd_target_weight > 0
      or self.body_orientation_std is not None
      or self.foot_slip_weight > 0
      or self.foot_landing_weight > 0
    )


BASELINE_TUNING = RewardTuningCfg()


def _arm_joint_indices(robot: Q1Robot) -> list[int]:
  indices = [
    i
    for i, name in enumerate(robot.joint_names)
    if "shoulder" in name or "elbow" in name
  ]
  if len(indices) != 8:
    raise ValueError("Expected Q1 V2's eight shoulder/elbow joints")
  return indices


def reward_tuning_terms(
  robot: Q1Robot,
  next_state: Tensor,
  target: Tensor,
  action: Tensor,
  tuning: RewardTuningCfg,
  context: Tensor | None = None,
  target_heads: tuple[Tensor, Tensor, Tensor, Tensor] | None = None,
  next_heads: tuple[Tensor, Tensor, Tensor, Tensor] | None = None,
) -> dict[str, Tensor]:
  """Return dimensionless reward changes, before multiplication by dt.

  A missing context explicitly means nominal motor offsets. When using the
  PD-target prior under domain randomization, pass the current simulator context.
  """
  indices = _arm_joint_indices(robot)
  qindices = [7 + i for i in indices]
  terms: dict[str, Tensor] = {}
  if tuning.body_orientation_std is not None:
    rotations = robot.kinematics(next_state)[1] if next_heads is None else next_heads[1]
    target_rotations = (
      robot.kinematics(target)[1] if target_heads is None else target_heads[1]
    )
    error = rotation_error(rotations[..., 1:, :], target_rotations[..., 1:, :]).mean(-1)
    original = (-error / robot.reward_scaling.orientation_std**2).exp()
    replacement = (-error / tuning.body_orientation_std**2).exp()
    terms["body_orientation_delta"] = replacement - original
  if tuning.arm_joint_weight:
    error = (next_state[..., qindices] - target[..., qindices]) / tuning.arm_error_std
    terms["arm_joint_penalty"] = -tuning.arm_joint_weight * error.square().mean(-1)
  if tuning.arm_pd_target_weight:
    zero = robot.default_q[indices]
    if context is not None:
      zero = zero + context[..., robot.context_layout.motor_offset][..., indices]
    pd_target = zero + tuning.action_scale * action[..., indices]
    error = (pd_target - target[..., qindices]) / tuning.arm_error_std
    terms["arm_pd_target_penalty"] = -tuning.arm_pd_target_weight * error.square().mean(
      -1
    )
  if tuning.foot_slip_weight or tuning.foot_landing_weight:
    positions, velocities = robot.foot_kinematics(next_state, next_heads)
    floor_z = float(robot.model.geom("floor").pos[2])
    height = positions[..., 2] - floor_z
    # shortcut: flat-ground geometry approximates contact; use a learned contact head for varied terrain.
    if tuning.foot_slip_weight:
      reference_positions, _ = robot.foot_kinematics(target, target_heads)
      reference_height = reference_positions[..., 2] - floor_z
      near = (1 - height / tuning.foot_contact_height).clamp(0, 1)
      reference_near = (1 - reference_height / tuning.foot_contact_height).clamp(0, 1)
      # Reference support plus detached gates remove the direct lift-to-escape gradient.
      support = torch.maximum(near, reference_near).detach()
      speed_sq = (velocities[..., :2] / tuning.foot_slip_speed_std).square().sum(-1)
      terms["foot_slip_penalty"] = -tuning.foot_slip_weight * (
        support * speed_sq
      ).mean((-2, -1))
    if tuning.foot_landing_weight:
      near_ground = (1 - height / tuning.foot_landing_height).clamp(0, 1).detach()
      downward = (-velocities[..., 2]).clamp_min(0) / tuning.foot_landing_speed_std
      terms["foot_landing_penalty"] = -tuning.foot_landing_weight * (
        near_ground * downward.square()
      ).mean((-2, -1))
  return terms


def tuned_reward(
  robot: Q1Robot,
  state: Tensor,
  next_state: Tensor,
  target: Tensor,
  action: Tensor,
  dt: float,
  *,
  tuning: RewardTuningCfg = BASELINE_TUNING,
  context: Tensor | None = None,
  target_heads: tuple[Tensor, Tensor, Tensor, Tensor] | None = None,
) -> Tensor:
  """Original Q1 reward with explicitly enabled experimental terms.

  The perfect static score remains 5*dt, and no tracking term can exceed its
  original maximum. Moving references can incur an action-prior penalty even
  with perfect tracking because a physical PD controller may need feedforward.
  This tradeoff is why the PD-target prior is a separate experimental switch.
  """
  next_heads = None
  if tuning.foot_slip_weight or tuning.foot_landing_weight:
    next_heads = robot.kinematics(next_state)
    if target_heads is None:
      target_heads = robot.kinematics(target)
    baseline = robot.reward(
      state, next_state, target, action, dt, target_heads, next_heads=next_heads
    )
  else:
    baseline = robot.reward(state, next_state, target, action, dt, target_heads)
  if not tuning.enabled:
    return baseline
  penalties = reward_tuning_terms(
    robot, next_state, target, action, tuning, context, target_heads, next_heads
  )
  return baseline + dt * sum(penalties.values())


def compare_fixed_rollout(args: argparse.Namespace) -> dict[str, Any]:
  """Compare reward/gradients on identical saved states, entirely on CPU."""
  torch.set_num_threads(1)
  tuning = RewardTuningCfg(**json.loads(args.config.read_text()))
  checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
  robot = Q1Robot(args.xml or Path(checkpoint["config"]["robot_xml"]))
  robot.load_state_dict(checkpoint["robot"])
  if robot.asset_sha256 != checkpoint["identity"]["asset_sha256"]:
    raise ValueError("The supplied XML/mesh differs from the checkpoint")
  if tuning.arm_pd_target_weight and not math.isclose(
    tuning.action_scale, checkpoint["config"]["action_scale"]
  ):
    raise ValueError("The PD prior action scale differs from the checkpoint")
  motion = Q1Motions(robot, [args.reference], device="cpu", storage="device").load(0)
  with np.load(args.rollout, allow_pickle=False) as data:
    states = torch.from_numpy(data["states"].copy()).to(robot.default_q.dtype)
  frames = min(args.frames, len(states) - 1, len(motion) - 1)
  if not 1 <= frames <= 50:
    raise ValueError("The fixed-rollout CPU comparison is bounded to 1..50 steps")
  actual = states[1 : frames + 1].clone().requires_grad_()
  state, target = states[:frames], motion[1 : frames + 1]
  action = states[1 : frames + 1, 57:79].clone().requires_grad_()
  baseline = robot.reward(state, actual, target, action, 0.02)
  tuned = tuned_reward(robot, state, actual, target, action, 0.02, tuning=tuning)
  baseline_qgrad = torch.autograd.grad(baseline.sum(), actual, retain_graph=True)[0]
  tuned_qgrad, direct_action_gradient = torch.autograd.grad(
    tuned.sum(), (actual, action)
  )
  arms = _arm_joint_indices(robot)
  qarms = [7 + i for i in arms]
  return {
    "comparison": "identical saved physical states; no policy update or simulation",
    "evaluation_context": "nominal motor offsets; use non-randomized recordings",
    "frames": frames,
    "arm_joint_names": [robot.joint_names[i] for i in arms],
    "tuning": asdict(tuning),
    "baseline_reward_mean": float(baseline.detach().mean()),
    "tuned_reward_mean": float(tuned.detach().mean()),
    "baseline_arm_q_gradient_mean_l2": float(
      baseline_qgrad[:, qarms].norm(dim=-1).mean()
    ),
    "tuned_arm_q_gradient_mean_l2": float(tuned_qgrad[:, qarms].norm(dim=-1).mean()),
    "direct_arm_action_gradient_mean_l2": float(
      direct_action_gradient[:, arms].norm(dim=-1).mean()
    ),
    "physical_arm_qpos_rmse_rad": float(
      (actual.detach()[:, qarms] - target[:, qarms]).square().mean().sqrt()
    ),
    "physical_improvement_demonstrated": False,
    "termination_unchanged": robot.reward_identity(),
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  subparsers = parser.add_subparsers(dest="command", required=True)
  config = subparsers.add_parser("config", help="Print an opt-in JSON configuration")
  config.add_argument(
    "--preset",
    choices=(
      "baseline",
      "arm-joint-0p05",
      "arm-pd-target-0p05",
      "body-orientation-0p6",
    ),
    default="baseline",
  )
  compare = subparsers.add_parser(
    "compare", help="CPU reward comparison of fixed saved states"
  )
  for name in ("config", "checkpoint", "reference", "rollout"):
    compare.add_argument(f"--{name}", type=Path, required=True)
  compare.add_argument("--xml", type=Path)
  compare.add_argument("--frames", type=int, default=25)
  compare.add_argument("--output", type=Path)
  args = parser.parse_args()
  if args.command == "config":
    presets = {
      "baseline": RewardTuningCfg(),
      "arm-joint-0p05": RewardTuningCfg(arm_joint_weight=0.05),
      "arm-pd-target-0p05": RewardTuningCfg(arm_pd_target_weight=0.05),
      "body-orientation-0p6": RewardTuningCfg(body_orientation_std=0.6),
    }
    result = asdict(presets[args.preset])
  else:
    result = compare_fixed_rollout(args)
  text = json.dumps(result, indent=2, allow_nan=False) + "\n"
  if args.command == "compare" and args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
      stream.write(text)
  else:
    print(text, end="")


if __name__ == "__main__":
  main()
