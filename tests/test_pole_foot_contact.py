"""Foot rewards, physical contact measurements and selection; no training runs."""

import copy
import json

import mujoco
import numpy as np
import pytest
import torch

from pole_task import COURSE_ROOT
from pole_task.data import load_course
from pole_task.foot_contact import foot_slip_passes
from pole_task.source import source_robot_xml
from pole_task.source_environment import SourcePoleEnvironment
from pole_task.source_profile import SourcePoleRobot
from pole_task.source_training import evaluation_score, source_acceptance
from supermimic.reward_tuning import RewardTuningCfg, reward_tuning_terms, tuned_reward


@pytest.fixture
def robot():
    return SourcePoleRobot(source_robot_xml(COURSE_ROOT / "pole_task/source_manifest.json")).double()


def standing(robot, height=0.0):
    state = torch.zeros((1, 79), dtype=torch.float64)
    state[:, :29] = torch.from_numpy(robot.model.qpos0.copy())
    points, _ = robot.foot_kinematics(state)
    state[:, 2] += height - points[..., 2].min()
    return state


def terms(robot, state, target, tuning):
    return reward_tuning_terms(robot, state, target, state[..., 57:79], tuning)


def test_sole_material_point_velocity_matches_mujoco_jacobian(robot):
    m, d = robot.model, mujoco.MjData(robot.model)
    d.qpos[:] = m.qpos0
    d.qpos[7:] += np.linspace(-0.05, 0.05, 22)
    d.qvel[:] = np.linspace(-0.3, 0.4, m.nv)
    mujoco.mj_forward(m, d)
    angular = np.empty(3)
    mujoco.mju_rotVecQuat(angular, d.qvel[3:6], d.qpos[3:7])
    state = torch.tensor(np.r_[d.qpos, d.qvel[:3], angular, d.qvel[6:], np.zeros(22)])[None]
    positions, velocities = robot.foot_kinematics(state)
    jp = np.zeros((3, m.nv))
    for foot, name in enumerate(("left", "right")):
        body = int(m.body(f"{name}_ankle_roll_link").id)
        for point, velocity in zip(positions[0, foot].numpy(), velocities[0, foot].numpy()):
            mujoco.mj_jac(m, d, jp, None, point, body)
            np.testing.assert_allclose(velocity, jp @ d.qvel, atol=1e-7)
    assert robot.foot_kinematics(state.expand(2, 3, -1))[0].shape == (2, 3, 2, 4, 3)


def test_slip_has_correct_gradient_and_reference_support_prevents_lift_escape(robot):
    tuning = RewardTuningCfg(foot_slip_weight=0.05)
    target = standing(robot)
    actual = target.clone()
    actual[:, 29] = 0.25
    actual.requires_grad_()
    penalty = terms(robot, actual, target, tuning)["foot_slip_penalty"]
    gradient = torch.autograd.grad(penalty.sum(), actual)[0]
    assert penalty.item() == pytest.approx(-0.05)
    assert gradient[0, 29] < 0 and torch.isfinite(gradient).all()
    epsilon = 1e-6
    shifted = []
    for sign in (1, -1):
        state = actual.detach().clone()
        state[:, 29] += sign * epsilon
        shifted.append(terms(robot, state, target, tuning)["foot_slip_penalty"])
    torch.testing.assert_close(gradient[:, 29], (shifted[0] - shifted[1]) / (2 * epsilon))
    airborne = actual.detach().clone()
    airborne[:, 2] += 0.2
    torch.testing.assert_close(terms(robot, airborne, target, tuning)["foot_slip_penalty"], penalty)
    swing_target = target.clone()
    swing_target[:, 2] += 0.2
    assert terms(robot, airborne, swing_target, tuning)["foot_slip_penalty"].item() == 0


def test_landing_only_penalizes_downward_motion_near_ground(robot):
    tuning = RewardTuningCfg(foot_landing_weight=0.02)
    state = standing(robot, height=0.02)
    state[:, 31] = -0.5
    state.requires_grad_()
    value = terms(robot, state, state, tuning)["foot_landing_penalty"]
    assert value.item() == pytest.approx(-0.01)
    gradient = torch.autograd.grad(value.sum(), state)[0]
    assert gradient[0, 31] > 0 and torch.isfinite(gradient).all()
    upward = state.detach().clone()
    upward[:, 31] = 0.5
    assert terms(robot, upward, upward, tuning)["foot_landing_penalty"].item() == 0
    distant = state.detach().clone()
    distant[:, 2] += 0.2
    assert terms(robot, distant, distant, tuning)["foot_landing_penalty"].item() == 0


def test_tuning_scales_once_by_dt_and_cached_reference_matches(robot):
    target = standing(robot)
    state = target.clone()
    state[:, 29] = 0.3
    state[:, 31] = -0.1
    tuning = RewardTuningCfg(foot_slip_weight=0.05, foot_landing_weight=0.02)
    action = state[:, 57:79]
    base = robot.reward(target, state, target, action, 0.02)
    actual = tuned_reward(robot, target, state, target, action, 0.02, tuning=tuning)
    torch.testing.assert_close(actual, base + 0.02 * sum(terms(robot, state, target, tuning).values()))
    cached = tuned_reward(robot, target, state, target, action, 0.02, tuning=tuning,
                          target_heads=robot.kinematics(target))
    torch.testing.assert_close(actual, cached)


def test_foot_rewards_reuse_baseline_kinematics(robot, monkeypatch):
    state = standing(robot)
    calls = []
    original = robot.kinematics

    def counted(value):
        calls.append(value)
        return original(value)

    monkeypatch.setattr(robot, "kinematics", counted)
    tuned_reward(robot, state, state, state, state[:, 57:79], 0.02,
                 tuning=RewardTuningCfg(foot_slip_weight=0.05, foot_landing_weight=0.02))
    assert len(calls) == 2


@pytest.mark.parametrize("field,value", [
    ("foot_slip_weight", -1), ("foot_landing_weight", float("nan")),
    ("foot_contact_height", 0), ("foot_landing_height", float("inf")),
    ("foot_slip_speed_std", 0), ("foot_landing_speed_std", -1),
])
def test_invalid_foot_reward_parameters_are_rejected(field, value):
    with pytest.raises(ValueError):
        RewardTuningCfg(**{field: value})


@pytest.mark.parametrize("seed", [20261103, 20261104, 20261105])
def test_randomized_contact_friction_equals_context_not_just_geom_values(robot, tmp_path, seed):
    env = SourcePoleEnvironment(load_course("nominal-01"), tmp_path / "scene.xml", robot,
                                seed=seed, randomize=True)
    env.reset(0)
    env.data.qpos[2] -= 0.005
    mujoco.mj_forward(env.model, env.data)
    contacts = [c for c in env.data.contact
                if ((int(c.geom1) in env.floor_geoms and int(c.geom2) in env.foot_geoms)
                    or (int(c.geom2) in env.floor_geoms and int(c.geom1) in env.foot_geoms))]
    assert contacts
    expected = float(env.context[0, env.layout.friction][0])
    assert all(c.friction[0] == pytest.approx(expected) for c in contacts)


def test_contact_metrics_measure_ground_sliding_and_reset(robot, tmp_path):
    env = SourcePoleEnvironment(load_course("nominal-01"), tmp_path / "scene.xml", robot)
    env.reset(0)
    assert env.foot_contact_metrics()["foot_slip_mean_m_s"] is None
    env.data.qpos[:] = standing(robot, height=-0.005)[0, :29].numpy()
    env.data.qvel[:] = 0
    env.data.qvel[0] = 0.2
    mujoco.mj_forward(env.model, env.data)
    env._record_foot_contacts(env.data.qvel.copy())
    row = env.foot_contact_metrics()
    assert all(t > 0 for t in row["foot_contact_time_s"])
    assert row["foot_slip_mean_m_s"] == pytest.approx(0.2)
    assert row["foot_slip_rms_m_s"] == pytest.approx(0.2)
    assert row["foot_slip_fraction"] == 1.0
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)
    env._record_foot_contacts(env.data.qvel.copy())
    assert env.foot_contact_metrics()["foot_slip_mean_m_s"] == pytest.approx(0.1)
    assert env.foot_contact_metrics()["foot_slip_fraction"] == pytest.approx(0.5)
    env.reset(0)
    assert env.foot_contact_metrics()["foot_contact_time_s"] == [0.0, 0.0]


def test_selection_rejects_sliding_missing_or_nonfinite_metrics_and_ranks_good_feet():
    config = json.loads((COURSE_ROOT / "pole_task/config.json").read_text())
    row = dict(route_completed=True, pole_collision=False, fall=False,
               root_xy_rmse_m=0.03, endpoint_distance_m=0.02,
               foot_contact_time_s=[1.0, 1.0], foot_slip_mean_m_s=0.02, foot_slip_fraction=0.05)
    reports = {name: row.copy() for name in ["nominal-01", *config["evaluation_case_ids"],
                                            *config["retention_case_ids"]]}
    assert source_acceptance(reports, config)
    lower_slip = copy.deepcopy(reports)
    lower_slip["nominal-01"]["foot_slip_mean_m_s"] = 0.01
    assert evaluation_score(lower_slip, config) < evaluation_score(reports, config)
    for overrides in ({"foot_slip_mean_m_s": 0.11}, {"foot_slip_fraction": 0.21},
                      {"foot_slip_mean_m_s": None}, {"foot_slip_fraction": float("nan")},
                      {"foot_contact_time_s": [0, 1]}, {"root_xy_rmse_m": float("nan")}):
        candidate = copy.deepcopy(reports)
        candidate["nominal-01"].update(overrides)
        assert not source_acceptance(candidate, config)
        assert evaluation_score(candidate, config) == float("inf")
    assert not foot_slip_passes({})
    assert not foot_slip_passes(row, {"foot_slip_mean_max_m_s": 0.01})


@pytest.mark.parametrize("config", [{"foot_slip_mean_max_m_s": 0},
                                   {"foot_slip_fraction_max": float("nan")},
                                   {"foot_slip_fraction_max": 1.1}])
def test_invalid_evaluation_limits_fail_closed(config):
    with pytest.raises(ValueError):
        foot_slip_passes({}, config)
