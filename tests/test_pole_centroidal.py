"""Independent MuJoCo physics and differentiable CAM checks."""
import mujoco
import numpy as np
import pytest
import torch
from pole_task import COURSE_ROOT
from supermimic.robot import quat_apply, quat_inverse, quat_mul
from pole_task.centroidal import CentroidalMomentum
from pole_task.data import load_course, load_retention
from pole_task.source import source_robot_xml
from pole_task.source_profile import SourcePoleRobot
from pole_task.source_runtime import source_nominal_context


def sample(dtype=torch.float32):
    torch.manual_seed(419)
    robot = SourcePoleRobot(source_robot_xml(COURSE_ROOT / 'pole_task/source_manifest.json')).to(dtype=dtype)
    states = []
    for case in ('nominal-01', 'eval-01', 'straight', 'left_turn', 'right_turn'):
        q = load_course(case).qpos if case.endswith('01') else load_retention(case)
        for frame in (0, len(q)//3, 2*len(q)//3):
            s = torch.zeros(79, dtype=dtype)
            s[:29] = torch.as_tensor(q[frame], dtype=dtype)
            s[29:57] = torch.randn(28, dtype=dtype) * .7
            states.append(s)
    state = torch.stack(states)
    context = source_nominal_context(robot, torch.zeros(len(state), 3, 2)).to(dtype)
    return robot, state, context


def mujoco_cam(robot, state, context):
    m = mujoco.MjModel.from_xml_path(str(robot.xml))
    d = mujoco.MjData(m)
    m.body_mass[robot.torso_id] += float(context[robot.context_layout.torso_added_mass][0])
    m.body_ipos[robot.root_id] += context[robot.context_layout.root_com].numpy()
    mujoco.mj_setConst(m, d)
    d.qpos[:] = state[:29].numpy()
    d.qvel[:] = state[29:57].numpy()
    d.qvel[3:6] = quat_apply(quat_inverse(state[3:7]), state[32:35]).numpy()
    mujoco.mj_forward(m, d)
    mujoco.mj_subtreeVel(m, d)
    return d.subtree_angmom[robot.root_id].copy()


@pytest.mark.parametrize('randomized', [False, True])
def test_mujoco_cam_including_fixed_links_and_context(randomized):
    robot, states, ctx = sample()
    if randomized:
        ctx[:, robot.context_layout.torso_added_mass] = .45
        ctx[:, robot.context_layout.root_com] = torch.tensor([.012, -.008, .015])
    cam = CentroidalMomentum(robot)
    actual, parts, center = cam(robot, states, ctx, components=True)
    expected = torch.tensor(np.stack([mujoco_cam(robot, s, c) for s,c in zip(states, ctx)]))
    error = (actual - expected).norm(dim=-1)
    print('CAM max absolute / normalized:', error.max().item(), (error/(1+expected.norm(dim=-1))).max().item())
    assert torch.all(error / (1 + expected.norm(dim=-1)) < 1e-5)
    assert torch.equal(actual, parts.sum(-2))
    assert len(cam.ids) == 26
    assert any(not robot.body_joints[i] for i in cam.ids.tolist())


def test_cam_invariants_and_nonzero_tilt():
    robot, s, ctx = sample()
    cam = CentroidalMomentum(robot)
    base = cam(robot, s, ctx)
    zero = s.clone(); zero[:,29:57] = 0
    torch.testing.assert_close(cam(robot, zero, ctx), torch.zeros_like(base))
    shifted = s.clone(); shifted[:,:3] += torch.tensor([4.,-2.,1.])
    torch.testing.assert_close(cam(robot, shifted, ctx), base, atol=2e-6, rtol=1e-5)
    shifted = s.clone(); shifted[:,29:32] += torch.tensor([2.,-.5,.3])
    torch.testing.assert_close(cam(robot, shifted, ctx), base, atol=2e-6, rtol=1e-5)
    q = torch.tensor([.8,.2,-.3,.4]); q /= q.norm()
    rotated = s.clone()
    rotated[:,:3] = quat_apply(q, s[:,:3])
    rotated[:,3:7] = quat_mul(q.expand(len(s),4), s[:,3:7])
    rotated[:,29:32] = quat_apply(q, s[:,29:32])
    rotated[:,32:35] = quat_apply(q, s[:,32:35])
    actual = cam(robot, rotated, ctx)
    torch.testing.assert_close(actual, quat_apply(q, base), atol=2e-6, rtol=1e-5)
    expected = torch.tensor(np.stack([mujoco_cam(robot,a,c) for a,c in zip(rotated,ctx)]))
    assert torch.all((actual-expected).norm(dim=-1)/(1+expected.norm(dim=-1)) < 1e-5)


def direction_check(function, state):
    state = state.detach().clone().requires_grad_()
    direction = torch.randn_like(state)
    direction[:,3:7] -= (direction[:,3:7]*state[:,3:7]).sum(-1,keepdim=True)*state[:,3:7]
    direction /= direction.norm()
    def evaluate(x):
        x = torch.cat((x[:,:3], x[:,3:7]/x[:,3:7].norm(dim=-1,keepdim=True), x[:,7:]), -1)
        return function(x).sum()
    derivative = (torch.autograd.grad(evaluate(state), state)[0]*direction).sum()
    assert derivative.abs() > 1e-8
    errors=[]
    for step in (1e-4,1e-5,1e-6):
        finite=(evaluate(state+step*direction)-evaluate(state-step*direction))/(2*step)
        errors.append(float((finite-derivative).abs().detach()))
    assert min(errors) < 1e-7 + 1e-4 * abs(float(derivative.detach()))


def test_cam_directional_derivative():
    robot, state, ctx = sample(torch.float64)
    cam = CentroidalMomentum(robot)
    direction_check(lambda s: cam(robot,s,ctx).square(),state)
