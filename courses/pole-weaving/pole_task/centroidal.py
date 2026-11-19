"""Differentiable world-frame momentum about the whole robot CoM."""

import torch
from torch import Tensor, nn
from supermimic.robot import quat_apply, quat_inverse, quat_mul


class CentroidalMomentum(nn.Module):
    """Constants come from the compiled source MJCF, never a second mass model."""

    def __init__(self, robot):
        super().__init__()
        model = robot.model
        ids = []
        for body in range(1, model.nbody):
            ancestor = body
            while ancestor and ancestor != robot.root_id:
                ancestor = int(model.body_parentid[ancestor])
            if ancestor == robot.root_id:
                ids.append(body)
        self.register_buffer("ids", torch.tensor(ids), persistent=False)
        for name in ("mass", "ipos", "iquat", "inertia"):
            self.register_buffer(name, torch.tensor(getattr(model, "body_" + name)[ids].copy()), persistent=False)
        self.root = ids.index(robot.root_id)
        self.torso = ids.index(robot.torso_id)
        self.layout = robot.context_layout

    def forward(self, robot, state: Tensor, context: Tensor, *, heads=None, components=False):
        if state.shape[-1] not in (57, 79) or context.shape[-1] != self.layout.size:
            raise ValueError("CAM requires source physical state and episode context")
        p, r, v, w = robot.kinematics(state) if heads is None else heads
        p, r, v, w = (x.index_select(-2, self.ids) for x in (p, r, v, w))
        mass = self.mass.to(state).expand(*state.shape[:-1], -1).clone()
        mass[..., self.torso] = mass[..., self.torso] + context[..., self.layout.torso_added_mass].squeeze(-1)
        ipos = self.ipos.to(state).expand(*state.shape[:-1], -1, 3).clone()
        ipos[..., self.root, :] = ipos[..., self.root, :] + context[..., self.layout.root_com]
        offset = quat_apply(r, ipos)
        centers = p + offset
        velocities = v + torch.linalg.cross(w, offset)
        total_mass = mass.sum(-1, keepdim=True)
        center = (centers * mass[..., None]).sum(-2) / total_mass
        velocity = (velocities * mass[..., None]).sum(-2) / total_mass
        inertia_rotation = quat_mul(r, self.iquat.to(state).expand_as(r))
        spin = quat_apply(inertia_rotation, self.inertia.to(state) * quat_apply(quat_inverse(inertia_rotation), w))
        orbital = torch.linalg.cross(centers - center[..., None, :], mass[..., None] * (velocities - velocity[..., None, :]))
        parts = spin + orbital
        return (parts.sum(-2), parts, center) if components else parts.sum(-2)
