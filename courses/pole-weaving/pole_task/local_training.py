"""Versioned LR diagnosis using V1 sampling; the source recipe remains immutable."""
import math
import numpy as np
import torch
from supermimic.replay import TemporalReplay
from supermimic.robot import quat_apply, rotation_error
from .foot_contact import foot_slip_passes

VARIANTS = {
    'control': (1e-5, 'constant'),
    'low_lr': (3e-6, 'constant'),
    'actor_cosine': (1e-5, 'cosine'),
}
QUALITY_LIMITS = {
    'schema_version': 1,
    'foot_slip_mean_max_m_s': .10,
    'foot_slip_fraction_max': .20,
    'tilt_error_rms_max_deg': 5.,
    'pd_acceleration_ratio_max': 2.,
    'reference_acceleration_floor_rad_s2': 10.,
}


def training_profile(config):
    value = config.get('local_training_profile')
    if value is None:
        return None
    if (not isinstance(value, dict) or set(value) != {'schema_version', 'variant'}
            or type(value['schema_version']) is not int or value['schema_version'] != 1
            or not isinstance(value['variant'], str) or value['variant'] not in VARIANTS):
        raise ValueError('local_training_profile requires schema_version=1 and a registered variant')
    lr, schedule = VARIANTS[value['variant']]
    if config.get('cam', {}).get('enabled', False):
        raise ValueError('LR/replay diagnosis keeps CAM disabled')
    if config.get('collection_episodes', 4) != 4 or config.get('retention_collection_episodes', 2) != 2:
        raise ValueError('V1 sampling requires 4 nominal and 2 retention episodes per batch')
    return {**value, 'experiment': 'lr-replay-v1', 'actor_learning_rate': lr,
            'actor_schedule': schedule, 'actor_final_ratio': .1,
            'actor_start_sampling': 'stratified_4', 'actor_collection_interval': 25,
            'actor_replay_batches': 2, 'quality_limits': dict(QUALITY_LIMITS)}


def stratified_start_frame(last_frame, batch, count, index, seed):
    # V1's exact partition and first seeded draw; reserve ten history transitions.
    bins = np.array_split(np.arange(1, last_frame - 9), 4)
    choices = bins[(batch * count + index) % 4]
    if not len(choices):
        raise ValueError('Reference too short for four strata with ten history transitions')
    return int(choices[int(np.random.default_rng(seed).random() * len(choices))])


def merge_replay_batches(batches, device):
    if not 1 <= len(batches) <= 2:
        raise ValueError('Replay must contain one or two whole collection batches')
    counts = [s['count'] for _, s in batches]
    if any(n <= 0 for n in counts) or any(b[0] <= a[0] for a, b in zip(batches, batches[1:])):
        raise ValueError('Invalid replay batch counts/order')
    for _, state in batches:
        if state['num_envs'] != 1 or not bool(state['boundaries'][-1].all()):
            raise ValueError('Collection batch must end at an episode boundary')
    count = sum(counts)
    merged = {'capacity': max(256, count), 'num_envs': 1, 'count': count}
    for name in TemporalReplay._storage_names():
        merged[name] = torch.cat([s[name] for _, s in batches], dim=0)
    replay = TemporalReplay(capacity=merged['capacity'], num_envs=1, state_dim=79,
                            context_dim=149, obs_dim=78, device=device)
    replay.load_state_dict(merged)
    return replay


def restore_replay_batches(runtime):
    merged = runtime['replay']
    result, start = [], 0
    for row in runtime['replay_batches']:
        count = row['count']
        if type(count) is not int or count <= 0:
            raise ValueError('Invalid saved replay batch size')
        state = {'capacity': max(256, count), 'num_envs': 1, 'count': count}
        for name in TemporalReplay._storage_names():
            state[name] = merged[name][start:start + count].clone()
            if len(state[name]) != count:
                raise ValueError('Saved replay batch exceeds initialized rows')
        result.append((row['update'], state))
        start += count
    if start != merged['count'] or not 1 <= len(result) <= 2:
        raise ValueError('Saved replay batch boundaries do not cover the replay')
    if any(type(u) is not int or u <= 0 or not bool(s['boundaries'][-1].all()) for u,s in result):
        raise ValueError('Invalid saved replay collection boundary')
    if any(b[0] <= a[0] for a,b in zip(result,result[1:])):
        raise ValueError('Saved replay updates must be ordered')
    return result


def motion_metrics(states, references, actions, dt):
    state = torch.as_tensor(np.asarray(states)[1:], dtype=torch.float64)
    target = torch.as_tensor(np.asarray(references), dtype=torch.float64)
    if len(state) == 0:
        return {'motion_metrics_version': 1}
    def tilt(q):
        axis = quat_apply(q / q.norm(dim=-1, keepdim=True), q.new_tensor([1., 0., 0.]))
        return torch.atan2(-axis[:, 2], axis[:, :2].norm(dim=-1)) * (180 / math.pi)
    actual_tilt, ref_tilt = tilt(state[:, 3:7]), tilt(target[:, 3:7])
    actual_accel = .5 * np.diff(np.asarray(actions), n=2, axis=0) / dt**2
    ref_accel = np.diff(target[:, 7:29].numpy(), n=2, axis=0) / dt**2
    rms = lambda x: float(np.sqrt(np.mean(x*x)))
    ref_rms = rms(ref_accel) if len(ref_accel) else None
    actual_rms = rms(actual_accel) if len(actual_accel) else None
    return {
        'motion_metrics_version': 1,
        'root_orientation_rms_rad': float(rotation_error(state[:, 3:7], target[:, 3:7]).mean().sqrt()),
        'root_forward_tilt_mean_deg': float(actual_tilt.mean()),
        'reference_forward_tilt_mean_deg': float(ref_tilt.mean()),
        'tilt_error_rms_deg': float((actual_tilt-ref_tilt).square().mean().sqrt()),
        'pd_target_acceleration_rms_rad_s2': actual_rms,
        'reference_joint_acceleration_rms_rad_s2': ref_rms,
        'pd_acceleration_ratio': actual_rms / max(ref_rms, QUALITY_LIMITS['reference_acceleration_floor_rad_s2']) if ref_rms is not None and actual_rms is not None else None,
    }


def quality_failures(row, profile):
    limits = profile['quality_limits']
    failures = []
    if not foot_slip_passes(row, limits):
        failures.append('loaded_foot_slip')
    for key, limit in (('tilt_error_rms_deg', limits['tilt_error_rms_max_deg']),
                       ('pd_acceleration_ratio', limits['pd_acceleration_ratio_max'])):
        value = row.get(key)
        if not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= limit:
            failures.append(key)
    value = row.get('root_orientation_rms_rad')
    if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
        failures.append('root_orientation_rms_rad')
    return failures


def selection_score(reports, profile):
    if profile is None:
        return sum(row['root_xy_rmse_m'] for row in reports.values())
    values = [reports[c].get('root_orientation_rms_rad') for c in ('nominal-01', 'eval-01')]
    if any(not isinstance(v, (int,float)) or not math.isfinite(v) for v in values):
        return float('inf')
    return sum(values) / 2
