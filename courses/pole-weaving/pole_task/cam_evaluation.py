"""Real rollout quality, fixed-action model error, and paired candidate decisions."""
from pathlib import Path
import json
import math
import mujoco
import numpy as np
import torch
from . import project_path
from .centroidal import CentroidalMomentum
from .cam_rewards import CAMReward, CAMConfig
from .inference import load_pole_actor
from .source_playback import load_course_bundle
from .source_environment import SourcePoleEnvironment
from .source_runtime import run_source_episode, advance_source_context
from .source_training import evaluation_cases_for
from .acceptance import full_acceptance
from .models import build_source_models
from supermimic.robot import rotation_error, quat_apply, quat_inverse

PROTOCOL = {
    'schema_version': 1,
    'primary': 'mean of nominal-01/eval-01 root SO3 angle RMS radians',
    'minimum_improvement': .05,
    'maximum_guardrail_regression': .05,
    'guardrails': ['foot_slip_mps','action_rate_rms_per_s','model_state_rmse','model_cam_rmse'],
    'evaluation_seed': 20261103,
    'model_horizon': 32,
    'model_window_stride': 32,
    'model_state_units': 'raw physical 57 columns, quaternion sign aligned; also report position/joint/velocity separately',
    'selection': 'original accepted-task then sum root XY RMSE',
    'validation_case': 'eval-01 participates in selection, not independent generalization',
    'tie_break': ['root_orientation_rms_rad','foot_slip_mps','action_rate_rms_per_s','B_ref'],
}


def foot_slip(episode, robot, path):
    env = SourcePoleEnvironment(episode.case, path, robot)
    squares = []
    for state, context in zip(episode.states[1:],episode.contexts):
        env.context = torch.tensor(context).unsqueeze(0)
        env._apply_context_physics()
        env.data.qpos[:] = state[:29]
        env.data.qvel[:] = state[29:57]
        env.data.qvel[3:6] = quat_apply(quat_inverse(torch.tensor(state[3:7])),torch.tensor(state[32:35])).numpy()
        mujoco.mj_forward(env.model,env.data)
        for contact in env.data.contact:
            a,b=int(contact.geom1),int(contact.geom2)
            if contact.dist>0: continue
            foot = a if a in env.foot_geoms and b in env.floor_geoms else b if b in env.foot_geoms and a in env.floor_geoms else None
            if foot is None: continue
            jac=np.zeros((3,env.model.nv))
            mujoco.mj_jac(env.model,env.data,jac,None,contact.pos,int(env.model.geom_bodyid[foot]))
            velocity=jac@env.data.qvel
            normal=contact.frame[:3]
            tangent=velocity-normal*np.dot(velocity,normal)
            squares.append(float(tangent@tangent))
    return math.sqrt(float(np.mean(squares))) if squares else None


@torch.no_grad()
def quality(episode, robot, motions, world, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    device=robot.default_q.device
    states=torch.tensor(episode.states,device=device)
    contexts=torch.tensor(episode.contexts,device=device)
    n=episode.steps
    references=motions.sequence(torch.full((n,),episode.clip,device=device),torch.tensor(episode.frames,device=device),torch.ones(1,device=device,dtype=torch.long))[:,0]
    cam=CentroidalMomentum(robot).to(device)
    actual=cam(robot,states[1:],contexts)
    reference=cam(robot,references[..., :57],contexts)
    own=robot.cam_reward
    own_error=(own.terms(robot,states[1:],references,contexts)[1][:,2]-actual[:,2]) if own.momentum else reference[:,2]-actual[:,2]
    errors=[]; cam_errors=[]; positions=[]; joints=[]; velocities=[]
    for start in range(0,n-31,32):
        predicted=states[start:start+1].clone(); context=contexts[start:start+1].clone()
        for step in range(32):
            index=start+step
            action=torch.tensor(episode.actions[index:index+1],device=device)
            predicted=world(predicted,action,context)
            truth=states[index+1:index+2]
            difference=predicted[:,:57]-truth[:,:57]
            sign=torch.where((predicted[:,3:7]*truth[:,3:7]).sum(-1,keepdim=True)<0,-1.,1.)
            difference[:,3:7]=predicted[:,3:7]*sign-truth[:,3:7]
            errors.append(difference.square().mean())
            positions.append(difference[:,:3].square().mean());joints.append(difference[:,7:29].square().mean());velocities.append(difference[:,29:57].square().mean())
            cam_errors.append((cam(robot,predicted,context)-cam(robot,truth,context)).square().mean())
            context=advance_source_context(robot,context,action)
    mean_rms=lambda values: float(torch.stack(values).mean().sqrt()) if values else None
    row={
        **episode.result,
        'root_orientation_rms_rad':float(rotation_error(states[1:,3:7],references[:,3:7]).mean().sqrt()),
        'cam_reference_z_rmse':float((actual[:,2]-reference[:,2]).square().mean().sqrt()),
        'cam_own_z_rmse':float(own_error.square().mean().sqrt()),
        'foot_slip_mps':foot_slip(episode,robot,output/'contact-measurement.xml'),
        'action_rate_rms_per_s':float(np.sqrt(np.mean(np.diff(np.concatenate((episode.states[:1,57:79],episode.actions)),axis=0)**2))/.02),
        'model_state_rmse':mean_rms(errors),'model_cam_rmse':mean_rms(cam_errors),
        'model_position_rmse_m':mean_rms(positions),'model_joint_rmse_rad':mean_rms(joints),'model_velocity_rmse':mean_rms(velocities),
        'model_windows':n//32,
    }
    arm=[i for i,name in enumerate(robot.joint_names) if 'shoulder' in name or 'elbow' in name]
    np.savez_compressed(output/f'{episode.case.case_id}-trajectory.npz',states=episode.states,actions=episode.actions,contexts=episode.contexts,frames=episode.frames,references=references.cpu().numpy(),cam=actual.cpu().numpy(),reference_cam=reference.cpu().numpy(),arm_joint_indices=arm)
    return row


def render_trajectory(case, robot, states, path):
    import imageio.v2 as imageio
    env=SourcePoleEnvironment(case,path.with_suffix('.xml'),robot)
    camera=mujoco.MjvCamera();camera.lookat[:]=[0.,2.4,.4];camera.distance=8.;camera.azimuth=135;camera.elevation=-35
    with mujoco.Renderer(env.model,height=720,width=960) as renderer, imageio.get_writer(path,fps=25) as writer:
        for state in states[::2]:
            env.data.qpos[:]=state[:29];mujoco.mj_forward(env.model,env.data)
            renderer.update_scene(env.data,camera=camera);writer.append_data(renderer.render())


def evaluate_checkpoint(checkpoint, output, device='cpu', video=True, config_path=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    actor,identity=load_pole_actor(checkpoint,device,config_path=config_path)
    config=identity['resolved_config']
    robot,motions,cases,_=load_course_bundle(reference_dir=output/'references',device=device,config=config)
    _,world,_=build_source_models(robot)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    world.load_state_dict(saved['model'],strict=True);world.to(device).eval()
    reports={};video_errors={}
    for index,case in enumerate(evaluation_cases_for(config,list(cases.values()))):
        clip=next(i for i,c in enumerate(motions.cases) if c.case_id==case.case_id)
        episode=run_source_episode(actor,robot,motions,case,clip,output/'scene.xml',start_frame=0,seed=PROTOCOL['evaluation_seed']+index,device=device)
        reports[case.case_id]=quality(episode,robot,motions,world,output)
        print(json.dumps({'case':case.case_id,'metrics':reports[case.case_id]}),flush=True)
        if video:
            try:render_trajectory(case,robot,episode.states,output/f'{case.case_id}.mp4')
            except (RuntimeError,ImportError,OSError) as exc:video_errors[case.case_id]=str(exc)
    print('Running original five-case and 20-trial nominal acceptance',flush=True)
    acceptance=full_acceptance(actor,robot,motions,cases,output/'acceptance.xml',config=config,seed=PROTOCOL['evaluation_seed'],device=device)
    # Repeated nominal rollouts estimate numerical reproducibility, not perturbation robustness.
    repeat=[];case=cases['nominal-01'];clip=next(i for i,c in enumerate(motions.cases) if c.case_id==case.case_id)
    for repeat_index in range(3):
        ep=run_source_episode(actor,robot,motions,case,clip,output/'repeat.xml',start_frame=0,seed=PROTOCOL['evaluation_seed'],device=device)
        repeat.append(quality(ep,robot,motions,world,output/f'repeat-{repeat_index}'))
    noise={key:max(r[key] for r in repeat)-min(r[key] for r in repeat) for key in ['root_orientation_rms_rad',*PROTOCOL['guardrails']] if all(r[key] is not None for r in repeat)}
    protocol=dict(PROTOCOL)
    if config.get('local_training_profile'):
        from .local_training import training_profile
        protocol.update(experiment='lr-replay-v1', selection='five-case task and motion gates, then mean nominal/eval SO3 RMS',
                        quality_limits=training_profile(config)['quality_limits'],
                        foot_slip_note='loaded force/time metrics are distinct from geometric foot_slip_mps RMS')
    result={'identity':identity,'protocol':protocol,'cases':reports,'acceptance':acceptance,'numerical_noise':noise,'video_errors':video_errors}
    (output/'evaluation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def promote(baseline,candidate):
    if not baseline['acceptance']['success'] or not candidate['acceptance']['success']:return False
    cases=('nominal-01','eval-01')
    mean=lambda result,key:sum(result['cases'][c][key] for c in cases)/2
    key='root_orientation_rms_rad';base=mean(baseline,key)
    if base-mean(candidate,key)<max(.05*base,baseline['numerical_noise'].get(key,0)):return False
    for key in PROTOCOL['guardrails']:
        for case in baseline['cases']:
            b=baseline['cases'][case][key];c=candidate['cases'][case][key]
            if b is None or c is None or not math.isfinite(b+c):return False
            if c-b>max(.05*abs(b),baseline['numerical_noise'].get(key,0)):return False
    return True


def summarize(evaluations):
    formal=[r for r in evaluations if not r['identity'].get('diagnostic_only')]
    local = [r for r in formal if r['identity']['resolved_config'].get('local_training_profile')]
    if local:
        if len(local) != len(formal):
            raise ValueError('Do not mix LR/replay experiments with original CAM controls')
        rows = {}
        for result in local:
            cfg = result['identity']['resolved_config']
            variant = cfg['local_training_profile']['variant']
            if variant in rows:
                raise ValueError('Select one qualified attempt per LR/replay variant')
            rows[variant] = {'seed':cfg['seed'], 'accepted':result['acceptance']['success'],
                'cases':result['cases'], 'checkpoint_sha256':result['identity']['checkpoint_sha256']}
        return {'experiment':'lr-replay-v1','variants':rows,'complete':len(rows)==3,
                'independent_generalization_test':False}

    by_slot={}
    for result in formal:
        config=result['identity']['resolved_config']
        slot=(config.get('cam',{}).get('group','B0'),config['seed'])
        if slot in by_slot:raise ValueError('Multiple evaluations for one formal slot; explicitly select the qualified attempt')
        by_slot[slot]=result
    base=by_slot.get(('B0',0))
    candidates={group:by_slot[(group,0)] for group in ('B_ref','B_task') if (group,0) in by_slot}
    eligible=[g for g,r in candidates.items() if base is not None and promote(base,r)]
    mean=lambda result,key:sum(result['cases'][c][key] for c in ('nominal-01','eval-01'))/2
    eligible.sort(key=lambda g:(mean(candidates[g],'root_orientation_rms_rad'),mean(candidates[g],'foot_slip_mps'),mean(candidates[g],'action_rate_rms_per_s'),g!='B_ref'))
    selected=eligible[0] if eligible else None
    paired=[]
    if selected:
        for seed in (0,1,2):
            if ('B0',seed) in by_slot and (selected,seed) in by_slot:
                b=by_slot[('B0',seed)];c=by_slot[(selected,seed)]
                paired.append({'seed':seed,'baseline':mean(b,'root_orientation_rms_rad'),'candidate':mean(c,'root_orientation_rms_rad'),'passes_promotion_rule':promote(b,c)})
    return {'schema_version':1,'protocol':PROTOCOL,'formal_slots':len(by_slot),'selected_candidate':selected,'paired_seeds':paired,'baseline_passed':bool(base and base['acceptance']['success']),'conclusion':'baseline_failed' if base and not base['acceptance']['success'] else 'no_qualified_candidate' if len(candidates)==2 and not selected else 'pending' if not selected else 'three_seed_consistent' if len(paired)==3 and all(p['passes_promotion_rule'] for p in paired) else 'candidate_requires_validation','independent_generalization_test':False}
