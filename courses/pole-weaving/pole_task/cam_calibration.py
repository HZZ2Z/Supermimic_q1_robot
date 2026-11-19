"""One fixed offline calibration; never performs an optimizer update."""
import json
import math
from pathlib import Path
import numpy as np
import torch
from .cam_rewards import CAMConfig, CAMReward
from .centroidal import CentroidalMomentum
from .source_playback import load_course_bundle
from .inference import load_pole_actor
from .source_runtime import SourcePoleSuperMimic, SOURCE_REWARD_TUNING
from .models import build_source_models
from .experiment import file_hash, capture_rng, restore_rng
from supermimic.replay import TemporalReplay


def calibrate(checkpoint, evaluation_dir, output, device='cuda:0'):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    actor,identity=load_pole_actor(checkpoint,device)
    config=identity['resolved_config']
    if config.get('cam',{}).get('group','B0')!='B0':raise ValueError('Calibration requires B0')
    robot,motions,cases,_=load_course_bundle(reference_dir=output/'references',device=device,config=config)
    _,model,critic=build_source_models(robot)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict(saved['model'],strict=True);critic.load_state_dict(saved['critic'],strict=True)
    model.to(device);critic.to(device)
    data=[];inputs={}
    for name in config['training_case_ids']:
        path=Path(evaluation_dir)/f'{name}-trajectory.npz';inputs[name]=file_hash(path)
        with np.load(path) as values:
            data.append(tuple(torch.tensor(values[k],device=device) for k in ('states','references','contexts')))
    cam=CentroidalMomentum(robot).to(device)
    reference_z=torch.cat([cam(robot,r[..., :57],c)[:,2] for _,r,c in data])
    scale=max(float(reference_z.square().mean().sqrt()),1e-6)
    errors=[]
    for group,mode in [('B_ref','reference'),('B_task','task')]:
        reward=CAMReward(robot,CAMConfig(group=group,enabled=True,target_mode=mode,momentum_scale=scale,tracking_weight=1))
        errors.extend(reward.terms(robot,s[1:],r,c)[2].square() for s,r,c in data)
    sigma=max(float(torch.cat(errors).median())/math.log(2),1e-4)
    state=saved['runtime']['replay']
    replay=TemporalReplay(capacity=state['capacity'],num_envs=1,state_dim=79,context_dim=149,obs_dim=78,device=device)
    replay.load_state_dict(state)
    torch.manual_seed(6021)
    batch=replay.sample(128,window=1,history=10)
    torch.save({'states':batch.states,'frames':batch.frames,'clips':batch.clips,'history':batch.history,'context':batch.context},output/'calibration-batch.pt')
    rng=capture_rng();rows={};weights=[]
    for group,mode in [('B_ref','reference'),('B_task','task')]:
        restore_rng(rng)
        cfg=CAMConfig(group=group,enabled=True,target_mode=mode,momentum_scale=scale,tracking_sigma=sigma,tracking_weight=1)
        algorithm=SourcePoleSuperMimic(robot,motions,model,actor,critic,horizon=32,reward_tuning=SOURCE_REWARD_TUNING,cam_config=cfg,clean_policy_actions=True,normalize_policy_objective=True,normalize_model_objective=True)
        objectives=algorithm.improve_policy(batch,diagnostic_objectives=True)
        base_grad=torch.autograd.grad(objectives['base'],objectives['actions'],retain_graph=True)
        cam_grad=torch.autograd.grad(objectives['cam'],objectives['actions'])
        rms=lambda values:float(torch.cat([v.detach().flatten() for v in values]).square().mean().sqrt())
        gbase,gcam=rms(base_grad),rms(cam_grad)
        rbase,rcam=abs(float(objectives['base'].detach())),abs(float(objectives['cam'].detach()))
        if not all(math.isfinite(x) and x>0 for x in (gbase,gcam,rbase,rcam)):
            raise ValueError('Degenerate calibration reward/action gradient; investigate before training')
        weight=min(.1*rbase/rcam,.1*gbase/gcam)
        rows[group]={'base_reward_abs':rbase,'unit_cam_reward_abs':rcam,'base_action_gradient_rms':gbase,'unit_cam_action_gradient_rms':gcam,'maximum_weight':weight}
        weights.append(weight)
        del objectives,base_grad,cam_grad,algorithm
    weight=min(weights)
    result={'schema_version':1,'checkpoint_sha256':file_hash(checkpoint),'inputs':inputs,'momentum_scale':scale,'tracking_sigma':sigma,'tracking_weight':weight,'groups':rows,'optimizer_updates':0,'evaluation_case_used':False,'batch':128,'horizon':32}
    (output/'calibration.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result
