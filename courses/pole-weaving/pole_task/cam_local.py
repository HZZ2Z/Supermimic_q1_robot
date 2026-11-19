"""Bounded local CAM operations over the existing source trainer."""
import argparse
import fcntl
from dataclasses import replace
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback
import random
import numpy as np
import torch
from . import ROOT, project_path
from .cam_rewards import CAMConfig, CAMReward
from .data import PoleMotions, write_vendor_motion
from .experiment import source_identity, semantic_hash, atomic_checkpoint, capture_rng, restore_rng
from .source_checkpoint import load_source_pole
from .source_recipe import resolve_source_recipe, freeze_pole_input_columns
from .source_runtime import SourcePoleSuperMimic, SOURCE_REWARD_TUNING, run_source_episode, make_source_replay
from .source_training import source_update, _make_optimizers, train_source, load_training_cases, atomic_json, _collect_episodes, TrainingBudget
from .local_training import training_profile, merge_replay_batches, VARIANTS
from .experiment import file_hash, environment_identity


def diagnostic(config_path, output, *, profile=False, device='cuda:0'):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    config=json.loads(Path(config_path).read_text())
    recipe=resolve_source_recipe(config_path)
    random.seed(0);np.random.seed(0);torch.manual_seed(0)
    loaded=load_source_pole(project_path(config['source_manifest']),device='cpu')
    robot=loaded.robot.to(device); actor=loaded.actor.to(device); world=loaded.world_model.to(device); critic=loaded.critic.to(device)
    training,_,cases=load_training_cases(config,manifest_path=project_path(config['manifest']))
    motions=PoleMotions(robot,cases,[write_vendor_motion(c,output/'references'/f'{c.case_id}.npz') for c in cases],device=device);motions.prepare()
    effective=recipe if profile else replace(recipe,model_batch_size=4,policy_batch_size=4,model_batches=1,critic_epochs=1,critic_minibatches=1,imagination_horizon=2)
    algorithm=SourcePoleSuperMimic(robot,motions,world,actor,critic,horizon=effective.imagination_horizon,reward_tuning=SOURCE_REWARD_TUNING,cam_config=config.get('cam'),clean_policy_actions=recipe.clean_policy_actions,normalize_policy_objective=True,normalize_model_objective=True)
    _make_optimizers(algorithm,recipe)
    by_id={c.case_id:i for i,c in enumerate(cases)}
    result={'diagnostic_only':True,'config':config,'effective_recipe':effective.to_dict(),'source':source_identity(),'device':device,'phases':[]}
    # Compare the same physical samples on both devices before any update.
    fixture=np.load(ROOT/'courses/pole-weaving/pole_task/data/fixtures/source-policy-fixture.npz')
    state=torch.tensor(fixture['state']).unsqueeze(0); context=torch.tensor(fixture['context']).unsqueeze(0); target=torch.tensor(fixture['references']).unsqueeze(0)[:,0]
    cpu_robot=load_source_pole(project_path(config['source_manifest']),device='cpu').robot
    cpu_reward=CAMReward(cpu_robot,config.get('cam'))
    if cpu_reward.momentum:
        a=cpu_reward.terms(cpu_robot,state,target,context)
        b=algorithm.cam_reward.terms(robot,state.to(device),target.to(device),context.to(device))
        result['cpu_gpu_max_error']=max(float((x-y.cpu()).abs().max()) for x,y in zip(a,b))
        for x,y in zip(a,b):torch.testing.assert_close(x,y.cpu(),rtol=1e-5,atol=1e-6)
        from supermimic.models import frozen
        with frozen(world):
            action=actor(torch.tensor(fixture['history'],device=device).unsqueeze(0),torch.tensor(fixture['goal'],device=device).unsqueeze(0))
            predicted=world(state.to(device),action,context.to(device))
            score=algorithm.cam_reward.terms(robot,predicted,target.to(device),context.to(device))[3].sum()
            gradient=torch.autograd.grad(score,action)[0]
            result['cam_action_gradient_l2']=float(gradient.norm())
            if not torch.isfinite(gradient).all() or gradient.norm()<=0:raise AssertionError('CAM gradient does not reach GPU actions')
        del action,predicted,score,gradient
    del cpu_robot,cpu_reward
    timings={}
    for name in ('fit_model','_fit_critic','_update_actor'):
        original=getattr(algorithm,name)
        def timed(*args,_original=original,_name=name,**kwargs):
            if device.startswith('cuda'):torch.cuda.synchronize()
            start=time.monotonic()
            result=_original(*args,**kwargs)
            if device.startswith('cuda'):torch.cuda.synchronize()
            timings[_name]=timings.get(_name,0)+time.monotonic()-start
            return result
        setattr(algorithm,name,timed)
    batches=[]
    local=recipe.local_training_profile
    updates=(1,2001,2026) if local is not None else (1,2001)
    for update in updates:
        timings.clear()
        stage=recipe.stage_for_update(update)
        if stage.freeze_pole_input_columns:freeze_pole_input_columns(actor)
        if local is not None:
            episodes=_collect_episodes(actor,robot,motions,training,output,stage=stage,seed=1000+update,
                device=device,episode_counts={c.case_id:4 if c.case_id=='nominal-01' else 2 for c in training},
                action_noise=0.,budget=TrainingBudget(10000),profile=local,collection_update=update,max_steps=64)
            fresh=make_source_replay(episodes,robot,by_id,device=device)
            batches=(batches[-1:] if update>2001 else [])+[(update,fresh.state_dict())]
            replay=merge_replay_batches(batches,device)
            del fresh
        else:
            episodes=[]
            for index,case in enumerate(training):
                episodes.append(run_source_episode(actor,robot,motions,case,by_id[case.case_id],output/'scene.xml',start_frame=0,seed=100+index,device=device,max_steps=64,observation_noise=True,randomize=stage.randomize))
            replay=make_source_replay(episodes,robot,by_id,device=device)
        before=[p.detach().clone() for p in actor.parameters()]
        if device.startswith('cuda'):
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        started=time.monotonic()
        metrics=source_update(algorithm,replay,effective,update)
        if device.startswith('cuda'):torch.cuda.synchronize()
        elapsed=time.monotonic()-started
        delta=sum(float((p.detach()-old).square().sum()) for p,old in zip(actor.parameters(),before))**.5
        if not all(np.isfinite(v) for v in metrics.values() if isinstance(v,(int,float))):raise FloatingPointError('Nonfinite diagnostic metrics')
        if stage.actor_updates_enabled:
            if delta<=0 or metrics['actor_grad_norm']<=0:raise AssertionError('Actor did not update')
        elif delta!=0:raise AssertionError('Warmup changed Actor')
        row={'stage':stage.name,'metrics':metrics,'actor_parameter_delta_l2':delta,'seconds':elapsed,'component_seconds':dict(timings),'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
        if device.startswith('cuda'):row.update(peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,free_mib=torch.cuda.mem_get_info()[0]/1024**2)
        row.update(update=update,replay_transitions=replay.count,replay_batch_updates=[u for u,_ in batches])
        result['phases'].append(row);print(json.dumps(row),flush=True)
        del replay,episodes,before
    result['status']='RESOURCE_PROFILED' if profile else 'GPU_SMOKE_PASS' if device.startswith('cuda') else 'CPU_PASS'
    atomic_json(output/'diagnostic.json',result)
    return result


def validate_readiness(config, config_path):
    local=training_profile(config)
    expected=ROOT/'logs/q1-pole-weaving/lr-replay-v1'/local['variant']/'seed-0'
    if config['seed']!=0 or project_path(config['output_root']).resolve()!=expected:
        raise ValueError('LR/replay pilot has exactly three registered seed=0 slots')
    receipt=ROOT/'logs/q1-pole-weaving/lr-replay-v1/readiness.json'
    if not receipt.is_file():
        raise ValueError('Run scripts/prepare_lr_replay.py before formal LR/replay training')
    ready=json.loads(receipt.read_text())
    if (ready.get('status')!='COMPLETE' or ready.get('source_sha256')!=source_identity()['sha256']
            or ready.get('variants')!=list(VARIANTS)):
        raise ValueError('Resource readiness does not match the current source/variants')
    if ready.get('environment')!=environment_identity():
        raise ValueError('Dependencies changed after resource validation')
    for variant in VARIANTS:
        path=ROOT/'configs/lr-replay-v1'/f'{variant}.json'
        if file_hash(path)!=ready['configuration_hashes'].get(variant):
            raise ValueError('Configuration changed after full three-variant resource validation')
    if file_hash(config_path)!=ready['configuration_hashes'][local['variant']]:
        raise ValueError('Formal config differs from resource validation')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=['smoke','profile','train','evaluate','summarize'])
    parser.add_argument('--config',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--device')
    parser.add_argument('--max-updates',type=int)
    args=parser.parse_args(argv)
    if args.operation in ('smoke','profile','train') and args.config is None:
        parser.error('--config is required for smoke/profile/train')
    if args.operation=='evaluate' and args.checkpoint is None:
        parser.error('--checkpoint is required for evaluate')
    if args.config is not None:args.config=project_path(args.config).resolve()
    if args.checkpoint is not None:args.checkpoint=project_path(args.checkpoint).resolve()
    torch.set_num_threads(4)
    output=project_path(args.output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    lock_path=ROOT/'logs/q1-pole-weaving/cam-local/.compute.lock'
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    compute_lock=lock_path.open('a')
    if args.operation!='summarize':
        fcntl.flock(compute_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.operation=='train':
        config=json.loads(args.config.read_text())
        if output.parent!=project_path(config['output_root']).resolve():parser.error('Formal output must be one attempt directory below configured group/seed root')
        if args.max_updates is not None and args.max_updates!=2500:parser.error('Use smoke/profile for diagnostics; formal update limit is 2500')
        if args.device is not None and args.device!=config.get('device','cpu'):parser.error('Training --device conflicts with frozen config')
        local=training_profile(config)
        group=config.get('cam',{}).get('group','B0');seed=config.get('seed')
        if local is not None:
            validate_readiness(config,args.config)
        if group not in ('B0','B_ref','B_task') or seed not in (0,1,2):parser.error('Outside the seven predefined group/seed slots')
        if seed in (1,2):
            promotion=json.loads(project_path(config['promotion_record']).read_text())
            if not promotion.get('passed') or group not in ('B0',promotion.get('candidate')):parser.error('Seed expansion requires the selected promoted candidate')
        attempts=[json.loads(p.read_text()) for p in output.parent.glob('*.run.json')]
        attempts=[r for r in attempts if r.get('operation')=='train']
        if len(attempts)>=2:parser.error('Slot has exhausted its one retry')
        if attempts and not (output.parent/'retry-justification.json').is_file():parser.error('Retry needs recorded root cause and semantic-equivalence evidence')
    if args.operation!='summarize' and output.exists():parser.error('Output already exists; select a new attempt directory')
    operation_id=output.name if args.operation!='summarize' else output.name+'-summary-'+str(time.time_ns())
    record_path=output.parent/(operation_id+'.run.json')
    record={'run_id':str(output.relative_to(ROOT)) if output.is_relative_to(ROOT) else str(output),'pid':os.getpid(),'command':[sys.executable,*sys.argv],'cwd':str(Path.cwd()),'python':sys.executable,'operation':args.operation,'started_unix':time.time(),'status':'IN_PROGRESS','config':str(args.config.resolve()) if args.config else None}
    atomic_json(record_path,record)
    log=output.parent/(operation_id+'.log')
    # The process itself owns the log, including tracebacks and W&B output.
    with log.open('x',buffering=1) as stream:
        os.dup2(stream.fileno(),1);os.dup2(stream.fileno(),2)
        try:
            if args.operation in ('smoke','profile'):
                diagnostic(args.config,output,profile=args.operation=='profile',device=args.device or 'cuda:0')
            elif args.operation=='train':
                train_source(args.config,args.checkpoint,max_updates=args.max_updates,run_dir=output)
            elif args.operation=='evaluate':
                from .cam_evaluation import evaluate_checkpoint
                evaluate_checkpoint(args.checkpoint,output,device=args.device or 'cuda:0',config_path=args.config)
            else:
                rows=[json.loads(p.read_text()) for p in output.rglob('evaluation.json')]
                from .cam_evaluation import summarize
                atomic_json(output/'summary.json',summarize(rows))
            record.update(status='COMPLETE',exit_code=0)
        except BaseException as exc:
            record.update(status='FAILED',exit_code=1,error=repr(exc));traceback.print_exc();raise
        finally:
            record.update(finished_unix=time.time(),log=str(log));atomic_json(record_path,record)
    return 0


if __name__=='__main__':raise SystemExit(main())
