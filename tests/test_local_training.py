"""Local recipe, V1 sampling equivalence, quality gates and replay retention."""
import ast
import copy
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
import torch
from pole_task import COURSE_ROOT, ROOT
from pole_task.local_training import (training_profile, stratified_start_frame, merge_replay_batches,
    restore_replay_batches, quality_failures, motion_metrics, selection_score)
from pole_task.source_recipe import resolve_source_recipe
from pole_task.source_training import source_acceptance, _collect_episodes, TrainingBudget
from supermimic.replay import TemporalReplay

CONFIG = COURSE_ROOT/'pole_task/config.json'


def config(variant='control'):
    value=json.loads(CONFIG.read_text())
    value['local_training_profile']={'schema_version':1,'variant':variant}
    return value


@pytest.mark.parametrize('variant,start,end',[('control',1e-5,1e-5),('low_lr',3e-6,3e-6),('actor_cosine',1e-5,1e-6)])
def test_actor_clock_and_verified_source(variant,start,end):
    original=resolve_source_recipe(CONFIG)
    recipe=resolve_source_recipe(None,config=config(variant))
    assert recipe.to_dict()['verified_source_recipe']==original.to_dict()
    assert recipe.optimizer_rates(2000)['actor']==pytest.approx(start)
    assert recipe.optimizer_rates(2499)['actor']==pytest.approx(end)
    assert recipe.optimizer_rates(0)['actor']==pytest.approx(start)
    for step in (0,1999,2000,2074,2499):
        for name in ('model','critic'):
            assert recipe.optimizer_rates(step)[name]==original.optimizer_rates(step)[name]
    rates=[recipe.optimizer_rates(i)['actor'] for i in range(2000,2500)]
    assert all(a>=b for a,b in zip(rates,rates[1:]))


def test_profile_rejects_unknown_variants_and_reward_changes():
    c=config();c['local_training_profile']['period']=500
    with pytest.raises(ValueError):resolve_source_recipe(None,config=c)
    c=config('unknown')
    with pytest.raises(ValueError):resolve_source_recipe(None,config=c)
    c=config();c['cam']={'group':'B_ref','enabled':True,'target_mode':'reference','tracking_weight':.1}
    with pytest.raises(ValueError,match='CAM disabled'):resolve_source_recipe(None,config=c)
    c=config();c['source_training']['actor_learning_rate']=3e-6
    with pytest.raises(ValueError):resolve_source_recipe(None,config=c)


@pytest.mark.parametrize('count',[2,4])
def test_four_strata_cover_two_batches_and_match_v1(count):
    last=298;bins=np.array_split(np.arange(1,last-9),4)
    visited=[]
    for batch in range(2):
        for index in range(count):
            frame=stratified_start_frame(last,batch,count,index,704+index)
            expected=(batch*count+index)%4
            assert frame in bins[expected]
            assert frame==stratified_start_frame(last,batch,count,index,704+index)
            visited.append(expected)
    assert set(visited)==set(range(4))
    # The checked-in port has the same result as V1's function, without importing V1 packages.
    path=ROOT.parent/'V1/courses/pole-weaving/pole_task/source_runtime.py'
    if path.exists():
        tree=ast.parse(path.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='stratified_start_frame')
        ns={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
        for last in (198,298,675):
            for b in range(4):
                for i in range(count):
                    assert stratified_start_frame(last,b,count,i,19+i)==ns['stratified_start_frame'](last,b,count,i,19+i)


def replay_state(mark,count=24):
    r=TemporalReplay(capacity=count,num_envs=1,state_dim=79,context_dim=149,obs_dim=78,device='cpu')
    for name in r._storage_names():
        getattr(r,name).zero_()
    r.states.fill_(mark);r.observations.fill_(mark);r.frames[:,0]=torch.arange(count)
    r.boundaries[-1]=True;r.count=count;r.cursor=0
    return r.state_dict()


def test_two_batches_expire_and_resume_without_cross_boundary_windows():
    a,b,c=(replay_state(i) for i in (1,2,3))
    batches=[(2001,a),(2026,b)]
    replay=merge_replay_batches(batches,'cpu')
    runtime={'replay':replay.state_dict(),'replay_batches':[{'update':u,'count':s['count']} for u,s in batches]}
    restored=restore_replay_batches(runtime)
    replay=merge_replay_batches(restored[-1:]+[(2051,c)],'cpu')
    assert replay.count==48 and set(replay.states[:48,0,0].tolist())=={2.,3.}
    torch.manual_seed(7);batch=replay.sample(512,window=8,history=10)
    assert torch.all(batch.states[:,:,0]==batch.history[:,-1:,0])
    runtime['replay_batches'][0]['count']+=1
    with pytest.raises(ValueError):restore_replay_batches(runtime)


def good_row():
    return {'route_completed':True,'pole_collision':False,'fall':False,'endpoint_distance_m':.1,'root_xy_rmse_m':.1,
        'foot_contact_time_s':[1.,1.],'foot_slip_mean_m_s':.03,'foot_slip_fraction':.05,
        'tilt_error_rms_deg':2.,'pd_acceleration_ratio':1.2,'root_orientation_rms_rad':.1}


def test_route_pass_cannot_hide_bad_motion_quality():
    c=config();profile=training_profile(c)
    cases={name:good_row() for name in ['nominal-01','eval-01','straight','left_turn','right_turn']}
    assert source_acceptance(cases,c)
    assert selection_score(cases,profile)==pytest.approx(.1)
    for field,value in [('foot_slip_mean_m_s',.2),('foot_slip_fraction',.3),('tilt_error_rms_deg',6.),('pd_acceleration_ratio',3.),('root_orientation_rms_rad',float('nan'))]:
        bad=copy.deepcopy(cases);bad['left_turn'][field]=value
        assert not source_acceptance(bad,c)
    missing=copy.deepcopy(cases);missing['nominal-01'].pop('pd_acceleration_ratio')
    assert not source_acceptance(missing,c)


def test_motion_metric_units_and_reference_tilt():
    states=np.zeros((21,79));states[:,3]=1
    refs=states[1:,:57].copy()
    actions=np.zeros((20,22))
    row=motion_metrics(states,refs,actions,.02)
    assert row['tilt_error_rms_deg']==0 and row['pd_acceleration_ratio']==0
    actions[:,0]=np.arange(20)**2*.01
    row=motion_metrics(states,refs,actions,.02)
    expected=.5*.02/.02**2/np.sqrt(22)
    assert row['pd_target_acceleration_rms_rad_s2']==pytest.approx(expected)
    assert row['pd_acceleration_ratio']==pytest.approx(expected/10)


def test_contact_instrumentation_does_not_change_physics(tmp_path):
    from test_pole_source_runtime import _motions, ZeroActor
    from pole_task.data import load_course
    from pole_task.source_runtime import run_source_episode
    robot,motions=_motions(tmp_path,[load_course('nominal-01')])
    kwargs=dict(start_frame=0,seed=10,max_steps=24)
    original=run_source_episode(ZeroActor(),robot,motions,motions.cases[0],0,tmp_path/'off.xml',**kwargs)
    robot.motion_quality_profile=training_profile(config())
    measured=run_source_episode(ZeroActor(),robot,motions,motions.cases[0],0,tmp_path/'on.xml',**kwargs)
    np.testing.assert_array_equal(original.states,measured.states)
    np.testing.assert_array_equal(original.rewards,measured.rewards)
    assert 'foot_slip_mean_m_s' in measured.result
    assert 'motion_quality_failures' in measured.result


def test_collection_uses_four_strata_only_in_actor_stage(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import pole_task.source_training as training
    recipe=resolve_source_recipe(None,config=config())
    case=SimpleNamespace(case_id='nominal-01',control_steps=298)
    starts=[]
    def episode(*args,**kwargs):
        starts.append(kwargs['start_frame'])
        assert kwargs['measure_quality'] is False
        return SimpleNamespace(case=case,start_frame=kwargs['start_frame'],steps=24,
                               frames=np.arange(24),boundaries=np.r_[np.zeros(23,dtype=bool),True])
    monkeypatch.setattr(training,'run_source_episode',episode)
    budget=TrainingBudget(1000)
    for update in (1,2001,2026):
        _collect_episodes(None,None,SimpleNamespace(cases=[case]),[case],tmp_path,
            stage=recipe.stage_for_update(update),seed=1000+update,device='cpu',
            episode_counts={'nominal-01':4},action_noise=0.,budget=budget,
            profile=recipe.local_training_profile,collection_update=update)
    assert starts[:4]==[None]*4
    for offset,update in ((4,2001),(8,2026)):
        assert starts[offset:offset+4]==[stratified_start_frame(298,(update-2001)//25,4,i,1000+update+i) for i in range(4)]
    assert budget.steps==288
    rows=[json.loads(line) for line in (tmp_path/'collection.jsonl').read_text().splitlines()]
    assert len(rows)==12 and len(rows[-1]['valid_window_frames']['policy'])==15


def test_profile_identity_and_wandb_quality_metrics(tmp_path):
    from pole_task.experiment import semantic_hash, checkpoint_config
    from pole_task.source_training import flat_metrics
    c=config()
    payload={'config':c,'identity':{'experiment_schema_version':2,'semantic_config_sha256':semantic_hash(c)}}
    path=tmp_path/'low.json';path.write_text(json.dumps(config('low_lr')))
    with pytest.raises(ValueError,match='conflicts'):checkpoint_config(payload,path)
    metrics=flat_metrics({'evaluation':{'nominal-01':good_row()},'best_score':float('inf'),
                          'sampled_windows':{'model':{'left_turn':128}},'stage':'actor'})
    assert metrics['evaluation/nominal-01/tilt_error_rms_deg']==2
    assert metrics['sampled_windows/model/left_turn']==128
    assert 'best_score' not in metrics and 'stage' not in metrics


def test_resume_does_not_import_future_best_and_hashes_warmup(tmp_path):
    from pole_task.source_training import carry_best_checkpoint, checkpoint_output_hashes
    parent=tmp_path/'parent.pt';target=tmp_path/'resumed';target.mkdir()
    torch.save({'update':2075,'best_score':.1},parent)
    carry_best_checkpoint(parent,target,2000,.2)
    assert not (target/'best.pt').exists()
    carry_best_checkpoint(parent,target,2100,.1)
    assert (target/'best.pt').read_bytes()==parent.read_bytes()
    (target/'warmup.pt').write_bytes(parent.read_bytes())
    hashes=checkpoint_output_hashes(target)
    assert hashes['warmup_checkpoint_sha256']==hashes['best_checkpoint_sha256']


@pytest.mark.parametrize('cut',[2,4,5])
def test_actual_resume_preserves_refresh_rng_and_two_batches(tmp_path,monkeypatch,cut):
    import pole_task.source_training as training
    from pole_task.source_recipe import SourceStage
    from pole_task.source_runtime import run_source_episode
    torch.set_num_threads(1)
    c=config('actor_cosine');c.update(device='cpu',output_root=str(tmp_path),evaluation_interval=1)
    path=tmp_path/'config.json';path.write_text(json.dumps(c))
    original=resolve_source_recipe(None,config=c)
    recipe=replace(original,stages=(SourceStage('warmup',2,1,2,False,False,False),SourceStage('actor',6,3,8,True,True,True)),
        model_batch_size=2,policy_batch_size=2,model_batches=1,critic_epochs=1,critic_minibatches=1,imagination_horizon=2,
        local_training_profile={**original.local_training_profile,'actor_collection_interval':2})
    monkeypatch.setattr(training,'resolve_source_recipe',lambda *a,**kw:recipe)
    monkeypatch.setattr(training,'preflight',lambda *a:{'training_ready':True,'hardware':{'ready':True}})
    calls=[]
    def collect(actor,robot,motions,cases,run_dir,*,stage,seed,device,budget,collection_update,**kwargs):
        calls.append(collection_update)
        ep=run_source_episode(actor,robot,motions,cases[0],0,run_dir/'collection.xml',
            start_frame=0,max_steps=24,seed=seed,device=device,randomize=stage.randomize,measure_quality=False)
        budget.charge(ep.steps);return [ep]
    monkeypatch.setattr(training,'_collect_episodes',collect)
    monkeypatch.setattr(training,'_evaluate',lambda *a,**kw:{name:{**good_row(),'steps':0} for name in ['nominal-01','eval-01','straight','left_turn','right_turn']})
    continuous=training.train_source(path,run_dir=tmp_path/'continuous')
    partial=training.train_source(path,max_updates=cut,run_dir=tmp_path/'part')
    resumed=training.train_source(path,partial/'latest.pt',run_dir=tmp_path/'resumed')
    a=torch.load(continuous/'latest.pt',weights_only=True);b=torch.load(resumed/'latest.pt',weights_only=True)
    def equal(x,y):
        if isinstance(x,torch.Tensor):assert torch.equal(x,y)
        elif isinstance(x,dict):
            assert x.keys()==y.keys()
            for k in x:equal(x[k],y[k])
        elif isinstance(x,(tuple,list)):
            assert len(x)==len(y)
            for u,v in zip(x,y):equal(u,v)
        else:assert x==y
    for key in ('actor','model','critic','actor_optimizer','model_optimizer','critic_optimizer','runtime'):
        equal(a[key],b[key])
    assert calls==[1,3,5,7]*2
    assert [r['update'] for r in a['runtime']['replay_batches']]==[5,7]
    assert a['runtime']['collection_environment_steps']==96
    assert len(list(continuous.glob('*.pt')))==3
