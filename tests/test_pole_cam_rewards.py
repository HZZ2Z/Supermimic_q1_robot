import json
from dataclasses import replace
import pytest
import torch
from pole_task import COURSE_ROOT
from pole_task.cam_rewards import CAMConfig, CAMReward
from pole_task.source_recipe import resolve_source_recipe
from pole_task.source_runtime import SOURCE_REWARD_TUNING, SourcePoleSuperMimic
from pole_task.source_checkpoint import load_source_pole
from supermimic.reward_tuning import tuned_reward
from test_pole_centroidal import sample, direction_check


def cfg(group):
    return CAMConfig(group=group, enabled=group!='B0', target_mode={'B0':'none','B_ref':'reference','B_task':'task'}[group], tracking_weight=0 if group=='B0' else .2)


def test_b0_exact_reward_and_gradient_and_no_cam(monkeypatch):
    robot,s,c=sample(); x=s.clone().requires_grad_(); a=torch.randn(len(s),22,requires_grad=True)
    target=s[:,:57]+.01
    old=tuned_reward(robot,s,x,target,a,.02,tuning=SOURCE_REWARD_TUNING,context=c)
    reward=CAMReward(robot)
    assert reward.momentum is None
    actual=reward(robot,s,x,target,a,.02,tuning=SOURCE_REWARD_TUNING,context=c)
    assert torch.equal(actual,old)
    old_grad=torch.autograd.grad(old.sum(),(x,a),retain_graph=True)
    new_grad=torch.autograd.grad(actual.sum(),(x,a))
    assert all(torch.equal(a,b) for a,b in zip(old_grad,new_grad))


@pytest.mark.parametrize('group',['B_ref','B_task'])
def test_cam_target_dt_and_full_derivative(group):
    robot,s,c=sample(torch.float64)
    target=s[:,:57].roll(1,0).clone()
    reward=CAMReward(robot,cfg(group))
    a=torch.zeros(len(s),22,dtype=s.dtype)
    term=reward.terms(robot,s,target,c)
    if group=='B_ref':
        torch.testing.assert_close(term[1],reward.momentum(robot,target,c))
    else:
        task=s[:,:57].clone();task[:,29:57]=0;task[:,29:31]=target[:,29:31];task[:,34]=target[:,34]
        torch.testing.assert_close(term[1],reward.momentum(robot,task,c))
        x=s.clone().requires_grad_()
        assert torch.autograd.grad(reward.terms(robot,x,target,c)[1].square().sum(),x)[0][:,7:29].norm()>0
    base=tuned_reward(robot,s,s,target,a,.02,tuning=SOURCE_REWARD_TUNING,context=c)
    value=reward(robot,s,s,target,a,.02,tuning=SOURCE_REWARD_TUNING,context=c)
    torch.testing.assert_close(value-base,.02*.2*term[3])
    direction_check(lambda x: reward.terms(robot,x,target,c)[3],s)
    assert not torch.allclose(term[3],reward.terms(robot,s,target.roll(1,0),c)[3])


def test_actor_gradient_through_frozen_real_world_model():
    loaded=load_source_pole(COURSE_ROOT/'pole_task/source_manifest.json')
    robot=loaded.robot
    fixture=__import__('numpy').load(COURSE_ROOT/'pole_task/data/fixtures/source-policy-fixture.npz')
    s=torch.tensor(fixture['state']).unsqueeze(0); c=torch.tensor(fixture['context']).unsqueeze(0)
    history=torch.tensor(fixture['history']).unsqueeze(0); goal=torch.tensor(fixture['goal']).unsqueeze(0)
    reference=torch.tensor(fixture['references']).unsqueeze(0)[:,0]
    for p in loaded.world_model.parameters(): p.requires_grad_(False)
    action=loaded.actor(history,goal); predicted=loaded.world_model(s,action,c)
    reward=CAMReward(robot,cfg('B_task'))
    score=reward.terms(robot,predicted,reference,c)[3].sum()
    gradients=torch.autograd.grad(score,tuple(loaded.actor.parameters()),allow_unused=True)
    assert all(torch.isfinite(g).all() for g in gradients if g is not None)
    assert sum(g.square().sum().item() for g in gradients if g is not None)>0
    assert all(p.grad is None for p in loaded.world_model.parameters())


@pytest.mark.parametrize('batch',[512,256,128])
def test_resource_profile_is_explicit_and_keeps_verified_source(batch):
    p=COURSE_ROOT/'pole_task/config.json'; config=json.loads(p.read_text())
    original=resolve_source_recipe(p)
    config['local_resource_profile']={'schema_version':1,'batch_size':batch}
    actual=resolve_source_recipe(None,config=config)
    assert actual.model_batch_size==actual.policy_batch_size==batch
    assert actual.imagination_horizon==32
    assert actual.to_dict()['verified_source_recipe']==original.to_dict()
    # Exercise the same full-update consumer used by train and diagnostics.
    from pole_task.source_training import source_update
    sizes=[]
    class Replay:
        def sample(self,size,**kwargs): sizes.append(size);return None
    class Optimizer: param_groups=[{}]
    class Algorithm:
        model_optimizer=Optimizer();actor_optimizer=Optimizer();critic_optimizer=Optimizer()
        def fit_model(self,batch): return {}
        def improve_policy(self,batch,*args,**kwargs): return {}
    source_update(Algorithm(),Replay(),actual,2001)
    assert sizes==[batch]*11


@pytest.mark.parametrize('values',[{'tracking_weigth':1},{'target_gradient':'detach'},{'damping_weight':.1},{'momentum_scale':float('nan')},{'tracking_sigma':0}])
def test_invalid_cam_rejected(values):
    with pytest.raises(ValueError): CAMConfig.parse(values)


def test_invalid_resource_override_rejected():
    config=json.loads((COURSE_ROOT/'pole_task/config.json').read_text())
    for profile in ({'schema_version':1,'batch_size':64},{'schema_version':1,'batch_size':128,'horizon':16}):
        config['local_resource_profile']=profile
        with pytest.raises(ValueError,match='local_resource_profile'):resolve_source_recipe(None,config=config)
    config.pop('local_resource_profile'); config['source_training']['model_batch_size']=128
    with pytest.raises(ValueError):resolve_source_recipe(None,config=config)


def test_reference_course_columns_do_not_enter_cam():
    robot,s,c=sample()
    reward=CAMReward(robot,cfg('B_ref'))
    reference=s[:,:57]
    paired=torch.cat((reference,torch.ones(len(s),6)*123),-1)
    torch.testing.assert_close(reward.terms(robot,s,paired,c)[1],reward.terms(robot,s,reference,c)[1])


def test_calibration_objectives_do_not_update_parameters(tmp_path):
    from test_pole_source_runtime import _motions, ZeroActor
    from pole_task.data import load_course
    from pole_task.source_runtime import run_source_episode, make_source_replay
    from pole_task.models import build_source_models
    robot,motions=_motions(tmp_path,[load_course('nominal-01')])
    actor,world,critic=build_source_models(robot)
    episode=run_source_episode(ZeroActor(),robot,motions,motions.cases[0],0,tmp_path/'scene.xml',start_frame=0,max_steps=24)
    replay=make_source_replay([episode],robot,{'nominal-01':0})
    algorithm=SourcePoleSuperMimic(robot,motions,world,actor,critic,horizon=2,cam_config=cfg('B_task'),reward_tuning=SOURCE_REWARD_TUNING)
    before=[p.clone().detach() for module in (actor,world,critic) for p in module.parameters()]
    result=algorithm.improve_policy(replay.sample(2,window=1,history=10),diagnostic_objectives=True)
    grads=torch.autograd.grad(result['cam'],result['actions'])
    assert all(torch.isfinite(g).all() for g in grads)
    assert all(torch.equal(p,old) for p,old in zip((p for m in (actor,world,critic) for p in m.parameters()),before))
    assert not algorithm.cam_reward.capture_terms
