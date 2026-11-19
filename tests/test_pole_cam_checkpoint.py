import copy
import json
import random
import numpy as np
import pytest
import torch
from pole_task import COURSE_ROOT
from pole_task.experiment import capture_rng, restore_rng, semantic_hash, checkpoint_config
from pole_task.source_training import validate_resume_payload, checkpoint_payload
from pole_task.source_recipe import resolve_source_recipe
from test_pole_source_training import _payload, CONFIG, MANIFEST, SOURCE_MANIFEST


@pytest.mark.parametrize('update',[1999,2000,2001,2500])
def test_completed_and_next_stage_boundaries(update):
    recipe=resolve_source_recipe(CONFIG)
    p=_payload(recipe,update=update,stage=recipe.stage_for_update(update).name)
    kw=dict(config=p['config'],config_path=CONFIG,recipe=recipe,manifest_path=MANIFEST,source_manifest_path=SOURCE_MANIFEST)
    assert validate_resume_payload(p,**kw)==update
    p.update(checkpoint_schema_version=2,completed_updates=update,completed_stage=p['stage'],next_stage=recipe.stage_for_update(update+1).name if update<2500 else 'complete')
    assert validate_resume_payload(p,**kw)==update
    p['next_stage']='wrong'
    with pytest.raises(ValueError,match='stage'): validate_resume_payload(p,**kw)


def test_rng_serializable_roundtrip_and_all_random_sources(tmp_path):
    state=capture_rng();path=tmp_path/'rng.pt';torch.save(state,path)
    expected=(random.random(),np.random.rand(),torch.rand(5))
    restore_rng(torch.load(path,weights_only=True))
    actual=(random.random(),np.random.rand(),torch.rand(5))
    assert expected[:2]==actual[:2];assert torch.equal(expected[2],actual[2])


def test_checkpoint_owns_config_and_rejects_explicit_conflict(tmp_path):
    config=json.loads(CONFIG.read_text())
    config['cam']={'group':'B_task','enabled':True,'target_mode':'task','tracking_weight':.1}
    saved={'config':config,'identity':{'experiment_schema_version':2,'semantic_config_sha256':semantic_hash(config)}}
    assert checkpoint_config(saved)['cam']['target_mode']=='task'
    explicit=copy.deepcopy(config);explicit['output_root']='logs/q1-pole-weaving/elsewhere'
    path=tmp_path/'config.json';path.write_text(json.dumps(explicit))
    assert checkpoint_config(saved,path)==explicit
    explicit['cam']['tracking_weight']=.2;path.write_text(json.dumps(explicit))
    with pytest.raises(ValueError,match='conflicts'):checkpoint_config(saved,path)
    explicit=copy.deepcopy(config);explicit['local_resource_profile']={'schema_version':1,'batch_size':128}
    assert semantic_hash(explicit)!=semantic_hash(config)

@pytest.mark.parametrize('cut',[1,2])
def test_actual_training_resume_matches_continuous_across_stage(tmp_path,monkeypatch,cut):
    from dataclasses import replace
    import pole_task.source_training as training
    from pole_task.source_recipe import SourceStage
    from pole_task.source_runtime import run_source_episode
    torch.set_num_threads(1)
    recipe=replace(resolve_source_recipe(CONFIG),stages=(SourceStage('warmup',2,1,2,False,False,False),SourceStage('actor',2,3,4,True,True,True)),model_batch_size=2,policy_batch_size=2,model_batches=1,critic_epochs=1,critic_minibatches=1,imagination_horizon=2)
    import os
    device=os.environ.get('CAM_RESUME_TEST_DEVICE','cpu')
    config=json.loads(CONFIG.read_text());config.update(device=device,output_root=str(tmp_path),evaluation_interval=1)
    config_path=tmp_path/'config.json';config_path.write_text(json.dumps(config))
    monkeypatch.setattr(training,'resolve_source_recipe',lambda *a,**k:recipe)
    monkeypatch.setattr(training,'preflight',lambda *a:{'training_ready':True,'hardware':{'ready':True}})
    collections=[]
    def collect(actor,robot,motions,cases,run_dir,*,stage,seed,device,budget,**kwargs):
        collections.append(stage.name)
        ep=run_source_episode(actor,robot,motions,cases[0],0,run_dir/'collection.xml',start_frame=0,max_steps=24,seed=seed,device=device,randomize=stage.randomize)
        budget.charge(ep.steps);return [ep]
    monkeypatch.setattr(training,'_collect_episodes',collect)
    def evaluate(*a,**kw):
        return {name:{'route_completed':True,'pole_collision':False,'fall':False,'endpoint_distance_m':.1,'root_xy_rmse_m':.1,'steps':0} for name in ['nominal-01','eval-01','straight','left_turn','right_turn']}
    monkeypatch.setattr(training,'_evaluate',evaluate)
    continuous=training.train_source(config_path,run_dir=tmp_path/'continuous')
    interrupted=training.train_source(config_path,max_updates=cut,run_dir=tmp_path/'part')
    resumed=training.train_source(config_path,interrupted/'latest.pt',run_dir=tmp_path/'resumed')
    a=torch.load(continuous/'latest.pt',weights_only=True);b=torch.load(resumed/'latest.pt',weights_only=True)
    for key in ['actor','model','critic']:
        for name,value in a[key].items():torch.testing.assert_close(value,b[key][name],rtol=0,atol=0)
    assert a['runtime']['environment_steps']==b['runtime']['environment_steps']
    assert torch.equal(a['runtime']['rng']['torch'],b['runtime']['rng']['torch'])
    assert collections==['warmup','actor','warmup','actor']
