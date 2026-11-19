import copy
import importlib.util
import json
import pytest
from pole_task import ROOT
from pole_task.experiment import file_hash
from pole_task.local_training import VARIANTS
from pole_task.source_recipe import resolve_source_recipe


def test_readiness_requires_matching_source_dependencies_and_all_configs(tmp_path,monkeypatch):
    import pole_task.cam_local as cli
    monkeypatch.setattr(cli,'ROOT',tmp_path)
    monkeypatch.setattr(cli,'project_path',lambda p:tmp_path/p)
    monkeypatch.setattr(cli,'source_identity',lambda:{'sha256':'source'})
    monkeypatch.setattr(cli,'environment_identity',lambda:{'python':'version'})
    configs=tmp_path/'configs/lr-replay-v1';configs.mkdir(parents=True)
    hashes={}
    for variant in VARIANTS:
        config=json.loads((ROOT/'configs/lr-replay-v1'/f'{variant}.json').read_text())
        path=configs/f'{variant}.json';path.write_text(json.dumps(config));hashes[variant]=file_hash(path)
    config=json.loads((configs/'control.json').read_text())
    with pytest.raises(ValueError,match='prepare_lr_replay'):cli.validate_readiness(config,configs/'control.json')
    receipt=tmp_path/'logs/q1-pole-weaving/lr-replay-v1/readiness.json';receipt.parent.mkdir(parents=True)
    ready={'status':'COMPLETE','source_sha256':'source','variants':list(VARIANTS),
           'environment':{'python':'version'},'configuration_hashes':hashes}
    receipt.write_text(json.dumps(ready));cli.validate_readiness(config,configs/'control.json')
    for key,value in [('source_sha256','changed'),('environment',{}),('status','FAILED')]:
        bad={**ready,key:value};receipt.write_text(json.dumps(bad))
        with pytest.raises(ValueError):cli.validate_readiness(config,configs/'control.json')
    receipt.write_text(json.dumps(ready))
    with (configs/'low_lr.json').open('a') as f:f.write('\n')
    with pytest.raises(ValueError,match='Configuration changed'):cli.validate_readiness(config,configs/'control.json')


def test_preparation_rejects_partial_cpu_or_wrong_scale_diagnostic():
    spec=importlib.util.spec_from_file_location('prepare_lr_replay',ROOT/'scripts/prepare_lr_replay.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    recipe=resolve_source_recipe(ROOT/'configs/lr-replay-v1/control.json')
    row={'diagnostic_only':True,'device':'cuda:0','effective_recipe':recipe.to_dict(),
         'phases':[{'update':u,'replay_batch_updates':b,'actor_parameter_delta_l2':d,
                    'peak_allocated_mib':100.,'peak_reserved_mib':120.,'metrics':{'actor_grad_norm':d}}
                   for u,b,d in [(1,[1],0.),(2001,[2001],.1),(2026,[2001,2026],.1)]]}
    module.check_diagnostic(row,recipe,'profile')
    for mutate in (lambda r:r.update(device='cpu'),lambda r:r['phases'].pop(),
                   lambda r:r['effective_recipe'].update(model_batch_size=4),
                   lambda r:r['phases'][2].update(actor_parameter_delta_l2=0.)):
        bad=copy.deepcopy(row);mutate(bad)
        with pytest.raises(ValueError):module.check_diagnostic(bad,recipe,'profile')


def test_evaluation_forwards_explicit_config(tmp_path,monkeypatch):
    import pole_task.cam_evaluation as evaluation
    seen={}
    def load(path,device,*,config_path):
        seen.update(path=path,device=device,config=config_path)
        raise ValueError('stop after validating caller arguments')
    monkeypatch.setattr(evaluation,'load_pole_actor',load)
    with pytest.raises(ValueError,match='caller arguments'):
        evaluation.evaluate_checkpoint('model.pt',tmp_path/'evaluation',config_path='config.json')
    assert seen=={'path':'model.pt','device':'cpu','config':'config.json'}


def test_new_summary_refuses_original_cam_control_and_duplicate_variant():
    from pole_task.cam_evaluation import summarize
    rows=[]
    for variant in VARIANTS:
        rows.append({'identity':{'resolved_config':{'seed':0,'local_training_profile':{'schema_version':1,'variant':variant}},
                                 'checkpoint_sha256':variant},'acceptance':{'success':False},'cases':{}})
    assert summarize(rows)['complete'] is True
    with pytest.raises(ValueError,match='one qualified'):summarize(rows+[rows[0]])
    old=copy.deepcopy(rows[0]);old['identity']['resolved_config'].pop('local_training_profile')
    with pytest.raises(ValueError,match='mix'):summarize(rows+[old])
