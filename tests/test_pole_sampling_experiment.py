import json
from dataclasses import replace
import numpy as np
import pytest
from pole_task.source_runtime import configured_reward_tuning, validate_start_sampling, stratified_start_frame, SOURCE_REWARD_TUNING
from pole_task import source_training as st


def test_defaults_preserve_registered_reward():
    assert configured_reward_tuning({}) == SOURCE_REWARD_TUNING
    assert validate_start_sampling({}) == 'uniform'
    assert configured_reward_tuning({'foot_slip_weight': .075}) == replace(SOURCE_REWARD_TUNING, foot_slip_weight=.075)


def test_v1_mainline_matches_B_without_changing_legacy_defaults():
    from pole_task import COURSE_ROOT, ROOT
    main = json.loads((COURSE_ROOT/'pole_task/config.json').read_text())
    experiment = json.loads((ROOT/'experiments/reward-sampling/B.json').read_text())
    legacy = json.loads((ROOT/'experiments/reward-sampling/pre-V1-main.json').read_text())
    assert main.pop('output_root') == 'logs/q1-pole-weaving'
    experiment.pop('output_root')
    assert main == experiment
    assert validate_start_sampling(main) == 'stratified_4'
    assert validate_start_sampling(legacy) == 'uniform'
    assert configured_reward_tuning(main) == configured_reward_tuning(legacy)


@pytest.mark.parametrize('value', [-1, float('nan'), float('inf')])
def test_invalid_weight_rejected(value):
    with pytest.raises(ValueError):configured_reward_tuning({'foot_slip_weight': value})


def test_invalid_sampling_rejected():
    with pytest.raises(ValueError):validate_start_sampling({'actor_start_sampling':'unknown'})


@pytest.mark.parametrize('count', [2,4])
def test_strata_are_reproducible_and_cover_two_batches(count):
    bins=np.array_split(np.arange(1,298-9),4);seen=[]
    for batch in range(2):
        for i in range(count):
            value=stratified_start_frame(298,batch,count,i,123+i+batch)
            assert value==stratified_start_frame(298,batch,count,i,123+i+batch)
            which=(batch*count+i)%4;assert value in bins[which];seen.append(which)
            assert 1<=value<=288
    assert set(seen)=={0,1,2,3}


def test_too_short_reference_rejected():
    with pytest.raises(ValueError):stratified_start_frame(10,0,2,0,123)


def test_collection_warmup_and_actor_weight_and_coverage(tmp_path,monkeypatch):
    from types import SimpleNamespace as NS
    calls=[]
    def run(*args,**kwargs):
        calls.append(kwargs)
        return NS(case=args[3],start_frame=kwargs['start_frame'] or 1,steps=12,
                  frames=np.arange(12),transitions=[{'boundary':i==11} for i in range(12)])
    monkeypatch.setattr(st,'run_source_episode',run)
    case=NS(case_id='left_turn',control_steps=298);motions=NS(cases=[case])
    config={'actor_start_sampling':'stratified_4','foot_slip_weight':.075}
    for actor in (False,True):
        st._collect_episodes(None,None,motions,[case],tmp_path,
            stage=NS(actor_updates_enabled=actor,randomize=actor,start_update=2001),seed=17,
            device='cpu',episode_counts={'left_turn':2},action_noise=0,
            budget=st.TrainingBudget(100),config=config,collection_update=2026)
    assert [c['start_frame'] for c in calls[:2]]==[None,None]
    assert [c['start_frame'] for c in calls[2:]]==[stratified_start_frame(298,1,2,i,17+i) for i in range(2)]
    assert all(c['reward_tuning'].foot_slip_weight==.075 for c in calls)
    rows=[json.loads(s) for s in (tmp_path/'collection.jsonl').read_text().splitlines()]
    assert all(r['valid_window_frames']=={'model':list(range(5)),'policy':[9,10,11]} for r in rows)


def test_checkpoint_reward_mismatch_is_not_silently_resumed(tmp_path):
    from dataclasses import asdict
    from pole_task import COURSE_ROOT
    from pole_task.source_recipe import resolve_source_recipe
    from pole_task.source_training import digest
    import torch
    parent=COURSE_ROOT.parent.parent/'版本记录/footflip/checkpoint-002100.pt'
    if not parent.exists():pytest.skip('Local experiment checkpoint is optional')
    payload=torch.load(parent,map_location='cpu',weights_only=True)
    config=json.loads((COURSE_ROOT/'pole_task/config.json').read_text());config['foot_slip_weight']=.075
    path=tmp_path/'config.json';path.write_text(json.dumps(config));recipe=resolve_source_recipe(path)
    payload['config']=config;payload['identity']['config_sha256']=digest(path)
    kwargs=dict(config=config,config_path=path,recipe=recipe,
                manifest_path=COURSE_ROOT/'pole_task/data/manifest.json',source_manifest_path=COURSE_ROOT/'pole_task/source_manifest.json')
    with pytest.raises(ValueError,match='reward tuning'):st.validate_resume_payload(payload,**kwargs)
    payload['identity']['reward_tuning']=asdict(configured_reward_tuning(config))
    assert st.validate_resume_payload(payload,**kwargs)==2100
