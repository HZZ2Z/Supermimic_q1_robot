"""Source warmup initialization, resume validation, and bounded training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

import pytest

from pole_task import COURSE_ROOT, ROOT
from pole_task.preflight import preflight
from pole_task.source_recipe import resolve_source_recipe
from pole_task.foot_contact import FOOT_CONTACT_VERSION
from pole_task.source_runtime import SOURCE_REWARD_TUNING
from pole_task.source_training import (
    SOURCE_CHECKPOINT_FORMAT,
    TrainingBudget,
    checkpoint_output_hashes,
    evaluation_cases_for,
    load_training_cases,
    select_checkpoint,
    source_acceptance,
    train_source,
    validate_resume_payload,
)

CONFIG = COURSE_ROOT / "pole_task/config.json"
MANIFEST = COURSE_ROOT / "pole_task/data/manifest.json"
SOURCE_MANIFEST = COURSE_ROOT / "pole_task/source_manifest.json"


def _digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(recipe, config_path=CONFIG, manifest_path=MANIFEST) -> dict:
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    return {
        "format": SOURCE_CHECKPOINT_FORMAT,
        "profile": "pole-weaving",
        "foot_contact_version": FOOT_CONTACT_VERSION,
        "reward_tuning": asdict(SOURCE_REWARD_TUNING),
        "source_final_checkpoint_loaded": False,
        "source_warmup_checkpoint_sha256": recipe.warmup_checkpoint_sha256,
        "source_revision": recipe.source_revision,
        "robot_xml_sha256": manifest["robot_contract"]["xml_sha256"],
        "data_manifest_sha256": _digest(manifest_path),
        "config_sha256": _digest(config_path),
    }


def _payload(recipe, *, update=10, stage="warmup", identity=None) -> dict:
    return {
        "format": SOURCE_CHECKPOINT_FORMAT,
        "stage": stage,
        "update": update,
        "config": json.loads(CONFIG.read_text()),
        "recipe": recipe.to_dict(),
        "identity": _identity(recipe) if identity is None else identity,
    }


def test_preflight_strictly_loads_the_source_warmup_and_reports_provenance():
    report = preflight(CONFIG)
    recipe = resolve_source_recipe(CONFIG)

    assert report["training_ready"] is True
    assert report["training_blocker"] is None
    initialization = report["initialization"]
    assert initialization["strictly_loaded"] is True
    assert initialization["source_checkpoint_id"] == "warmup_checkpoint"
    assert initialization["sha256"] == recipe.warmup_checkpoint_sha256
    assert initialization["warmup_iteration"] == 2000
    assert initialization["source_revision"] == recipe.source_revision
    assert initialization["source_final_checkpoint_loaded"] is False
    assert initialization["final_checkpoint_sha256"] == recipe.final_checkpoint_sha256
    assert report["recipe"] == recipe.to_dict()
    assert report["recipe"]["stages"][0]["updates"] == 2000
    assert report["recipe"]["stages"][1]["updates"] == 500


def test_project_checkpoint_validation_rejects_wrong_lineage_and_stage():
    recipe = resolve_source_recipe(CONFIG)
    valid = _payload(recipe)
    assert (
        validate_resume_payload(
            valid,
            config=valid["config"],
            config_path=CONFIG,
            recipe=recipe,
            manifest_path=MANIFEST,
            source_manifest_path=SOURCE_MANIFEST,
        )
        == 10
    )

    wrong_format = {**valid, "format": "q1_pole_weaving_project_v9"}
    with pytest.raises(ValueError, match="project-trained pole-profile"):
        validate_resume_payload(
            wrong_format,
            config=valid["config"],
            config_path=CONFIG,
            recipe=recipe,
            manifest_path=MANIFEST,
            source_manifest_path=SOURCE_MANIFEST,
        )

    final_actor = _payload(
        recipe,
        identity={**_identity(recipe), "source_final_checkpoint_loaded": True},
    )
    with pytest.raises(ValueError, match="final actor"):
        validate_resume_payload(
            final_actor,
            config=valid["config"],
            config_path=CONFIG,
            recipe=recipe,
            manifest_path=MANIFEST,
            source_manifest_path=SOURCE_MANIFEST,
        )

    wrong_warmup = _payload(
        recipe,
        identity={
            **_identity(recipe),
            "source_warmup_checkpoint_sha256": "0" * 64,
        },
    )
    with pytest.raises(ValueError, match="different source warmup"):
        validate_resume_payload(
            wrong_warmup,
            config=valid["config"],
            config_path=CONFIG,
            recipe=recipe,
            manifest_path=MANIFEST,
            source_manifest_path=SOURCE_MANIFEST,
        )

    wrong_stage = _payload(recipe, update=2001, stage="warmup")
    with pytest.raises(ValueError, match="stage"):
        validate_resume_payload(
            wrong_stage,
            config=valid["config"],
            config_path=CONFIG,
            recipe=recipe,
            manifest_path=MANIFEST,
            source_manifest_path=SOURCE_MANIFEST,
        )

    beyond_budget = _payload(recipe, update=2501, stage="actor")
    with pytest.raises(ValueError, match="Resume update"):
        validate_resume_payload(
            beyond_budget,
            config=valid["config"],
            config_path=CONFIG,
            recipe=recipe,
            manifest_path=MANIFEST,
            source_manifest_path=SOURCE_MANIFEST,
        )


def test_checkpoint_selection_never_searches_outside_the_run(tmp_path):
    with pytest.raises(FileNotFoundError):
        select_checkpoint(tmp_path)
    (tmp_path / "latest.pt").write_bytes(b"latest")
    assert select_checkpoint(tmp_path) == tmp_path / "latest.pt"
    (tmp_path / "best.pt").write_bytes(b"best")
    assert select_checkpoint(tmp_path) == tmp_path / "best.pt"
    assert select_checkpoint(tmp_path, prefer_best=False) == tmp_path / "latest.pt"


@pytest.mark.parametrize("field", ["foot_contact_version", "reward_tuning"])
def test_resume_rejects_missing_or_changed_foot_contract(field):
    recipe = resolve_source_recipe(CONFIG)
    payload = _payload(recipe)
    payload["identity"].pop(field)
    with pytest.raises(ValueError, match="foot-contact physics or reward"):
        validate_resume_payload(
            payload, config=payload["config"], config_path=CONFIG, recipe=recipe,
            manifest_path=MANIFEST, source_manifest_path=SOURCE_MANIFEST,
        )


def test_published_checkpoints_are_hashed_for_run_provenance(tmp_path):
    (tmp_path / "latest.pt").write_bytes(b"latest")
    hashes = checkpoint_output_hashes(tmp_path)
    assert hashes == {
        "latest_checkpoint_sha256": _digest(tmp_path / "latest.pt")
    }
    (tmp_path / "best.pt").write_bytes(b"best")
    hashes = checkpoint_output_hashes(tmp_path)
    assert hashes["best_checkpoint_sha256"] == _digest(tmp_path / "best.pt")
    assert hashes["latest_checkpoint_sha256"] == _digest(tmp_path / "latest.pt")


def test_documented_preflight_is_read_only_and_opens_no_viewer(monkeypatch):
    config = json.loads(CONFIG.read_text())
    output_root = ROOT / config["output_root"]
    before = set(output_root.glob("run-*")) if output_root.exists() else set()

    def blocked(*args, **kwargs):
        raise AssertionError("preflight opened a MuJoCo viewer")

    monkeypatch.setattr("mujoco.viewer.launch_passive", blocked)
    report = preflight(CONFIG)

    assert report["training_ready"]
    assert report["initialization"]["strictly_loaded"]
    after = set(output_root.glob("run-*")) if output_root.exists() else set()
    assert after == before


def test_failed_source_preflight_creates_no_run_directory(tmp_path, monkeypatch):
    config_path = tmp_path / "pole_task/config.json"
    config_path.parent.mkdir()
    config_path.write_text("{}")

    def missing_source(_):
        raise FileNotFoundError("registered source warmup is missing")

    monkeypatch.setattr(
        "pole_task.source_training.preflight", missing_source
    )
    with pytest.raises(FileNotFoundError, match="warmup is missing"):
        train_source(config_path)
    assert not (tmp_path / "logs").exists()


def test_training_budget_is_strictly_bounded():
    budget = TrainingBudget(5)
    budget.charge(3)
    assert budget.remaining() == 2
    with pytest.raises(RuntimeError, match="environment-step budget"):
        budget.charge(3)


def test_training_cases_exclude_the_held_out_pair():
    config = json.loads(CONFIG.read_text())
    training, evaluation, all_cases = load_training_cases(config, manifest_path=MANIFEST)

    assert [case.case_id for case in evaluation] == ["eval-01"]
    assert evaluation[0].split == "held_out"
    assert all(case.split != "held_out" for case in training)
    assert {case.case_id for case in all_cases} == {
        *[case.case_id for case in training],
        "eval-01",
    }
    # The retention references are paired with their registered hashes rather
    # than being regenerated from soccer checkpoints.
    assert {case.case_id for case in training} == {
        "nominal-01",
        "straight",
        "left_turn",
        "right_turn",
    }
    assert all(case.sha256 for case in all_cases)


def _report(*, completed, collision=False, fall=False, endpoint=0.1, rmse=0.1):
    return {
        "route_completed": completed,
        "pole_collision": collision,
        "fall": fall,
        "endpoint_distance_m": endpoint,
        "root_xy_rmse_m": rmse,
        "foot_contact_time_s": [1.0, 1.0],
        "foot_slip_mean_m_s": 0.02,
        "foot_slip_fraction": 0.1,
    }


def test_training_evaluation_includes_every_acceptance_case():
    config = json.loads(CONFIG.read_text())
    _, _, all_cases = load_training_cases(config, manifest_path=MANIFEST)
    ordered = evaluation_cases_for(config, all_cases)
    assert [case.case_id for case in ordered] == [
        "nominal-01",
        "eval-01",
        "straight",
        "left_turn",
        "right_turn",
    ]


def test_checkpoint_selection_requires_nominal_held_out_and_retention():
    config = json.loads(CONFIG.read_text())
    good = {
        "nominal-01": _report(completed=True),
        "eval-01": _report(completed=True),
        "straight": _report(completed=True),
        "left_turn": _report(completed=True),
        "right_turn": _report(completed=True),
    }
    assert source_acceptance(good, config)
    assert not source_acceptance(
        {**good, "nominal-01": _report(completed=False)}, config
    )
    assert not source_acceptance(
        {**good, "eval-01": _report(completed=True, collision=True)}, config
    )
    assert not source_acceptance(
        {**good, "straight": _report(completed=True, fall=True)}, config
    )
    assert not source_acceptance(
        {**good, "eval-01": _report(completed=True, rmse=0.6)}, config
    )


@pytest.mark.parametrize('key', ['actor_collection_interval', 'actor_checkpoint_interval'])
@pytest.mark.parametrize('value', [0, -1, 1.5, True])
def test_actor_intervals_reject_invalid_config(tmp_path, key, value):
    config = json.loads(CONFIG.read_text())
    config[key] = value
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=key):
        preflight(path)


@pytest.mark.parametrize('resume_update,requested_scale,inherited_scale', [
    (0, None, 1.0), (3, None, 1.0), (4, None, 1.0),
    (3, 0.5, 1.0), (3, None, 0.5), (3, 1.0, 0.5),
])
def test_actor_refresh_and_snapshots_without_optimizer_updates(
    tmp_path, monkeypatch, resume_update, requested_scale, inherited_scale,
):
    from dataclasses import replace
    from types import SimpleNamespace as NS
    import torch
    from pole_task import source_training as st

    original = resolve_source_recipe(CONFIG)
    warmup = replace(original.stages[0], updates=2, end_update=2)
    actor_stage = replace(original.stages[1], updates=6, start_update=3, end_update=8)
    recipe = replace(original, stages=(warmup, actor_stage), model_batches=1)
    config = json.loads(CONFIG.read_text())
    config.update(output_root=str(tmp_path / 'runs'), device='cpu', evaluation_interval=3,
                  actor_collection_interval=2, actor_checkpoint_interval=2)
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    monkeypatch.setattr(st, 'preflight', lambda _: {'training_ready': True, 'hardware': {'ready': True}})
    monkeypatch.setattr(st, 'resolve_source_recipe', lambda *a, **k: recipe)
    modules = [torch.nn.Linear(1, 1) for _ in range(4)]
    loaded = NS(robot=modules[0], actor=modules[1], world_model=modules[2], critic=modules[3])
    monkeypatch.setattr(st, 'load_source_pole', lambda *a, **k: loaded)
    cases = [NS(case_id='nominal-01', split='train'), NS(case_id='left_turn', split='retention')]
    held_out = NS(case_id='eval-01', split='held_out')
    monkeypatch.setattr(st, 'load_training_cases', lambda *a, **k: (cases, [held_out], cases + [held_out]))
    monkeypatch.setattr(st, 'write_vendor_motion', lambda case, path: path)
    monkeypatch.setattr(st, 'PoleMotions', lambda *a, **k: NS(prepare=lambda: None))
    monkeypatch.setattr(st, '_checkpoint_identity', lambda **k: {})
    monkeypatch.setattr(st, '_make_optimizers', lambda *a: None)
    monkeypatch.setattr(st, 'freeze_pole_input_columns', lambda *a: None)
    updates, collections, replays, rates, restored, replay_members = [], [], [], [], [], []
    def optimizer():
        return NS(param_groups=[{}], state_dict=lambda: {}, load_state_dict=lambda state: restored.append(state))
    def improve(batch, *args, **kwargs):
        updates.append((batch, kwargs))
        rates.append({name: getattr(algorithm, name+'_optimizer').param_groups[0]['lr']
                      for name in ('model', 'actor', 'critic')})
        return {'imagined_reward': 0.1}
    algorithm = NS(robot=modules[0], actor=modules[1], model=modules[2], critic=modules[3],
                   model_optimizer=optimizer(), actor_optimizer=optimizer(), critic_optimizer=optimizer(),
                   fit_model=lambda batch: {}, improve_policy=improve)
    monkeypatch.setattr(st, 'SourcePoleSuperMimic', lambda *a, **k: algorithm)
    def collect(*args, **kwargs):
        collections.append((kwargs['stage'].name, kwargs['seed']))
        kwargs['budget'].charge(12)
        return [NS(steps=7, case=cases[0], collection=len(collections)),
                NS(steps=5, case=cases[1], collection=len(collections))]
    monkeypatch.setattr(st, '_collect_episodes', collect)
    def replay(*args, **kwargs):
        generation = len(replays) + 1
        replays.append(generation)
        replay_members.append(list(dict.fromkeys(e.collection for e in args[0])))
        return NS(sample=lambda *a, **k: NS(
            generation=generation,
            clips=torch.tensor([0, 1, 1] if k['window'] == 8 else [0, 1]),
        ))
    monkeypatch.setattr(st, 'make_source_replay', replay)
    monkeypatch.setattr(st, 'evaluation_cases_for', lambda *a: cases)
    def evaluate(*args, **kwargs):
        kwargs['budget'].charge(3)
        return {}
    monkeypatch.setattr(st, '_evaluate', evaluate)
    monkeypatch.setattr(st, 'source_acceptance', lambda *a: False)
    monkeypatch.setattr(st, 'evaluation_score', lambda *a: float('inf'))
    def forbidden(*args, **kwargs):
        raise AssertionError('Tests must not execute optimizer steps')
    monkeypatch.setattr(torch.optim.Adam, 'step', forbidden)
    if resume_update:
        payload = {name: module.state_dict() for name, module in zip(['actor', 'model', 'critic'], modules[1:])}
        payload.update({name: {} for name in ['actor_optimizer', 'model_optimizer', 'critic_optimizer']})
        payload.update(identity={'actor_learning_rate_scale': inherited_scale}, stage='actor')
        monkeypatch.setattr(st, 'load_resume_checkpoint', lambda *a, **k: (payload, resume_update))
    run = train_source(path, tmp_path / 'fake-resume.pt' if resume_update else None,
                       resume_actor_lr_scale=requested_scale)
    scale = inherited_scale if requested_scale is None else requested_scale
    assert len(restored) == (3 if resume_update else 0)
    identity = json.loads((run / 'identity.json').read_text())
    assert identity['actor_learning_rate_scale'] == scale
    assert identity['effective_actor_base_learning_rate'] == pytest.approx(1e-5 * scale)
    assert identity['actor_replay_batches'] == 2
    expected = {0: [1, 1, 2, 2, 3, 3, 4, 4], 3: [1, 2, 2, 3, 3], 4: [1, 1, 2, 2]}[resume_update]
    assert [batch.generation for batch, _ in updates] == expected
    assert replay_members == {
        0: [[1], [2], [2, 3], [3, 4]],
        3: [[1], [1, 2], [2, 3]],
        4: [[1], [1, 2]],
    }[resume_update]
    assert len(set(seed for _, seed in collections)) == len(collections)
    assert not (run / 'best.pt').exists()
    for update in [4, 6, 8]:
        snapshot = run / f'checkpoint-{update:06d}.pt'
        assert snapshot.exists() == (update > resume_update)
        if snapshot.exists():
            saved = torch.load(snapshot, weights_only=True)
            assert saved['update'] == update and saved['stage'] == 'actor'
            assert 'actor_optimizer' in saved and saved['config'] == config
            assert saved['recipe'] == recipe.to_dict()
            assert saved['identity']['actor_learning_rate_scale'] == scale
    metrics = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
    assert metrics[-1]['collection_environment_steps'] == len(collections) * 12
    assert metrics[-1]['replay_transitions'] == 24
    assert metrics[-1]['replay_batch_updates'] == [5, 7]
    assert metrics[-1]['environment_steps'] > metrics[-1]['collection_environment_steps']
    # Snapshot-only updates do not reuse an older evaluation or select best.
    for row in metrics:
        assert row['actor_learning_rate_scale'] == scale
        batches = len(row['replay_batch_updates'])
        assert batches in (1, 2)
        assert row['replay_transitions'] == 12 * batches
        assert row['replay_case_transitions'] == {
            'nominal-01': 7 * batches, 'left_turn': 5 * batches, 'eval-01': 0,
        }
        elapsed = row['update'] - resume_update
        assert row['sampled_model_windows_by_case'] == {
            'nominal-01': elapsed, 'left_turn': elapsed * 2, 'eval-01': 0,
        }
        assert row['sampled_policy_windows_by_case'] == {
            'nominal-01': elapsed, 'left_turn': elapsed, 'eval-01': 0,
        }
        if row['update'] in [4]:
            assert 'evaluation' not in row
    for update, (_, kwargs) in enumerate(updates, start=resume_update + 1):
        assert kwargs['update_actor'] == (update >= 3)
        assert kwargs['log_reward_terms'] == (update in [2, 3, 4, 6, 8])
        expected_rates = recipe.optimizer_rates(update - 1)
        expected_rates['actor'] *= scale
        assert rates[update - resume_update - 1] == expected_rates


@pytest.mark.parametrize('value', [0, -0.5, 1.1, float('nan'), float('inf'), True, '0.5'])
def test_resume_actor_lr_scale_rejects_invalid_values(value):
    from pole_task.source_training import _resume_actor_lr_scale
    payload = {'stage': 'actor', 'identity': {}}
    with pytest.raises(ValueError, match='scale'):
        _resume_actor_lr_scale(payload, value)
    payload['identity']['actor_learning_rate_scale'] = value
    with pytest.raises(ValueError, match='scale'):
        _resume_actor_lr_scale(payload, 0.5)


def test_resume_scale_requires_actor_stage_and_inherits_old_checkpoint_default():
    from pole_task.source_training import _resume_actor_lr_scale
    assert _resume_actor_lr_scale(None, None) == 1
    assert _resume_actor_lr_scale({'stage': 'actor', 'identity': {}}, None) == 1
    for payload in [None, {'stage': 'warmup', 'identity': {}}]:
        with pytest.raises(ValueError, match='Actor-stage resume'):
            _resume_actor_lr_scale(payload, 0.5)
    with pytest.raises(ValueError, match='null'):
        _resume_actor_lr_scale({'stage': 'actor', 'identity': {'actor_learning_rate_scale': None}}, None)


@pytest.mark.parametrize('changed', ['config', 'recipe', 'actor_learning_rate_scale'])
def test_experimental_resume_keeps_config_and_recipe_checks(changed):
    recipe = resolve_source_recipe(CONFIG)
    payload = _payload(recipe, update=2100, stage='actor')
    payload['identity']['actor_learning_rate_scale'] = 0.5
    config = json.loads(CONFIG.read_text())
    if changed == 'config':
        payload['config']['source_training']['actor_learning_rate'] = 5e-6
    elif changed == 'recipe':
        payload['recipe']['optimizers']['actor']['learning_rate'] = 5e-6
    else:
        payload['identity']['actor_learning_rate_scale'] = float('nan')
    with pytest.raises(ValueError):
        validate_resume_payload(payload, config=config, config_path=CONFIG, recipe=recipe,
                                manifest_path=MANIFEST, source_manifest_path=SOURCE_MANIFEST)


def test_resume_scale_without_checkpoint_fails_before_preflight(monkeypatch):
    from pole_task import source_training as st
    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid resume must not initialize training')
    monkeypatch.setattr(st, 'preflight', forbidden)
    with pytest.raises(ValueError, match='Actor-stage resume'):
        st.train_source(CONFIG, resume_actor_lr_scale=0.5)


def test_cli_forwards_resume_experiment_without_training(monkeypatch):
    from pole_task import source_training as st
    calls = []
    monkeypatch.setattr(st, 'train_source', lambda *a, **k: calls.append((a, k)))
    assert st.main(['--resume-checkpoint', '/tmp/2100.pt', '--max-updates', '2200',
                    '--resume-actor-lr-scale', '0.5']) == 0
    assert calls[0][0][1].name == '2100.pt'
    assert calls[0][1] == {'max_updates': 2200, 'resume_actor_lr_scale': 0.5,
                          'actor_end_update': None, 'run_name': None}


@pytest.mark.parametrize('limit', [2099, 2100])
def test_resume_limit_fails_before_model_loading_or_run_creation(tmp_path, monkeypatch, limit):
    from pole_task import source_training as st
    config = json.loads(CONFIG.read_text())
    config['output_root'] = str(tmp_path / 'runs')
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    recipe = resolve_source_recipe(CONFIG)
    payload = _payload(recipe, update=2100, stage='actor')
    monkeypatch.setattr(st, 'preflight', lambda _: {'training_ready': True, 'hardware': {'ready': True}})
    monkeypatch.setattr(st, 'resolve_source_recipe', lambda *a, **k: recipe)
    monkeypatch.setattr(st, 'load_resume_checkpoint', lambda *a, **k: (payload, 2100))
    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid resume limit must not initialize models')
    monkeypatch.setattr(st, 'load_source_pole', forbidden)
    with pytest.raises(ValueError, match='must exceed'):
        st.train_source(path, tmp_path / 'fake.pt', max_updates=limit, resume_actor_lr_scale=0.5)
    assert not (tmp_path / 'runs').exists()


@pytest.mark.parametrize('update', [0, 1999, 2000, 2001, 2100, 2500])
def test_resume_validates_completed_stage_at_boundaries(update):
    recipe = resolve_source_recipe(CONFIG)
    payload = _payload(recipe, update=update, stage=recipe.stage_for_update(max(1, update)).name)
    assert validate_resume_payload(payload, config=payload['config'], config_path=CONFIG, recipe=recipe,
                                   manifest_path=MANIFEST, source_manifest_path=SOURCE_MANIFEST) == update


def test_warmup_end_cannot_be_mislabeled_as_actor():
    recipe = resolve_source_recipe(CONFIG)
    payload = _payload(recipe, update=2000, stage='actor')
    with pytest.raises(ValueError, match='stage'):
        validate_resume_payload(payload, config=payload['config'], config_path=CONFIG, recipe=recipe,
                                manifest_path=MANIFEST, source_manifest_path=SOURCE_MANIFEST)


@pytest.fixture
def control_flow_run(tmp_path, monkeypatch):
    """Production save/resume/control flow; all learning and collection are fake."""
    from dataclasses import replace
    from types import SimpleNamespace as NS
    import torch
    from pole_task import source_training as st
    original = resolve_source_recipe(CONFIG)
    recipe = replace(original, stages=(replace(original.stages[0], updates=2, end_update=2),
                     replace(original.stages[1], updates=6, start_update=3, end_update=8)), model_batches=1)
    config = json.loads(CONFIG.read_text())
    config.update(output_root=str(tmp_path / 'runs'), device='cpu', evaluation_interval=1,
                  actor_checkpoint_interval=2, success_patience=2)
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    modules = [torch.nn.Linear(1, 1) for _ in range(4)]
    original_states = [{k: v.clone() for k, v in m.state_dict().items()} for m in modules]
    flags = []
    def optimizer():
        return NS(param_groups=[{}], state_dict=lambda: {}, load_state_dict=lambda _: None)
    def improve(*args, **kwargs):
        flags.append(kwargs['update_actor'])
        return {}
    algorithm = NS(actor=modules[1], model=modules[2], critic=modules[3], fit_model=lambda _: {},
                   improve_policy=improve, model_optimizer=optimizer(), actor_optimizer=optimizer(), critic_optimizer=optimizer())
    case = NS(case_id='nominal-01', split='train')
    monkeypatch.setattr(st, 'preflight', lambda _: {'training_ready': True, 'hardware': {'ready': True}})
    monkeypatch.setattr(st, 'resolve_source_recipe', lambda *a, **k: recipe)
    monkeypatch.setattr(st, 'load_source_pole', lambda *a, **k: NS(robot=modules[0], actor=modules[1], world_model=modules[2], critic=modules[3]))
    monkeypatch.setattr(st, 'load_training_cases', lambda *a, **k: ([case], [], [case]))
    monkeypatch.setattr(st, 'write_vendor_motion', lambda c, p: p)
    monkeypatch.setattr(st, 'PoleMotions', lambda *a, **k: NS(prepare=lambda: None))
    monkeypatch.setattr(st, '_checkpoint_identity', lambda **k: _identity(recipe, path))
    monkeypatch.setattr(st, '_make_optimizers', lambda *a: None)
    monkeypatch.setattr(st, 'freeze_pole_input_columns', lambda *a: None)
    monkeypatch.setattr(st, 'SourcePoleSuperMimic', lambda *a, **k: algorithm)
    monkeypatch.setattr(st, '_collect_episodes', lambda *a, **k: [NS(steps=12, case=case)])
    monkeypatch.setattr(st, 'make_source_replay', lambda *a, **k: NS(sample=lambda *a, **k: NS(clips=torch.tensor([0]))))
    monkeypatch.setattr(st, 'evaluation_cases_for', lambda *a: [case])
    monkeypatch.setattr(st, 'source_acceptance', lambda r, c: r['score'] is not None)
    monkeypatch.setattr(st, 'evaluation_score', lambda r, c: r['score'] if r['score'] is not None else float('inf'))
    def forbidden(*a, **k):
        raise AssertionError('No real optimizer updates allowed')
    monkeypatch.setattr(torch.optim.Adam, 'step', forbidden)
    def run(resume=None, limit=None, scores=(), **kwargs):
        flags.clear()
        values = iter(scores)
        monkeypatch.setattr(st, '_evaluate', lambda *a, **k: {'score': next(values, None)})
        directory = st.train_source(path, resume, max_updates=limit, **kwargs)
        for m, before in zip(modules, original_states):
            for key, value in m.state_dict().items():
                assert torch.equal(value, before[key])
        return NS(path=directory, flags=flags.copy(),
                  metrics=[json.loads(s) for s in (directory / 'metrics.jsonl').read_text().splitlines()],
                  saved=torch.load(directory / 'latest.pt', weights_only=True),
                  stop=json.loads((directory / 'stop_reason.json').read_text()),
                  identity=json.loads((directory / 'identity.json').read_text()))
    return run


def test_patience_does_not_stop_frozen_warmup(control_flow_run):
    result = control_flow_run()
    assert result.flags == [False, False, True, True]
    assert [r['patience'] for r in result.metrics] == [0, 0, 1, 2]
    assert result.stop['update'] == 4
    assert result.stop['reason'] == 'success_patience_exhausted'


def test_saved_warmup_boundary_resumes_into_actor(control_flow_run):
    first = control_flow_run(limit=2)
    assert first.saved['stage'] == 'warmup' and first.saved['patience'] == 0
    second = control_flow_run(first.path / 'latest.pt', limit=4)
    assert second.flags == [True, True]
    assert second.metrics[0]['update'] == 3
    assert second.metrics[0]['replay_batch_updates'] == [3]
    assert second.saved['stage'] == 'actor'


def test_new_checkpoint_preserves_actor_patience_on_resume(control_flow_run):
    first = control_flow_run(limit=3)
    assert first.saved['patience'] == 1
    assert first.saved['identity']['patience_scope'] == 'actor'
    second = control_flow_run(first.path / 'latest.pt')
    assert second.flags == [True]
    assert second.stop['update'] == 4
    assert second.saved['patience'] == 2


def test_legacy_patience_resets_once_without_resetting_best_score(control_flow_run):
    import torch
    first = control_flow_run(limit=3)
    legacy = first.saved
    legacy['identity'].pop('patience_scope', None)
    legacy['patience'], legacy['best_score'] = 84, 9.0
    path = first.path / 'legacy.pt'
    torch.save(legacy, path)
    second = control_flow_run(path, limit=4)
    assert second.saved['patience'] == 1
    assert second.saved['best_score'] == 9.0
    assert second.identity['legacy_patience_reset'] is True
    assert second.identity['initial_actor_patience'] == 0
    third = control_flow_run(second.path / 'latest.pt')
    assert third.identity['legacy_patience_reset'] is False
    assert third.identity['initial_actor_patience'] == 1
    assert third.stop['update'] == 5 and third.saved['patience'] == 2


def test_actor_score_improvement_resets_patience(control_flow_run):
    result = control_flow_run(scores=[None, None, None, 2.0, None, None])
    assert [r['patience'] for r in result.metrics] == [0, 0, 1, 0, 1, 2]
    assert result.saved['best_score'] == 2.0 and result.stop['update'] == 6
    assert (result.path / 'best.pt').exists()


@pytest.mark.parametrize('scope,value', [('unknown', 0), ('actor', -1), ('actor', True),
                                        ('actor', 1.5), ('actor', None)])
def test_new_patience_metadata_rejects_invalid_values(scope, value):
    from pole_task.source_training import _resume_actor_patience
    payload = {'identity': {'patience_scope': scope}, 'patience': value, 'stage': 'actor'}
    with pytest.raises(ValueError, match='patience'):
        _resume_actor_patience(payload)


def test_warmup_boundary_uses_default_lr_without_actor_override():
    from pole_task.source_training import _resume_actor_lr_scale
    recipe = resolve_source_recipe(CONFIG)
    payload = _payload(recipe, update=2000, stage='warmup')
    assert validate_resume_payload(payload, config=payload['config'], config_path=CONFIG, recipe=recipe,
                                   manifest_path=MANIFEST, source_manifest_path=SOURCE_MANIFEST) == 2000
    assert _resume_actor_lr_scale(payload, None) == 1.0
    with pytest.raises(ValueError, match='Actor-stage resume'):
        _resume_actor_lr_scale(payload, 0.5)


def test_actor_extension_changes_only_duration_and_preserves_cosine():
    from pole_task.source_training import _runtime_recipe
    recipe = resolve_source_recipe(CONFIG)
    payload = _payload(recipe, update=2200, stage='actor')
    assert _runtime_recipe(recipe) is recipe
    extended = _runtime_recipe(recipe, payload, 2600)
    assert recipe.total_updates == 2500
    assert extended.total_updates == 2600 and extended.actor_updates == 600
    assert extended.stages[0] == recipe.stages[0]
    assert extended.stage_for_update(2600).actor_updates_enabled
    original, modified = recipe.to_dict(), extended.to_dict()
    original.pop('stages'); modified.pop('stages')
    assert original == modified
    assert extended.optimizer_rates(2599) == recipe.optimizer_rates(2599)


@pytest.mark.parametrize('end', [True, None, 2499, 2600.5, '2600', float('nan')])
def test_invalid_saved_actor_budget_is_rejected(end):
    recipe = resolve_source_recipe(CONFIG)
    payload = _payload(recipe, update=2200, stage='actor')
    payload['identity']['actor_end_update'] = end
    with pytest.raises(ValueError, match='Actor end update'):
        validate_resume_payload(payload, config=payload['config'], config_path=CONFIG, recipe=recipe,
                                manifest_path=MANIFEST, source_manifest_path=SOURCE_MANIFEST)


def test_extension_requires_actor_resume_and_cannot_shrink():
    from pole_task.source_training import _runtime_recipe
    recipe = resolve_source_recipe(CONFIG)
    for payload in (None, _payload(recipe, update=2000, stage='warmup')):
        with pytest.raises(ValueError, match='Actor-stage resume'):
            _runtime_recipe(recipe, payload, 2600)
    payload = _payload(recipe, update=2550, stage='actor')
    payload['identity']['actor_end_update'] = 2600
    assert _runtime_recipe(recipe, payload).total_updates == 2600
    with pytest.raises(ValueError, match='shrink'):
        _runtime_recipe(recipe, payload, 2500)
    assert validate_resume_payload(payload, config=payload['config'], config_path=CONFIG, recipe=recipe,
                                   manifest_path=MANIFEST, source_manifest_path=SOURCE_MANIFEST) == 2550
    payload['recipe']['optimizers']['actor']['learning_rate'] *= 2
    with pytest.raises(ValueError, match='recipe'):
        validate_resume_payload(payload, config=payload['config'], config_path=CONFIG, recipe=recipe,
                                manifest_path=MANIFEST, source_manifest_path=SOURCE_MANIFEST)


def test_completed_registered_budget_can_resume_only_with_extension(tmp_path):
    import torch
    from pole_task.source_training import load_resume_checkpoint
    recipe = resolve_source_recipe(CONFIG)
    payload = _payload(recipe, update=2500, stage='actor')
    path = tmp_path / '2500.pt'
    torch.save(payload, path)
    kwargs = dict(config=payload['config'], config_path=CONFIG, recipe=recipe,
                  manifest_path=MANIFEST, source_manifest_path=SOURCE_MANIFEST)
    with pytest.raises(ValueError, match='completed all 2500'):
        load_resume_checkpoint(path, **kwargs)
    assert load_resume_checkpoint(path, **kwargs, actor_end_update=2600)[1] == 2500


def test_extended_resume_roundtrip_and_named_directory(control_flow_run):
    first = control_flow_run(limit=7, scores=range(20, 13, -1))
    with pytest.raises(ValueError, match='max_updates must be in 1..8'):
        control_flow_run(first.path / 'latest.pt', limit=9)
    second = control_flow_run(first.path / 'latest.pt', limit=9, actor_end_update=10,
                              run_name='A-from7-to10', scores=[12, 11])
    assert second.path.name == 'A-from7-to10'
    assert second.identity['run_name'] == second.path.name
    assert second.identity['actor_end_update'] == 10
    assert second.saved['recipe']['stages'][1]['end_update'] == 8
    assert second.saved['stage'] == 'actor' and second.saved['update'] == 9
    assert second.flags == [True, True]
    third = control_flow_run(second.path / 'latest.pt', run_name='A-continue', scores=[10])
    assert third.saved['update'] == 10 and third.flags == [True]
    assert third.stop['reason'] == 'extended_budget_exhausted'
    assert (third.path / 'checkpoint-000010.pt').is_file()
    before = (second.path / 'latest.pt').read_bytes()
    with pytest.raises(FileExistsError):
        control_flow_run(first.path / 'latest.pt', limit=9, actor_end_update=10,
                         run_name='A-from7-to10')
    assert (second.path / 'latest.pt').read_bytes() == before


@pytest.mark.parametrize('name', ['', '.', '..', '../escape', '/tmp/escape', 'a/b', 'a\\b', 'a\nb', 'x'*97])
def test_invalid_run_name_fails_before_preflight(monkeypatch, name):
    from pole_task import source_training as st
    monkeypatch.setattr(st, 'preflight', lambda *a: pytest.fail('Must reject before initialization'))
    with pytest.raises(ValueError, match='Run name'):
        st.train_source(CONFIG, run_name=name)


def test_cli_forwards_named_actor_extension(monkeypatch):
    from pole_task import source_training as st
    calls = []
    monkeypatch.setattr(st, 'train_source', lambda *a, **k: calls.append((a, k)))
    assert st.main(['--resume-checkpoint', '/tmp/2200.pt', '--max-updates', '2600',
                    '--actor-end-update', '2600', '--run-name', 'A-2200-to2600']) == 0
    assert calls[0][1] == dict(max_updates=2600, resume_actor_lr_scale=None,
                             actor_end_update=2600, run_name='A-2200-to2600')
