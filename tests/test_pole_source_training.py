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


@pytest.mark.parametrize('resume_update', [0, 3, 4])
def test_actor_refresh_and_snapshots_without_optimizer_updates(tmp_path, monkeypatch, resume_update):
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
    cases = [NS(case_id='nominal-01', split='train')]
    monkeypatch.setattr(st, 'load_training_cases', lambda *a, **k: (cases, [], cases))
    monkeypatch.setattr(st, 'write_vendor_motion', lambda case, path: path)
    monkeypatch.setattr(st, 'PoleMotions', lambda *a, **k: NS(prepare=lambda: None))
    monkeypatch.setattr(st, '_checkpoint_identity', lambda **k: {})
    monkeypatch.setattr(st, '_make_optimizers', lambda *a: None)
    monkeypatch.setattr(st, 'freeze_pole_input_columns', lambda *a: None)
    updates, collections, replays = [], [], []
    def optimizer():
        return NS(param_groups=[{}], state_dict=lambda: {}, load_state_dict=lambda _: None)
    def improve(batch, *args, **kwargs):
        updates.append((batch, kwargs))
        return {'imagined_reward': 0.1}
    algorithm = NS(robot=modules[0], actor=modules[1], model=modules[2], critic=modules[3],
                   model_optimizer=optimizer(), actor_optimizer=optimizer(), critic_optimizer=optimizer(),
                   fit_model=lambda batch: {}, improve_policy=improve)
    monkeypatch.setattr(st, 'SourcePoleSuperMimic', lambda *a, **k: algorithm)
    def collect(*args, **kwargs):
        collections.append((kwargs['stage'].name, kwargs['seed']))
        kwargs['budget'].charge(12)
        return [NS(steps=12)]
    monkeypatch.setattr(st, '_collect_episodes', collect)
    def replay(*args, **kwargs):
        generation = len(replays) + 1
        replays.append(generation)
        return NS(sample=lambda *a, **k: generation)
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
        monkeypatch.setattr(st, 'load_resume_checkpoint', lambda *a, **k: (payload, resume_update))
    run = train_source(path, tmp_path / 'fake-resume.pt' if resume_update else None)
    expected = {0: [1, 1, 2, 2, 3, 3, 4, 4], 3: [1, 2, 2, 3, 3], 4: [1, 1, 2, 2]}[resume_update]
    assert [batch for batch, _ in updates] == expected
    assert len(set(seed for _, seed in collections)) == len(collections)
    assert not (run / 'best.pt').exists()
    for update in [4, 6, 8]:
        snapshot = run / f'checkpoint-{update:06d}.pt'
        assert snapshot.exists() == (update > resume_update)
        if snapshot.exists():
            saved = torch.load(snapshot, weights_only=True)
            assert saved['update'] == update and saved['stage'] == 'actor'
            assert 'actor_optimizer' in saved and saved['config'] == config
    metrics = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
    assert metrics[-1]['collection_environment_steps'] == len(collections) * 12
    assert metrics[-1]['replay_transitions'] == 12
    assert metrics[-1]['environment_steps'] > metrics[-1]['collection_environment_steps']
    # Snapshot-only updates do not reuse an older evaluation or select best.
    for row in metrics:
        if row['update'] in [4]:
            assert 'evaluation' not in row
    for update, (_, kwargs) in enumerate(updates, start=resume_update + 1):
        assert kwargs['update_actor'] == (update >= 3)
        assert kwargs['log_reward_terms'] == (update in [2, 3, 4, 6, 8])
