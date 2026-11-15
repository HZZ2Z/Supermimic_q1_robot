import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from q1_supermimic.profiles import resolve_profile

ROOT = Path(__file__).resolve().parents[1]


def test_default_profile_selects_pole_config_and_checkpoint_contract():
    profile = resolve_profile()
    assert profile.name == 'pole-weaving'
    assert profile.config_path == ROOT / 'courses/pole-weaving/pole_task/config.json'
    assert profile.output_root == ROOT / 'logs/q1-pole-weaving'
    assert profile.model_contract == 'source-q1-course-v1'
    assert profile.checkpoint_format == 'supermimic_pole_weaving_profile_v1'
    assert profile == resolve_profile('pole-weaving')


def test_removed_soccer_profile_is_rejected():
    with pytest.raises(ValueError, match='Unknown SuperMimic profile'):
        resolve_profile('soccer')


@pytest.mark.parametrize('output', ['logs/auto', 'logs/custom/run-1'])
def test_custom_config_allows_any_descendant_of_logs(tmp_path, output):
    path = tmp_path / 'custom.json'
    path.write_text(json.dumps({'output_root': output}))
    profile = resolve_profile('pole-weaving', 'custom.json', project_root=tmp_path)
    assert profile.config_path == path
    assert profile.output_root == tmp_path / output


@pytest.mark.parametrize('output', ['logs', 'outside', 'logs/../../outside'])
def test_output_must_stay_strictly_inside_logs(tmp_path, output):
    path = tmp_path / 'custom.json'
    path.write_text(json.dumps({'output_root': output}))
    with pytest.raises(ValueError, match='output must stay inside'):
        resolve_profile('pole-weaving', path, project_root=tmp_path)


def test_output_cannot_escape_logs_through_symlink(tmp_path):
    (tmp_path / 'logs').mkdir()
    (tmp_path / 'elsewhere').mkdir()
    (tmp_path / 'logs/link').symlink_to(tmp_path / 'elsewhere', target_is_directory=True)
    path = tmp_path / 'custom.json'
    path.write_text(json.dumps({'output_root': 'logs/link/run'}))
    with pytest.raises(ValueError, match='output must stay inside'):
        resolve_profile('pole-weaving', path, project_root=tmp_path)


@pytest.mark.parametrize('output', [None, '', 3])
def test_output_root_is_required(tmp_path, output):
    path = tmp_path / 'custom.json'
    path.write_text(json.dumps({'output_root': output}))
    with pytest.raises(ValueError, match='requires config field'):
        resolve_profile('pole-weaving', path, project_root=tmp_path)


def test_missing_config_reports_path(tmp_path):
    with pytest.raises(FileNotFoundError, match='missing.json'):
        resolve_profile('pole-weaving', tmp_path / 'missing.json', project_root=tmp_path)


@pytest.mark.parametrize('selection', [[], ['--profile', 'pole-weaving']])
def test_entrypoint_defaults_to_pole_and_translates_check(monkeypatch, selection):
    from q1_supermimic import train

    # Training loads models and runs on a GPU; capture only its CLI boundary.
    captured = []
    module = ModuleType('pole_task.training')
    def pole_main(argv):
        captured.extend(argv)
        return 7
    module.main = pole_main
    monkeypatch.setitem(sys.modules, 'pole_task.training', module)
    assert train.main([*selection, '--check']) == 7
    assert captured == ['--config', str(ROOT / 'courses/pole-weaving/pole_task/config.json'), '--preflight-only']


def test_entrypoint_forwards_custom_config_and_resume(monkeypatch, tmp_path):
    from q1_supermimic import train

    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'output_root': 'logs/custom'}))
    checkpoint = tmp_path / 'latest.pt'
    captured = []
    module = ModuleType('pole_task.training')
    module.main = lambda argv: captured.extend(argv)
    monkeypatch.setitem(sys.modules, 'pole_task.training', module)
    train.main(['--config', str(config), '--resume', str(checkpoint)])
    assert captured == ['--config', str(config), '--resume-checkpoint', str(checkpoint)]


def test_entrypoint_rejects_soccer_before_loading_training(capsys):
    from q1_supermimic import train

    with pytest.raises(SystemExit) as exc:
        train.main(['--profile', 'soccer'])
    assert exc.value.code == 2
    assert 'invalid choice' in capsys.readouterr().err
