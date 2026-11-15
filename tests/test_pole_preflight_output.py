"""The real preflight must use the same output policy as the convenience CLI."""

import importlib
import json

import pytest


class ReachedDataValidation(Exception):
    pass


@pytest.fixture
def preflight_output_boundary(tmp_path, monkeypatch):
    module = importlib.import_module('pole_task.preflight')
    # Output checks precede course parsing. Replace external checkpoint validation
    # and stop at that next boundary; keep the actual preflight path checks live.
    manifest_path = module.COURSE_ROOT / 'pole_task/source_manifest.json'
    report = module.verify_source_manifest(manifest_path, purpose='runtime')
    report['artifact_paths'][report['warmup_checkpoint_id']] = str(tmp_path / 'warmup.pt')
    monkeypatch.setattr(module, 'verify_source_manifest', lambda _: report)
    def stop_at_data(_):
        raise ReachedDataValidation
    monkeypatch.setattr(module, 'load_manifest', stop_at_data)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    def check(output):
        config = tmp_path / 'config.json'
        config.write_text(json.dumps({'output_root': output, 'manifest': 'unused.json'}))
        return module.preflight(config)
    return check


@pytest.mark.parametrize('output', ['logs/auto', 'logs/latest/run', 'logs/q1', 'logs/q1-pole-weaving'])
def test_preflight_accepts_outputs_without_soccer_reservations(preflight_output_boundary, output):
    with pytest.raises(ReachedDataValidation):
        preflight_output_boundary(output)


@pytest.mark.parametrize('output', ['logs', 'outside', 'logs/../../outside'])
def test_preflight_rejects_outputs_outside_dedicated_logs(preflight_output_boundary, output):
    with pytest.raises(ValueError, match='output must stay'):
        preflight_output_boundary(output)
