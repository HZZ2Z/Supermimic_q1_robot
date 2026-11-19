"""Versioned experiment identity and complete update-boundary state."""

import hashlib
import json
import random
import sys
import importlib.metadata
from pathlib import Path
import numpy as np
import torch
from . import ROOT, project_path
from .cam_rewards import CAMConfig


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def semantic_config(config):
    result = dict(config)
    for key in ("output_root", "device", "gpu_min_free_gb", "wandb_mode"):
        result.pop(key, None)
    for key in ("manifest", "source_manifest"):
        result[key] = file_hash(project_path(config[key]))
    result["cam"] = CAMConfig.parse(config.get("cam")).to_dict()
    return result


def semantic_hash(config):
    return hashlib.sha256(json.dumps(semantic_config(config), sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_identity():
    paths = []
    for directory in ("courses/pole-weaving", "shared/supermimic", "shared/q1-mujoco", "3rdparty/SuperMimic/src"):
        paths.extend((ROOT / directory).rglob("*.py"))
    paths += [ROOT / name for name in ("requirements.txt", "environment.yml", "setup.py")]
    paths += [ROOT / name for name in ("scripts/cam_local.py", "scripts/prepare_lr_replay.py") if (ROOT / name).is_file()]
    files = {str(path.relative_to(ROOT)): file_hash(path) for path in sorted(paths)}
    return {"sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(), "files": files}


def environment_identity():
    return {"python": sys.version, "packages": dict(sorted(
        (d.metadata['Name'], d.version) for d in importlib.metadata.distributions() if d.metadata['Name']))}


def capture_rng():
    numpy = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [numpy[0], torch.from_numpy(numpy[1].astype(np.int64)), *numpy[2:]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], numpy[1].numpy().astype(np.uint32), *numpy[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("CUDA RNG checkpoint requires CUDA for exact resume")
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def checkpoint_config(saved, explicit_path=None):
    config = saved.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint is missing its experiment configuration")
    CAMConfig.parse(config.get("cam"))
    identity = saved.get("identity", {})
    if identity.get("experiment_schema_version") == 2:
        if semantic_hash(config) != identity.get("semantic_config_sha256"):
            raise ValueError("Checkpoint experiment configuration or asset contents differ")
    else:
        if CAMConfig.parse(config.get("cam")).enabled:
            raise ValueError("Legacy checkpoint cannot claim a CAM experiment")
    if explicit_path is not None:
        explicit = json.loads(Path(explicit_path).read_text())
        if semantic_config(explicit) != semantic_config(config):
            raise ValueError("Explicit configuration conflicts with checkpoint experiment")
        config = explicit
    return config
