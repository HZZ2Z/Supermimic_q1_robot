"""User-run, serial GPU checks and source freeze; never starts formal training."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

from pole_task import ROOT
from pole_task.experiment import source_identity, file_hash, environment_identity
from pole_task.local_training import VARIANTS
from pole_task.source_recipe import resolve_source_recipe
from pole_task.source_training import atomic_json


def check_diagnostic(row, recipe, operation):
    if not row.get('diagnostic_only') or row.get('device') != 'cuda:0':
        raise ValueError('Readiness requires CUDA diagnostics, not formal/CPU results')
    phases=row['phases']
    if [p['update'] for p in phases] != [1,2001,2026]:
        raise ValueError('Missing warmup, first Actor or refreshed Actor phase')
    if [p['replay_batch_updates'] for p in phases] != [[1],[2001],[2001,2026]]:
        raise ValueError('Resource check did not exercise two Actor replay batches')
    if operation=='profile' and row['effective_recipe']!=recipe.to_dict():
        raise ValueError('Resource diagnostic did not use the complete resolved recipe')
    for p in phases:
        delta=p['actor_parameter_delta_l2']
        if not math.isfinite(delta) or (p['update']==1 and delta!=0) or (p['update']>1 and delta<=0):
            raise ValueError('Actor freeze/update check failed')
        if not 0 < p['peak_allocated_mib'] <= p['peak_reserved_mib']:
            raise ValueError('Missing GPU memory evidence')
        if p['update']>1 and not p['metrics']['actor_grad_norm']>0:
            raise ValueError('Actor gradient check failed')


def main():
    root=ROOT/'logs/q1-pole-weaving/lr-replay-v1'
    output=root/'preflight'/time.strftime('%Y%m%d-%H%M%S')
    output.mkdir(parents=True,exist_ok=False)
    receipt=root/'readiness.json'
    # A failed recheck cannot leave an older receipt looking like the new result.
    receipt.unlink(missing_ok=True)
    source=source_identity();environment=environment_identity()
    configs={v:ROOT/'configs/lr-replay-v1'/f'{v}.json' for v in VARIANTS}
    hashes={v:file_hash(p) for v,p in configs.items()}
    record={'pid':os.getpid(),'command':[sys.executable,*sys.argv],'cwd':str(Path.cwd()),
            'python':sys.executable,'started_unix':time.time(),'status':'IN_PROGRESS',
            'source_sha256':source['sha256'],'environment':environment,
            'configuration_hashes':hashes,'variants':list(VARIANTS),'diagnostics':[]}
    atomic_json(output/'run.json',record)
    try:
        for operation in ('smoke','profile'):
            for variant,path in configs.items():
                target=output/f'{variant}-{operation}'
                command=[sys.executable,str(ROOT/'scripts/cam_local.py'),operation,
                         '--config',str(path),'--output',str(target),'--device','cuda:0']
                print(f'{variant} {operation}: {target}.log',flush=True)
                subprocess.run(command,cwd=ROOT,env={**os.environ,'WANDB_MODE':'offline'},check=True)
                row=json.loads((target/'diagnostic.json').read_text())
                check_diagnostic(row,resolve_source_recipe(path),operation)
                if row['source']['sha256']!=source['sha256']:
                    raise ValueError('Source changed during resource checks')
                record['diagnostics'].append(str(target/'diagnostic.json'))
                atomic_json(output/'run.json',record)
        if source_identity()!=source or environment_identity()!=environment or any(file_hash(p)!=hashes[v] for v,p in configs.items()):
            raise ValueError('Source, dependencies or configs changed during resource checks')
        atomic_json(output/'source.json',source)
        with tarfile.open(output/'frozen-source.tar.gz','w:gz') as archive:
            for name in source['files']:
                archive.add(ROOT/name,arcname=name)
            for path in configs.values():
                archive.add(path,arcname=str(path.relative_to(ROOT)))
        record.update(status='COMPLETE',exit_code=0,finished_unix=time.time(),
                      frozen_source=str(output/'frozen-source.tar.gz'))
        atomic_json(receipt,record)
        print(f'Resource checks passed. Frozen receipt: {receipt}\nFormal training has NOT started.',flush=True)
    except BaseException as exc:
        record.update(status='FAILED',exit_code=1,error=repr(exc),finished_unix=time.time())
        raise
    finally:
        atomic_json(output/'run.json',record)


if __name__=='__main__':
    main()
