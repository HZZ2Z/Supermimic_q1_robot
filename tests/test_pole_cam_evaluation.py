from copy import deepcopy
from pole_task.cam_evaluation import promote, PROTOCOL


def result():
    row={'root_orientation_rms_rad':1.,**{k:1. for k in PROTOCOL['guardrails']}}
    return {'acceptance':{'success':True},'cases':{k:dict(row) for k in ['nominal-01','eval-01','straight']},'numerical_noise':{}}


def test_promotion_requires_task_and_common_quality():
    base=result();candidate=deepcopy(base)
    for r in candidate['cases'].values():r['root_orientation_rms_rad']=.9
    assert promote(base,candidate)
    candidate['cases']['straight']['foot_slip_mps']=1.06
    assert not promote(base,candidate)
    candidate['cases']['straight']['foot_slip_mps']=1.
    candidate['acceptance']['success']=False
    assert not promote(base,candidate)
    candidate=deepcopy(base)
    for r in candidate['cases'].values():r['cam_own_z_rmse']=0.
    assert not promote(base,candidate)


def test_summary_excludes_diagnostics_and_reports_baseline_failure():
    from pole_task.cam_evaluation import summarize
    b=result();b['identity']={'resolved_config':{'seed':0,'cam':{'group':'B0'}}}
    b['acceptance']['success']=False
    diagnostic=deepcopy(b);diagnostic['identity']['diagnostic_only']=True
    report=summarize([b,diagnostic])
    assert report['formal_slots']==1 and report['conclusion']=='baseline_failed'
    assert report['selected_candidate'] is None


def test_summary_uses_common_metric_and_fixed_tie_break():
    from pole_task.cam_evaluation import summarize
    b=result();b['identity']={'resolved_config':{'seed':0,'cam':{'group':'B0'}}}
    candidates=[]
    for group in ['B_task','B_ref']:
        c=deepcopy(b);c['identity']['resolved_config']['cam']['group']=group
        for row in c['cases'].values():row['root_orientation_rms_rad']=.9
        candidates.append(c)
    report=summarize([b,*candidates])
    assert report['selected_candidate']=='B_ref'
    assert report['conclusion']=='candidate_requires_validation'
