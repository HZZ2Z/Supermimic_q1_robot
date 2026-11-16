"""Read-only paired action screening. Never call train_source or an optimizer."""
from pathlib import Path
import sys,json,argparse,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'courses/pole-weaving'),str(ROOT/'3rdparty/SuperMimic/src')]
import torch,mujoco
from supermimic.data import Q1Motions
from supermimic.reward_tuning import reward_tuning_terms
from pole_task.source_runtime import SOURCE_REWARD_TUNING,run_source_episode,advance_source_context
from pole_task.source_playback import load_course_bundle
from pole_task.models import build_source_models
from pole_task.source_profile import source_goal_features
from pole_task.source_environment import SourcePoleEnvironment
from pole_task.foot_contact import foot_slip_phase_metrics

def masks(samples):
    loaded=samples[...,0]>.5;phase=np.full(loaded.shape,-1)
    def runs(a):
        b=np.flatnonzero(np.diff(np.r_[False,a,False].astype(int)));return zip(b[::2],b[1::2])
    for f in range(2):
        closed=loaded[:,f].copy()
        for a,b in runs(~closed):
            if a>0 and b<len(closed) and b-a<=10:closed[a:b]=True
        for a,b in runs(closed):
            ix=np.arange(a,b)[loaded[a:b,f]];age=ix-a;left=b-1-ix
            touch=(age<20)&((left>=20)|(age<=left));lift=(left<20)&~touch
            phase[ix,f]=0;phase[ix[touch],f]=3 if a==0 else 1;phase[ix[lift],f]=3 if b==len(closed) else 2
    idx=np.arange(len(samples))[:,None];age=idx-np.maximum.accumulate(np.where(~loaded,idx,-1),axis=0)
    continuous=(phase==0)&(age>5);reload=(phase==0)&(age<=5)
    expected=foot_slip_phase_metrics(samples,.002)
    np.testing.assert_allclose((samples*(phase==0)[...,None]).sum(axis=(0,1))[0]*.002,sum(expected['sustained']['contact_time_s']))
    return continuous,reload,age,loaded

def gradients(actor,world,critic,robot,motions,clip,ep,anchors):
    refs=motions.sequence(torch.full((12,),clip),torch.tensor(anchors),torch.arange(43))
    state=torch.from_numpy(ep.states[anchors]);context=torch.from_numpy(ep.contexts[anchors]);history=torch.stack([torch.from_numpy(ep.observations[a-9:a+1]) for a in anchors])
    first=actor(history,source_goal_features(robot,refs[:,1:11],state)).detach().requires_grad_(True)
    survival=torch.ones(12);parts={k:torch.zeros(12) for k in ('base','arm','slip','landing')}
    for t in range(32):
        action=first if t==0 else actor(history,source_goal_features(robot,refs[:,t+1:t+11],state))
        nxt=world(state,action,context);target=refs[:,t+1]
        reward_state=torch.cat((state[:,:57],robot.previous_action_from_observation(history[:,-1])),1)
        terms=reward_tuning_terms(robot,nxt,target,action,SOURCE_REWARD_TUNING,context)
        values={'base':robot.reward(reward_state,nxt,target,action,.02),'arm':.02*terms['arm_joint_penalty'],'slip':.02*terms['foot_slip_penalty'],'landing':.02*terms['foot_landing_penalty']}
        for k,v in values.items():parts[k]=parts[k]+.99**t*survival*v
        survival=survival*(1-world.termination_logits(robot,nxt,target).sigmoid())
        obs=robot.observation(nxt,context)
        noise=torch.randn(obs.shape,generator=torch.Generator().manual_seed(20261116+t))*robot.noise_scale()
        history=torch.cat((history[:,1:],(obs+noise)[:,None]),1);context=advance_source_context(robot,context,action);state=nxt
    if float(survival.min().detach())<=.01:raise RuntimeError('Screen anchors reached absorption; do not silently change the screening population')
    parts['bootstrap']=.99**32*survival*critic(state,source_goal_features(robot,refs[:,33:43],state),context)
    gs={k:torch.autograd.grad(v.sum(),first,retain_graph=True)[0].detach() for k,v in parts.items()}
    total=sum(gs.values());torch.testing.assert_close(total,torch.autograd.grad(sum(parts.values()).sum(),first)[0],rtol=1e-4,atol=1e-5)
    torch.testing.assert_close(first.detach(), torch.from_numpy(ep.actions[anchors]), rtol=1e-4, atol=1e-5)
    return first.detach(),gs

@torch.no_grad()
def trial(env,snap,ep,anchor,action,actor,robot,motions,clip,continuous_out):
    mujoco.mj_copyData(env.data,env.model,snap['data'])
    env.frame=anchor;env.steps=anchor;env.collision=False;env.pole_contacts.clear()
    env.action[:]=snap['action'];env.last_action[:]=snap['last_action'];env.context=snap['context'].clone()
    env._foot_contact_samples=list(snap['samples']);env._foot_contact_totals[:]=snap['totals']
    history=torch.from_numpy(ep.observations[anchor-9:anchor+1])[None].clone()
    state=torch.from_numpy(ep.states[anchor])[None];context=torch.from_numpy(ep.contexts[anchor])[None].clone()
    fall=False;collision=False;errors=[]
    # Two extra controls supply the trailing 40 ms needed by the phase classifier.
    for t in range(12):
        refs=motions.sequence(torch.tensor([clip]),torch.tensor([anchor+t]),torch.arange(1,11))
        a=action[None] if t==0 else actor(history,source_goal_features(robot,refs,state))
        ns,tr=env.step(a[0].numpy());state=torch.from_numpy(ns)[None]
        if t<10:
            fall |= bool(robot.terminated(state,refs[:,0])[0]);collision |= tr['pole_collision']
            errors.append(float(np.square(ns[:2]-refs[0,0,:2].numpy()).mean()))
        obs=robot.observation(state,context);history=torch.cat((history[:,1:],obs[:,None]),1)
        context=advance_source_context(robot,context,a)
    samples=np.asarray(env._foot_contact_samples);continuous,reload,age,loaded=masks(samples)
    begin=len(snap['samples']);end=begin+100;region=np.zeros_like(loaded);region[begin:end]=True
    c=(samples*(continuous&region)[...,None]).sum(axis=(0,1))*.002
    r=(samples*(reload&region)[...,None]).sum(axis=(0,1))*.002
    events=0
    for foot in range(2):
        for i in np.flatnonzero((age[:,foot]==1)&loaded[:,foot]):
            if i>0 and (reload[i:min(i+5,len(samples)),foot]&region[i:min(i+5,len(samples)),foot]).any():events+=1
    return {'continuous_time':float(c[0]),'continuous_integral':float(c[1]),'reload_integral':float(r[1]),'events':events,'loaded_time':float(samples[begin:end,:,0].sum()*.002),'fall':fall,'collision':collision,'root_rmse':float(np.sqrt(np.mean(errors)))}

def select(rows):
    decisions={};aggregates={}
    for w in (.05,.075,.10):
        aggregates[str(w)]={}
        for eps in (.01,.02):
            aggregates[str(w)][str(eps)]={}
            for case in ('nominal-01','straight','left_turn','right_turn'):
                selected=[r for r in rows if r['weight']==w and r['epsilon']==eps and r['sign']==1 and r['case']==case]
                assert len(selected)==36
                time=sum(r['continuous_time'] for r in selected)
                if time<=0:raise RuntimeError('No continuous-load observations; screening cannot pass')
                aggregates[str(w)][str(eps)][case]={'continuous':sum(r['continuous_integral'] for r in selected)/time,'reload':sum(r['reload_integral'] for r in selected)/36,'events':sum(r['events'] for r in selected),'loaded':sum(r['loaded_time'] for r in selected),'conflicts':sum(r['conflict'] for r in selected),'failures':sum(r['fall'] or r['collision'] for r in selected)}
    for w in (.075,.10):
        reasons=[]
        for eps in (.01,.02):
            for case in ('nominal-01','straight','left_turn','right_turn'):
                a=aggregates['0.05'][str(eps)][case];b=aggregates[str(w)][str(eps)][case]
                checks={'continuous_not_worse':b['continuous']<=a['continuous']+1e-12,'reload_not_worse':b['reload']<=a['reload']+1e-12,'events_not_worse':b['events']<=a['events'],'load_within_5pct':abs(b['loaded']/a['loaded']-1)<=.05,'conflicts_not_worse':b['conflicts']<=a['conflicts'],'no_failures':b['failures']==0}
                if case=='left_turn':checks['left_minimum_improvement']=b['continuous']<=a['continuous']*.95 or b['reload']<=a['reload']*.9
                reasons.extend(f'{eps}/{case}/{k}' for k,v in checks.items() if not v)
        decisions[str(w)]={'passed':not reasons,'failed_checks':reasons}
    winner=next((w for w in (.075,.10) if decisions[str(w)]['passed']),None)
    return {'selected_weight':winner,'decisions':decisions,'aggregates':aggregates}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'outputs/reward-sampling-screen');args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'trials.jsonl').exists():raise FileExistsError('Use a new output directory to preserve the frozen screening run')
    torch.set_num_threads(1)
    def forbidden(*a,**k):raise AssertionError('Optimizer steps forbidden')
    torch.optim.Adam.step=forbidden
    original=Q1Motions.prepare
    def prepare(self):self.cache_dir=Path('/tmp/q1-phase-test-cache');return original(self)
    Q1Motions.prepare=prepare
    robot,motions,cases,config=load_course_bundle(reference_dir=out/'references',device='cpu')
    checkpoints={'parent':'run-20261115-160417/checkpoint-002100.pt','single':'run-20261115-221026/checkpoint-002200.pt','two':'run-20261115-230813/checkpoint-002200.pt'}
    (out/'protocol.json').write_text(json.dumps({'checkpoints':{k:{'path':v,'sha256':hashlib.sha256((ROOT/'logs/q1-pole-weaving'/v).read_bytes()).hexdigest()} for k,v in checkpoints.items()},'weights':[.05,.075,.10],'epsilon_rms':[.01,.02],'horizon':32,'physical_controls':10,'tail_controls':2,'optimizer_steps':0},indent=2))
    rows=[];probe=[]
    for label,path in checkpoints.items():
        actor,world,critic=build_source_models(robot);saved=torch.load(ROOT/'logs/q1-pole-weaving'/path,map_location='cpu',weights_only=True)
        for name,module in [('actor',actor),('model',world),('critic',critic)]:
            module.load_state_dict(saved[name],strict=True);module.eval()
            for p in module.parameters():p.requires_grad_(False)
        del saved
        for case_name in ('nominal-01','straight','left_turn','right_turn'):
            case=cases[case_name];clip=next(i for i,c in enumerate(motions.cases) if c.case_id==case_name)
            anchors=np.linspace(9,case.control_steps-32,12,dtype=int);snapshots={}
            env=SourcePoleEnvironment(case,out/f'{label}-{case_name}.xml',robot,seed=20261103)
            original_step=env.step
            def capture(action):
                if env.frame in anchors:
                    d=mujoco.MjData(env.model);mujoco.mj_copyData(d,env.model,env.data)
                    snapshots[env.frame]={'data':d,'action':env.action.copy(),'last_action':env.last_action.copy(),'context':env.context.clone(),'samples':list(env._foot_contact_samples),'totals':env._foot_contact_totals.copy()}
                return original_step(action)
            env.step=capture
            ep=run_source_episode(actor,robot,motions,case,clip,out/f'{label}-{case_name}.xml',start_frame=0,environment=env,device='cpu')
            env.step=original_step
            original_masks=masks(np.asarray(env._foot_contact_samples))
            if len(snapshots)!=12:raise RuntimeError('Reference rollout ended before all anchors; C cannot be selected')
            actions,gs=gradients(actor,world,critic,robot,motions,clip,ep,anchors)
            for j,anchor in enumerate(anchors):
                base=trial(env,snapshots[int(anchor)],ep,int(anchor),actions[j],actor,robot,motions,clip,None)
                assert base == trial(env,snapshots[int(anchor)],ep,int(anchor),actions[j],actor,robot,motions,clip,None)
                rows.append({'checkpoint':label,'case':case_name,'frame':int(anchor),'weight':.05,'epsilon':0.,'sign':0,'conflict':False,**base})
                for w in (.05,.075,.10):
                    g=sum(gs.values())[j]+(w/.05-1)*gs['slip'][j];conflict=bool(torch.dot(g,gs['slip'][j])<0)
                    rms=g.square().mean().sqrt();direction=g/rms.clamp_min(1e-12)
                    probe.append({'checkpoint':label,'case':case_name,'frame':int(anchor),'weight':w,'anchor_contact_state':[('continuous' if original_masks[0][int(anchor)*10-1,f] else 'reloading' if original_masks[1][int(anchor)*10-1,f] else 'loaded_other' if original_masks[3][int(anchor)*10-1,f] else 'unloaded') for f in range(2)],'gradient_rms':float(rms),'component_norms':{k:float(v[j].norm()) for k,v in gs.items()},'conflict':conflict})
                    for eps in (.01,.02):
                        for sign in (1,-1):
                            result=trial(env,snapshots[int(anchor)],ep,int(anchor),actions[j]+sign*eps*direction,actor,robot,motions,clip,None)
                            row={'checkpoint':label,'case':case_name,'frame':int(anchor),'weight':w,'epsilon':eps,'sign':sign,'conflict':conflict,'predicted_first_order_objective_delta':float(torch.dot(g, sign*eps*direction)),**result};rows.append(row)
            with (out/'trials.jsonl').open('w') as f:
                for row in rows:f.write(json.dumps(row)+'\n')
            (out/'gradients.json').write_text(json.dumps(probe,indent=2))
            print(label,case_name,'complete',len(rows),'trials',flush=True)
    result=select(rows);result['anchors']=144;result['optimizer_steps']=0
    (out/'decision.json').write_text(json.dumps(result,indent=2)+'\n')
    if result['selected_weight'] is not None:
        cfg=json.loads((ROOT/'experiments/reward-sampling/B.json').read_text());cfg.update(foot_slip_weight=result['selected_weight'],output_root='logs/q1-pole-weaving/experiment-C')
        path=ROOT/'experiments/reward-sampling/C.json'
        with path.open('x') as f:json.dump(cfg,f,indent=2)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
