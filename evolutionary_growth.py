"""Experimental connection proposals and held-out evaluation; never a motor policy.

Proposals do not mutate the running graph. An evaluation accepts a candidate for
further testing, not proof of biological function or permission for live promotion.
"""
import re
import numpy as np

TARGET_CLASSES={'cb_intrinsic','ol_intrinsic','vnc_intrinsic','ascending_neuron',
                'descending_neuron','visual_projection','visual_projection_tbc'}

def _side(core,index):
    side=str(core.side[index]).lower()
    if side in ('l','left'):return 'L'
    if side in ('r','right'):return 'R'
    match=re.search(r'_([LR])(?:__G.*)?$',str(core.instance[index]))
    return match.group(1) if match else None

def propose_connections(core,candidate,max_edges=4):
    source=int(candidate['index'])
    result=dict(format='connection-trial-v1',status='blocked',source_index=source,
                source_body=int(core.bodies[source]),edges=[],reason='',live_graph_modified=False,
                locality='annotated same-type/same-nerve peers, not physical distance',
                required_evidence=['held-out stimulus discrimination','quiet-state recovery',
                                   'bounded saturation','matched control replay'])
    kind=str(core.types[source]);side=_side(core,source)
    if not kind or side is None or float(core.sign[source])==0:
        result['reason']='Missing type/side or unsupported modulatory source';return result
    nerve=str(core.entry_nerve[source])
    if not nerve:
        result['reason']='No peripheral/circuit locality annotation for safe recruitment';return result
    peers=np.flatnonzero((core.types==kind)&(core.entry_nerve==nerve)&(core.nt_code==core.nt_code[source]))
    peers=[int(i) for i in peers if int(i)!=source and _side(core,int(i))==side]
    existing=set(map(int,core.indices[core.indptr[source]:core.indptr[source+1]]))
    options={}
    for peer in peers:
        for edge in range(int(core.indptr[peer]),int(core.indptr[peer+1])):
            target=int(core.indices[edge])
            if target==source or target in existing or str(core.superclass[target]) not in TARGET_CLASSES:continue
            strength=abs(float(core.w0[edge])*float(core.edge_gain[edge]))
            if not np.isfinite(strength) or strength<=0:continue
            # No newly recruited branch receives more than 5% of its donor's
            # structural envelope or 0.05 mV per event, whichever is smaller.
            fraction=min(.05,.05/strength)
            option=dict(source_index=source,target_index=target,donor_edge_index=edge,
                        donor_source_index=peer,donor_fraction=fraction,
                        trial_event_mv=strength*fraction,transmitter=str(core.nt[source]))
            if target not in options or strength>options[target][0]:options[target]=(strength,option)
    result['edges']=[v[1] for _,v in sorted(options.items(),key=lambda item:(-item[1][0],item[0]))[:max_edges]]
    result['status']='proposed' if result['edges'] else 'blocked'
    result['reason']='Bounded connection candidates require isolated replay evaluation' if result['edges'] else 'No compatible annotated peer pathway found'
    return result

def evaluate_trial(control_evoked,trial_evoked,control_quiet,trial_quiet,*,saturation_hz=120.):
    """Evaluate held-out rate arrays: stimulus x repetition x target.

    Matched replay inputs and graph provenance must be supplied by the caller.
    This score measures a limited discriminability proxy, not behavioral fitness.
    """
    if not np.isfinite(saturation_hz) or saturation_hz<=0:raise ValueError("Invalid saturation threshold")
    ce,te,cq,tq=[np.asarray(a,dtype=float) for a in (control_evoked,trial_evoked,control_quiet,trial_quiet)]
    if ce.ndim!=3 or ce.shape!=te.shape or min(ce.shape)<2 or cq.shape!=tq.shape or cq.ndim!=2 or cq.shape[1]!=ce.shape[2] or cq.size==0:
        raise ValueError('Need matched repeated held-out stimuli and quiet measurements')
    if any(np.any(~np.isfinite(a)) or np.any(a<0) for a in (ce,te,cq,tq)):
        raise ValueError('Rates must be finite and nonnegative')
    def separation(a):
        means=a.mean(axis=1)
        differences=means[:,None,:]-means[None,:,:]
        signal=float(np.mean(differences**2))
        noise=float(np.mean((a-means[:,None,:])**2))
        return signal/(1.+noise)
    control_score=separation(ce);trial_score=separation(te)
    quiet_excess=float(np.max(tq.mean(axis=0)-cq.mean(axis=0)))
    saturation=float(np.mean(te>=saturation_hz))
    control_saturation=float(np.mean(ce>=saturation_hz))
    # Demand additional selectivity, not uniform gain. Normalize each target's
    # response scale before a second comparison.
    def normalized(a):return a/np.maximum(a.mean(axis=(0,1),keepdims=True),1e-9)
    selective=separation(normalized(te))>separation(normalized(ce))+1e-3
    accepted=bool(selective and trial_score>max(.01,control_score*1.1) and quiet_excess<=.5
                  and saturation<=min(.10,control_saturation+.01))
    return dict(accepted_for_extended_trial=accepted,control_separation=control_score,
                trial_separation=trial_score,selectivity_improved=bool(selective),
                quiet_excess_hz=quiet_excess,saturation_fraction=saturation,
                live_promotion_authorized=False)
