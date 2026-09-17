"""Common bounded geometric path objective; no episode outcome or method input."""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class QualityPolicy:
    clearance: float = .15
    joint_length: float = .15
    cartesian_length: float = .10
    source_deviation: float = .30
    endpoint_continuity: float = .10
    posture_excursion: float = .10
    roughness: float = .07
    joint_margin: float = .03


def resample(path, count=17):
    q=np.asarray(path,float);distance=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(q,axis=0),axis=1))]
    if distance[-1]<1e-12:return np.repeat(q[:1],count,axis=0)
    keep=np.r_[True,np.diff(distance)>1e-12];t=np.linspace(0,distance[-1],count)
    return np.column_stack([np.interp(t,distance[keep],q[keep,j]) for j in range(q.shape[1])])


def evaluate(path, lower, upper, active=None, features=None, reference=None, clearance=None, policy=None):
    """Feasibility is a separate hard gate. All score terms are bounded [0,1].

    Length penalizes travel, endpoint continuity penalizes the endpoint branch,
    and posture excursion measures only motion outside the endpoint joint box.
    No second natural-posture distance duplicates endpoint continuity.
    """
    policy=policy or QualityPolicy();q=np.asarray(path,float);sample=resample(q)
    active=np.arange(q.shape[1]) if active is None else np.asarray(active,int)
    span=np.maximum(np.asarray(upper)-lower,1e-6);a=q[0];b=q[-1]
    dq=np.diff(q[:,active],axis=0);length=float(np.linalg.norm(dq,axis=1).sum())
    norm_length=float(np.linalg.norm(dq/span[active],axis=1).sum()/np.sqrt(len(active)))
    continuity=float(np.mean(((b[active]-a[active])/span[active])**2))
    excess=np.maximum(np.maximum(np.minimum(a,b)-sample,sample-np.maximum(a,b)),0)
    excursion=float(np.sqrt(np.mean((excess[:,active]/span[active])**2)))
    directions=dq/np.maximum(np.linalg.norm(dq,axis=1,keepdims=True),1e-12)
    roughness=float(np.sum(np.linalg.norm(np.diff(directions,axis=0),axis=1)**2))
    margin=float(np.min(np.minimum(sample[:,active]-np.asarray(lower)[active],np.asarray(upper)[active]-sample[:,active])/span[active]))
    positions=np.asarray([features(x)['wrists'] for x in sample]) if features else sample[:,None,:min(3,q.shape[1])]
    cartesian=float(np.linalg.norm(np.diff(positions,axis=0),axis=2).sum(axis=0).mean())
    chord=max(.10,float(np.linalg.norm(positions[-1]-positions[0],axis=1).mean()))
    reference_points=reference(np.linspace(0,1,len(sample))) if reference else positions
    source=float(np.sqrt(np.mean(np.sum((positions-reference_points)**2,axis=2)))) if reference else 0.
    t=np.linspace(0,1,len(sample))[:,None,None]
    source_shape_scale=max(.05,float(np.sqrt(np.mean(np.sum((reference_points-((1-t)*reference_points[0]+t*reference_points[-1]))**2,axis=2)))))
    clearances=np.asarray([clearance(x) for x in sample]) if clearance else None
    minimum=float(np.min(clearances)) if clearance is not None else None
    def squash(x):return float(max(0.,x)/(1.+max(0.,x)))
    terms=dict(clearance=float(np.mean(np.exp(-np.maximum(0.,clearances)/.01))) if minimum is not None else 0.,
        joint_length=squash(norm_length),cartesian_length=squash(cartesian/chord),
        source_deviation=squash((source/source_shape_scale)**2),endpoint_continuity=squash(continuity),
        posture_excursion=squash(excursion/.05),roughness=squash(roughness),joint_margin=float(np.exp(-max(0.,margin)/.05)))
    return dict(joint_path_length_rad=length,cartesian_wrist_path_length_m=cartesian,
        source_motion_deviation_m=source,posture_deviation=excursion,endpoint_continuity=continuity,
        roughness=roughness,minimum_clearance_m=minimum,joint_limit_margin_fraction=margin,
        normalized_terms=terms,weights=asdict(policy),score=float(sum(getattr(policy,k)*v for k,v in terms.items())),
        quality_samples=len(sample),source_normalization_m=source_shape_scale,clearance_scope='Forbidden geometry, capped at30mm; sampled path quality, separate full edge safety gate')


def simplify(path, validator, score, rng, attempts=32):
    """Accept only collision-valid, non-worsening deterministic geometric edits."""
    q=np.asarray(path,float);initial=q.copy();best=score(q);changes=[]
    proposals=[]
    # Longest redundant intervals first, then deterministic random intervals.
    for width in range(len(q)-1,1,-1):
        proposals.extend((i,i+width) for i in range(len(q)-width))
    proposals=proposals[:attempts//2]
    for trial in range(attempts):
        if len(q)<=2:break
        if trial<len(proposals):i,j=proposals[trial];j=min(j,len(q)-1);i=min(i,j-1)
        else:i,j=sorted(rng.choice(len(q),2,replace=False))
        if j<=i+1 or not validator.edge(q[i],q[j]):continue
        candidate=np.r_[q[:i+1],q[j:]];quality=score(candidate)
        if quality['score']<=best['score']+1e-12:
            q=candidate;best=quality;changes.append(dict(kind='COLLISION_AWARE_SHORTCUT',trial=trial))
    # Laplacian corner relaxation, never changing either interaction endpoint.
    for _ in range(2):
        for i in range(1,len(q)-1):
            point=.5*q[i]+.25*(q[i-1]+q[i+1])
            if not validator.edge(q[i-1],point) or not validator.edge(point,q[i+1]):continue
            candidate=q.copy();candidate[i]=point;quality=score(candidate)
            if quality['score']<best['score']-1e-12:
                q=candidate;best=quality;changes.append(dict(kind='COLLISION_AWARE_CORNER_RELAXATION',index=i))
    valid=all(validator.edge(a,b) for a,b in zip(q[:-1],q[1:]))
    if not valid:raise AssertionError('Smoothing generated invalid edge')
    return q,dict(active=True,changes=changes,input_waypoints=initial,final_quality=best,
        full_geometry_revalidated=True,interaction_endpoints_preserved=bool(np.array_equal(q[0],initial[0]) and np.array_equal(q[-1],initial[-1])))
