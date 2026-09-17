"""Deterministic source-guided bidirectional joint-space RRT-Connect.

The validity oracle is supplied by the common runtime-hull adapter. A compact
soft source guide and common path objective are optional; no method or outcome
is consumed. Every valid-endpoint free-space call expands the RRT, including
when the direct chord is valid. Edges are subdivided
at a declared maximum joint increment, including both ends and midpoints.
This is resolution-complete collision checking, not exact swept-mesh CCD.
"""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class Budget:
    seed: int = 1729
    iterations: int = 480
    state_checks: int = 18000
    step_rad: float = .18
    edge_resolution_rad: float = .005
    shortcut_attempts: int = 32
    max_solutions: int = 4
    improvement_iterations: int = 24


class Exhausted(RuntimeError):
    pass


class Validator:
    def __init__(self, lower, upper, state_valid, resolution=.005, max_checks=18000):
        self.lower=np.asarray(lower,float);self.upper=np.asarray(upper,float)
        self.oracle=state_valid;self.resolution=float(resolution);self.max_checks=int(max_checks)
        if self.resolution<=0 or np.any(self.lower>self.upper):raise ValueError('Invalid limits/resolution')
        self.calls=0;self.edges=0;self.cache={};self.first_invalid=None

    def state(self,q):
        q=np.asarray(q,float)
        if q.shape!=self.lower.shape or not np.isfinite(q).all() or np.any(q<self.lower) or np.any(q>self.upper):
            self.first_invalid=self.first_invalid or {'reason':'JOINT_LIMIT_OR_NONFINITE'}
            return False
        key=q.tobytes()
        if key not in self.cache:
            if self.calls>=self.max_checks:raise Exhausted('STATE_CHECK_BUDGET')
            self.calls+=1;result=self.oracle(q)
            result=result if isinstance(result,dict) else {'valid':bool(result)}
            self.cache[key]=bool(result['valid'])
            if not result['valid'] and self.first_invalid is None:self.first_invalid=result
        return self.cache[key]

    def edge(self,a,b):
        a=np.asarray(a,float);b=np.asarray(b,float);self.edges+=1
        # Test midpoint even for short chords. Dyadic order quickly detects
        # interior obstacles before completing all declared-resolution samples.
        segments=max(2,int(np.ceil(np.max(np.abs(b-a))/self.resolution)))
        depth=int(np.ceil(np.log2(segments)));segments=2**depth
        if not self.state(a) or not self.state(b):return False
        for level in range(1,depth+1):
            denominator=2**level
            for numerator in range(1,denominator,2):
                if not self.state(a+(numerator/denominator)*(b-a)):return False
        return True


def path_length(path):
    return float(np.linalg.norm(np.diff(np.asarray(path),axis=0),axis=1).sum())


def plan(q_start,q_goal,lower,upper,state_valid,active_indices=None,budget=None,guide=None,clearance=None):
    budget=budget or Budget();a=np.asarray(q_start,float);b=np.asarray(q_goal,float)
    lower=np.asarray(lower,float);upper=np.asarray(upper,float)
    active=np.arange(len(a)) if active_indices is None else np.asarray(active_indices,int)
    passive=np.setdiff1d(np.arange(len(a)),active)
    if not np.array_equal(a[passive],b[passive]):raise ValueError('Planner cannot move fixed joints')
    validator=Validator(lower,upper,state_valid,budget.edge_resolution_rad,budget.state_checks)
    from .path_quality import evaluate,simplify
    result=dict(status='NO_CONNECTING_PATH',path=None,algorithm='SOURCE_GUIDED_RRT_CONNECT',
                budget=asdict(budget),direct_path_valid=False,search_used=False,iterations=0,
                active_indices=active.tolist(),edge_check_type='DYADIC_SUBDIVISION_MAX_JOINT_INCREMENT',
                rrt_api_called=True,rrt_search_expanded=0,source_guided_samples=0,global_samples=0,
                local_samples=0,goal_samples=0,path_candidates=[],direct_early_bypass=False,
                canonical_edge_rejections=0)
    features=getattr(guide,'features',None);reference=getattr(guide,'reference',None)
    quality_cache={}
    def quality(path):
        key=np.asarray(path,float).tobytes()
        if key not in quality_cache:quality_cache[key]=evaluate(path,lower,upper,active,features,reference,clearance)
        return quality_cache[key]
    solutions=[];signatures=set();first_solution_iteration=None;solution_counter=0
    def retain(path,origin):
        nonlocal first_solution_iteration,solution_counter
        path=np.asarray(path)
        # Canonicalize collinear geometry for meaningful solution multiplicity.
        ids=[0]
        for i in range(1,len(path)-1):
            v=path[i]-path[ids[-1]];w=path[i+1]-path[i]
            if np.linalg.norm(v)<1e-10:continue
            if np.linalg.norm(w)>1e-10 and np.dot(v,w)/(np.linalg.norm(v)*np.linalg.norm(w))>1.-1e-10:continue
            ids.append(i)
        ids.append(len(path)-1);path=path[ids]
        key=path.round(8).tobytes()
        if key in signatures:return
        # Collinear removal changes the dyadic sample locations of an edge.
        # Checked short tree edges do not certify the merged representation.
        # Reject newly detected invalid geometry before it enters ranking;
        # these checks consume the existing shared state-check budget.
        if not all(validator.edge(x,y) for x,y in zip(path[:-1],path[1:])):
            result['canonical_edge_rejections']+=1
            return
        signatures.add(key);solution_counter+=1
        row=dict(path_id='path_'+str(solution_counter),planning_seed=budget.seed,origin=origin,
            waypoints=path,**quality(path))
        solutions.append(row);solutions.sort(key=lambda x:(x['score'],x['path_id']))
        del solutions[budget.max_solutions:]
        if first_solution_iteration is None:first_solution_iteration=result['iterations']
    def finish(reason=None,path=None):
        result.update(reason=reason,state_checks=validator.calls,edge_checks=validator.edges,
                      first_invalid=validator.first_invalid)
        if path is not None:
            result.update(status='PATH_FOUND',path=np.asarray(path),path_length_rad=path_length(path))
        return result
    rng=np.random.default_rng(budget.seed)
    trees=[([a.copy()],[-1]),([b.copy()],[-1])]
    def extend(tree,target):
        nodes,parents=tree
        distance=np.linalg.norm(np.asarray(nodes)[:,active]-target[active],axis=1)
        near=int(np.argmin(distance));old=nodes[near];delta=target-old
        length=float(np.max(np.abs(delta[active])))
        if length<1e-12:return near,True
        new=old+min(1.,budget.step_rad/length)*delta
        if not validator.edge(old,new):return None,False
        nodes.append(new);parents.append(near)
        result['rrt_search_expanded']+=1
        return len(nodes)-1,length<=budget.step_rad
    def trace(tree,index):
        nodes,parents=tree;path=[]
        while index>=0:path.append(nodes[index]);index=parents[index]
        return list(reversed(path))
    try:
        if not validator.state(a):return finish('INVALID_START_STATE')
        if not validator.state(b):return finish('INVALID_GOAL_STATE')
        if validator.edge(a,b):
            result['direct_path_valid']=True;retain([a,b],'STRAIGHT_BASELINE_INSIDE_RRT')
        result['search_used']=True
        # Seed one tree along the compact differential proposals. They are
        # ordinary checked RRT extensions, never required intermediate goals.
        # A blocked proposal abandons this seed route; global exploration below
        # still runs under the same total state-check budget.
        proposals=getattr(guide,'proposals',[])
        if len(proposals):
            index=0;route_ok=True
            for target in [*proposals,b]:
                result['source_guided_samples']+=1
                reached=False
                for _ in range(int(np.ceil(np.max(upper-lower)/budget.step_rad))+2):
                    if validator.calls>=int(.5*budget.state_checks):break
                    index,reached=extend(trees[0],np.asarray(target))
                    if index is None or reached:break
                if not reached:route_ok=False;break
            if route_ok:retain(trace(trees[0],index),'SOURCE_GUIDED_RRT_EXTENSIONS')
        for iteration in range(budget.iterations):
            # Reserve a fixed part of the same budget for final geometry edits.
            if validator.calls>=int(.75*budget.state_checks):break
            if first_solution_iteration is not None and iteration>=first_solution_iteration+budget.improvement_iterations and result['rrt_search_expanded']>0:break
            result['iterations']=iteration+1;side=iteration%2;other=1-side
            sample=a.copy()
            proposals=getattr(guide,'proposals',[])
            if iteration%4==0 and len(proposals):
                sample[active]=np.asarray(proposals[(iteration//4)%len(proposals)])[active]
                if iteration>=4:sample[active]=np.clip(sample[active]+rng.normal(0,.08,len(active)),lower[active],upper[active])
                result['source_guided_samples']+=1
            elif iteration%4==3:
                sample[active]=trees[other][0][-1][active]
                result['goal_samples']+=1
            elif iteration%4==2:
                # Deterministic local-biased samples plus full-box exploration.
                sample[active]=np.clip((a[active]+b[active])/2+rng.normal(0,.55,len(active)),lower[active],upper[active])
                result['local_samples']+=1
            else:
                sample[active]=rng.uniform(lower[active],upper[active]);result['global_samples']+=1
            index,_=extend(trees[side],sample)
            if index is None:continue
            target=trees[side][0][index]
            for _ in range(int(np.ceil(np.max(upper-lower)/budget.step_rad))+2):
                match,reached=extend(trees[other],target)
                if match is None:break
                if reached:
                    p=trace(trees[side],index);q=trace(trees[other],match)
                    path=p+list(reversed(q))[1:] if side==0 else q+list(reversed(p))[1:]
                    retain(path,'RRT_TREE_CONNECTION');break
    except Exhausted as error:
        result['budget_stop_reason']=str(error)
    if not solutions:return finish(result.get('budget_stop_reason','FIXED_ITERATION_BUDGET_EXHAUSTED'))
    selected=solutions[0];path=np.asarray(selected['waypoints']);result['path_candidates']=solutions
    result.update(selected_path_id=selected['path_id'],selected_path_quality=quality(path),
        distinct_solutions_generated=solution_counter,source_guidance=guide.evidence() if hasattr(guide,'evidence') else None)
    # Smoothing has its own reserved share, never an episode-specific retry.
    try:path,smoothing=simplify(path,validator,quality,rng,budget.shortcut_attempts)
    except Exhausted:
        # The retained path already has checked edges; keep it if a fixed
        # simplification budget expires, without claiming an unchecked edit.
        path=np.asarray(selected['waypoints']);smoothing=dict(active=True,budget_exhausted=True,changes=[],full_geometry_revalidated=False)
    try:
        final_valid=all(validator.edge(x,y) for x,y in zip(path,path[1:]))
    except Exhausted:return finish('FINAL_REVALIDATION_BUDGET')
    if not final_valid:return finish('FINAL_REVALIDATION_FAILED')
    smoothing['full_geometry_revalidated']=True
    result.update(smoothing=smoothing,final_path_quality=quality(path),
        straight_path_selected=len(path)==2,nontrivial_rrt_path_selected=len(path)>2,
        full_geometry_revalidated=True)
    return finish(path=path)


def validate_path(path,lower,upper,state_valid,resolution=.005,max_checks=200000):
    checker=Validator(lower,upper,state_valid,resolution,max_checks)
    try:valid=all(checker.edge(a,b) for a,b in zip(path[:-1],path[1:])) and checker.state(path[0])
    except Exhausted:valid=False
    return dict(valid=bool(valid),state_checks=checker.calls,edge_checks=checker.edges,
                max_joint_increment_rad=resolution,first_invalid=checker.first_invalid)
