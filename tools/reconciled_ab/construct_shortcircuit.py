"""Acceptance-equivalent geometry short-circuit; no numerical solver changes.

A conjunction cannot become valid after a demonstrated false conjunct.
Every valid candidate still receives exhaustive geometry inspection. Truncated
invalid diagnostics are labeled and never reported as complete frame counts.
"""
import os,sys,signal,concurrent.futures
from .common import *
from . import construct

AMEND=RUN/'freeze/GEOMETRY_SHORTCIRCUIT_IMPLEMENTATION.json'
def inspect(c,q,h):
    rows=[]
    for f,(v,hh) in enumerate(zip(q,h)):
        try:records=c.inspect(v,*hh)
        except Exception as exc:records=[dict(classification='UNRESOLVED_GEOMETRY',reason=str(exc))]
        row=dict(frame=f,records=records);rows.append(row)
        if any(x['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY') for x in records):
            row.update(inspection_stopped_after_proven_rejection=True,total_trajectory_frames=len(q),remaining_geometry='NOT_INSPECTED; trajectory already rejected',collision_counts_are_lower_bounds=True)
            break
    return rows

def run(key):
    amend=read(AMEND);assert record(Path(__file__))==amend['implementation']
    folder=RUN/'construction'/key;done=folder/'RESULT.json'
    if done.exists() and read(done)['outcome']!='INFRASTRUCTURE_INVALID':
        expected=amend['preexisting_results'].get(key)
        if expected is not None:assert record(done)==expected
        else:
            context=read(folder/'SHORTCIRCUIT_CONTEXT.json');assert context['amendment']==record(AMEND)
        return key,read(done)['outcome']
    save(folder/'SHORTCIRCUIT_CONTEXT.json',dict(amendment=record(AMEND),original_protocol=record(construct.CONTRACT),input_record=next(x for x in read(construct.CONTRACT)['cases'] if x['case']['key']==key),solver_budgets_and_trajectories_unchanged=True))
    construct.inspect=inspect
    return construct.run_case(key)

def stop_and_freeze():
    # Resolve only this run's own numerical workers. No simulator, external
    # user process, or previous recovery/training process is touched.
    victims=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:args=(p/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if b'tools.reconciled_ab.construct' in args:
            victims.append(dict(pid=int(p.name),command=[x.decode() for x in args if x]))
    save(RUN/'audit/OWN_CONSTRUCTION_PROCESS_INTERRUPTION.json',dict(timestamp=now(),reason='Common acceptance-equivalent geometry short-circuit; resume saved q, never expand search or tune outcomes',processes=victims))
    for v in victims:
        try:os.kill(v['pid'],signal.SIGTERM)
        except ProcessLookupError:pass
    # Wait for explicit process termination before any resumed worker starts.
    import time
    for _ in range(60):
        active=[]
        for v in victims:
            try:
                stat=Path(f"/proc/{v['pid']}/stat").read_text().split()
                if stat[2]!='Z':active.append(v['pid'])
            except FileNotFoundError:pass
        if not active:break
        time.sleep(.1)
    assert not active,active
    class Fake:
        def __init__(self,labels):self.labels=iter(labels)
        def inspect(self,*args):return [dict(classification=next(self.labels))]
    for labels in [['CLEAR']*4,['CLEAR','PROXY_ONLY_OVERLAP','CLEAR','CLEAR'],['CLEAR','HARD_SELF_COLLISION','CLEAR','UNRESOLVED_GEOMETRY'],['UNRESOLVED_GEOMETRY']+['CLEAR']*3]:
        short=inspect(Fake(labels),range(4),[(None,None)]*4)
        assert any(x in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY') for x in labels)==any(x['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY') for r in short for x in r['records'])
        if len(short)<4:assert short[-1]['inspection_stopped_after_proven_rejection']
    save(AMEND,dict(created_at=now(),implementation=record(Path(__file__)),original_protocol=record(construct.CONTRACT),
        equivalence_proof='For the common all-frames conjunction, an observed HARD or UNRESOLVED makes the candidate invalid regardless of unexamined suffix. Clear candidates are fully examined. Same q, targets, tracking cost, motion gates, candidate order and maximum budgets. No accepted candidate or selection can change.',
        tests='CLEAR, PROXY_ONLY, HARD, UNRESOLVED suffix equivalence PASS',
        diagnostics='Invalid-candidate collision counts are lower bounds if inspection stops early; remaining frames are NOT_INSPECTED, never CLEAR. Historical exhaustive counts retained.',
        preexisting_results={p.parent.name:record(p) for p in (RUN/'construction').glob('*/RESULT.json')},
        no_physical_behavior_changed=True,no_search_budget_increase=True,method_identity_used=False))
    save(RUN/'freeze/PHYSICS_PERFORMANCE_BINDING.json',dict(physics_protocol=record(RUN/'freeze/PHYSICS_PROTOCOL.json'),construction_protocol=record(construct.CONTRACT),performance_amendment=record(AMEND),physical_behavior_unchanged=True,per_case_cache_binding='Original input/dependency signature plus SHORTCIRCUIT_CONTEXT.json; immutable preexisting result hashes in amendment',no_replay_or_task_outcome_tuning=True))
    log('COMMON_GEOMETRY_SHORTCIRCUIT','QUALIFIED','Proven acceptance-equivalent early rejection; numerical core unchanged; completed results preserved and interrupted searches resume cached arrays.',artifacts=[AMEND],next_stage='RESUME_FULL_COHORT_CONSTRUCTION')

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='prepare':stop_and_freeze()
    else:
        keys=[x['case']['key'] for x in read(construct.CONTRACT)['cases']]
        with concurrent.futures.ProcessPoolExecutor(max_workers=6) as pool:
            fs={pool.submit(run,k):k for k in keys}
            for i,f in enumerate(concurrent.futures.as_completed(fs),1):
                try:print('BOUNDED_CASE_COMPLETE',i,len(keys),f.result(),flush=True)
                except Exception as exc:print('CASE_INFRASTRUCTURE_ERROR',fs[f],repr(exc),flush=True)
