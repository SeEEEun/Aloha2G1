#!/usr/bin/env python3
"""Eight-slot independent-case streaming; no change to frozen search budgets."""
from pathlib import Path
import concurrent.futures,json,os,signal,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.final_paper_position_scheduler import run,DEST
from tools.final_paper_io import atomic_json,read,now

def main():
    parent=int(sys.argv[1]);older=int(sys.argv[2]);cmd=Path(f'/proc/{parent}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert 'final_paper_position_scheduler_v2.py' in cmd
    os.kill(parent,signal.SIGSTOP)
    active={}
    for line in subprocess.check_output(['ps','-eo','pid=,ppid=,args='],text=True).splitlines():
        fields=line.split(None,2)
        if len(fields)==3 and int(fields[1])==parent and 'final_paper_position_run.py' in fields[2]:active[fields[2].split()[-1]]=int(fields[0])
    atomic_json(DEST/'STREAMING_SCHEDULER_HANDOFF.json',dict(timestamp=now(),paused_scheduler=parent,older_paused_scheduler=older,active_numerical_jobs=active,
        workers=8,change='Remove the scheduling barrier between independently frozen TRAIN and DEV characterization; retain8 total slots and every per-case numerical parameter/budget',numeric_workers_interrupted=False,seed_bank_immutable=True,dev_outcomes_not_used_for_training_calibration=True))
    def task(row):
        key=row['key']
        if key in active:
            while True:
                path=Path(f'/proc/{active[key]}/stat')
                if not path.exists() or path.read_text().split()[2]=='Z':break
                time.sleep(5)
        return run(row)
    rows=read(DEST/'CASE_MANIFEST.json');results=[];pending=[]
    for row in rows:
        path=DEST/'position'/row['key']/'RESULT.json'
        if path.exists() and read(path)['outcome']!='INFRASTRUCTURE_INVALID':results.append(read(path))
        else:pending.append(row)
    pending.sort(key=lambda row:(row['key'] not in active,rows.index(row)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures={pool.submit(task,row):row for row in pending}
        for future in concurrent.futures.as_completed(futures):
            r=future.result();results.append(r)
            with (DEST/'EPISODE_COMPLETION_LOG_V3.jsonl').open('a') as stream:stream.write(json.dumps(dict(timestamp=now(),key=r['case']['key'],outcome=r['outcome']))+'\n')
            print('STREAMED_COHORT_PROGRESS',len(results),'/150',r['case']['key'],r['outcome'],flush=True)
    assert len(results)==150 and len({r['case']['key'] for r in results})==150
    atomic_json(DEST/'POSITION_COHORT_COMPLETE.json',dict(results=results,all_infrastructure_valid=all(r['outcome']!='INFRASTRUCTURE_INVALID' for r in results)))
    for pid,name in [(parent,'final_paper_position_scheduler_v2.py'),(older,'final_paper_position_scheduler.py')]:
        p=Path(f'/proc/{pid}/cmdline')
        if p.exists():
            assert name in p.read_bytes().replace(b'\0',b' ').decode()
            os.kill(pid,signal.SIGKILL)

if __name__=='__main__':main()
