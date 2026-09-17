#!/usr/bin/env python3
"""Concurrency-only handoff; never interrupt/duplicate an active numerical job."""
from pathlib import Path
import concurrent.futures,json,os,signal,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.final_paper_position_scheduler import run,DEST

def main():
    parent=int(sys.argv[1])
    # Pause only our owned scheduling process; its solver children keep running.
    cmd=Path(f'/proc/{parent}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert 'tools/final_paper_position_scheduler.py' in cmd
    os.kill(parent,signal.SIGSTOP)
    active={}
    for line in subprocess.check_output(['ps','-eo','pid=,ppid=,args='],text=True).splitlines():
        fields=line.split(None,2)
        if len(fields)==3 and int(fields[1])==parent and 'final_paper_position_run.py' in fields[2]:active[fields[2].split()[-1]]=int(fields[0])
    (DEST/'SCHEDULER_CONCURRENCY_HANDOFF.json').write_text(json.dumps(dict(timestamp=time.time(),old_scheduler_pid=parent,active_numerical_jobs=active,old_workers=4,new_workers=8,solver_budget_changed=False,solver_processes_interrupted=False),indent=2))
    def task(row):
        key=row['key']
        if key in active:
            while True:
                stat=Path(f'/proc/{active[key]}/stat')
                if not stat.exists() or stat.read_text().split()[2]=='Z':break
                time.sleep(5)
        return run(row)
    rows=json.loads((DEST/'CASE_MANIFEST.json').read_text());results=[]
    for group in ('TRAIN40','DEV35'):
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures={pool.submit(task,r):r for r in rows if r['group']==group}
            for future in concurrent.futures.as_completed(futures):
                r=future.result();results.append(r)
                with (DEST/'EPISODE_COMPLETION_LOG_V2.jsonl').open('a') as log:log.write(json.dumps(dict(timestamp=time.time(),key=r['case']['key'],outcome=r['outcome']))+'\n')
                print('COHORT_PROGRESS',len(results),'/150',r['case']['key'],r['outcome'],flush=True)
    (DEST/'POSITION_COHORT_COMPLETE.json').write_text(json.dumps(dict(results=results,all_infrastructure_valid=all(r['outcome']!='INFRASTRUCTURE_INVALID' for r in results)),indent=2))
    # All inherited children are done and their atomic artifacts ingested. Only
    # the obsolete paused scheduler is retired; no numerical process is killed.
    os.kill(parent,signal.SIGKILL)

if __name__=='__main__':main()
