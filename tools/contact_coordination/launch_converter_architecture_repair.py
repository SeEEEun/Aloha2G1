"""Detach the locked repair supervisor from an interactive tool session."""
import argparse
import os
from pathlib import Path
import subprocess
import time
from .io import ROOT,OFFLINE,atomic_json,atomic_text


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True)
    a=p.parse_args();out=a.run_dir.resolve()
    args=[OFFLINE,'-B','-m','tools.contact_coordination.run_converter_architecture_repair','--run-dir',str(out),'--resume']
    with (out/'SUPERVISOR.log').open('a') as stream:
        worker=subprocess.Popen(args,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
            start_new_session=True,close_fds=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
    time.sleep(.5)
    atomic_json(out/'LAUNCHER.json',dict(pid=worker.pid,command=args,detached_session=True,initial_returncode=worker.poll()))
    print('Persistent repair supervisor PID',worker.pid,'initial returncode',worker.poll(),flush=True)


if __name__=='__main__':main()
