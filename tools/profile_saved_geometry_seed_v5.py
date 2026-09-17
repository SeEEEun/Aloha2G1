#!/usr/bin/env python3
"""Profile one persisted TRAIN seed; no solver, target or acceptance mutation."""
from pathlib import Path
import sys,cProfile,pstats,io,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_common_train11_position_v5 import *
from tools.common_fast_hard_witness_v5 import hard_witness

def main(case,path,frame):
    verified_oracle_contract();g,c,n=model();t,h,ts,source=load_input(case,g,n)
    path=Path(path);q=np.load(path)['q'][frame];assert np.isfinite(q).all()
    folder=ST5/'geometry_profiling'/case/f'FRAME_{frame}'
    for label,fn in [('early_rejection',lambda:hard_witness(c,q,h[frame])),('full_original_classifier',lambda:c.inspect(q,*h[frame]))]:
        profiler=cProfile.Profile();start=time.monotonic();profiler.enable()
        try:result=fn()
        except Exception as exc:result=dict(error=str(exc))
        profiler.disable();runtime=time.monotonic()-start;stream=io.StringIO();pstats.Stats(profiler,stream=stream).sort_stats('cumulative').print_stats(24)
        atomic_text(folder/(label+'.txt'),stream.getvalue());atomic_json(folder/(label+'.json'),dict(runtime_s=runtime,result=result,input=file_record(path),frame=frame,source=source))
        print(label,runtime,stream.getvalue(),flush=True)

if __name__=='__main__':main(sys.argv[1],sys.argv[2],int(sys.argv[3]))
