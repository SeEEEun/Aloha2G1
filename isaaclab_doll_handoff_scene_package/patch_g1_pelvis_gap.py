#!/usr/bin/env python3
"""Safely patch only the copied G1 preview/common code from the historical 0.20 m root/forward gap to 0.15 m.
If a unique root-related literal cannot be identified, do not guess: write a report and stop with code 4.
"""
from __future__ import annotations
import argparse, json, re, shutil
from pathlib import Path

NUM_RE=re.compile(r'(?<![\d.])0\.(?:20|2)(?![\d])')
KEYS={"g1":3,"root":4,"offset":3,"forward":2,"base":1,"pelvis":4,"position":1,"pos":1,"translation":1}
BAD={"camera":-4,"light":-4,"speed":-4,"scale":-3,"phone":-3,"charger":-3}

def score(lines,i):
    ctx=' '.join(lines[max(0,i-2):min(len(lines),i+3)]).lower()
    s=sum(v for k,v in KEYS.items() if k in ctx)+sum(v for k,v in BAD.items() if k in ctx)
    return s,ctx

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--scene-dir',type=Path,required=True); ap.add_argument('--gap',type=float,default=0.15)
    args=ap.parse_args(); scene=args.scene_dir.resolve()
    files=[scene/'preview_doll_handoff_g1.py',scene/'robot_model_preview_common.py']
    cands=[]
    for f in files:
        if not f.is_file(): continue
        lines=f.read_text(encoding='utf-8').splitlines(True)
        for i,line in enumerate(lines):
            if NUM_RE.search(line):
                sc,ctx=score(lines,i); cands.append((sc,f,i,line,ctx))
    cands.sort(key=lambda x:x[0],reverse=True)
    report={'requested_gap_m':args.gap,'candidates':[{'score':c[0],'file':str(c[1]),'line':c[2]+1,'text':c[3].strip()} for c in cands]}
    out=scene/'g1_pelvis_gap_patch_report.json'
    if not cands or cands[0][0] < 4 or (len(cands)>1 and cands[1][0]==cands[0][0]):
        report['status']='AMBIGUOUS_NO_PATCH'; out.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2)); return 4
    sc,f,i,line,ctx=cands[0]
    lines=f.read_text(encoding='utf-8').splitlines(True)
    new=NUM_RE.sub(f'{args.gap:.2f}', lines[i], count=1)
    backup=f.with_suffix(f.suffix+'.before_doll_handoff')
    if not backup.exists(): shutil.copy2(f,backup)
    lines[i]=new; f.write_text(''.join(lines),encoding='utf-8')
    report.update({'status':'PATCHED','patched_file':str(f),'patched_line':i+1,'before':line.strip(),'after':new.strip(),'backup':str(backup)})
    out.write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2)); return 0

if __name__=='__main__': raise SystemExit(main())
