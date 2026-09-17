#!/usr/bin/env python3
"""Read-only final cohort, raw-target and artifact hash verification."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.final_paper_io import *

def main():
    rows=cases();assert len(rows)==150 and len({r['key'] for r in rows})==150
    paper=OUT/'07_paper_artifacts/final';freeze=read(OUT/'03_common_execution_freeze/COMMON_RETARGETING_FREEZE_MANIFEST.json');q0=np.array(freeze['acceptance']['natural_q0']);records={};hands={};six_count=0
    def collect(value):
        if isinstance(value,dict):
            if {'path','sha256','bytes'}<=set(value):
                key=value['path'];rec={k:value[k] for k in ('path','sha256','bytes')}
                if key in records:assert records[key]==rec,'Conflicting artifact identity'
                records[key]=rec
            for v in value.values():collect(v)
        elif isinstance(value,list):
            for v in value:collect(v)
    for row in rows:
        stage={s:read(DEST/s/row['key']/'RESULT.json') for s in ('position','full6d','complete_action')}
        for s,r in stage.items():assert r['case']['key']==row['key'] and r['outcome']!='INFRASTRUCTURE_INVALID';collect(r)
        mode=row['representation_mode']
        with np.load(row['source']['path']) as z:
            target=np.stack([z[f'{mode}_{side}_wrist_position_model'] for side in ('left','right')],axis=1)
            rotation=np.stack([z[f'{mode}_{side}_wrist_rotation_model'] for side in ('left','right')],axis=1);ts=z['source_timestamp'].copy()
        for family in stage['position']['families']:
            with np.load(family['trajectory']['path']) as z:
                assert np.array_equal(z['RAW_REPRESENTATION_TARGET'],target)
                assert np.array_equal(z['source_timestamp'],ts)
                assert np.array_equal(z['full_q'][0],q0)
                assert np.array_equal(z['execution_timestamp'][21:],ts-ts[0]+.7)
        with np.load(stage['position']['selected']['trajectory']['path']) as z:hands[(row['group'],row['index'],mode)]=z['common_hand_q'].copy()
        if stage['full6d'].get('trajectory'):
            six_count+=1
            with np.load(stage['full6d']['trajectory']['path']) as z:
                assert np.array_equal(z['RAW_REPRESENTATION_TARGET'],target) and np.array_equal(z['RAW_ORIENTATION_TARGET'],rotation)
                assert np.array_equal(z['full_q'][0],q0)
    for row in rows:
        if row['representation_mode']=='WRIST':assert np.array_equal(hands[(row['group'],row['index'],'WRIST')],hands[(row['group'],row['index'],'INTERACTION')])
    reference=read(DEST/'REFERENCE_PHYSICS_COMPLETE.json');assert len(reference['results'])==70 and len({r['case']['key'] for r in reference['results']})==70 and reference['all_infrastructure_valid']
    collect(reference)
    for rec in records.values():assert file_record(Path(rec['path']))==rec,rec['path']
    old=DEST/'provenance/paper_layout_v1/FINAL_NUMERICAL_RESULTS.json'
    assert old.read_bytes()==(paper/'FINAL_NUMERICAL_RESULTS.json').read_bytes(),'Presentation edits changed numerical results'
    atomic_json(paper/'FINAL_COHORT_INTEGRITY_AUDIT.json',dict(status='PASS',cases=150,matched_pairs=75,attempted_full6d=six_count,reference_outcomes=70,checked_file_records=len(records),raw_position_arrays_exact=True,raw_orientation_arrays_exact=True,natural_q0_exact=True,source_timestamps_exact=True,preparation_offset_seconds=.7,common_hand_arrays_exact_ab=True,numerical_results_byte_identical_after_layout_changes=True,implementation=file_record(Path(__file__)),records=list(records.values())))
    print('FINAL_COHORT_INTEGRITY_PASS',len(records),'artifact hashes;',six_count,'6D attempts; targets unchanged')

if __name__=='__main__':main()
