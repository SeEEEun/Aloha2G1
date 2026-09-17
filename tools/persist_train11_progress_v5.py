#!/usr/bin/env python3
"""Snapshot completed gates without conflating numeric candidates with passes."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_common_train11_position_v5 import *

def main():
    verified_oracle_contract();ids=read(OUT/'00_contract/AUTHORITATIVE_SPLIT_MANIFEST.json')['selection']['qualification_ids']
    passes=sorted(TRAIN.glob('*/SOURCE_POSITION_PASS.json'));rows=[]
    for p in passes:
        r=read(p);rows.append(dict(case=p.parent.name,qualification=file_record(p),trajectory=r['trajectory'],
            raw_acceptance=r['metrics']['raw_acceptance'],correction_mm=r['metrics']['correction_mm']))
    counts={mode:sum(x['case'].startswith(mode+'_') for x in rows) for mode in ('WRIST','INTERACTION')}
    p=ST5/'TRAIN11_PROGRESS_SNAPSHOT.json';atomic_json(p,dict(fixed_ids=ids,qualified=rows,counts=counts,expected_each=11,
        smoke3='A3/3 B3/3 qualified',full_6d='NOT_RUN',later_stages='NOT_RUN'))
    korean=f"SMOKE3는 A3/3·B3/3 통과했습니다. 고정 TRAIN11 전체 위치 검증은 현재 A{counts['WRIST']}/11·B{counts['INTERACTION']}/11입니다. A05와 B05의 실제 충돌도 공통 복구로 해소하고 전체 준비·궤적 검증을 마쳤습니다. 다른 후보의 수치 통과와 전체 형상·분기 통과는 구분해 기록합니다. 원시 목표·시간·물리 한계·충돌 기준은 그대로이며 A/B 공정성을 유지합니다. 나머지 공통 복구와 검증을 계속합니다."
    log('FIXED_TRAIN11_POSITION','IN_PROGRESS',[OUT/'00_contract/AUTHORITATIVE_SPLIT_MANIFEST.json',ST5/'INDEPENDENT_SMOKE_AND_PREPARATION_RESULT.json'],
        'Some TRAIN-side branches require hard-temporal and geometry-aware local restoration',
        'COMMON_NUMERICAL_AND_GEOMETRY_RECOVERY','Run bounded common conic restoration and detailed-boundary repair; independently requalify entire trajectories',
        [p], 'COMPLETE_TRAIN11_POSITION_THEN_FULL_6D',korean)

if __name__=='__main__':main()
