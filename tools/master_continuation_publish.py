#!/usr/bin/env python3
"""Publish only valid diagnostics and an explicit blocked-stage handoff."""
from pathlib import Path
import sys
import shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.master_continuation_preflight import MASTER,verified_oracle_contract,log
from tools.final_single_variable_prepare import OUT,read,file_record,status
from tools.run_reference_motion_scientific_reset import atomic_json,atomic_text


def main():
    verified_oracle_contract()
    result=read(MASTER/'MASTER_RESULT.json');assert result['status']=='MASTER_SINGLE_VARIABLE_AB_SCIENTIFIC_BLOCKER'
    paper=OUT/'07_paper_artifacts';paper.mkdir(parents=True,exist_ok=True)
    table=paper/'TABLE_MASTER_POSITION_BOUNDARY_DIAGNOSTIC.md'
    atomic_text(table,(MASTER/'INITIAL_BOUNDARY_TABLE.md').read_text())
    figures=[]
    for suffix in ('png','pdf','svg'):
        source=MASTER/f'INITIAL_BOUNDARY_DIAGNOSTIC.{suffix}'
        destination=paper/f'FigXX_Master_Position_Boundary_Diagnostic.{suffix}'
        if destination.exists():assert file_record(destination)['sha256']==file_record(source)['sha256']
        else:shutil.copyfile(source,destination)
        figures.append(destination)
    terminal='''MASTER SINGLE-VARIABLE A/B CONTINUATION

==================================================
CARTESIAN FORENSIC
Root cause: MIXED_SOLVER_AND_TARGET_FEASIBILITY
A raw reachable: 1772 / 2066 known framewise witnesses (TRAIN smoke)
A certified unreachable: 290 / 2066; 4 additional uncertified no-witness frames
B raw reachable: 686 / 686 known framewise witnesses (B49 only)
B certified unreachable: 0 / 686 (B49 only)

==================================================
COMMON EXECUTABLE POSITION
A complete executable trajectories: 0 / 3 qualify the new fixed-boundary gate
B complete executable trajectories: 0 / 3 qualify the new fixed-boundary gate
Scope: necessary-condition audit, not six new optimized rollouts.
Blocker: fixed natural q(0) produces A 56.18-56.45 mm and B 67.40-67.48 mm wrist error on proven-reachable frame-zero targets; the new requirement is <=10 mm on every reachable frame.
A correction mean / p95 / max: NOT APPLIED; no qualified closest-feasible trajectory
B correction mean / p95 / max: NOT APPLIED
Hard collision: full new trajectories NOT QUALIFIED
Hard-limit violations: full new trajectories NOT QUALIFIED
Branch discontinuities: full new trajectories NOT QUALIFIED
No target, initial state, event clock or collision threshold was changed.

==================================================
FULL 6D
A: NOT RUN
B: NOT RUN
Orientation residual: NOT AVAILABLE

==================================================
LOADED DEX3
Mapping: NOT RUN
Sign: NOT RUN
Readback: NOT RUN
Runtime limits: NOT RUN
Measured violations: NOT MEASURED

==================================================
DATASET / TRAINING
A actions changed: NOT AUDITED
B actions changed: NOT AUDITED
Datasets regenerated: NO
ACT retrained: NO
ACT-A checkpoint: NOT RESELECTED / NOT APPROVED FOR THIS REBUILD
ACT-B checkpoint: NOT RESELECTED / NOT APPROVED FOR THIS REBUILD

==================================================
DEV35 PHYSICAL RESULT
ACT-A valid: NOT RUN (0 / 35 executed)
ACT-B valid: NOT RUN (0 / 35 executed)
ACT-A full task: NOT AVAILABLE
ACT-B full task: NOT AVAILABLE
B - A: NOT AVAILABLE

==================================================
MAIN FIGURE: NOT GENERATED (no DEV35 results)
RESULT TABLE: NOT GENERATED (no DEV35 results)
TRAIN DIAGNOSTIC FIGURE: outputs/final_single_variable_ab/07_paper_artifacts/FigXX_Master_Position_Boundary_Diagnostic.png
TRAIN DIAGNOSTIC TABLE: outputs/final_single_variable_ab/07_paper_artifacts/TABLE_MASTER_POSITION_BOUNDARY_DIAGNOSTIC.md
A TOP: NOT GENERATED
A OVERVIEW: NOT GENERATED
B TOP: NOT GENERATED
B OVERVIEW: NOT GENERATED
FINAL REPORT: outputs/final_single_variable_ab/FINAL_SINGLE_VARIABLE_AB_REBUILD_AND_EVALUATION_REPORT.md
CHATGPT UPDATE: outputs/final_single_variable_ab/master_continuation/CHATGPT_UPDATE.md

==================================================
MASTER_SINGLE_VARIABLE_AB_SCIENTIFIC_BLOCKER
'''
    atomic_text(MASTER/'FINAL_TERMINAL_OUTPUT.txt',terminal)
    log('FINAL_DIAGNOSTIC_PUBLICATION','MASTER_SINGLE_VARIABLE_AB_SCIENTIFIC_BLOCKER',
        [MASTER/'INITIAL_BOUNDARY_PROOF.json',MASTER/'MASTER_RESULT.json'],
        'Common startup boundary must be reconciled with the every-frame reachable-target rule.',
        'INITIAL_BOUNDARY_CONTRACT_CONFLICT',
        'Published valid TRAIN diagnostics only; preserved old report provenance; no fabricated DEV35 results or loaded-articulation passes.',
        [table,*figures,MASTER/'FINAL_TERMINAL_OUTPUT.txt'],
        'REQUIRES_EXPLICIT_COMMON_STARTUP_BOUNDARY_CONVENTION_BEFORE_POSITION_SOLVING')
    status('MASTER_SINGLE_VARIABLE_AB_SCIENTIFIC_BLOCKER','REQUIRES_COMMON_STARTUP_BOUNDARY_CONVENTION',
        [MASTER/'CURRENT_STAGE.md',MASTER/'MASTER_RESULT.json',MASTER/'CHATGPT_UPDATE.md',
         MASTER/'BLOCKERS/INITIAL_BOUNDARY_CONTRACT_CONFLICT.md',MASTER/'INITIAL_BOUNDARY_PROOF.json',
         OUT/'FINAL_SINGLE_VARIABLE_AB_REBUILD_AND_EVALUATION_REPORT.md',OUT/'READY_FOR_UNTOUCHED_FINAL_TEST.md'])
    files=[p for p in MASTER.rglob('*') if p.is_file() and p.name!='MASTER_ARTIFACT_MANIFEST.json']
    files +=[table,*figures,OUT/'FINAL_SINGLE_VARIABLE_AB_REBUILD_AND_EVALUATION_REPORT.md',OUT/'READY_FOR_UNTOUCHED_FINAL_TEST.md',OUT/'CURRENT_STATUS.md',OUT/'CURRENT_STATUS.json']
    atomic_json(MASTER/'MASTER_ARTIFACT_MANIFEST.json',{'artifacts':[file_record(p) for p in sorted(files)],
        'implementations':[file_record(Path(__file__)),file_record(ROOT/'tools/master_continuation_preflight.py')]})
    print(terminal,flush=True)


if __name__=='__main__':main()
