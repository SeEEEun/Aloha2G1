#!/usr/bin/env python3
"""Persist the primary agent's completed visual review and refresh provenance."""
from pathlib import Path
import sys,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.final_paper_io import *

def main():
    paper=OUT/'07_paper_artifacts/final'
    names=['Fig_Main_SingleVariable_AB.png','Fig_Raw_Cartesian_Feasibility_AB.png','Fig_Morphology_Correction_Distribution.png','Fig_DEV35_First_Failure_Stage.png','Fig_Matched_DEV35_Matrix.png','Fig_Representative_Physical_Rollouts.png','Fig_TRAIN_Frozen_Solver_Bottleneck.png','verification/DECODED_VIDEO_FIRST_MIDDLE_LAST.png','replays/A_top_RETARGETING_FAILURE_CARDS.png']
    assert read(paper/'MEDIA_DECODE_VERIFICATION.json')['status']=='PASS' and read(paper/'FINAL_COHORT_INTEGRITY_AUDIT.json')['status']=='PASS'
    atomic_json(paper/'VISUAL_REVIEW.json',dict(status='PASS',reviewed_at=now(),reviewer='Primary Codex agent; direct image inspection during this run',artifacts=[file_record(paper/n) for n in names],
        observations=['Main four-panel figure is readable and explicitly labels DEV35 development evaluation.','Zero physical trials and upstream feasibility failure are explicitly stated; zero cumulative counts are not described as observed grasp failures.','The absence of a qualified A correction distribution is explicitly labeled, not replaced with zero values.','All35 matched cells and exact first-failure categories are preserved; full IDs are available in the supplementary matrix and CSV.','Actual encoded MP4 first/middle/last samples of all four videos were inspected. They are labeled static failure cards, with no fabricated robot/object motion.','Representative selection is objective; absent success categories are not fabricated.','TRAIN diagnostic plots are labeled kinematic diagnostics, not physical evidence.'],
        limitation='Video review sampled first, middle and last decoded frames; these are intentional static failure-card videos. No DEV35 physical trajectory exists.',
        numerical_integrity=file_record(paper/'FINAL_COHORT_INTEGRITY_AUDIT.json'),media_integrity=file_record(paper/'MEDIA_DECODE_VERIFICATION.json')))
    archive=DEST/'provenance/pre_visual_final_report';archive.mkdir(parents=True,exist_ok=True)
    for p in (OUT/'FINAL_PAPER_READY_SINGLE_VARIABLE_AB_REPORT.md',DEST/'FINAL_ARTIFACT_VERIFICATION.json'):
        if p.exists():shutil.copy2(p,archive/p.name)
    # Refresh only the artifact inventory; no analysis or simulation is repeated.
    manifest=read(paper/'ANALYSIS_MANIFEST.json')
    manifest['artifacts']=[file_record(p) for p in sorted(paper.glob('*')) if p.is_file() and p.name!='ANALYSIS_MANIFEST.json']
    atomic_json(paper/'ANALYSIS_MANIFEST.json',manifest)
    print('VISUAL_REVIEW_PERSISTED; numerical results unchanged')

if __name__=='__main__':main()
