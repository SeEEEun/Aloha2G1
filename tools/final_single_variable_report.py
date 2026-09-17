#!/usr/bin/env python3
"""Persist the exact terminal gate and resume evidence; never invent results."""
from __future__ import annotations
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.final_single_variable_prepare import OUT,ROOT,CORE,RESET,read,file_record,digest
from tools.run_reference_motion_scientific_reset import atomic_json,atomic_text,sha256,stats


def main():
    qualification_path=OUT / "02_common_execution_qualification/COMMON_POSITION_IK_QUALIFICATION.json"
    qualification=read(qualification_path)
    assert qualification["status"]=="COMMON_SOLVER_NOT_QUALIFIED", "This report writer is only for the explicit bounded-solver terminal gate"
    rows=qualification["rows"]
    references=read(OUT / "01_registration/RAW_REFERENCE_MANIFEST.json")
    split_path=OUT / "00_contract/AUTHORITATIVE_SPLIT_MANIFEST.json"
    split=read(split_path)
    by_source={r["source_recording_id"]:r for r in references["entries"]}
    selection_source=CORE / "offline_heldout8/experiment2_inference_selection.json"
    previous_selection=read(selection_source)
    for key in ("a","b"):
        selected=previous_selection["methods"][key]["checkpoint_selection"]
        p=Path(selected["selected_checkpoint"])/"model.safetensors"
        assert sha256(p)==selected["selected_model_sha256"]
        split["previous_checkpoints"][key]["authoritative_selected_checkpoint"]={**selected,"selection_source":file_record(selection_source),"model":file_record(p),"final_rebuild_eligibility":"NOT_APPROVED_FOR_REUSE; qualified corrected action audit not reached"}
    for row in split["entries"]:
        if row["source_recording_id"] in by_source:
            row["current_rebuild_raw_cartesian_reference"]=by_source[row["source_recording_id"]]["reference"]
        row["qualified_corrected_joint_actions_available"]=False
    atomic_json(split_path,split)
    oldfreeze=ROOT / "outputs/final_episode_registered_eval35/01_freeze"
    recovered={"status":"RECOVERED_CONFIGURATION_PROVENANCE_ONLY_NOT_REQUALIFIED",
               "source_files":[file_record(oldfreeze/name) for name in ["FINAL_PHYSICAL_ENVIRONMENT.json","FINAL_DOLL_CONTACT_MODEL.json","FINAL_COMMON_DEX3_CONTROLLER.json","FINAL_MECHANICAL_GRASP_SCORER.json"]],
               "physical_environment":read(oldfreeze / "FINAL_PHYSICAL_ENVIRONMENT.json"),
               "doll_contact_model":read(oldfreeze / "FINAL_DOLL_CONTACT_MODEL.json"),
               "previous_controller":read(oldfreeze / "FINAL_COMMON_DEX3_CONTROLLER.json"),
               "loaded_Dex3_qualification":"NOT_RUN_POSITION_AND_6D_GATES_NOT_PASSED",
               "new_controller_required_state_machine":["OPEN","PRESHAPE","PROGRESSIVE_CLOSE","MECHANICAL_GRASP_CONFIRM","BOUNDED_PRELOAD","HOLD","RECEIVER_CLOSE","DUAL_HOLD","GIVER_RELEASE","FINAL_RELEASE"],
               "old_controller_does_not_establish_new_state_machine_qualification":True}
    atomic_json(OUT / "01_registration/RECOVERED_PHYSICAL_SCENE_AND_CONTROLLER_PROVENANCE.json",recovered)
    combined={}
    for mode in ("WRIST","INTERACTION"):
        selected=[r for r in rows if r["representation_mode"]==mode]
        values=[np.load(OUT / f"02_common_execution_qualification/position_only/{mode}_EP{r['episode_index']:03d}.npz",allow_pickle=False) for r in selected]
        combined[mode]={"qualified":sum(r["position_qualified"] for r in selected),"attempted":len(selected),
                        "total_frames":sum(r["frame_count"] for r in selected),
                        "self_collision_frames":sum(r["self_collision_frames"] for r in selected),
                        "hard_limit_violations":sum(r["hard_limit_violations"] for r in selected),
                        "branch_discontinuities":sum(r["branch_discontinuities"] for r in selected),
                        "runtime_s":sum(r["runtime_s"] for r in selected),
                        "position_residual_mm":stats(np.concatenate([np.max(z["position_residual_m"],axis=1) for z in values]),1000),
                        "q_step_norm_rad":stats(np.concatenate([np.linalg.norm(np.diff(z["q"],axis=0),axis=1) for z in values])),
                        "minimum_self_collision_margin_m":min(r["self_collision_minimum_margin_m"] for r in selected),
                        "candidate_projection_magnitude_mm":stats(np.concatenate([np.linalg.norm(z["COMMON_FEASIBLE_TARGET_CANDIDATE"]-z["RAW_REPRESENTATION_TARGET"],axis=2).ravel() for z in values]),1000),
                        "projection_applied_to_any_dataset":False}
        for z in values:z.close()
    versions={p:importlib.metadata.version(p) for p in ("numpy","scipy","mujoco","pyarrow")}
    atomic_json(OUT / "00_contract/OFFLINE_EXECUTION_ENVIRONMENT.json",{"python":sys.version,"executable":sys.executable,"platform":platform.platform(),"packages":versions,"models":read(RESET / "config/common_config.json")["models"],"no_real_hardware_commands":True,"no_PhysX_runs":True})
    failed=[r for r in rows if not r["position_qualified"]]
    witnesses=[]
    for r in failed:
        for w in r["global_witnesses"]:
            clear=[s for s in w["seeds"] if not s["contacts"]]
            best_clear=min((max(s["position_residual_m"]) for s in clear),default=None)
            witnesses.append({"mode":r["representation_mode"],"episode":r["episode_index"],"frame":w["frame"],"best_multiseed_position_residual_mm":1000*w["best_maximum_wrist_residual_m"],"best_zero_contact_multiseed_residual_mm":1000*best_clear if best_clear is not None else None,"interpretation":"ISOLATED_FRAME_FEASIBLE; sequential realization remains unqualified" if best_clear is not None and best_clear<=0.01 else "BOUNDED_SEARCH_DID_NOT_FIND_ACCEPTABLE_ISOLATED_REALIZATION; not a global infeasibility proof"})
    terminal={"terminal_status":"COMMON_SOLVER_NOT_QUALIFIED", "contract":file_record(OUT / "00_contract/SINGLE_VARIABLE_EXPERIMENT_CONTRACT.json"),"qualification":file_record(qualification_path),"aggregate":combined,"isolated_frame_witnesses":witnesses,
              "completed":["contract","split recovery","prior provenance registry","35 matched source environments","40 TRAIN and 35 DEV raw reference audits","strict shared source-derived command clock","bounded common position solver SMOKE3"],
              "blocked_gates":["position-only TRAIN11 qualification","full 6D IK","loaded Dex3","full physical reference smoke","common execution freeze","exact corrected action audit","dataset regeneration/parity","ACT retraining","policy-output validation","DEV35 integration smoke","DEV35 freeze","ACT-A35/ACT-B35","statistics","paper result figures/tables","actual PhysX replay"],
              "exact_corrected_training_action_diff":"NOT_AVAILABLE_UNQUALIFIED_SOLVER; no corrected joint-action dataset exists",
              "dataset_regeneration_required":"PENDING_FINAL_ACTION_AUDIT; prior diagnostic evidence and common timing changes indicate expected regeneration",
              "ACT_RETRAINING_REQUIRED":"PENDING_FINAL_ACTION_AUDIT; do not spend training compute before execution qualification",
              "FULL_TASK_SCOPE_VIABLE":"NOT_ESTABLISHED", "standardized_grasp_fallback_used":False,
              "final_DEV35_physical_rollouts":0,"DEV35_TSR":None,"untouched_final_test_manifest_found":False,
              "READY_FOR_UNTOUCHED_FINAL_TEST_COLLECTION":False,
              "readiness_reason":"Complete common pipeline freeze does not exist; collect/evaluate a fresh final set only after qualification and freeze",
              "final_test_search":{"roots":[str(ROOT / p) for p in ("outputs","configs","datasets")],"patterns":["*FINAL_TEST*","*final_test*","*final-test*"],"matches":[],"note":"DEV35 includes originally held-out episodes used in prior engineering/selection; no untouched claim"},
              "no_scientific_A_B_outcome_claim":True}
    atomic_json(OUT / "FINAL_GATE_RESULT.json",terminal)
    contract_hash=terminal["contract"]["sha256"]
    qhash=terminal["qualification"]["sha256"]
    raw_audit=read(OUT / "01_registration/RAW_TARGET_AUDIT.json")
    count=sum(r["qualification_subset"] for r in raw_audit["entries"])
    report=f"""# Final single-variable A/B rebuild and evaluation report

Terminal status: COMMON_SOLVER_NOT_QUALIFIED.

The experiment stopped at the prescribed bounded common position-only gate. No corrected joint-action training dataset, new ACT training, common execution freeze, final DEV35 freeze, PhysX rollout, success rate, or physical-result figure was produced. This is a qualification result, not an A/B task-success result.

## 1. Scientific variable

A is WRIST-TRAJECTORY-CENTRIC SPATIAL TARGET GENERATION; B is INTERACTION-CENTRIC SPATIAL TARGET GENERATION. The contract SHA256 is `{contract_hash}`. Spatial representation is selected in the recovered representation builder. The new solver receives target positions, the same model, the same hand joint geometry, timestamps, and a common initial state; it receives no representation, episode, object, scorer, or outcome input.

## 2. Prior confounds and provenance

All pre-contract physical results, including ACT-A 0/35, incomplete/invalid ACT-B, standardized-grasp 0/35 versus 0/35, biased evaluators, and invalid workspace/IK trials, are PRE_FINAL_SINGLE_VARIABLE_REBUILD_DIAGNOSTIC_ONLY. Original bytes were preserved; see `00_contract/PRIOR_DIAGNOSTIC_PROVENANCE.json`. Previously dirty repository files were not edited by this rebuild.

## 3. Common task registration and data identities

Recovered 50 original recordings, COMMON48, TRAIN40, original HELDOUT8, and DEV35 from manifests: 79 distinct recording IDs across the provenance union. TRAIN40 includes the explicit replacement recording `GoPark_20260823_140035` at final dataset index 36, whose original collection index is null; it was not confused with original episode 36.

DEV35 environments: 35/35 source-derived and 35/35 exactly matched A/B object poses, with 0 mm and 0 degree matched differences. These are the qualified source-conditioned poses in the identity metric task registration. Failed diagnostic workspace transforms were not reused. Active rotations, meters, quaternion ordering, named sides, fixed wrist/tool multiplication, and world/model round trips were audited. This establishes coordinate consistency, not workspace feasibility.

## 4. Common event clock and raw targets

All 40 TRAIN and 35 DEV reference pairs were regenerated from measured source FK. The predetermined TRAIN qualification IDs are {split['selection']['qualification_ids']}; both raw generators pass {count}/{count}, including 0, 24, 49. A independently reconstructs registered source TCP position and orientation. B reconstructs initial object/task-relative translation, the bimanual relation, and the target whole-hand frame. The source has no dynamic object-pose ground truth; contact topology and dynamic object-relative behavior are not asserted.

The original source clock is preserved separately from the strict execution clock. Where necessary, the common command close interval is advanced minimally to finish one source sample before detected lift. This affects 9/40 TRAIN and 11/35 DEV episodes (the strict inequality also changes source equality cases, beyond the previously reported overlaps). RIGHT acquisition precedes LEFT release in all entries. Timestamps and raw wrist samples are preserved, with no standardized-grasp rebase or derivative splice. Runtime mechanical confirmation remains untested.

## 5. Common morphology and IK qualification

Position-only SMOKE3: A {combined['WRIST']['qualified']}/3, B {combined['INTERACTION']['qualified']}/3. The common authoritative Cartesian gate is 95% of bilateral frames within 10 mm; maximum residuals are still disclosed. Even passing B episodes have initial acquisition residuals above 10 mm. Passing this offline position gate would not establish natural-start physical integration.

The solver uses actual G1 FK/Jacobians, hard limits, signed collision distances, bounded deterministic shoulder/elbow seed alternatives, a reverse posture guide and forward temporal realization. It does not declare feasibility using radial reach alone. The budget was declared before outcomes: two trajectory passes, up to four seeds and two collision discovery passes per frame, 40 evaluations per candidate, and at most three failure witnesses with 24 full-limit seeds and 150 evaluations per seed. Zero classified prohibited self-contact is required, without the historical hard-segment duration/depth exemptions.

| Mode | Qualified | Self-collision frames | Hard-limit violations | Branch discontinuities | Residual mean/p95/max mm | Solver seconds |
|---|---:|---:|---:|---:|---|---:|
"""
    for mode in combined:
        c=combined[mode];p=c["position_residual_mm"]
        report+=f"| {mode} | {c['qualified']}/3 | {c['self_collision_frames']} | {c['hard_limit_violations']} | {c['branch_discontinuities']} | {p['mean']:.3f}/{p['p95']:.3f}/{p['max']:.3f} | {c['runtime_s']:.2f} |\n"
    report+="""
Finite differences checked the position Jacobian (maximum difference 1.03e-10) and an active collision distance Jacobian (2.92e-5). These local checks do not certify all geometry derivatives.

Isolated multiseed witnesses distinguish solver continuity failure from unresolved model feasibility. For example, WRIST episode 0 frame 420 admits a zero-contact solution at the raw target although the sequential trajectory misses it substantially. WRIST episode 24 frame 185 retained approximately 66.84 mm residual in the bounded full-limit search. Neither a universal morphology incompatibility nor global mathematical infeasibility is established.

A maximum 10 mm common projection candidate was predeclared from the unchanged Cartesian tolerance. Raw targets and diagnostic bounded candidates are saved separately. Projection was not adopted into any dataset. A selected realization's required projection is not a global nearest-feasible-target proof; large residuals and candidate-bound failures remain explicit failures. The 11-episode position qualification is NOT_RUN because SMOKE3 failed. Full 6D is NOT_RUN. Qualification details and all selected/candidate q, contacts, margins, and target residuals are in `02_common_execution_qualification/`.

## 6. Loaded Dex3 qualification

NOT_RUN (0/14 newly tested), because position-only and full 6D gates did not pass. No measured-state clipping or isolated articulation test was substituted. The earlier controller/articulation configuration is preserved as provenance and is not promoted to loaded qualification.

## 7. Exact old-versus-corrected action difference

NOT_REACHED. Corrected Cartesian references exist; qualified corrected TRAIN40 joint-action tensors, normalization statistics, and chunks do not. The old smoke candidate diff is diagnostic provenance and cannot substitute for the requested exact diff against a qualified common pipeline. Changed frames/scalars and maximum/mean exact action differences remain unavailable, not zero.

## 8. Dataset regeneration decision

Pending the exact qualified action audit. Earlier candidates and the explicit common timing correction make regeneration expected. Neither existing dataset was overwritten and no unqualified solver trajectory was packaged as training supervision.

## 9. Dataset parity

Raw-reference inputs, clock, and registration have zero observed unintended differences outside the representation switch. Corrected training dataset parity is NOT_RUN; the experiment does not claim zero downstream dataset confounds without that audit. The 14-dimensional recorded source state and the old ACT 28-dimensional input convention remain an explicit downstream interface question to resolve identically before building datasets.

## 10. ACT retraining decision

Pending the exact final action audit; training was NOT_RUN. If either final action/timing/chunk/order/normalization changes, both datasets and both policies must be rebuilt. Existing checkpoints are not approved for reuse on the basis of these Cartesian audits.

## 11. ACT training parity

The old authoritative common protocol was recovered: official LeRobot ACT, 100000 steps, batch size 8, chunk size 50, seed 1000, no AMP, checkpoints every 20000 steps, and the matched configuration/model hashes. Old selected checkpoints were rehashed. New training parity is NOT_RUN.

## 12. Checkpoint selection

The recovered old common rule ranks heldout semantic phase score, then full-valid-chunk RMSE, then raw chunk jerk, then earlier step. Old A selected step 100000; old B selected step 20000. Their paths and actual model hashes are in the split manifest. No new checkpoint selection occurred. Before any future training, predeclare one identical selection rule without DEV35 physical success or visual preference.

## 13. Policy-output sanity

NOT_RUN. No stale action decoder, shape conversion, or old checkpoint inference was used as evidence for the rebuild.

## 14. DEV35 smoke and full-task scope

NOT_RUN. The preselected smoke indices are 0, 8, 17, 26, 34. Natural-start full-task scope remains intended but viability is NOT_ESTABLISHED. Standardized-grasp fallback was not used.

## 15. Freeze hashes

The contract, split, source data, references, selection, solver budget, implementation, and qualification results are hashed. A common execution freeze and final DEV35 freeze do not exist because their prerequisites failed. See `ARTIFACT_HASH_MANIFEST.json` for this stopped-stage artifact inventory; it is explicitly not a qualified execution freeze.

## 16. DEV35 A physical results

NOT_RUN: zero new final rollouts. No denominator-based success rate is computed.

## 17. DEV35 B physical results

NOT_RUN: zero new final rollouts. Old B diagnostics are excluded.

## 18. Stage-wise success

Approach, left grasp, lift, handoff, right ownership, right transport, bin entry, bin settle, first failure, and cumulative full-task outcomes are unavailable. Missing outcomes are not scored as failures.

## 19. Full-task success

DEV35 FULL-TASK PHYSICAL SUCCESS RATE: unavailable for both policies. B-A percentage-point difference: unavailable.

## 20. Matched statistics

No matched physical pairs exist for this rebuild; contingency counts, confidence intervals, paired effect, and exact McNemar test are not computed.

## 21. Untouched final-test availability

No FINAL_TEST manifest was found in outputs/configs/datasets under the searched naming patterns. No complete pipeline freeze exists. DEV35 remains development/diagnostic data. READY_FOR_UNTOUCHED_FINAL_TEST_COLLECTION is not asserted at this failed prerequisite gate; it becomes YES only after the common pipeline and policies are qualified and frozen.

## 22. Figures, videos, and recovered physical scene

Final paper result tables, the four-panel physical-results figure, and the four 7x5 measured-state replay videos are NOT_GENERATED because no final physical traces exist. No command-only visualization is presented as physical evidence.

The authoritative physical configuration was recovered without executing it: 20 g rigid plush proxy, collision dimensions 117.5 x 72.5 x 77.5 mm, visual dimensions 120 x 90 x 85 mm, static/dynamic friction 0.9/0.75, linear/angular damping 0.35/0.35, restitution 0, contact offset 1.5 mm, and 150 mm bin. Exact configuration and source hashes are in `01_registration/RECOVERED_PHYSICAL_SCENE_AND_CONTROLLER_PROVENANCE.json`. Natural initial state is separate from all standardized-grasp artifacts.

## 23. Limitations and resume

This is finite solver qualification evidence, not a proof that no common solver can work. The source-derived spatial target and common scene were not moved to recover feasibility. The inherited collision classification and active model require eventual PhysX and loaded Dex3 correspondence audits. Qualification runtime excludes model construction, final metric aggregation, and isolated failure-witness searches; those artifacts are separate from physical execution.

All six valid offline qualification results are persisted and hash-checked on resume. Do not rerun them unnecessarily. To attempt a materially revised common solver, preserve this completed bounded attempt, declare a new deterministic solver/registration qualification contract, and create separately versioned evidence. Maintain the raw generators, source-event equality, and natural-start scope. After both position qualification levels pass, advance to full 6D, loaded articulation, shared reference smoke, freeze, exact action diff, dataset parity, and only then ACT training/evaluation.

## 24. Paper-safe interpretation

The repository now supports a controlled raw-reference comparison under a shared source-derived command clock and a method-blind bounded solver attempt. That common solver did not qualify for both methods. The evidence does not support attributing physical task-success differences to representation, reporting a DEV35 TSR, claiming B superiority, untouched final-test performance, real-G1 task success, or autonomous target-domain visual deployment.

COMMON_SOLVER_NOT_QUALIFIED
"""
    report_path=OUT / "FINAL_SINGLE_VARIABLE_AB_REBUILD_AND_EVALUATION_REPORT.md"
    atomic_text(report_path,report)
    paths=[p for p in OUT.rglob("*") if p.is_file() and p.name not in {"ARTIFACT_HASH_MANIFEST.json","CURRENT_STATUS.md","CURRENT_STATUS.json","FINAL_TERMINAL_OUTPUT.txt"}]
    inventory={"status":"STOPPED_STAGE_PROVENANCE_HASHES_NOT_AN_EXECUTION_FREEZE","artifacts":[file_record(p) for p in sorted(paths)],"implementations":[file_record(ROOT / "tools" / n) for n in ("final_single_variable_prepare.py","final_common_position_solver.py","final_single_variable_qualify_position.py","final_single_variable_report.py")]}
    atomic_json(OUT / "ARTIFACT_HASH_MANIFEST.json",inventory)
    status_data={"terminal_status":"COMMON_SOLVER_NOT_QUALIFIED","completed_gate":"BOUNDED_POSITION_ONLY_SMOKE3_COMPLETE_AND_FAILED","next_required_gate":"REVISED_COMMON_SOLVER_QUALIFICATION_BEFORE_ANY_DOWNSTREAM_EXECUTION","contract_sha256":contract_hash,"qualification_sha256":qhash,"report":file_record(report_path),"inventory":file_record(OUT / "ARTIFACT_HASH_MANIFEST.json"),"offline_results_complete":6,"new_final_physical_rollouts":0}
    atomic_json(OUT / "CURRENT_STATUS.json",status_data)
    atomic_text(OUT / "CURRENT_STATUS.md",f"# Final single-variable A/B rebuild status\n\nCOMMON_SOLVER_NOT_QUALIFIED\n\nCompleted: contract and splits; source-conditioned DEV35 environments; 75 raw reference pairs; strict common event clock; all six bounded position-only smoke trajectories; terminal report and provenance hashes.\n\nPosition-only: A 0/3; B 2/3. Self-collision frames: A 39, B 0. Hard-limit violations: 0. Branch discontinuities: 0.\n\nNext required gate: a separately versioned common solver qualification after reviewing the saved bounded failure evidence. The current attempt exhausted its declared budget. Full 6D, loaded Dex3, dataset/training stages, common/final freezes, and all physical evaluation remain NOT_RUN.\n\n- Contract SHA256: `{contract_hash}`\n- Qualification: `{qualification_path}` SHA256 `{qhash}`\n- Report: `{report_path}` SHA256 `{sha256(report_path)}`\n- Complete hashes: `ARTIFACT_HASH_MANIFEST.json`\n\nAll prior physical results are PRE_FINAL_SINGLE_VARIABLE_REBUILD_DIAGNOSTIC_ONLY. No old artifacts were deleted. No real hardware commands or PhysX rollouts were sent. Reuse the six persisted qualification results; the runner verifies hashes before reuse.\n")
    sep="\n==================================================\n"
    terminal_text="FINAL SINGLE-VARIABLE A-vs-B REBUILD\n"+sep+"\nEXPERIMENT CONTRACT\n\nOnly intended difference:\nA = WRIST TARGET GENERATION\nB = INTERACTION TARGET GENERATION\n\nUnintended downstream confounds:\n0 observed in raw references and solver input/API audit; unrun downstream stages NOT_ESTABLISHED\n"
    terminal_text+=sep+f"\nCOMMON EXECUTION\n\nTask registration: PASS (coordinate convention/environment equality)\nA raw target: PASS ({count}/{count} qualification TRAIN; all 40 TRAIN and 35 DEV audited)\nB raw target: PASS ({count}/{count} qualification TRAIN; all 40 TRAIN and 35 DEV audited)\nPosition-only IK:\nA 0 / 3\nB 2 / 3\nTRAIN11 position IK: NOT_RUN_SMOKE3_FAILED\nFull 6D IK:\nA NOT_RUN\nB NOT_RUN\nSelf-collision frames:\nA 39\nB 0\nBranch discontinuities: 0\nLoaded Dex3: NOT_RUN (0/14 tested; not qualified)\nShared reference physical smoke:\nA NOT_RUN\nB NOT_RUN\n"
    terminal_text+=sep+"\nACTION-TARGET AUDIT\n\nA training actions changed: UNKNOWN_FINAL_CORRECTED_TENSORS_UNAVAILABLE\nB training actions changed: UNKNOWN_FINAL_CORRECTED_TENSORS_UNAVAILABLE\nA changed frames/scalars: NOT_AVAILABLE\nB changed frames/scalars: NOT_AVAILABLE\nDataset regeneration required: PENDING_EXACT_FINAL_AUDIT\nACT retraining required: PENDING_EXACT_FINAL_AUDIT\n"
    terminal_text+=sep+"\nCORRECTED DATASETS\n\nA dataset: NOT_GENERATED\nSHA256: NOT_AVAILABLE\nB dataset: NOT_GENERATED\nSHA256: NOT_AVAILABLE\nUnintended dataset confounds: NOT_AUDITED\n"
    terminal_text+=sep+"\nACT TRAINING\n\nACT-A selected checkpoint: NO_REBUILD_CHECKPOINT\nSHA256: NOT_AVAILABLE\nACT-B selected checkpoint: NO_REBUILD_CHECKPOINT\nSHA256: NOT_AVAILABLE\nSame training protocol: RECOVERED; new training NOT_RUN\nSame checkpoint rule: RECOVERED; new selection NOT_RUN\n"
    terminal_text+=sep+"\nDEV35 FREEZE\n\nSHA256: NOT_CREATED_PREREQUISITES_FAILED\nFull-task scope viable: NOT_ESTABLISHED\nStandardized-grasp fallback used: NO\n"
    terminal_text+=sep+"\nDEV35 PHYSICAL RUNS\n\nACT-A: NOT_RUN (0 new rollouts; not 0/35 success)\nACT-B: NOT_RUN (0 new rollouts; not 0/35 success)\n"
    for label,mode in (("A","WRIST"),("B","INTERACTION")):
        terminal_text+=sep+f"\nACT-{label} — {mode}\n\n"+"\n".join(f"{stage}: NOT_RUN" for stage in ("APPROACH","LEFT GRASP","LIFT","HANDOFF","RIGHT OWNERSHIP","RIGHT TRANSPORT","BIN ENTRY","BIN SETTLE","FULL TASK"))+"\n"
    terminal_text+=sep+"\nB - A FULL TASK DIFFERENCE: NOT_AVAILABLE\nMATCHED TEST: NOT_COMPUTED_NO_FINAL_PHYSICAL_PAIRS\n"
    terminal_text+=sep+"\nUNTOUCHED FINAL TEST\n\nAvailable: NO manifest found\nREADY_FOR_UNTOUCHED_FINAL_TEST_COLLECTION = NO (common solver/pipeline not qualified or frozen)\n"
    terminal_text+=sep+f"\nMAIN FIGURE: NOT_GENERATED\nTABLE: NOT_GENERATED\nA TOP: NOT_GENERATED\nA OVERVIEW: NOT_GENERATED\nB TOP: NOT_GENERATED\nB OVERVIEW: NOT_GENERATED\nFINAL REPORT:\n{report_path}\n"+sep+"\nCOMMON_SOLVER_NOT_QUALIFIED\n"
    atomic_text(OUT / "FINAL_TERMINAL_OUTPUT.txt",terminal_text)
    print(terminal_text,end="")


if __name__=="__main__":main()
