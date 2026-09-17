#!/usr/bin/env python3
"""Run finite common position qualification; stop all later gates on failure."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.final_single_variable_prepare import OUT, RESET, read, file_record, digest, status
from tools.run_reference_motion_scientific_reset import atomic_json,atomic_npz,atomic_text,atomic_csv,sha256,stats
from tools.doll_handoff_retargeting.common import load_common_config,load_scene,branch_flags
from tools.doll_handoff_retargeting.models import G1Kinematics
from tools.common_g1_morphology_adapter import CommonG1MorphologyAdapter
from tools.final_common_position_solver import CommonPositionSolver

CONFIG=OUT / "01_registration/POSITION_SOLVER_QUALIFICATION_CONTRACT.json"
RESULTS=OUT / "02_common_execution_qualification/position_only"


def inspect_implementation():
    path=ROOT / "tools/final_common_position_solver.py"
    tree=ast.parse(path.read_text())
    names={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}
    forbidden=names & {"representation_mode","method","episode","object_pose","task_success","physical_outcome"}
    assert not forbidden
    return {"implementation":file_record(path),"forbidden_input_identifiers":sorted(forbidden),"pass":not forbidden,
            "scope":"solver API and complete implementation; driver labels paths but passes no method label to solver"}


def evaluate(solver,q,targets,hands,timestamps,cfg):
    actual=[]; contacts=[]; margins=[]
    pairs=set(p for ps in solver.g1.clearance_pairs.values() for p in ps) | solver.tracked
    for frame in range(len(q)):
        actual.append(solver.pose_jacobian(q[frame],hands[frame])[0])
        records=solver.contacts(q[frame],hands[frame])
        if records: contacts.append({"frame":frame,"records":records})
        # Signed distances saturate at 20mm, explicitly recorded in the report.
        model,data=solver.g1.model,solver.g1.data
        margins.append(min((float(mujoco.mj_geomDistance(model,data,*p,0.02,None)) for p in pairs),default=0.02))
    actual=np.asarray(actual)
    residual=np.linalg.norm(actual-targets,axis=2)
    step=np.diff(q,axis=0); dt=float(np.median(np.diff(timestamps)))
    velocity=np.abs(step)/dt; accel=np.abs(np.diff(q,n=2,axis=0))/dt**2
    branch=branch_flags(q,cfg["branch_absolute_step_norm_rad"],cfg["branch_local_multiplier"])
    limits=int(np.count_nonzero((q<solver.g1.arm_limits[:,0]-1e-9)|(q>solver.g1.arm_limits[:,1]+1e-9)))
    temporal=bool(np.max(np.abs(step))<=cfg["maximum_joint_step_rad"]+1e-7 and np.max(velocity)<=cfg["maximum_velocity_rad_s"]+1e-7 and np.max(accel)<=cfg["maximum_acceleration_rad_s2"]+1e-5)
    acceptance=float(np.mean(np.max(residual,axis=1)<=cfg["position_tolerance_m"]))
    # A point projection into a selected feasible wrist ball is the minimum
    # target change for that realization. It is not a claim of global optimum.
    needed=np.maximum(residual-cfg["position_tolerance_m"]+1e-5,0)
    bounded=np.minimum(needed,cfg["projection"]["maximum_translation_m"])
    direction=(actual-targets)/np.maximum(residual[:,:,None],1e-12)
    feasible_candidate=targets+direction*bounded[:,:,None]
    candidate_residual=np.linalg.norm(actual-feasible_candidate,axis=2)
    projection_allowed=bool(not contacts and limits==0 and not np.any(branch) and temporal)
    required_projection_within_bound=bool(np.max(needed)<=cfg["projection"]["maximum_translation_m"]+1e-9)
    projected_qualified=bool(projection_allowed and required_projection_within_bound and np.mean(np.max(candidate_residual,axis=1)<=cfg["position_tolerance_m"])>=cfg["required_frame_acceptance_rate"])
    raw_qualified=bool(acceptance>=cfg["required_frame_acceptance_rate"] and not contacts and limits==0 and not np.any(branch) and temporal and np.isfinite(q).all())
    metrics={"frame_count":len(q),"raw_position_acceptance_rate":acceptance,"position_residual_mm":stats(np.max(residual,axis=1),1000),
             "q_step_norm_rad":stats(np.linalg.norm(step,axis=1)),"q_step_scalar_abs_rad":stats(np.abs(step)),
             "self_collision_frames":len(contacts),"self_collision_minimum_margin_m":float(min(margins)),"distance_query_upper_saturation_m":0.02,
             "hard_limit_violations":limits,"branch_discontinuities":int(np.count_nonzero(branch)),"finite":bool(np.isfinite(q).all()),
             "maximum_velocity_rad_s":float(np.max(velocity)),"maximum_acceleration_rad_s2":float(np.max(accel)),"temporal_pass":temporal,
             "raw_position_qualified":raw_qualified,"position_qualified":raw_qualified or projected_qualified,
             "projection":{"required_for_selected_realization_mm":stats(needed,1000),"bounded_candidate_mm":stats(bounded,1000),"permitted_by_collision_and_temporal_gates":projection_allowed,"all_required_changes_within_bound":required_projection_within_bound,"qualified":projected_qualified,"applied_to_training":False,"interpretation":"No globally nearest target claim; candidate is not a qualified feasible target unless every common gate passes"},"contacts":contacts,
             "failing_cartesian_frames":np.flatnonzero(np.max(residual,axis=1)>cfg["position_tolerance_m"]).tolist()}
    return metrics,actual,residual,feasible_candidate,np.asarray(margins)


def run(episode,mode):
    cfg=read(CONFIG)
    raw=OUT / "01_registration/raw_references" / f"TRAIN_EP{episode:03d}.npz"
    base=RESULTS / f"{mode}_EP{episode:03d}"
    result_json=base.with_suffix(".json"); result_npz=base.with_suffix(".npz")
    dependencies={"source":file_record(raw),"config":file_record(CONFIG),"solver":file_record(ROOT / "tools/final_common_position_solver.py"),"runner":file_record(Path(__file__)),"common":file_record(RESET / "config/common_config.json"),"collision":file_record(ROOT / "tools/doll_handoff_feasibility/solver.py"),"initial_state":file_record(ROOT / "outputs/paper_core_ab/COMMON_G1_POLICY_INITIAL_STATE_AB_V1.json")}
    if result_json.exists():
        cached=read(result_json)
        if cached["dependencies"]==dependencies and sha256(result_npz)==cached["trajectory_sha256"]:
            print(f"REUSE {mode} {episode}: {cached['position_qualified']}",flush=True)
            return cached
        raise RuntimeError(f"Existing result dependencies changed: preserve {result_json} before creating a separately versioned rerun")
    assert read(OUT / "01_registration/RAW_TARGET_AUDIT.json")["unintended_reference_confounds"]==0
    common=load_common_config(RESET / "config/common_config.json"); scene=load_scene(common)
    g1=G1Kinematics(common,scene)
    nominal=np.asarray(common["resolved"]["canonical_g1_nominal_q"])
    initial=np.asarray(read(dependencies["initial_state"]["path"])["g1_14_arm_initial_q_rad"])
    collision=CommonG1MorphologyAdapter(common,g1,nominal)
    primitives=g1.derive_hand_primitives(scene,read(RESET / "config/proposed_config.json"),nominal)
    solver=CommonPositionSolver(g1,collision,cfg,nominal)
    with np.load(raw,allow_pickle=False) as source:
        targets=np.stack([source[f"{mode}_{side}_wrist_position_model"] for side in ("left","right")],axis=1)
        hands=np.stack([np.asarray(primitives["states"][side]["OPEN"])+source[f"common_{side}_close_fraction"][:,None]*(np.asarray(primitives["states"][side]["GRASP"])-np.asarray(primitives["states"][side]["OPEN"])) for side in ("left","right")],axis=1)
        timestamps=source["source_timestamp"].copy()
    for i,side in enumerate(("left","right")):
        assert np.all(hands[:,i]>=g1.hand_limits[side][:,0]-1e-9) and np.all(hands[:,i]<=g1.hand_limits[side][:,1]+1e-9)
    input_hash=hashlib.sha256(targets.tobytes()).hexdigest()
    print(f"POSITION {mode} EP{episode:03d} START {len(targets)} frames",flush=True)
    q,details=solver.solve(targets,hands,timestamps,initial)
    assert hashlib.sha256(targets.tobytes()).hexdigest()==input_hash
    metrics,actual,residual,projected,margins=evaluate(solver,q,targets,hands,timestamps,cfg)
    witness=[]
    if not metrics["position_qualified"]:
        # First/worst/last failing frames are deterministic evidence extraction
        # after the trajectory; these are not calibration episodes or outcomes.
        failed=metrics["failing_cartesian_frames"]
        if failed:
            frames=list(dict.fromkeys([failed[0],int(np.argmax(np.max(residual,axis=1))),failed[-1]]))
            for frame in frames[:cfg["global_witness_frames_per_trajectory"]]:
                witness.append({"frame":frame,"incoming_target_model_m":targets[frame],**solver.witness(targets[frame],hands[frame])})
    atomic_npz(result_npz,q=q,source_timestamp=timestamps,RAW_REPRESENTATION_TARGET=targets,COMMON_FEASIBLE_TARGET_CANDIDATE=projected,common_feasible_target_qualified=np.asarray(metrics["position_qualified"]),actual_wrist_position_model=actual,position_residual_m=residual,self_collision_margin_m=margins,common_hand_q=hands,initial_q=initial,reverse_posture_guide=details.pop("reverse_guide"))
    atomic_json(base.with_name(base.name+"_CANDIDATES").with_suffix(".json"),details)
    result={"representation_mode":mode,"episode_index":episode,"dependencies":dependencies,"implementation_audit":inspect_implementation(),"trajectory_sha256":sha256(result_npz),"runtime_s":details["runtime_s"],"function_evaluations":details["function_evaluations"],"global_witnesses":witness,**metrics}
    atomic_json(result_json,result)
    status(f"POSITION_TRAJECTORY_{mode}_EP{episode:03d}_PERSISTED", "REMAINING_SMOKE3_POSITION_TRAJECTORIES",[result_json,result_npz])
    print(json.dumps({k:result[k] for k in ("representation_mode","episode_index","position_qualified","raw_position_acceptance_rate","position_residual_mm","self_collision_frames","hard_limit_violations","branch_discontinuities","maximum_acceleration_rad_s2","runtime_s")}),flush=True)
    return result


def summarize():
    rows=[read(RESULTS/f"{mode}_EP{ep:03d}.json") for mode in ("WRIST","INTERACTION") for ep in (0,24,49)]
    complete=all(r["position_qualified"] for r in rows)
    summary={"status":"POSITION_SMOKE3_PASS" if complete else "COMMON_SOLVER_NOT_QUALIFIED","rows":rows,
             "counts":{mode:sum(r["position_qualified"] for r in rows if r["representation_mode"]==mode) for mode in ("WRIST","INTERACTION")},
             "training_qualification11":"NEXT_GATE" if complete else "NOT_RUN_SMOKE3_FAILED", "full_6d":"NOT_RUN", "loaded_dex3":"NOT_RUN", "training":"NOT_RUN","physics":"NOT_RUN"}
    atomic_json(RESULTS.parent / "COMMON_POSITION_IK_QUALIFICATION.json",summary)
    table="| Mode | Episode | Accepted frames | Residual mean/p95/max mm | Self-contact frames | Limit violations | Branches | qdot max | qddot max | Pass |\n|---|---:|---:|---|---:|---:|---:|---:|---:|---|\n"
    for r in rows:
        p=r["position_residual_mm"]
        table+=f"| {r['representation_mode']} | {r['episode_index']} | {100*r['raw_position_acceptance_rate']:.2f}% | {p['mean']:.3f}/{p['p95']:.3f}/{p['max']:.3f} | {r['self_collision_frames']} | {r['hard_limit_violations']} | {r['branch_discontinuities']} | {r['maximum_velocity_rad_s']:.3f} | {r['maximum_acceleration_rad_s2']:.3f} | {r['position_qualified']} |\n"
    atomic_text(RESULTS.parent / "COMMON_POSITION_IK_QUALIFICATION.md", "# Common position-only qualification\n\n"+summary["status"]+"\n\n"+table+"\nActual sequential G1 kinematics, hard limits, signed collision geometry, reverse posture propagation, and bounded forward continuity were evaluated. No radial-only reachability verdict or method-specific solver branch was used. Source registration and raw wrist targets were preserved. Candidate projections are explicitly unqualified unless all gates pass. Full-limit multiseed witnesses provide bounded search evidence, not a proof of global infeasibility.\n\nThe declared budget is exhausted after these six trajectories; later gates must remain unrun if any trajectory fails.\n")
    status(summary["status"],"TRAIN_QUALIFICATION11" if complete else "STOP_REQUIRED_BY_BOUNDED_COMMON_SOLVER_CONTRACT",[RESULTS.parent / "COMMON_POSITION_IK_QUALIFICATION.json",RESULTS.parent / "COMMON_POSITION_IK_QUALIFICATION.md"])
    print(summary["status"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--episode",type=int,choices=[0,24,49]);p.add_argument("--mode",choices=["WRIST","INTERACTION"]);p.add_argument("--summarize",action="store_true")
    a=p.parse_args()
    if a.summarize:summarize()
    elif a.episode is not None and a.mode:run(a.episode,a.mode)
    else:p.error("provide episode and mode, or summarize")
