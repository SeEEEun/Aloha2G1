#!/usr/bin/env python3
"""Recover and qualify inputs for the final experiment without policy/physics use."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_reference_motion_scientific_reset import (
    RESET, DETECTOR, SOURCE_EVENTS, SOURCE_MANIFEST, REGISTRATION,
    atomic_json, atomic_npz, atomic_text, atomic_csv, sha256, native,
    load_recording, event_proxy, target_audit, source_object, enriched_registration,
    transform_matrix, save_reference_archive,
)
from tools.doll_handoff_retargeting.common import load_common_config, load_scene
from tools.doll_handoff_retargeting.models import ALOHAKinematics, G1Kinematics
from tools.doll_handoff_retargeting.retarget import RepresentationBuilder
from tools.reference_motion_common_timeline import build_common_timeline

OUT = ROOT / "outputs/final_single_variable_ab"
CORE = ROOT / "outputs/paper_core_ab"
REF = ROOT / "outputs/reference_motion_scientific_reset"
ALIASES = {"LEFT_GRASP_CONFIRMED_SOURCE": "LEFT_GRASP_SOURCE",
           "LEFT_TRANSPORT": "LEFT_TRANSPORT_BEGIN", "DUAL_CONTACT_SOURCE": "DUAL_SUPPORT_SOURCE",
           "RIGHT_OWNED_SOURCE": "RIGHT_OWNERSHIP_SOURCE"}


def read(path):
    return json.loads(Path(path).read_text())


def file_record(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}


def digest(value):
    return hashlib.sha256(json.dumps(native(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def status(gate, next_gate, paths):
    record = {"completed_gate": gate, "next_required_gate": next_gate,
              "artifacts": [file_record(p) for p in paths],
              "downstream_training_and_physics": "NOT_RUN"}
    atomic_json(OUT / "CURRENT_STATUS.json", record)
    atomic_text(OUT / "CURRENT_STATUS.md", "# Final single-variable A/B rebuild status\n\n"
                + f"Completed gate: {gate}\n\nNext required gate: {next_gate}\n\n"
                + "\n".join(f"- `{a['path']}` SHA256 `{a['sha256']}`" for a in record["artifacts"])
                + "\n\nTraining and final physical evaluation: NOT_RUN. Prior results are diagnostic provenance only.\n")


def recover():
    hashes = read(OUT / "00_contract/SINGLE_VARIABLE_EXPERIMENT_CONTRACT.sha256.json")
    for name, expected in hashes.items():
        assert sha256(OUT / "00_contract" / name) == expected, name
    manifests = {k: read(CORE / f"{k}_manifest.json") for k in ("common48", "train40", "heldout8")}
    original = read(SOURCE_MANIFEST)
    reg = read(REGISTRATION)
    train = {r["original_source_recording_id"] for r in manifests["train40"]["entries"]}
    held = {r["original_source_recording_id"] for r in manifests["heldout8"]["entries"]}
    common = {r["original_source_recording_id"] for r in manifests["common48"]["entries"]}
    dev = {r["source_recording"]: r for r in reg["entries"]}
    assert len(train) == 40 and len(held) == 8 and train.isdisjoint(held)
    assert train | held == common and len(dev) == 35 and train.isdisjoint(dev)
    candidates = sorted(r["final_dataset_index"] for r in manifests["train40"]["entries"])
    spread = [candidates[i] for i in np.linspace(0, len(candidates)-1, 10).round().astype(int)]
    selection = {"rule": read(OUT / "00_contract/SINGLE_VARIABLE_EXPERIMENT_CONTRACT.json")["qualification_selection"],
                 "train40_ids": candidates, "train_spread10_ids": spread,
                 "qualification_ids": sorted(set(spread) | {0,24,49}),
                 "dev35_smoke_indices": np.linspace(0,34,5).round().astype(int).tolist(),
                 "episode_id_namespace": "authoritative final_dataset_index; original source index remains separate, including replacement recordings",
                 "selected_before_new_solver_outcomes": True}
    selection["selection_content_sha256"] = digest(selection)
    atomic_json(OUT / "00_contract/QUALIFICATION_SELECTION.json", selection)
    training = read(CORE / "act_a_b_training_contract.json")
    audits = read(CORE / "act_a_b_training_audit.json")
    checkpoints = {}
    for key in ("a", "b"):
        root = CORE / f"act_{key}40"
        possible = list(root.rglob("*selection*.json"))
        checkpoints[key] = {"training_contract": training["records"][key],
                            "training_config": file_record(training["records"][key]["config"]),
                            "training_audit": audits["methods"][key] if "methods" in audits and key in audits["methods"] else audits.get("methods", {}),
                            "selection_artifacts": [file_record(p) for p in possible],
                            "checkpoint_files": [file_record(p) for p in sorted(root.glob("train/checkpoints/*/pretrained_model/model.safetensors"))],
                            "eligibility": "OLD_CHECKPOINT_REUSE_PENDING_EXACT_CORRECTED_ACTION_AUDIT"}
    entries = []
    by_name = {r["source_name"]: r for r in original["records"]}
    common_by_name = {r["original_source_recording_id"]:r for r in manifests["common48"]["entries"]}
    for name in sorted(set(by_name) | set(dev) | set(common_by_name)):
        row = by_name.get(name)
        c = common_by_name.get(name)
        source = Path(row["parquet_path"]) if row else Path(c["source_parquet_path"]) if c else Path(dev[name]["source_grasp_window"]["evidence_source"])
        expected = row["parquet_sha256"] if row else c["source_parquet_sha256"] if c else dev[name]["source_grasp_window"]["evidence_source_sha256"]
        actual = sha256(source)
        assert actual == expected, source
        old_refs = list((REF / "corrected_references").rglob(f"EP{row['episode_index']:02d}_*.npz")) if row else []
        if name in dev:
            old_refs += list((REF / "corrected_references/dev35").glob(f"EVAL{dev[name]['eval_index']:02d}_*.npz"))
        entry = {"source_recording_id": name,
                 "stable_episode_id": row["stable_episode_id"] if row else c["stable_episode_id"] if c else dev[name]["stable_episode_id"],
                 "original_episode_index": row["episode_index"] if row else None,
                 "final_dataset_index": c["final_dataset_index"] if c else None,
                 "source_parquet": str(source), "source_parquet_sha256": actual,
                 "original50": row is not None, "COMMON48": name in common,
                 "TRAIN40": name in train, "original_HELDOUT8": name in held,
                 "DEV35_DIAGNOSTIC35": name in dev,
                 "dev_index": dev[name]["eval_index"] if name in dev else None,
                 "previous_checkpoint_usage": "ACT_A40_AND_ACT_B40_TRAINING" if name in train else ("HELDOUT_CHECKPOINT_SELECTION_AND_REPEATED_DIAGNOSTICS" if name in held else "REPEATED_DEVELOPMENT_DIAGNOSTICS" if name in dev else "ORIGINAL_COLLECTION_EXCLUDED_FROM_COMMON48"),
                 "corrected_cartesian_references": [file_record(p) for p in old_refs],
                 "qualified_corrected_joint_actions_available": False}
        entries.append(entry)
    split = {"schema_version": "final_authoritative_split_v1", "evaluation_label": "DEV35 / DIAGNOSTIC35",
             "counts": {"original50": len(by_name), "COMMON48": len(common), "TRAIN40": len(train), "original_HELDOUT8": len(held), "DEV35": len(dev), "unique_recordings": len(entries)},
             "sources": [file_record(SOURCE_MANIFEST), file_record(REGISTRATION)] + [file_record(CORE / f"{k}_manifest.json") for k in manifests],
             "split_contract": manifests["train40"]["split_contract"], "selection": selection,
             "previous_checkpoints": checkpoints, "entries": entries}
    atomic_json(OUT / "00_contract/AUTHORITATIVE_SPLIT_MANIFEST.json", split)
    prior_dirs = [p for p in (ROOT / "outputs").iterdir() if p.is_dir() and (
        p.name.startswith(("final_", "single_variable_", "standardized_grasp_", "dex3_")) or p.name == "paper_physics_task_eval") and p != OUT]
    prior = {"status": "PRE_FINAL_SINGLE_VARIABLE_REBUILD_DIAGNOSTIC_ONLY",
             "applies_recursively_to": [str(p) for p in sorted(prior_dirs)],
             "all_pre_contract_physical_results": True,
             "preserve_original_bytes": True, "eligible_for_final_statistics": False,
             "includes": ["previous ACT-A 0/35", "invalid/incomplete ACT-B", "standardized-grasp 0/35 vs 0/35", "biased graspability evaluators", "invalid IK/workspace experiments"],
             "report_hashes": [file_record(p) for d in prior_dirs for p in sorted(d.glob("*.md"))]}
    atomic_json(OUT / "00_contract/PRIOR_DIAGNOSTIC_PROVENANCE.json", prior)
    atomic_text(OUT / "00_contract/PRIOR_DIAGNOSTIC_PROVENANCE.md", "# Previous result classification\n\nPRE_FINAL_SINGLE_VARIABLE_REBUILD_DIAGNOSTIC_ONLY\n\nAll physical results produced before this contract, including ACT-A 0/35, incomplete ACT-B, standardized-grasp 0/35 versus 0/35, biased evaluators, and invalid workspace/IK trials, are excluded from final statistics. Original files are preserved. The JSON companion identifies directory scopes and hashes.\n")
    envs = []
    for r in reg["entries"]:
        source = r["source_to_target_transform"]
        assert source["uniform_metric_scale"] == 1.0
        q = Rotation.from_quat(source["quaternion_xyzw"])
        expected = q.apply(r["source_object_pose"]["position_xyz_m"]) + source["translation_xyz_m"]
        assert np.max(np.abs(expected-r["target_object_pose"]["position_xyz_m"])) < 1e-12
        assert (q * Rotation.from_quat(r["source_object_pose"]["quaternion_xyzw"])
                * Rotation.from_quat(r["target_object_pose"]["quaternion_xyzw"]).inv()).magnitude() < 1e-12
        assert not r["source_object_task_frame_source"]["ACT_outputs_used_for_object_pose"]
        assert not r["source_object_task_frame_source"]["physical_A_B_outcomes_read"]
        pose = copy.deepcopy(r["target_object_pose"])
        e = {"dev_index": r["eval_index"], "source_recording_id": r["source_recording"], "stable_episode_id": r["stable_episode_id"],
             "source_object_pose": r["source_object_pose"], "source_to_target_transform": source,
             "T_object_A": pose, "T_object_B": copy.deepcopy(pose),
             "source_evidence": {k:r[k] for k in ("source_object_task_frame_source", "source_grasp_window", "orientation_evidence")},
             "bin_pose": reg["bin_pose"], "translation_difference_mm": 0.0, "rotation_difference_deg": 0.0}
        assert digest(e["T_object_A"]) == digest(e["T_object_B"])
        e["entry_sha256"] = digest(e)
        envs.append(e)
    env = {"status": "PASS_REGISTRATION_ONLY", "evaluation_label": "DEV35 / DIAGNOSTIC35",
           "source": file_record(REGISTRATION), "source_derived_count": len(envs), "matched_equality_count": len(envs),
           "maximum_translation_difference_mm": 0.0, "maximum_rotation_difference_deg": 0.0,
           "bin_pose": reg["bin_pose"], "entries": envs}
    atomic_json(OUT / "01_registration/DEV35_EPISODE_TASK_ENVIRONMENTS.json", env)
    atomic_csv(OUT / "01_registration/DEV35_EPISODE_TASK_ENVIRONMENTS.csv", [{"dev_index":e["dev_index"], "source_recording_id":e["source_recording_id"], **dict(zip(("x_m","y_m","z_m"),e["T_object_A"]["position_xyz_m"])), **dict(zip(("qx","qy","qz","qw"),e["T_object_A"]["quaternion_xyzw"])), "A_B_equal":True, "entry_sha256":e["entry_sha256"]} for e in envs])
    atomic_text(OUT / "01_registration/DEV35_EPISODE_TASK_ENVIRONMENTS.md", "# DEV35 source-conditioned environments\n\n35/35 source-derived; 35/35 exact matched A/B pose equality; maximum translation difference 0 mm and rotation difference 0 degrees.\n\nRecovered from source task geometry and source images. Policy predictions and rollout outcomes are not inputs. The bin is fixed across episodes. This is registration qualification only.\n")
    atomic_json(OUT / "00_contract/RECOVERED_ACT_PROTOCOL.json", {"contract":training,"selection_rule":read(CORE / "checkpoint_selection_rule.json"),"source":file_record(CORE / "act_a_b_training_contract.json")})
    status("CONTRACT_SPLITS_PROVENANCE_AND_DEV35_REGISTRATION", "RAW_TARGET_AND_SHARED_EVENT_CLOCK_AUDIT", [OUT / "00_contract/AUTHORITATIVE_SPLIT_MANIFEST.json", OUT / "00_contract/QUALIFICATION_SELECTION.json", OUT / "01_registration/DEV35_EPISODE_TASK_ENVIRONMENTS.json"])
    print(json.dumps({"recovered_counts":split["counts"], "qualification_ids": selection["qualification_ids"], "dev_smoke_indices":selection["dev35_smoke_indices"]}), flush=True)


def strict_clock(timeline, timestamps):
    """Common command-phase mapping; retain measured motion and source events."""
    source = {ALIASES.get(k,k):v for k,v in timeline.times_sec.items()}
    target = dict(source)
    dt = float(np.median(np.diff(timestamps)))
    duration = source["LEFT_CLOSE_COMPLETE"] - source["LEFT_CLOSE_BEGIN"]
    # Advance closing only where necessary. No wrist sample is cut or rebased.
    target["LEFT_CLOSE_COMPLETE"] = min(source["LEFT_CLOSE_COMPLETE"], source["LEFT_LIFT_BEGIN"]-dt)
    target["LEFT_CLOSE_BEGIN"] = min(source["LEFT_CLOSE_BEGIN"], target["LEFT_CLOSE_COMPLETE"]-duration)
    assert target["LEFT_CLOSE_BEGIN"] >= timestamps[0]
    assert target["LEFT_CLOSE_COMPLETE"] < target["LEFT_LIFT_BEGIN"]
    assert target["RIGHT_ACQUIRE_SOURCE"] < target["LEFT_RELEASE_BEGIN"]
    arrays = {}
    for side, begin, end, release in (("left","LEFT_CLOSE_BEGIN","LEFT_CLOSE_COMPLETE","LEFT_RELEASE_BEGIN"), ("right","RIGHT_CLOSE_BEGIN","RIGHT_ACQUIRE_SOURCE","FINAL_RELEASE_BEGIN")):
        u = np.clip((timestamps-target[begin])/max(target[end]-target[begin],dt),0,1)
        close = u**3 * (10-15*u+6*u**2)
        release_duration = max(duration, dt)
        v = np.clip((timestamps-target[release])/release_duration,0,1)
        close *= 1-v**3*(10-15*v+6*v**2)
        arrays[f"common_{side}_close_fraction"] = close
    return {"source_times_sec":source, "execution_times_sec":target,
            "execution_minus_source_sec":{k:target[k]-source[k] for k in source},
            "command_close_advanced":target["LEFT_CLOSE_COMPLETE"] != source["LEFT_CLOSE_COMPLETE"],
            "motion_timestamps_modified":False,
            "mapping":"shared timestamp-normalized quintic close/release; measured source motion retained",
            "mechanical_confirmation":"NOT_TESTED; runtime controller must enforce before lift"}, arrays


def references():
    split = read(OUT / "00_contract/AUTHORITATIVE_SPLIT_MANIFEST.json")
    selection = split["selection"]
    common = load_common_config(RESET / "config/common_config.json")
    scene = load_scene(common)
    aloha, g1 = ALOHAKinematics(common,scene), G1Kinematics(common,scene)
    alignment = read(RESET / "config/tool_frame_report.json")["source_to_target_axis_alignment"]
    representation = RepresentationBuilder(common, scene, g1, alignment, read(RESET / "config/proposed_config.json"), read(RESET / "config/baseline_workspace_mapping_report.json"))
    detector, previous_events = read(DETECTOR), read(SOURCE_EVENTS)
    registration = {r["source_recording"]:r for r in read(REGISTRATION)["entries"]}
    clock_rows, audit_rows, manifest = [], [], []
    implementation = [file_record(Path(__file__)), file_record(ROOT / "tools/reference_motion_common_timeline.py"), file_record(ROOT / "tools/doll_handoff_retargeting/retarget.py"), file_record(RESET / "config/common_config.json"), file_record(DETECTOR)]
    for row in split["entries"]:
        if not (row["TRAIN40"] or row["DEV35_DIAGNOSTIC35"]):
            continue
        name, ep = row["source_recording_id"], row["final_dataset_index"]
        recording = load_recording(name)
        fk = aloha.fk(recording["state"])
        coarse = previous_events.get(str(row["original_episode_index"]),{}).get("frames") if row["original_episode_index"] is not None else None
        coarse = {k:int(v) for k,v in coarse.items() if v is not None} if coarse else None
        timeline = build_common_timeline(action=recording["action"], timestamps=recording["timestamp"], left_tcp_position_world=fk["left_tcp_position_world"], right_tcp_position_world=fk["right_tcp_position_world"], detector_config=detector, coarse_events=coarse)
        reg = enriched_registration(registration[name],scene) if name in registration else None
        matrix = transform_matrix(registration[name]) if name in registration else np.eye(4)
        event = event_proxy(timeline,name,ep if ep is not None else row["dev_index"])
        targets = {mode:representation.build(mode,fk,event,reg) for mode in ("WRIST","INTERACTION")}
        object_pos = np.asarray(registration[name]["target_object_pose"]["position_xyz_m"]) if name in registration else source_object(scene)
        audit = target_audit(fk,{"A":targets["WRIST"],"B":targets["INTERACTION"]},matrix,g1,object_pos)
        # Independent A orientation check against registered source orientation.
        a_orientation = []
        for side in ("left","right"):
            expected = np.einsum("ij,tjk,kl->til", matrix[:3,:3], fk[f"{side}_tcp_rotation_world"], np.asarray(alignment["sides"][side]["source_interaction_to_g1_grasp_axis_alignment"]))
            local = np.asarray(targets["WRIST"]["static_wrist_to_tool"][side])
            actual = g1.model_to_world_rotation(np.einsum("tij,jk->tik",targets["WRIST"][f"{side}_wrist_rotation"],local[:3,:3]))
            a_orientation.extend((Rotation.from_matrix(actual) * Rotation.from_matrix(expected).inv()).magnitude())
        audit["A"]["independent_source_orientation_max_deg"] = float(np.degrees(max(a_orientation)))
        clock, clock_arrays = strict_clock(timeline,recording["timestamp"])
        key = f"TRAIN_EP{ep:03d}" if row["TRAIN40"] else f"DEV{row['dev_index']:02d}"
        path = OUT / "01_registration/raw_references" / f"{key}.npz"
        values = {"source_recording_id":np.asarray(name),"source_timestamp":recording["timestamp"], "source_action":recording["action"], "source_state":recording["state"], "source_event_names":np.asarray(list(clock["source_times_sec"])), "source_event_times_sec":np.asarray(list(clock["source_times_sec"].values())), "common_execution_event_times_sec":np.asarray(list(clock["execution_times_sec"].values())), "common_normalized_time":(recording["timestamp"]-recording["timestamp"][0])/(recording["timestamp"][-1]-recording["timestamp"][0]), **clock_arrays}
        for mode in targets:
            for side in ("left","right"):
                for component in ("position","rotation"):
                    values[f"{mode}_{side}_wrist_{component}_model"] = targets[mode][f"{side}_wrist_{component}"]
                values[f"{mode}_{side}_wrist_to_tool"] = targets[mode]["static_wrist_to_tool"][side]
        for k,v in fk.items():
            if isinstance(v,np.ndarray): values[f"source_fk_{k}"] = v
        atomic_npz(path,**values)
        clock_rows.append({"source_recording_id":name,"scope":"TRAIN40" if row["TRAIN40"] else "DEV35", "key":key, **clock})
        passed = all(a["finite"] and a["position_reconstruction_error_mm"]["max"]<1e-3 and a["orientation_reconstruction_error_deg"]["max"]<1e-3 and a["bimanual_relation_error_mm"]["max"]<1e-3 for a in audit.values()) and max(a_orientation)<1e-8
        assert passed and not timeline.order_violations, key
        audit_rows.append({"key":key,"original_episode_index":ep,"source_recording_id":name,"TRAIN40":row["TRAIN40"],"qualification_subset":row["TRAIN40"] and ep in selection["qualification_ids"], "frame_count":len(recording["timestamp"]),"pass":passed,"audit":audit})
        manifest.append({"key":key,"original_episode_index":ep,"source_recording_id":name,"TRAIN40":row["TRAIN40"],"reference":file_record(path),"source":file_record(recording["parquet"])})
        print(f"reference {key} PASS {len(recording['timestamp'])} frames",flush=True)
    atomic_json(OUT / "01_registration/COMMON_SOURCE_AND_EXECUTION_EVENT_CLOCK.json", {"implementation":implementation, "entries":clock_rows,"shared_between_A_B":True,"source_clock_preserved":True,"retiming_disclosed":True})
    atomic_json(OUT / "01_registration/RAW_REFERENCE_MANIFEST.json", {"implementation":implementation,"entries":manifest})
    atomic_json(OUT / "01_registration/RAW_TARGET_AUDIT.json", {"entries":audit_rows,"unintended_reference_confounds":0,"orientation_and_position_audit_scope":"source TCP / initial task relation and target whole-hand frame; no dynamic object tracking or physical competence claim"})
    for label,name in (("A","WRIST"),("B","INTERACTION")):
        subset=[r for r in audit_rows if r["qualification_subset"]]
        maxp=max(r["audit"][label]["position_reconstruction_error_mm"]["max"] for r in subset)
        maxr=max(r["audit"][label]["orientation_reconstruction_error_deg"]["max"] for r in subset)
        atomic_text(OUT / "01_registration" / f"{label}_{name}_RAW_TARGET_AUDIT.md", f"# {label} {name} raw target audit\n\nPASS: {len(subset)}/{len(subset)} predetermined TRAIN qualification episodes, including 0, 24, 49.\n\nMaximum position reconstruction error: {maxp:.12g} mm. Maximum orientation reconstruction error: {maxr:.12g} degrees. All 40 TRAIN and 35 DEV references independently regenerated and audited.\n\nA reconstructs registered source TCP position and orientation through the fixed wrist/tool transform. B reconstructs the registered initial object-relative and bimanual translation relation with target whole-hand orientation. Source ownership/event labels are common. Source dynamic object poses are unavailable; this audit does not establish dynamic contact topology. No IK or physical outcome was used.\n\nThe source clock is retained. A common disclosed command-phase mapping advances close completion where needed to precede measured lift by one source sample interval. Wrist positions, timestamps, and natural-start frame zero are preserved. Mechanical confirmation remains a downstream gate.\n")
    registration_audit={"status":"PASS_COORDINATE_CONVENTION_ONLY", "selected_workspace_transform":np.eye(4),"selection_reason":"authoritative qualified reference and scene registration; previous workspace/IK trials are unqualified diagnostic provenance", "common_task_registration":common["task_registration"],"frame_convention":{"points":"R @ p + t; row batch p @ R.T + t", "rotations":"active right-handed matrices; target R = registration R @ source R", "quaternions":"source configuration WXYZ explicitly reordered to XYZW; exported XYZW", "tool":"T_world_tool = T_world_wrist @ T_wrist_tool", "length_unit":"meter", "left_right":"named channels and model IDs", "root_conversion_roundtrip_max_m":float(np.max(np.abs(g1.model_to_world_position(g1.world_to_model_position(np.eye(3))))-np.eye(3)))},"previous_workspace_registration_reused":False,"source_and_environment_equality":True}
    atomic_json(OUT / "01_registration/COMMON_TASK_WORKSPACE_REGISTRATION_AUDIT.json",registration_audit)
    atomic_text(OUT / "01_registration/COMMON_TASK_WORKSPACE_REGISTRATION_AUDIT.md","# Common task/workspace registration\n\nPASS for coordinate conventions and matched environments. The qualified source-derived identity metric registration is used for both representations. No additional workspace transform is applied. Previous radial/workspace trials are diagnostic only and cannot establish sequential IK feasibility. Active right-handed rotations, meters, explicit WXYZ-to-XYZW conversion, named left/right chains, and wrist/tool multiplication order are audited in the JSON companion. Solver feasibility remains a separate gate.\n")
    train_advanced=sum(r["command_close_advanced"] for r in clock_rows if r["scope"]=="TRAIN40")
    dev_advanced=sum(r["command_close_advanced"] for r in clock_rows if r["scope"]=="DEV35")
    atomic_text(OUT / "01_registration/COMMON_SOURCE_EVENT_CLOCK.md", f"# Shared source and execution clock\n\nAll 40 TRAIN and 35 DEV sources have one shared timestamp-based event clock. Original event names are explicitly aliased to the experiment contract.\n\nThe source detector is preserved. To meet strict close-complete-before-lift, the target command close interval is advanced by the minimum required amount to end one source sample before detected lift. This affects {train_advanced}/40 TRAIN and {dev_advanced}/35 DEV episodes. The original and execution times and every delta are saved. This is an explicit common target timing correction, not an assertion that recorded source timing was already strict.\n\nNatural-start motion samples and source observations are unchanged; no grasp-to-lift splice is introduced. Both representations receive the identical normalized command phase. RIGHT acquisition precedes LEFT release in all entries. Runtime mechanical confirmation and loaded articulation remain unqualified.\n")
    status("RAW_TARGETS_AND_COMMON_SOURCE_EXECUTION_CLOCK_PASS", "POSITION_ONLY_COMMON_IK_SMOKE3", [OUT / "01_registration/RAW_TARGET_AUDIT.json", OUT / "01_registration/RAW_REFERENCE_MANIFEST.json", OUT / "01_registration/COMMON_SOURCE_AND_EXECUTION_EVENT_CLOCK.json"])


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("stage",choices=["recover","references"])
    args=parser.parse_args()
    recover() if args.stage=="recover" else references()
