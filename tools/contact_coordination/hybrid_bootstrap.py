"""Recover exact source identities and predeclare small TRAIN development set."""
import numpy as np
from . import bootstrap
from .io import ROOT, atomic_json, atomic_text, read, record


def run(out, resume=False):
    result = bootstrap.run(out, resume)
    entries = read(out/'bootstrap/SPLITS.json')['entries']
    train = sorted((r for r in entries if r['TRAIN40']), key=lambda r: r['source_recording_id'])
    dev = sorted((r for r in entries if r['DEV35_DIAGNOSTIC35']), key=lambda r: r['dev_index'])
    assert len(train) == 40 and len(dev) == 35
    assert set(r['source_recording_id'] for r in train).isdisjoint(r['source_recording_id'] for r in dev)
    indices = np.linspace(0,34,10).round().astype(int).tolist()
    assert len(set(indices)) == 10
    selection = dict(prototype_source_id=train[0]['source_recording_id'],
        train_source_ids=[train[i]['source_recording_id'] for i in (0,13,26)],
        train_rule='lexicographically ordered recording IDs at positions 0,13,26; prototype position 0',
        ablation_dev_positions=indices, ablation_source_ids=[dev[i]['source_recording_id'] for i in indices],
        dev_source_ids=[r['source_recording_id'] for r in dev], selected_before_hybrid_outcomes=True,
        scheduled_primary_instances=70, scheduled_ablation_instances=10, frozen=False)
    path = out/'bootstrap/SELECTION.json'
    if path.exists():
        assert read(path) == selection, 'selection drift'
    atomic_json(path, selection)
    components = [
        ('TRAIN40/DEV35 and replacements','outputs/final_single_variable_ab/00_contract/AUTHORITATIVE_SPLIT_MANIFEST.json','75 source/reference hashes; 40 TRAIN,35 DEV disjoint','reuse IDs, never index assumptions'),
        ('raw FK/reference lineage','outputs/final_single_variable_ab/01_registration/RAW_REFERENCE_MANIFEST.json','source ID checked against each NPZ and raw parquet','reuse arrays'),
        ('corrected events','outputs/final_single_variable_ab/01_registration/COMMON_SOURCE_AND_EXECUTION_EVENT_CLOCK.json','same source clock for both representations','preserve event order; replace dense execution clock'),
        ('source FK','tools/doll_handoff_retargeting/models.py','named ALOHA model and task TCP recorded in common config','reuse FK and G1 transforms'),
        ('functional tool mapping','outputs/single_variable_ab_reset/shared_pipeline_train_smoke_v4/config/tool_frame_report.json','fixed wrist-to-tool transform; XYZW and metres','reuse in source reference'),
        ('initial object registration','outputs/doll_handoff_retargeting/scene_recalibration.json','per-recording pre-motion images and homography','reuse per-recording pose; infer z explicitly'),
        ('source yaw inference','tools/build_eval35_episode_object_registration.py','existing image PCA; documented 180-degree ambiguity','reuse pure function, no batch side effects'),
        ('interaction representation','tools/doll_handoff_retargeting/retarget.py','existing object/inter-hand quantities and raw references','thin phase/contact wrapper; dense arrays remain priors'),
        ('common G1 config/models','outputs/single_variable_ab_reset/shared_pipeline_train_smoke_v4/config/common_config.json','XML/USD hashes and named arm/Dex3 mappings','reuse assets; no legacy dense solver orchestration'),
        ('joint limits/order','outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json','28 named hard limits','reuse exact values and name permutation'),
        ('P14 hands/object/physics','configs/dex3_simple_graspable_doll_grasp_v2.json','current compatible standalone traces; 20g intermediate rigid proxy','reuse exact material/geometry/controller settings'),
        ('150mm bin','outputs/final_single_variable_ab/01_registration/RECOVERED_PHYSICAL_SCENE_AND_CONTROLLER_PROVENANCE.json','150mm height,3mm rim; authoritative environment','reuse dimensions; no scene search'),
        ('measured controller','tools/direct_physical_execution_layer.py','opposing candidate before lift; receiver before giver release','reuse common measured control'),
        ('PhysX runner/logger','tools/run_direct_physical_execution_isaac.py','same dependency hashes as compatible standalone controls','explicit read-only checkpoint adapter'),
        ('runtime collider inventory','tools/reconciled_ab/runtime_geometry.py','read-only USD API/filter capture','reuse; offline/runtime mismatch remains a gate'),
        ('common scorer','tools/finalize_common_dex3_grasp_qualification.py','raw numerical traces, measured contact/support and settle','reuse functions; separate candidate from physical ownership'),
        ('SE3 IK backend','tools/build_doll_handoff_graspable_proxy_v2_primitives.py','existing scipy least_squares, FK full SE3 and named bounds','reuse installed backend/residual formulation on phase goals'),
        ('retiming bounds','outputs/policy_execution_stability_review/jerk_limited_otg/common_g1_28d_ruckig_config.json','per-joint velocity ceilings; acceleration is empirical calibration, not manufacturer-rated','reuse calibration transparently; no invented authoritative acceleration'),
        ('measured replay renderer','tools/render_episode_registered_physical_evidence.py','must verify available entry point before invoking','reuse measured-state rendering after numerical evidence')]
    rows=[]
    for component, relative, evidence, action in components:
        p=ROOT/relative
        # Resolve the existing renderer name without guessing a nonexistent path.
        if component=='measured replay renderer' and not p.exists():
            p=ROOT/'tools/render_final_episode_registered_physical_evidence.py'
        rows.append(dict(component=component, file=record(p), compatibility_evidence=evidence, reuse_action=action))
    atomic_json(out/'bootstrap/REUSE_COMPONENTS.json', rows)
    atomic_text(out/'REUSE_MAP.md', '# Existing-asset reuse\n\nExact versions are SHA-256-bound in `bootstrap/REUSE_COMPONENTS.json`. Compatibility statements distinguish current checks from pending gates.\n\n'
        +'| Component | Exact file | Compatibility evidence | Reuse / minimal change |\n|---|---|---|---|\n'
        +'\n'.join(f"| {r['component']} | {r['file']['path']} | {r['compatibility_evidence']} | {r['reuse_action']} |" for r in rows)
        +'\n\nRobot model dependencies and raw recording replacements remain in the authoritative manifests. The rigid surrogate is not validated soft-body physics. Acceleration limits recovered here are empirical manipulation bounds; no manufacturer acceleration rating was found.\n')
    return dict(result, selection=selection)
