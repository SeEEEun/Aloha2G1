#!/usr/bin/env python3
"""Finalize report, rerun commands, and manifest for source accessory audit v11c."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_accessory_semantics_audit_v11c"
ACTION = ROOT / "evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz"
TIMELINE = ROOT / "configs/episode49_task_timeline.approved.json"
PHONE = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_phone_carrier_audit_v11b/reconstructed_phone_trajectories.npz"
MODEL = Path("/home/jbnu/trossen_arm_mujoco/trossen_arm_mujoco/assets/stationary_ai/stationary_ai.xml")
ACCESSORY_USD = ROOT / "outputs/episode49_source_scene/generated/source_accessory.usda"
SCENE_USD = ROOT / "outputs/episode49_source_scene/generated/source_magsafe_fixed_scene.usda"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def dump(path: Path, payload: Any) -> None:
    write(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def mm(value: float) -> str:
    return f"{float(value) * 1000.0:.3f}"


def video_info(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    result = {
        "decoded_frame_count": int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "width": int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
        "height": int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
    }
    capture.release()
    return result


def main() -> int:
    required_json = {
        "input": OUT / "input_audit.json",
        "asset": OUT / "accessory_asset_frame_audit.json",
        "local": OUT / "accessory_local_frame_attachment_audit.json",
        "gap": OUT / "ring_center_gap_orientation_audit.json",
        "contact": OUT / "right_contact_proxy_audit.json",
        "semantics": OUT / "frame_326_341_semantics_audit.json",
        "correction": OUT / "required_unapplied_accessory_correction.json",
        "decision": OUT / "five_cause_decision.json",
        "constraints": OUT / "constraint_invariants.json",
        "visual": OUT / "visual_evidence_audit.json",
    }
    missing = [str(path) for path in [*required_json.values(), ACTION, TIMELINE, PHONE, MODEL, ACCESSORY_USD, SCENE_USD] if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    values = {name: load(path) for name, path in required_json.items()}
    asset = values["asset"]
    local = values["local"]
    gap = values["gap"]
    contact = values["contact"]
    semantics = values["semantics"]
    correction = values["correction"]
    decision = values["decision"]
    constraints = values["constraints"]
    visual = values["visual"]

    video_path = Path(visual["video"]["path"])
    info = video_info(video_path)
    if info["decoded_frame_count"] != 990 or abs(info["fps"] - 7.5) > 1e-6:
        raise RuntimeError(f"Video invariant failed: {info}")
    if constraints["status"] != "PASS_NO_FORBIDDEN_MUTATION_OR_DOWNSTREAM_GENERATION":
        raise RuntimeError("Constraint audit is not PASS")

    opt_events = contact["event_details"]["optimized_action_aligned"]
    obs_events = contact["event_details"]["observation_state"]
    component_rows: list[str] = []
    for frame in (300, 310, 319, 326, 329, 334, 341, 350):
        row = opt_events[str(frame)]
        queries = row["component_queries"]
        component_rows.append(
            f"| {frame} | {row['aligned_action_index']} | {mm(queries['main_c_ring']['best']['gap_m'])} | "
            f"{mm(queries['support_ring']['best']['gap_m'])} | {mm(queries['hinge']['best']['gap_m'])} | "
            f"{mm(queries['authoritative_all']['best']['gap_m'])} | {queries['authoritative_all']['best']['geom_name']} |"
        )

    ranked_326 = opt_events["326"]["actual_ALOHA_contact_boxes_ranked"]
    ranked_rows = "\n".join(
        f"| {index + 1} | `{row['geom_name']}` | {mm(row['gap_m'])} |"
        for index, row in enumerate(ranked_326)
    )
    best_axis = local["best_frame_326_axis_candidate"]
    best_hinge = local["support_ring_hinge_articulation_diagnostic"]["best_frame_326"]
    parity_326 = contact["optimized_vs_observation_gripper_geometry"]["326"]
    semantics_numeric = semantics["numeric_interpretation"]
    command = semantics["aligned_optimized_command"]
    removal = semantics["right_TCP_removal_motion"]
    source_hashes = {
        "optimized_action": sha256(ACTION),
        "approved_timeline": sha256(TIMELINE),
        "phone_carrier_npz": sha256(PHONE),
        "stationary_ALOHA_model": sha256(MODEL),
        "source_accessory_USD": sha256(ACCESSORY_USD),
        "source_scene_USD": sha256(SCENE_USD),
    }
    if source_hashes["optimized_action"] != constraints["optimized_action_sha256"]:
        raise RuntimeError("optimized_action hash changed")
    if source_hashes["approved_timeline"] != constraints["timeline_sha256"]:
        raise RuntimeError("approved timeline hash changed")
    if source_hashes["phone_carrier_npz"] != constraints["phone_carrier_npz_sha256"]:
        raise RuntimeError("phone carrier input hash changed")

    report = f"""# Episode 49 source accessory geometry/contact audit v11c

1. `CHARGER_ANCHORED_530`은 observed frame 223 이후의 진단 가설로만 유지했고, frames 176–222는 `OBJECT STATE UNRESOLVED DURING GRASP ACQUISITION`으로 유지했다.
2. 실제 Stationary ALOHA의 six named right pad/tip collision OBB, main C-ring, support ring, hinge를 분리 측정했지만 frame 326의 10 mm 접촉 gate를 만족한 후보는 없었다.
3. G1 target/IK, phasewarp, orientation retargeting, Dex3 target motion, physics는 생성하지 않았다.

## 1. 최종 상태

- **{decision['status']}**
- `CHARGER_ANCHORED_530`: source object-state diagnostic only, G1 미승인
- optimized_action task validity를 failure로 확정하지 않음
- 정확한 blocker: charger-anchored carrier를 frame 326까지 역전파한 3-D phone/accessory pose가 raw 영상에서 보이는 오른손과 co-locate하지 않는다. 테스트한 다섯 원인 중 frame 341 semantic 오류만 확정됐고, frame 326의 80.358 mm는 해결되지 않았다.

## 2. 고정 입력과 alignment

- `action_to_observation_lag_frames = 7`
- `action_sample_for_observed_frame = observed_frame - 7` (frames 0–6만 sample 0 pre-command hold)
- frame 326 → optimized_action[319]
- frame 329 → optimized_action[322]
- frame 341 → optimized_action[334]
- optimized_action SHA-256: `{source_hashes['optimized_action']}`
- approved timeline SHA-256: `{source_hashes['approved_timeline']}`
- phone carrier diagnostic NPZ SHA-256: `{source_hashes['phone_carrier_npz']}`

## 3. Accessory local axes, centers, attachment

- Main C-ring: local X–Z plane, normal/thickness `+Y`; center `[0,0,0]` m; inner/outer radius `22.5/27.5 mm`.
- Main opening: `36 deg`, center `-90 deg`, local opening direction `-Z`; material interval `[-72,252] deg`.
- Ring-frame basis: x=`accessory +X`, y=`accessory -Z` (opening), z=`accessory +Y` (phone rear outward).
- Support ring: local X–Y plane, normal `+Z`; center `[0,30.0,-33.2] mm`.
- Hinge: local center `[0,1.5,-24.0] mm`, axis `+/-X`.
- Phone→accessory root: identity rotation, translation `[0,6.425,0] mm` = phone half-thickness `3.975` + clearance `0.700` + main half-depth `1.750` mm.
- Generated composite USD transform error: `{asset['generated_USD_parity']['translation_max_abs_error_m']:.3e} m`; first main-ring mesh point error: `{asset['generated_USD_parity']['first_point_max_abs_error_m']:.3e} m`; no authored accessory rotation: `{asset['generated_USD_parity']['no_authored_accessory_rotation']}`.

## 4. Frame/attachment alternatives (diagnostic only)

- Simple attachment-origin variants: best was `DOUBLE_MAIN_HALF_DEPTH_WRONG`, frame-326 gap `{mm(local['attachment_origin_candidates']['DOUBLE_MAIN_HALF_DEPTH_WRONG']['events']['326']['gap_m'])} mm`.
- All 24 proper signed-axis permutations: best `{best_axis['candidate_id']}`, frame-326 gap `{mm(best_axis['events']['326']['gap_m'])} mm`.
- Support-ring hinge sweep −180…+180 deg: best `{best_hinge['support_ring_rotation_about_hinge_x_degrees']:+d} deg`, frame-326 gap `{mm(best_hinge['events']['326']['gap_m'])} mm`; frame-329 gap `{mm(best_hinge['events']['329']['gap_m'])} mm`.
- None reached the 10 mm gate; no diagnostic transform/angle was adopted.
- Closing the current nearest pair would require an unapplied rigid translation of `{mm(correction['326']['norm_m'])} mm`, local vector `{np.asarray(correction['326']['translation_to_close_current_nearest_pair_accessory_local_m']) * 1000.0} mm`. This correction was **not** applied.

## 5. Ring gap and component distances

- Authoritative gapped asset frame-326 gap: `{mm(gap['authoritative_frame_326_gap_m'])} mm`.
- Complete-main-ring lower bound: `{mm(gap['complete_main_ring_lower_bound_frame_326_m'])} mm`.
- Best 5-deg gap-orientation sweep: `{gap['best_gap_orientation_frame_326']['gap_center_degrees']:+d} deg`, `{mm(gap['best_gap_orientation_frame_326']['events']['326']['gap_m'])} mm`.
- Therefore gap direction cannot explain the large miss.

| Observed frame | Action index | Main C-ring mm | Support mm | Hinge mm | Best actual asset mm | Nearest ALOHA OBB |
|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(component_rows)}

Frame 341 values above are a **counterfactual still-attached boundary diagnostic**; `accessory_removed` 이후 contact validity 값이 아니다.

## 6. Right TCP versus actual contact proxy

- TCP point is not used as contact geometry. Frame 326 TCP-to-surface: `{mm(opt_events['326']['right_TCP_to_authoritative_surface_m'])} mm`.
- Actual contact proxy: six named Stationary ALOHA right gripper pad/tip collision OBBs.

| Rank | Actual ALOHA contact OBB | Frame-326 gap mm |
|---:|---|---:|
{ranked_rows}

- Best optimized-action OBB at frame 326: `{ranked_326[0]['geom_name']}`, `{mm(ranked_326[0]['gap_m'])} mm`.
- Same object hypothesis with observation.state[326]: `{mm(obs_events['326']['component_queries']['authoritative_all']['best']['gap_m'])} mm`.
- Optimized vs observation named-proxy centers at frame 326: mean `{mm(parity_326['mean_difference_m'])} mm`, max `{mm(parity_326['max_difference_m'])} mm`; TCP difference `{mm(parity_326['right_TCP_position_difference_m'])} mm`.
- Dex3 world distance was intentionally not fabricated: there is no authorized G1/Dex3 pose. Static semantic role only is retained (`right C = middle insertion/hook`).

## 7. Frame 326/341 semantics

- Right gripper maximum opening: observed frame `{command['maximum_open_frame']}`, `{command['maximum_open_value_m']*1000.0:.3f} mm`.
- Closing crosses 10% open at frame `{command['first_at_or_below_10pct_open_frame']}`; by frame 326 command is `{command['event_values_m']['326']*1000.0:.3f} mm` (near closed).
- Approved frame 326 remains `right_accessory_grasp_start`; it is not redefined as actuator-close onset or a rigid carrier lock.
- Frame 326→341 optimized right TCP displacement: `{mm(removal['frame_326_to_341_displacement_m'])} mm`.
- Frame 341 is `accessory_removed`; keeping the accessory attached to the phone there is semantically invalid. Thus `{mm(semantics_numeric['frame_341_counterfactual_attached_gap_m'])} mm` must not be used as a grasp-failure metric.
- Within frames 280–370 the diagnostic attached-model minimum is `{mm(semantics_numeric['minimum_gap_frames_280_370_m'])} mm` at observed frame `{semantics_numeric['minimum_gap_observed_frame']}`; it never enters 10 mm.

## 8. Five-possibility decision

1. **Wrong local frame/attachment:** active builder/layout/USD parity passes. Tested origin errors, 24 axis frames, and one-axis support hinge sweep all remain >10 mm. Rejected for the tested variants.
2. **Wrong ring center/gap orientation:** complete-ring lower bound is still `{mm(gap['complete_main_ring_lower_bound_frame_326_m'])} mm`; rejected as the large-gap cause.
3. **Wrong contact proxy:** TCP proxy was replaced by all six actual ALOHA collision OBBs; best remains `{mm(ranked_326[0]['gap_m'])} mm`. Rejected.
4. **Wrong frame semantics:** partial cause. Frame 341 still-attached comparison is invalid; frame 326 is a visual acquisition boundary, but semantics alone does not close its gap.
5. **optimized_action does not reach:** not established. Aligned optimized vs observation approach/removal direction cosines are `0.999594/0.982470`, and frame-326 proxy-center difference is only `{mm(parity_326['mean_difference_m'])} mm`; both disagree with the same reconstructed object pose.

Conclusion: none of 1–3 or 5 explains frame 326 under the frozen hypothesis; 4 removes the frame-341 false failure only. The remaining blocker is the asserted 3-D accessory object state produced by backward use of the diagnostic phone carrier—not evidence that optimized_action is invalid.

## 9. Visual evidence

- 990-frame 4-panel video (7.5 fps): `{visual['video']['path']}`
  - SHA-256: `{visual['video']['sha256']}`
  - Panels: raw cam_high | actual ALOHA + authoritative asset | six-OBB X-ray | five-cause dashboard
- Raw-vs-proxy contact sheet: `{visual['images']['accessory_raw_vs_proxy_contact_sheet.png']['path']}`
- Event contact sheet: `{visual['images']['accessory_event_contact_sheet.png']['path']}`
- Distance timeseries: `{visual['images']['accessory_gap_timeseries.png']['path']}`
- Gap/axis/hinge sweeps: `{visual['images']['ring_gap_axis_hinge_sweeps.png']['path']}`
- Five-cause matrix: `{visual['images']['accessory_five_cause_matrix.png']['path']}`

## 10. Single next user decision

{decision['single_next_user_decision']}

Do **not** approve G1 retargeting from this result.

THE APPROVED ALOHA ACTION, TIMELINE, AND SEVEN-FRAME ALIGNMENT WERE NOT CHANGED
CHARGER_ANCHORED_530 REMAINED A DIAGNOSTIC PHONE-CARRIER HYPOTHESIS ONLY
OBSERVED FRAMES 176-222 REMAINED OBJECT STATE UNRESOLVED DURING GRASP ACQUISITION
NO G1 TARGET, IK, PHASEWARP, OR ORIENTATION TARGET WAS GENERATED
NO DEX3 TARGET MOTION OR PHYSICS WAS GENERATED
SIMULATION AND SOURCE OBJECT-RELATION DIAGNOSTICS ONLY
"""
    write(OUT / "report.md", report)

    commands = f"""#!/usr/bin/env bash
set -euo pipefail
cd {ROOT}

PY=/home/jbnu/miniconda3/envs/lerobot-smolvla/bin/python

"$PY" tools/audit_episode49_accessory_semantics_v11c.py
MUJOCO_GL=egl "$PY" tools/render_episode49_accessory_semantics_v11c.py
"$PY" tools/finalize_episode49_accessory_semantics_v11c.py

# Visual review only; no robot publisher or physics:
ffplay -loop 0 {video_path}
"""
    write(OUT / "commands.sh", commands)
    os.chmod(OUT / "commands.sh", 0o755)

    forbidden_name_fragments = ("g1_target", "g1_ik", "arm_trajectory", "phasewarp", "orientation_target", "dex3_target")
    forbidden_files = [path.name for path in OUT.iterdir() if any(fragment in path.name.lower() for fragment in forbidden_name_fragments)]
    if forbidden_files:
        raise RuntimeError(f"Forbidden downstream outputs present: {forbidden_files}")

    output_files = sorted(path for path in OUT.iterdir() if path.is_file() and path.name != "run_manifest.json")
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": decision["status"],
        "scope": "EPISODE_49_SOURCE_ACCESSORY_GEOMETRY_CONTACT_SEMANTICS_AUDIT_ONLY",
        "source_hashes": source_hashes,
        "video_validation": {**info, "sha256": sha256(video_path), "metadata_status": visual["video"]["metadata"]["status"]},
        "carrier_policy": {
            "frames_176_222": "OBJECT STATE UNRESOLVED DURING GRASP ACQUISITION",
            "frames_223_onward": "CHARGER_ANCHORED_530 diagnostic only",
            "approved_for_G1": False,
        },
        "five_cause_summary": {name: row["classification"] for name, row in decision["five_possibilities"].items()},
        "constraints": constraints,
        "forbidden_output_filename_matches": forbidden_files,
        "outputs": {
            path.name: {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in output_files
        },
        "manifest_self_hash_recorded": False,
    }
    dump(OUT / "run_manifest.json", manifest)
    print(json.dumps({
        "status": decision["status"],
        "report": str((OUT / "report.md").resolve()),
        "commands": str((OUT / "commands.sh").resolve()),
        "manifest": str((OUT / "run_manifest.json").resolve()),
        "video": str(video_path.resolve()),
        "decoded_frames": info["decoded_frame_count"],
        "forbidden_outputs": forbidden_files,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
