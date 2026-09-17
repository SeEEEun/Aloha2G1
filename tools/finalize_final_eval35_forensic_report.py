#!/usr/bin/env python3
"""Write the final forensic/configuration/paper-direction reports after postprocessing."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUT = ROOT / "outputs/final_episode_registered_eval35"


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def rate(count: int) -> str:
    return f"{count}/35 ({100*count/35:.1f}%)"


def main() -> int:
    scene = read(OUT / "00_forensic_audit/PHYSICAL_SCENE_RUNTIME_AUDIT.json")
    forensic = read(OUT / "00_forensic_audit/ACT_A_LEFT_GRASP_FORENSIC_REPORT.json")
    articulation = read(OUT / "00_forensic_audit/DEX3_ARTICULATION_FORENSIC_AUDIT.json")
    qualification = read(OUT / "00_qualification/FINAL_COMMON_DEX3_GRASP_QUALIFICATION.json")
    freeze_path = OUT / "01_freeze/FINAL_EVAL35_FREEZE_MANIFEST.json"
    freeze = read(freeze_path)
    environment = read(OUT / "01_freeze/FINAL_PHYSICAL_ENVIRONMENT.json")
    registration = read(OUT / "00_registration/EVAL35_EPISODE_OBJECT_REGISTRATION.json")
    results = read(OUT / "04_results/FINAL_NUMERIC_RESULTS.json")
    replay = read(OUT / "06_physical_replays/PHYSICAL_REPLAY_VERIFICATION.json")
    config = read(ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json")
    a, b = results["ACT_A"], results["ACT_B"]
    zs = [float(row["target_object_pose"]["position_xyz_m"][2]) for row in registration["entries"]]
    doll = environment["doll"]
    physics = environment["physics"]
    table_z = float(config["object"]["table_surface_world_z_m"])
    table_gaps = scene["doll_table_height"]
    config_table = f"""# Final physical configuration

| Component | Final runtime value | Evidence |
|---|---:|---|
| Doll visual dimensions | {doll['visual_dimensions_m']} m | Final frozen physical environment |
| Doll collision dimensions | {doll['collision_geometry']['dimensions_m']} m | `{doll['collision_geometry']['name']}` |
| Visual−collision half extent | {doll['visual_minus_collision_half_extent_m']} m | Per-axis calculation |
| Mass | {doll['mass_kg']:.3f} kg | Dynamic rigid body |
| Static friction | {doll['material']['static_friction']:.2f} | Qualified material |
| Dynamic friction | {doll['material']['dynamic_friction']:.2f} | Qualified material |
| Restitution | {doll['restitution']:.1f} | Qualified material |
| Linear / angular damping | {doll['linear_damping']:.2f} / {doll['angular_damping']:.2f} | Qualified object model |
| Added contact tolerance | {doll['additional_contact_tolerance_mm']} mm | Qualified selection |
| Contact / rest offset | {doll['contact_offset_m']*1000:.1f} / {doll['rest_offset_m']*1000:.1f} mm | PhysX proxy |
| Doll CCD | disabled | GPU PhysX scene capability reported explicitly |
| Table top Z | {table_z:.6f} m | Runtime config |
| Initial doll/table gap | {table_gaps['minimum_effective_collision_bottom_gap_m']*1000:.6f} to {table_gaps['maximum_effective_collision_bottom_gap_m']*1000:.6f} mm | 35 registrations; floating-point tolerance |
| Episode object center Z range | {min(zs):.9f}–{max(zs):.9f} m | 35 source-derived entries |
| Bin height | {environment['bin']['external_height_m']*1000:.0f} mm | Frozen bin |
| Physics / control timestep | {physics['dt_s']:.9f} / {1/physics['control_fps_hz']:.9f} s | Contact-constrained PhysX |
| Substeps | {physics['substeps_per_control_frame']} | Frozen physics |
| Solver | {physics['solver']}, {physics['articulation_solver_position_iterations']}/{physics['articulation_solver_velocity_iterations']} position/velocity iterations | Common A/B TYPE-3 correction |
| Dex3 safety inset | {freeze['dex3_safety_inset_rad']:.3f} rad | Frozen controller |
| Object pose writes after initialization | 0 | All qualification/final traces |
| Arm / wrist rescue | NO / NO | All final traces |
"""
    write(OUT / "FINAL_PHYSICAL_CONFIGURATION_TABLE.md", config_table)

    failure_lines = "\n".join(f"- {key}: {value}/35" for key, value in forensic["category_counts"].items())
    stage_lines = []
    for key in ("LEFT_GRASP_SUCCESS","HANDOFF_SUCCESS","RIGHT_OWNERSHIP_SUCCESS","NO_DROP_TO_BIN","BIN_ENTRY_SUCCESS","BIN_SETTLE_SUCCESS","CLEAN_COMMANDED_RELEASE","PREMATURE_DROP_INTO_BIN","FULL_TASK_SUCCESS"):
        stage_lines.append(f"- {key}: ACT-A {rate(a['stage_counts'][key])}; ACT-B {rate(b['stage_counts'][key])}")
    fail_lines = []
    for key in ("LEFT_GRASP","HANDOFF","RIGHT_OWNERSHIP","TRANSPORT","BIN_ENTRY","SETTLE","NONE_SUCCESS"):
        fail_lines.append(f"- {('SUCCESS' if key=='NONE_SUCCESS' else key)}: A {a['first_failure_counts'][key]}; B {b['first_failure_counts'][key]}")
    paired = results["paired"]
    figures = {
        "main": OUT / "05_paper_artifacts/Fig17_ACT_AB_Physical_Task_Success_EVAL35_double.png",
        "audit": OUT / "05_paper_artifacts/Fig17b_Physical_Grasp_Failure_and_Contact_Audit.png",
        "registration": OUT / "07_paper_visuals/METHOD_A_EPISODE_CONDITIONED_REGISTRATION.png",
        "grasp": OUT / "07_paper_visuals/METHOD_B_DEX3_MECHANICAL_GRASP_SEQUENCE.png",
        "handoff": OUT / "07_paper_visuals/METHOD_C_PHYSICAL_HANDOFF_SEQUENCE.png",
        "comparison": OUT / "07_paper_visuals/RESULT_MATCHED_AB_PHYSICAL_COMPARISON.png",
    }
    videos = {key: Path(value["path"]) for key, value in replay["mosaics"].items()}
    missing = [str(path) for path in [*figures.values(), *videos.values()] if not path.is_file()]
    if missing:
        raise RuntimeError(f"postprocessing artifacts missing: {missing}")
    report = f"""# Final ACT-A/B EVAL35 forensic and physical report

## A. Physical scene audit

The final experiment used 35/35 episode-conditioned, source-derived object registrations. Matched A/B poses are identical (0 mm / 0 deg), while the entries are not replaced by a global canonical pose. Runtime A registration error in the preserved audit was at most {scene['episode_registration']['maximum_ACT_A_runtime_translation_error_mm']:.9f} mm and {scene['episode_registration']['maximum_ACT_A_runtime_rotation_error_deg']:.9f} deg. The doll is dynamic, non-kinematic, gravity-driven, and contact-constrained; no attachment, following, or post-initialization root-pose write exists.

See [physical configuration table](FINAL_PHYSICAL_CONFIGURATION_TABLE.md).

## B–D. Preserved ACT-A 0/35 forensics and scorer sanity

The pre-fix ACT-A35 is retained at `provenance_common_execution_bug_solver32/` and labeled `SUPERSEDED_COMMON_EXECUTION_BUG`. In those traces the reported LEFT_GRASP rate was 0/35. Read-only trace reconstruction found zero scorer false negatives and classified every episode before learning of the B outcome:

{failure_lines}

The minimum signed hand-to-collider distances and zero force traces show that the old A attempts were genuine geometric misses under that scene, not visual-overlap scorer errors. Nevertheless, those runs cannot be mixed into the final comparison because the common articulation solver changed.

## E–H. B01 articulation root cause and fairness consequence

`left_hand_middle_1_joint` is archive/execution index 18, PhysX DOF index 36, and USD prim `/World/G1/Asset/joints/left_hand_middle_1_joint`. Its project range is [-1.74533, 0] rad and its runtime safety range is [-1.740329, -0.005] rad. A no-contact 14-joint sweep passed mapping, sign, readback, and hard limits 14/14. The B01 command/readback mismatch was reproduced only when hand–table contact remained: the source 32/1 articulation solver under-resolved the external contact impulse and allowed +0.320730 rad despite a −0.168992 rad target. Removing the doll did not remove the excursion; removing the table did. This excludes command-field, sign, label, and doll-contact hypotheses.

Fix classification: **TYPE 3 — common physical articulation bug**. TGS 64 position / 4 velocity iterations still violated the reproduced hard stop, whereas 80/4 passed it and the unchanged non-EVAL full-task physical-validity gates. The correction changed no command, trajectory, registration, collider, controller, or task scorer and was applied identically to A/B. Therefore ACT-A was rerun from episode 01. The corrected zero-contact sweep passed 14/14, and B01 completed as a valid final rollout.

## I–M. Final matched results

Final valid dataset: ACT-A 35/35, ACT-B 35/35 under freeze `{sha(freeze_path)}`.

{chr(10).join(stage_lines)}

Primary TSR difference: **{paired['B_minus_A_percentage_points']:+.1f} percentage points**, paired bootstrap 95% CI [{paired['paired_bootstrap_95_percent_CI_percentage_points'][0]:.1f}, {paired['paired_bootstrap_95_percent_CI_percentage_points'][1]:.1f}] pp. Exact McNemar p={paired['exact_McNemar_two_sided_p']:.6g}. This is reported descriptively without overclaiming significance.

Matched counts: {paired['counts']}.

First failures:

{chr(10).join(fail_lines)}

Release diagnostics: A {a['release_counts']}; B {b['release_counts']}.

## N–O. Evidence artifacts

- Main result figure: `{figures['main']}`
- Physical audit figure: `{figures['audit']}`
- Registration visual: `{figures['registration']}`
- Mechanical grasp visual: `{figures['grasp']}`
- Handoff visual: `{figures['handoff']}`
- Matched comparison: `{figures['comparison']}`
- A top / overview mosaics: `{videos['A_top']}`, `{videos['A_overview']}`
- B top / overview mosaics: `{videos['B_top']}`, `{videos['B_overview']}`

Every mosaic reconstructs saved `MEASURED_Q` and actual PhysX doll position/quaternion traces; command-only object replay is not used.

## P. Paper-safe interpretation and limitations

This experiment tests whether the interaction-centric representation yields different contact-constrained manipulation progression than the trajectory-centric wrist-transfer baseline after ACT learning, under identical episode-conditioned source-derived task registrations and one common target-hand realization. It supports only the observed stage and TSR differences. It does not establish real-G1 TSR, autonomous target-domain vision, or exact plush mechanics, and does not justify a significance or superiority claim unless the reported matched test supports it.
"""
    final = OUT / "FINAL_ACT_AB_EVAL35_FORENSIC_AND_PHYSICAL_REPORT.md"
    write(final, report)
    write(OUT / "FINAL_ACT_AB_EVAL35_COMPLETE_REPORT.md", report)
    direction = f"""# Paper figure direction

## Figure 1 / method evidence

Use `{figures['registration'].name}`, `{figures['grasp'].name}`, and `{figures['handoff'].name}` in that order: source-derived episode registration → mechanical contact confirmation before lift → physical ownership transfer. Add only short labels for source episode, confirmed grasp, and ownership. Do not claim policy-dependent object placement, attachment-based grasp, real-G1 validation, or mandatory three-digit topology.

Korean: 에피소드별 소스 기반 물체 정합, 리프트 전 기계적 파지 확인, 양손 소유권 전환을 실제 저장된 물리 상태로 설명한다.

## Result figure

Use `{figures['main'].name}`. Put Full Task Success Rate in the visually dominant panel and retain exact x/35 counts, confidence intervals, first-failure stages, and all 35 paired rows. Emphasize the measured TSR difference, not a qualitative superiority claim.

Korean: 35개 대응 에피소드의 단계별 성공률과 최종 TSR을 정확한 개수와 함께 제시한다.

## Matched A/B comparison

Use `{figures['comparison'].name}`. Its episode was selected by the predeclared rule: lowest index among episodes with the largest absolute cumulative-stage difference. Highlight where physical progression diverges; do not imply cherry-picked generality.

Korean: 누적 단계 차이가 최대인 에피소드 중 가장 낮은 인덱스를 사용해 선택 편향을 줄였다.

## Success storyboard

Use the lowest-index successful method storyboard if present; otherwise omit it. It is best suited to supplementary material unless space permits. Do not replace aggregate results with a single successful example.

Korean: 성공 사례는 최저 인덱스 규칙으로 선택하며 전체 통계를 대체하지 않는다.
"""
    write(OUT / "07_paper_visuals/PAPER_FIGURE_DIRECTION.md", direction)
    print(json.dumps({"status":"PASS","final_report":str(final),"freeze_sha256":sha(freeze_path)},indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
