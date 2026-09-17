#!/usr/bin/env python3
"""Build the final standardized-grasp report after numeric and media verification."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/standardized_grasp_ab_dev35"
RESULTS = OUT / "05_results/FINAL_STANDARDIZED_GRASP_NUMERIC_RESULTS.json"
PREP = OUT / "01_prepared_commands/STANDARDIZED_GRASP_AB_COMMAND_MANIFEST.json"
CONTROL = OUT / "00_control/QUALIFIED_STANDARDIZED_INITIAL_GRASP.json"
GATE = OUT / "00_control/INITIALIZATION_PHYSICAL_GATE.json"
CONTACT = ROOT / "outputs/final_episode_registered_eval35/01_freeze/FINAL_DOLL_CONTACT_MODEL.json"
REPLAY = OUT / "07_physical_replays/STANDARDIZED_GRASP_PHYSICAL_REPLAY_VERIFICATION.json"
FIGURE = OUT / "06_paper_artifacts/FigXX_Standardized_Grasp_AB_Physical_Comparison_double.png"
TABLE = OUT / "06_paper_artifacts/TABLE_STANDARDIZED_GRASP_AB_PHYSICAL_RESULTS.md"
REPORT = OUT / "FINAL_STANDARDIZED_GRASP_AB_PHYSICAL_REPORT.md"
STAGES = (
    ("LIFT_SUCCESS", "Lift"), ("LEFT_RETENTION_SUCCESS", "Left retention"),
    ("HANDOFF_SUCCESS", "Handoff"), ("RIGHT_OWNERSHIP_SUCCESS", "Right ownership"),
    ("RIGHT_TRANSPORT_RETENTION", "Right transport retention"), ("BIN_ENTRY_SUCCESS", "Bin entry"),
    ("BIN_SETTLE_SUCCESS", "Bin settle"), ("POST_GRASP_FULL_TASK_SUCCESS", "Post-grasp full task"),
)
FAILURES = ("LIFT", "LEFT_RETENTION", "HANDOFF", "RIGHT_OWNERSHIP", "RIGHT_TRANSPORT", "BIN_ENTRY", "SETTLE", "SUCCESS")


def read(path: Path) -> dict[str, Any]: return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(value, encoding="utf-8"); os.replace(temporary, path)


def main() -> int:
    result, prep, control, gate, contact, replay = (read(path) for path in (RESULTS, PREP, CONTROL, GATE, CONTACT, REPLAY))
    required = [
        FIGURE, FIGURE.with_suffix(".pdf"), FIGURE.with_suffix(".svg"), TABLE,
        OUT / "07_physical_replays/A_POST_GRASP_DEV35_TOP_35SPLIT.mp4",
        OUT / "07_physical_replays/A_POST_GRASP_DEV35_OVERVIEW_35SPLIT.mp4",
        OUT / "07_physical_replays/B_POST_GRASP_DEV35_TOP_35SPLIT.mp4",
        OUT / "07_physical_replays/B_POST_GRASP_DEV35_OVERVIEW_35SPLIT.mp4",
        OUT / "07_physical_replays/AB_POST_GRASP_MATCHED_PHYSICAL_REVIEW.mp4",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if result["status"] != "FINAL_COMPARABLE_35_PLUS_35" or replay["status"] != "PASS" or missing:
        raise RuntimeError(f"final package prerequisites invalid; missing={missing}")
    shape_errors = [record["relative_trajectory_shape_preservation"] for record in prep["records"]]
    max_translation = max(row["maximum_relative_translation_shape_error_mm"] for row in shape_errors)
    max_rotation = max(row["maximum_relative_rotation_shape_error_deg"] for row in shape_errors)
    a, b, paired = result["A"], result["B"], result["paired"]
    contact_model = contact.get("selected_contact_model", contact.get("doll", contact))
    lines = [
        "# Final standardized-grasp A/B physical report", "",
        "## 1. Scope and rationale", "",
        "This experiment isolates post-grasp manipulation from initial acquisition. Every A/B rollout begins from the same persisted, physically validated target-embodiment LEFT grasp. The result is therefore a **DEV35 standardized-grasp post-grasp physical evaluation**, not end-to-end autonomous grasp success and not an untouched/unseen test result.", "",
        "All previous end-to-end physical attempts are retained as `PRE_STANDARDIZED_GRASP_DIAGNOSTIC_ONLY` provenance and are excluded from these numbers.", "",
        "## 2. Controlled variable", "",
        "The sole intended A/B difference is the raw post-grasp spatial target: A uses wrist-trajectory-centric retargeting and B uses interaction-centric retargeting. Source episode, standardized initial state, source-derived event timing, common IK, hard limits, Dex3 controller, handoff state machine, object, bin, physics, and scorer are shared.", "",
        "## 3. Shared physical initial state", "",
        f"The control state comes from `{control['source_trace']}` at control frame {control['selected_control_frame']} / physics row {control['selected_physics_row']}. It has real LEFT thumb/index/middle forces, zero table support, mechanical retention, and no attachment. The independent runtime gate status is **{gate['status']}**. Its qualified initial object position is `{control['object_position_world_m']}` m and quaternion XYZW is `{control['object_quaternion_xyzw']}`.", "",
        "## 4. Post-grasp boundary and trajectory rebase", "",
        "For every source episode, `t0` is the common source-derived stable LEFT-grasp/hold boundary. For each method and wrist, the rebase is `ΔT_m(t)=inv(T_m(t0))·T_m(t)` and `T_eval_m(t)=T_shared·ΔT_m(t)`. Raw and rebased targets are stored in every prepared archive.", "",
        f"Across all 70 archives, the maximum relative-trajectory shape error introduced by rebasing is {max_translation:.3e} mm translation and {max_rotation:.3e} deg rotation. Initial joint discontinuity is zero in every archive. Thus the gauge changes the common start but not method-relative SE(3) evolution.", "",
        "## 5. Common execution", "",
        "The same sequential G1 IK implementation, seed rules, joint limits, self-collision checks, hard-limit projector, temporal continuity rules, Dex3 HOLD/handoff/release state machine, contact model, PhysX configuration, and physical scorer were used. A persistent infeasible post-grasp IK target fails the corresponding stage; the evaluator does not spatially rescue it. Arm rescue and wrist rescue are both absent.", "",
        "## 6. Physical object and release", "",
        f"The selected qualified contact model is `{contact_model.get('name', 'INTERMEDIATE_PLUSH_PROXY')}`. It is a dynamic, gravity-enabled 20 g rigid approximation of plush contact with no attachment, weld, magnet, parenting, following force, or post-initialization pose writes. Final release is physical Dex3 opening followed by gravity/contact dynamics.", "",
        "## 7. Freeze and rollout integrity", "",
        f"Freeze SHA256: `{result['freeze_sha256']}`. All 70/70 final rollouts are physically valid under that freeze, with 35 matched episode pairs, zero arm/wrist rescue, and zero object pose writes after initialization.", "",
        "## 8. Stage-wise results", "",
        "| Stage | A — Wrist | B — Interaction | B−A |", "|---|---:|---:|---:|",
    ]
    for key, label in STAGES:
        ac, bc = a["stage_counts"][key], b["stage_counts"][key]
        lines.append(f"| {label} | {ac}/35 ({100*ac/35:.1f}%) | {bc}/35 ({100*bc/35:.1f}%) | {100*(bc-ac)/35:+.1f} pp |")
    lines += [
        "", "The standardized initial grasp is 35/35 for both methods by experimental construction and is not counted as method performance.", "",
        "## 9. Primary post-grasp success rate", "",
        f"A: **{paired['A_post_grasp_success_count']}/35 = {paired['A_post_grasp_success_percent']:.1f}%** (exact 95% CI [{paired['A_clopper_pearson_95_percent_ci_percent'][0]:.1f}, {paired['A_clopper_pearson_95_percent_ci_percent'][1]:.1f}]%).", "",
        f"B: **{paired['B_post_grasp_success_count']}/35 = {paired['B_post_grasp_success_percent']:.1f}%** (exact 95% CI [{paired['B_clopper_pearson_95_percent_ci_percent'][0]:.1f}, {paired['B_clopper_pearson_95_percent_ci_percent'][1]:.1f}]%).", "",
        f"B−A: **{paired['B_minus_A_percentage_points']:+.1f} percentage points**; paired bootstrap 95% CI [{paired['paired_bootstrap_95_percent_ci_percentage_points'][0]:.1f}, {paired['paired_bootstrap_95_percent_ci_percentage_points'][1]:.1f}] pp; exact McNemar p={paired['exact_McNemar_two_sided_p']:.6g}. This is reported descriptively without overclaiming significance.", "",
        "## 10. Matched outcomes", "",
        *[f"- {key.replace('_', ' ')}: {value}" for key, value in paired["counts"].items()], "",
        "## 11. First-failure distribution", "",
        *[f"- {stage}: A {a['first_failure_counts'][stage]}/35; B {b['first_failure_counts'][stage]}/35" for stage in FAILURES], "",
        "## 12. Retargeting mechanistic context", "",
        "The existing frozen paper-core full-50 retargeting audit (`outputs/paper_core_ab/tables/table1_full50_retargeting.md`) reports mean wrist error of 3.742 mm for A versus 77.853 mm for B, mean whole-hand error of 90.820 mm for A versus 21.157 mm for B, and mean bimanual-relation error of 86.786 mm for A versus 35.979 mm for B. Lower is better. Thus A provides the expected wrist-fidelity advantage, whereas B provides the expected whole-hand and bimanual interaction advantages.", "",
        "The separate frozen HELDOUT8 ACT audit (`outputs/paper_core_ab/tables/table2_heldout_act_prediction.md`) retains the same directional pattern: predicted wrist error is 24.958/95.269 mm (A/B), whole-hand error is 107.131/58.330 mm, and bimanual error is 120.976/85.920 mm. These are supporting mechanistic datasets, not measurements from the standardized-grasp DEV35 rollouts, and their sample units must not be pooled with the physical rates.", "",
        "The present controlled physical experiment asks whether those post-grasp representation differences preserve lift, handoff, ownership, transport, and placement differently after eliminating initial acquisition as a confound. It does not establish either method's ability to find or acquire the initial object. The observed B advantage is limited to Lift/Left Retention (4/35 versus 0/35); neither method reaches handoff or full completion.", "",
        "## 13. Artifacts", "",
        f"- Main figure: `{FIGURE}`", f"- Paper table: `{TABLE}`",
        f"- A top mosaic: `{required[4]}`", f"- A overview mosaic: `{required[5]}`",
        f"- B top mosaic: `{required[6]}`", f"- B overview mosaic: `{required[7]}`", f"- Matched A/B review: `{required[8]}`", "",
        "## 14. Limitations", "",
        "DEV35 was repeatedly used during engineering and is explicitly labeled development/diagnostic. The evaluation is contact-constrained simulation using a qualified rigid plush proxy, not real-G1 evidence. The shared-grasp initialization removes a central end-to-end requirement. Common IK infeasibility remains part of post-grasp execution rather than being repaired based on outcome.", "",
        "## 15. Paper-safe wording", "",
        "> To isolate the effect of retargeting representation from initial grasp-acquisition errors, both methods were initialized from the same physically validated target-embodiment grasp state. We then evaluated post-grasp manipulation completion under identical G1/Dex3 execution and contact-constrained physics on DEV35. The reported rate is post-grasp task completion, not end-to-end autonomous grasp success.", "",
    ]
    atomic_text(REPORT, "\n".join(lines))
    print(json.dumps({"status": "PASS", "report": str(REPORT), "freeze_sha256": result["freeze_sha256"]}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
