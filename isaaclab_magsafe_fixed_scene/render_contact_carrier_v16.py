#!/usr/bin/env python3
"""Run the verified v15 zero-physics Isaac/Fabric path for a v16 trajectory.

The v15 renderer is treated as the synchronization implementation of record.
This wrapper substitutes only artifact names and labels in memory, preserving
the articulation write/readback/link-sync/render ordering byte-for-byte.
"""
from __future__ import annotations

from pathlib import Path


HERE = Path(__file__).resolve().parent
source_path = HERE / "render_orientation_dex3_v15.py"
source = source_path.read_text(encoding="utf-8")
replacements = (
    (
        'parser.add_argument("--height", type=int, default=405)\nAppLauncher.add_app_launcher_args(parser)',
        'parser.add_argument("--height", type=int, default=405)\n'
        'parser.add_argument("--camera-pass", choices=("all", "overview", "side", "left_close", "right_close", "charger_close"), default="all")\n'
        'AppLauncher.add_app_launcher_args(parser)',
    ),
    ("current_layout_ep49_orientation_dex3_v15", "current_layout_ep49_contact_carrier_v16"),
    ("full_arm_dex3_trajectory.npz", "arm_dex3_coupled_trajectory.npz"),
    ("ALOHA_PRIMARY_EP49_ORIENTATION_DEX3_V15", "ALOHA_PRIMARY_CONTACT_CARRIER_V16"),
    ("v15_isaaclab_kinematic_replay.usda", "v16_isaaclab_kinematic_replay.usda"),
    ("v15_g1_dex3_robot_only_overview.mp4", "v16_g1_dex3_robot_only_overview.mp4"),
    ("v15_g1_dex3_robot_only_side.mp4", "v16_g1_dex3_robot_only_side.mp4"),
    ("v15_left_phone_grasp_closeup.mp4", "v16_left_phone_carrier_closeup.mp4"),
    ("v15_right_accessory_hook_closeup.mp4", "v16_right_hook_carrier_closeup.mp4"),
    ("v15_charger_placement_closeup.mp4", "v16_charger_closeup.mp4"),
    ("v15_object_follow_overview.mp4", "v16_object_follow_overview.mp4"),
    ("v15_object_follow_side.mp4", "v16_object_follow_side.mp4"),
    ("v15_semantic_keyframe_overview.png", "v16_semantic_keyframe_overview.png"),
    ("v15_left_ab_contact_closeup.png", "v16_left_ab_contact_closeup.png"),
    ("v15_right_c_contact_closeup.png", "v16_right_c_contact_closeup.png"),
    ("v15_charger_alignment_closeup.png", "v16_charger_alignment_closeup.png"),
    ("v15_orientation_axes_contact_sheet.png", "v16_orientation_axes_contact_sheet.png"),
    ("V15 SEMANTIC KEYFRAMES - FAILED DIAGNOSTIC", "V16 SEMANTIC CONTACT-CARRIER KEYFRAMES"),
    ("LEFT A+B CONTACT - FAILED DIAGNOSTIC", "LEFT A+B PINCH CARRIER"),
    ("RIGHT C CONTACT - FAILED DIAGNOSTIC", "RIGHT C HOOK CARRIER"),
    ("CHARGER ALIGNMENT - FAILED DIAGNOSTIC", "V16 CHARGER ALIGNMENT"),
    ("Episode49 v15", "Episode49 v16"),
    ("EP49 v15", "EP49 v16"),
    (
        'numeric_gates["numeric_pass"]',
        'numeric_gates.get("numeric_pass", numeric_gates["numerical_pass"])',
    ),
    (
        '    cameras = {}\n    if not args.gui:',
        '    if not args.gui and args.camera_pass != "all":\n'
        '        camera_poses = {args.camera_pass: camera_poses[args.camera_pass]}\n'
        '    cameras = {}\n'
        '    if not args.gui:',
    ),
    (
        '    writers = {}\n    for label, (_, output) in video_specs.items():',
        '    if args.camera_pass != "all":\n'
        '        video_specs = {label: value for label, value in video_specs.items() if value[0] == args.camera_pass}\n'
        '    writers = {}\n'
        '    for label, (_, output) in video_specs.items():',
    ),
    (
        '    contact_sheet("overview", OUT / "v16_semantic_keyframe_overview.png", "V16 SEMANTIC CONTACT-CARRIER KEYFRAMES")\n'
        '    contact_sheet("left_close", OUT / "v16_left_ab_contact_closeup.png", "LEFT A+B PINCH CARRIER")\n'
        '    contact_sheet("right_close", OUT / "v16_right_c_contact_closeup.png", "RIGHT C HOOK CARRIER")\n'
        '    contact_sheet("charger_close", OUT / "v16_charger_alignment_closeup.png", "V16 CHARGER ALIGNMENT")\n'
        '    contact_sheet("overview", OUT / "v16_orientation_axes_contact_sheet.png", "SOURCE-RELATIVE ORIENTATION STAGES")',
        '    sheet_specs = (("overview", OUT / "v16_semantic_keyframe_overview.png", "V16 SEMANTIC CONTACT-CARRIER KEYFRAMES"),\n'
        '                   ("left_close", OUT / "v16_left_ab_contact_closeup.png", "LEFT A+B PINCH CARRIER"),\n'
        '                   ("right_close", OUT / "v16_right_c_contact_closeup.png", "RIGHT C HOOK CARRIER"),\n'
        '                   ("charger_close", OUT / "v16_charger_alignment_closeup.png", "V16 CHARGER ALIGNMENT"),\n'
        '                   ("overview", OUT / "v16_orientation_axes_contact_sheet.png", "SOURCE-RELATIVE ORIENTATION STAGES"))\n'
        '    for camera_name, output, title in sheet_specs:\n'
        '        if camera_name in cameras:\n'
        '            contact_sheet(camera_name, output, title)',
    ),
    (
        '    dump(OUT / "isaaclab_kinematic_validation.json", result)',
        '    result["camera_pass"] = args.camera_pass\n'
        '    audit_name = "isaaclab_kinematic_validation.json" if args.camera_pass == "all" else f"isaaclab_kinematic_validation_{args.camera_pass}.json"\n'
        '    dump(OUT / audit_name, result)',
    ),
    (
        '"FAILED DIAGNOSTIC CANDIDATE | NOT APPROVED"',
        '("CONTACT-CARRIER CANDIDATE | USER REVIEW PENDING" if object_follow_enabled else "FAILED DIAGNOSTIC CANDIDATE | NOT APPROVED")',
    ),
)
for old, new in replacements:
    if old not in source:
        raise RuntimeError(f"verified renderer token missing: {old}")
    source = source.replace(old, new)

namespace = {"__name__": "__main__", "__file__": str(Path(__file__).resolve())}
exec(compile(source, str(source_path), "exec"), namespace)
