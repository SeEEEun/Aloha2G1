"""Canonical event and progress names for the MagSafe task."""

SCHEMA_NAME = "aloha_magsafe_semantic_timeline_v1"

REQUIRED_EVENTS = (
    "left_phone_grasp_start",
    "phone_rotation_to_portrait_start",
    "phone_portrait_reached",
    "right_accessory_grasp_start",
    "accessory_detachment_start",
    "accessory_removed",
    "phone_move_to_charger_start",
    "phone_charger_attachment_complete",
    "left_phone_release_complete",
    "right_accessory_release_complete",
    "left_arm_return_near_home",
    "task_end",
)

OPTIONAL_EVENTS = (
    "left_phone_grasp_stable",
    "right_accessory_hook_stable",
    "phone_transport_stable",
    "terminal_hold_start",
)

PROGRESS_NAMES = (
    "phone_acquisition",
    "phone_rotation",
    "accessory_acquisition",
    "accessory_removal",
    "phone_to_charger",
    "left_release",
    "right_release",
)

PARTIAL_ORDER_EDGES = (
    ("left_phone_grasp_start", "phone_rotation_to_portrait_start", False),
    ("phone_rotation_to_portrait_start", "phone_portrait_reached", False),
    ("phone_portrait_reached", "right_accessory_grasp_start", True),
    ("right_accessory_grasp_start", "accessory_detachment_start", False),
    ("accessory_detachment_start", "accessory_removed", False),
    ("accessory_removed", "phone_move_to_charger_start", True),
    ("phone_move_to_charger_start", "phone_charger_attachment_complete", False),
    ("phone_charger_attachment_complete", "left_phone_release_complete", False),
    ("left_phone_release_complete", "left_arm_return_near_home", True),
    ("left_arm_return_near_home", "task_end", True),
    ("accessory_removed", "right_accessory_release_complete", False),
    ("right_accessory_release_complete", "task_end", True),
)

CONFIDENCE_CLASSES = ("HIGH", "MEDIUM", "LOW", "AMBIGUOUS")

