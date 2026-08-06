import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/g1_world_task_retargeting/right_tcp_debug"
SCRIPT = ROOT / "tools/debug_aloha_right_tcp_alignment.py"


def load_json(name):
    return json.loads((OUT / name).read_text())


def test_exact_frame_window_and_candidates():
    rows = list(csv.DictReader((OUT / "right_tcp_candidate_distances.csv").open()))
    assert {int(r["frame"]) for r in rows} == set(range(300, 361))
    assert len({r["candidate"] for r in rows}) == 8


def test_channel_mapping_is_explicit_and_name_based():
    audit = load_json("right_tcp_mapping_audit.json")
    assert audit["joint_channels"] == {
        "left_arm": "0:6", "left_gripper": 6,
        "right_arm": "7:13", "right_gripper": 13,
    }
    assert audit["qpos_application"]["name_based_mapping"] is True
    assert audit["missing_joints"] == []


def test_no_automatic_tcp_change_or_root_search():
    audit = load_json("right_tcp_mapping_audit.json")
    assert audit["automatic_tcp_change"] is False
    assert audit["g1_root_search_run"] is False
    assert audit["status"] == "USER_APPROVAL_REQUIRED"


def test_frame_alignment_not_shifted_or_changed():
    audit = load_json("frame_alignment_audit.json")
    assert audit["best_shift_frames"] == 0
    assert audit["frame_index_exact"] is True
    assert audit["event_frame_changed"] is False


def test_separate_fk_disagreement_is_recorded_not_applied():
    audit = load_json("right_tcp_mapping_audit.json")
    comparison = audit["separate_fk_vs_isaac_articulation"]
    assert comparison["status"] == "MISMATCH_OBSERVED"
    assert comparison["difference_norm_m"] > 0.1


def test_no_hardware_or_publisher_code():
    text = SCRIPT.read_text()
    for forbidden in ("ChannelPublisher", "ChannelFactoryInitialize", "rt/lowstate"):
        assert forbidden not in text

