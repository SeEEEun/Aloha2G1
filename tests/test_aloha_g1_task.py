import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("aloha_g1_task", ROOT / "tools/aloha_g1_task.py")
task = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(task)


def test_authoritative_scene_reused():
    assert task.AUTHORITATIVE_SCENE.name == "magsafe_magnetic_scene_v2.usda"
    assert task.ALOHA_REPLAY.parent == task.G1_REPLAY.parent == task.SCENE


def test_bundle_layout(tmp_path, monkeypatch):
    monkeypatch.setattr(task, "TASK_ROOT", tmp_path)
    base = task.bundle("episode_1"); task.make_dirs(base)
    assert {p.name for p in base.iterdir()} == {"source", "converted", "videos", "metrics", "logs"}


def test_action_validation(tmp_path):
    p = tmp_path / "action.npz"; np.savez(p, optimized_action=np.zeros((12, 14)), fps=30.0)
    a, fps, key = task.inspect_action(p)
    assert a.shape == (12, 14) and fps == 30 and key == "optimized_action"


def test_prepare_blocks_without_approved_calibration(tmp_path, monkeypatch):
    source = tmp_path / "input.npz"; np.savez(source, optimized_action=np.zeros((12, 14)), fps=30.0)
    monkeypatch.setattr(task, "TASK_ROOT", tmp_path / "bundles")
    monkeypatch.setattr(task, "CONFIGS", {k: tmp_path / f"{k}.json" for k in ("root", "dex3", "grasp")})
    args = Namespace(episode_id="ep", aloha_action=source, source_type="smolvla", timeline=None, force=False, dry_run=False)
    assert task.prepare(args) == 4
    m = json.loads((tmp_path / "bundles/ep/manifest.json").read_text())
    assert m["validation_status"]["conversion"] == "NOT_RUN"
    assert m["validation_status"]["calibration"] == "CALIBRATION_REQUIRED"


def test_recalibration_is_never_implicit():
    text = (ROOT / "tools/aloha_g1_task.py").read_text()
    assert "--recalibrate" not in text
    assert "ONE_TIME_APPROVED_CALIBRATION_CONFIGS_MISSING" in text


class FakeAdapter:
    def __init__(self): self.inferences = 0; self.executions = 0
    def observe(self): return {"rgb": np.zeros((2, 2, 3)), "robot_state": np.zeros(28), "instruction": "task"}
    def infer(self, observation, horizon): self.inferences += 1; return np.zeros((horizon, 14))
    def convert(self, chunk): return np.zeros((len(chunk), 28))
    def execute(self, chunk, horizon): self.executions += 1; return {"terminated": self.executions == 3}


def test_closed_loop_replans_instead_of_replaying_full_trajectory():
    adapter = FakeAdapter()
    rows = task.minimum_closed_loop(adapter, 8, 2, 10)
    assert len(rows) == adapter.inferences == adapter.executions == 3
    assert all(row["replanned"] for row in rows)


def test_policy_input_requires_rgb_state_instruction():
    adapter = FakeAdapter(); adapter.observe = lambda: {"rgb": np.zeros((1, 1, 3))}
    try: task.minimum_closed_loop(adapter, 4, 2, 1)
    except RuntimeError as exc: assert "rgb, robot_state and instruction" in str(exc)
    else: raise AssertionError("missing state was accepted")


def test_physics_has_no_object_snap_and_requires_controller():
    text = (ROOT / "tools/aloha_g1_task.py").read_text()
    assert "VERIFIED_G1_PHYSICS_TRAJECTORY_CONTROLLER_NOT_AVAILABLE" in text
    assert "palm-follow" not in text.lower()


def test_no_hardware_apis():
    text = (ROOT / "tools/aloha_g1_task.py").read_text()
    for forbidden in ("ChannelPublisher", "ChannelFactoryInitialize", "rt/lowstate", "xr_teleop"):
        assert forbidden not in text


def test_manifest_dashboard_marks_simulation(tmp_path):
    base = tmp_path / "ep"; task.make_dirs(base)
    m = {"episode_id": "ep", "validation_status": {}, "failure_reasons": [], "simulation_only": True}
    task.write_dashboard(base, m)
    assert "SIMULATION ONLY" in (base / "index.html").read_text()

