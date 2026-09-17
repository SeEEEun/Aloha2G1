from __future__ import annotations

from pathlib import Path
import json
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.fair_a_full50_audit.common import (
    OUTPUT_ROOT,
    load_a_trajectory,
    stable_episode_id,
)
from tools.fair_a_full50_audit.full_pose_resolver import (
    FullPoseCommonWristResolver,
)
from tools.fair_a_full50_audit.wrist_resolver import IMMUTABLE_SOURCE_KEYS


class FairAFull50AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.resolver = FullPoseCommonWristResolver(output_root=OUTPUT_ROOT)

    def test_wrist_adapter_does_not_consume_interaction_target(self) -> None:
        values = load_a_trajectory(3)
        position, rotation = self.resolver._source_targets(values)
        for side in ("left", "right"):
            expected = self.resolver.g1.model_to_world_position(
                values[f"target_{side}_wrist_position_model"]
            )
            np.testing.assert_array_equal(position[side], expected)
            np.testing.assert_array_equal(
                rotation[side],
                values[f"target_{side}_wrist_rotation_model"].astype(np.float64),
            )
            self.assertFalse(
                np.array_equal(
                    position[side].astype(np.float32),
                    values[f"target_{side}_interaction_frame_position_world"],
                )
            )

    def test_all_repaired_outputs_preserve_fair_source_arrays(self) -> None:
        for episode in range(50):
            original = load_a_trajectory(episode)
            trajectory, _ = self.resolver._cache_paths(episode)
            self.assertTrue(trajectory.is_file())
            with np.load(trajectory, allow_pickle=False) as payload:
                for key in IMMUTABLE_SOURCE_KEYS:
                    np.testing.assert_array_equal(payload[key], original[key])
                self.assertFalse(
                    bool(payload["interaction_frame_used_as_solver_target"])
                )
                self.assertFalse(
                    bool(payload["source_targets_modified_semantically"])
                )
                self.assertFalse(bool(payload["episode_specific_correction"]))
                self.assertFalse(bool(payload["phase_specific_correction"]))
                self.assertEqual(
                    str(payload["realization_frame"]), "G1_WRIST_ORIGIN"
                )

    def test_independent_gate_retains_two_hard_collisions(self) -> None:
        path = OUTPUT_ROOT / "after/full50_validation.json"
        self.assertTrue(path.is_file())
        report = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(report["episode_count"], 50)
        self.assertEqual(report["total_frames"], 34478)
        self.assertEqual(
            report["classification_counts"],
            {
                "CLEAN_PASS": 27,
                "USABLE_WITH_WARNING": 21,
                "HARD_FAIL": 2,
            },
        )
        self.assertEqual(report["collision_hard_frame_count"], 366)
        self.assertEqual(report["joint_limit_violation_count"], 0)
        self.assertEqual(report["branch_discontinuity_count"], 0)


if __name__ == "__main__":
    unittest.main()
