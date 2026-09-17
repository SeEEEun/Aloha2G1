#!/usr/bin/env python3
"""Fit, validate, and freeze (or invalidate) the blind physical envelope."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path("/home/jbnu/aloha_g1_dataset")
DEFAULT = ROOT / "outputs/final_representation_neutral_eval/00_frozen_evaluator"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def feature(row: dict[str, Any]) -> np.ndarray:
    translation = np.asarray(row["delta_translation_object_m"], dtype=np.float64) / 0.01
    rotation = np.rad2deg(
        Rotation.from_euler(
            "xyz", row["delta_rotation_object_rpy_deg"], degrees=True
        ).as_rotvec()
    ) / 10.0
    return np.r_[translation, rotation]


def predict(
    query: np.ndarray,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    radius: float,
    guard: float,
) -> tuple[bool, float, float, float]:
    distances = np.linalg.norm(train_features - query[None], axis=1)
    positive = distances[train_labels]
    negative = distances[~train_labels]
    d_positive = float(np.min(positive)) if len(positive) else float("inf")
    d_negative = float(np.min(negative)) if len(negative) else float("inf")
    radius_margin = radius - d_positive
    guard_margin = guard * d_negative - d_positive
    margin = float(min(radius_margin, guard_margin))
    return bool(margin >= 0.0), margin, d_positive, d_negative


def confusion(truth: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(truth, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    tp = int(np.count_nonzero(truth & predicted))
    fp = int(np.count_nonzero(~truth & predicted))
    tn = int(np.count_nonzero(~truth & ~predicted))
    fn = int(np.count_nonzero(truth & ~predicted))
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    fpr = float(fp / (fp + tn)) if fp + tn else 0.0
    fnr = float(fn / (fn + tp)) if fn + tp else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT)
    args = parser.parse_args()
    root = args.root.resolve()
    atlas = root / "atlas_v2"
    protocol_path = root / "BLIND_ATLAS_PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    rows: dict[str, dict[str, Any]] = {}
    labels: list[dict[str, Any]] = []
    manifests: list[Path] = []
    for round_index in (1, 2):
        manifest_path = atlas / f"ROUND{round_index}_COMMAND_MANIFEST.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"missing bounded atlas round {round_index}")
        manifests.append(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in manifest["records"]:
            rows[row["sample_id"]] = row
        for sample_id in manifest["selected_sample_ids"]:
            label_path = atlas / f"round_{round_index}_physx" / sample_id / "PHYSICAL_LABEL.json"
            if not label_path.is_file():
                raise RuntimeError(f"missing physical label: {label_path}")
            label = json.loads(label_path.read_text(encoding="utf-8"))
            label["physical_label_path"] = str(label_path)
            label["physical_label_sha256"] = sha256(label_path)
            labels.append(label)
    calibration = [label for label in labels if label["split"] == "calibration"]
    validation = [label for label in labels if label["split"] == "validation"]
    if not calibration or not validation:
        raise RuntimeError("predeclared calibration/validation split is empty")
    calibration_features = np.stack([feature(rows[label["sample_id"]]) for label in calibration])
    calibration_truth = np.asarray(
        [label["physically_graspable"] for label in calibration], dtype=bool
    )
    if not np.any(calibration_truth) or np.all(calibration_truth):
        raise RuntimeError("calibration requires both physical classes")

    candidates: list[dict[str, Any]] = []
    for radius in protocol["envelope"]["candidate_positive_radii"]:
        for guard in protocol["envelope"]["negative_guard_scale_candidates"]:
            predicted: list[bool] = []
            margins: list[float] = []
            for index, query in enumerate(calibration_features):
                keep = np.arange(len(calibration_features)) != index
                value, margin, _, _ = predict(
                    query,
                    calibration_features[keep],
                    calibration_truth[keep],
                    float(radius),
                    float(guard),
                )
                predicted.append(value)
                margins.append(margin)
            metrics = confusion(calibration_truth, np.asarray(predicted))
            candidates.append(
                {
                    "positive_radius": float(radius),
                    "negative_guard_scale": float(guard),
                    "leave_one_out": metrics,
                    "leave_one_out_margins": margins,
                }
            )
    chosen = min(
        candidates,
        key=lambda row: (
            row["leave_one_out"]["false_positive_rate"],
            -row["leave_one_out"]["precision"],
            -row["leave_one_out"]["recall"],
            row["positive_radius"],
            row["negative_guard_scale"],
        ),
    )
    validation_features = np.stack([feature(rows[label["sample_id"]]) for label in validation])
    validation_truth = np.asarray(
        [label["physically_graspable"] for label in validation], dtype=bool
    )
    validation_predictions: list[bool] = []
    validation_rows: list[dict[str, Any]] = []
    for label, query in zip(validation, validation_features, strict=True):
        predicted, margin, d_positive, d_negative = predict(
            query,
            calibration_features,
            calibration_truth,
            chosen["positive_radius"],
            chosen["negative_guard_scale"],
        )
        validation_predictions.append(predicted)
        validation_rows.append(
            {
                "sample_id": label["sample_id"],
                "physical_truth": bool(label["physically_graspable"]),
                "predicted_ready": predicted,
                "physical_readiness_margin": margin,
                "nearest_positive_distance": d_positive,
                "nearest_negative_distance": d_negative,
            }
        )
    validation_metrics = confusion(validation_truth, np.asarray(validation_predictions))
    acceptance = protocol["envelope"]["validation_acceptance"]
    accepted = bool(
        validation_metrics["true_positive"] >= int(acceptance["minimum_true_positive_count"])
        and validation_metrics["precision"] >= float(acceptance["minimum_precision"])
        and validation_metrics["false_positive_rate"]
        <= float(acceptance["maximum_false_positive_rate"])
    )

    atlas_manifest = {
        "schema_version": "blind_dex3_graspability_atlas_manifest_v1",
        "status": "PHYSICALLY_LABELED",
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "sample_count": len(labels),
        "calibration_count": len(calibration),
        "validation_count": len(validation),
        "positive_count": sum(label["physically_graspable"] for label in labels),
        "negative_count": sum(not label["physically_graspable"] for label in labels),
        "calibration_positive_count": int(np.count_nonzero(calibration_truth)),
        "validation_positive_count": int(np.count_nonzero(validation_truth)),
        "round_manifests": [
            {"path": str(path), "sha256": sha256(path)} for path in manifests
        ],
        "physical_labels": labels,
        "ab_artifacts_accessed_before_selection": False,
    }
    dump(root / "GRASPABILITY_ATLAS_MANIFEST.json", atlas_manifest)
    text(
        root / "GRASPABILITY_ATLAS_MANIFEST.md",
        "\n".join(
            [
                "# Independent Dex3 physical graspability atlas",
                "",
                "The atlas was generated and physically labeled without loading A/B or ACT artifacts.",
                "",
                f"- Samples: {len(labels)}",
                f"- Calibration: {len(calibration)} ({int(np.count_nonzero(calibration_truth))} positive)",
                f"- Validation: {len(validation)} ({int(np.count_nonzero(validation_truth))} positive)",
                f"- Seed: {protocol['atlas']['seed']}",
                f"- Protocol SHA256: `{sha256(protocol_path)}`",
            ]
        ),
    )
    envelope = {
        "schema_version": "dex3_physical_graspability_envelope_v1",
        "status": "VALIDATED_AND_FROZEN" if accepted else "INVALID_NOT_FROZEN",
        "features": protocol["envelope"]["features"],
        "normalization": protocol["envelope"]["normalization"],
        "classifier": protocol["envelope"]["family"],
        "positive_radius": chosen["positive_radius"],
        "negative_guard_scale": chosen["negative_guard_scale"],
        "calibration_positive_features": calibration_features[calibration_truth].tolist(),
        "calibration_negative_features": calibration_features[~calibration_truth].tolist(),
        "selection_priority": protocol["envelope"]["selection_priority"],
        "selection_metrics": chosen["leave_one_out"],
        "margin": "min(positive_radius - nearest_positive_distance, negative_guard_scale * nearest_negative_distance - nearest_positive_distance)",
        "ab_information_used": False,
    }
    dump(root / "GRASPABILITY_ENVELOPE.json", envelope)
    text(
        root / "GRASPABILITY_ENVELOPE.md",
        "\n".join(
            [
                "# Dex3 physical graspability envelope",
                "",
                f"Status: **{envelope['status']}**",
                f"Positive radius: {chosen['positive_radius']}",
                f"Negative guard scale: {chosen['negative_guard_scale']}",
                f"Calibration LOO precision: {chosen['leave_one_out']['precision']:.6f}",
                f"Calibration LOO recall: {chosen['leave_one_out']['recall']:.6f}",
                f"Calibration LOO FPR: {chosen['leave_one_out']['false_positive_rate']:.6f}",
            ]
        ),
    )
    validation_result = {
        "schema_version": "dex3_physical_graspability_validation_v1",
        "status": "PASS" if accepted else "FAIL",
        "acceptance_rule": acceptance,
        "metrics": validation_metrics,
        "records": validation_rows,
        "classifier_parameters_selected_before_validation": True,
        "validation_used_for_tuning": False,
    }
    dump(root / "VALIDATION_RESULTS.json", validation_result)
    text(
        root / "VALIDATION_RESULTS.md",
        "\n".join(
            [
                "# Held-out synthetic validation",
                "",
                f"Status: **{validation_result['status']}**",
                f"Confusion: TP={validation_metrics['true_positive']}, FP={validation_metrics['false_positive']}, TN={validation_metrics['true_negative']}, FN={validation_metrics['false_negative']}",
                f"Precision: {validation_metrics['precision']:.6f}",
                f"Recall: {validation_metrics['recall']:.6f}",
                f"False-positive rate: {validation_metrics['false_positive_rate']:.6f}",
                f"False-negative rate: {validation_metrics['false_negative_rate']:.6f}",
            ]
        ),
    )

    authoritative = [
        protocol_path,
        root / "COMMON_GRASP_INTENT_SPEC.json",
        ROOT / "outputs/final_representation_neutral_eval/PREDECLARED_TASK_SUCCESS_CRITERIA.md",
        root / "GRASPABILITY_ATLAS_MANIFEST.json",
        root / "GRASPABILITY_ENVELOPE.json",
        root / "VALIDATION_RESULTS.json",
        ROOT / "outputs/final_contact_constrained_eval/03_freeze/FREEZE_MANIFEST.json",
        ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json",
        ROOT / "tools/build_blind_dex3_graspability_atlas.py",
        ROOT / "tools/prepare_blind_dex3_atlas_round2.py",
        ROOT / "tools/run_blind_dex3_atlas_physx.py",
        Path(__file__).resolve(),
    ]
    freeze_manifest = {
        "schema_version": "representation_neutral_evaluator_freeze_manifest_v1",
        "status": "FROZEN_BEFORE_A_B_ACCESS" if accepted else "EVALUATOR_INVALID_NOT_FROZEN",
        "ab_artifacts_accessed_before_freeze": False,
        "post_freeze_tuning_allowed": False,
        "validation_status": validation_result["status"],
        "files": [
            {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in authoritative
        ],
    }
    dump(root / "EVALUATOR_FREEZE_MANIFEST.json", freeze_manifest)
    freeze_sha = sha256(root / "EVALUATOR_FREEZE_MANIFEST.json")
    text(
        root / "EVALUATOR_FREEZE_MANIFEST.md",
        "\n".join(
            [
                "# Representation-neutral physical evaluator freeze",
                "",
                f"Status: **{freeze_manifest['status']}**",
                f"Validation: **{validation_result['status']}**",
                f"Manifest SHA256: `{freeze_sha}`",
                "",
                "No Fair-A, Proposed-B, ACT-A, ACT-B, method, or episode outcome was used in calibration, selection, or validation.",
            ]
        ),
    )
    print(
        json.dumps(
            {
                "status": freeze_manifest["status"],
                "samples": len(labels),
                "calibration": len(calibration),
                "validation": len(validation),
                "validation_metrics": validation_metrics,
                "frozen_sha256": freeze_sha,
            },
            indent=2,
        )
    )
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
