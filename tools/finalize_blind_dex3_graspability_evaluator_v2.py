#!/usr/bin/env python3
"""Stratify, fit, blind-validate, and freeze the synthetic-only evaluator v2."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUT = ROOT / "outputs/final_representation_neutral_eval/00_frozen_evaluator_v2"
SOURCE = ROOT / "outputs/final_representation_neutral_eval/00_frozen_evaluator"
SOURCE_ATLAS = SOURCE / "atlas_v2"
ATLAS = OUT / "atlas_extension"
PROTOCOL = OUT / "BLIND_ATLAS_EXTENSION_PROTOCOL.json"


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


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def split_hash(seed: int, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()


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
    k: int,
    radius: float,
    minimum_vote: float,
) -> tuple[bool, float, float, float]:
    distances = np.linalg.norm(train_features - query[None], axis=1)
    order = np.argsort(distances, kind="stable")[: min(k, len(distances))]
    nearest_distance = float(distances[order[0]])
    positive_vote = float(np.mean(train_labels[order]))
    margin = float(min(radius - nearest_distance, positive_vote - minimum_vote))
    return bool(margin >= 0.0), margin, nearest_distance, positive_vote


def confusion(truth: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(truth, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    tp = int(np.count_nonzero(truth & predicted))
    fp = int(np.count_nonzero(~truth & predicted))
    tn = int(np.count_nonzero(~truth & ~predicted))
    fn = int(np.count_nonzero(truth & ~predicted))
    return {
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
    }


def source_rows_and_labels() -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    rows: dict[str, dict[str, Any]] = {}
    labels: list[dict[str, Any]] = []
    for round_index in (1, 2):
        manifest = json.loads(
            (SOURCE_ATLAS / f"ROUND{round_index}_COMMAND_MANIFEST.json").read_text(
                encoding="utf-8"
            )
        )
        for row in manifest["records"]:
            rows[row["sample_id"]] = row
        for sample_id in manifest["selected_sample_ids"]:
            label_path = (
                SOURCE_ATLAS
                / f"round_{round_index}_physx"
                / sample_id
                / "PHYSICAL_LABEL.json"
            )
            label = json.loads(label_path.read_text(encoding="utf-8"))
            label.update(
                {
                    "origin": "existing_v1_known_calibration",
                    "split": "calibration",
                    "physical_label_path": str(label_path),
                    "physical_label_sha256": sha256(label_path),
                }
            )
            labels.append(label)
    return rows, labels


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    command_manifest_path = ATLAS / "EXTENSION_COMMAND_MANIFEST.json"
    command_manifest = json.loads(command_manifest_path.read_text(encoding="utf-8"))
    source_rows, source_labels = source_rows_and_labels()
    rows = dict(source_rows)
    extension_records = {
        row["sample_id"]: row for row in command_manifest["records"]
    }
    rows.update(extension_records)
    new_labels: list[dict[str, Any]] = []
    for sample_id in command_manifest["selected_sample_ids"]:
        label_path = ATLAS / "physx" / sample_id / "PHYSICAL_LABEL.json"
        if not label_path.is_file():
            raise RuntimeError(f"extension physics incomplete: {sample_id}")
        label = json.loads(label_path.read_text(encoding="utf-8"))
        label.update(
            {
                "origin": "new_blind_extension",
                "physical_label_path": str(label_path),
                "physical_label_sha256": sha256(label_path),
            }
        )
        new_labels.append(label)

    split_cfg = protocol["split"]
    random_seed = int(protocol["extension_sampling"]["seed"])
    new_positive = [row for row in new_labels if row["physically_graspable"]]
    new_negative = [row for row in new_labels if not row["physically_graspable"]]
    balance = {
        "new_positive": len(new_positive),
        "new_negative": len(new_negative),
        "minimum_each_for_preregistered_split": 15,
    }
    if len(new_positive) < 15 or len(new_negative) < 15:
        dump(
            OUT / "BALANCE_REQUIRED.json",
            {
                "schema_version": "blind_atlas_balance_required_v2",
                "status": "SUPPLEMENTAL_BALANCE_ROUND_REQUIRED",
                **balance,
                "rule": protocol["extension_sampling"]["supplemental_balance_round"],
            },
        )
        print(json.dumps({"status": "SUPPLEMENTAL_BALANCE_ROUND_REQUIRED", **balance}, indent=2))
        return 3

    validation_ids: set[str] = set()
    for class_rows, minimum in (
        (new_positive, int(split_cfg["minimum_new_validation_positive"])),
        (new_negative, int(split_cfg["minimum_new_validation_negative"])),
    ):
        count = max(
            minimum,
            int(round(float(split_cfg["new_validation_fraction_per_class"]) * len(class_rows))),
        )
        count = min(count, len(class_rows) - 6)
        ordered = sorted(
            class_rows, key=lambda row: split_hash(random_seed, row["sample_id"])
        )
        validation_ids.update(row["sample_id"] for row in ordered[:count])
    for row in new_labels:
        row["split"] = "validation" if row["sample_id"] in validation_ids else "calibration"

    calibration = source_labels + [
        row for row in new_labels if row["split"] == "calibration"
    ]
    validation = [row for row in new_labels if row["split"] == "validation"]
    validation_positive = sum(row["physically_graspable"] for row in validation)
    validation_negative = len(validation) - validation_positive
    calibration_positive = sum(row["physically_graspable"] for row in calibration)
    calibration_negative = len(calibration) - calibration_positive
    required_counts = {
        "validation_positive": int(split_cfg["minimum_new_validation_positive"]),
        "validation_negative": int(split_cfg["minimum_new_validation_negative"]),
        "new_calibration_positive": int(split_cfg["minimum_new_calibration_positive"]),
        "new_calibration_negative": int(split_cfg["minimum_new_calibration_negative"]),
    }
    new_calibration_positive = sum(
        row["physically_graspable"]
        for row in new_labels
        if row["split"] == "calibration"
    )
    new_calibration_negative = sum(
        not row["physically_graspable"]
        for row in new_labels
        if row["split"] == "calibration"
    )
    if (
        validation_positive < required_counts["validation_positive"]
        or validation_negative < required_counts["validation_negative"]
        or new_calibration_positive < required_counts["new_calibration_positive"]
        or new_calibration_negative < required_counts["new_calibration_negative"]
    ):
        raise RuntimeError("deterministic stratified split violates preregistered class counts")
    split_manifest = {
        "schema_version": "blind_dex3_stratified_split_v2",
        "status": "FROZEN_BEFORE_CLASSIFIER_FIT",
        "seed": random_seed,
        "existing_known_samples_assignment": "calibration_only",
        "new_sample_rule": split_cfg["new_samples"],
        "calibration_ids": [row["sample_id"] for row in calibration],
        "validation_ids": [row["sample_id"] for row in validation],
        "counts": {
            "calibration": len(calibration),
            "calibration_positive": calibration_positive,
            "calibration_negative": calibration_negative,
            "new_calibration_positive": new_calibration_positive,
            "new_calibration_negative": new_calibration_negative,
            "validation": len(validation),
            "validation_positive": validation_positive,
            "validation_negative": validation_negative,
        },
        "validation_used_for_model_selection": False,
        "ab_artifacts_accessed": False,
    }
    dump(OUT / "STRATIFIED_SPLIT.json", split_manifest)

    x_cal = np.stack([feature(rows[row["sample_id"]]) for row in calibration])
    y_cal = np.asarray([row["physically_graspable"] for row in calibration], dtype=bool)
    fold = np.empty(len(calibration), dtype=np.int64)
    for class_value in (False, True):
        indices = np.flatnonzero(y_cal == class_value)
        ordered = sorted(
            indices.tolist(),
            key=lambda index: split_hash(random_seed + 5, calibration[index]["sample_id"]),
        )
        for ordinal, index in enumerate(ordered):
            fold[index] = ordinal % 5
    classifier_cfg = protocol["classifier"]
    candidates: list[dict[str, Any]] = []
    for k in classifier_cfg["candidate_k"]:
        for radius in classifier_cfg["candidate_nearest_distance_radius"]:
            for vote in classifier_cfg["candidate_minimum_positive_vote_fraction"]:
                predictions = np.zeros(len(calibration), dtype=bool)
                margins = np.empty(len(calibration), dtype=np.float64)
                for fold_index in range(5):
                    train = fold != fold_index
                    test = np.flatnonzero(fold == fold_index)
                    for index in test:
                        value, margin, _, _ = predict(
                            x_cal[index],
                            x_cal[train],
                            y_cal[train],
                            int(k),
                            float(radius),
                            float(vote),
                        )
                        predictions[index] = value
                        margins[index] = margin
                metrics = confusion(y_cal, predictions)
                candidates.append(
                    {
                        "k": int(k),
                        "nearest_distance_radius": float(radius),
                        "minimum_positive_vote_fraction": float(vote),
                        "cross_validation": metrics,
                    }
                )
    selection = classifier_cfg["selection"]
    eligible = [
        row
        for row in candidates
        if row["cross_validation"]["true_positive"]
        >= int(selection["eligible_minimum_true_positive"])
        and row["cross_validation"]["recall"]
        >= float(selection["eligible_minimum_recall"])
    ]
    if not eligible:
        dump(
            OUT / "CLASSIFIER_SELECTION.json",
            {
                "status": "FAIL_NO_NONTRIVIAL_CALIBRATION_CLASSIFIER",
                "candidates": candidates,
            },
        )
        print("FAIL_NO_NONTRIVIAL_CALIBRATION_CLASSIFIER")
        return 2
    chosen = min(
        eligible,
        key=lambda row: (
            row["cross_validation"]["false_positive_rate"],
            -row["cross_validation"]["precision"],
            -row["cross_validation"]["recall"],
            row["nearest_distance_radius"],
            row["k"],
            -row["minimum_positive_vote_fraction"],
        ),
    )
    dump(
        OUT / "CLASSIFIER_SELECTION.json",
        {
            "schema_version": "blind_dex3_classifier_selection_v2",
            "status": "SELECTED_BEFORE_VALIDATION",
            "selection_rule": selection,
            "selected": chosen,
            "candidates": candidates,
            "validation_used": False,
            "ab_artifacts_accessed": False,
        },
    )

    x_validation = np.stack([feature(rows[row["sample_id"]]) for row in validation])
    y_validation = np.asarray(
        [row["physically_graspable"] for row in validation], dtype=bool
    )
    predicted = np.zeros(len(validation), dtype=bool)
    validation_records: list[dict[str, Any]] = []
    for index, row in enumerate(validation):
        value, margin, distance, vote = predict(
            x_validation[index],
            x_cal,
            y_cal,
            chosen["k"],
            chosen["nearest_distance_radius"],
            chosen["minimum_positive_vote_fraction"],
        )
        predicted[index] = value
        validation_records.append(
            {
                "sample_id": row["sample_id"],
                "physical_truth": bool(row["physically_graspable"]),
                "predicted_ready": value,
                "physical_readiness_margin": margin,
                "nearest_distance": distance,
                "positive_vote_fraction": vote,
            }
        )
    validation_metrics = confusion(y_validation, predicted)
    acceptance = protocol["blind_validation_acceptance"]
    passed = bool(
        validation_metrics["true_positive"] >= int(acceptance["minimum_true_positive"])
        and validation_metrics["true_negative"] >= int(acceptance["minimum_true_negative"])
        and validation_metrics["precision"] >= float(acceptance["minimum_precision"])
        and validation_metrics["recall"] >= float(acceptance["minimum_recall"])
        and validation_metrics["false_positive_rate"]
        <= float(acceptance["maximum_false_positive_rate"])
        and validation_metrics["false_negative_rate"]
        <= float(acceptance["maximum_false_negative_rate"])
    )
    validation_result = {
        "schema_version": "blind_dex3_validation_v2",
        "status": "PASS" if passed else "FAIL",
        "acceptance": acceptance,
        "metrics": validation_metrics,
        "records": validation_records,
        "non_degenerate_positive_denominator": validation_positive > 0,
        "non_degenerate_negative_denominator": validation_negative > 0,
        "classifier_selected_before_validation": True,
        "validation_used_for_tuning": False,
        "ab_artifacts_accessed": False,
    }
    dump(OUT / "VALIDATION_RESULTS.json", validation_result)

    all_labels = source_labels + new_labels
    atlas_manifest = {
        "schema_version": "independent_dex3_physical_graspability_atlas_v2",
        "status": "PHYSICALLY_LABELED_AND_STRATIFIED",
        "sample_count": len(all_labels),
        "positive_count": sum(row["physically_graspable"] for row in all_labels),
        "negative_count": sum(not row["physically_graspable"] for row in all_labels),
        "existing_sample_count": len(source_labels),
        "new_sample_count": len(new_labels),
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256(PROTOCOL),
        "split": str(OUT / "STRATIFIED_SPLIT.json"),
        "split_sha256": sha256(OUT / "STRATIFIED_SPLIT.json"),
        "labels": all_labels,
        "ab_artifacts_accessed": False,
    }
    dump(OUT / "GRASPABILITY_ATLAS_MANIFEST.json", atlas_manifest)
    envelope = {
        "schema_version": "dex3_physical_graspability_envelope_v2",
        "status": "VALIDATED_AND_FROZEN" if passed else "INVALID_NOT_FROZEN",
        "family": classifier_cfg["family"],
        "features": classifier_cfg["features"],
        "normalization": classifier_cfg["normalization"],
        "k": chosen["k"],
        "nearest_distance_radius": chosen["nearest_distance_radius"],
        "minimum_positive_vote_fraction": chosen[
            "minimum_positive_vote_fraction"
        ],
        "calibration_features": x_cal.tolist(),
        "calibration_labels": y_cal.astype(int).tolist(),
        "cross_validation_metrics": chosen["cross_validation"],
        "margin_definition": classifier_cfg["margin"],
        "all_negative_classifier": False,
        "ab_information_used": False,
    }
    dump(OUT / "GRASPABILITY_ENVELOPE.json", envelope)

    invalid_v1 = SOURCE / "EVALUATOR_FREEZE_MANIFEST.json"
    invalid_copy = OUT / "PRESERVED_INVALID_V1_EVALUATOR_MANIFEST.json"
    if not invalid_copy.exists():
        shutil.copy2(invalid_v1, invalid_copy)
    authoritative = [
        PROTOCOL,
        OUT / "STRATIFIED_SPLIT.json",
        OUT / "CLASSIFIER_SELECTION.json",
        OUT / "VALIDATION_RESULTS.json",
        OUT / "GRASPABILITY_ATLAS_MANIFEST.json",
        OUT / "GRASPABILITY_ENVELOPE.json",
        ROOT / "outputs/final_representation_neutral_eval/PREDECLARED_TASK_SUCCESS_CRITERIA.md",
        SOURCE / "COMMON_GRASP_INTENT_SPEC.json",
        ROOT / protocol["preserved_common_execution_layer"]["freeze_manifest"],
        ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json",
        ROOT / "tools/build_blind_dex3_graspability_atlas_extension_v2.py",
        ROOT / "tools/run_blind_dex3_atlas_extension_physx.py",
        Path(__file__).resolve(),
    ]
    freeze_manifest = {
        "schema_version": "representation_neutral_evaluator_freeze_manifest_v2",
        "status": "FROZEN_BEFORE_A_B_ACCESS" if passed else "EVALUATOR_INVALID_NOT_FROZEN",
        "validation_status": validation_result["status"],
        "common_execution_layer_modified": False,
        "common_execution_layer_freeze_sha256": protocol[
            "preserved_common_execution_layer"
        ]["freeze_manifest_sha256"],
        "ab_artifacts_accessed_before_freeze": False,
        "post_freeze_evaluator_tuning_allowed": False,
        "files": [
            {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in authoritative
        ],
    }
    dump(OUT / "EVALUATOR_FREEZE_MANIFEST.json", freeze_manifest)
    write_text(
        OUT / "EVALUATOR_FREEZE_MANIFEST.md",
        "\n".join(
            [
                "# Representation-neutral physical evaluator v2",
                "",
                f"Status: **{freeze_manifest['status']}**",
                f"Validation: **{validation_result['status']}**",
                f"Samples: {len(all_labels)} ({atlas_manifest['positive_count']} positive, {atlas_manifest['negative_count']} negative)",
                f"Calibration: {len(calibration)} ({calibration_positive} positive, {calibration_negative} negative)",
                f"Validation: {len(validation)} ({validation_positive} positive, {validation_negative} negative)",
                f"Validation precision: {validation_metrics['precision']:.6f}",
                f"Validation recall: {validation_metrics['recall']:.6f}",
                f"Validation FPR: {validation_metrics['false_positive_rate']:.6f}",
                f"Validation FNR: {validation_metrics['false_negative_rate']:.6f}",
                "",
                "The common execution layer was not modified. No A/B artifact was accessed before this decision.",
            ]
        ),
    )
    summary = {
        "status": freeze_manifest["status"],
        "samples": len(all_labels),
        "calibration": len(calibration),
        "validation": len(validation),
        "validation_metrics": validation_metrics,
        "freeze_manifest_sha256": sha256(OUT / "EVALUATOR_FREEZE_MANIFEST.json"),
    }
    print(json.dumps(summary, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
