# Pre-mount G1 deployment readiness report

Date: 2026-08-25

The camera-independent deployment stack is prepared and fail-closed. The final D455 is still `CAMERA_NOT_MOUNTED` / `EXTRINSIC_NOT_CALIBRATED`; no final-view render, XR collection, post-training, command publication, mode switch, or robot motion was performed.

The overall gate is **BLOCKED_BEFORE_HELMET_D455_INSTALL** because the frozen Fair Baseline A converter produced 22 genuine hard failures on the exact common 50-source set (18 `FAIL_IK`, 4 `FAIL_COLLISION`). As required, all 50 were attempted, none were excluded, Dataset A was not packaged, and Policy A training was not started. The complete machine-readable gate is `outputs/pre_mount_readiness/dataset_a/dataset_a_validation.json`.

## Recovery and frozen inputs

- Pre-change checkpoint: commit `04780407e25b0388a4b9db6489b7cdc40f116a02`, tag `pre-mount-g1-readiness-20260825`.
- Final Dataset B manifest SHA-256: `693b8b5531c8adf1a7921452c7af6dbc89feb99f84aa70185ab33dca2cb11b44`.
- Common 50-source manifest SHA-256: `8fd073d78cfe2e27075954cd43910b1cac43ccf7f373ea528866a82d3cd04f76`.
- Frozen Policy B source checkpoint model SHA-256: `bfe3e2aa6529967a12831a6f0bb91104b704835b2f43733072489b9ea2b68395`.
- `SOURCE_LIKE_CAM_HIGH` is retained only as `SOURCE_LIKE_CAM_HIGH_DIAGNOSTIC`; no further tuning was performed.

## Official upstream pins

See `docs/pre_mount_upstream_audit.json`. External clones are ignored under `external/third_party`; no upstream repository was modified or vendored.

| Repository | Commit | License |
| --- | --- | --- |
| unitreerobotics/unitree_sdk2 | `f29ee9f234851e9e79f75102c0f9e83008d8fdd1` | BSD-3-Clause |
| unitreerobotics/xr_teleoperate | `845b25a32f7febedf220e830952a7134897adb9d` | Apache-2.0 |
| unitreerobotics/unitree_lerobot | `41c2805742de879ddab2d8d6beaeaf215f876395` | Apache-2.0 |
| unitreerobotics/unitree_sim_isaaclab | `e30c25b1dffdf92ada1d6c8c1fe9a47bdde0fecc` | Apache-2.0 |
| realsenseai/librealsense | `e196cefa896e312d79c2df400c7623aa1e9c62ac` | Apache-2.0 |

## Helmet D455 kit

- Read-only probe: `tools/probe_helmet_d455.py`.
- RGB/depth capture: `tools/capture_helmet_d455_calibration.py`.
- ChArUco/task solve: `tools/solve_helmet_d455_task_extrinsic.py`.
- ChArUco plus black-workspace validation and fail-closed freeze: `tools/validate_helmet_d455_calibration.py`.
- Pending template: `configs/helmet_d455_mount_template.json`.
- Exact board: 7 × 5 squares, 50.0 mm squares, 37.5 mm markers, `DICT_4X4_50`, 350.0 × 250.0 mm pattern on A3 at 100%.
- Printable: `calibration/helmet_d455/printable/HELMET_D455_CHARUCO_7X5_50MM_A3_PRINT_AT_100_PERCENT.pdf`.
- Workflow: `docs/HELMET_D455_ONE_SHOT_CALIBRATION.md`.
- Offline capture → solve → validation: `PASS_SYNTHETIC_OFFLINE_ONLY_NOT_FREEZABLE`; synthetic freeze is refused.
- Host probe found a D405, not a D455. Deployment status therefore remains `NOT_MOUNTED` / `NOT_CALIBRATED`.
- A movable parent is fail-closed: runtime requires either the exact configured locked pose ID or a timestamp-aligned read-only parent FK JSON bridge.

## Camera-config-driven A/B path

The same required `--camera-config` interface is used by `tools/prepare_doll_handoff_g1visual_render_plan.py`, `tools/render_doll_handoff_g1visual_dataset.py`, and `tools/run_policy_isaac_doll_handoff.py`. The renderer and rollout runner take an explicit A/B variant; the final pending preset is rejected by operational consumers. B render-plan identity was proven with both the historical diagnostic camera and the pending helmet config; action and state hashes stayed identical. No final 50-episode rendering was run.

The generalized Isaac runner now instantiates one policy camera only; its intrinsics, pose, resolution, clipping range, and distortion all come from the selected config. A Policy B stage-0 diagnostic smoke passed camera intrinsic/pose/image-shape readback and emitted a finite 50 × 28 chunk. The raw sample's small Dex3 hard-limit excess was recorded rather than hidden; the common hard-limit and deployment-safety projections passed. This smoke was inference-only and did not tune the diagnostic view, advance physics, or expose a robot command path.

`tools/prepare_final_view_ab_pipeline.py` generates the gated A/B command plan. It covers frozen label intake, common camera rendering, phase-consistent doll reconstruction, LeRobot packaging/readback, exact label audit, 1:1 paired rehearsal, the common nine-phase probe, and identical closed-loop rollout. `tools/build_paired_visual_rehearsal_dataset.py` passed a read-only diagnostic dry run over 50/50 episodes and 68,956 total paired frames. Equal adaptation configs are prepared but disabled.

## Dataset and Policy A

- Converter input: exact final common 50-source manifest, common backend/natural-arm/safety contracts, Fair Baseline A only, no Proposed-B interaction projection, and no episode correction.
- Conversion: 50/50 attempted; 28 pass, 18 `FAIL_IK`, 4 `FAIL_COLLISION`.
- Packaging: stopped; `datasets/doll_handoff_trajectory_a_50` does not exist.
- Silent exclusions: zero.
- Policy A source config: frozen equal to original Policy B conditions (same SmolVLA base, 20,000 steps, batch 16, AdamW/cosine LR principle, chunk 50, 28-D interfaces, `MEAN_STD`, `cam_high`, task, seed, and augmentation setting).
- Policy A checkpoint: not created.
- Policy A nine-phase probe: not run because no checkpoint exists.

## XR bridge

`tools/convert_unitree_xr_to_canonical_g1.py` accepts official Unitree XR `data.json` recordings and produces canonical final-camera RGB, measured 28-D state, commanded 28-D action, timestamps, task/episode metadata, camera serial, calibration hash, exclusion manifest, adaptation/reference split, LeRobot conversion, and readback. Simulation dry-run: 3 episodes / 36 frames, PASS. Real collection remains `NOT_STARTED`.

`configs/xr_post_training/policy_a_equal.json` and `policy_b_equal.json` both reference the single disabled `shared_equal_ab.json`. Only variant identity, source checkpoint, and output path may differ. Neither post-training run was started.

## Dex3 contact instrumentation

`tools/record_dex3_contact_readonly.py` contains exactly two authoritative `HandState_` subscribers (`rt/lf/dex3/left/state`, `rt/lf/dex3/right/state`) and no publisher, command message/client, or mode switch. It records seven q/dq/ddq/tau fields plus all nine raw pressure records (12 raw values each), temperatures, lost/reserve fields, host timestamps, rate, and estimated dropout.

`tools/analyze_dex3_contact_readonly.py` produces baseline/noise statistics, raw channel and delta plots, plateau/dropout candidates, left/right comparison, and a manual sensor-index annotation template. Synthetic capture/analysis passed. Force units and contact thresholds remain `NOT_ASSIGNED`.

`ContactFeedbackGraspAdapter` is a disabled advisory skeleton with `OPEN`, `CLOSING`, `CONTACT_DETECTED`, `HOLD`, and `RELEASE`; activation against the uncalibrated template is refused.

## Real-G1 inference-only harness

`tools/run_real_g1_inference_only.py` implements D455 RGB + named measured G1/Dex3 state_28d + task → selected Policy A/B → raw action_chunk_50x28 → timestamped logs. It loads checkpoint preprocessors/postprocessors, runs common hard-limit/collision preflight, and has no real-command publisher/client or mode-switch path.

Both A/B mock interfaces passed. A real frozen Policy B inference passed saved normalization, emitted one finite 50 × 28 chunk, and passed hard-limit/collision preflight. Policy A selection is implemented but its checkpoint is blocked by the Dataset A gate. Status is `INFERENCE_ONLY_NO_COMMAND_PUBLISHER`.

## Offline tests

`tests/test_pre_mount_g1_readiness.py`: 14 passed. The suite checks frozen B hashes, upstream pins, pending-camera refusal, synthetic ChArUco/black-frame validation, movable-parent enforcement, single config-driven Isaac camera behavior and smoke evidence, final-view identity, Dataset A hard-stop, Policy A training gate, XR conversion/equality, Dex3 subscriber-only behavior, disabled contact adapter, and A/B log-only inference.

## Remaining physical inputs

- mount D455
- capture serial/intrinsics
- solve final extrinsic
- run real Dex3 read-only contact calibration
- collect final-camera XR data

## Non-physical blocker

Fair Baseline A cannot be silently repaired or subsetted. A comparison requires an explicit research decision about the 22 common-source hard failures; Policy A source training and all later A/B comparisons remain blocked until that decision is made.

BLOCKED_BEFORE_HELMET_D455_INSTALL
