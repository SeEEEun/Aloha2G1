# Helmet D455 one-shot calibration

The pre-mount state is intentionally `CAMERA_NOT_MOUNTED` and `EXTRINSIC_NOT_CALIBRATED`. The D405 currently visible to the host is not accepted as the deployment camera. Never rename a synthetic candidate or the pending template to `helmet_d455_final.json`.

## Before installation

Print `calibration/helmet_d455/printable/HELMET_D455_CHARUCO_7X5_50MM_A3_PRINT_AT_100_PERCENT.pdf` on A3 at 100%, with all fit/scaling options disabled. Measure the printed chessboard pattern. It must be 350.0 × 250.0 mm; reject it if either dimension differs by more than 0.5 mm. The exact board is 7 × 5 squares, 50.0 mm per square, 37.5 mm ArUco markers, `DICT_4X4_50`. Its canonical physical frame begins at the pattern’s bottom-left, with +X right, +Y up, and +Z out of the printed face.

That bottom-left/up/out frame is the kit's canonical physical frame. The detector internally converts OpenCV's raw top-left/down/into object-point frame using the exact matrix recorded in `configs/helmet_d455_charuco_board.json`; physical measurements and `task_from_board_matrix` must use only the canonical frame.

## Physical-day sequence

1. Connect the mounted D455 and run `python3 tools/probe_helmet_d455.py --serial SERIAL --output CAPTURE/probe.json`. Confirm product, serial, firmware, USB mode, and the requested 640 × 480 @ 30 Hz RGB/depth profiles.
2. Complete a copy of `configs/helmet_d455_mount_measurement.template.json`. If the parent link can move, record either a named locked parent/head pose, or a timestamp-aligned parent FK provider plus the measured rigid `parent_from_camera_matrix`.
3. Measure the printed board frame in the task frame and store a finite 4 × 4 `task_from_board_matrix` in JSON. This measurement is mandatory; the solver has no guessed default.
4. Capture at least 12 well-focused board views with `python3 tools/capture_helmet_d455_calibration.py --serial SERIAL --frames 12 --output-dir CAPTURE`. Use the exposure/gain switches if fixed settings are required. The command logs the effective supported settings after warmup, both stream intrinsics, color distortion, `color_from_depth`, firmware/USB identity, device timestamps, and host monotonic/wall timestamps.
5. Remove the board and capture a clear RGB image of the complete black workspace frame. It may be supplied to validation with `--black-image` or its four corners can be manually annotated in the configured task-corner order.
6. Solve with `python3 tools/solve_helmet_d455_task_extrinsic.py --capture-manifest CAPTURE/capture_manifest.json --task-from-board-json task_from_board.json --mount-json mount.json --output CAPTURE/helmet_d455_candidate.json`.
7. Validate with `python3 tools/validate_helmet_d455_calibration.py --candidate CAPTURE/helmet_d455_candidate.json --black-image black_workspace.png --report CAPTURE/validation.json --freeze configs/camera/helmet_d455_final.json`. Freeze succeeds only for a real D455, complete mount contract, passing ChArUco spread/reprojection, and passing black-frame reprojection. The required final basename is exactly `helmet_d455_final.json`.

That frozen file is then passed unchanged as `--camera-config` to both A/B rendering and both A/B Isaac rollouts. If the mount is disturbed, the calibration hash is invalid and the workflow must be repeated.

## Offline proof

The synthetic fixture under `outputs/pre_mount_readiness/helmet_d455` exercises capture → solve → ChArUco validation → black-frame validation. Its expected status is `PASS_SYNTHETIC_OFFLINE_ONLY_NOT_FREEZABLE`; `freeze_permitted` must remain false.
