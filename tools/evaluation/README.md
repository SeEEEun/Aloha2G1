# Paper evaluation contract

`tools/evaluate_paper_metrics.py` is the single policy-independent entry point.
It evaluates frozen retargeting trajectories, ACT prediction chunks, source-video-conditioned
rollouts, and later physical rollouts through the same episode schema. A/B pairing is exact
on `source_episode_id`; an unmatched identity is an error, never an implicit exclusion.

The evaluator uses the project's frozen 28-joint names/ranges and the existing Dex3
whole-hand/interaction-frame definition. Missing inputs remain `status: NA`; it never
constructs an orientation reference or contact/ownership annotation.

## Portable bundle

A bundle is JSON with `schema_version: paper_evaluation_bundle_v1`, one explicit
`evaluation_mode`, an optional explicit `success_kind`, and a list of episodes. Each episode
contains `source_episode_id`, `arrays_path`, 30 Hz timing, annotations, existing feasibility
diagnostics, and provenance. The NPZ may contain:

- bilateral `reference_*` and `candidate_*_wrist_position_m` arrays;
- optional authoritative bilateral wrist rotations as `[T,3,3]`;
- bilateral `reference_*` and `candidate_*_whole_hand_position_m` arrays;
- `reference_action_rad` / `candidate_action_rad` as `[T,28]`, or corresponding action
  chunks as `[N,K,28]`; optional `action_chunk_valid_lengths` excludes padding;
- `candidate_q_rad` as `[T,28]`, or `candidate_q_chunk_rad` for boundary-safe raw-chunk
  smoothness, and `feasibility_projection_m` per frame.

Semantic event frames are accepted only when `phase_events_authoritative: true`.
Handoff ordering is accepted only when `handoff_events_authoritative: true`. For
non-physical rollouts the success kind is `SEMANTIC_SUCCESS`; physical logs use the separate
`PHYSICAL_SUCCESS` contract.

## Definitions

- PCS is the longest correctly ordered canonical phase prefix divided by eight.
- Whole-hand error uses the authoritative project whole-hand frame; no new frame is defined.
- Bimanual relation is the error in `p_R(t) - p_L(t)`.
- HOA is one only when authoritative `RIGHT_ACQUIRE` strictly precedes `LEFT_RELEASE`.
- RPL is predicted divided by reference combined bilateral path length.
- SWPE is `S * L_ref / max(L_pred, L_ref)` and is explicitly labeled semantic or physical.
- Feasibility consumes validated project diagnostics; collision geometry is not recomputed.
- Smoothness uses 30 Hz finite differences. Direction reversal uses a fixed velocity
  deadband, and low-motion oscillation is peak-to-peak after removing the endpoint trend.

Paired bootstrap confidence intervals use 10,000 episode resamples and seed `20260826`.
The exact paired sign test is supplementary only.

The rigid-proxy event detector thresholds live only in
`configs/doll_handoff_rigid_proxy_v1.json`. They must be frozen before either policy is run
and cannot vary by method.

## Project adapters

The read-only subcommands `project-retargeting`, `paper-core-offline`,
`paper-core-source`, and `physical-batch` convert the repository's native artifacts to this
same contract. The latter three remain fail-closed until all exact source identities exist.
Each command emits episode metrics, paired statistics, and refreshes all four table formats
without replacing unavailable results with zeros.
