#!/usr/bin/env python3
"""Package the five author-selected JKROS figures with self-contained provenance.

This is an archival/documentation utility.  It reads frozen figure/result artifacts,
copies the authoritative PNG bytes, writes Markdown, and validates a Markdown/PNG-
only ZIP.  It does not run retargeting, training, inference, or simulation.
"""

from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Iterable
import zipfile
from zoneinfo import ZoneInfo

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_DIR = ROOT / "outputs/paper_final_5_archive"
ZIP_PATH = ROOT / "outputs/ALOHA_G1_FINAL_5_FIGURES_COMPLETE_ARCHIVE.zip"
NOW = datetime.now(ZoneInfo("Asia/Seoul"))


FIGURES = [
    {
        "number": 1,
        "name": "Method Overview",
        "folder": "Fig01_Method_Overview",
        "stem": "Fig01_Method_Overview",
        "question": "What exactly is compared?",
        "location": "Methods; experimental-design overview before the quantitative results.",
        "format": "double-column",
    },
    {
        "number": 2,
        "name": "Retargeting Trade-off",
        "folder": "Fig02_Retargeting_Tradeoff",
        "stem": "Fig02_Retargeting_Tradeoff",
        "question": "Does accurate wrist transfer preserve manipulation interaction?",
        "location": "Results; primary retargeting-quality result.",
        "format": "double-column",
    },
    {
        "number": 3,
        "name": "Paired Statistical Effect",
        "folder": "Fig03_Paired_Statistics",
        "stem": "Fig03_Paired_Statistical_Effect",
        "question": "Is the A/B difference consistent over the 50 matched demonstrations?",
        "location": "Results; paired statistical evidence immediately after Fig. 2.",
        "format": "single-column",
    },
    {
        "number": 4,
        "name": "Feasibility",
        "folder": "Fig04_Feasibility",
        "stem": "Fig04_Feasibility",
        "question": "Can the retargeted trajectories be mechanically realized on G1?",
        "location": "Results; feasibility and hard-failure accounting.",
        "format": "double-column",
    },
    {
        "number": 5,
        "name": "Supervision to Policy",
        "folder": "Fig05_Supervision_to_Policy",
        "stem": "Fig05_Supervision_to_Policy",
        "question": "Does the representation characteristic survive policy learning?",
        "location": "Results; downstream held-out ACT prediction.",
        "format": "double-column",
    },
]


P = {
    "final_root": ROOT / "outputs/paper_final_figures",
    "final_manifest": ROOT / "outputs/paper_final_figures/manifest/FINAL_SIX_FIGURE_MANIFEST.md",
    "final_manifest_json": ROOT / "outputs/paper_final_figures/manifest/final_six_figure_manifest.json",
    "validation": ROOT / "outputs/paper_final_figures/manifest/validation_report.json",
    "plot_script": ROOT / "tools/generate_jkros_final_figures.py",
    "plot_validator": ROOT / "tools/validate_jkros_final_figures.py",
    "bank_script": ROOT / "tools/generate_paper_figure_bank.py",
    "pub_script": ROOT / "tools/generate_jkros_pubready_figures.py",
    "table1": ROOT / "outputs/paper_core_ab/tables/table1_full50_retargeting.csv",
    "table2": ROOT / "outputs/paper_core_ab/tables/table2_heldout_act_prediction.csv",
    "table1_json": ROOT / "outputs/paper_core_ab/tables/table1_full50_retargeting.json",
    "table2_json": ROOT / "outputs/paper_core_ab/tables/table2_heldout_act_prediction.json",
    "s1": ROOT / "outputs/paper_figure_bank/tables/TableS1_per_episode_retargeting.csv",
    "forest": ROOT / "outputs/paper_figure_bank_pubready/source_data/04_effect_forest.csv",
    "ecdf_errors": ROOT / "outputs/paper_figure_bank_pubready/source_data/10_error_ecdf.csv",
    "ecdf_projection": ROOT / "outputs/paper_figure_bank_pubready/source_data/11_projection_ecdf.csv",
    "a_manifest": ROOT / "outputs/fair_a_full50_hard_fail_audit/after/fair_a_repair_manifest.json",
    "a_per_episode": ROOT / "outputs/fair_a_full50_hard_fail_audit/after/full50_per_episode.csv",
    "a_projection": ROOT / "outputs/fair_a_full50_hard_fail_audit/after/projection_statistics.json",
    "a_b_comparison": ROOT / "outputs/fair_a_full50_hard_fail_audit/comparison/fair_a_vs_proposed_b.json",
    "a_report": ROOT / "outputs/fair_a_full50_hard_fail_audit/FAIR_A_FULL50_HARD_FAIL_AUDIT_REPORT.md",
    "b_manifest": ROOT / "outputs/doll_handoff_dataset_b_final/final_source_manifest.json",
    "b_freeze": ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json",
    "common48": ROOT / "outputs/paper_core_ab/common48_manifest.json",
    "train40": ROOT / "outputs/paper_core_ab/train40_manifest.json",
    "heldout8": ROOT / "outputs/paper_core_ab/heldout8_manifest.json",
    "dataset_audit": ROOT / "outputs/paper_core_ab/dataset_packaging_audit.json",
    "training_contract": ROOT / "outputs/paper_core_ab/act_a_b_training_contract.json",
    "training_audit": ROOT / "outputs/paper_core_ab/act_a_b_training_audit.json",
    "checkpoint_rule": ROOT / "outputs/paper_core_ab/checkpoint_selection_rule.json",
    "eval_contract": ROOT / "outputs/paper_core_ab/offline_evaluation_contract.json",
    "exp2": ROOT / "outputs/paper_core_ab/offline_heldout8/experiment2_result.json",
    "eval_script": ROOT / "tools/evaluate_paper_core_act_ab.py",
    "a_report_script": ROOT / "tools/fair_a_full50_audit/report.py",
    "a_wrist_resolver": ROOT / "tools/fair_a_full50_audit/wrist_resolver.py",
    "a_full_resolver": ROOT / "tools/fair_a_full50_audit/full_pose_resolver.py",
    "b_evaluator": ROOT / "tools/doll_handoff_feasibility/evaluate.py",
    "b_solver": ROOT / "tools/doll_handoff_feasibility/solver.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_text(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def md_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", "<br>")

    result = ["| " + " | ".join(cell(x) for x in headers) + " |"]
    result.append("|" + "|".join("---" for _ in headers) + "|")
    result.extend("| " + " | ".join(cell(x) for x in row) + " |" for row in rows)
    return "\n".join(result)


def fmt_float(value: Any, digits: int = 9) -> str:
    if value in (None, ""):
        return ""
    return f"{float(value):.{digits}f}"


def artifact_table(records: list[tuple[Path, str, str]]) -> str:
    rows = []
    for path, fields, scope in records:
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(
            (
                f"`{path.resolve()}`<br>`{rel(path)}`",
                sha256(path),
                fields,
                scope,
            )
        )
    return md_table(["Absolute / repository-relative path", "SHA256", "Fields used", "Scope"], rows)


def common_method_definitions() -> str:
    return """### Canonical methods

**Trajectory-Centric A (Fair-A).** The method transfers independent left/right 6-D wrist poses (position plus orientation). It does **not** use the source interaction frame, ownership state, or bimanual semantic relation as optimization targets. The archived result is the fairness-corrected version: a wrist-frame representation adapter feeds the frozen common temporal arm solver and generic feasibility machinery. The correction is global, not episode- or phase-specific, and does not semantically edit source targets. Whole-hand and bimanual values for A are diagnostic outputs, not targets that A optimized.

**Interaction-Centric B.** The method constructs an interaction representation comprising source interaction frames, whole-hand grasp geometry, the right-minus-left bimanual relation, and ownership transition, then realizes G1 arms and two Dex3 hands. Its output shares the same final 28-D G1/Dex3 joint convention and the same common temporal inverse-kinematics / generic-feasibility machinery used for the fairness-corrected A evaluation; only the representation adapter and motion source differ.

**ACT-A / ACT-B.** Both are official LeRobot ACT policies trained from fresh, same-seed initialization under an identical 100,000-step budget. ACT-A uses the Fair-A TRAIN40 labels; ACT-B uses Proposed-B TRAIN40 labels. The only allowed training-configuration substitutions were dataset identity/root, output directory, and job name. Each has a method-specific MEAN_STD normalization fit on its own matched TRAIN40 data. Evaluation is strict-reload, `eval` mode, raw 50-step chunks with no temporal ensemble or smoothing, on the same HELDOUT8 ALOHA RGB/probe inputs and each method's own 28-D state/target.

### Shared target action representation

Both policy datasets use 28 named joints: 14 G1 arm joints (7 left + 7 right), 7 left Dex3 joints, and 7 right Dex3 joints. The source ALOHA action accepted by the frozen retargeting model is `[T, 14]`; that source action is not used as the G1 policy target. Dataset state/action are the method-specific 28-D G1/Dex3 labels."""


def metric_definitions(include_policy: bool = False) -> str:
    policy = ""
    if include_policy:
        policy = r"""

For policy prediction, the same geometric definitions are applied after named-joint forward kinematics to every valid frame of each raw ACT chunk. The source wrist target is transformed from model to world coordinates; the source interaction-frame positions are the byte-identical A/B reference geometry. For a probe (k) with (V_k) valid predicted frames, per-episode values are weighted averages of its nine probe means using (V_k). In this frozen evaluation every probe has (V_k=50), so each held-out episode contributes 450 valid predicted frames.

Action RMSE, although not plotted in this figure, is defined against each method's own held-out retargeted action target. For a set of valid action elements (mathcal{V}),

\[
\operatorname{RMSE}=\sqrt{\frac{1}{|\mathcal{V}|}\sum_{i\in\mathcal{V}}(\hat q_i-q_i)^2}\quad[\mathrm{rad}].
\]
"""
    return r"""Let (e) index a source episode, (t) a frame, and (h\in\{L,R\}) a hand. All position arrays are stored in metres and multiplied by (1000) for publication in millimetres. Lower is better for every error metric.

**Wrist trajectory error.**

\[
e_{w,e,t,h}=\left\|\mathbf p^{\mathrm{ach}}_{w,e,t,h}-\mathbf p^{\mathrm{src}}_{w,e,t,h}\right\|_2.
\]

For A, the achieved/source arrays are `achieved_*_realization_frame_position_world` and `source_*_realization_frame_position_world`. For B, `target_*_wrist_position_model` is mapped into world coordinates with the recovered frozen model-to-world affine transform and compared with `achieved_*_wrist_position_world`. The affine fit was accepted only with maximum residual below (10^{-6}) m.

**Whole-hand interaction error.** If (mathbf p_g) is the achieved physical whole-hand grasp-frame origin and (mathbf p_I) is the source target interaction-frame origin,

\[
e_{g,e,t,h}=\left\|\mathbf p^{\mathrm{ach}}_{g,e,t,h}-\mathbf p^{\mathrm{src}}_{I,e,t,h}\right\|_2.
\]

The scalar pool contains both hands at every frame.

**Bimanual relation error.**

\[
e_{b,e,t}=\left\|\left(\mathbf p^{\mathrm{ach}}_{g,R}-\mathbf p^{\mathrm{ach}}_{g,L}\right)-\left(\mathbf p^{\mathrm{src}}_{I,R}-\mathbf p^{\mathrm{src}}_{I,L}\right)\right\|_2.
\]

This is translation-relation preservation, not a physical grasp-success metric.

**Feasibility projection magnitude.** `feasibility_projection_translation_m[e,t,h]` stores the translation applied by the shared nearest-feasible target resolver. The resolver's local projection is described as

\[
\underset{\mathbf r}{\operatorname{argmin}}\;\|\mathbf r-\mathbf s\|_2^2
\quad\text{subject to}\quad
\|\mathbf a-\mathbf r\|_2\leq\epsilon,
\]

where (mathbf s) is the unmodified source target, (mathbf r) the realizable target, and (mathbf a) the achieved realization under the frozen common constraints. A projection does not waive collision, limit, branch, or hard-IK validation. Orientation projection is zero in the Fair-A repair.

**Aggregation.** An episode mean is the arithmetic mean of all scalar samples in that episode (two hands per frame for wrist, whole-hand, and projection; one relation per frame for bimanual). A frame-weighted full50 mean concatenates the relevant samples across all episodes. The 50 episodes contain 34,478 frames, yielding 68,956 hand-frame samples and 34,478 bimanual samples.""" + policy


def common_style_table() -> str:
    return md_table(
        ["Element", "Recovered specification"],
        [
            ("Background", "white (`figure.facecolor`, `axes.facecolor`, and save face color)"),
            ("Typography", "serif; configured fallback `DejaVu Serif`, `Times New Roman`, `Times`; local font resolution was DejaVu Serif"),
            ("Base / axis label / panel title", "7.2 / 7.5 / 7.8 pt"),
            ("Tick / legend / panel-label", "6.7 / 6.6 / 8.0 pt; panel labels bold"),
            ("Axis / default line", "0.65 / 1.1 pt"),
            ("Default marker size", "4.6 pt"),
            ("Tick width / length", "0.6 / 2.6 pt"),
            ("Grid", "`#D9D9D9`, 0.48 pt, alpha 0.78; drawn below data"),
            ("Ink / secondary text", "`#252525` / `#717171`"),
            ("A", "`#3F5F8F`; circle marker; solid line where a line is used; light fill `#DCE3ED`"),
            ("B", "`#A9553B`; square marker; dashed line where a line is used; light fill `#EDDCD6`"),
            ("Outcome colors", "CLEAN `#6D8C73`, WARNING `#C3A64B`, HARD `#9B4A42`; hatches `//`, `..`, `xx`"),
            ("Vector text", "PDF font type 42; SVG text retained (`svg.fonttype = none`)"),
            ("Raster export", "600 dpi; embedded PNG DPI is 599.9988 in each axis"),
        ],
    )


def reconstruction_checklist() -> str:
    return """- [x] exact source files identified
- [x] exact numerical values transcribed
- [x] metric definitions included
- [x] episode provenance included
- [x] visual encoding documented
- [x] caption included
- [x] reproduction command included
- [x] claims and caveats documented
- [x] PNG hash recorded"""


def figure_identity(fig: dict[str, Any], copy_path: Path, git_head: str, dirty: bool) -> str:
    number = fig["number"]
    folder = P["final_root"] / fig["folder"]
    original = folder / f"{fig['stem']}.png"
    meta_path = folder / f"{fig['stem']}_metadata.json"
    metadata = read_json(meta_path)
    with Image.open(original) as image:
        pixels = image.size
        dpi = image.info.get("dpi", (0, 0))
        mode = image.mode
        image.verify()
    stat = original.stat()
    created = datetime.fromtimestamp(stat.st_mtime, ZoneInfo("Asia/Seoul")).isoformat()
    png_hash = sha256(original)
    copied_hash = sha256(copy_path)
    scripts = [P["plot_script"], P["bank_script"], P["pub_script"]]
    if number == 4:
        scripts += [P["a_report_script"], P["a_wrist_resolver"], P["a_full_resolver"], P["b_evaluator"], P["b_solver"]]
    if number == 5:
        scripts += [P["eval_script"]]
    script_rows = [(f"`{x.resolve()}`", sha256(x)) for x in scripts]
    return f"""- Paper figure number: **Fig. {number}**
- Figure name: **{fig['name']}**
- Internal filename: `{fig['stem']}.png`
- Archival filename: `{copy_path.name}`
- Original PNG: `{original.resolve()}`
- Original vector PDF: `{(folder / (fig['stem'] + '.pdf')).resolve()}`
- Original vector SVG: `{(folder / (fig['stem'] + '.svg')).resolve()}`
- Original rendering script: `{P['plot_script'].resolve()}`
- Original generation command: `{metadata['generation_command']}`
- Metadata artifact: `{meta_path.resolve()}` (SHA256 `{sha256(meta_path)}`)
- Filesystem modification timestamp (best recoverable PNG creation proxy): `{created}`
- Original PNG SHA256: `{png_hash}`
- Archival PNG SHA256: `{copied_hash}`
- Byte-for-byte copy: **{'YES' if png_hash == copied_hash else 'NO'}**
- Original/archival size: `{stat.st_size}` / `{copy_path.stat().st_size}` bytes
- Pixel dimensions / mode: `{pixels[0]} × {pixels[1]}` / `{mode}`
- Embedded DPI: `{dpi[0]:.4f} × {dpi[1]:.4f}`
- Live repository Git HEAD: `{git_head}`
- Live worktree state: **{'DIRTY' if dirty else 'CLEAN'}**. The final PNG and all scientific inputs are hash-pinned; the current dirty state is not used to regenerate the figure.
- Generation-time Git commit: the Fair-A manifest records `{read_json(P['a_manifest']).get('git_head', 'NOT RECOVERED')}`. A separate commit for the untracked final plotting script is **NOT RECOVERED**; its content hash below is authoritative.

### Relevant implementation hashes

{md_table(['Implementation file', 'SHA256'], script_rows)}"""


def full_identity_table(s1: list[dict[str, str]]) -> str:
    return md_table(
        ["Final index", "Stable source episode ID", "Frames", "A outcome", "B outcome"],
        [
            (r["episode_index"], r["stable_episode_id"], r["frame_count"], r["a_status"], r["b_status"])
            for r in s1
        ],
    )


def tradeoff_episode_table(s1: list[dict[str, str]]) -> str:
    return md_table(
        ["Ep.", "Stable source ID", "T", "A wrist", "B wrist", "A whole", "B whole", "A bimanual", "B bimanual"],
        [
            (
                r["episode_index"], r["stable_episode_id"], r["frame_count"],
                r["a_wrist_mean_mm"], r["b_wrist_mean_mm"],
                r["a_whole_hand_mean_mm"], r["b_whole_hand_mean_mm"],
                r["a_bimanual_mean_mm"], r["b_bimanual_mean_mm"],
            )
            for r in s1
        ],
    )


def effect_episode_table(s1: list[dict[str, str]]) -> str:
    def difference(row: dict[str, str], metric: str) -> str:
        value = float(row['b_' + metric + '_mean_mm']) - float(row['a_' + metric + '_mean_mm'])
        return ("+" if value >= 0 else "") + repr(value)

    return md_table(
        ["Ep.", "Stable source ID", "Δ wrist", "Δ whole", "Δ bimanual", "Δ projection"],
        [
            (r["episode_index"], r["stable_episode_id"], difference(r, "wrist"), difference(r, "whole_hand"), difference(r, "bimanual"), difference(r, "projection"))
            for r in s1
        ],
    )


def aggregate_error_table(ecdf: list[dict[str, str]], projection: list[dict[str, str]]) -> str:
    rows = []
    for r in ecdf:
        if r["record_type"] == "summary":
            rows.append((r["metric"], r["method"], r["mean_mm"], r["median_mm"], r["p95_mm"], r["max_mm"], "mm"))
    for r in projection:
        if r["record_type"] == "summary":
            rows.append(("Projection magnitude", r["method"], r["mean_mm"], r["median_mm"], r["p95_mm"], r["max_mm"], "mm"))
    return md_table(["Metric", "Method", "Mean", "Median", "p95", "Maximum", "Unit"], rows)


def tradeoff_summary_table(fig2_rows: list[dict[str, str]]) -> str:
    rows = []
    for r in fig2_rows:
        if r["record_type"] == "frame_weighted_summary":
            rows.append((r["metric"], r["method"], r["mean_mm"], r["paired_cluster_bootstrap_ci95_low_mm"], r["paired_cluster_bootstrap_ci95_high_mm"], r["seed"], "50", "20,000"))
    return md_table(["Panel metric", "Method", "Filled-marker mean [mm]", "CI low", "CI high", "Effective RNG seed", "Episodes", "Resamples"], rows)


def policy_summary_table(fig5_rows: list[dict[str, str]]) -> str:
    rows = []
    for r in fig5_rows:
        if r["record_type"] != "frame_weighted_summary":
            continue
        metric_index = {"Wrist": 0, "Whole-hand": 1, "Bimanual": 2}[r["metric"]]
        effective_seed = 20260827 + (100 if r["level"] == "Retargeted supervision" else 200) + metric_index
        rows.append(
            (
                r["level"], r["metric"], r["method"], r["mean_mm"],
                r["paired_cluster_bootstrap_ci95_low_mm"], r["paired_cluster_bootstrap_ci95_high_mm"],
                effective_seed, r["episodes"] or r["heldout_episodes"], r["resamples"],
            )
        )
    return md_table(["Row", "Metric", "Method", "Plotted mean [mm]", "CI low", "CI high", "Effective RNG seed", "n episodes", "Resamples"], rows)


def heldout_episode_table(fig5_rows: list[dict[str, str]], heldout: dict[str, Any]) -> str:
    records: dict[int, dict[str, Any]] = {}
    stable = {int(e["final_dataset_index"]): e["stable_episode_id"] for e in heldout["entries"]}
    for r in fig5_rows:
        if r["record_type"] != "heldout_episode_mean":
            continue
        episode = int(r["episode_index"])
        record = records.setdefault(episode, {"id": stable[episode], "valid": r["valid_predicted_frames"], "probes": r["probe_count"]})
        record[f"{r['method']}_{r['metric']}"] = r["sample_mean_mm"]
    rows = []
    for episode in sorted(records):
        r = records[episode]
        rows.append((episode, r["id"], r["probes"], r["valid"], r["ACT-A_Wrist"], r["ACT-B_Wrist"], r["ACT-A_Whole-hand"], r["ACT-B_Whole-hand"], r["ACT-A_Bimanual"], r["ACT-B_Bimanual"]))
    return md_table(["Final ep.", "Stable source ID", "Probes", "Valid frames / metric", "ACT-A wrist", "ACT-B wrist", "ACT-A whole", "ACT-B whole", "ACT-A bimanual", "ACT-B bimanual"], rows)


def heldout_probe_table(heldout: dict[str, Any]) -> str:
    phases = ["initial_approach", "pre_grasp", "left_grasp_owned", "left_transport", "handoff_approach", "dual_contact_transfer", "right_owned", "right_transport", "release"]
    stable = {int(e["final_dataset_index"]): e["stable_episode_id"] for e in heldout["entries"]}
    rows = []
    for episode in sorted(int(x) for x in heldout["complete_source_phase_audit"]):
        frames = heldout["complete_source_phase_audit"][str(episode)]["nine_probe_frames"]
        rows.append((episode, stable[episode], *(frames[p] for p in phases)))
    return md_table(["Ep.", "Stable ID", "Initial approach", "Pre-grasp", "Left grasp owned", "Left transport", "Handoff approach", "Dual contact", "Right owned", "Right transport", "Release"], rows)


def exact_table_csv(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "(empty)"
    return md_table(list(rows[0]), [[r[k] for k in rows[0]] for r in rows])


def figure_1_doc(data: dict[str, Any], fig: dict[str, Any], copy_path: Path) -> str:
    s1 = data["s1"]
    stages = read_csv(P["final_root"] / fig["folder"] / f"{fig['stem']}_source_data.csv")
    artifacts = artifact_table([
        (P["final_root"] / fig["folder"] / f"{fig['stem']}_source_data.csv", "ordered stage/branch/description records", "conceptual pipeline"),
        (P["table1"], "full50 method labels, episode counts, evaluation metrics", "FULL50"),
        (P["a_manifest"], "fairness-corrected A provenance, trajectory hashes, classifications", "FULL50"),
        (P["b_manifest"], "frozen B source ordering, trajectories, classifications", "FULL50"),
        (P["common48"], "paired feasible source identities and construction", "COMMON48"),
        (P["train40"], "training identities and split contract", "TRAIN40"),
        (P["heldout8"], "held-out identities, probes, split seed", "HELDOUT8"),
        (P["training_contract"], "ACT architecture/budget/initialization/normalization", "TRAIN40"),
        (P["exp2"], "held-out evaluation contract and selected results", "HELDOUT8"),
        (P["b_freeze"], "28-D action schema and B frozen implementation hashes", "FULL50"),
    ])
    caption = data["captions"][1]
    return f"""# FIGURE 01 — COMPLETE PROVENANCE

Archive-local image: [figure_01.png](figure_01.png)

## A. FIGURE IDENTITY

{figure_identity(fig, copy_path, data['git_head'], data['dirty'])}

## B. SCIENTIFIC QUESTION

**Question:** {fig['question']}

This methods figure defines the controlled comparison before any result is interpreted. It supports the paper's fairness argument: A and B differ at the retargeting representation, then pass through shared G1 realization and matched policy/evaluation machinery. A reader should conclude that the experiment isolates representation-dependent supervision characteristics under a common backend. The reader must **not** infer real-G1 execution, closed-loop success, physical doll handoff, onboard perception, XR, or D455 deployment; none appears in this pipeline.

## C. METHODS / CONDITIONS SHOWN

{common_method_definitions()}

The flow is deliberately symmetric. “SHARED G1 REALIZATION” means the frozen common temporal DLS plus SEW/null-space arm backend and generic feasibility processing; A uses a wrist-frame adapter and B an interaction-frame adapter. Both produce 28-D G1/Dex3 dataset labels. The figure's “common evaluation” box names wrist fidelity, whole-hand interaction, bimanual relation, and held-out prediction; it is not a closed-loop rollout box.

## D. EXACT DATA SOURCES

{artifacts}

No episodes are excluded at the FULL50 retargeting stage. COMMON48 removes only final Fair-A HARD episodes 35 and 46 from **both** methods before the policy split. No Experiment-3 artifact was read to draw this figure.

## E. EXACT NUMERICAL VALUES

### Pipeline records transcribed from the final figure source CSV

{exact_table_csv(stages)}

### Scope and dimensionality

{md_table(['Quantity', 'Exact value', 'Meaning'], [
    ('ALOHA source demonstrations', 50, 'final common source order 0–49'),
    ('FULL50 frames', '34,478', 'same source-frame count for A and B'),
    ('Source ALOHA action', '14-D', 'two 7-D ALOHA arms/grippers; accepted as `[T,14]`'),
    ('G1/Dex3 state/action', '28-D', '14 arm + 7 left Dex3 + 7 right Dex3'),
    ('COMMON48', 48, 'FULL50 minus A hard episodes 35 and 46, removed symmetrically'),
    ('TRAIN40', 40, 'matched training sources'),
    ('HELDOUT8', 8, 'matched held-out sources'),
    ('Held-out probes', '72 = 8 × 9', 'nine frozen semantic probe frames per held-out episode'),
    ('Predicted chunk', '50 × 28', '50 action steps, 28 named joints'),
    ('ACT training budget', '100,000 steps', 'batch 8, chunk 50, AMP disabled'),
])}

### Exact FULL50 identity/order

{full_identity_table(s1)}

## F. STATISTICS

This schematic contains no error bar and performs no statistical test. The pipeline's quantitative branches are documented in Figs. 2–5. Where those figures use bootstrap intervals, the common base seed is 20260827 and the figure-specific effective streams are recorded in their own provenance files.

## G. METRIC DEFINITIONS

{metric_definitions(include_policy=True)}

## H. EPISODE / SPLIT PROVENANCE

Final episode indices 0–49 are the frozen source-collection indices. Final indices 13 and 36 are in-place replacements (`doll_handoff_20260823_135848` and `doll_handoff_20260823_140035`) already present in the authoritative B source manifest and therefore used identically by A. COMMON48 excludes A hard episodes 35 and 46 from both branches. The split uses NumPy `Generator(PCG64)` seed **20260826**: permute ascending COMMON48 indices, take the first eight as held-out, use the remaining 40 for training, then sort within each packaged subset. HELDOUT8 is `[2, 13, 23, 27, 28, 31, 37, 40]`; selection was frozen before training and not performance-based.

## I. VISUAL ENCODING

- The top neutral rectangle is the shared set of 50 source demonstrations.
- The left blue branch is A; the right rust branch is B. Equal box size and symmetric arrows encode equal experimental standing, not equal metric values.
- The method boxes list the only intended representational contrast.
- The full-width gray box is shared realization, with a faint vertical dashed divider emphasizing parallel A/B adapters within common machinery.
- The paired Dataset and ACT boxes show method-specific supervision and policies.
- Converging arrows lead to the neutral common-evaluation box.
- Arrowheads encode processing direction only; box area, position, or color intensity does not encode a quantitative result.

## J. EXACT FIGURE DESIGN SPEC

{common_style_table()}

{md_table(['Design item', 'Exact recovered value'], [
    ('Recommended format / final width', 'double-column / 178.0 mm'),
    ('Canvas', '4204 × 1710 px; aspect ratio 2.458480'),
    ('Matplotlib figure size', '178 mm × 2.85 in'),
    ('Normalized plotting coordinates', 'x and y both [0,1]; axes hidden'),
    ('Primary box border', '0.85 pt'),
    ('Arrow', '`-|>`; `#717171`; 0.75 pt; mutation scale 7'),
    ('Source box', 'x=.37, y=.89, w=.26, h=.075'),
    ('A/B representation boxes', 'A (.035,.66,.41,.145); B (.555,.66,.41,.145)'),
    ('Shared realization box', '(.035,.46,.93,.105)'),
    ('Dataset boxes', 'A (.12,.315,.25,.075); B (.63,.315,.25,.075)'),
    ('ACT boxes', 'A (.16,.19,.17,.068); B (.67,.19,.17,.068)'),
    ('Evaluation box', '(.25,.025,.50,.095)'),
    ('Margins / subplot parameters', 'axes fill from `plt.subplots`; no explicit `subplots_adjust`'),
    ('Internal title', 'none'),
])}

## K. PAPER CAPTION

### Exact final English caption

{caption}

### Korean explanation

50개의 동일한 ALOHA 시연에서 출발하여, 손목 6-D 궤적을 옮기는 A와 상호작용 프레임·전체 손 형상·양손 관계·소유권 전이를 표현하는 B를 공정하게 비교한다. 두 분기는 동일한 G1 시간적 IK와 일반 타당성 실현을 거친 뒤 각각 Dataset A/B와 ACT-A/B를 만들며, 공통 지표로 평가된다.

### Extended panel/flow explanation

This is a single-flow schematic rather than a multi-panel plot. The upper fork locates the scientific intervention, the middle shared box locates the controlled backend, and the lower reconvergence locates the common evaluation. It explicitly prevents the mistaken reading that B received a stronger generic IK/feasibility solver.

## L. PAPER TEXT CONNECTION

The final-six manifest recommends this figure for **Methods** and describes it as the answer to “What exactly is compared?” No current manuscript `.tex`, `.docx`, `.odt`, or submission draft was present in the repository, so an exact manuscript sentence is **NOT RECOVERED**. The figure supports a methods paragraph explaining the isolated representation contrast and should be read before Table 1 (FULL50 retargeting) and Table 2 (HELDOUT8 ACT prediction).

## M. REPRODUCTION PIPELINE

```text
frozen A/B manifests + Table 1 + HELDOUT8 Experiment 2
→ ordered conceptual stage records
→ build_method_overview(double)
→ Matplotlib vector canvas
→ authoritative 600-dpi PNG
→ byte-for-byte archival copy
```

Exact original command (do not run unless intentionally rebuilding all final figures):

```bash
cd /home/jbnu/aloha_g1_dataset
python3 tools/generate_jkros_final_figures.py
```

Hash-only verification used for this archive:

```bash
sha256sum outputs/paper_final_figures/Fig01_Method_Overview/Fig01_Method_Overview.png \
  outputs/paper_final_5_archive/FIGURE_01/figure_01.png
```

## N. VALIDATION / INTEGRITY CHECKS

- Existing final validation report: **PASS**, 164 checks passed and 0 failed for the six-figure source package.
- Recommended double-column PNG equals the main PNG byte-for-byte.
- Source hashes in figure metadata were rechecked.
- The archive copy hash equals the source PNG hash.
- FULL50 A/B source IDs, ordering, and frame counts match for all 50 entries.
- No scalar result was recomputed or changed; no GPU, simulator, training, or inference was invoked.
- No Experiment-3, placeholder, or physical-execution element appears.

## O. INTERPRETATION

**Main takeaway:** the scientific comparison changes representation, not the shared realization/evaluation backend. **Secondary takeaway:** the distinction propagates into two matched policy-supervision datasets and policies. **Trade-off context:** later figures show that A preserves the wrist trajectory while B preserves whole-hand/bimanual interaction geometry more closely.

## P. LIMITATIONS / CAVEATS

- The diagram is an experimental abstraction, not proof that every implementation detail is identical; the hash-pinned contracts provide that audit trail.
- Results are offline/simulation-side geometry and feasibility, not physical G1 task success.
- It does not claim onboard visual autonomy, closed-loop control, XR, D455 deployment, or a real doll handoff.
- “Shared feasibility” follows a fairness correction to the original A baseline; the archived A is the corrected result, not the pre-fairness run.
- The box “held-out prediction” does not include Experiment 3.

## Q. ORAL PRESENTATION NOTES

### 20-second Korean explanation

“50개 ALOHA 시연에서 표현만 바꿉니다. A는 6-D 손목 궤적, B는 전체 손과 양손 상호작용을 표현하며, 그 뒤의 G1 시간적 IK와 타당성 처리, 학습 예산, 평가 절차는 공통입니다.”

### 60-second Korean explanation

“이 그림은 공정성의 핵심을 보여 줍니다. 동일한 50개 ALOHA 시연을 두 분기로 나누되 A는 독립적인 6-D 손목 궤적을, B는 상호작용 프레임·전체 손 형상·양손 관계·소유권 전이를 사용합니다. 이후에는 동일한 G1 시간적 IK와 일반 타당성 백엔드를 적용해 28-D G1/Dex3 Dataset A와 B를 만들고, 같은 ACT 설정으로 학습합니다. 평가는 손목, 전체 손, 양손 관계 및 held-out 예측으로 공통화했습니다. 따라서 뒤의 차이는 전체 시스템 우열이 아니라 표현 선택에 따른 특성으로 해석해야 합니다.”

### Likely reviewer questions

1. **Q: Did B use a stronger IK solver?** A: No. The fairness-corrected A uses the frozen common temporal arm backend and generic feasibility resolver through a wrist-frame adapter; B uses an interaction-frame adapter.
2. **Q: Are the policy datasets the same?** A: Source RGB, episode identities, splits, action dimensionality, architecture, and budget are matched; target state/action labels differ because they come from A versus B retargeting.
3. **Q: Does the diagram claim real-robot success?** A: No. It ends at common offline/held-out evaluation and contains no real-G1 or closed-loop task-success branch.

## R. RECONSTRUCTION CHECKLIST

{reconstruction_checklist()}
"""


def figure_2_doc(data: dict[str, Any], fig: dict[str, Any], copy_path: Path) -> str:
    s1, fig2 = data["s1"], data["fig2"]
    caption = data["captions"][2]
    artifacts = artifact_table([
        (P["final_root"] / fig["folder"] / f"{fig['stem']}_source_data.csv", "300 episode means + 6 frame-weighted summary/CI rows", "FULL50"),
        (P["s1"], "episode identity, frame count, A/B means/medians/p95/max/status", "FULL50"),
        (P["table1"], "published rounded mean/p95/max values", "FULL50"),
        (P["a_manifest"], "A trajectory set/hash and fairness conditions", "FULL50"),
        (P["b_manifest"], "B trajectory set/hash and source order", "FULL50"),
        (P["a_b_comparison"], "full-precision aggregate errors", "FULL50"),
        (P["ecdf_errors"], "full-frame distribution summaries used here only to transcribe median/p95/max", "FULL50"),
        (P["ecdf_projection"], "projection distribution summary for cross-figure audit", "FULL50"),
    ])
    return f"""# FIGURE 02 — COMPLETE PROVENANCE

Archive-local image: [figure_02.png](figure_02.png)

## A. FIGURE IDENTITY

{figure_identity(fig, copy_path, data['git_head'], data['dirty'])}

## B. SCIENTIFIC QUESTION

**Question:** {fig['question']}

This is the primary quantitative figure. It tests whether preserving the source wrist trajectory is equivalent to preserving the geometry that mediates manipulation. It supports the paper's trade-off claim: A is wrist-fidelity oriented, whereas B is whole-hand/bimanual-interaction oriented. The reader should not conclude that B is universally better, that either method physically completes the doll handoff, or that these are closed-loop policy-success measurements.

## C. METHODS / CONDITIONS SHOWN

{common_method_definitions()}

Both methods are evaluated against the same 50 frozen source demonstrations and their same frame sequence. The two final A HARD episodes (35 and 46) remain in this FULL50 quality figure; nothing is hidden or imputed.

## D. EXACT DATA SOURCES

{artifacts}

All 50 final source identities are included. No episode is excluded. The A trajectory path for each source is `outputs/fair_a_full50_hard_fail_audit/after_full_pose/trajectories/<stable_id>.npz`; the B path is `outputs/doll_handoff_dataset_b_final/retargeted_actions/trajectories/episode_<index:06d>.npz`. Every path was verified against its frozen manifest hash before the final source CSV was created.

## E. EXACT NUMERICAL VALUES

### Plotted frame-weighted summary markers and confidence intervals

{tradeoff_summary_table(fig2)}

The in-panel annotations round these means to one decimal: wrist `3.7 vs 77.9 mm`; whole-hand `90.8 vs 21.2 mm`; bimanual `86.8 vs 36.0 mm`. The paper caption/table use three decimals.

### Full-frame descriptive statistics

{aggregate_error_table(data['ecdf_errors'], data['ecdf_projection'])}

### Every plotted episode-level open marker (episode mean, mm)

{tradeoff_episode_table(s1)}

Raw positions are stored in metres. Transformation is exactly `error_mm = 1000 × Euclidean_error_m`. No normalization, log scale, clipping, or interpolation is applied.

## F. STATISTICS

- Design: paired by source episode; 50 clusters; A and B resample the **same** episode indices together.
- Summary marker: frame-weighted/concatenated-frame mean. Episode weight is its frame count (T_e); for two-hand metrics the factor two cancels in the weighted mean.
- Bootstrap: 20,000 cluster resamples of size 50 with replacement.
- CI: empirical percentile interval `[2.5th, 97.5th]` of the resampled frame-weighted mean for each method.
- Base seed: 20260827. Effective NumPy `default_rng` seeds are **20260837** (wrist), **20260838** (whole-hand), and **20260839** (bimanual), as transcribed from the final figure CSV.
- The A/B method estimates are resampled with paired source indices, but the displayed vertical intervals are marginal intervals around each method mean—not an interval for B−A. The latter appears in Fig. 3.
- No p-value, null-hypothesis test, or statistical-significance claim is made.

## G. METRIC DEFINITIONS

{metric_definitions()}

## H. EPISODE / SPLIT PROVENANCE

Scope is FULL50: final indices 0–49, 34,478 source frames. Episode order and stable IDs are fully transcribed in the table in Section E. No training split is involved. A HARD episodes 35 (`doll_handoff_20260820_ep035`, 692 frames) and 46 (`doll_handoff_20260820_ep046`, 688 frames) are present in both A/B pairs. Final indices 13 and 36 are the frozen in-place replacement source recordings documented by the final B source manifest; A was regenerated on that exact final-common source order.

## I. VISUAL ENCODING

- Panels (a), (b), and (c) show wrist, whole-hand, and bimanual errors, respectively.
- x=0/A: blue open circles for 50 episode means; x=1/B: rust open squares.
- Deterministic horizontal offsets span −0.075 to +0.075 solely to prevent marker overplotting; they do not encode another variable.
- Large filled circle/square: full-frame mean, not the median.
- Vertical whisker with caps: method-specific 95% paired episode-cluster bootstrap interval.
- y-axis: error in millimetres; lower is better; lower limit is zero.
- Light horizontal grid guides reading. No bar area, violin density, regression, or significance symbol is used.
- The common legend maps both color and marker shape, retaining grayscale distinguishability.

## J. EXACT FIGURE DESIGN SPEC

{common_style_table()}

{md_table(['Design item', 'Exact recovered value'], [
    ('Recommended format / final width', 'double-column / 178.0 mm'),
    ('Canvas', '4204 × 1452 px; aspect ratio 2.895317'),
    ('Matplotlib figure size', '178 mm × 2.42 in'),
    ('Subplots', '1 × 3'),
    ('Margins', 'left=.075, right=.985, bottom=.18, top=.79, wspace=.38'),
    ('x limits / ticks', '[-0.30, 1.30]; ticks 0=A, 1=B'),
    ('y limits', 'lower=0; upper NOT EXPLICITLY FIXED (Matplotlib autoscale from points/CI)'),
    ('Episode symbols', 'scatter area 9 pt²; white face; edge width .42; alpha .58'),
    ('Summary symbols', '6.0 pt; filled method color; `#252525` edge .45'),
    ('CI', '1.25 pt line; capsize 2.5 pt'),
    ('Panel label x', '−.18 axes fraction; y=1.04'),
    ('Mean annotation', '6.3 pt, top-center, one decimal'),
    ('Legend', 'top center, 2 columns, no frame'),
    ('Internal figure title', 'none; only compact panel titles'),
])}

## K. PAPER CAPTION

### Exact final English caption

{caption}

### Korean explanation

동일한 50개 시연에서 A는 손목 오차가 3.742 mm로 매우 작지만 전체 손과 양손 관계 오차가 크다. B는 손목 경로에서 더 벗어나지만 전체 손 오차를 21.157 mm, 양손 관계 오차를 35.979 mm로 줄인다. 모든 지표는 낮을수록 좋으며, 작은 빈 표식은 각 에피소드 평균, 큰 채운 표식은 프레임 가중 평균, 수직 막대는 20,000회 부트스트랩 95% 구간이다.

### Extended panel explanation

- **(a) Wrist:** A is the fidelity reference; B's interaction-centered realization permits substantial wrist deviation.
- **(b) Whole-hand:** B moves the physical grasp-frame origin closer to the source interaction-frame origin.
- **(c) Bimanual:** B better preserves the right-minus-left grasp-frame displacement.

## L. PAPER TEXT CONNECTION

The source manifest places this figure in **Results—primary retargeting result**, directly related to Table 1. It supports a sentence paraphrasable as: high wrist-trajectory fidelity does not guarantee preservation of whole-hand or bimanual manipulation geometry. No current manuscript file was found, so exact manuscript wording is **NOT RECOVERED**.

## M. REPRODUCTION PIPELINE

```text
50 hash-verified A NPZ + 50 hash-verified B NPZ
→ source-ID equality check and model→world affine verification
→ per-frame Euclidean errors
→ per-episode means + concatenated-frame means
→ paired cluster bootstrap (20,000; deterministic metric seeds)
→ three-panel Matplotlib plot
→ 600-dpi authoritative PNG
→ byte-for-byte archival copy
```

```bash
cd /home/jbnu/aloha_g1_dataset
python3 tools/generate_jkros_final_figures.py
sha256sum outputs/paper_final_figures/Fig02_Retargeting_Tradeoff/Fig02_Retargeting_Tradeoff.png \
  outputs/paper_final_5_archive/FIGURE_02/figure_02.png
```

The original command regenerates all final figures and is documented for reproducibility only; this archival task did not invoke it.

## N. VALIDATION / INTEGRITY CHECKS

- 50 episode points × 2 methods × 3 metrics = **300** episode rows in the final source CSV.
- Six frame-weighted summary rows are present with finite means and ordered CI bounds.
- A/B stable IDs and frame counts match at every final index.
- All final means round exactly to Table 1: 3.742/77.853, 90.820/21.157, and 86.786/35.979 mm.
- Full-precision wrist aggregates are not bit-identical across older derivative artifacts: the final Fig. 2 CSV stores A/B `3.741914639249444/77.85337558460229` mm, the frozen comparison JSON stores `3.7419145592286923/77.85338704250391` mm, and the publication-bank ECDF summary stores `3.741914987564087/77.85337558460229` mm. These ranges are below `0.0000005` mm for A and `0.000012` mm for B and all round to the authoritative Table-1 values `3.742/77.853` mm. The exact final PNG uses the final Fig. 2 CSV values; the archive does not silently collapse this source-precision discrepancy.
- FULL50 p95/max values match Table 1 and the frozen comparison JSON.
- HARD episodes 35 and 46 are included.
- The existing final-figure validator reports the source PNG nonblank, 600 dpi, correct width, free of placeholder text, and identical to its recommended double-column variant.
- Source and archival PNG hashes match byte-for-byte.

## O. INTERPRETATION

**Main takeaway:** wrist fidelity and interaction fidelity are not interchangeable. **Secondary takeaway:** the directional contrast is distributed over the matched source set rather than being expressed only by one aggregate bar. **Important trade-off:** A strongly minimizes wrist error; B substantially reduces whole-hand and bimanual errors. The appropriate method therefore depends on which geometry the task representation must preserve.

## P. LIMITATIONS / CAVEATS

- These are offline kinematic/geometric errors after shared realization, not semantic task-success rates.
- Whole-hand and bimanual metrics are diagnostic for A and were not A solver targets.
- The figure does not establish causality beyond the controlled representation comparison, nor physical grasp stability.
- A's two hard episodes remain in the metrics; mechanical realizability is addressed separately in Fig. 4.
- Confidence intervals express episode-cluster resampling uncertainty and are not presented as significance tests.
- No real G1, onboard vision, or closed-loop policy claim is supported.

## Q. ORAL PRESENTATION NOTES

### 20-second Korean explanation

“핵심은 우열이 아니라 트레이드오프입니다. A는 손목 오차가 3.742 mm로 작고, B는 손목을 더 움직이는 대신 전체 손과 양손 관계 오차를 각각 21.157 mm와 35.979 mm로 줄입니다.”

### 60-second Korean explanation

“동일한 50개 소스 시연을 에피소드 단위로 비교했습니다. 작은 빈 점은 각 시연의 평균이고, 큰 점은 모든 프레임을 가중한 평균이며, 막대는 에피소드를 함께 재표집한 20,000회 부트스트랩 구간입니다. 패널 (a)에서는 궤적 중심 A가 손목을 훨씬 정확히 보존합니다. 그러나 패널 (b), (c)에서는 상호작용 중심 B가 실제 전체 손 그립 프레임과 양손 상대 관계를 더 정확히 보존합니다. 따라서 손목 궤적만 정확히 옮겼다고 조작 상호작용까지 보존되는 것은 아닙니다.”

### Likely reviewer questions

1. **Q: Are the 50 dots paired?** A: Yes. A and B at each final index use the same stable source episode and frame count; bootstrap resampling uses the same episode indices.
2. **Q: Why are means frame-weighted?** A: The plotted large marker represents the error over the complete frame pool; the small markers retain episode-level heterogeneity. Fig. 3 gives equal weight to episodes for its paired effect.
3. **Q: Does lower whole-hand error prove successful handoff?** A: No. It quantifies representation-level geometric preservation, not contact physics or task success.

## R. RECONSTRUCTION CHECKLIST

{reconstruction_checklist()}
"""


def figure_3_doc(data: dict[str, Any], fig: dict[str, Any], copy_path: Path) -> str:
    effects, s1 = data["fig3"], data["s1"]
    caption = data["captions"][3]
    artifacts = artifact_table([
        (P["final_root"] / fig["folder"] / f"{fig['stem']}_source_data.csv", "four plotted effect/CI rows", "FULL50 paired"),
        (P["forest"], "authoritative stored effects, intervals, seed", "FULL50 paired"),
        (P["s1"], "50 A/B per-episode mean pairs for four metrics", "FULL50 paired"),
        (P["a_manifest"], "A source/trajectory hashes", "FULL50"),
        (P["b_manifest"], "B source/trajectory hashes", "FULL50"),
        (P["table1"], "frame-weighted descriptive values used for cross-check only", "FULL50"),
    ])
    return f"""# FIGURE 03 — COMPLETE PROVENANCE

Archive-local image: [figure_03.png](figure_03.png)

## A. FIGURE IDENTITY

{figure_identity(fig, copy_path, data['git_head'], data['dirty'])}

## B. SCIENTIFIC QUESTION

**Question:** {fig['question']}

This forest plot asks whether the directional A/B contrasts persist when each of the 50 matched demonstrations receives equal weight. It supports the same trade-off argument as Fig. 2 with an explicit paired B−A estimand and uncertainty. The reader should not interpret a confidence interval that does not cross zero as a declared hypothesis test, nor should negative values be called universally “better” without naming the metric.

## C. METHODS / CONDITIONS SHOWN

{common_method_definitions()}

The same final FULL50 A/B trajectory pairs used in Fig. 2 are used here. The paired estimator first reduces each method/episode/metric to its episode mean and then takes B minus A.

## D. EXACT DATA SOURCES

{artifacts}

All final indices 0–49 are included; none is excluded. The effect file was generated from the hash-verified Table S1 episode records and was independently rechecked against frozen NPZ arrays by the final figure generator.

## E. EXACT NUMERICAL VALUES

### Four plotted effects and intervals

{md_table(['Metric', 'Mean B−A [mm]', '95% CI low', '95% CI high', 'n', 'Resamples', 'Seed', 'CI crosses 0?'], [
    (r['metric'], r['paired_B_minus_A_mean_mm'], r['ci95_low_mm'], r['ci95_high_mm'], r['episodes'], r['bootstrap_resamples'], r['seed'], 'NO' if float(r['ci95_low_mm']) * float(r['ci95_high_mm']) > 0 else 'YES') for r in effects
])}

### Every paired episode effect used by the forest estimator

All entries below are `B episode mean − A episode mean` in millimetres. They are derived directly from the A/B values transcribed in Table S1; no interpolation is used.

{effect_episode_table(s1)}

Transformation chain: per-frame metric in metres → multiply by 1000 → episode mean for A and B → subtract B−A → unweighted mean over 50 episodes. This is why the Fig. 3 effect is not required to equal the difference between the frame-weighted Fig. 2 summary markers.

## F. STATISTICS

- Estimand for metric (m):

\[
\widehat\Delta_m=\frac{1}{50}\sum_{{e=1}}^{{50}}\left(\bar e^B_{{m,e}}-\bar e^A_{{m,e}}\right).
\]

- Pairing: A and B share the same episode index/stable ID; no cross-episode pairing.
- Episode weighting: equal (unweighted), unlike the frame-weighted filled markers in Figs. 2 and 5.
- Bootstrap: NumPy `default_rng(20260827)`; for each metric in row order wrist, whole-hand, bimanual, projection, draw a `20000 × 50` matrix of episode indices with replacement; calculate the mean paired difference in each row.
- Random stream nuance: one generator is instantiated once and consumed sequentially across the four metrics. The stored `seed=20260827` identifies that base stream; the generator is **not** reset per row.
- 95% interval: percentile `[2.5, 97.5]` of 20,000 bootstrap means.
- Sign: Δ<0 means lower error for B; Δ>0 means lower error for A because all four quantities are errors.
- None of the four intervals crosses zero. This is reported descriptively, not as a p-value or formal significance declaration.

## G. METRIC DEFINITIONS

{metric_definitions()}

The projection row uses the per-episode mean over two hand-frame projection magnitudes before forming B−A.

## H. EPISODE / SPLIT PROVENANCE

FULL50 includes final indices 0–49 and the two A HARD episodes 35 and 46. It is not COMMON48, TRAIN40, or HELDOUT8. Stable IDs and effects for every pair are listed in Section E. The B source manifest's in-place replacements at final indices 13 and 36 are part of the frozen final source set and occur identically for A. There is no representative-episode selection and no cherry-picking.

## I. VISUAL ENCODING

- y rows, top to bottom: Wrist, Whole-hand, Bimanual relation, Projection magnitude.
- x axis: paired mean difference B−A in millimetres.
- Open black diamond: the mean of 50 paired episode differences.
- Horizontal black whisker with caps: 95% paired-bootstrap percentile interval.
- Vertical gray line at x=0: no mean paired difference.
- Text by each point gives mean and `[low, high]`, rounded to one decimal for compact reading; full precision is in Section E.
- Bottom annotations explicitly decode the sign; there is no A/B color legend and thus no misleading universal “winner” color.
- Light vertical grid supports x-value reading.

## J. EXACT FIGURE DESIGN SPEC

{common_style_table()}

{md_table(['Design item', 'Exact recovered value'], [
    ('Recommended format / final width', 'single-column / 88.0 mm'),
    ('Canvas', '2078 × 1608 px; aspect ratio 1.292289'),
    ('Matplotlib figure size', '88 mm × 2.68 in'),
    ('Subplots', 'single axes'),
    ('Margins', 'left=.32, right=.975, bottom=.37, top=.92'),
    ('x limits', '[-88, +88] mm'),
    ('y limits', '[-.7, 3.55]'),
    ('Zero line', '`#717171`, .75 pt'),
    ('Effect marker', 'open diamond `D`; 4.6 pt; edge .85'),
    ('CI', '1.25 pt; capsize 2.4 pt'),
    ('Value label', '5.8 pt; ±6 point horizontal and +6 point vertical offset'),
    ('Grid', 'x-axis only'),
    ('Internal figure title', 'none'),
])}

## K. PAPER CAPTION

### Exact final English caption

{caption}

### Korean explanation

각 에피소드에서 B−A 오차를 계산한 뒤 50개를 동일 가중으로 평균하였다. 손목은 +74.115 mm이므로 A의 오차가 더 작고, 전체 손·양손 관계·projection은 각각 음수이므로 B의 오차가 더 작다. 점은 평균 차이, 가로 막대는 20,000회 paired bootstrap 95% 구간이다.

### Extended row explanation

- **Wrist:** positive effect; the paired data favor lower wrist error for A.
- **Whole-hand and bimanual:** negative effects; paired episode means are lower for B.
- **Projection:** smaller negative effect; B requires less mean generic-feasibility translation at the episode level, while rare B tail values are not hidden by this mean effect.

## L. PAPER TEXT CONNECTION

The final manifest places the figure in **Results—paired statistical evidence** after Fig. 2. It supports the claim that the observed directions are not solely a consequence of frame-count weighting or a few selected demonstrations. Table 1 gives frame-pooled descriptive means; Fig. 3 gives equal-episode paired effects. No current manuscript was found, so exact section numbering and wording are **NOT RECOVERED**.

## M. REPRODUCTION PIPELINE

```text
Table S1 per-episode A/B means (50 matched rows)
→ Δ_e = B_e − A_e for each metric
→ unweighted mean Δ
→ 20,000 paired episode bootstrap resamples, base seed 20260827
→ stored 04_effect_forest.csv
→ final single-column forest plot
→ byte-for-byte archival PNG copy
```

```bash
cd /home/jbnu/aloha_g1_dataset
python3 tools/generate_jkros_final_figures.py
sha256sum outputs/paper_final_figures/Fig03_Paired_Statistics/Fig03_Paired_Statistical_Effect.png \
  outputs/paper_final_5_archive/FIGURE_03/figure_03.png
```

## N. VALIDATION / INTEGRITY CHECKS

- Exactly four forest rows, each with 50 episodes, 20,000 resamples, and finite ordered bounds.
- Stored effect CSV hash is recorded and its values were compared against frozen-array recomputation by the final generator.
- Per-episode B−A values in Section E were regenerated only as lightweight arithmetic from the frozen Table S1 A/B means.
- Source order and identity agree for all 50 A/B pairs.
- Effects match the final validator at full precision.
- Source PNG is nonblank, 600 dpi, exactly the recommended single-column variant, and contains no placeholder.
- Archival PNG hash equals original.

## O. INTERPRETATION

**Main takeaway:** the episode-level paired directions reproduce the trade-off. **Secondary takeaway:** B also has lower average projection magnitude on an equal-episode basis. **Counterintuitive point:** positive is favorable to A only because the plotted quantity is B−A and lower error is preferred; negative is favorable to B for the same mathematical reason.

## P. LIMITATIONS / CAVEATS

- Bootstrap intervals quantify resampling variability of these 50 demonstrations; they are not a population-wide guarantee.
- No multiple-comparison correction or p-value is reported, and the caption deliberately avoids “statistically significant.”
- Episode-level averaging gives short and long episodes equal weight; Fig. 2 supplies the complementary frame-weighted view.
- Projection is a solver intervention magnitude, not physical task success.
- A's hard episodes are retained; the hard-failure categories are shown in Fig. 4.
- All evidence is offline/simulation-side, with no real-G1 or closed-loop autonomy claim.

## Q. ORAL PRESENTATION NOTES

### 20-second Korean explanation

“50개 대응 시연에서 B−A를 직접 계산했습니다. 손목은 양수라 A가 낮고, 전체 손·양손·projection은 음수라 B가 낮습니다. 모든 부트스트랩 구간은 0의 한쪽에 있지만 유의성 검정으로 주장하지는 않습니다.”

### 60-second Korean explanation

“Fig. 2의 큰 점은 프레임 가중 평균이므로 여기서는 각 시연을 동일하게 취급해 확인했습니다. 각 소스 에피소드에서 B 오차에서 A 오차를 빼고, 50개 paired difference의 평균을 구했습니다. 손목은 +74.115 mm로 A가 유리하고, 전체 손은 −69.678 mm, 양손 관계는 −50.819 mm, projection은 −2.450 mm로 B가 유리합니다. 가로 구간은 동일 에피소드 쌍을 20,000회 재표집한 95% percentile 구간입니다. 이 그림의 목적은 승자를 하나 정하는 것이 아니라 트레이드오프가 대응 데이터 전반에 일관됨을 보여 주는 것입니다.”

### Likely reviewer questions

1. **Q: Why does the effect not exactly equal the difference of Fig. 2 means?** A: Fig. 3 equally weights 50 episode means; Fig. 2 weights by frame count. Both estimands are documented.
2. **Q: Are intervals significance tests?** A: No. They are paired-bootstrap uncertainty intervals; no p-value or formal significance claim is made.
3. **Q: Was the bootstrap paired?** A: Yes. Each resampled index selects the same source episode for A and B before B−A is averaged.

## R. RECONSTRUCTION CHECKLIST

{reconstruction_checklist()}
"""


def figure_4_doc(data: dict[str, Any], fig: dict[str, Any], copy_path: Path) -> str:
    s1 = data["s1"]
    fig4 = data["fig4"]
    caption = data["captions"][4]
    hard_a = [r for r in s1 if r["a_status"] == "HARD_FAIL"]
    a_full = {int(r["episode_index"]): r for r in data["a_full"]}
    artifacts = artifact_table([
        (P["final_root"] / fig["folder"] / f"{fig['stem']}_source_data.csv", "outcome and hard-failure-mode counts", "FULL50"),
        (P["table1"], "final rounded outcome/failure counts", "FULL50"),
        (P["a_per_episode"], "A per-episode classification and failure reasons", "FULL50"),
        (P["a_manifest"], "fairness-corrected A trajectory/classification freeze", "FULL50"),
        (P["b_manifest"], "B per-episode final classifications", "FULL50"),
        (P["a_report"], "backend parity and hard-collision audit narrative", "FULL50"),
        (P["a_projection"], "A shared projection statistics", "FULL50"),
    ])
    return f"""# FIGURE 04 — COMPLETE PROVENANCE

Archive-local image: [figure_04.png](figure_04.png)

## A. FIGURE IDENTITY

{figure_identity(fig, copy_path, data['git_head'], data['dirty'])}

## B. SCIENTIFIC QUESTION

**Question:** {fig['question']}

The figure separates soft diagnostic warnings from mechanical hard failures after shared G1 realization. It supports the bounded statement that B retains zero hard-fail episodes whereas A retains two arm/torso collision failures. It does **not** support ranking the methods by CLEAN count, because WARNING is explicitly usable and not a hard failure.

## C. METHODS / CONDITIONS SHOWN

{common_method_definitions()}

For fairness-corrected A, the common generic repair is frozen bidirectional box-constrained tracking, global full-pose acceptance conditioning, hard SO(3) feasibility, frozen 16/32/48-frame minimum-acceleration windows, signed-distance collision repair, and nearest-feasible translation projection. No A-specific, episode-specific, or phase-specific values were fitted. B uses the frozen generic resolver through its interaction-frame adapter.

## D. EXACT DATA SOURCES

{artifacts}

All 50 episodes per method are shown. The two A HARD episodes are not excluded. No policy split, training output, or Experiment-3 result enters this figure.

## E. EXACT NUMERICAL VALUES

### Exact source rows used by the plot

{exact_table_csv(fig4)}

### Counts and percentages

{md_table(['Method', 'CLEAN', 'CLEAN %', 'WARNING', 'WARNING %', 'HARD', 'HARD %', 'Total'], [
    ('Trajectory-Centric A', 27, '54%', 21, '42%', 2, '4%', 50),
    ('Interaction-Centric B', 13, '26%', 37, '74%', 0, '0%', 50),
])}

### Hard-failure modes

{md_table(['Mode', 'A episode count', 'B episode count'], [('Hard collision', 2, 0), ('Hard IK', 0, 0), ('Joint limit', 0, 0), ('Branch discontinuity', 0, 0)])}

### Hard episode identities and collision evidence

{md_table(['Final episode', 'Stable ID', 'Frames', 'A status', 'B status', 'A shared-hard collision frames', 'Maximum penetration'], [
    (35, hard_a[0]['stable_episode_id'], hard_a[0]['frame_count'], hard_a[0]['a_status'], hard_a[0]['b_status'], a_full[35]['hard_collision_frames_after'], f"{float(a_full[35]['maximum_penetration_after_m']) * 1000!r} mm"),
    (46, hard_a[1]['stable_episode_id'], hard_a[1]['frame_count'], hard_a[1]['a_status'], hard_a[1]['b_status'], a_full[46]['hard_collision_frames_after'], f"{float(a_full[46]['maximum_penetration_after_m']) * 1000!r} mm"),
])}

### Every episode classification

{full_identity_table(s1)}

## F. STATISTICS

This figure reports complete-set counts, not sample estimates with uncertainty. No bootstrap, confidence interval, p-value, or significance test is used. Percentages are exact count/50 conversions. The unit of classification is an episode; collision frame counts are supporting diagnostics and are not used as independent statistical samples.

## G. METRIC DEFINITIONS

**CLEAN_PASS.** No hard reason and no warning reason; stored reason `ALL_HARD_AND_WARNING_GATES_CLEAR`.

**USABLE_WITH_WARNING.** No hard reason, but at least one warning: nonzero feasibility projection, non-hard self-contact, source target outside the strict tolerance, and/or realized target outside the strict tolerance. A warning is explicitly not a hard failure.

**HARD_FAIL.** At least one hard reason: physical hard IK; hard arm/torso, distal hand, or other self-collision; joint-limit violation; branch discontinuity; or acceleration hard-gate breach. Hard reasons dominate warning reasons in episode classification.

**Collision hard-gate rules.** For `DISTAL_HAND_HAND_CONTACT`, depth ≥15 mm and duration ≥0.25 s. For `ARM_TORSO_INVALID`, depth ≥15 mm, or depth ≥5 mm and duration ≥1.0 s. For other classified contacts, depth ≥5 mm or duration ≥0.25 s. A and B final hard-collision counts use the same shared-severity logic.

**Hard IK.** The trajectory remains outside the physical IK acceptance after shared bidirectional/temporal realization; isolated strict-tolerance misses alone do not necessarily make an episode HARD. Final A has zero hard-IK episodes but seven isolated strict frames.

**Joint limit / branch.** Nonzero post-realization joint-limit violation count or branch-discontinuity count, respectively.

{metric_definitions()}

## H. EPISODE / SPLIT PROVENANCE

Scope is FULL50, final indices 0–49 in the exact stable order listed in Section E. A and B are evaluated on the same sources and 34,478 frames. A HARD episodes are 35 and 46; B is warning/clean rather than HARD on those same sources. These episodes are excluded symmetrically only later when constructing COMMON48 for policy learning, not from this figure.

## I. VISUAL ENCODING

- **Panel (a):** horizontal stacked bars, one bar per method, length 50 episodes.
- CLEAN is muted green with `//`; WARNING amber with `..`; HARD red with `xx`. Hatch redundantly encodes status for grayscale.
- White/ink integers centered inside nonzero segments are counts.
- A appears above B; the y axis is inverted to preserve that order.
- **Panel (b):** rows are hard-collision, hard-IK, joint-limit, and branch modes.
- Open blue circle=A; open rust square=B; a thin gray connector joins their counts within a mode.
- The bottom note `WARNING ≠ HARD_FAIL` is an interpretation guard, not data.
- Bar area in panel (a) does not represent quality beyond category count; the figure intentionally avoids a “CLEAN winner” annotation.

## J. EXACT FIGURE DESIGN SPEC

{common_style_table()}

{md_table(['Design item', 'Exact recovered value'], [
    ('Recommended format / final width', 'double-column / 178.0 mm'),
    ('Canvas', '4204 × 1392 px; aspect ratio 3.020115'),
    ('Matplotlib figure size', '178 mm × 2.32 in'),
    ('Subplots', '1 × 2; width ratio 1.30:1.0'),
    ('Margins', 'left=.205, right=.985, bottom=.20, top=.80, wspace=.47'),
    ('Panel (a) x limits', '[0,50] episodes'),
    ('Stacked bar height / edge', '.54 / .55 pt'),
    ('Panel (b) x limits / ticks', '[-.15,2.35]; ticks 0,1,2'),
    ('Panel (b) A/B vertical offsets', '+.09 / −.09'),
    ('Connector', '`#B9B9B9`, .65 pt'),
    ('Legends', 'top-center, frameless; 3 columns in (a), 2 in (b)'),
    ('Bottom note', '6.3 pt, `#717171`'),
    ('Internal figure title', 'none'),
])}

## K. PAPER CAPTION

### Exact final English caption

{caption}

### Korean explanation

공통 G1 시간적 IK와 타당성 처리 후 A는 CLEAN/WARNING/HARD가 27/21/2, B는 13/37/0이다. WARNING은 실패가 아니며, A의 두 HARD는 모두 팔-몸통 충돌이다. 두 방법 모두 hard IK, 관절 한계, branch 실패는 0이다.

### Extended panel explanation

- **(a)** reports the complete classification distribution without collapsing warning into failure.
- **(b)** identifies the mechanical source of the two A hard failures and confirms the other hard modes are zero for both methods.

## L. PAPER TEXT CONNECTION

The final manifest places this figure in **Results—feasibility** and ties it to the feasibility rows in Table 1. It supports a result sentence that B has zero hard-fail episodes after shared realization while A retains two collision failures; it does not support a sentence equating CLEAN with total performance. Exact manuscript prose is **NOT RECOVERED** because no current manuscript artifact was found.

## M. REPRODUCTION PIPELINE

```text
Fair-A final full50 classifications + Proposed-B final source manifest
→ count CLEAN/WARNING/HARD by episode
→ count hard mode flags (collision, IK, limit, branch)
→ verify totals against Table 1
→ two-panel plot
→ 600-dpi authoritative PNG
→ byte-for-byte archival copy
```

```bash
cd /home/jbnu/aloha_g1_dataset
python3 tools/generate_jkros_final_figures.py
sha256sum outputs/paper_final_figures/Fig04_Feasibility/Fig04_Feasibility.png \
  outputs/paper_final_5_archive/FIGURE_04/figure_04.png
```

## N. VALIDATION / INTEGRITY CHECKS

- A totals: 27+21+2=50; B totals: 13+37+0=50.
- Hard-mode sums are consistent: A collision=2 and all other modes=0; B all modes=0.
- Per-episode status table reproduces the aggregate counts.
- Hard episode IDs 35 and 46 agree across A per-episode CSV, A repair manifest, COMMON48 exclusion manifest, Table S1, and the audit report.
- A/B source identity and episode ordering are unchanged.
- Existing final validator confirms exact counts, nonblank 600-dpi PNG, correct double-column width, and no placeholder text.
- Source/copy hashes match.

## O. INTERPRETATION

**Main takeaway:** B retains no hard-fail episode under the shared final realization, while A retains two hard arm/torso collision episodes. **Secondary takeaway:** neither method has final hard IK, joint-limit, or branch failures. **Important caution:** B has fewer CLEAN episodes because many trajectories carry warnings; that does not contradict its zero HARD count.

## P. LIMITATIONS / CAVEATS

- Classification is defined by the frozen kinematic/self-collision feasibility model; it is not a physical hardware safety certification.
- Collision analysis covers modeled robot self-collision and does not prove object-contact success or grasp stability.
- WARNING includes several heterogeneous diagnostics and should not be read as a single severity scale.
- Zero HARD does not mean zero projection or zero warning.
- A hard episodes remain visible here and in FULL50 geometry metrics; policy learning uses COMMON48 with symmetric exclusion.
- No real-G1, onboard-vision, or closed-loop success claim follows.

## Q. ORAL PRESENTATION NOTES

### 20-second Korean explanation

“WARNING은 실패가 아닙니다. 공통 타당성 처리 후 B는 HARD가 0개이고, A는 2개이며 둘 다 팔-몸통 충돌입니다. hard IK, 관절 한계, branch 실패는 양쪽 모두 0입니다.”

### 60-second Korean explanation

“왼쪽은 50개 전체 분류입니다. A는 27 CLEAN, 21 WARNING, 2 HARD이고 B는 13 CLEAN, 37 WARNING, 0 HARD입니다. 여기서 WARNING은 projection이나 비-hard 접촉 같은 진단이므로 HARD_FAIL과 구분해야 합니다. 오른쪽에서 hard mode를 분해하면 A의 두 실패는 에피소드 35와 46의 지속적인 팔-몸통 충돌이고, hard IK·관절 한계·branch 실패는 양쪽 모두 없습니다. 따라서 결론은 CLEAN 개수 경쟁이 아니라 공통 실현 후 B가 hard failure를 남기지 않았다는 제한된 결과입니다.”

### Likely reviewer questions

1. **Q: Why does B have fewer CLEAN episodes but no HARD failures?** A: Projection or non-hard residual/contact diagnostics trigger WARNING; warning is a separate usable category.
2. **Q: Were A's failed episodes removed?** A: No. They are included in FULL50 figures and shown explicitly here; only the later matched policy split excludes them from both methods.
3. **Q: Does zero hard failure establish physical safety?** A: No. It is a result under the frozen G1 kinematic/self-collision feasibility model, not hardware validation.

## R. RECONSTRUCTION CHECKLIST

{reconstruction_checklist()}
"""


def figure_5_doc(data: dict[str, Any], fig: dict[str, Any], copy_path: Path) -> str:
    fig5 = data["fig5"]
    exp2 = data["exp2"]
    heldout = data["heldout"]
    training = data["training"]
    contract = data["training_contract"]
    caption = data["captions"][5]
    artifacts = artifact_table([
        (P["final_root"] / fig["folder"] / f"{fig['stem']}_source_data.csv", "FULL50 episode means/CIs and HELDOUT8 episode means/CIs", "FULL50 + HELDOUT8"),
        (P["table1"], "rounded retargeting means", "FULL50"),
        (P["table2"], "rounded policy geometry/action/phase/control values", "HELDOUT8"),
        (P["exp2"], "selected checkpoints, per-probe geometry, exact aggregates", "HELDOUT8"),
        (P["eval_contract"], "probe, action, geometry and phase definitions", "HELDOUT8"),
        (P["checkpoint_rule"], "predeclared checkpoint ranking", "HELDOUT8"),
        (P["training_contract"], "matched architecture/budget/normalization", "TRAIN40"),
        (P["training_audit"], "checkpoint hashes and training health", "TRAIN40"),
        (P["dataset_audit"], "matched RGB, state/action packaging and dataset hashes", "TRAIN40 + HELDOUT8"),
        (P["common48"], "feasible paired pool", "COMMON48"),
        (P["train40"], "exact training indices", "TRAIN40"),
        (P["heldout8"], "exact held-out indices and nine probe frames", "HELDOUT8"),
        (P["a_manifest"], "A retargeting freeze", "FULL50"),
        (P["b_manifest"], "B retargeting/source freeze", "FULL50"),
    ])
    exact_geometry = []
    for method_key, label in (("a", "ACT-A"), ("b", "ACT-B")):
        geo = exp2["methods"][method_key]["common_source_geometry"]
        for metric_key, metric_label in (("source_wrist_trajectory_fidelity_error_mm", "Wrist"), ("whole_hand_interaction_frame_error_mm", "Whole-hand"), ("bimanual_relation_error_mm", "Bimanual")):
            s = geo[metric_key]
            exact_geometry.append((label, metric_label, s["mean"], s["median"], s["p95"], s["max"], "mm"))
    rankings = []
    for method_key, label in (("a", "ACT-A"), ("b", "ACT-B")):
        for r in exp2["methods"][method_key]["checkpoint_selection"]["ranking_keys"]:
            rankings.append((label, r["step"], r["phase_score"], r["full_valid_chunk_rmse_rad"], r["raw_jerk_rms_rad_s3"], "YES" if r["step"] == exp2["methods"][method_key]["checkpoint_selection"]["selected_step"] else "NO"))
    train_indices = data["train40"]["split_contract"]["train_final_dataset_indices"]
    held_indices = data["train40"]["split_contract"]["heldout_final_dataset_indices"]
    return f"""# FIGURE 05 — COMPLETE PROVENANCE

Archive-local image: [figure_05.png](figure_05.png)

## A. FIGURE IDENTITY

{figure_identity(fig, copy_path, data['git_head'], data['dirty'])}

## B. SCIENTIFIC QUESTION

**Question:** {fig['question']}

This figure connects representation-dependent retargeting supervision to held-out ACT geometry. It supports the paper's second main result: A/ACT-A remains wrist-fidelity oriented and B/ACT-B remains whole-hand/bimanual-interaction oriented. It does not claim that top and bottom rows are paired before/after samples, that B has a higher phase score, or that either policy completes a closed-loop physical task.

## C. METHODS / CONDITIONS SHOWN

{common_method_definitions()}

### Exact selected checkpoints

- ACT-A selected step **100,000**: `{exp2['methods']['a']['checkpoint_selection']['selected_checkpoint']}`; model SHA256 `{exp2['methods']['a']['checkpoint_selection']['selected_model_sha256']}`.
- ACT-B selected step **20,000**: `{exp2['methods']['b']['checkpoint_selection']['selected_checkpoint']}`; model SHA256 `{exp2['methods']['b']['checkpoint_selection']['selected_model_sha256']}`.
- The ACT-B 100,000-step file is the final training checkpoint, but it is **not** the selected evaluation checkpoint. Selection used the same predeclared lexicographic rule for both methods: maximize eight-behavior phase score; then minimize full-valid-chunk RMSE; then minimize raw-chunk jerk; then choose the earlier step for an exact tie.

Training was official LeRobot ACT with ResNet18 ImageNet initialization, training seed 1000, 100,000 steps, batch 8, chunk 50, no AMP, and checkpoints every 20,000 steps. Held-out data were excluded from normalization and training.

## D. EXACT DATA SOURCES

{artifacts}

Top row: all 50 retargeting episodes, including A HARD 35/46. Bottom row: HELDOUT8 only, selected from COMMON48 before training. The two rows are different evaluation sets and are not linked as repeated measurements.

## E. EXACT NUMERICAL VALUES

### Every plotted summary marker and CI

{policy_summary_table(fig5)}

### Exact ACT geometry distribution summaries from Experiment 2

{md_table(['Policy', 'Metric', 'Mean', 'Median', 'p95', 'Maximum', 'Unit'], exact_geometry)}

The filled bottom-row markers were read from rounded Table 2 values (three decimals): ACT-A/ACT-B wrist `24.958/95.269`, whole-hand `107.131/58.330`, and bimanual `120.976/85.920` mm. The exact Experiment-2 means above round to those plotted values. CI calculation uses the full-precision per-episode means.

### All HELDOUT8 episode-level open markers (mm)

{heldout_episode_table(fig5, heldout)}

### All FULL50 top-row episode markers (mm)

{tradeoff_episode_table(data['s1'])}

### Table 2 values associated with this evaluation

These values are not all drawn in Fig. 5, but they identify and audit the same selected policy evaluation.

{exact_table_csv(data['table2'])}

### Checkpoint candidates and selection inputs

{md_table(['Policy', 'Step', '8-behavior phase score /64', 'Full valid chunk RMSE [rad]', 'Raw jerk RMS [rad/s³]', 'Selected'], rankings)}

Final training-health loss at step 100,000 was **0.036** for ACT-A and **0.038** for ACT-B. This loss was not a checkpoint-selection criterion and is not plotted.

## F. STATISTICS

- Within each row A/B data are source-paired; the top and bottom rows are not paired to one another.
- Top-row filled means: concatenated FULL50 frame means; 50 clusters and frame-count weights.
- Bottom-row per-episode means: weighted average of nine probe means by valid predicted frames. All 72 probes have 50 valid frames, so each held-out episode has 450 valid predicted frames per metric.
- Bottom filled means: rounded Table 2 aggregate values; exact full-prediction-frame means are in Section E.
- Bootstrap: 20,000 paired episode-cluster resamples; empirical percentile 95% interval.
- Base seed: 20260827. Effective seeds recovered from code are top wrist/whole/bimanual **20260927/20260928/20260929** and bottom **20261027/20261028/20261029**.
- The final Fig. 5 CSV stores interval endpoints but omits its seed column; effective seeds above are recovered from the hash-pinned plotting implementation (`base + 100/200 + metric index`).
- Fig. 5 top-row CIs differ slightly from Fig. 2 because a different deterministic Monte Carlo stream is used. Point data and means are identical.
- No p-value or significance claim is made.

## G. METRIC DEFINITIONS

{metric_definitions(include_policy=True)}

**Phase score (context, not plotted):** eight frozen behaviors per held-out episode—left approach, left grasp, left transport, right handoff approach, dual-hand configuration, right owned, right transport, release—yield 64 binary opportunities per policy. ACT-A scores 56/64 and ACT-B 52/64. This figure does not imply B wins phase score.

## H. EPISODE / SPLIT PROVENANCE

- FULL50: 0–49; 34,478 frames; used for top row.
- COMMON48: remove only A HARD episodes 35 and 46 from both methods; canonical entry hash `{data['common48']['entries_canonical_sha256']}`.
- Split seed: **20260826**, NumPy `Generator(PCG64)`, frozen before training.
- TRAIN40 indices: `{train_indices}`; canonical entry hash `{data['train40']['entries_canonical_sha256']}`.
- HELDOUT8 indices: `{held_indices}`; canonical entry hash `{heldout['entries_canonical_sha256']}`.
- HELDOUT8 stable IDs are transcribed in the episode table in Section E.
- A/B decoded RGB tensors were bit-exact for all **72/72** probes; feature names, source interaction targets, and episode identities matched. Method-specific 28-D states and targets were intentionally different.

### Exact frozen nine-probe source frames

{heldout_probe_table(heldout)}

The held-out split is not the later representative-episode rule and contains no visual cherry-picking.

## I. VISUAL ENCODING

- Layout is 2 rows × 3 columns. Columns are Wrist, Whole-hand, and Bimanual.
- Top panels (a–c): Retargeted supervision, n=50; labels A/B.
- Bottom panels (d–f): Held-out ACT prediction, n=8; labels ACT-A/ACT-B.
- Small open blue circles and rust squares are episode means; deterministic horizontal jitter only avoids overlap.
- Large filled method markers are the aggregate mean; vertical bars are paired episode-cluster bootstrap intervals within that row.
- Each metric column shares a common y scale across top and bottom so orientation can be compared without asserting sample pairing.
- The left-side vertical row labels explicitly state n=50 and n=8.
- Combined legend `A / ACT-A` and `B / ACT-B` maps representation lineage, not identical sample identity.

## J. EXACT FIGURE DESIGN SPEC

{common_style_table()}

{md_table(['Design item', 'Exact recovered value'], [
    ('Recommended format / final width', 'double-column / 178.0 mm'),
    ('Canvas', '4204 × 1890 px; aspect ratio 2.224339'),
    ('Matplotlib figure size', '178 mm × 3.15 in'),
    ('Subplots', '2 × 3, shared y within each column'),
    ('Margins', 'left=.11, right=.985, bottom=.12, top=.82, hspace=.47, wspace=.33'),
    ('x limits / ticks', '[-.30,1.30]; A/B or ACT-A/ACT-B'),
    ('FULL50 episode points', '6.2 pt²; jitter ±.068; edge .34; alpha .48'),
    ('HELDOUT8 episode points', '10.0 pt²; jitter ±.055; edge .50; alpha .70'),
    ('Summary marker / CI', '5.2 pt; CI 1.05 pt; capsize 2.0'),
    ('Column y upper rule', '1.08 × max of top/bottom episode values and CI upper endpoints; lower=0'),
    ('Row-label text', '6.0 pt bold, `#717171`, rotated 90°'),
    ('Panel label x', '−.22; y=1.04'),
    ('Legend', 'top center, 2 columns, no frame'),
    ('Internal figure title', 'none'),
])}

## K. PAPER CAPTION

### Exact final English caption

{caption}

### Korean explanation

위 행은 50개 전체 retargeting supervision, 아래 행은 8개 held-out 에피소드의 ACT 예측이다. 서로 다른 평가 집합이므로 전후 paired 비교가 아니지만, A/ACT-A는 손목 오차가 낮고 B/ACT-B는 전체 손과 양손 관계 오차가 낮다는 방향이 학습 뒤에도 유지된다.

### Extended panel explanation

- **(a,d) Wrist:** A and ACT-A occupy the lower-error side within their respective rows.
- **(b,e) Whole-hand:** B and ACT-B have lower interaction-frame error.
- **(c,f) Bimanual:** B and ACT-B have lower right-minus-left relation error.
- Shared column scale visualizes pattern preservation; it does not convert the two rows into paired before/after data.

## L. PAPER TEXT CONNECTION

The final manifest recommends **Results—downstream policy**. The figure links Table 1 retargeting geometry to the geometric portion of Table 2 held-out ACT prediction. It supports the bounded sentence that representation-dependent geometry remains visible in held-out policy predictions. It must be accompanied by the separate Table 2 fact that phase score is ACT-A 56/64 versus ACT-B 52/64. Exact manuscript section numbering/wording is **NOT RECOVERED**.

## M. REPRODUCTION PIPELINE

```text
FULL50 frozen retargeting arrays
→ top episode/aggregate geometry

TRAIN40 method-specific 28-D datasets + common ACT contract
→ checkpoint candidates 20k/60k/100k
→ frozen lexicographic checkpoint selection
→ strict-reload raw held-out predictions (72 × 50 × 28)
→ named G1/Dex3 FK
→ common source wrist/interaction geometry errors
→ nine-probe → eight-episode aggregation
→ within-row paired cluster bootstrap
→ 2×3 final plot
→ byte-for-byte archival PNG copy
```

```bash
cd /home/jbnu/aloha_g1_dataset
python3 tools/generate_jkros_final_figures.py
sha256sum outputs/paper_final_figures/Fig05_Supervision_to_Policy/Fig05_Supervision_to_Policy.png \
  outputs/paper_final_5_archive/FIGURE_05/figure_05.png
```

The frozen prediction archives already exist. This archival pass did not run ACT inference, training, FK evaluation, or simulation.

## N. VALIDATION / INTEGRITY CHECKS

- Top row: 50 episode points × 2 methods × 3 metrics = 300 records.
- Bottom row: 8 episodes × 2 policies × 3 metrics = 48 records; nine probes and 450 valid frames for every episode/method/metric.
- Held-out IDs equal `[2,13,23,27,28,31,37,40]` for both methods.
- Source RGB tensors matched exactly for 72/72 probes; named 28-D feature mapping passed.
- Selected checkpoint hashes match Experiment 2 and the training audit.
- Retargeting means match Table 1; policy rounded means match Table 2; exact means/p95/max match Experiment 2.
- The existing validator confirms sample counts, identities, means, 600 dpi, correct double-column variant, and no placeholder.
- Original and archival PNG hashes match.

## O. INTERPRETATION

**Main takeaway:** the supervision representation leaves a corresponding signature in held-out ACT predictions. **Secondary takeaway:** policy learning increases absolute geometry error relative to retargeting supervision on a different evaluation set, but preserves the A/B directional pattern. **Important trade-off:** ACT-A is better in wrist error; ACT-B is better in whole-hand and bimanual error. ACT-B does not win the separate phase score.

## P. LIMITATIONS / CAVEATS

- Top (FULL50 retargeting) and bottom (HELDOUT8 policy predictions) use different evaluation sets and are not paired before/after samples.
- Policy geometry is computed from raw predicted chunks under source-conditioned offline probes; it is not closed-loop rollout success.
- Each policy is scored against source geometry but action RMSE against its own retargeted target; those are different questions.
- ACT-B's lower full-chunk RMSE and interaction errors do not imply higher phase score; observed phase scores are 56/64 for A and 52/64 for B.
- No physical doll grasp, real-G1 execution, onboard autonomy, or Experiment-3 result is claimed.
- Checkpoint selection includes phase score, RMSE, jerk, and step in a predeclared lexicographic order; it is not based on the plotted geometry means.

## Q. ORAL PRESENTATION NOTES

### 20-second Korean explanation

“위는 50개 retargeting, 아래는 8개 held-out ACT 예측입니다. 서로 다른 집합이지만 A 계열은 손목 오차가 낮고 B 계열은 전체 손·양손 오차가 낮다는 표현 특성이 학습 후에도 유지됩니다.”

### 60-second Korean explanation

“위 행은 50개 전체 시연에서 만든 supervision의 기하 오차이고, 아래는 사전에 분리한 8개 held-out 시연에서 9개 probe씩 예측한 raw ACT chunk의 기하 오차입니다. 같은 열은 같은 정의와 단위를 쓰지만 위아래는 paired sample이 아닙니다. A는 supervision에서 손목 3.742 mm, ACT-A는 24.958 mm이고, B는 77.853/95.269 mm입니다. 반대로 전체 손과 양손 관계에서는 B와 ACT-B가 모두 더 낮습니다. 따라서 정책이 단순히 모든 차이를 지우지 않고 supervision 표현의 방향성을 유지한다고 해석합니다. 다만 phase score는 A 56/64, B 52/64이므로 B가 모든 정책 지표에서 우수하다고 말하지 않습니다.”

### Likely reviewer questions

1. **Q: Are top and bottom directly comparable?** A: Definitions/units match, but sets differ (FULL50 vs HELDOUT8), so they show directional preservation rather than paired change.
2. **Q: Were checkpoints chosen to favor geometry?** A: No. A common rule frozen before training ranks phase score, then full-chunk RMSE, jerk, and earlier step; geometry means are not selection keys.
3. **Q: Does this show closed-loop task success?** A: No. It is offline held-out raw-chunk prediction and named forward-kinematics geometry.

## R. RECONSTRUCTION CHECKLIST

{reconstruction_checklist()}
"""


def build_readme(data: dict[str, Any], zip_hash_placeholder: str = "computed after packaging") -> str:
    return f"""# ALOHA→G1 Final Five Figure Provenance Archive

Archive created: **{NOW.isoformat()}**  
Repository root used: `{ROOT}`  
Git HEAD: `{data['git_head']}`  
Worktree at packaging time: **{'DIRTY' if data['dirty'] else 'CLEAN'}**

## Purpose

This self-contained archive preserves the exact five PNGs selected by the current author directive for the ALOHA→G1 Doll-Handoff paper and dense Markdown provenance sufficient to audit, explain, or redraw them. No scientific artifact was modified and no retargeting, training, inference, Isaac simulation, or GPU job was run.

The research question is whether accurate transfer of independent wrist trajectories is sufficient to preserve manipulation interaction when ALOHA demonstrations are realized on G1 with Dex3 hands, and whether representation-dependent supervision remains visible after ACT policy learning.

## Canonical methods and collective result

- **Trajectory-Centric A (Fair-A)** transfers independent 6-D wrist poses. It uses the fairness-corrected shared temporal IK and generic feasibility backend and intentionally omits interaction-frame, ownership, and bimanual-semantic targets.
- **Interaction-Centric B** represents interaction frames, physical whole-hand grasp geometry, bimanual right-minus-left relation, and ownership transition, then uses the shared backend.
- **ACT-A / ACT-B** are matched official ACT policies trained on A/B TRAIN40 supervision.

Collectively the five figures establish a trade-off, not a universal winner: A has much lower wrist error; B has much lower whole-hand and bimanual interaction errors and no final hard-fail episodes; the same orientation is visible in held-out ACT prediction.

## Authoritative final-five selection

{md_table(['Order', 'Paper figure', 'Scientific question', 'Data scope', 'Main result', 'Archive PNG'], [
    (1, 'Fig. 1 — Method Overview', 'What exactly is compared?', 'FULL50; COMMON48/TRAIN40/HELDOUT8 context', 'Representation differs; realization and evaluation are shared.', '[FIGURE_01/figure_01.png](FIGURE_01/figure_01.png)'),
    (2, 'Fig. 2 — Retargeting Trade-off', 'Does wrist transfer preserve interaction?', 'FULL50, 50 paired episodes', 'A lower wrist; B lower whole-hand/bimanual error.', '[FIGURE_02/figure_02.png](FIGURE_02/figure_02.png)'),
    (3, 'Fig. 3 — Paired Statistical Effect', 'Is the contrast consistent across matched demonstrations?', 'FULL50, 50 paired episodes', 'Paired B−A directions reproduce the trade-off.', '[FIGURE_03/figure_03.png](FIGURE_03/figure_03.png)'),
    (4, 'Fig. 4 — Feasibility', 'Are trajectories mechanically realizable?', 'FULL50, 50 per method', 'A has 2 collision HARD; B has 0 HARD.', '[FIGURE_04/figure_04.png](FIGURE_04/figure_04.png)'),
    (5, 'Fig. 5 — Supervision to Policy', 'Does the representation characteristic survive learning?', 'FULL50 + HELDOUT8', 'A/ACT-A wrist-oriented; B/ACT-B interaction-oriented.', '[FIGURE_05/figure_05.png](FIGURE_05/figure_05.png)'),
])}

Selection evidence was inspected in the requested priority order. **No current paper manuscript or submission draft was present in the repository.** The next authority, `outputs/paper_final_figures/manifest/FINAL_SIX_FIGURE_MANIFEST.md`, still calls Figs. 1–6 the final six. The current author's 2026-08-30 archival directive explicitly requests the final five and uniquely names the method pipeline, core trade-off, paired statistics, feasibility, and supervision→policy content. Those map exactly to source Figs. 1–5, so Fig. 6 representative motion is excluded from this five-figure archive. This selection discrepancy is recorded below rather than silently reconciled.

The lower-priority paper artifacts do not resolve that conflict: `outputs/paper_final_figures/captions/` contains final captions for Figs. 1–6, and `outputs/paper_final_figures/manifest/validation_report.json` explicitly validates an “exact six” directory set. The publication-ready bank contains candidates rather than a newer manuscript selection. Thus the only five-item authority is the current explicit author directive.

## Navigation

Each `FIGURE_XX` directory contains one exact PNG and one `FIGURE_XX_COMPLETE_PROVENANCE.md`. Every provenance file follows sections A–R: identity; scientific question; method; sources; all numerical values; statistics; metrics; episode/split provenance; visual encoding; exact design; captions; paper connection; reproduction; validation; interpretation; caveats; oral notes; checklist. `MANIFEST.md` records file and frozen-artifact hashes.

## Dataset and result scopes

{md_table(['Scope', 'Exact definition', 'Use in archive'], [
    ('FULL50', 'Final frozen indices 0–49; 50 A/B source-matched episodes; 34,478 frames; includes A HARD 35 and 46.', 'Figs. 1–4 and Fig. 5 top row'),
    ('COMMON48', 'FULL50 minus only A HARD 35 and 46, removed from both methods.', 'Policy split parent'),
    ('TRAIN40', '40 matched COMMON48 episodes; split seed 20260826.', 'ACT-A/ACT-B training'),
    ('HELDOUT8', 'Indices [2,13,23,27,28,31,37,40]; 8 matched episodes; 9 probes each.', 'Fig. 5 bottom row / Table 2'),
])}

## Allowed paper claim

Under matched frozen sources and a shared final G1 realization backend, Trajectory-Centric A preserves wrist trajectories more accurately, while Interaction-Centric B preserves whole-hand and bimanual manipulation-interaction geometry more accurately. B retains zero hard-fail episodes versus two collision hard failures for A. The representation-dependent geometric pattern remains visible in offline HELDOUT8 ACT predictions.

## Prohibited / unsupported claims

- B is better at every metric.
- Either method achieves physical doll-task success.
- Real G1 execution, safety, or grasp robustness was demonstrated.
- Onboard visual autonomy, XR, D455, or closed-loop Experiment-3 success was demonstrated.
- A bootstrap CI is a declared statistical-significance test.
- ACT-B wins phase behavior; the frozen scores are ACT-A 56/64 and ACT-B 52/64.
- FULL50 and HELDOUT8 are paired before/after samples.

## KNOWN PROVENANCE DISCREPANCIES

1. **Figure-count selection:** the repository's final manifest declares six figures, while the current author directive declares five and describes content matching Figs. 1–5. With no manuscript available to resolve the conflict, the current explicit selection governs this archive; Fig. 6 is not copied.
2. **Bootstrap seed metadata granularity:** final metadata for Figs. 2 and 5 records base seed 20260827. The plotting code uses deterministic offsets per metric/row. Fig. 2's effective seeds are stored in its source CSV; Fig. 5's are recoverable only from code and are documented in its provenance Markdown.
3. **Repeated FULL50 CIs:** Fig. 2 and the top row of Fig. 5 use identical episode data and means but different deterministic bootstrap streams, producing slightly different Monte Carlo percentile endpoints. This is not a scientific-value conflict.
4. **Frame-weighted versus episode-weighted estimands:** Fig. 2/Fig. 5 filled retargeting markers are frame-weighted; Fig. 3 is the unweighted mean of episode-level B−A differences. Their point arithmetic therefore differs slightly by design.
5. **Sub-micrometre derivative precision:** full-precision wrist means differ slightly among the final figure CSV, frozen comparison JSON, and older ECDF source (maximum span below `0.000012` mm). All agree at the paper's three-decimal precision. The final PNG and its documentation retain the exact final-figure CSV value; Fig. 2 provenance transcribes every variant.

No discrepancy was found in rounded scientific values between final figure source CSVs, Table 1, Table 2, the final validator, and frozen JSON/NPZ-derived manifests. There is no manuscript artifact against which wording or numbering can be checked.

## Archive integrity

- Intended content: exactly 5 `.png` and 7 `.md` files; no other extension.
- ZIP SHA256: **{zip_hash_placeholder}** (reported externally after ZIP construction because embedding a ZIP's own hash would be recursive).
- PNG bytes are direct copies; source and archival SHA256 values are identical.
- See `MANIFEST.md` for per-file hashes and frozen implementation/data hashes.
"""


def build_manifest(data: dict[str, Any], included_without_manifest: list[tuple[Path, str]]) -> str:
    dataset_audit = data["dataset_audit"]
    training = data["training"]
    b_freeze = data["b_freeze"]
    a_manifest = data["a_manifest"]
    frozen_rows = [
        ("Git HEAD", "repository", data["git_head"]),
        ("Final plotting implementation", rel(P["plot_script"]), sha256(P["plot_script"])),
        ("Final validation implementation", rel(P["plot_validator"]), sha256(P["plot_validator"])),
        ("Retargeting data loader/statistics", rel(P["bank_script"]), sha256(P["bank_script"])),
        ("Held-out ACT evaluator", rel(P["eval_script"]), sha256(P["eval_script"])),
        ("Fair-A repair manifest", rel(P["a_manifest"]), sha256(P["a_manifest"])),
        ("Fair-A trajectory set", "manifest field `trajectory_set_sha256`", a_manifest["trajectory_set_sha256"]),
        ("Fair-A original implementation", "manifest field", a_manifest["original_a_implementation_sha256"]),
        ("Frozen generic resolver implementation", "A manifest field", a_manifest["frozen_b_generic_resolver_implementation_sha256"]),
        ("Frozen generic resolver config", "A manifest field", a_manifest["frozen_b_generic_resolver_config_sha256"]),
        ("Proposed-B final source manifest", rel(P["b_manifest"]), sha256(P["b_manifest"])),
        ("Proposed-B trajectory set", "B freeze field `trajectory_set_sha256`", b_freeze["trajectory_set_sha256"]),
        ("Proposed-B canonical policy action set", "B freeze field", b_freeze["canonical_policy_action_set_sha256"]),
        ("COMMON48 entries", rel(P["common48"]), data["common48"]["entries_canonical_sha256"]),
        ("TRAIN40 entries", rel(P["train40"]), data["train40"]["entries_canonical_sha256"]),
        ("HELDOUT8 entries", rel(P["heldout8"]), data["heldout"]["entries_canonical_sha256"]),
        ("Table 1 CSV", rel(P["table1"]), sha256(P["table1"])),
        ("Table 2 CSV", rel(P["table2"]), sha256(P["table2"])),
        ("Stored paired effects", rel(P["forest"]), sha256(P["forest"])),
        ("ACT-A selected model", str(data["exp2"]["methods"]["a"]["checkpoint_selection"]["selected_checkpoint"]), data["exp2"]["methods"]["a"]["checkpoint_selection"]["selected_model_sha256"]),
        ("ACT-B selected model", str(data["exp2"]["methods"]["b"]["checkpoint_selection"]["selected_checkpoint"]), data["exp2"]["methods"]["b"]["checkpoint_selection"]["selected_model_sha256"]),
        ("ACT-A final 100k model", training["methods"]["a"]["final_checkpoint"], training["methods"]["a"]["final_checkpoint_model_sha256"]),
        ("ACT-B final 100k model", training["methods"]["b"]["final_checkpoint"], training["methods"]["b"]["final_checkpoint_model_sha256"]),
        ("ACT training contract", rel(P["training_contract"]), sha256(P["training_contract"])),
        ("ACT-A TRAIN40 normalization stats", dataset_audit["datasets"]["a_train40"], data["training_contract"]["records"]["a"]["dataset_stats_sha256"]),
        ("ACT-B TRAIN40 normalization stats", dataset_audit["datasets"]["b_train40"], data["training_contract"]["records"]["b"]["dataset_stats_sha256"]),
        ("ACT-A TRAIN40 action array", dataset_audit["datasets"]["a_train40"], dataset_audit["manifests"]["a_train40"]["action_array_sha256"]),
        ("ACT-B TRAIN40 action array", dataset_audit["datasets"]["b_train40"], dataset_audit["manifests"]["b_train40"]["action_array_sha256"]),
        ("ACT-A HELDOUT8 action array", dataset_audit["datasets"]["a_heldout8"], dataset_audit["manifests"]["a_heldout8"]["action_array_sha256"]),
        ("ACT-B HELDOUT8 action array", dataset_audit["datasets"]["b_heldout8"], dataset_audit["manifests"]["b_heldout8"]["action_array_sha256"]),
    ]
    file_rows = []
    for archive_path, original in sorted(included_without_manifest, key=lambda x: rel(x[0])):
        file_rows.append((rel(archive_path.relative_to(ARCHIVE_DIR)), archive_path.stat().st_size, sha256(archive_path), original))
    file_rows.append(("MANIFEST.md", "SELF-REFERENTIAL", "SELF-REFERENTIAL", "generated in archive"))
    return f"""# ARCHIVE MANIFEST

Archive creation timestamp: **{NOW.isoformat()}**  
Repository: `{ROOT}`  
Git HEAD: `{data['git_head']}`  
Worktree state: **{'DIRTY' if data['dirty'] else 'CLEAN'}**

## Included files

{md_table(['Archival path', 'Size [bytes]', 'SHA256', 'Original path / origin'], file_rows)}

`MANIFEST.md` cannot contain its own final byte size and SHA256 without changing those values (a cryptographic self-reference). It is therefore marked `SELF-REFERENTIAL`; the packaging validator computes its actual size/hash after extraction, and the ZIP comment stores the final manifest hash. All other included files have exact embedded sizes and hashes.

## Original PNG correspondence

{md_table(['Figure', 'Original PNG', 'Archival PNG', 'SHA256', 'Byte-identical'], [
    (f"Fig. {fig['number']}", str((P['final_root']/fig['folder']/(fig['stem']+'.png')).resolve()), f"FIGURE_{fig['number']:02d}/figure_{fig['number']:02d}.png", sha256(P['final_root']/fig['folder']/(fig['stem']+'.png')), 'YES') for fig in FIGURES
])}

## Relevant frozen implementation, dataset, split, and checkpoint hashes

{md_table(['Artifact', 'Path / field', 'SHA256 or canonical digest'], frozen_rows)}

## Selection authorities

{md_table(['Priority', 'Evidence', 'Finding'], [
    (1, 'Current manuscript / figure references', 'NOT PRESENT in repository; no authoritative manuscript file recovered'),
    (2, rel(P['final_manifest']), 'Declares six figures, Figs. 1–6'),
    (3, 'Final caption directory', 'Contains captions for Figs. 1–6; does not select five'),
    (4, rel(P['validation']), 'Validates an exact six-figure directory set; does not select five'),
    (5, 'Final publication-ready bank', 'Candidate bank; no newer manuscript-selected five found'),
    ('governing instruction', 'Current author directive dated 2026-08-30', 'Explicitly requests final five and defines content mapping to Figs. 1–5; governs this archive'),
])}

## Packaging constraints and result mutation audit

- Included extensions: `.md`, `.png` only.
- Included counts: 5 PNG; 7 Markdown.
- Source code, PDF, SVG, JSON, CSV, NPZ, video, and checkpoints are not included.
- Important source data are transcribed into per-figure Markdown tables.
- Scientific results modified: **NO**.
- Retargeting/training/inference/Isaac rerun: **NO**.
- Experiment 3 used: **NO**.
"""


def load_data() -> dict[str, Any]:
    required = list(P.values())
    for path in required:
        if isinstance(path, Path) and not path.exists():
            raise FileNotFoundError(path)
    git_head = run_text("git", "rev-parse", "HEAD")
    dirty = bool(run_text("git", "status", "--porcelain"))
    captions = {}
    for fig in FIGURES:
        captions[fig["number"]] = (P["final_root"] / "captions" / f"Fig{fig['number']:02d}_caption.txt").read_text(encoding="utf-8").strip()
    fig2 = read_csv(P["final_root"] / "Fig02_Retargeting_Tradeoff/Fig02_Retargeting_Tradeoff_source_data.csv")
    fig3 = read_csv(P["final_root"] / "Fig03_Paired_Statistics/Fig03_Paired_Statistical_Effect_source_data.csv")
    fig4 = read_csv(P["final_root"] / "Fig04_Feasibility/Fig04_Feasibility_source_data.csv")
    fig5 = read_csv(P["final_root"] / "Fig05_Supervision_to_Policy/Fig05_Supervision_to_Policy_source_data.csv")
    return {
        "git_head": git_head,
        "dirty": dirty,
        "captions": captions,
        "s1": read_csv(P["s1"]),
        "a_full": read_csv(P["a_per_episode"]),
        "fig2": fig2,
        "fig3": fig3,
        "fig4": fig4,
        "fig5": fig5,
        "table1": read_csv(P["table1"]),
        "table2": read_csv(P["table2"]),
        "ecdf_errors": read_csv(P["ecdf_errors"]),
        "ecdf_projection": read_csv(P["ecdf_projection"]),
        "a_manifest": read_json(P["a_manifest"]),
        "b_manifest": read_json(P["b_manifest"]),
        "b_freeze": read_json(P["b_freeze"]),
        "common48": read_json(P["common48"]),
        "train40": read_json(P["train40"]),
        "heldout": read_json(P["heldout8"]),
        "dataset_audit": read_json(P["dataset_audit"]),
        "training_contract": read_json(P["training_contract"]),
        "training": read_json(P["training_audit"]),
        "exp2": read_json(P["exp2"]),
        "validation": read_json(P["validation"]),
    }


def scientific_consistency_checks(data: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    s1 = data["s1"]
    if len(s1) != 50 or [int(r["episode_index"]) for r in s1] != list(range(50)):
        issues.append("Table S1 is not the exact ordered final 0..49 set")
    if sum(int(r["frame_count"]) for r in s1) != 34478:
        issues.append("FULL50 frame total differs from 34,478")
    if {x: sum(r["a_status"] == x for r in s1) for x in ("CLEAN_PASS", "USABLE_WITH_WARNING", "HARD_FAIL")} != {"CLEAN_PASS": 27, "USABLE_WITH_WARNING": 21, "HARD_FAIL": 2}:
        issues.append("A status counts disagree")
    if {x: sum(r["b_status"] == x for r in s1) for x in ("CLEAN_PASS", "USABLE_WITH_WARNING", "HARD_FAIL")} != {"CLEAN_PASS": 13, "USABLE_WITH_WARNING": 37, "HARD_FAIL": 0}:
        issues.append("B status counts disagree")
    if [int(r["episode_index"]) for r in s1 if r["a_status"] == "HARD_FAIL"] != [35, 46]:
        issues.append("A hard episode identities disagree")
    if data["heldout"]["split_contract"]["heldout_final_dataset_indices"] != [2, 13, 23, 27, 28, 31, 37, 40]:
        issues.append("HELDOUT8 identities disagree")
    if data["exp2"]["methods"]["a"]["checkpoint_selection"]["selected_step"] != 100000 or data["exp2"]["methods"]["b"]["checkpoint_selection"]["selected_step"] != 20000:
        issues.append("selected checkpoint identity disagrees")
    expected_effects = [74.11501871520937, -69.67791922449261, -50.81853092092401, -2.4498772634494976]
    if any(abs(float(r["paired_B_minus_A_mean_mm"]) - e) > 1e-12 for r, e in zip(data["fig3"], expected_effects)):
        issues.append("forest effects disagree")
    if data["validation"].get("status") != "PASS" or data["validation"].get("checks_failed") != 0:
        issues.append("existing final figure validation is not PASS")
    # Selection discrepancy is known and non-numerical but must be surfaced.
    issues.append("SELECTION: repository final manifest declares six figures; current explicit author directive selects Figs. 1–5")
    issues.append("METADATA: Figs. 2/5 metadata records a base bootstrap seed while plotting uses documented deterministic offsets")
    return issues


def ensure_output_layout() -> None:
    if ARCHIVE_DIR.exists() or ZIP_PATH.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing archive target: {ARCHIVE_DIR} or {ZIP_PATH}"
        )
    ARCHIVE_DIR.mkdir(parents=True)
    for number in range(1, 6):
        (ARCHIVE_DIR / f"FIGURE_{number:02d}").mkdir()


def create_archive() -> dict[str, Any]:
    data = load_data()
    issues = scientific_consistency_checks(data)
    ensure_output_layout()

    included: list[tuple[Path, str]] = []
    copies: dict[int, Path] = {}
    for fig in FIGURES:
        number = fig["number"]
        original = P["final_root"] / fig["folder"] / f"{fig['stem']}.png"
        copied = ARCHIVE_DIR / f"FIGURE_{number:02d}" / f"figure_{number:02d}.png"
        shutil.copy2(original, copied)
        if sha256(original) != sha256(copied):
            raise RuntimeError(f"PNG copy hash mismatch: Fig. {number}")
        with Image.open(copied) as image:
            image.verify()
        copies[number] = copied
        included.append((copied, str(original.resolve())))

    builders = [figure_1_doc, figure_2_doc, figure_3_doc, figure_4_doc, figure_5_doc]
    for fig, builder in zip(FIGURES, builders):
        number = fig["number"]
        path = ARCHIVE_DIR / f"FIGURE_{number:02d}" / f"FIGURE_{number:02d}_COMPLETE_PROVENANCE.md"
        path.write_text(builder(data, fig, copies[number]).strip() + "\n", encoding="utf-8")
        included.append((path, "generated in archive from frozen provenance artifacts"))

    readme_path = ARCHIVE_DIR / "README.md"
    readme_path.write_text(build_readme(data).strip() + "\n", encoding="utf-8")
    included.append((readme_path, "generated in archive from current author selection and frozen manifests"))

    manifest_path = ARCHIVE_DIR / "MANIFEST.md"
    manifest_path.write_text(build_manifest(data, included).strip() + "\n", encoding="utf-8")

    all_files = sorted(p for p in ARCHIVE_DIR.rglob("*") if p.is_file())
    if len(all_files) != 12:
        raise RuntimeError(f"archive directory must contain 12 files, got {len(all_files)}")
    if sum(p.suffix.lower() == ".png" for p in all_files) != 5 or sum(p.suffix.lower() == ".md" for p in all_files) != 7:
        raise RuntimeError("archive directory extension counts are not 5 PNG / 7 MD")
    if any(p.suffix.lower() not in {".png", ".md"} for p in all_files):
        raise RuntimeError("forbidden extension in archive directory")

    manifest_hash = sha256(manifest_path)
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in all_files:
            archive.write(path, path.relative_to(ARCHIVE_DIR).as_posix())
        archive.comment = f"MANIFEST.md SHA256={manifest_hash}".encode("ascii")

    # Independent extraction and content validation.
    with tempfile.TemporaryDirectory(prefix="paper_final_5_validate_") as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(ZIP_PATH, "r") as archive:
            archive.extractall(tmp_path)
            names = archive.namelist()
            if archive.comment.decode("ascii") != f"MANIFEST.md SHA256={manifest_hash}":
                raise RuntimeError("ZIP comment manifest hash mismatch")
        extracted = sorted(p for p in tmp_path.rglob("*") if p.is_file())
        pngs = [p for p in extracted if p.suffix.lower() == ".png"]
        mds = [p for p in extracted if p.suffix.lower() == ".md"]
        if len(pngs) != 5 or len(mds) != 7 or len(extracted) != 12:
            raise RuntimeError("extracted counts differ from 5 PNG / 7 MD")
        if any(p.suffix.lower() not in {".png", ".md"} for p in extracted):
            raise RuntimeError("forbidden extracted extension")
        for p in pngs:
            with Image.open(p) as image:
                image.verify()
            source = ARCHIVE_DIR / p.relative_to(tmp_path)
            if sha256(p) != sha256(source):
                raise RuntimeError(f"extracted PNG hash mismatch: {p}")
        for p in mds:
            if p.stat().st_size == 0 or not p.read_text(encoding="utf-8").strip():
                raise RuntimeError(f"empty Markdown file: {p}")
            text = p.read_text(encoding="utf-8")
            for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
                if re.match(r"^[a-z]+://", target):
                    continue
                target_path = (p.parent / target).resolve()
                if not target_path.exists():
                    raise RuntimeError(f"broken relative Markdown reference: {p} -> {target}")
        if sha256(tmp_path / "MANIFEST.md") != manifest_hash:
            raise RuntimeError("extracted MANIFEST.md hash mismatch")

    return {
        "zip_path": str(ZIP_PATH.resolve()),
        "zip_size": ZIP_PATH.stat().st_size,
        "zip_sha256": sha256(ZIP_PATH),
        "manifest_sha256": manifest_hash,
        "issues": issues,
        "png_count": 5,
        "md_count": 7,
    }


def main() -> None:
    result = create_archive()
    print("FINAL FIVE FIGURES:")
    for fig in FIGURES:
        print(f"{fig['number']}. Fig. {fig['number']} — {fig['name']}")
    print("Evidence: no current manuscript was found; the current explicit final-five author directive maps to source Figs. 1–5 and overrides the older final-six package selection for this archive.")
    print(f"ZIP path: {result['zip_path']}")
    print(f"ZIP size: {result['zip_size']} bytes")
    print(f"ZIP SHA256: {result['zip_sha256']}")
    print(f"MANIFEST.md SHA256 (also in ZIP comment): {result['manifest_sha256']}")
    print(f"Validated contents: {result['png_count']} PNG, {result['md_count']} MD, no other extensions")
    print("Known provenance issues:")
    for issue in result["issues"]:
        print(f"- {issue}")


if __name__ == "__main__":
    main()
