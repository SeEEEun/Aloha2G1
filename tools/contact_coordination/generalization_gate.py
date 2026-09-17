"""Fail closed before downstream work until fixed-TRAIN5 evidence is verified."""
from pathlib import Path
from .io import read, record

DOWNSTREAM = frozenset(('generation_freeze', 'matched_dataset_realization',
    'effective_data_audit', 'ACT_train_or_verified_reuse', 'ACT_interface_test',
    'final_freeze', 'ACT_A35_B35', 'reference_coupling10', 'analysis', 'figures',
    'replays', 'report', 'final_verification', 'TRAIN40_conversion',
    'REFERENCE_coupling10', 'target_dataset'))


def require(out, stage):
    # The current A/B/C contract has distinct calibration/final-generation
    # gates. Never inherit a historical TRAIN5 ready flag into a new study.
    from .abc_contract import study_root, require as abc_require
    if study_root(out) is not None:
        return abc_require(out, stage)
    if stage not in DOWNSTREAM:
        return
    out = Path(out).resolve()
    root = next((p for p in (out, *out.parents)
                 if (p / 'SPLIT_CONTRACT.json').exists()), out)
    path = root / 'GENERALIZATION_REPAIR_READY.json'
    reason = 'Missing current Golden/fixed-TRAIN5 generalization evidence'
    if path.exists():
        value = read(path)
        selection = read(root / 'bootstrap/SELECTION.json')
        pilot = set(selection['additional_train_source_ids'])
        planned = set(value.get('non_golden_complete_valid_plan_source_ids', []))
        executed = set(value.get('non_golden_physical_beyond_lift_source_ids', []))
        flags = ('golden_common_batch_pass', 'source_specific_targets_verified',
                 'no_known_common_registration_cache_contact_bug')
        valid = (value.get('GENERALIZATION_REPAIR_READY') == 'YES'
                 and all(value.get(k) is True for k in flags)
                 and len(planned) >= 2 and planned <= pilot
                 and bool(executed) and executed <= planned)
        evidence = value.get('verified_evidence', [])
        valid = valid and bool(evidence) and all(record(e['path']) == e for e in evidence)
        if valid:
            return value
        reason = 'Readiness evidence missing, insufficient, or changed'
    raise RuntimeError('GENERALIZATION_GATE_CLOSED: ' + stage + ': ' + reason)
