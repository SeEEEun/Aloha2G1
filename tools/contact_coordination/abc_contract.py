"""Versioned A/B/C identities and evidence-backed development gates."""
from pathlib import Path
from .io import read, record

METHODS = {
    'A_WRIST': dict(name='WRIST_REFERENCE', legacy_key='A', coupling=None),
    'B_INDEPENDENT': dict(name='INDEPENDENT_INTERACTION', legacy_key='B_NO_COUPLING', coupling=False),
    'C_COUPLED': dict(name='COUPLED_INTERACTION_OURS', legacy_key='B', coupling=True),
}
PRECALIBRATION = frozenset(('inspect', 'environment_golden_audit', 'core_regression',
    'video_contract_test', 'parameter_inventory', 'folds', 'status', 'report',
    'TRAIN_control', 'SOURCE_SENSITIVITY'))
CALIBRATION = frozenset(('bounded_calibration', 'TRAIN_calibration', 'final_refit', 'freeze'))


def study_root(path):
    path = Path(path).resolve()
    return next((p for p in (path, *path.parents) if (p/'ABC_STUDY.json').exists()), None)


def verified_gate(root, name):
    path = root / (name+'.json')
    if not path.exists():
        return False
    value = read(path)
    flags = value.get('checks', {})
    evidence = value.get('evidence', [])
    return (value.get(name) is True and bool(flags) and all(v is True for v in flags.values())
            and bool(evidence) and all(record(e['path']) == e for e in evidence))


def require(out, stage):
    root = study_root(out)
    if root is None:
        raise ValueError('A/B/C stage requires an explicit ABC_STUDY.json')
    if (root/'ABORTED_FOR_CONVERTER_ARCHITECTURE_REPAIR.json').exists():
        if stage not in ('inspect','status','report'):
            raise RuntimeError('ABORTED_FOR_CONVERTER_ARCHITECTURE_REPAIR: calibration run cannot resume')
    if (root/'ARCHITECTURE_REPAIR.json').exists():
        if stage in ('ARCHITECTURE_REPAIR','TRAIN_control','inspect','status','report'):
            return
        raise RuntimeError('ARCHITECTURE_REPAIR_ONLY: calibration and downstream conversion are prohibited')
    if stage in PRECALIBRATION:
        return
    if not verified_gate(root, 'CALIBRATION_READY'):
        raise RuntimeError('ABC_GATE_CLOSED: '+stage+': CALIBRATION_READY evidence is absent or changed')
    if stage in CALIBRATION:
        return
    if not verified_gate(root, 'FINAL_CONFIG_FROZEN'):
        raise RuntimeError('ABC_GATE_CLOSED: '+stage+': FINAL_CONFIG_FROZEN evidence is absent or changed')


def eligible_supervision(row):
    """Task success is a quality label, not this predeclared eligibility rule."""
    return (row.get('evidence_channel') == 'OFFICIAL_NOMINAL'
            and not row.get('synthetic_contract_test',False)
            and not row.get('contract_test_only',False)
            and row.get('physical_validity') is True
            and row.get('complete_command_sequence') is True
            and row.get('observation_action_alignment_verified') is True)
