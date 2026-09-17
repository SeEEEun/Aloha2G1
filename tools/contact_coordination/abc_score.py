"""Independent nominal scoring with an irreversible official admission latch."""
from .io import read,record,atomic_json


def score(folder, context):
    config=read(folder/'input/FULL_ATTEMPT_CONFIG.json')
    if config['evidence_channel']!='OFFICIAL_NOMINAL' or config.get('contract_test_only'):
        raise ValueError('Diagnostic/contract-control physics is ineligible for official scoring')
    from .score_hybrid import score as measured_score
    raw=measured_score(folder,context)
    recording=read(folder/'FULL_ATTEMPT_RECORDING.json')
    result=dict(raw,evidence_channel='OFFICIAL_NOMINAL',method_key=config['method_key'],
                source_id=config['source_id'],recording=record(folder/'FULL_ATTEMPT_RECORDING.json'))
    failure=recording['first_failure_latched'] or recording['official_admission_stop']
    if failure:
        result['stages']=dict(raw['stages'],FULL_TASK=False)
        result['terminal']='VALID_PLAN_TASK_FAILURE' if raw['physical_validity'] else 'EXECUTION_ABORT_PHYSICAL_VALIDITY'
        result['official_failure_latch']=failure
        result['late_bin_entry_does_not_clear_latch']=True
    if (folder/'NUMERICAL_ABORT.json').exists():
        result.update(terminal='EXECUTION_ABORT_PHYSICAL_VALIDITY',physical_validity=False,
                      validity_abort=read(folder/'NUMERICAL_ABORT.json'))
        result['stages']=dict(result['stages'],FULL_TASK=False)
    protection=folder/'PREGRASP_OBJECT_PROTECTION.json'
    if protection.exists() and read(protection)['status']=='PREGRASP_OBJECT_CONTACT':
        result.update(physical_validity=False,terminal='PREGRASP_OBJECT_CONTACT',pregrasp_object_protection=record(protection))
        result['stages']=dict(result['stages'],FULL_TASK=False)
    result['source_conditioned_full_task']='DEMONSTRATED' if result['stages']['FULL_TASK'] else 'NOT_DEMONSTRATED'
    result['scorer_adapter']=record(__file__)
    atomic_json(folder/'ABC_NOMINAL_SCORE.json',result);return result
