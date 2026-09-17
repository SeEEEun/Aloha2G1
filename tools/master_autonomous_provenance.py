"""Preserve frozen execution bytes while permitting requested report publication.

The predecessor included a report destination in its execution dependencies.
Only that reporting path can be superseded, with its exact original preserved.
No scientific executable, target, model, timing or limit receives an exception.
"""
from pathlib import Path
from tools.cartesian_reachability_forensic import DEST,BASELINE_CONTRACT
from tools.final_single_variable_prepare import OUT,read,file_record


def verified_oracle_contract():
    baseline=read(BASELINE_CONTRACT);oracle=read(DEST/'ORACLE_CONTRACT.json');substitutions=[]
    records=baseline['protected_artifacts']+baseline['new_implementations']+oracle['protected']+oracle['implementations']
    for r in records:
        p=Path(r['path']);actual=file_record(p)
        if actual['sha256']==r['sha256']:continue
        assert p==OUT/'FINAL_SINGLE_VARIABLE_AB_REBUILD_AND_EVALUATION_REPORT.md',str(p)
        archive=OUT/'master_continuation/PREVIOUS_FINAL_REPORT.md';original=file_record(archive)
        assert original['sha256']==r['sha256'],'Frozen report original is not preserved'
        substitutions.append(dict(original_record=r,verified_original_archive=original,current_report=actual,
            execution_dependency_changed=False,reason='User-requested publication at the same reporting destination'))
    return oracle,substitutions
