#!/usr/bin/env python3
"""Hash and interface audit of the frozen common experiment; no solver edits."""
from pathlib import Path
import ast,inspect,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import final_paper_position_run as position
from tools import final_paper_full6d_run as full6d
from tools.final_paper_reference_physics import *
from tools.final_paper_source_clock_dex3 import SourceClockDex3

def main():
    position.verify();full6d.verify6();checks=[]
    for label,function in [('position realization',position.realize),('full6D realization',full6d.solve6),('Dex3 controller',SourceClockDex3)]:
        source=inspect.getsource(function);tree=ast.parse(source)
        guards=[]
        for node in ast.walk(tree):
            if isinstance(node,(ast.If,ast.IfExp)):
                literals=[n.value for n in ast.walk(node.test) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
                guards.extend(x for x in literals if x in ('A','B','WRIST','INTERACTION','ACT-A40','ACT-B40'))
        assert not guards
        checks.append(dict(component=label,signature=str(inspect.signature(function)),method_label_conditions=guards))
    pairs={};n=0
    for row in cases():
        key=(row['group'],row['index']);source=row['source']
        assert file_record(Path(source['path']))==source
        if key in pairs:assert pairs[key]==source
        else:pairs[key]=source
        n+=1
    clock=read(DEST/'dex3/SOURCE_CLOCK_INPUT_AUDIT.json');assert clock['status']=='PASS' and clock['matched_pairs']==75
    manifests=[FREEZE,OUT/'03_common_execution_freeze/COMMON_FULL6D_FREEZE_MANIFEST.json',OUT/'03_common_execution_freeze/COMMON_DEX3_FREEZE_MANIFEST.json',PF]
    for p in manifests:
        manifest=read(p)
        for rec in manifest.get('files',manifest.get('records',[])):assert file_record(Path(rec['path']))==rec
    result=dict(status='PASS',cases=n,matched_source_pairs=len(pairs),source_hash_equality=True,
        common_algorithm_interface_checks=checks,common_clock_input_audit=file_record(DEST/'dex3/SOURCE_CLOCK_INPUT_AUDIT.json'),
        representation_switch='Only selects archived raw spatial SE(3) arrays. Algorithm and parameters unchanged across cohort.',
        natural_start_and_preparation=position.verify()['acceptance'],common_compute_budget=position.verify()['budget'],
        freeze_manifests=[file_record(p) for p in manifests],unintended_downstream_method_specific_branches_detected=0,
        legacy_metadata_note='One constant ACT-A40 metadata token is supplied for every SourceClockDex3 instance; inherited execution only stores/prints it. The controller consumes no representation identity.',
        prior_results='Prior selected diagnostic portfolios and physical runs are not reused as final outcomes.',
        unresolved_limitations='Interface/static audit is not a formal proof of all software behavior. Each actual final outcome remains bound to the frozen hashes; failed numeric searches do not prove mathematical infeasibility.')
    atomic_json(DEST/'FINAL_COMMON_PIPELINE_PARITY_AUDIT.json',result)
    atomic_text(DEST/'FINAL_COMMON_PIPELINE_PARITY_AUDIT.md','# Final common pipeline parity audit\n\n'+json.dumps(result,indent=2)+'\n')
    print('FINAL_COMMON_PIPELINE_PARITY_PASS',n,len(pairs))

if __name__=='__main__':main()
