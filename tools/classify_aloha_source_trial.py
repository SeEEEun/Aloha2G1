#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
from aloha_source_validation_common import dump
def main():
 p=argparse.ArgumentParser();p.add_argument('--trial-dir',type=Path,required=True);a=p.parse_args();v=json.loads((a.trial_dir/'stage_verdicts.json').read_text());reviewed=all(x['verdict']!='REQUIRES_HUMAN_REVIEW' for x in v['rows']);r={'classification':'UNCLASSIFIED_REQUIRES_HUMAN_REVIEW','confidence':'LOW','evidence':[str(a.trial_dir/'stage_verdicts.json')],'requires_human_review':not reviewed,'allowed_classes':['SOURCE_ACTION_VALID','ARM_VALID_CONTACT_FAILED','SOURCE_ACTION_INVALID','OPEN_LOOP_DIVERGENCE','HARDWARE_TRACKING_FAILURE'],'rule':'Classification requires reviewed semantic/object outcomes plus tracking metrics; no object success is fabricated.'};dump(a.trial_dir/'final_classification.json',r);print(json.dumps(r,indent=2))
if __name__=='__main__':main()
