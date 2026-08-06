#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
from aloha_source_validation_common import dump
ROWS=['Phone approach','Phone grasp','Phone rotation','Accessory grasp','Accessory removal','Charger placement','Full task']
def main():
 p=argparse.ArgumentParser();p.add_argument('--trial-dir',type=Path,required=True);p.add_argument('--annotation-json',type=Path);a=p.parse_args();human=json.loads(a.annotation_json.read_text()) if a.annotation_json else {};rows=[]
 for n in ROWS:rows.append({'stage':n,'hand_path':'UNREVIEWED','grasp':'N/A' if n=='Phone approach' else 'UNREVIEWED','object_motion':'N/A' if n=='Phone approach' else 'UNREVIEWED','tracking':'UNREVIEWED','verdict':'REQUIRES_HUMAN_REVIEW','note':'',**human.get(n,{})})
 report={'status':'REQUIRES_HUMAN_REVIEW','rows':rows,'automatic_object_success_inferred':False};dump(a.trial_dir/'stage_verdicts.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
