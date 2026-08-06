#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from aloha_source_validation_common import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--state-record',type=Path,required=True);p.add_argument('--output',type=Path,default=BASE/'preflight');a=p.parse_args();d=load_record(a.state_record);r=validate_mapping(d['raw_joint_names'].astype(str).tolist());r.update({'state_record':str(a.state_record.resolve()),'state_data_provenance':str(d.get('data_provenance','UNKNOWN')),'source_trajectory':str(TRAJECTORY),'source_action_shape':[990,14],'gripper_direction':{'left':'INCREASING_OPEN_OFFLINE_EVIDENCE_LIVE_UNVERIFIED','right':'INCREASING_OPEN_OFFLINE_EVIDENCE_LIVE_UNVERIFIED'}});dump(a.output/'joint_mapping_report.json',r)
 lines=['# ALOHA joint mapping','','| Action index | Side | Semantic joint | Runtime joint name | Verified source |','|---:|---|---|---|---|']+[f"| {x['action_index']} | {x['side']} | {x['semantic_joint']} | {x['runtime_joint_name']} | `{x['verified_source']}` |" for x in r['rows']]
 (a.output/'joint_mapping_report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(r,indent=2));return 0 if r['status']=='PASS' else 2
if __name__=='__main__':raise SystemExit(main())
