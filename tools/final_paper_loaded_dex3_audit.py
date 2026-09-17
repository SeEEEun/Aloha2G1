#!/usr/bin/env python3
"""Use the unchanged 14-joint audit at a declared nonzero G1 arm posture.

This branch audits gravity-loaded articulation only; real doll-contact and
transition tests are separately run through the contact-constrained engine.
"""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(add_help=False)
parser.add_argument('--arm-state-json',type=Path,required=True)
known,remaining=parser.parse_known_args()
arm=json.loads(known.arm_state_json.read_text())['g1_14_arm_initial_q_rad']
assert len(arm)==14
source=ROOT/'tools/audit_final_dex3_articulation_isaac.py'
text=source.read_text()
old='    midpoint[:14] = 0.0\n'
assert text.count(old)==1
text=text.replace(old,f'    midpoint[:14] = np.asarray({arm!r}, dtype=np.float64)\n')
old='"scope": "same scene/articulation/actuator/timebase as final EVAL35; task object collision-isolated",'
assert text.count(old)==1
text=text.replace(old,'"scope": "Common gravity-loaded G1 arm posture, task object collision-isolated; not doll-contact qualification",\n        "loaded_arm_state": '+repr(str(known.arm_state_json.resolve()))+',\n        "loaded_arm_q": '+repr(arm)+',')
sys.argv=[str(source),*remaining]
exec(compile(text,str(source),'exec'),{'__name__':'__main__','__file__':str(source)})
