from pathlib import Path
import json,shutil,sys
import pytest
ROOT=Path('/home/jbnu/aloha_g1_dataset');sys.path.insert(0,str(ROOT/'tools'))
import approve_episode49_final_events as f
ORIG=json.load(open(ROOT/'configs/episode49_task_timeline.approved.json'))
def rewritten(tmp_path,a=600,b=620):
 p=tmp_path/'approved.json';p.write_text(json.dumps(ORIG));backup=f.rewrite(ORIG,{'left_arm_return_near_home':a,'right_accessory_place_on_table_start':b},p);return json.load(open(p)),backup
def test_existing_ten_preserved(tmp_path):
 d,_=rewritten(tmp_path);m={x['event']:x['frame'] for x in d['events']};assert all(m[k]==v for k,v in f.FIXED.items())
def test_chronological(tmp_path):
 d,_=rewritten(tmp_path);frames=[x['frame'] for x in d['events']];assert frames==sorted(frames)
def test_release_after_phone_attachment():assert f.FIXED['right_accessory_release_complete']>f.FIXED['phone_charger_attachment_complete']
def test_release_after_left_release():assert f.FIXED['right_accessory_release_complete']>f.FIXED['left_phone_release_complete']
def test_two_manual_events_required():assert tuple(x[0] for x in f.NEW)==('left_arm_return_near_home','right_accessory_place_on_table_start')
def test_no_auto_frame():
 assert not f.DRAFT.exists() or len(json.load(open(f.DRAFT)).get('events',[]))<2
def test_accessory_held_until_646():
 d=json.load(open(f.SEMANTICS));assert d['phase_rules']['ACCESSORY_HOLD_AND_TRANSPORT']['end_event']=='right_accessory_place_on_table_start'
def test_no_release_at_341():assert json.load(open(f.SEMANTICS))['kinematic_replay']['frame_341_release'] is False
def test_hold_phase():assert 'ACCESSORY_HOLD_AND_TRANSPORT' in json.load(open(f.SEMANTICS))['right_accessory_phase_order']
def test_table_phase():assert 'ACCESSORY_TABLE_PLACEMENT' in json.load(open(f.SEMANTICS))['right_accessory_phase_order']
def test_alias(tmp_path):
 d,_=rewritten(tmp_path);m={x['event']:x for x in d['events']};assert m['accessory_placed_on_table_complete']['alias']=='right_accessory_release_complete' and m['right_accessory_release_complete']['alias_of']=='accessory_placed_on_table_complete'
def test_backup(tmp_path):
 _,b=rewritten(tmp_path);assert b.exists()
def test_no_trajectory():assert json.load(open(f.SEMANTICS))['trajectory_generated'] is False
def test_no_ik():assert json.load(open(f.SEMANTICS))['ik_run'] is False
def test_no_dds():assert json.load(open(f.SEMANTICS))['dds_or_publisher_used'] is False
def test_no_publisher():assert 'channelpublisher' not in Path(f.__file__).read_text().lower()
def test_no_real_robot():assert json.load(open(f.SEMANTICS))['real_robot_used'] is False
