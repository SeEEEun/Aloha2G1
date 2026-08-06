from pathlib import Path
import hashlib,json,sys
import pytest
ROOT=Path('/home/jbnu/aloha_g1_dataset');sys.path.insert(0,str(ROOT/'tools'))
import approve_episode49_events_interactive as w
def test_frame_range():
 assert w.preview_frames(0)==[0,1,5,10]
 with pytest.raises(ValueError):w.preview_frames(990)
def test_timestamp():assert w.timestamp(123)==4.1
def test_preview_boundary_safe():assert w.preview_frames(989)==[979,984,988,989]
def test_no_frame_auto_selected():assert w.new_draft()['events']==[]
def test_confirmation_required(tmp_path,monkeypatch):
 monkeypatch.setattr(w,'DRAFT',tmp_path/'draft.json');monkeypatch.setattr(w,'PREVIEWS',tmp_path/'previews');inputs=iter(['12','n','q']);w.run(lambda _:next(inputs),False);assert json.load(open(w.DRAFT))['events']==[]
def test_event_ordering():
 e=[w.record(n,f) for (n,_),f in zip(w.EVENTS,[10,20,30,40,50,60,70,80,90,100])];assert w.validate_order(e)[0];e[1]['frame']=31;assert not w.validate_order(e)[0]
def test_partial_resume(tmp_path,monkeypatch):
 monkeypatch.setattr(w,'DRAFT',tmp_path/'draft.json');d=w.new_draft();d['events']=[w.record(w.EVENT_NAMES[0],12)];w.save_draft(d);assert w.load_draft()['events'][0]['frame']==12
def test_draft_saved_on_exit(tmp_path,monkeypatch):
 monkeypatch.setattr(w,'DRAFT',tmp_path/'draft.json');w.run(lambda _:'q',False);assert w.DRAFT.exists() and json.load(open(w.DRAFT))['status']=='DRAFT_WAITING_FOR_USER'
def test_approved_requires_all(tmp_path):
 with pytest.raises(ValueError):w.save_approved(w.new_draft(),tmp_path/'approved.json')
def test_source_manual():assert w.record(w.EVENT_NAMES[0],1)['source']=='manual_video_review'
def test_raw_unchanged():
 p=w.IMAGES/'frame_000000.png';before=hashlib.sha256(p.read_bytes()).hexdigest();w.make_preview(w.EVENT_NAMES[0],0);assert hashlib.sha256(p.read_bytes()).hexdigest()==before
def test_no_trajectory_generation():assert 'np.savez' not in Path(w.__file__).read_text().lower()
def test_no_ik():assert 'solve_ik' not in Path(w.__file__).read_text().lower() and 'temporal_solve' not in Path(w.__file__).read_text().lower()
def test_no_dds():assert 'import dds' not in Path(w.__file__).read_text().lower()
def test_no_publisher():assert 'channelpublisher' not in Path(w.__file__).read_text().lower()
def test_no_real_robot():assert 'unitree_sdk' not in Path(w.__file__).read_text().lower()
