import json
from pathlib import Path

ROOT = Path('/home/jbnu/aloha_g1_dataset')


def test_episode49_manual_end_events_are_equal_and_approved():
    data = json.loads((ROOT/'configs/episode49_task_timeline.approved.json').read_text())
    events = {x['event']: x for x in data['events']}
    assert data['frame_range'] == [0, 989]
    assert data['fps'] == 30.0
    for name in ('left_arm_return_near_home', 'task_end'):
        event = events[name]
        assert event['frame'] == 702
        assert event['timestamp_s'] == 23.4 == 702/30
        assert event['source'] == 'manual_video_review'
        assert event['approval'] == 'APPROVED_BY_USER'
        assert event['scope'] == 'EPISODE_49_ONLY'


def test_chronology_allows_equal_frames():
    data = json.loads((ROOT/'configs/episode49_task_timeline.approved.json').read_text())
    frames = sorted(x['frame'] for x in data['events'])
    assert all(a <= b for a, b in zip(frames, frames[1:]))
    assert frames.count(702) == 2
