#!/usr/bin/env python3
"""Decode the actual final MP4 files and preserve samples for visual review."""
from pathlib import Path
import sys,xml.etree.ElementTree as ET
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.final_paper_io import OUT,read,file_record,atomic_json
from tools.render_final_episode_registered_physical_evidence import atomic_image

def main():
    paper=OUT/'07_paper_artifacts/final';manifest=read(paper/'REPLAY_MANIFEST.json');folder=paper/'verification';folder.mkdir(parents=True,exist_ok=True)
    panels=[];records=[]
    for name,row in sorted(manifest['products'].items()):
        assert file_record(Path(row['artifact']['path']))==row['artifact']
        cap=cv2.VideoCapture(row['artifact']['path']);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
        assert cap.isOpened() and n==manifest['common_video_frames'] and abs(fps-30)<1e-8
        images=[];frames=[0,n//2,n-1]
        for index in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,im=cap.read();assert ok and im.shape==(2160,3840,3)
            small=cv2.resize(im,(960,540),interpolation=cv2.INTER_AREA)
            cv2.rectangle(small,(0,0),(960,28),(245,245,245),-1)
            cv2.putText(small,f'{name} | decoded MP4 frame {index}/{n-1}',(8,21),cv2.FONT_HERSHEY_SIMPLEX,.6,(20,20,20),1,cv2.LINE_AA)
            images.append(small)
        cap.release();panels.append(np.hstack(images));records.append(dict(name=name,video=row['artifact'],decoded_frames=frames,frame_count=n,fps=fps,dimensions=[3840,2160],all_samples_decoded=True))
    contact=folder/'DECODED_VIDEO_FIRST_MIDDLE_LAST.png';atomic_image(contact,np.vstack(panels))
    svg=paper/'Fig_Main_SingleVariable_AB.svg';ET.parse(svg)
    pdf=paper/'Fig_Main_SingleVariable_AB.pdf';assert pdf.read_bytes().startswith(b'%PDF-')
    atomic_json(paper/'MEDIA_DECODE_VERIFICATION.json',dict(status='PASS',videos=records,contact_sheet=file_record(contact),main_vector_files=[file_record(svg),file_record(pdf)],physical_trials=manifest['physical_traces'],interpretation='Actual encoded files decoded, not just source images inspected. With zero qualified trajectories these videos intentionally contain static retargeting-failure cards, not physics replay.'))
    print('FINAL_MEDIA_DECODE_PASS',len(records),'videos',n,'frames each')

if __name__=='__main__':main()
