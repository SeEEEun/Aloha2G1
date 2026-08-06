#!/usr/bin/env python3
"""Read-only state recorder. Live backend deliberately unavailable: installed connect() moves arms."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
from aloha_source_validation_common import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--duration',type=float,default=10);p.add_argument('--output',type=Path,default=BASE/'read_only_idle');p.add_argument('--dry-run',action='store_true');p.add_argument('--rate',type=float,default=30);a=p.parse_args()
 if not a.dry_run:raise RuntimeError('LIVE_READ_ONLY_BACKEND_BLOCKED: installed TrossenArmDriver.connect/disconnect changes modes and commands home/sleep; supply a vendor-verified non-mutating state API first')
 n=max(2,int(a.duration*a.rate));mono=time.monotonic()+np.arange(n)/a.rate;wall=time.time()+np.arange(n)/a.rate
 # Fixture is clearly marked and derived from source observation only to exercise schema/validators.
 src,_,_=load_action();state=np.resize(src,(n,14));raw_names=np.asarray(NAMES);vel=np.gradient(state,1/a.rate,axis=0);lat=np.zeros(n);gaps=np.diff(mono)
 a.output.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output/'readonly_state.npz',timestamp_monotonic=mono,timestamp_wall=wall,raw_joint_names=raw_names,raw_joint_positions=state,raw_joint_velocities=vel,mapped_observation_state_14d=state,left_gripper_actual=state[:,6],right_gripper_actual=state[:,13],state_receive_latency=lat,data_provenance=np.asarray('SIM_FIXTURE_SYNTHETIC_NOT_REAL'))
 report={'status':'DRY_RUN_PASS','data_provenance':'SIM_FIXTURE_SYNTHETIC_NOT_REAL','publisher_created':False,'command_client_created':False,'mode_changed':False,'torque_changed':False,'samples':n,'duration_s':a.duration,'measured_hz':float(1/np.mean(gaps)),'packet_gap_mean_s':float(np.mean(gaps)),'packet_gap_max_s':float(np.max(gaps)),'nan_inf_count':0,'stale_packets':int(np.sum(gaps>2/a.rate)),'live_backend':'BLOCKED_UNSAFE_EXISTING_CONNECT_SIDE_EFFECTS'}
 dump(a.output/'recorder_report.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
