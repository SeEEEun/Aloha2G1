#!/usr/bin/env python3
"""Hardware-inert audit skeleton for the installed Trossen low-level binding.

The installed 1.9.0 driver's daemon transmits joint inputs, so this module has
no live connection implementation. Importing and all supported modes create no
hardware object and open no network connection.
"""
from __future__ import annotations
import argparse,json,signal
from pathlib import Path

ROOT=Path('/home/jbnu/aloha_g1_dataset')
OUT=ROOT/'outputs/aloha_source_validation/episode49/readonly_driver_audit'
VERDICT='UNSAFE'
ALLOWLIST=('get_robot_output','get_all_positions','get_all_velocities','get_modes','get_joint_limits','get_error_information','get_driver_version','get_controller_version')
DENYLIST=('configure','cleanup','disconnect','set_all_modes','set_arm_modes','set_gripper_mode','set_joint_modes','set_all_positions','set_arm_positions','set_gripper_position','write','send_action','reboot_controller','clear_error','home','sleep')

def audit_plan():
 return {'verdict':VERDICT,'hardware_object_created':False,'network_connection_opened':False,
  'live_backend_implemented':False,'reason':'libtrossen_arm 1.9.0 daemon calls set_joint_inputs every communication cycle; destructor invokes cleanup',
  'allowlist_if_a_separately_verified_preconfigured_read_handle_ever_exists':list(ALLOWLIST),'denylist':list(DENYLIST)}

def read_preconfigured_fake_for_test(driver):
 """Exercise getter-only logic on an injected fake; never constructs/configures it."""
 return {name:getattr(driver,name)() for name in ALLOWLIST}

def main():
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group();g.add_argument('--audit-only',action='store_true');g.add_argument('--dry-run',action='store_true');g.add_argument('--print-plan',action='store_true')
 p.add_argument('--enable-live-readonly',action='store_true');p.add_argument('--confirmed-no-command-path',action='store_true');p.add_argument('--confirmed-driver-version',action='store_true');p.add_argument('--confirmed-safe-cleanup',action='store_true');p.add_argument('--operator-present',action='store_true');a=p.parse_args()
 if a.enable_live_readonly:raise RuntimeError('LIVE_READONLY_BLOCKED_UNSAFE: installed daemon transmits joint inputs and lifecycle is not state-only')
 report=audit_plan();OUT.mkdir(parents=True,exist_ok=True);(OUT/'lowlevel_candidate_plan.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
