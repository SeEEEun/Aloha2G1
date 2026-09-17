"""Causal ACT interface over the existing simulator, camera, OTG and checker.

Only current RGB and measured named state enter ACT. This adapter has no source
event schedule, reference fallback, contact-seeking controller, or arm planner.
"""
import ast
import hashlib
import json
import os
import subprocess
import time
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any
import numpy as np
from scipy.spatial.transform import Rotation
from .io import ROOT,read,record,atomic_json,atomic_npz


class PolicySafetyViolation(RuntimeError):
    """A specifically identified policy-command safety violation."""


def bridge_class():
    """Extract the existing bridge without importing its executable Isaac CLI."""
    path=ROOT/'tools/run_act_b_isaac_diagnostic.py'
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='ACTBridge')
    ns=dict(Path=Path,Any=Any,np=np,hashlib=hashlib,os=os,time=time,subprocess=subprocess,
        Client=Client,ROOT=ROOT,POLICY_PYTHON=Path('/home/jbnu/miniconda3/envs/lerobot-smolvla/bin/python'),
        POLICY_WORKER=ROOT/'tools/act_b_inference_worker.py',AUTHKEY=b'act-b-isaac-local-v1',read_json=read)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    return ns['ACTBridge']


class PolicyRuntime:
    event_field_names=('MEASURED_Q','EXECUTED_COMMAND','RAW_POLICY_COMMAND','ACT_NORMALIZED_ACTION')

    def __init__(self,config_path,names):
        self.cfg=read(config_path);self.names=list(names);self.folder=Path(self.cfg['output_dir'])
        assert self.cfg['runtime_mode']=='CURRENT_RGB_MEASURED_STATE_TO_ACT_TO_COMMON_OTG'
        assert not set(self.cfg)&{'reference_trajectory','source_events','handoff_times','demonstration_frame_index'}
        self.initial_q_rad=np.asarray(self.cfg['initial_q_rad'],float)
        assert self.cfg['joint_names']==self.names and self.initial_q_rad.shape==(28,)
        for row in self.cfg['dependencies']:
            assert record(row['path'])['sha256']==row['sha256'],row['path']
        self.study_compatible=False
        if self.cfg.get('selected_checkpoint_record'):
            selection_record=self.cfg['selected_checkpoint_record']
            assert record(selection_record['path'])==selection_record
            selection=read(selection_record['path'])
            assert Path(selection['checkpoint']).resolve()==Path(self.cfg['checkpoint']).resolve()
            for item in selection['files']+[selection['lineage']]:assert record(item['path'])==item
            assert record(Path(self.cfg['checkpoint'])/'model.safetensors')['sha256']==self.cfg['model_sha256']
            lineage=read(selection['lineage']['path'])
            assert lineage['paired_source_ids'] and selection['step']==lineage['configs'][self.cfg['condition']]['steps']
            self.study_compatible=True
        self.bridge=bridge_class()(Path(self.cfg['checkpoint']),self.cfg['model_sha256'],self.cfg['execution'],self.folder)
        import atexit
        atexit.register(self.close)
        self.bridge.connection.send({'command':'reset'})
        self.reset_response=self.bridge.connection.recv()
        from tools.common_deployment_safety_projection import NamedJointDeploymentSafetyProjector
        from tools.common_jerk_limited_otg import CommonJerkLimitedOTG
        self.projector=NamedJointDeploymentSafetyProjector.from_path(Path(self.cfg['deployment_projection']))
        self.otg=CommonJerkLimitedOTG.from_path(Path(self.cfg['otg']))
        assert self.projector.names==self.otg.names==self.names
        self.previous=self.initial_q_rad.copy();self.previous_velocity=np.zeros(28);self.previous_acceleration=np.zeros(28)
        self.rows=[];self.aborted=None;self.closed=False
        self.raw=self.initial_q_rad.copy();self.normalized=np.zeros(28);self.executed=self.initial_q_rad.copy()

    def create_camera(self):
        from .g1_rgb_observer import G1RGBObserver
        self.observer=G1RGBObserver(self.cfg['camera'],'G1 ACT current-observation input')
        self.observer.create_camera()

    def bind(self,sim,robot,doll):
        self.sim,self.robot,self.doll=sim,robot,doll
        self.observer.bind(sim)
        from tools.doll_handoff_retargeting.common import load_common_config,load_scene
        from tools.doll_handoff_retargeting.models import G1Kinematics
        from .source_phase import COMMON
        from .runtime_hulls import Checker
        conf=load_common_config(COMMON);g=G1Kinematics(conf,load_scene(conf))
        self.checker=Checker(g,Path(self.cfg['checker_dir']),self.names)

    def snapshot(self,*,measured_q_rad,object_pose_xyzw,body_names,body_positions_world_m,body_quaternions_xyzw,records):
        # Privileged object state is exclusively a safety input, never ACT input.
        return dict(q=measured_q_rad,object=object_pose_xyzw)

    def rgb(self):
        return self.observer.read()

    def step(self,frame,snapshot):
        import cv2
        from .source_phase import pose
        rgb=self.rgb();q=np.asarray(snapshot['q']);start=time.monotonic()
        raw,normalized,nchunk,pchunk,response=self.bridge.infer(rgb,q)
        self.raw,self.normalized=raw,normalized
        image_path=self.folder/'observations'/f'frame_{frame:06d}.png';image_path.parent.mkdir(parents=True,exist_ok=True)
        assert cv2.imwrite(str(image_path),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        row=dict(control_frame=frame,observation_time_s=frame/30.,observation_precedes_action=True,
            measured_state=q,raw_action=raw,normalized_action=normalized,worker=response,image=record(image_path))
        try:
            if np.any(raw[:14]<self.projector.hard_lower[:14]) or np.any(raw[:14]>self.projector.hard_upper[:14]):
                raise PolicySafetyViolation('POLICY_ARM_COMMAND_OUTSIDE_NAMED_HARD_LIMITS')
            projection=self.projector.project(raw[None],inference_index=frame)
            target=projection.deployment_safe_action[0]
            # Ruckig carries its own previous command derivatives. Actual
            # measured state is unmodified and remains the policy observation.
            generated=self.otg.generate(current_position=self.previous,current_velocity=self.previous_velocity,
                current_acceleration=self.previous_acceleration,target_position=target,steps=1)
            command=generated.position[0]
            assert np.all(command>=self.projector.hard_lower) and np.all(command<=self.projector.hard_upper)
            obj=snapshot['object'];x=pose(Rotation.from_quat(obj[3:7]).as_matrix(),obj[:3])
            allq=self.robot.data.joint_pos.torch[0].detach().cpu().numpy()
            extras={str(n):float(v) for n,v in zip(self.robot.data.joint_names,allq,strict=True) if n not in self.names}
            bad=[]
            for u in (0.,.5,1.):
                test=q*(1-u)+command*u
                hits=[h for h in self.checker.check(test,x,('left','right'),extras) if not h['allowed_contact']]
                if frame==0 and u==0. and hits:raise RuntimeError('COMMON_INITIAL_SCENE_COLLISION: '+str(hits[:2]))
                bad.extend(hits)
            if bad:raise PolicySafetyViolation('COLLISION_INVALID_SERVO_EDGE: '+str(bad[:2]))
            self.previous=command.copy();self.previous_velocity=generated.velocity[0];self.previous_acceleration=generated.acceleration[0]
            self.executed=command.copy()
            row.update(executed_command=command,hard_projected=projection.hard_limit_projected_action[0],
                deployment_projected=target,projection=projection.summary,otg=generated.audit,safety='VALID')
        except PolicySafetyViolation as error:
            self.aborted=dict(frame=frame,reason=str(error),status='POLICY_ADAPTER_SAFETY_ABORT')
            row.update(safety=self.aborted,executed_command=None)
        except Exception as error:
            # Preserve the observation/output, but do not blame an internal
            # projector, solver or checker exception on the learned policy.
            row.update(infrastructure_error=dict(type=type(error).__name__,message=str(error)),executed_command=None)
            if pchunk is not None:atomic_npz(self.folder/'chunks'/f'query_{frame:06d}.npz',normalized=nchunk,physical=pchunk)
            atomic_json(self.folder/'policy_steps'/f'{frame:06d}.json',row)
            raise
        if pchunk is not None:
            atomic_npz(self.folder/'chunks'/f'query_{frame:06d}.npz',normalized=nchunk,physical=pchunk)
        row['runtime_s']=time.monotonic()-start
        atomic_json(self.folder/'policy_steps'/f'{frame:06d}.json',row);self.rows.append(row)
        return self.executed

    def event_values(self,measured):
        return dict(MEASURED_Q=measured,EXECUTED_COMMAND=self.executed,RAW_POLICY_COMMAND=self.raw,ACT_NORMALIZED_ACTION=self.normalized)

    def summary(self):
        return dict(mode=self.cfg['runtime_mode'],checkpoint=self.cfg['checkpoint'],model_sha256=self.cfg['model_sha256'],
            calls=len(self.rows),executed_commands=sum(r.get('executed_command') is not None for r in self.rows),
            reset_response=self.reset_response,aborted=self.aborted,execution=self.cfg['execution'],
            teacher_reference_used=False,source_event_clock_used=False,high_level_planner_used=False,
            event_driven_demo_controller_used=False,task_or_object_specific_arm_override=False,
            contact_state_input_to_ACT=False,checkpoint_compatible_with_required_study=self.study_compatible,
            selected_checkpoint_record=self.cfg.get('selected_checkpoint_record'),
            interpretation='Verified matched target-domain checkpoint; actual current-observation ACT execution' if self.study_compatible else 'Legacy-checkpoint TRAIN interface diagnostic only; not primary ACT35 evaluation')

    def write_summary(self,path):
        atomic_json(path,self.summary())
        self.close()

    def close(self):
        if not getattr(self,'closed',False):
            self.bridge.close();self.closed=True


def build_runtime(command_path,commands,names):
    # The engine's constant administrative command array is never consulted.
    return PolicyRuntime(Path(command_path).parent/'ACT_RUNTIME.json',names)
