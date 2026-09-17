"""Read-only pre-action observations around the unchanged reference controller."""
from pathlib import Path
import numpy as np
from .io import read,record,atomic_json,atomic_npz
from .phase_clock_runtime import build_runtime as controller_runtime
from .g1_rgb_observer import G1RGBObserver


class MeasuredDemonstrationRuntime:
    def __init__(self,delegate,config):
        self.delegate=delegate;self.cfg=config;self.folder=Path(config['output_dir'])
        self.observer=G1RGBObserver(config['camera'],'Dynamic G1 demonstration pre-action observation')
        self.observation_rows=[]

    def __getattr__(self,name):return getattr(self.delegate,name)
    def create_camera(self):self.observer.create_camera()
    def bind(self,sim,robot,doll):self.observer.bind(sim)

    def step(self,frame,snapshot):
        import cv2
        rgb=self.observer.read();state=np.asarray(snapshot.measured_q_rad,dtype=np.float64).copy()
        command=self.delegate.step(frame,snapshot)
        path=self.folder/'observations'/f'frame_{frame:06d}.png';path.parent.mkdir(parents=True,exist_ok=True)
        assert cv2.imwrite(str(path),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        row=dict(control_frame=int(frame),timestamp_s=frame/30.,observation_precedes_action=True,state=state,executed_command=np.asarray(command).copy(),image=record(path))
        atomic_json(self.folder/'observation_action'/f'{frame:06d}.json',row);self.observation_rows.append(row)
        return command

    def write_summary(self,path):
        self.delegate.write_summary(path)
        rows=self.observation_rows
        atomic_npz(self.folder/'OBSERVATION_ACTION.npz',control_frame=np.asarray([r['control_frame'] for r in rows]),timestamp_s=np.asarray([r['timestamp_s'] for r in rows]),measured_state=np.asarray([r['state'] for r in rows]),executed_command=np.asarray([r['executed_command'] for r in rows]),joint_names=np.asarray(self.cfg['joint_names']))
        atomic_json(self.folder/'OBSERVATION_ACTION_CONTRACT.json',dict(observation='Current dynamic G1 RGB and measured named28 joints',action='Executed absolute28-joint command before next8 physics substeps',rate_hz=30,pre_action=True,frames=len(rows),camera=record(self.cfg['camera']),privileged_phase_object_fields_in_ACT_inputs=False,source_images_used=False,observer=record(__file__)))


def build_runtime(command_path,commands,names):
    return MeasuredDemonstrationRuntime(controller_runtime(command_path,commands,names),read(Path(command_path).parent/'DEMO_OBSERVATION.json'))
