"""One read-only G1 RGB observation algorithm for demonstrations and ACT."""
from pathlib import Path
import numpy as np


class G1RGBObserver:
    def __init__(self,config_path,purpose):
        from tools.deployment_camera_config import load_camera_config
        self.config=load_camera_config(Path(config_path),purpose=purpose)

    def create_camera(self):
        from isaaclab.sensors import Camera,CameraCfg
        import isaaclab.sim as sim_utils
        c=self.config
        self.camera=Camera(CameraCfg(prim_path='/World/HybridACTCamera',update_period=0.,width=c.width,height=c.height,data_types=['rgb'],spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(c.intrinsic_matrix.reshape(-1).tolist(),width=c.width,height=c.height,clipping_range=c.clipping_range_m,lock_camera=True)))

    def bind(self,sim):
        self.sim=sim;c=self.config
        self.camera.set_world_poses(c.position_world_xyz_m.astype(np.float32)[None],c.orientation_world_xyzw_ros_camera.astype(np.float32)[None],convention='ros')

    def read(self):
        from tools.deployment_camera_config import apply_configured_distortion
        self.sim.forward();self.sim.render();self.sim.render_context.reset_transform_cadence()
        self.camera.update(1/240.,force_recompute=True)
        rgb=self.camera.data.output['rgb'].torch[0].detach().cpu().numpy()[...,:3].copy()
        assert rgb.shape==(self.config.height,self.config.width,3) and rgb.dtype==np.uint8
        return apply_configured_distortion(rgb,self.config)
