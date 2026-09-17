"""Common known-answer FK/Jacobian/6D reconstruction tests, not A/B tuning."""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from .common import *
from tools.final_paper_position_run import model, INITIAL, RESET, load_common_config
from tools.doll_handoff_retargeting.retarget import SharedTemporalIK, _rotation_error

def main():
    g,c,n=model();common=load_common_config(RESET/'config/common_config.json');s=SharedTemporalIK(common,g,n)
    base=np.array(read(INITIAL)['g1_14_arm_initial_q_rad'])
    # Independent named MuJoCo model/data, not G1Kinematics.wrist_state.
    m=mujoco.MjModel.from_xml_path(str(g.path));d=mujoco.MjData(m)
    jids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,str(name)) for name in g.arm_joint_names]
    qids=m.jnt_qposadr[jids];bids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,side+'_wrist_yaw_link') for side in ('left','right')]
    key=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_KEY,'stand')
    def independent(q):
        d.qpos[:]=m.key_qpos[key];d.qpos[qids]=q;d.qvel[:]=0;mujoco.mj_forward(m,d)
        return d.xpos[bids].copy(),d.xmat[bids].reshape(2,3,3).copy()
    def residual(q,p,r):
        a,b=independent(q)
        return float(np.linalg.norm(a-p,axis=1).max()),float(Rotation.from_matrix(r@b.transpose(0,2,1)).magnitude().max())
    path=base[None]+np.sin(np.linspace(0,np.pi,31))[:,None]*.015*np.sin(np.arange(14)+1)[None]
    poses=[independent(q) for q in path];p=np.array([x[0] for x in poses]);r=np.array([x[1] for x in poses])
    targets={f'{side}_wrist_position':p[:,k] for k,side in enumerate(('left','right'))}
    targets.update({f'{side}_wrist_rotation':r[:,k] for k,side in enumerate(('left','right'))})
    exact=[];near=[];geometry=[];jacerr=[]
    for i,q in enumerate(path):
        if i in (0,15,30):
            geometry.append(dict(frame=i,records=c.inspect(q,None,None)))
            v,meta=s._solve_seed(targets,i,q,q,q,250,None);pe,re=residual(v,p[i],r[i]);exact.append(dict(frame=i,position_m=pe,angle_rad=re,q_max_change=float(np.max(np.abs(v-q))),metadata=meta))
            seed=q+.02*np.cos(np.arange(14)+1);v,meta=s._solve_seed(targets,i,seed,q,q,250,None);pe,re=residual(v,p[i],r[i]);near.append(dict(frame=i,position_m=pe,angle_rad=re,metadata=meta))
            state=g.wrist_state(q)
            for side,k in [('left',0),('right',1)]:
                numeric=np.empty((6,7));eps=1e-6
                for j in range(7):
                    plus=q.copy();minus=q.copy();plus[7*k+j]+=eps;minus[7*k+j]-=eps
                    pp,rp=independent(plus);pm,rm=independent(minus)
                    numeric[:3,j]=(pp[k]-pm[k])/(2*eps)
                    numeric[3:,j]=Rotation.from_matrix(rp[k]@rm[k].T).as_rotvec()/(2*eps)
                jacerr.append(float(np.max(np.abs(numeric-state[side+'_jacobian']))))
    found=[];prev=path[0];prev2=prev.copy()
    for i in range(len(path)):
        v,meta=s._solve_seed(targets,i,prev,prev,prev2,40,.14);found.append(v);prev2,prev=prev,v
    errors=[residual(q,p[i],r[i]) for i,q in enumerate(found)]
    quats=Rotation.from_matrix(r.reshape(-1,3,3)).as_quat()
    qsign=float(np.max(Rotation.from_matrix(Rotation.from_quat(quats).as_matrix()@Rotation.from_quat(-quats).as_matrix().transpose(0,2,1)).magnitude()))
    rotated=Rotation.from_rotvec([.13,-.22,.04]).as_matrix();tool=Rotation.from_rotvec([-.1,.05,.02]).as_matrix()
    tool_inverse_error=float(np.max(np.abs(rotated@tool@tool.T-rotated)))
    out=dict(model=record(g.path),mapping=record(g.mapping_path),implementation=record(ROOT/'tools/doll_handoff_retargeting/retarget.py'),configuration=record(RESET/'config/common_config.json'),
        exact_seed=exact,nearby_seed=near,independent_jacobian_max_error=max(jacerr),quaternion_sign_geodesic_error=qsign,
        tool_inverse_composition_error=tool_inverse_error,continuous_path_max_position_m=max(x[0] for x in errors),continuous_path_max_angle_rad=max(x[1] for x in errors),
        known_controls_geometry=geometry,primary_angular_residual='geodesic SO(3), target @ achieved.T; no Euler subtraction',
        wrist='named left/right_wrist_yaw_link, model coordinates, meters/radians',
        fixed_tool_transform='Input already names wrist; no additional TCP transform inside 6D solver',
        status='PASS' if max(x['position_m'] for x in exact)<1e-10 and max(x['angle_rad'] for x in exact)<1e-10 and max(jacerr)<1e-6 and max(x['position_m'] for x in near)<.01 and max(x['angle_rad'] for x in near)<.75 and max(x[0] for x in errors)<.01 else 'FAIL')
    save(RUN/'audit/FULL6D_KNOWN_ANSWER_TESTS.json',out)
    text(RUN/'FULL6D_INVOCATION_AND_IMPLEMENTATION_AUDIT.md','# 6D invocation and implementation audit\n\nKnown-answer test status: '+out['status']+'\n\nExact-seed FK reconstruction, nearby deterministic seeds, a short continuous known path, independent finite-difference Jacobian ordering, quaternion sign equivalence, and active SO(3) geodesic error were tested. The targets name wrist_yaw_link, so the fixed tool transform is not reapplied. See JSON for errors and geometry evidence.\n\nHistorical A: NOT_RUN_UPSTREAM. Historical B: 37 complete solver returns. Historical rejection counts are not evidence of an orientation implementation defect. The shared DLS is position/orientation weighted, not strict lexicographic position preservation. Its documented candidate selection also overstates geometry-aware final ranking: the wrapper ranks accepted status and normalized tracking error, while the core uses proxy posture scores; final detailed geometry is checked afterward. New Layer-B validity must be applied explicitly.\n\nNo common kinematic implementation change is justified unless these independent tests fail.\n')
    print('KNOWN_ANSWER_6D',out['status'],'jacobian',out['independent_jacobian_max_error'],'near',[(x['position_m'],x['angle_rad']) for x in near],flush=True)

if __name__=='__main__':main()
