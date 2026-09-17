"""Model-derived position branches for fixed shoulder configuration.

No method, episode, object, event or task-outcome input. For the actual G1
elbow-Y / wrist-X / wrist-Y chain, the wrist origin satisfies
w = Ry(e) [b + Rx(r) Ry(p) v], b=v3+v4, v=(L,0,0).
The elbow equation is one cosine equation; both elbow and wrist branches are
enumerated, filtered by authoritative limits, then independently FK checked.
"""
import numpy as np
import mujoco

def _rot_y(angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def _equivalents(angle,lower,upper):
    return [angle+2*np.pi*k for k in range(-2,3) if lower-1e-12<=angle+2*np.pi*k<=upper+1e-12]

class AnalyticPositionBranches:
    def __init__(self,g):
        self.g=g;self.rows=[];m=g.model
        for arm,side in enumerate(('left','right')):
            bodies=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,side+'_'+name+'_link') for name in ('elbow','wrist_roll','wrist_pitch','wrist_yaw')]
            for body,axis in zip(bodies,([0,1,0],[1,0,0],[0,1,0],[0,0,1])):
                np.testing.assert_allclose(m.body_quat[body],[1,0,0,0],atol=1e-15,rtol=0)
                np.testing.assert_array_equal(m.jnt_axis[m.body_jntadr[body]],axis)
                np.testing.assert_array_equal(m.jnt_pos[m.body_jntadr[body]],[0,0,0])
            v3,v4,v5=(m.body_pos[k].copy() for k in bodies[1:])
            np.testing.assert_array_equal(v4[1:],[0,0]);np.testing.assert_array_equal(v5[1:],[0,0]);assert v5[0]>0
            self.rows.append(dict(side=side,arm=arm,elbow_body=bodies[0],b=v3+v4,L=float(v5[0])))

    def candidates(self,target,shoulders,reference,hands,arm,lower=None,upper=None,allow_closest=False):
        row=self.rows[arm];base=7*arm;q=np.array(reference).copy();q[base:base+3]=shoulders;q[base+3:base+6]=0
        self.g.assign(q,*hands);bid=row['elbow_body'];R=self.g.data.xmat[bid].reshape(3,3);p=self.g.data.xpos[bid]
        w=R.T@(np.asarray(target)-p);b=row['b'];L=row['L']
        A=b[0]*w[0]+b[2]*w[2];B=b[2]*w[0]-b[0]*w[2];rho=np.hypot(A,B)
        C=(w@w+b@b-L*L)/2-b[1]*w[1];phi=np.arctan2(B,A)
        lo=self.g.arm_limits[:,0] if lower is None else np.asarray(lower)
        hi=self.g.arm_limits[:,1] if upper is None else np.asarray(upper)
        elbows=[];exact=rho>1e-15 and abs(C)<=rho+1e-14
        if exact:
            angle=np.arccos(np.clip(C/rho,-1,1))
            for e in (phi-angle,phi+angle):elbows.extend(_equivalents(e,lo[base+3],hi[base+3]))
        elif allow_closest:
            for e in (phi,phi+np.pi):elbows.extend(_equivalents(e,lo[base+3],hi[base+3]))
            elbows.extend([lo[base+3],hi[base+3]])
        values=[]
        for e in elbows:
            u=_rot_y(-e)@w-b;length=np.linalg.norm(u)
            if length<1e-15:continue
            v=u/length;pitch=np.arccos(np.clip(v[0],-1,1))
            for candidate_pitch in (pitch,-pitch):
                for pangle in _equivalents(candidate_pitch,lo[base+5],hi[base+5]):
                    ss=np.sin(pangle)
                    rolls=[reference[base+4]] if abs(ss)<1e-10 else _equivalents(np.arctan2(v[1]/ss,-v[2]/ss),lo[base+4],hi[base+4])
                    for roll in rolls:
                        vq=np.array(reference).copy();vq[base:base+3]=shoulders;vq[base+3:base+6]=[e,roll,pangle]
                        if np.any(vq<lo-1e-12) or np.any(vq>hi+1e-12):continue
                        self.g.assign(vq,*hands);actual=self.g.wrist_pose(row['side'])[:3,3].copy();residual=float(np.linalg.norm(actual-target))
                        if exact and residual>1e-8:raise AssertionError('Analytic branch failed independent FK')
                        values.append(dict(q=vq,residual_m=residual,exact_equation=exact))
        return values
