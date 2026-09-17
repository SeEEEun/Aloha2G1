"""Target-blind certified outer radius retaining coupled hinge translations.

For the actual zero-origin G1 arm hinges and identity fixed child rotations:
v1 + Rz(a)[v2 + Ry(b)(v3 + Rx(c)(v4 + Ry(d)v5))].
v4 and v5 lie on their local x axes. Relaxing c,d makes the final v5 vector
any vector of norm L5, an OUTER relaxation. Maximizing b exactly leaves a
one-dimensional a maximization. A global Taylor upper enclosure certifies it.
No target, episode, method, success, or measured residual is an input.
"""
import heapq
import numpy as np
import mujoco
from tools.common_hinge_orbit_bounds import orbit_enclosures

def certified_radius(v1,v2,v3,v4,v5,maximum_intervals=250000,gap_m=1e-9):
    np.testing.assert_array_equal(v1[:1],[0.])
    np.testing.assert_array_equal(v2[1:2],[0.])
    np.testing.assert_array_equal(v4[1:],[0.,0.])
    np.testing.assert_array_equal(v5[1:],[0.,0.])
    Y=float(v1[1]);X=float(v2[0]);Z=float(v1[2]+v2[2]);b=v3+v4
    B=float(np.hypot(b[0],b[2]));by=float(b[1]);L=float(np.linalg.norm(v5))
    if abs(Z)<1e-6:raise ValueError('This analytic certificate needs a nonzero axial lower bound')
    C=Y*Y+by*by+X*X+Z*Z+B*B
    umax=abs(X)+abs(Y)
    r2_bound=(Y*Y+umax*abs(Y))/abs(Z)+(umax*Y)**2/abs(Z)**3
    M2=2*abs(Y)*(abs(by)+abs(X))+2*B*r2_bound
    def value(theta):
        u=X+Y*np.sin(theta);r=np.hypot(u,Z)
        G=C+2*Y*(by*np.cos(theta)+X*np.sin(theta))+2*B*r
        derivative=2*Y*(-by*np.sin(theta)+X*np.cos(theta))+2*B*u*Y*np.cos(theta)/r
        return float(G),float(derivative)
    def enclosure(left,right):
        mid=(left+right)/2;half=(right-left)/2;G,dG=value(mid)
        # Outward rounding budget covers these float64 scalar operations. It
        # is a bound-computation guard, not a Cartesian acceptance threshold.
        upper=np.nextafter(G+abs(dG)*half+.5*M2*half*half+1e-14,np.inf)
        return (-float(upper),left,right),G
    heap=[];lower=-np.inf;best_angle=None
    edges=np.linspace(-np.pi,np.pi,9)
    for left,right in zip(edges[:-1],edges[1:]):
        item,G=enclosure(left,right);heapq.heappush(heap,item)
        if G>lower:lower=G;best_angle=(left+right)/2
    iterations=0
    while iterations<maximum_intervals and np.sqrt(-heap[0][0])-np.sqrt(lower)>gap_m:
        _,left,right=heapq.heappop(heap);mid=(left+right)/2
        for a,bound in ((left,mid),(mid,right)):
            item,G=enclosure(a,bound);heapq.heappush(heap,item)
            if G>lower:lower=G;best_angle=(a+bound)/2
        iterations+=1
    upper=-heap[0][0]
    return dict(radius_upper_m=float(np.sqrt(upper)+L),unrestricted_radius_lower_m=float(np.sqrt(lower)+L),
        certified_radius_gap_m=float(np.sqrt(upper)-np.sqrt(lower)),interval_splits=iterations,
        maximizer_sample_angle_rad=best_angle,second_derivative_absolute_bound_m2=M2,
        float64_outward_guard_m2=1e-14,no_target_input=True,
        proof='Triangle inequality for the last link after an unrestricted spherical relaxation; exact unrestricted middle-y hinge maximum; globally covering Taylor upper bounds for the remaining z hinge')

def full_chain_enclosures(g):
    rows=orbit_enclosures(g);m=g.model
    for row in rows:
        side=row['side']
        ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,side+'_'+name+'_link') for name in ('shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')]
        axes=[[0,0,1],[0,1,0],[1,0,0],[0,1,0],[0,0,1]]
        for body,axis in zip(ids,axes):
            np.testing.assert_allclose(m.body_quat[body],[1,0,0,0],atol=1e-15,rtol=0)
            np.testing.assert_array_equal(m.jnt_axis[m.body_jntadr[body]],axis)
            np.testing.assert_array_equal(m.jnt_pos[m.body_jntadr[body]],[0,0,0])
        certificate=certified_radius(*(m.body_pos[b].copy() for b in ids))
        old=row['orbit']['remainder_radius_m'];new=certificate['radius_upper_m']
        # Never replace an existing outer radius by a looser one.
        row['orbit']['remainder_radius_m']=min(old,new)
        row['full_chain_certificate']=dict(previous_outer_radius_m=old,tightening_m=max(0.,old-new),**certificate)
    return rows
