"""Compact event-normalized wrist shape in the episode's object/task frame.

No ALOHA joint targets and no required intermediate G1 IK goals. Proposals use
bounded differential kinematics; global RRT samples always remain available.
"""
from pathlib import Path
import numpy as np
from .io import read,record

INTERVALS={
 'APPROACH_CLEARANCE':('APPROACH_START','LEFT_CLOSE_BEGIN',0.,.65),
 'PREGRASP':('APPROACH_START','LEFT_CLOSE_BEGIN',.65,1.),
 'LEFT_ACQUISITION':('LEFT_CLOSE_BEGIN','LEFT_GRASP_SOURCE',0.,1.),
 'LIFT':('LEFT_LIFT_BEGIN','LEFT_TRANSPORT_BEGIN',0.,1.),
 'LEFT_CARRY':('LEFT_TRANSPORT_BEGIN','RIGHT_ACQUIRE_SOURCE',0.,1.),
 'RECEIVER_APPROACH':('RIGHT_APPROACH_BEGIN','RIGHT_ACQUIRE_SOURCE',0.,1.),
 'GIVER_CLEARANCE':('LEFT_RELEASE_BEGIN','RIGHT_TRANSPORT_BEGIN',0.,1.),
 'RECEIVER_DEPARTURE':('RIGHT_OWNERSHIP_SOURCE','RIGHT_TRANSPORT_BEGIN',0.,1.),
 'RIGHT_TRANSPORT':('RIGHT_TRANSPORT_BEGIN','FINAL_RELEASE_BEGIN',0.,.8),
 'PLACE':('RIGHT_TRANSPORT_BEGIN','FINAL_RELEASE_BEGIN',.8,1.),
 'POST_RELEASE_RETREAT':('FINAL_RELEASE_BEGIN','TASK_END',0.,.2)}


def extract(source_folder, phase_name, hands, count=7):
    folder=Path(source_folder);phase=read(folder/'PHASE_RECORD.json')
    name=phase_name.split('_CONNECT')[0]
    spec=INTERVALS.get(name)
    if spec is None:return None
    begin,end,lo,hi=spec;events=phase['events'];a=events[begin]['time_s'];b=events[end]['time_s']
    times=np.linspace(a+(b-a)*lo,a+(b-a)*hi,count);frame=np.asarray(phase['initial_object_pose_world'])
    with np.load(folder/'FUNCTIONAL_WRIST_PRIORS.npz') as data:
        anchors={}
        for hand in hands:
            world=data[hand+'_wrist_world'][:,:3,3]
            local=(world-frame[:3,3])@frame[:3,:3]
            anchors[hand]=np.column_stack([np.interp(times,data['source_timestamp'],local[:,j]) for j in range(3)])
    return dict(schema='compact_source_motion_v1',phase=phase_name,interval_events=[begin,end],interval_fraction=[lo,hi],
        source_times_s=times,phase_u=np.linspace(0,1,count),object_task_frame_world=frame,
        wrist_anchors_object_frame=anchors,anchor_count=count,hard_tracking=False,
        provenance=record(folder/'FUNCTIONAL_WRIST_PRIORS.npz'),source_joint_angles_used=False)


def attach(goal,source_folder):
    result=dict(goal)
    # A's primary trajectory reference is retained; the same soft guide can
    # still aid the common connector between its geometric keyframes.
    # Include both source wrists so common passive-arm preparation can reuse
    # this interval without borrowing the active hand's curve.
    prior=extract(source_folder,goal['name'],['left','right'])
    if prior:result['source_motion_prior']=prior
    return result


class MotionGuide:
    def __init__(self,g1,goal,a,b,active):
        from .planner import assign_kinematic_state
        self.g=g1;self.goal=goal;self.a=np.asarray(a);self.b=np.asarray(b);self.active=np.asarray(active);self.cache={}
        self.calibration=goal.get('kinematic_calibration',{});self.hands=list(goal['active_hands'])
        def features(q):
            key=np.asarray(q,float).tobytes()
            if key not in self.cache:
                assign_kinematic_state(g1,q,self.calibration)
                wrists=np.asarray([g1.model_to_world_position(g1.wrist_pose(s)[:3,3]) for s in self.hands])
                self.cache[key]=dict(wrists=wrists)
            return self.cache[key]
        self.features=features;start=features(a)['wrists'];end=features(b)['wrists']
        prior=goal.get('source_motion_prior');u=np.linspace(0,1,7);residual=np.zeros((7,len(self.hands),3))
        if prior:
            rotation=np.asarray(prior['object_task_frame_world'])[:3,:3]
            for j,side in enumerate(self.hands):
                local=np.asarray(prior['wrist_anchors_object_frame'][side]);t=np.asarray(prior['phase_u'])
                shape=local-((1-t[:,None])*local[0]+t[:,None]*local[-1])
                residual[:,j]=np.column_stack([np.interp(u,t,shape[:,k]) for k in range(3)])@rotation.T
        self.anchors=(1-u[:,None,None])*start+u[:,None,None]*end+residual
        self.prior=prior;self.proposals=[];self.proposal_errors=[]
        if prior:
            for fraction in (.2,.4,.6,.8):
                q=(1-fraction)*self.a+fraction*self.b;target=self.reference([fraction])[0].ravel()
                # Fixed six Jacobian steps, not a target-admission IK solve.
                for _ in range(6):
                    current=features(q)['wrists'].ravel();error=target-current;columns=[]
                    for index in self.active:
                        qp=q.copy();qp[index]+=1e-4;columns.append((features(qp)['wrists'].ravel()-current)/1e-4)
                    jac=np.asarray(columns).T;step=jac.T@np.linalg.solve(jac@jac.T+.0025*np.eye(len(error)),error)
                    q[self.active]+=np.clip(step,-.12,.12);q=np.clip(q,g1.arm_limits[:,0],g1.arm_limits[:,1])
                self.proposals.append(q);self.proposal_errors.append(float(np.linalg.norm(features(q)['wrists'].ravel()-target)))

    def reference(self,u):
        u=np.asarray(u);t=np.linspace(0,1,len(self.anchors));flat=self.anchors.reshape(len(t),-1)
        return np.column_stack([np.interp(u,t,flat[:,j]) for j in range(flat.shape[1])]).reshape(len(u),len(self.hands),3)

    def evidence(self):
        return dict(compact_prior=self.prior,adapted_reference_anchors_world=self.anchors,
            proposal_q=self.proposals,proposal_residual_m=self.proposal_errors,
            rule='Endpoint-preserving source curvature in registered object/task axes; finite differential proposals; never hard intermediate IK targets')
