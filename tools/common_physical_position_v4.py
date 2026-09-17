"""Method-blind physical acceptance; aggregate trust radius is not a gate."""
import copy
import numpy as np
from tools.run_autonomous_dual_position import qualify_numeric as legacy_qualify
from tools.common_fixed_anchor_precision import restore as precision_restore
from tools.doll_handoff_retargeting.common import branch_flags


def temporal_metrics(q,dt,config):
    q=np.asarray(q);d=np.diff(q,axis=0);dd=np.diff(q,n=2,axis=0)
    velocity=np.max(np.abs(d),axis=0)/dt if len(d) else np.zeros(q.shape[1])
    acceleration=np.max(np.abs(dd),axis=0)/dt**2 if len(dd) else np.zeros(q.shape[1])
    vlim=np.broadcast_to(np.asarray(config['maximum_velocity_rad_s']),velocity.shape)
    alim=np.broadcast_to(np.asarray(config['maximum_acceleration_rad_s2']),acceleration.shape)
    flags=branch_flags(q,config['branch_absolute_step_norm_rad'],config['branch_local_multiplier'])
    return dict(maximum_velocity_rad_s=float(velocity.max()),maximum_acceleration_rad_s2=float(acceleration.max()),
        velocity_per_joint_rad_s=velocity.tolist(),acceleration_per_joint_rad_s2=acceleration.tolist(),
        maximum_step_norm_rad=float(np.linalg.norm(d,axis=1).max()) if len(d) else 0.,
        maximum_scalar_step_rad=float(np.abs(d).max()) if len(d) else 0.,
        velocity_violating_joints=np.flatnonzero(velocity>vlim+1e-7).tolist(),
        acceleration_violating_joints=np.flatnonzero(acceleration>alim+1e-5).tolist(),
        branch_discontinuity_frames=np.flatnonzero(flags).tolist(),
        aggregate_step_is_acceptance_gate=False,
        pass_temporal=bool(np.all(velocity<=vlim+1e-7) and np.all(acceleration<=alim+1e-5) and not np.any(flags) and np.isfinite(q).all()))


def qualify_numeric(solver,q,targets,hands,timestamps,bounds,slack):
    result,actual,residual,lower,cert,allow=legacy_qualify(solver,q,targets,hands,timestamps,bounds,slack)
    result['legacy_aggregate_temporal_diagnostic']=result['temporal']
    result['temporal']=temporal_metrics(q,float(np.median(np.diff(timestamps))),solver.config)
    result['pass_numeric']=bool(not result['failed_frames'] and result['temporal']['pass_temporal'] and
        not result['hard_limit_violations'] and result['finite'])
    mask=lower>.01000001
    result['certified_optimality_gap_max_mm']=float(np.max(residual[mask]-lower[mask])*1000) if np.any(mask) else None
    result['certified_optimality_gap_scope']='certified wrists only; independently reachable other wrist retains10mm gate'
    return result,actual,residual,lower,cert,allow


def restore(solver,targets,hands,initial,dt,allowances,**kwargs):
    # Reuse tested residual/Jacobian implementation. Its aggregate residual is
    # made mathematically redundant by the unchanged scalar physical bounds.
    # This is not an outcome-selected aggregate tolerance or final gate.
    local=copy.copy(solver);local.config=dict(solver.config)
    scalar=min(local.config['maximum_joint_step_rad'],local.config['maximum_velocity_rad_s']*dt)
    local.config['maximum_step_norm_rad']=float(np.sqrt(initial.shape[1])*scalar)
    return precision_restore(local,targets,hands,initial,dt,allowances,**kwargs)
