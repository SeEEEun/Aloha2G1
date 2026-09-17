"""Bounded method-blind local SQP: unchanged targets and physical acceptance.

Geometry-derived progress goals are search constraints, never collision waivers.
Only a subsequent complete detailed-geometry classification can accept the q.
"""
import numpy as np
from scipy.optimize import minimize, LinearConstraint


def solve(solver, targets, hands, initial, allowances, dt, arm, goals,
          max_iterations=300):
    n = len(initial)
    ids = np.arange(arm * 7, arm * 7 + 6)
    base = initial[:, ids].copy()
    active = np.ones_like(base, dtype=bool)
    active[:2] = active[-2:] = False
    active = active.ravel()
    fixed = base.ravel()
    d1 = np.kron(np.diff(np.eye(n), axis=0), np.eye(6))
    d2 = np.kron(np.diff(np.eye(n), n=2, axis=0), np.eye(6))
    cfg = solver.config
    step = min(cfg['maximum_joint_step_rad'], cfg['maximum_velocity_rad_s'] * dt)
    acc = cfg['maximum_acceleration_rad_s2'] * dt ** 2
    matrix = np.vstack((d1, d2))
    offset = matrix[:, ~active] @ fixed[~active]
    limit = np.r_[np.full(len(d1), step), np.full(len(d2), acc)]
    linear = LinearConstraint(matrix[:, active], -limit-offset, limit-offset)

    def expand(x):
        v = fixed.copy()
        v[active] = x
        return v

    cache = [None, None]

    def constraints(x):
        if cache[0] is not None and np.array_equal(x, cache[0]):
            return cache[1]
        q = initial.copy()
        q[:, ids] = expand(x).reshape(n, 6)
        values, rows = [], []
        for f in range(2, n-2):
            pos, jac = solver.pose_jacobian(q[f], hands[f])
            error = pos[arm] - targets[f, arm]
            values.append(1e6 * ((allowances[f, arm]-1e-8) ** 2 - error @ error))
            row = np.zeros(len(fixed))
            row[f*6:f*6+6] = -2e6 * error @ jac[arm*3:arm*3+3, ids]
            rows.append(row[active])
            if f in goals:
                ds, js = solver.clearance_values(goals[f])
                for dist, deriv in zip(ds, js):
                    values.append(1000 * (dist - solver.clearance))
                    row = np.zeros(len(fixed))
                    row[f*6:f*6+6] = 1000 * deriv[ids]
                    rows.append(row[active])
        result = np.array(values), np.array(rows)
        cache[:] = [x.copy(), result]
        return result

    history = []
    def callback(x):
        if len(history) % 25 == 0:
            print('LOCAL_SQP', len(history), 'min_constraint', float(constraints(x)[0].min()), flush=True)
        history.append(float(constraints(x)[0].min()))

    fit = minimize(lambda x: .5 * np.sum((x-fixed[active])**2), fixed[active],
                   jac=lambda x: x-fixed[active], method='SLSQP',
                   bounds=list(zip(np.tile(solver.lower[ids], n)[active],
                                   np.tile(solver.upper[ids], n)[active])),
                   constraints=[linear, dict(type='ineq', fun=lambda x: constraints(x)[0],
                                              jac=lambda x: constraints(x)[1])],
                   callback=callback, options=dict(maxiter=max_iterations, ftol=1e-12))
    q = initial.copy()
    q[:, ids] = expand(fit.x).reshape(n, 6)
    return q, dict(success=bool(fit.success), message=str(fit.message),
                   iterations=int(fit.nit), min_constraint=float(constraints(fit.x)[0].min()),
                   joint_deviation_norm=float(np.linalg.norm(q-initial)), history=history)
