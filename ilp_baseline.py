"""Fort-neighborhood constraint generation with SciPy's HiGHS MILP solver.

The core fort-profile solver has no third-party dependencies. This optional
comparison requires NumPy and SciPy >= 1.9. Small nonnegative integer costs
are used; MILP optimality is assessed with zero requested relative gap, and
every returned placement and violated cut are checked in integer arithmetic.
"""
from __future__ import annotations

from math import inf
from time import perf_counter
import warnings

from fort_profiles import adjacency, minimum_failure_cost


def solve_ilp(n, edges, costs=None, allowed=None, failures=1, time_limit=10.0):
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp

    edges = list(edges)
    costs = [1] * n if costs is None else list(costs)
    allowed = sorted(range(n) if allowed is None else set(allowed))
    if n < 1 or len(costs) != n:
        raise ValueError('One cost per vertex of a nonempty graph is required')
    if (not isinstance(failures, int)) or failures < 0:
        raise ValueError('failures must be a nonnegative integer')
    if any(not isinstance(c, int) or c < 0 for c in costs) or sum(costs) >= 2**50:
        raise ValueError('Use nonnegative integer costs with sum below 2**50')
    if time_limit <= 0:
        raise ValueError('time_limit must be positive')
    start = perf_counter()
    attack = minimum_failure_cost(n, edges, allowed, certificate=True)
    precheck_seconds = perf_counter() - start
    stats = {'precheck_seconds': precheck_seconds,
             'feasibility_margin': attack['cost'], 'cuts': 0, 'master_solves': 0,
             'lower_bound': 0, 'upper_bound': None, 'seconds': 0.0}
    if failures >= attack['cost']:
        stats.update(status='infeasible', cost=inf, sensors=[],
                     certificate=attack, seconds=perf_counter()-start)
        return stats

    adj = adjacency(n, edges)
    stats['upper_bound'] = sum(costs[v] for v in allowed)
    best_sensors = allowed[:]
    # F=V is a fort and gives a useful initial valid covering constraint.
    rows = [np.ones(len(allowed))]
    seen_cuts = {tuple(allowed)}
    stats['cuts'] = 1
    target = failures + 1
    deadline = start + time_limit
    while perf_counter() < deadline:
        remaining = max(1e-6, deadline - perf_counter())
        # SciPy forwards the threads option to HiGHS. Keep the comparison
        # single-threaded; suppress only its documented forwarding warning.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Unrecognized options detected:.*threads.*')
            result = milp(
                c=np.array([costs[v] for v in allowed], dtype=float),
                integrality=np.ones(len(allowed)),
                bounds=Bounds(np.zeros(len(allowed)), np.ones(len(allowed))),
                constraints=LinearConstraint(np.array(rows), target, np.inf),
                options={'time_limit': remaining, 'mip_rel_gap': 0.0, 'threads': 1},
            )
        stats['master_solves'] += 1
        dual = getattr(result, 'mip_dual_bound', None)
        if dual is not None and np.isfinite(dual):
            stats['lower_bound'] = max(stats['lower_bound'], float(dual))
        if result.status not in (0, 1):
            raise RuntimeError(f'Master failed after a feasible oracle check: {result.message}')
        if result.x is None:
            break
        if np.max(np.abs(result.x - np.rint(result.x))) > 1e-5:
            raise RuntimeError('MILP returned a nonintegral candidate')
        sensors = [v for v, x in zip(allowed, result.x) if x > 0.5]
        selected = set(sensors)
        candidate_cost = sum(costs[v] for v in sensors)
        if any(sum(v in selected for v in cut) < target for cut in seen_cuts):
            raise RuntimeError('Rounded candidate violates an existing integer cut')
        witness = minimum_failure_cost(n, edges, sensors, certificate=True)
        if witness['cost'] >= target:
            if candidate_cost < stats['upper_bound']:
                stats['upper_bound'], best_sensors = candidate_cost, sensors
            if result.status == 0:
                stats.update(status='optimal', cost=candidate_cost, sensors=sensors,
                             lower_bound=candidate_cost, upper_bound=candidate_cost,
                             seconds=perf_counter()-start)
                return stats
            break
        if result.status != 0:
            break
        neighborhood = set(witness['fort'])
        for v in witness['fort']:
            neighborhood.update(adj[v])
        cut = tuple(v for v in allowed if v in neighborhood)
        if cut in seen_cuts or sum(v in selected for v in cut) >= target:
            raise RuntimeError('Separation did not produce a new violated cut')
        seen_cuts.add(cut)
        rows.append(np.array([float(v in neighborhood) for v in allowed]))
        stats['cuts'] += 1
    stats.update(status='timeout', cost=None, sensors=best_sensors,
                 seconds=perf_counter()-start)
    return stats
