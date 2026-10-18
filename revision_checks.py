"""Regression checks for feasibility preprocessing and the optional MILP solver."""
import json
import math
import random
from pathlib import Path

from fort_profiles import adjacency, minimum_failure_cost, power_closure, solve_cactus
from ilp_baseline import solve_ilp
from structural_checks import spider
from validate import brute_optimum


def check_certificate(n, edges, allowed, failures, certificate):
    adj = adjacency(n, edges)
    fort = set(certificate['fort'])
    assert fort
    assert all(len(adj[v] & fort) != 1 for v in range(n) if v not in fort)
    neighborhood = fort | set().union(*(adj[v] for v in fort))
    assert len(set(allowed) & neighborhood) == certificate['cost'] <= failures


def shuffled_cactus(n, rng):
    edges, size = [], 1
    while size < n:
        root = rng.randrange(size)
        if n-size >= 2 and rng.random() < 0.6:
            count = rng.randint(2, min(7, n-size))
            path = [root] + list(range(size, size+count)) + [root]
            edges.extend(zip(path, path[1:]))
            size += count
        else:
            edges.append((root, size))
            size += 1
    labels = list(range(n))
    rng.shuffle(labels)
    return [(labels[u], labels[v]) for u, v in edges]


def main(repetitions=40, seed=20261019):
    rng = random.Random(seed)
    comparisons = feasible = rejected = 0
    for trial in range(repetitions):
        n = rng.randint(4, 11)
        edges = shuffled_cactus(n, rng)
        costs = [rng.randint(0, 15) for _ in range(n)]
        allowed = {v for v in range(n) if rng.random() > 0.2 or trial % 3 == 0}
        for k in range(5):
            exact, witnesses = brute_optimum(n, edges, costs, allowed, k)
            for precheck in (False, True):
                value, sensors, stats = solve_cactus(
                    n, iter(edges), costs, iter(allowed), k,
                    dominance=(trial % 2 == 0), precheck=precheck)
                assert value == exact
                if math.isfinite(value):
                    assert sum(1 << v for v in sensors) in witnesses
                elif precheck:
                    assert stats['operations'] == 0
                    check_certificate(n, edges, allowed, k,
                                      stats['infeasibility_certificate'])
            ilp = solve_ilp(n, edges, costs, allowed, k, time_limit=10)
            assert ilp['status'] in ('optimal', 'infeasible')
            assert ilp['cost'] == exact
            if math.isfinite(exact):
                assert sum(1 << v for v in ilp['sensors']) in witnesses
                feasible += 1
            else:
                rejected += 1
                check_certificate(n, edges, allowed, k, ilp['certificate'])
            comparisons += 1
    # Empty permitted set and budgets at and beyond the exact margin.
    for n, edges, allowed in [(1, [], set()), (1, [], {0}),
                               (4, [(0,1),(1,2),(2,3)], {1,3})]:
        r = minimum_failure_cost(n, edges, allowed)
        for k in (0, r, r+1):
            value, sensors, stats = solve_cactus(n, edges, allowed=allowed, failures=k)
            assert math.isfinite(value) == (k < r)
            if k >= r:
                assert stats['operations'] == stats['max_entries'] == 0
                check_certificate(n, edges, allowed, k, stats['infeasibility_certificate'])
    n, edges, arms = spider([2,2,2])
    assert minimum_failure_cost(n, edges, range(n)) == 5
    assert solve_cactus(n, edges, failures=4)[0] == brute_optimum(
        n, edges, [1]*n, set(range(n)), 4)[0] == 7
    result = {'seed': seed, 'graph_budget_instances': comparisons,
              'feasible': feasible, 'infeasible': rejected,
              'methods': ['raw_dp', 'prechecked_dp', 'fort_cut_ilp'],
              'edge_case_checks': 10, 'mismatches': 0}
    out = Path(__file__).parent / 'results'
    out.mkdir(exist_ok=True)
    (out/'revision_checks.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    main()
