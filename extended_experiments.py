"""Budget scaling and solver comparisons on deterministic cactus instances.

Run with Python 3.10+ and SciPy 1.9+: python3 extended_experiments.py
Each method receives the same 10-second solve limit; process startup and
imports are excluded. Timeout rows have no claimed optimum or table size.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import platform
import random
import time
import traceback
from pathlib import Path

from fort_profiles import minimum_failure_cost, solve_cactus
from structural_checks import spider, spider_optimum
from validate import random_cactus


def cycle_chain(count=3, length=10):
    edges, n, root = [], 1, 0
    for _ in range(count):
        ring = [root] + list(range(n, n+length-1))
        edges.extend(zip(ring, ring[1:]+ring[:1]))
        root = ring[length//2]
        n += length-1
    return n, edges


def instances(seed=20261020):
    n_spider, e_spider, arms = spider([10,10,10])
    n_chain, e_chain = cycle_chain()
    families = [
        ('path', 30, [(v,v+1) for v in range(29)], None),
        ('cycle', 30, [(v,(v+1)%30) for v in range(30)], None),
        ('spider', n_spider, e_spider, arms),
        ('cycle_chain', n_chain, e_chain, None),
    ]
    result = []
    for family_index, (family, n, edges, arms) in enumerate(families):
        for cost_instance in range(2):
            rng = random.Random(seed+100*family_index+cost_instance)
            result.append({'id': f'{family}_{cost_instance}', 'family': family,
                           'n': n, 'edges': edges, 'allowed': list(range(n)),
                           'costs': [rng.randint(1,100) for _ in range(n)],
                           'cost_instance': cost_instance, 'arms': arms})
    return result


def _worker(conn, instance, k, method, time_limit):
    try:
        if method == 'ilp':
            # Load the solver before signaling the start of timed work.
            import scipy.optimize
            from ilp_baseline import solve_ilp
        conn.send(('ready', None))
        start = time.perf_counter()
        args = (instance['n'], instance['edges'], instance['costs'],
                instance['allowed'], k)
        if method == 'ilp':
            result = solve_ilp(*args, time_limit=time_limit)
            result['seconds'] = time.perf_counter()-start
        else:
            value, sensors, stats = solve_cactus(
                *args, dominance=method == 'dp_dominance',
                precheck=method != 'dp_raw')
            result = {'status': 'optimal' if math.isfinite(value) else 'infeasible',
                      'cost': value, 'sensors': sensors,
                      'seconds': time.perf_counter()-start, **stats}
        conn.send(('result', result))
    except BaseException:
        conn.send(('error', traceback.format_exc()))
    finally:
        conn.close()


def run_case(instance, k, method, time_limit):
    context = mp.get_context('spawn')
    parent, child = context.Pipe(duplex=False)
    proc = context.Process(target=_worker,
                           args=(child, instance, k, method, time_limit))
    proc.start()
    child.close()
    try:
        if not parent.poll(30):
            raise RuntimeError('Worker startup exceeded 30 seconds')
        event, payload = parent.recv()
        if event != 'ready':
            raise RuntimeError(payload)
        if not parent.poll(time_limit):
            proc.terminate()
            proc.join(5)
            return {'status': 'timeout', 'seconds': None, 'cost': None,
                    'time_limit': time_limit}
        event, payload = parent.recv()
        if event != 'result':
            raise RuntimeError(payload)
        return payload
    finally:
        parent.close()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
            proc.join()


def main(time_limit=10.0, seed=20261020):
    # Limits used by numerical libraries in spawned workers.
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    out = Path(__file__).parent/'results'
    out.mkdir(exist_ok=True)
    cases = instances(seed)
    for case in cases:
        case['rho'] = minimum_failure_cost(case['n'], case['edges'], case['allowed'])
        assert case['rho'] >= 9, (case['id'], case['rho'])
    (out/'extended_instances.json').write_text(json.dumps(cases, indent=2)+'\n')
    rows, by_instance_budget = [], {}
    with (out/'extended_runs.jsonl').open('w') as log:
        for case in cases:
            for k in range(9):
                expected = None
                if case['family'] in ('path','cycle'):
                    expected = sum(sorted(case['costs'])[:k+1])
                elif case['family'] == 'spider':
                    expected = spider_optimum(case['arms'], case['costs'],
                                              set(case['allowed']), k)
                for method in ('dp','dp_dominance','ilp'):
                    result = run_case(case, k, method, time_limit)
                    assert result['status'] in ('optimal', 'timeout'), result
                    if result['status'] == 'optimal':
                        key = (case['id'], k)
                        if expected is not None:
                            assert result['cost'] == expected, (key, method, result, expected)
                        if key in by_instance_budget:
                            assert result['cost'] == by_instance_budget[key]
                        by_instance_budget[key] = result['cost']
                        sensors = result['sensors']
                        assert set(sensors) <= set(case['allowed'])
                        assert sum(case['costs'][v] for v in sensors) == result['cost']
                        assert minimum_failure_cost(case['n'], case['edges'], sensors) > k
                    row = {'instance': case['id'], 'family': case['family'],
                           'n': case['n'], 'rho': case['rho'], 'k': k,
                           'method': method, 'time_limit': time_limit, **result}
                    rows.append(row)
                    log.write(json.dumps(row)+'\n')
                    log.flush()
                    print(json.dumps({x:row.get(x) for x in
                        ('instance','k','method','status','seconds','max_entries','cost')}),
                          flush=True)
    (out/'extended.json').write_text(json.dumps(rows, indent=2)+'\n')
    fields = ['instance','family','n','rho','k','method','status','seconds','time_limit',
              'cost','precheck_seconds','max_entries','max_entries_before_prune',
              'pairs','dominance_seconds','retained_entries_total','cuts',
              'master_solves','lower_bound','upper_bound']
    with (out/'extended.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    # Two infeasible budgets on one reproducible branching cactus, comparing
    # the identical DP with and without the oracle preprocessing.
    rng = random.Random(seed+1000)
    edges = random_cactus(60, rng)
    control = {'id':'infeasible_control', 'n':60, 'edges':edges,
               'allowed':list(range(60)), 'costs':[rng.randint(1,100) for _ in range(60)]}
    r = minimum_failure_cost(60, edges, control['allowed'])
    control['rho'] = r
    control_rows = []
    for k in (r, r+5):
        for method in ('dp_raw','dp'):
            result = run_case(control, k, method, time_limit)
            assert result['status'] in ('infeasible','timeout')
            if method == 'dp':
                assert result['status'] == 'infeasible' and result['operations'] == 0
            control_rows.append({'k':k, 'method':method, **result})
    (out/'precheck_comparison.json').write_text(json.dumps(
        {'instance':control, 'rows':control_rows}, indent=2)+'\n')
    import scipy
    from scipy.optimize._highspy import _core
    metadata = {'seed':seed, 'instances':len(cases), 'budgets':list(range(9)),
                'runs':len(rows), 'time_limit_seconds':time_limit,
                'python':platform.python_version(), 'platform':platform.platform(),
                'scipy':scipy.__version__,
                'highs':'.'.join(str(getattr(_core,'HIGHS_VERSION_'+s))
                                 for s in ('MAJOR','MINOR','PATCH')),
                'threads':1, 'timing':'single run per cost instance; imports excluded',
                'mismatches':0}
    (out/'extended_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata, indent=2), flush=True)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--time-limit', type=float, default=10.0)
    args = parser.parse_args()
    main(time_limit=args.time_limit)
