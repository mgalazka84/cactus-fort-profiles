"""Independent exhaustive checks and reproducible cactus experiments."""
import argparse
import csv
import itertools
import json
import math
import random
import time
from pathlib import Path

from fort_profiles import adjacency, cactus_blocks, solve_cactus


def brute_optimum(n, edges, costs, allowed, failures):
    """Direct observation for every placement; recursive deletion of sensors.

    This verifier does not use forts, boundary states, or the DP recurrences.
    """
    neighbors = [0] * n
    for u, v in edges:
        neighbors[u] |= 1 << v
        neighbors[v] |= 1 << u
    full = (1 << n) - 1
    good = [False] * (1 << n)
    for s in range(1 << n):
        observed = s
        for v in range(n):
            if s >> v & 1:
                observed |= neighbors[v]
        while True:
            before = observed
            for v in range(n):
                if observed >> v & 1:
                    missing = neighbors[v] & (full ^ observed)
                    if missing and not missing & (missing - 1):
                        observed |= missing
            if observed == before:
                break
        good[s] = observed == full
    for _ in range(failures):
        previous = good
        good = previous.copy()
        for s in range(1 << n):
            good[s] = previous[s] and all(
                previous[s ^ (1 << v)] for v in range(n) if s >> v & 1
            )
    permit = sum(1 << v for v in allowed)
    best = math.inf
    witnesses = []
    for s in range(1 << n):
        if s & ~permit or not good[s]:
            continue
        cost = sum(costs[v] for v in range(n) if s >> v & 1)
        if cost < best:
            best, witnesses = cost, [s]
        elif cost == best:
            witnesses.append(s)
    return best, set(witnesses)


def connected(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    seen, todo = {0}, [0]
    while todo:
        v = todo.pop()
        for u in adj[v] - seen:
            seen.add(u)
            todo.append(u)
    return len(seen) == n


def all_connected_cacti(n):
    possible = list(itertools.combinations(range(n), 2))
    # A connected cactus has at most floor(3(n-1)/2) edges.
    for m in range(max(0, n - 1), 3 * (n - 1) // 2 + 1):
        for edges in itertools.combinations(possible, m):
            if not connected(n, edges):
                continue
            try:
                cactus_blocks(adjacency(n, edges))
            except ValueError:
                continue
            yield list(edges)


def random_cactus(n, rng, cycle_probability=0.5):
    edges, size = [], 1
    while size < n:
        root = rng.randrange(size)
        remaining = n - size
        if remaining >= 2 and rng.random() < cycle_probability:
            count = rng.randint(2, min(6, remaining))
            path = [root] + list(range(size, size + count)) + [root]
            edges.extend(zip(path, path[1:]))
            size += count
        else:
            edges.append((root, size))
            size += 1
    return edges


def check(n, edges, costs, allowed, failures, dominance=False):
    exact, witnesses = brute_optimum(n, edges, costs, allowed, failures)
    cost, sensors, stats = solve_cactus(
        n, edges, costs, allowed, failures, dominance
    )
    sensor_mask = sum(1 << v for v in sensors)
    assert cost == exact, (n, edges, costs, allowed, failures, exact, cost, sensors)
    assert cost == math.inf or sensor_mask in witnesses
    stats['feasible'] = cost != math.inf
    return stats


def exhaustive(max_n=6):
    counts, comparisons, start = {}, 0, time.perf_counter()
    for n in range(1, max_n + 1):
        count = 0
        for index, edges in enumerate(all_connected_cacti(n)):
            count += 1
            costs = [1 + ((3 * v + 5 * index) % 9) for v in range(n)]
            for k in range(min(3, n + 1)):
                allowed = set(range(n))
                check(n, edges, costs, allowed, k)
                comparisons += 1
            # Extra location-restricted instance on each graph.
            allowed = {v for v in range(n) if (v + index) % 4}
            check(n, edges, costs, allowed, 1)
            comparisons += 1
        counts[str(n)] = count
        print(json.dumps({"n": n, "connected_labeled_cacti": count,
                          "comparisons_so_far": comparisons}), flush=True)
    return {"counts": counts, "comparisons": comparisons,
            "mismatches": 0, "seconds": time.perf_counter() - start}


def random_checks(repetitions=200, seed=20261018):
    rng, comparisons, start = random.Random(seed), 0, time.perf_counter()
    by_budget = {str(k): {'feasible': 0, 'infeasible': 0} for k in range(4)}
    for i in range(repetitions):
        n = rng.randint(3, 10)
        edges = random_cactus(n, rng)
        costs = [rng.randint(0, 20) for _ in range(n)]
        allowed = {v for v in range(n) if rng.random() > 0.2}
        for k in range(4):
            stats = check(n, edges, costs, allowed, k, dominance=(i % 2 == 0))
            by_budget[str(k)]['feasible' if stats['feasible'] else 'infeasible'] += 1
            comparisons += 1
    # Every possible permitted set on representative branching/cyclic graphs.
    graphs = [
        (5, [(0, 1), (0, 2), (0, 3), (0, 4)]),
        (5, [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4)]),
        (6, [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3)])
    ]
    for n, edges in graphs:
        for a in range(1 << n):
            allowed = {v for v in range(n) if a >> v & 1}
            for k in range(3):
                check(n, edges, list(range(1, n + 1)), allowed, k)
                comparisons += 1
    return {"seed": seed, "comparisons": comparisons, "mismatches": 0,
            "random_instances_by_budget": by_budget,
            "additional_permitted_set_comparisons": comparisons - 4 * repetitions,
            "seconds": time.perf_counter() - start}


def benchmark(seed=20261018, sizes=(20, 50, 100, 200), repetitions=3):
    rng, rows = random.Random(seed), []
    for n in sizes:
        for i in range(repetitions):
            edges = random_cactus(n, rng)
            costs = [rng.randint(1, 100) for _ in range(n)]
            for k in (0, 1, 2):
                start = time.perf_counter()
                cost, sensors, stats = solve_cactus(n, edges, costs, failures=k,
                                                   dominance=True)
                row = {"n": n, "m": len(edges), "instance": i, "failures": k,
                       "cost": cost, "sensors": len(sensors),
                       "seconds": time.perf_counter() - start,
                       "max_entries": stats["max_entries"], "pairs": stats["pairs"]}
                rows.append(row)
                print(json.dumps(row), flush=True)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["exhaustive", "random", "benchmark"])
    parser.add_argument("--max-n", type=int, default=6)
    parser.add_argument("--repetitions", type=int, default=200)
    args = parser.parse_args()
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    if args.mode == "exhaustive":
        result = exhaustive(args.max_n)
    elif args.mode == "random":
        result = random_checks(args.repetitions)
    else:
        result = benchmark(repetitions=args.repetitions)
        with (out / "benchmark.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(result[0]))
            writer.writeheader()
            writer.writerows(result)
    (out / (args.mode + ".json")).write_text(json.dumps(result, indent=2) + "\n")
    if args.mode != "benchmark":
        print(json.dumps(result, indent=2))
