"""Checks for the adversarial oracle and the weighted spider formula."""
import itertools
import json
import math
import random
from pathlib import Path

from fort_profiles import minimum_failure_cost, power_closure, solve_cactus
from validate import brute_optimum, random_cactus


def direct_attack(n, edges, sensors, attack_costs):
    sensors = list(sensors)
    best = math.inf
    for a in range(1 << len(sensors)):
        failed = {v for i, v in enumerate(sensors) if a >> i & 1}
        if len(power_closure(n, edges, set(sensors) - failed)) < n:
            best = min(best, sum(attack_costs[v] for v in failed))
    return best


def spider(arm_lengths):
    edges, arms, size = [], [], 1
    for length in arm_lengths:
        arm = list(range(size, size + length))
        path = [0] + arm
        edges.extend(zip(path, path[1:]))
        arms.append(arm)
        size += length
    return size, edges, arms


def spider_optimum(arms, costs, allowed, failures):
    """Prefix-sum closed formula, independently of the profile algorithm."""
    prefix = []
    for arm in arms:
        marginal = sorted(costs[v] for v in arm if v in allowed)
        prefix.append([0] + list(itertools.accumulate(marginal)))

    def c(i, q):
        return prefix[i][q] if q < len(prefix[i]) else math.inf

    best = math.inf
    for center in range(2):
        if center and 0 not in allowed:
            continue
        h = failures + 1 - center
        for t in range(h // 2 + 1):
            q = h - t
            for exceptional in range(len(arms)):
                cost = center * costs[0] + sum(
                    c(i, t if i == exceptional else q) for i in range(len(arms))
                )
                best = min(best, cost)
    return best


def main(seed=20261018):
    rng, oracle_checks, spider_checks = random.Random(seed), 0, 0
    for i in range(200):
        n = rng.randint(1, 9)
        edges = random_cactus(n, rng)
        sensors = {v for v in range(n) if rng.random() < 0.65}
        attack = [rng.randint(0, 20) for _ in range(n)]
        exact = direct_attack(n, edges, sensors, attack)
        result = minimum_failure_cost(n, edges, sensors, attack, certificate=True)
        assert result['cost'] == exact, (n, edges, sensors, attack, result, exact)
        assert len(power_closure(n, edges, sensors - set(result['failed_sensors']))) < n
        oracle_checks += 1
    for lengths in itertools.product((1, 2, 3), repeat=3):
        n, edges, arms = spider(lengths)
        for trial in range(4):
            costs = [rng.randint(0, 20) for _ in range(n)]
            allowed = {v for v in range(n) if rng.random() > 0.2}
            for k in range(5):
                formula = spider_optimum(arms, costs, allowed, k)
                exact, _ = brute_optimum(n, edges, costs, allowed, k)
                dp, _, _ = solve_cactus(n, edges, costs, allowed, k, dominance=True)
                assert formula == exact == dp, (lengths, costs, allowed, k, formula, exact, dp)
                spider_checks += 1
    # Attack formula rho(S) = 1_{center in S} + two least arm occupancies.
    for lengths in ((1, 1, 1), (1, 2, 3), (2, 2, 2), (1, 1, 1, 1)):
        n, edges, arms = spider(lengths)
        for mask in range(1 << n):
            sensors = {v for v in range(n) if mask >> v & 1}
            counts = sorted(len(sensors & set(arm)) for arm in arms)
            formula = (0 in sensors) + sum(counts[:2])
            oracle = minimum_failure_cost(n, edges, sensors, certificate=True)
            assert oracle['cost'] == formula
            oracle_checks += 1
    result = {'seed': seed, 'weighted_attack_and_margin_checks': oracle_checks,
              'weighted_spider_optimization_checks': spider_checks, 'mismatches': 0}
    path = Path(__file__).parent / 'results' / 'structural.json'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
