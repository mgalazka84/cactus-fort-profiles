"""Exact minimum-cost fault-tolerant power domination on cactus graphs.

One sensor per permitted vertex; a failed sensor leaves its vertex and edges
in place.  The dynamic program keeps an entire adversarial fort profile for
each placement, rather than minimizing its coordinates independently.
Only Python's standard library is required.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from math import inf
import sys


@dataclass(frozen=True)
class Trace:
    left: object
    right: object
    selected: tuple[int, ...]


@dataclass
class Table:
    boundary: tuple[int, ...]
    # (boundary sensor mask, profile) -> (installation cost, decision trace)
    entries: dict
    fort_traces: tuple | None = None


def adjacency(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if not (0 <= u < n and 0 <= v < n) or u == v:
            raise ValueError("A finite simple graph on 0,...,n-1 is required")
        if v in adj[u]:
            raise ValueError("Duplicate edge")
        adj[u].add(v)
        adj[v].add(u)
    return adj


def cactus_blocks(adj):
    """Tarjan edge blocks, plus validation; returns (blocks, vertex incidence)."""
    n = len(adj)
    disc, low = [-1] * n, [0] * n
    stack, blocks = [], []
    clock = 0

    def visit(v, parent):
        nonlocal clock
        disc[v] = low[v] = clock
        clock += 1
        for u in sorted(adj[v]):
            if disc[u] < 0:
                stack.append((v, u))
                visit(u, v)
                low[v] = min(low[v], low[u])
                if low[u] >= disc[v]:
                    block = []
                    while True:
                        e = stack.pop()
                        block.append(e)
                        if e == (v, u):
                            break
                    blocks.append(block)
            elif u != parent and disc[u] < disc[v]:
                stack.append((v, u))
                low[v] = min(low[v], disc[u])

    for v in range(n):
        if disc[v] < 0:
            visit(v, -1)
    incidence = [[] for _ in range(n)]
    for i, block in enumerate(blocks):
        degree = {}
        for u, v in block:
            degree[u] = degree.get(u, 0) + 1
            degree[v] = degree.get(v, 0) + 1
        if len(block) > 1 and not (
            len(degree) == len(block) and all(d == 2 for d in degree.values())
        ):
            raise ValueError("Graph is not a cactus")
        for v in degree:
            incidence[v].append(i)
    return blocks, incidence


@lru_cache(maxsize=None)
def words(b):
    # 0,1,2: outside F with 0,1,at least 2 F-neighbors; 3: inside F.
    return tuple(product(range(4), repeat=b))


@lru_cache(maxsize=None)
def word_index(b):
    return {q: i for i, q in enumerate(words(b))}


class FortDP:
    def __init__(self, costs, allowed=None, failures=1, dominance=False,
                 fixed_sensors=None, attack_costs=None):
        if failures < 0 or not isinstance(failures, int):
            raise ValueError("failures must be a nonnegative integer")
        if any(c < 0 or c == inf for c in costs):
            raise ValueError("Finite nonnegative costs are required")
        self.costs = list(costs)
        self.allowed = set(range(len(costs))) if allowed is None else set(allowed)
        if not self.allowed <= set(range(len(costs))):
            raise ValueError("Invalid permitted vertex")
        self.t = failures + 1
        self.unreachable = self.t + 1
        self.dominance = dominance
        self.fixed_sensors = None if fixed_sensors is None else set(fixed_sensors)
        self.attack_costs = [1] * len(costs) if attack_costs is None else list(attack_costs)
        if fixed_sensors is not None:
            # No saturation occurs in the fixed-placement separation oracle.
            self.t = sum(self.attack_costs) + 1
            self.unreachable = self.t + 1
        self.stats = {"operations": 0, "max_entries": 0, "pairs": 0,
                      "max_by_boundary": {0: 0, 1: 0, 2: 0}}

    def finish(self, table):
        if self.dominance:
            table.entries = self.prune(table.entries)
        self.stats["operations"] += 1
        self.stats["max_entries"] = max(self.stats["max_entries"], len(table.entries))
        b = len(table.boundary)
        self.stats["max_by_boundary"][b] = max(
            self.stats["max_by_boundary"][b], len(table.entries)
        )
        return table

    @staticmethod
    def prune(entries):
        """Optional safe componentwise dominance, for identical boundary masks."""
        kept = {}
        by_mask = {}
        for (a, p), value in sorted(entries.items(), key=lambda kv: kv[1][0]):
            candidates = by_mask.setdefault(a, [])
            if any(all(x >= y for x, y in zip(q, p)) for q in candidates):
                continue
            candidates.append(p)
            kept[a, p] = value
        return kept

    def primitive(self, boundary, edges=()):
        b = tuple(sorted(boundary))
        local = {v: set() for v in b}
        for u, v in edges:
            local[u].add(v)
            local[v].add(u)
        p = [self.unreachable] * (4 ** len(b))
        for f in range(1, 1 << len(b)):
            fs = {v for i, v in enumerate(b) if f >> i & 1}
            q = tuple(3 if v in fs else min(2, len(local[v] & fs)) for v in b)
            p[word_index(len(b))[q]] = 0
        p = tuple(p)
        entries = {}
        for a in range(1 << len(b)):
            if any(a >> i & 1 and v not in self.allowed for i, v in enumerate(b)):
                continue
            if self.fixed_sensors is not None and any(
                ((a >> i) & 1) != (v in self.fixed_sensors) for i, v in enumerate(b)
            ):
                continue
            entries[a, p] = (0, 0)
        traces = tuple(None for _ in p) if self.fixed_sensors is not None else None
        return self.finish(Table(b, entries, traces))

    def singleton(self, v):
        return self.primitive((v,))

    def edge(self, u, v):
        return self.primitive((u, v), ((u, v),))

    def combine(self, left, right, out_boundary):
        """Union edge-disjoint fragments sharing only their named boundary.

        Vertices in the union boundary but not out_boundary become internal.
        All their incident edges must already belong to the union fragment.
        """
        b1, b2 = left.boundary, right.boundary
        out = tuple(sorted(out_boundary))
        union = tuple(sorted(set(b1) | set(b2)))
        if not set(out) <= set(union) or len(out) > 2:
            raise ValueError("Invalid output interface")
        shared = sorted(set(b1) & set(b2))
        i1, i2 = {v: i for i, v in enumerate(b1)}, {v: i for i, v in enumerate(b2)}
        forgotten = tuple(v for v in union if v not in out)
        loc = {v: i for i, v in enumerate(union)}
        transitions = []
        w1, w2 = words(len(b1)), words(len(b2))
        # The final extra index is the EMPTY fort, separate from nonempty
        # partial forts that happen to have an all-zero boundary state.
        for j1 in range(len(w1) + 1):
            q1 = w1[j1] if j1 < len(w1) else (0,) * len(b1)
            for j2 in range(len(w2) + 1):
                if j1 == len(w1) and j2 == len(w2):
                    continue
                q2 = w2[j2] if j2 < len(w2) else (0,) * len(b2)
                if any((q1[i1[v]] == 3) != (q2[i2[v]] == 3) for v in shared):
                    continue
                q = {}
                for v in union:
                    if v in i1 and v in i2:
                        x, y = q1[i1[v]], q2[i2[v]]
                        q[v] = 3 if x == 3 else min(2, x + y)
                    elif v in i1:
                        q[v] = q1[i1[v]]
                    else:
                        q[v] = q2[i2[v]]
                if any(q[v] == 1 for v in forgotten):
                    continue
                covered = sum(1 << loc[v] for v in forgotten if q[v] in (2, 3))
                j = word_index(len(out))[tuple(q[v] for v in out)]
                in_fort = tuple(v for v in forgotten if q[v] == 3)
                transitions.append((j1, j2, j, covered, in_fort))
        result = {}
        result_traces = None
        for (a1, p1), (c1, s1) in left.entries.items():
            for (a2, p2), (c2, s2) in right.entries.items():
                if any(((a1 >> i1[v]) & 1) != ((a2 >> i2[v]) & 1) for v in shared):
                    continue
                self.stats["pairs"] += 1
                selected = {v: ((a1 >> i1[v]) & 1) if v in i1 else ((a2 >> i2[v]) & 1)
                            for v in union}
                amask = sum(selected[v] << loc[v] for v in union)
                aout = sum(selected[v] << i for i, v in enumerate(out))
                pc1, pc2 = p1 + (0,), p2 + (0,)
                p = [self.unreachable] * (4 ** len(out))
                best_forts = [None] * len(p) if self.fixed_sensors is not None else None
                for j1, j2, j, covered, in_fort in transitions:
                    if pc1[j1] == self.unreachable or pc2[j2] == self.unreachable:
                        continue
                    additional = sum(self.attack_costs[v] for v in forgotten
                                     if selected[v] and covered >> loc[v] & 1)
                    x = min(self.t, pc1[j1] + pc2[j2] + additional)
                    if x < p[j]:
                        p[j] = x
                        if best_forts is not None:
                            f1 = (left.fort_traces[j1] if left.fort_traces is not None
                                  and j1 < len(left.fort_traces) else None)
                            f2 = (right.fort_traces[j2] if right.fort_traces is not None
                                  and j2 < len(right.fort_traces) else None)
                            best_forts[j] = Trace(f1, f2, in_fort)
                p = tuple(p)
                newly_selected = tuple(v for v in forgotten if selected[v])
                c = c1 + c2 + sum(self.costs[v] for v in newly_selected)
                key = aout, p
                if key not in result or c < result[key][0]:
                    result[key] = (c, Trace(s1, s2, newly_selected))
                    if best_forts is not None:
                        result_traces = tuple(best_forts)
        return self.finish(Table(out, result, result_traces))

    def forget(self, table, out_boundary=()):
        # Union with an empty graph is a unary boundary-forget operation.
        empty = Table((), {(0, (self.unreachable,)): (0, 0)})
        return self.combine(table, empty, out_boundary)


def _build_cactus_table(n, edges, dp):
    if n < 1:
        raise ValueError("At least one vertex is required")
    # The graph-theoretic recursion has depth O(n).
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 4 * n + 100))
    adj = adjacency(n, edges)
    blocks, incidence = cactus_blocks(adj)
    seen_vertices = set()

    def vertex(v, parent_block):
        seen_vertices.add(v)
        branches = []
        for bi in incidence[v]:
            if bi == parent_block:
                continue
            block = blocks[bi]
            if len(block) == 1:
                u = next(x for x in block[0] if x != v)
                branch = dp.combine(dp.edge(v, u), vertex(u, bi), (v,))
            else:
                local = {}
                for x, y in block:
                    local.setdefault(x, []).append(y)
                    local.setdefault(y, []).append(x)
                sequence, prev, cur = [v], v, min(local[v])
                while cur != v:
                    sequence.append(cur)
                    nxt = next(x for x in local[cur] if x != prev)
                    prev, cur = cur, nxt
                u = sequence[1]
                branch = dp.combine(dp.edge(v, u), vertex(u, bi), (v, u))
                for w in sequence[2:]:
                    branch = dp.combine(branch, dp.edge(u, w), (v, w))
                    branch = dp.combine(branch, vertex(w, bi), (v, w))
                    u = w
                branch = dp.combine(branch, dp.edge(u, v), (v,))
            branches.append(branch)
        if not branches:
            return dp.singleton(v)
        table = branches[0]
        for branch in branches[1:]:
            table = dp.combine(table, branch, (v,))
        return table

    final = None
    for v in range(n):
        if v in seen_vertices:
            continue
        component = dp.forget(vertex(v, -1))
        final = component if final is None else dp.combine(final, component, ())
    return final


def solve_cactus(n, edges, costs=None, allowed=None, failures=1, dominance=False):
    """Return optimal cost, sensor list, and instrumentation; inf if infeasible."""
    costs = [1] * n if costs is None else list(costs)
    if len(costs) != n:
        raise ValueError("One cost per vertex is required")
    dp = FortDP(costs, allowed, failures, dominance)
    final = _build_cactus_table(n, edges, dp)
    answers = [val for (_, p), val in final.entries.items() if p[0] == dp.t]
    if not answers:
        return inf, [], dp.stats
    cost, trace = min(answers, key=lambda val: val[0])
    sensors, todo = set(), [trace]
    while todo:
        trace = todo.pop()
        if isinstance(trace, Trace):
            sensors.update(trace.selected)
            todo.extend((trace.left, trace.right))
    return cost, sorted(sensors), dp.stats


def minimum_failure_cost(n, edges, sensors, attack_costs=None, certificate=False):
    """Minimum total sensor-removal cost that destroys full observation.

    Returns zero if the placement does not power-dominate. With unit attack
    costs this is the minimum NUMBER of failed sensors required to break it.
    The supplied sensors are fixed; no installation choices are optimized.
    """
    sensors = set(sensors)
    if not sensors <= set(range(n)):
        raise ValueError("Invalid sensor vertex")
    attack_costs = [1] * n if attack_costs is None else list(attack_costs)
    if len(attack_costs) != n or any(c < 0 or c == inf for c in attack_costs):
        raise ValueError("Finite nonnegative attack costs are required")
    dp = FortDP([0] * n, fixed_sensors=sensors, attack_costs=attack_costs)
    final = _build_cactus_table(n, edges, dp)
    assert len(final.entries) == 1
    value = next(iter(final.entries))[1][0]
    if not certificate:
        return value
    fort, todo = set(), [final.fort_traces[0]]
    while todo:
        trace = todo.pop()
        if isinstance(trace, Trace):
            fort.update(trace.selected)
            todo.extend((trace.left, trace.right))
    adj = adjacency(n, edges)
    neighborhood = fort.copy()
    for v in fort:
        neighborhood.update(adj[v])
    failed = sensors & neighborhood
    assert fort and all(len(adj[v] & fort) != 1 for v in range(n) if v not in fort)
    assert sum(attack_costs[v] for v in failed) == value
    return {"cost": value, "fort": sorted(fort), "failed_sensors": sorted(failed)}


def power_closure(n, edges, sensors):
    """Independent direct observation simulator, without forts or interfaces."""
    adj = adjacency(n, edges)
    observed = set(sensors)
    for v in sensors:
        observed.update(adj[v])
    changed = True
    while changed:
        changed = False
        for v in tuple(observed):
            missing = adj[v] - observed
            if len(missing) == 1:
                observed.update(missing)
                changed = True
    return observed


if __name__ == "__main__":
    edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3)]
    print(solve_cactus(6, edges, costs=[6, 2, 4, 7, 1, 3], failures=1))
