"""solve_relax 的随机差分测试：与暴力枚举的最小统一放宽量交叉验证。

对每个随机连通小实例，暴力从 e=0 起逐个检查“每条约束放宽为
[lower−e, upper+e]、基准点固定 0”后是否存在整数可行赋值，首个可行 e
即为全局最优（覆盖全部冲突环）；与 SPFA 负环检测 + 单调性二分求得的
relaxation 比对，并核对返回坐标是满足放宽后全部约束的同一组赋值。
"""

from __future__ import annotations

import itertools
import random

from app.schemas import ChainRequest, ConstraintIn
from app.solver import solve, solve_relax
from tests.test_random import random_instance


def brute_force_relaxation(datum, measure, constraints, cap: int = 60) -> int | None:
    """暴力求最小 e：枚举基准为 0 的全部整数赋值（放宽后界内），逐个 e 检查。"""
    nodes = list(dict.fromkeys([*datum, *measure]))
    datum_set = set(datum)
    bound = sum(max(abs(c.lower), abs(c.upper)) + 1 for c in constraints)
    value_ranges = [
        range(0, 1) if p in datum_set else range(-bound, bound + 1) for p in nodes
    ]
    idx = {p: i for i, p in enumerate(nodes)}
    assignments = list(itertools.product(*value_ranges))
    for e in range(cap + 1):
        for vals in assignments:
            if all(
                c.lower - e <= vals[idx[c.end]] - vals[idx[c.start]] <= c.upper + e
                for c in constraints
            ):
                return e
    return None


def _multi_datum_instance(rng: random.Random):
    datums = ["D1", "D2"]
    measure = ["M0", "M1"]
    nodes = datums + measure
    connected = [nodes[0]]
    cons = []
    cid = 0
    for child in nodes[1:]:
        parent = rng.choice(connected)
        lo, hi = sorted((rng.randint(-5, 5), rng.randint(-5, 5)))
        cons.append(ConstraintIn(id=f"c{cid}", start=parent, end=child, lower=lo, upper=hi))
        connected.append(child)
        cid += 1
    for _ in range(rng.randint(0, 2)):
        a, b = rng.sample(nodes, 2)
        lo, hi = sorted((rng.randint(-5, 5), rng.randint(-5, 5)))
        cons.append(ConstraintIn(id=f"c{cid}", start=a, end=b, lower=lo, upper=hi))
        cid += 1
    return datums, measure, cons


def test_random_relax_matches_brute_force():
    rng = random.Random(20260925)
    feasible_cases = infeasible_cases = 0
    for _ in range(200):
        datum, measure, constraints = random_instance(rng)
        req = ChainRequest(
            datum_points=datum, measure_points=measure, constraints=constraints
        )
        res = solve_relax(req)
        assert res.feasible and res.relaxation is not None
        e_star = res.relaxation

        bf = brute_force_relaxation(datum, measure, constraints)
        assert bf is not None, "暴力搜索范围内未找到可行 e"
        assert e_star == bf, (
            f"e mismatch: got {e_star}, bf {bf}, "
            f"constraints={[(c.start, c.end, c.lower, c.upper) for c in constraints]}"
        )

        coords = res.coordinates
        assert all(coords[d] == 0 for d in datum)
        assert all(
            c.lower - e_star <= coords[c.end] - coords[c.start] <= c.upper + e_star
            for c in constraints
        )
        assert all(isinstance(coords[p], int) for p in measure)

        if solve(req).feasible:
            feasible_cases += 1
            assert e_star == 0
        else:
            infeasible_cases += 1
            assert e_star > 0

    assert feasible_cases > 80
    assert infeasible_cases > 30  # 必须覆盖到需要放宽的不可行实例


def test_random_relax_multiple_datums_match_brute_force():
    rng = random.Random(7717)
    checked = 0
    for _ in range(150):
        datum, measure, constraints = _multi_datum_instance(rng)
        req = ChainRequest(
            datum_points=datum, measure_points=measure, constraints=constraints
        )
        res = solve_relax(req)
        e_star = res.relaxation
        bf = brute_force_relaxation(datum, measure, constraints)
        assert bf is not None
        assert e_star == bf, (
            f"got {e_star}, bf {bf}, "
            f"cons={[(c.start, c.end, c.lower, c.upper) for c in constraints]}"
        )
        coords = res.coordinates
        assert all(coords[d] == 0 for d in datum)
        assert all(
            c.lower - e_star <= coords[c.end] - coords[c.start] <= c.upper + e_star
            for c in constraints
        )
        checked += 1
    assert checked == 150
