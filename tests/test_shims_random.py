"""solve_shims 的随机差分测试：可达紧区间与选配结果对暴力枚举交叉验证。

对每个随机连通小实例枚举全部可行整数赋值，直接计算
``coord(end) − coord(start)`` 的真实最小/最大值、落入区间的规格集合与
``|t−target|`` 最小（平手取薄）的选配结果，与最短路实现逐项比对，
并核对返回坐标满足全部原约束且间隙恰为所选厚度。
"""

from __future__ import annotations

import random

from app.schemas import ShimsRequest
from app.solver import solve_shims
from tests.test_random import brute_force_solutions, random_instance


def _gap(solution, p, datum_set):
    return 0 if p in datum_set else solution[p]


def test_random_shims_match_brute_force():
    rng = random.Random(20260925)
    feasible_checked = 0
    infeasible_checked = 0
    for _ in range(200):
        datum, measure, constraints = random_instance(rng)
        nodes = list(dict.fromkeys([*datum, *measure]))
        start, end = rng.sample(nodes, 2) if rng.random() < 0.8 else (
            rng.choice(nodes), rng.choice(nodes)
        )
        # 非负整数微米标准厚度目录（含重复、含区间外值）
        catalog = [rng.randint(0, 14) for _ in range(rng.randint(1, 6))]
        catalog += [rng.randint(0, 14) for _ in range(2)]
        target = rng.randint(-6, 20)

        req = ShimsRequest(
            datum_points=datum,
            measure_points=measure,
            constraints=constraints,
            start=start,
            end=end,
            standard_thicknesses=catalog,
            target_thickness=target,
        )
        solutions = list(brute_force_solutions(datum, measure, constraints))
        res = solve_shims(req)

        if not solutions:
            assert not res.feasible
            assert res.cycle_weight < 0
            infeasible_checked += 1
            continue

        assert res.feasible
        datum_set = set(datum)
        gaps = [_gap(s, end, datum_set) - _gap(s, start, datum_set) for s in solutions]
        bf_min, bf_max = min(gaps), max(gaps)
        assert res.shim_range == (bf_min, bf_max), (
            f"range mismatch: got {res.shim_range}, bf {(bf_min, bf_max)}, "
            f"start={start}, end={end}, "
            f"cons={[(c.start, c.end, c.lower, c.upper) for c in constraints]}"
        )

        bf_candidates = sorted({t for t in catalog if bf_min <= t <= bf_max})
        assert res.shim_candidates == bf_candidates
        bf_selected = (
            min(bf_candidates, key=lambda t: (abs(t - target), t))
            if bf_candidates
            else None
        )
        assert res.selected_thickness == bf_selected

        if bf_selected is None:
            assert res.coordinates == {}
        else:
            coords = res.coordinates
            assert all(coords[d] == 0 for d in datum)
            assert all(
                c.lower <= coords[c.end] - coords[c.start] <= c.upper
                for c in constraints
            )
            assert coords[end] - coords[start] == bf_selected
            assert all(isinstance(v, int) for v in coords.values())
        feasible_checked += 1

    assert feasible_checked > 120
    assert infeasible_checked > 20  # 必须覆盖到原链矛盾、沿用闭环响应的实例


def test_random_shims_every_interval_integer_is_attainable():
    """紧区间内每个整数（含端点）都必须由某组可行赋值达到——与“整数紧区间”语义核对。"""
    rng = random.Random(31415)
    checked = 0
    for _ in range(160):
        datum, measure, constraints = random_instance(rng)
        nodes = list(dict.fromkeys([*datum, *measure]))
        start, end = rng.choice(nodes), rng.choice(nodes)
        solutions = list(brute_force_solutions(datum, measure, constraints))
        if not solutions:
            continue
        req = ShimsRequest(
            datum_points=datum,
            measure_points=measure,
            constraints=constraints,
            start=start,
            end=end,
            standard_thicknesses=[0],
            target_thickness=0,
        )
        lo, hi = solve_shims(req).shim_range
        datum_set = set(datum)
        attainable = {
            _gap(s, end, datum_set) - _gap(s, start, datum_set) for s in solutions
        }
        assert all(t in attainable for t in range(lo, hi + 1))
        checked += 1
    assert checked > 80
