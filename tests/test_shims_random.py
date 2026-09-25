"""solve_shims 的随机差分测试：紧区间与枚举可达差值集合交叉验证。

对每个随机可行实例取两个声明点作起终点，暴力枚举全部可行整数坐标，
收集 coord(end) − coord(start) 的真实可达整数集合，与最短路给出的
紧区间逐整数比对；再给随机非负标准厚度表与目标厚度，核对选配结果及
返回坐标确实满足原约束与所选厚度。
"""

from __future__ import annotations

import random

from app.schemas import ShimRequest
from app.solver import solve_shims
from tests.test_random import brute_force_solutions, random_instance


def _brute_achievable_diffs(datum, measure, constraints, start, end):
    diffs = set()
    # 基准点恒为 0：枚举结果只含测点，基准点差值按 0 补入
    for s in brute_force_solutions(datum, measure, constraints):
        xs = {p: s.get(p, 0) for p in {*datum, *measure}}
        diffs.add(xs[end] - xs[start])
    return diffs


def test_random_shim_interval_matches_brute_force():
    rng = random.Random(861219)
    checked = 0
    for _ in range(220):
        if checked >= 130:
            break
        datum, measure, constraints = random_instance(rng)
        solutions = list(brute_force_solutions(datum, measure, constraints))
        if not solutions:
            continue  # 不可行分支由矛盾闭环测试覆盖

        start, end = rng.sample(list(dict.fromkeys([*datum, *measure])), 2)
        # 非负整数标准厚度（含 0 附近与较大值），目标厚度同样非负
        catalog = [rng.randint(0, 12) for _ in range(rng.randint(0, 6))]
        target = rng.randint(0, 12)

        r = ShimRequest(
            datum_points=datum,
            measure_points=measure,
            constraints=constraints,
            start=start,
            end=end,
            standard_thicknesses=catalog,
            target_thickness=target,
        )
        res = solve_shims(r)
        assert res.feasible
        lo, hi = res.shim_interval

        diffs = _brute_achievable_diffs(datum, measure, constraints, start, end)
        assert diffs, "可行实例至少应有一个可达差值"
        assert (lo, hi) == (min(diffs), max(diffs)), (
            f"interval mismatch {lo,hi} vs {min(diffs),max(diffs)}; "
            f"{start=} {end=} "
            f"{[(c.start, c.end, c.lower, c.upper) for c in constraints]}"
        )
        # 可行系统中差值集合在紧区间上无整数缺口（差分约束可行域是 Z^n 凸多胞体）
        assert diffs == set(range(lo, hi + 1))

        in_range = sorted({t for t in catalog if lo <= t <= hi})
        if not in_range:
            assert res.selected_thickness is None
            assert res.coordinates == {}
        else:
            expect = min(in_range, key=lambda t: (abs(t - target), t))
            assert res.selected_thickness == expect
            coords = res.coordinates
            assert all(coords[d] == 0 for d in datum)
            assert all(
                c.lower <= coords[c.end] - coords[c.start] <= c.upper
                for c in constraints
            )
            assert coords[end] - coords[start] == expect
        checked += 1
    assert checked >= 130
