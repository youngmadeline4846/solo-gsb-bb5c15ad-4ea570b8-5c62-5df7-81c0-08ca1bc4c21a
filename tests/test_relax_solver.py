"""统一放宽（solve_relax）求解器单元测试。"""

from __future__ import annotations

import pytest

from app.schemas import ChainRequest
from app.solver import ValidationError, solve, solve_relax
from tests.test_solver import C


def req(datum, measure, constraints) -> ChainRequest:
    return ChainRequest(
        datum_points=list(datum),
        measure_points=list(measure),
        constraints=list(constraints),
    )


def check_relaxed(r: ChainRequest, res, e: int) -> None:
    """断言：返回 e 与同一组坐标满足基准固定、放宽后全部约束，且 e 最小。"""
    assert res.feasible
    assert res.relaxation == e
    coords = res.coordinates
    for d in r.datum_points:
        assert coords[d] == 0
    for c in r.constraints:
        assert c.lower - e <= coords[c.end] - coords[c.start] <= c.upper + e
    assert all(isinstance(v, int) for v in coords.values())
    # 最小性：e-1 必不可行（放宽后仍存在负环）
    if e > 0:
        assert not _feasible_with_slack(r, e - 1)
    else:
        # e=0 ⇔ 原请求可行
        assert solve(r).feasible


def _feasible_with_slack(r: ChainRequest, slack: int) -> bool:
    """直接构造放宽 slack 的差分图做一次 Bellman-Ford 可行性判定（测试用）。"""
    from app.solver import _bellman_ford, _build_edges

    anchor = r.datum_points[0]
    nodes = list(dict.fromkeys([*r.datum_points, *r.measure_points]))
    edges = _build_edges(r, anchor, slack=slack)
    _, _, bad_edge = _bellman_ford(nodes, edges, anchor)
    return bad_edge is None


# ---------- 原请求可行：e=0，坐标与 solve 一致 ----------

def test_feasible_request_needs_no_relaxation():
    r = req(
        ["A"], ["B", "C"],
        [C("c1", "A", "B", 10, 12), C("c2", "B", "C", 5, 8)],
    )
    res = solve_relax(r)
    check_relaxed(r, res, 0)
    assert res.coordinates == solve(r).coordinates == {"A": 0, "B": 12, "C": 20}


def test_zero_width_feasible_chain_e_zero():
    r = req(
        ["A"], ["B", "C"],
        [C("c1", "A", "B", 10, 10), C("c2", "B", "C", 5, 5)],
    )
    res = solve_relax(r)
    check_relaxed(r, res, 0)
    assert res.coordinates == {"A": 0, "B": 10, "C": 15}


# ---------- 单个矛盾闭环 ----------

def test_single_negative_cycle_min_e():
    # A→B∈[10,11]、B→C∈[10,11]、C→A∈[-1,0]：取下限反向边的环权 -21+3e
    # ⇒ e ≥ 7（e=6 时环权 -3 仍矛盾，不能整除向下取整）
    r = req(
        ["A"], ["B", "C"],
        [
            C("c1", "A", "B", 10, 11),
            C("c2", "B", "C", 10, 11),
            C("c3", "C", "A", -1, 0),
        ],
    )
    res = solve_relax(r)
    check_relaxed(r, res, 7)
    assert not solve(r).feasible


def test_three_edge_cycle_rounding_matters():
    # 环权 -17（-9-9+1），含 3 条约束边 ⇒ ⌈17/3⌉ = 6（e=5 仍有环权 -2）
    r = req(
        ["A"], ["B", "C"],
        [
            C("c1", "A", "B", 9, 10),
            C("c2", "B", "C", 9, 10),
            C("c3", "C", "A", -1, 0),
        ],
    )
    res = solve_relax(r)
    check_relaxed(r, res, 6)
    assert not _feasible_with_slack(r, 5)
    assert _feasible_with_slack(r, 6)


def test_single_point_overconstrained_e():
    # A→B 同时要求 B=10 和 B=-4：e=7 时两区间 [3,17]、[-11,3] 恰在 3 相交
    r = req(
        ["A"], ["B"],
        [C("c1", "A", "B", 10, 10), C("c2", "A", "B", -4, -4)],
    )
    res = solve_relax(r)
    check_relaxed(r, res, 7)
    assert res.coordinates["B"] in (3,)


# ---------- 多个冲突环：不能只按首个矛盾环估算 ----------

def test_two_disjoint_conflict_loops_require_max():
    # A-B-A：B=10 与 B=-3 两条精确约束 ⇒ 环权 -13+2e ⇒ e ≥ ⌈13/2⌉ = 7
    # A-C-A：C=6 与 C=0 ⇒ 环权 -6+2e ⇒ e ≥ 3
    # 只看先被检测到的一个环会估成 3 或 7；全局最小 e 必须为 7
    r = req(
        ["A"], ["B", "C"],
        [
            C("c1", "A", "B", 10, 10),
            C("c2", "A", "B", -3, -3),
            C("c3", "A", "C", 6, 6),
            C("c4", "A", "C", 0, 0),
        ],
    )
    res = solve_relax(r)
    check_relaxed(r, res, 7)
    assert not _feasible_with_slack(r, 6)
    assert _feasible_with_slack(r, 7)


def test_two_chained_cycles_with_different_k():
    # 环一（c1+c2）：2 条边、环权 -8 ⇒ e ≥ 4
    # 环二（c3 自环思路改为 A-C-A 精确矛盾）：单独构造 k 不同的环
    r = req(
        ["A"], ["B", "C"],
        [
            C("c1", "A", "B", 5, 5),
            C("c2", "A", "B", -3, -3),    # A-B-A：e ≥ ⌈8/2⌉ = 4
            C("c3", "A", "C", 0, 0),
            C("c4", "C", "C", -9, -1),    # C 自环：环权 1+e（上界边）⇒ e≥0；
        ],                                 # 下限边 -(-9)+e=9+e，不自相矛盾
    )
    # 自环 c4 要求 0 ∈ [-9-e, -1+e] ⇒ e ≥ 1；主导环仍是 A-B-A 的 4
    res = solve_relax(r)
    check_relaxed(r, res, 4)


# ---------- 自环矛盾 ----------

def test_self_loop_positive_lower_requires_e():
    # B-B ∈ [1,2]：必须 0 ∈ [1-e, 2+e] ⇒ e ≥ 1
    r = req(
        ["A"], ["B"],
        [C("c1", "A", "B", 0, 5), C("c2", "B", "B", 1, 2)],
    )
    res = solve_relax(r)
    check_relaxed(r, res, 1)


def test_self_loop_negative_upper_requires_e():
    # B-B ∈ [-5,-2]：必须 0 ∈ [-5-e, -2+e] ⇒ e ≥ 2
    r = req(
        ["A"], ["B"],
        [C("c1", "A", "B", 0, 5), C("c2", "B", "B", -5, -2)],
    )
    res = solve_relax(r)
    check_relaxed(r, res, 2)


# ---------- 基准点矛盾：pin 边不参与放宽 ----------

def test_datum_datum_conflict_pin_not_relaxed():
    # D1→D2 ∈ [5,8] 与 pin(D2→D1,0) 成环：5+e 上界边 + 0 pin 边不成负环；
    # 下限反向边 D2→D1 权 -5+e 与 pin(D1→D2,0) 成环 ⇒ e ≥ 5
    r = req(
        ["D1", "D2"], ["M"],
        [
            C("c1", "D1", "D2", 5, 8),
            C("c2", "D1", "M", 0, 5),
        ],
    )
    res = solve_relax(r)
    check_relaxed(r, res, 5)
    assert res.coordinates["D1"] == 0 and res.coordinates["D2"] == 0


def test_datum_pin_edges_keep_weight_zero():
    # 放宽后重建的图中 datum_pin 边权必须恒为 0（不随 e 变化）
    from app.solver import _build_edges

    r = req(["D1", "D2"], ["M"], [C("c1", "D1", "D2", 3, 7)])
    for slack in (0, 4, 100):
        pin_weights = [
            e.weight
            for e in _build_edges(r, "D1", slack=slack)
            if e.bound == "datum_pin"
        ]
        assert pin_weights and all(w == 0 for w in pin_weights)


# ---------- 同一组坐标 ----------

def test_returned_coordinates_are_one_joint_assignment():
    r = req(
        ["A"], ["B", "C"],
        [
            C("c1", "A", "B", 9, 10),
            C("c2", "B", "C", 9, 10),
            C("c3", "C", "A", -1, 0),
        ],
    )
    res = solve_relax(r)
    e = res.relaxation
    coords = res.coordinates
    # 同一组坐标同时让放宽后的三条约束全部成立（不只是各端点可达到）
    for c in r.constraints:
        d = coords[c.end] - coords[c.start]
        assert c.lower - e <= d <= c.upper + e


# ---------- 静态校验沿用 ChainRequest ----------

def relax_codes(r: ChainRequest):
    with pytest.raises(ValidationError) as ei:
        solve_relax(r)
    return {e["code"] for e in ei.value.errors}


def test_unknown_point_rejected():
    r = req(["A"], ["B"], [C("c1", "A", "GHOST", 1, 2)])
    assert relax_codes(r) == {"unknown_point"}


def test_reversed_interval_rejected():
    r = req(["A"], ["B"], [C("c1", "A", "B", 5, 4)])
    assert relax_codes(r) == {"reversed_interval"}


def test_unconnected_measure_rejected():
    r = req(
        ["A"], ["B", "C"],
        [C("c1", "A", "B", 1, 2)],
    )
    assert relax_codes(r) == {"point_not_connected_to_datum"}


def test_negative_coordinates_possible_after_relaxation():
    # A→B∈[-20,-10] 与 A→B∈[-5,-3] 矛盾；e≥5（⌈7/2⌉=... 计算：
    # 环 (-(-10)+e)+(-5+e) = 5+e-5+e = 2e ... 下界边 10+e、上界边 -5+e ⇒ 5+2e ⇒ 恒非负；
    # 另一方向：上界边 -10+e 与下界反向边 5+e ⇒ -5+2e<0 ⇒ e≥3）
    r = req(
        ["A"], ["B"],
        [C("c1", "A", "B", -20, -10), C("c2", "A", "B", -5, -3)],
    )
    res = solve_relax(r)
    check_relaxed(r, res, 3)
    assert res.coordinates["B"] < 0
