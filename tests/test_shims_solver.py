"""solve_shims 的单元测试：可达紧区间、规格筛选、平手取薄与坐标一致性。"""

from __future__ import annotations

import pytest

from app.schemas import ConstraintIn, ShimsRequest
from app.solver import ValidationError, solve_shims


def _req(start="A", end="C", thicknesses=(2, 5, 10, 12, 14, 16, 18, 20, 25, 30),
         target=17, constraints=None):
    if constraints is None:
        constraints = [
            ConstraintIn(id="c1", start="A", end="B", lower=10, upper=12),
            ConstraintIn(id="c2", start="B", end="C", lower=5, upper=8),
        ]
    return ShimsRequest(
        datum_points=["A"],
        measure_points=["B", "C"],
        constraints=constraints,
        start=start,
        end=end,
        standard_thicknesses=list(thicknesses),
        target_thickness=target,
    )


def test_reachable_interval_and_selection():
    res = solve_shims(_req())
    assert res.feasible
    assert res.shim_range == (15, 20)
    # 落入 [15,20] 的规格去重升序
    assert res.shim_candidates == [16, 18, 20]
    # 目标 17：|16−17|=|18−17|=1 平手取薄
    assert res.selected_thickness == 16
    coords = res.coordinates
    assert coords == {"A": 0, "B": 11, "C": 16}
    # 同一组坐标满足全部原约束且间隙恰为所选厚度
    assert 10 <= coords["B"] - coords["A"] <= 12
    assert 5 <= coords["C"] - coords["B"] <= 8
    assert coords["C"] - coords["A"] == 16


def test_coordinates_satisfy_original_constraints_for_each_candidate():
    req_obj = _req()
    for t in (16, 18, 20):
        res = solve_shims(_req(thicknesses=[t], target=t))
        assert res.selected_thickness == t
        coords = res.coordinates
        assert coords["A"] == 0
        assert 10 <= coords["B"] - coords["A"] <= 12
        assert 5 <= coords["C"] - coords["B"] <= 8
        assert coords["C"] - coords["A"] == t
    # 避免未使用变量告警，req_obj 的构造本身也是一个可用性检查
    assert isinstance(req_obj, ShimsRequest)


def test_no_candidate_returns_null():
    res = solve_shims(_req(thicknesses=[21, 22, 30], target=100))
    assert res.feasible
    assert res.shim_range == (15, 20)
    assert res.shim_candidates == []
    assert res.selected_thickness is None
    assert res.coordinates == {}


def test_reverse_direction_negative_interval():
    # C − A ∈ [-20,-15]：非负厚度无规格可选
    res = solve_shims(_req(start="C", end="A", thicknesses=[0, 2, 16], target=-17))
    assert res.feasible
    assert res.shim_range == (-20, -15)
    assert res.shim_candidates == []
    assert res.selected_thickness is None


def test_start_equals_end_is_zero_gap():
    res = solve_shims(_req(start="B", end="B", thicknesses=[0, 1], target=5))
    assert res.feasible
    assert res.shim_range == (0, 0)
    assert res.shim_candidates == [0]
    assert res.selected_thickness == 0
    assert res.coordinates["B"] - res.coordinates["B"] == 0


def test_duplicate_thicknesses_deduped():
    res = solve_shims(_req(thicknesses=[16, 16, 20, 20, 16], target=16))
    assert res.shim_candidates == [16, 20]
    assert res.selected_thickness == 16


def test_selection_closest_then_thinner():
    # 目标恰为某规格时选它
    assert solve_shims(_req(target=20)).selected_thickness == 20
    assert solve_shims(_req(target=15)).selected_thickness == 16
    # 目标远离时仍选最近者
    assert solve_shims(_req(target=100)).selected_thickness == 20
    assert solve_shims(_req(target=-100)).selected_thickness == 16


def test_infeasible_chain_returns_contradiction_cycle():
    constraints = [
        ConstraintIn(id="c1", start="A", end="B", lower=9, upper=10),
        ConstraintIn(id="c2", start="B", end="C", lower=9, upper=10),
        ConstraintIn(id="c3", start="C", end="A", lower=-1, upper=0),
    ]
    res = solve_shims(_req(thicknesses=[1, 2], target=1, constraints=constraints))
    assert not res.feasible
    assert res.cycle_weight < 0
    assert {e.cid for e in res.cycle_edges if e.cid} == {"c1", "c2", "c3"}


def test_unknown_shim_point_validation_error():
    with pytest.raises(ValidationError) as exc:
        solve_shims(_req(end="X"))
    codes = {e["code"] for e in exc.value.errors}
    assert codes == {"unknown_point"}
    assert exc.value.errors[0]["location"] == ["end"]


def test_datum_points_allowed_as_shim_endpoints():
    res = solve_shims(_req(start="A", end="A", thicknesses=[0], target=0))
    assert res.feasible
    assert res.shim_range == (0, 0)
    assert res.selected_thickness == 0


def test_chain_request_validation_reused():
    # 原尺寸链的静态校验（反向区间）同样对 shims 生效
    req = ShimsRequest(
        datum_points=["A"],
        measure_points=["B"],
        constraints=[ConstraintIn(id="c1", start="A", end="B", lower=9, upper=1)],
        start="A",
        end="B",
        standard_thicknesses=[1],
        target_thickness=1,
    )
    with pytest.raises(ValidationError) as exc:
        solve_shims(req)
    assert exc.value.errors[0]["code"] == "reversed_interval"


def test_zero_width_constraint_pins_gap():
    constraints = [
        ConstraintIn(id="c1", start="A", end="B", lower=10, upper=10),
    ]
    res = solve_shims(
        ShimsRequest(
            datum_points=["A"],
            measure_points=["B"],
            constraints=constraints,
            start="A",
            end="B",
            standard_thicknesses=[9, 10, 11],
            target_thickness=0,
        )
    )
    assert res.shim_range == (10, 10)
    assert res.shim_candidates == [10]
    assert res.selected_thickness == 10
    assert res.coordinates == {"A": 0, "B": 10}


def test_shim_endpoints_between_measure_points():
    # 间隙取 B->C（非基准点之间）：区间 [5,8]
    res = solve_shims(_req(start="B", end="C", thicknesses=[5, 6, 8, 9], target=7))
    assert res.shim_range == (5, 8)
    # |6−7|=|8−7|=1 平手取薄
    assert res.selected_thickness == 6
    coords = res.coordinates
    assert coords["C"] - coords["B"] == 6
    assert 10 <= coords["B"] - coords["A"] <= 12
