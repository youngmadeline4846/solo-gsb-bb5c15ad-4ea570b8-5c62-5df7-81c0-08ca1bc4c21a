"""solve_shims 单元测试：可达紧区间、目标厚度选配与不可行沿用。"""

from __future__ import annotations

import pytest

from app.schemas import ConstraintIn, ShimRequest
from app.solver import (
    ValidationError,
    solve_shims,
    validate_request,
    validate_shim_endpoints,
)


def C(cid: str, start: str, end: str, lower: int, upper: int) -> ConstraintIn:
    return ConstraintIn(id=cid, start=start, end=end, lower=lower, upper=upper)


def req(
    datum,
    measure,
    constraints,
    start,
    end,
    thicknesses,
    target,
) -> ShimRequest:
    return ShimRequest(
        datum_points=list(datum),
        measure_points=list(measure),
        constraints=list(constraints),
        start=start,
        end=end,
        standard_thicknesses=list(thicknesses),
        target_thickness=target,
    )


def _coords_valid(r: ShimRequest, coords: dict[str, int], selected: int) -> bool:
    if any(coords[d] != 0 for d in r.datum_points):
        return False
    if not all(
        c.lower <= coords[c.end] - coords[c.start] <= c.upper for c in r.constraints
    ):
        return False
    return coords[r.end] - coords[r.start] == selected


def test_tight_interval_simple_chain():
    # A→B [10,12]、B→C [5,8] ⇒ C−A 可达区间 [15,20]
    r = req(["A"], ["B", "C"], [C("c1", "A", "B", 10, 12), C("c2", "B", "C", 5, 8)],
            "A", "C", [15, 16, 20], 17)
    res = solve_shims(r)
    assert res.feasible
    assert res.shim_interval == (15, 20)


def test_selects_nearest_to_target():
    r = req(["A"], ["B", "C"], [C("c1", "A", "B", 10, 12), C("c2", "B", "C", 5, 8)],
            "A", "C", [15, 18, 20], 19)
    res = solve_shims(r)
    assert res.selected_thickness == 18  # 18 与 20 距 19 同为 1，取较薄者
    assert _coords_valid(r, res.coordinates, 18)


def test_exact_target_hit():
    r = req(["A"], ["B", "C"], [C("c1", "A", "B", 10, 12), C("c2", "B", "C", 5, 8)],
            "A", "C", [15, 18, 20], 20)
    res = solve_shims(r)
    assert res.selected_thickness == 20
    assert _coords_valid(r, res.coordinates, 20)
    assert res.coordinates["C"] == 20


def test_duplicate_thicknesses_dont_matter():
    r = req(["A"], ["B"], [C("c1", "A", "B", 10, 12)],
            "A", "B", [12, 12, 10, 10], 11)
    res = solve_shims(r)
    assert res.selected_thickness == 10
    assert _coords_valid(r, res.coordinates, 10)


def test_no_candidate_in_interval_returns_null():
    r = req(["A"], ["B", "C"], [C("c1", "A", "B", 10, 12), C("c2", "B", "C", 5, 8)],
            "A", "C", [0, 5, 10, 22, 25], 18)
    res = solve_shims(r)
    assert res.feasible
    assert res.shim_interval == (15, 20)
    assert res.selected_thickness is None
    assert res.coordinates == {}


def test_empty_catalog_returns_null():
    r = req(["A"], ["B"], [C("c1", "A", "B", 10, 12)], "A", "B", [], 11)
    res = solve_shims(r)
    assert res.feasible
    assert res.shim_interval == (10, 12)
    assert res.selected_thickness is None


def test_zero_thickness_selectable():
    # A→B 区间包含 0：规格 0 在界内
    r = req(["A"], ["B"], [C("c1", "A", "B", -2, 3)], "A", "B", [0, 2, 3], 1)
    res = solve_shims(r)
    assert res.shim_interval == (-2, 3)
    assert res.selected_thickness == 0 or res.selected_thickness == 2
    # 目标 1：0 与 2 等距，并列取较薄的 0
    assert res.selected_thickness == 0
    assert _coords_valid(r, res.coordinates, 0)


def test_interval_may_be_entirely_negative():
    # 反向公差：C−A 落在负半轴，非负规格无一入选
    r = req(["A"], ["B"], [C("c1", "A", "B", -12, -10)], "A", "B", [0, 5, 10], 0)
    res = solve_shims(r)
    assert res.feasible
    assert res.shim_interval == (-12, -10)
    assert res.selected_thickness is None
    assert res.coordinates == {}


def test_start_equals_end_is_zero_interval():
    r = req(["A"], ["B"], [C("c1", "A", "B", 10, 12)], "A", "A", [0, 1], 0)
    res = solve_shims(r)
    assert res.feasible
    assert res.shim_interval == (0, 0)
    assert res.selected_thickness == 0
    assert _coords_valid(r, res.coordinates, 0)


def test_measure_points_allowed_as_endpoints():
    r = req(["A"], ["B", "C"], [C("c1", "A", "B", 10, 12), C("c2", "B", "C", 5, 8)],
            "C", "B", [4, 6, 8], 7)
    res = solve_shims(r)
    # B−C ∈ [−8,−5]，非负规格无一入选
    assert res.shim_interval == (-8, -5)
    assert res.selected_thickness is None


def test_cross_measure_constraint_tightens_interval():
    # c3 精确钉住 C−B=5：B−C 的区间被压成 [-5,-5]
    r = req(
        ["A"], ["B", "C"],
        [C("c1", "A", "B", 0, 10), C("c2", "A", "C", 0, 10), C("c3", "B", "C", 5, 5)],
        "C", "B", [0, 5], 0,
    )
    res = solve_shims(r)
    assert res.shim_interval == (-5, -5)
    assert res.selected_thickness is None

    # 正向取终点 C−起点 B 时区间被压成 [5,5]
    r2 = req(
        ["A"], ["B", "C"],
        [C("c1", "A", "B", 0, 10), C("c2", "A", "C", 0, 10), C("c3", "B", "C", 5, 5)],
        "B", "C", [0, 5, 10], 0,
    )
    res2 = solve_shims(r2)
    assert res2.shim_interval == (5, 5)
    assert res2.selected_thickness == 5
    assert _coords_valid(r2, res2.coordinates, 5)


def test_multiple_datums_pin_included_in_interval():
    r = req(
        ["D1", "D2"], ["M"],
        [C("c1", "D1", "M", 3, 7), C("c2", "M", "D2", -7, -3)],
        "D1", "M", [3, 5, 7], 4,
    )
    res = solve_shims(r)
    assert res.feasible
    assert res.shim_interval == (3, 7)
    # 距目标 4：|3−4|=1、|5−4|=1 并列 → 取较薄 3
    assert res.selected_thickness == 3
    assert _coords_valid(r, res.coordinates, 3)


def test_infeasible_chain_returns_same_cycle():
    r = req(
        ["A"], ["B", "C"],
        [C("c1", "A", "B", 9, 10), C("c2", "B", "C", 9, 10), C("c3", "C", "A", -1, 0)],
        "A", "C", [10, 15, 20], 18,
    )
    res = solve_shims(r)
    assert not res.feasible
    assert res.cycle_weight < 0
    assert {e.cid for e in res.cycle_edges if e.cid is not None} == {"c1", "c2", "c3"}
    assert res.shim_interval is None
    assert res.selected_thickness is None


def test_unknown_shim_endpoint_reports_unknown_point():
    r = ShimRequest(
        datum_points=["A"],
        measure_points=["B"],
        constraints=[C("c1", "A", "B", 10, 12)],
        start="A",
        end="GHOST",
        standard_thicknesses=[10],
        target_thickness=10,
    )
    # validate_request 通过（它不检查 start/end），再由专项校验拦截
    validate_request(r)
    with pytest.raises(ValidationError) as ei:
        validate_shim_endpoints(r)
    assert {e["code"] for e in ei.value.errors} == {"unknown_point"}
    with pytest.raises(ValidationError) as ei2:
        solve_shims(r)
    assert {e["code"] for e in ei2.value.errors} == {"unknown_point"}


def test_both_endpoints_unknown_collects_two_errors():
    r = ShimRequest(
        datum_points=["A"],
        measure_points=["B"],
        constraints=[C("c1", "A", "B", 10, 12)],
        start="X",
        end="Y",
        standard_thicknesses=[10],
        target_thickness=10,
    )
    with pytest.raises(ValidationError) as ei:
        solve_shims(r)
    assert sum(e["code"] == "unknown_point" for e in ei.value.errors) == 2


def test_chain_validation_still_applies():
    # 原尺寸链校验（未连通、反向区间）沿用，不被垫片字段掩盖
    r = req(["A"], ["B", "C"], [C("c1", "A", "B", 9, 1)], "A", "C", [10], 10)
    with pytest.raises(ValidationError) as ei:
        solve_shims(r)
    codes = {e["code"] for e in ei.value.errors}
    assert "reversed_interval" in codes
    assert "point_not_connected_to_datum" in codes


def test_negative_thickness_rejected_by_pydantic():
    from pydantic import ValidationError as PydanticValidationError

    with pytest.raises(PydanticValidationError):
        ShimRequest(
            datum_points=["A"],
            measure_points=["B"],
            constraints=[C("c1", "A", "B", 10, 12)],
            start="A",
            end="B",
            standard_thicknesses=[10, -1],
            target_thickness=10,
        )
    with pytest.raises(PydanticValidationError):
        ShimRequest(
            datum_points=["A"],
            measure_points=["B"],
            constraints=[C("c1", "A", "B", 10, 12)],
            start="A",
            end="B",
            standard_thicknesses=[10],
            target_thickness=-2,
        )


def test_extra_fields_forbidden():
    from pydantic import ValidationError as PydanticValidationError

    with pytest.raises(PydanticValidationError):
        ShimRequest(
            datum_points=["A"],
            measure_points=["B"],
            constraints=[C("c1", "A", "B", 10, 12)],
            start="A",
            end="B",
            standard_thicknesses=[10],
            target_thickness=10,
            bogus=1,
        )


def test_endpoint_fields_stripped():
    r = req(["A"], ["B"], [C("c1", "A", "B", 10, 12)], " A ", " B ", [12], 12)
    res = solve_shims(r)
    assert res.feasible and res.selected_thickness == 12
