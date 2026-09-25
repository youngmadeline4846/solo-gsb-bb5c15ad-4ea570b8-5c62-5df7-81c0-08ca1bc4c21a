"""POST /api/v1/chains/relax 的 API 层测试。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def payload(constraints, datum=("A",), measure=("B", "C")):
    return {
        "datum_points": list(datum),
        "measure_points": list(measure),
        "constraints": constraints,
    }


FEASIBLE = [
    {"id": "c1", "start": "A", "end": "B", "lower": 10, "upper": 12},
    {"id": "c2", "start": "B", "end": "C", "lower": 5, "upper": 8},
]

INFEASIBLE = [
    {"id": "c1", "start": "A", "end": "B", "lower": 9, "upper": 10},
    {"id": "c2", "start": "B", "end": "C", "lower": 9, "upper": 10},
    {"id": "c3", "start": "C", "end": "A", "lower": -1, "upper": 0},
]


def test_relax_feasible_returns_e_zero_with_intervals():
    r = client.post("/api/v1/chains/relax", json=payload(FEASIBLE))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "feasible"
    assert body["unit"] == "um"
    assert body["relaxation"] == 0
    assert body["coordinates"] == {"A": 0, "B": 12, "C": 20}
    # e=0 时放宽区间与原区间逐字段一致，顺序与请求 constraints 相同
    assert body["constraints"] == [
        {
            "id": "c1", "start": "A", "end": "B",
            "original_lower": 10, "original_upper": 12,
            "relaxed_lower": 10, "relaxed_upper": 12,
        },
        {
            "id": "c2", "start": "B", "end": "C",
            "original_lower": 5, "original_upper": 8,
            "relaxed_lower": 5, "relaxed_upper": 8,
        },
    ]


def test_relax_infeasible_returns_min_e_and_relaxed_intervals():
    r = client.post("/api/v1/chains/relax", json=payload(INFEASIBLE))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["relaxation"] == 6
    coords = body["coordinates"]
    assert coords["A"] == 0
    # 同一组坐标满足放宽后的全部约束（下界减 e、上界加 e）
    for item in body["constraints"]:
        e = body["relaxation"]
        assert item["relaxed_lower"] == item["original_lower"] - e
        assert item["relaxed_upper"] == item["original_upper"] + e
        diff = coords[item["end"]] - coords[item["start"]]
        assert item["relaxed_lower"] <= diff <= item["relaxed_upper"]
    assert [i["id"] for i in body["constraints"]] == ["c1", "c2", "c3"]


def test_relax_covers_multiple_conflict_cycles():
    # 两个互不共享边的矛盾环：A-B-A 需 e≥7、A-C-A 需 e≥3
    body = client.post(
        "/api/v1/chains/relax",
        json=payload(
            [
                {"id": "c1", "start": "A", "end": "B", "lower": 10, "upper": 10},
                {"id": "c2", "start": "A", "end": "B", "lower": -3, "upper": -3},
                {"id": "c3", "start": "A", "end": "C", "lower": 6, "upper": 6},
                {"id": "c4", "start": "A", "end": "C", "lower": 0, "upper": 0},
            ]
        ),
    ).json()
    assert body["relaxation"] == 7
    coords = body["coordinates"]
    for item in body["constraints"]:
        diff = coords[item["end"]] - coords[item["start"]]
        assert item["relaxed_lower"] <= diff <= item["relaxed_upper"]


def test_relax_unknown_point_422():
    bad = [{"id": "c1", "start": "A", "end": "GHOST", "lower": 1, "upper": 2}]
    r = client.post("/api/v1/chains/relax", json=payload(bad))
    assert r.status_code == 422
    assert r.json()["detail"][0]["code"] == "unknown_point"


def test_relax_reversed_interval_422():
    bad = [
        {"id": "c1", "start": "A", "end": "B", "lower": 9, "upper": 8},
        {"id": "c2", "start": "B", "end": "C", "lower": 1, "upper": 2},
    ]
    r = client.post("/api/v1/chains/relax", json=payload(bad))
    assert r.status_code == 422
    assert r.json()["detail"][0]["code"] == "reversed_interval"


def test_relax_unconnected_measure_422():
    bad = [{"id": "c1", "start": "A", "end": "B", "lower": 1, "upper": 2}]
    r = client.post("/api/v1/chains/relax", json=payload(bad))
    assert r.status_code == 422
    assert r.json()["detail"][0]["code"] == "point_not_connected_to_datum"


def test_relax_non_integer_bound_422():
    bad = [
        {"id": "c1", "start": "A", "end": "B", "lower": 10, "upper": 12.5},
        {"id": "c2", "start": "B", "end": "C", "lower": 5, "upper": 8},
    ]
    r = client.post("/api/v1/chains/relax", json=payload(bad))
    assert r.status_code == 422


def test_relax_unknown_field_422():
    p = payload(FEASIBLE)
    p["bogus"] = 1
    r = client.post("/api/v1/chains/relax", json=p)
    assert r.status_code == 422


def test_relax_shares_request_body_with_check():
    # /relax 沿用 ChainRequest：与 /check 同请求体，不接受 readings 等额外字段
    p = payload(FEASIBLE)
    p["readings"] = {"B": 1, "C": 2}
    r = client.post("/api/v1/chains/relax", json=p)
    assert r.status_code == 422


def test_relax_datum_conflict_pins_stay_zero():
    # 基准点互钉边不参与放宽：D1→D2∈[5,8] 需 e≥5，两基准点坐标仍为 0
    body = client.post(
        "/api/v1/chains/relax",
        json=payload(
            [
                {"id": "c1", "start": "D1", "end": "D2", "lower": 5, "upper": 8},
                {"id": "c2", "start": "D1", "end": "M", "lower": 0, "upper": 5},
            ],
            datum=("D1", "D2"),
            measure=("M",),
        ),
    ).json()
    assert body["relaxation"] == 5
    assert body["coordinates"]["D1"] == 0
    assert body["coordinates"]["D2"] == 0


def test_relax_openapi_documented():
    schema = client.get("/openapi.json").json()
    assert "/api/v1/chains/relax" in schema["paths"]
    assert "RelaxResponse" in schema["components"]["schemas"]
    assert "RelaxedConstraint" in schema["components"]["schemas"]
