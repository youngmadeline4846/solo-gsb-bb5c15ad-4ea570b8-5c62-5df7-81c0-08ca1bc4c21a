"""/api/v1/chains/shims 接口测试（FastAPI TestClient）。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

PAYLOAD = {
    "datum_points": ["A"],
    "measure_points": ["B", "C"],
    "constraints": [
        {"id": "c1", "start": "A", "end": "B", "lower": 10, "upper": 12},
        {"id": "c2", "start": "B", "end": "C", "lower": 5, "upper": 8},
    ],
    "start": "A",
    "end": "C",
    "standard_thicknesses": [2, 5, 10, 12, 14, 16, 18, 20, 25, 16, 30],
    "target_thickness": 17,
}


def test_shims_feasible_selects_and_returns_coordinates():
    r = client.post("/api/v1/chains/shims", json=PAYLOAD)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body == {
        "status": "feasible",
        "unit": "um",
        "start": "A",
        "end": "C",
        "minimum": 15,
        "maximum": 20,
        "target_thickness": 17,
        "candidates": [16, 18, 20],
        "selected_thickness": 16,
        "coordinates": {"A": 0, "B": 11, "C": 16},
    }


def test_shims_tie_prefers_thinner():
    payload = {**PAYLOAD, "standard_thicknesses": [16, 20], "target_thickness": 18}
    body = client.post("/api/v1/chains/shims", json=payload).json()
    assert body["selected_thickness"] == 16
    assert body["coordinates"]["C"] - body["coordinates"]["A"] == 16


def test_shims_no_candidate_nulls_selection_and_coordinates():
    payload = {**PAYLOAD, "standard_thicknesses": [21, 30], "target_thickness": 100}
    r = client.post("/api/v1/chains/shims", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "feasible"
    assert body["minimum"] == 15 and body["maximum"] == 20
    assert body["candidates"] == []
    assert body["selected_thickness"] is None
    assert body["coordinates"] is None


def test_shims_infeasible_uses_contradiction_cycle_response():
    payload = {
        **PAYLOAD,
        "constraints": [
            {"id": "c1", "start": "A", "end": "B", "lower": 9, "upper": 10},
            {"id": "c2", "start": "B", "end": "C", "lower": 9, "upper": 10},
            {"id": "c3", "start": "C", "end": "A", "lower": -1, "upper": 0},
        ],
    }
    r = client.post("/api/v1/chains/shims", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "infeasible"
    assert set(body["conflicting_constraint_ids"]) == {"c1", "c2", "c3"}
    assert body["cycle_weight"] < 0
    assert body["cycle_edges"]
    # 不可行响应不含选配字段
    assert "selected_thickness" not in body


def test_shims_unknown_endpoint_422():
    payload = {**PAYLOAD, "end": "X"}
    r = client.post("/api/v1/chains/shims", json=payload)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert any(e["code"] == "unknown_point" and e["location"] == ["end"] for e in detail)


def test_shims_negative_thickness_422():
    payload = {**PAYLOAD, "standard_thicknesses": [1, -2, 3]}
    r = client.post("/api/v1/chains/shims", json=payload)
    assert r.status_code == 422


def test_shims_empty_and_non_integer_thickness_list_422():
    for bad in ([], [1.5], [True]):
        payload = {**PAYLOAD, "standard_thicknesses": bad}
        r = client.post("/api/v1/chains/shims", json=payload)
        assert r.status_code == 422, bad


def test_shims_missing_fields_422():
    r = client.post("/api/v1/chains/shims", json={"datum_points": ["A"]})
    assert r.status_code == 422


def test_shims_extra_field_422():
    payload = {**PAYLOAD, "unexpected": 1}
    r = client.post("/api/v1/chains/shims", json=payload)
    assert r.status_code == 422


def test_shims_original_validation_reused():
    # 未连通测点 + 未知约束点：沿用原尺寸链校验码与位置
    payload = {
        "datum_points": ["A"],
        "measure_points": ["B", "D"],
        "constraints": [
            {"id": "c1", "start": "A", "end": "B", "lower": 1, "upper": 2},
        ],
        "start": "A",
        "end": "B",
        "standard_thicknesses": [1],
        "target_thickness": 1,
    }
    r = client.post("/api/v1/chains/shims", json=payload)
    assert r.status_code == 422
    assert r.json()["detail"][0]["code"] == "point_not_connected_to_datum"


def test_shims_whitespace_stripped_in_endpoints():
    payload = {**PAYLOAD, "start": " A ", "end": " C "}
    body = client.post("/api/v1/chains/shims", json=payload).json()
    assert body["start"] == "A" and body["end"] == "C"
    assert body["selected_thickness"] == 16


def test_shims_openapi_documented():
    schema = client.get("/openapi.json").json()
    assert "/api/v1/chains/shims" in schema["paths"]
