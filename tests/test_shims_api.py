"""/api/v1/chains/shims 接口测试（FastAPI TestClient）。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

BASE = {
    "datum_points": ["A"],
    "measure_points": ["B", "C"],
    "constraints": [
        {"id": "c1", "start": "A", "end": "B", "lower": 10, "upper": 12},
        {"id": "c2", "start": "B", "end": "C", "lower": 5, "upper": 8},
    ],
}


def test_shims_feasible_selects_thickness():
    payload = {
        **BASE,
        "start": "A",
        "end": "C",
        "standard_thicknesses": [0, 5, 10, 15, 18, 20, 22, 25],
        "target_thickness": 19,
    }
    r = client.post("/api/v1/chains/shims", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "feasible"
    assert body["unit"] == "um"
    assert body["interval"] == {"start": "A", "end": "C", "minimum": 15, "maximum": 20}
    assert body["target_thickness"] == 19
    assert body["selected_thickness"] == 18  # 18/20 并列，取较薄
    coords = body["coordinates"]
    assert coords["A"] == 0
    assert coords["C"] - coords["A"] == 18
    for c in BASE["constraints"]:
        assert c["lower"] <= coords[c["end"]] - coords[c["start"]] <= c["upper"]


def test_shims_no_match_returns_nulls():
    payload = {
        **BASE,
        "start": "A",
        "end": "C",
        "standard_thicknesses": [0, 5, 10, 22, 25],
        "target_thickness": 18,
    }
    r = client.post("/api/v1/chains/shims", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "feasible"
    assert body["interval"]["minimum"] == 15 and body["interval"]["maximum"] == 20
    assert body["selected_thickness"] is None
    assert body["coordinates"] is None


def test_shims_empty_catalog_returns_nulls():
    payload = {**BASE, "start": "A", "end": "C",
               "standard_thicknesses": [], "target_thickness": 18}
    r = client.post("/api/v1/chains/shims", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["selected_thickness"] is None and body["coordinates"] is None


def test_shims_infeasible_uses_same_response_as_check():
    payload = {
        "datum_points": ["A"],
        "measure_points": ["B", "C"],
        "constraints": [
            {"id": "c1", "start": "A", "end": "B", "lower": 9, "upper": 10},
            {"id": "c2", "start": "B", "end": "C", "lower": 9, "upper": 10},
            {"id": "c3", "start": "C", "end": "A", "lower": -1, "upper": 0},
        ],
        "start": "A",
        "end": "C",
        "standard_thicknesses": [10, 15, 20],
        "target_thickness": 18,
    }
    r = client.post("/api/v1/chains/shims", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "infeasible"
    assert set(body["conflicting_constraint_ids"]) == {"c1", "c2", "c3"}
    assert body["cycle_weight"] == -17
    assert body["cycle_edges"]


def test_shims_unknown_endpoint_422():
    payload = {**BASE, "start": "A", "end": "GHOST",
               "standard_thicknesses": [10], "target_thickness": 10}
    r = client.post("/api/v1/chains/shims", json=payload)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert {e["code"] for e in detail} == {"unknown_point"}
    assert detail[0]["location"] == ["end"]


def test_shims_chain_static_validation_422():
    payload = {
        "datum_points": ["A"],
        "measure_points": ["B", "C"],
        "constraints": [
            {"id": "c1", "start": "A", "end": "B", "lower": 1, "upper": 2},
        ],
        "start": "A",
        "end": "C",
        "standard_thicknesses": [1],
        "target_thickness": 1,
    }
    r = client.post("/api/v1/chains/shims", json=payload)
    assert r.status_code == 422
    assert r.json()["detail"][0]["code"] == "point_not_connected_to_datum"


def test_shims_negative_thickness_422():
    payload = {**BASE, "start": "A", "end": "C",
               "standard_thicknesses": [10, -3], "target_thickness": 10}
    assert client.post("/api/v1/chains/shims", json=payload).status_code == 422

    payload = {**BASE, "start": "A", "end": "C",
               "standard_thicknesses": [10], "target_thickness": -1}
    assert client.post("/api/v1/chains/shims", json=payload).status_code == 422


def test_shims_non_integer_thickness_422():
    payload = {**BASE, "start": "A", "end": "C",
               "standard_thicknesses": [10, 10.5], "target_thickness": 10}
    assert client.post("/api/v1/chains/shims", json=payload).status_code == 422


def test_shims_missing_field_422():
    # 缺 target_thickness
    payload = {**BASE, "start": "A", "end": "C",
               "standard_thicknesses": [10]}
    assert client.post("/api/v1/chains/shims", json=payload).status_code == 422


def test_shims_extra_field_422():
    payload = {**BASE, "start": "A", "end": "C",
               "standard_thicknesses": [10], "target_thickness": 10,
               "readings": {"B": 1}}
    assert client.post("/api/v1/chains/shims", json=payload).status_code == 422


def test_shims_openapi_documented():
    schema = client.get("/openapi.json").json()
    assert "/api/v1/chains/shims" in schema["paths"]
