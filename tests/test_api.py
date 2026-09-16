from __future__ import annotations

from fastapi.testclient import TestClient

from main import app


EARTH_ANALOGUE = {
    "planet_radius_earth": 1.0,
    "orbital_period_days": 365.25,
    "semi_major_axis_au": 1.0,
    "eccentricity": 0.0167,
    "equilibrium_temperature_k": 255.0,
    "stellar_temperature_k": 5772.0,
    "stellar_radius_solar": 1.0,
    "stellar_mass_solar": 1.0,
    "star_count": 1,
    "known_planet_count": 1,
    "discovery_method": "Transit",
}


def test_health_and_prediction_endpoints() -> None:
    with TestClient(app) as client:
        health = client.get("/health")
        prediction = client.post("/predict", json=EARTH_ANALOGUE)

    assert health.status_code == 200
    assert health.json()["model_loaded"] is True
    assert prediction.status_code == 200
    payload = prediction.json()
    assert payload["predicted_mass_earth"] > 0
    assert payload["predicted_mass_jupiter"] > 0
    assert payload["mass_band"]


def test_api_rejects_impossible_radius() -> None:
    invalid = {**EARTH_ANALOGUE, "planet_radius_earth": -1}
    with TestClient(app) as client:
        response = client.post("/predict", json=invalid)
    assert response.status_code == 422

