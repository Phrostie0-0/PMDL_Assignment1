"""FastAPI service for the packaged exoplanet mass model."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field


EARTH_MASSES_PER_JUPITER = 317.82838


class PredictionInput(BaseModel):
    """Human-friendly API contract, mapped to NASA column names internally."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )

    planet_radius_earth: float = Field(gt=0, le=30)
    orbital_period_days: float = Field(gt=0, le=1_000_000)
    semi_major_axis_au: float = Field(gt=0, le=1_000)
    eccentricity: float = Field(ge=0, le=1)
    equilibrium_temperature_k: float = Field(gt=0, le=10_000)
    stellar_temperature_k: float = Field(ge=1_000, le=50_000)
    stellar_radius_solar: float = Field(gt=0, le=100)
    stellar_mass_solar: float = Field(gt=0, le=100)
    star_count: int = Field(ge=1, le=10)
    known_planet_count: int = Field(ge=1, le=20)
    discovery_method: str = Field(min_length=1, max_length=100)

    def to_model_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "pl_rade": self.planet_radius_earth,
                    "pl_orbper": self.orbital_period_days,
                    "pl_orbsmax": self.semi_major_axis_au,
                    "pl_eqt": self.equilibrium_temperature_k,
                    "pl_orbeccen": self.eccentricity,
                    "st_teff": self.stellar_temperature_k,
                    "st_rad": self.stellar_radius_solar,
                    "st_mass": self.stellar_mass_solar,
                    "sy_snum": self.star_count,
                    "sy_pnum": self.known_planet_count,
                    "discoverymethod": self.discovery_method,
                }
            ]
        )


class PredictionOutput(BaseModel):
    predicted_mass_earth: float
    predicted_mass_jupiter: float
    mass_band: str
    model_version: str


def resolve_model_path() -> Path:
    configured_path = os.getenv("MODEL_PATH")
    if configured_path:
        return Path(configured_path).expanduser().resolve()

    module_path = Path(__file__).resolve()
    container_model = module_path.parent / "models" / "model.joblib"
    if container_model.exists():
        return container_model

    return module_path.parents[3] / "models" / "model.joblib"


def load_model_package(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run the training stage before starting the API."
        )
    package = joblib.load(path)
    if not isinstance(package, dict) or "model" not in package or "metadata" not in package:
        raise ValueError("Model package must contain `model` and `metadata`")
    return package


def mass_band(mass_earth: float) -> str:
    """Return a descriptive mass band, not a formal composition classification."""
    if mass_earth < 2:
        return "Earth-mass"
    if mass_earth < 10:
        return "Super-Earth-mass"
    if mass_earth < 50:
        return "Neptune-mass"
    if mass_earth < 300:
        return "Sub-Jovian-mass"
    return "Jupiter-mass"


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = resolve_model_path()
    app.state.model_path = model_path
    app.state.model_package = load_model_package(model_path)
    yield


app = FastAPI(
    title="Exoplanet Mass Predictor API",
    version="1.0.0",
    description=(
        "Estimate an exoplanet's mass from planetary, orbital, and stellar "
        "parameters using a model trained on the NASA Exoplanet Archive."
    ),
    lifespan=lifespan,
)


@app.get("/")
def root() -> Dict[str, str]:
    return {
        "service": "Exoplanet Mass Predictor API",
        "health": "/health",
        "documentation": "/docs",
    }


@app.get("/health")
def health(request: Request) -> Dict[str, Any]:
    package = request.app.state.model_package
    metadata = package["metadata"]
    return {
        "status": "ok",
        "model_loaded": True,
        "model_version": metadata.get("model_version", "unknown"),
    }


@app.get("/model-info")
def model_info(request: Request) -> Dict[str, Any]:
    return request.app.state.model_package["metadata"]


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: PredictionInput, request: Request) -> PredictionOutput:
    package = request.app.state.model_package
    try:
        prediction = float(package["model"].predict(payload.to_model_frame())[0])
    except Exception as error:
        raise HTTPException(status_code=500, detail="Model prediction failed") from error

    if not np.isfinite(prediction) or prediction < 0:
        raise HTTPException(status_code=500, detail="Model returned an invalid mass")

    metadata = package["metadata"]
    return PredictionOutput(
        predicted_mass_earth=round(prediction, 4),
        predicted_mass_jupiter=round(prediction / EARTH_MASSES_PER_JUPITER, 6),
        mass_band=mass_band(prediction),
        model_version=str(metadata.get("model_version", "unknown")),
    )
