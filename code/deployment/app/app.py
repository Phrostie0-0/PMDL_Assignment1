"""Streamlit interface for the Exoplanet Mass Predictor API."""

from __future__ import annotations

import os
from typing import Any, Dict

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

DISCOVERY_METHODS = [
    "Transit",
    "Radial Velocity",
    "Microlensing",
    "Imaging",
    "Transit Timing Variations",
    "Eclipse Timing Variations",
    "Astrometry",
    "Orbital Brightness Modulation",
    "Pulsar Timing",
    "Pulsation Timing Variations",
    "Disk Kinematics",
]

PRESETS: Dict[str, Dict[str, Any]] = {
    "Earth analogue": {
        "radius": 1.0,
        "period": 365.25,
        "axis": 1.0,
        "eccentricity": 0.0167,
        "temperature": 255.0,
        "star_temperature": 5772.0,
        "star_radius": 1.0,
        "star_mass": 1.0,
        "star_count": 1,
        "planet_count": 1,
        "method": "Transit",
    },
    "Mini-Neptune": {
        "radius": 3.0,
        "period": 12.0,
        "axis": 0.1,
        "eccentricity": 0.08,
        "temperature": 800.0,
        "star_temperature": 5000.0,
        "star_radius": 0.8,
        "star_mass": 0.8,
        "star_count": 1,
        "planet_count": 3,
        "method": "Transit",
    },
    "Hot Jupiter": {
        "radius": 11.2,
        "period": 3.5,
        "axis": 0.045,
        "eccentricity": 0.05,
        "temperature": 1500.0,
        "star_temperature": 6000.0,
        "star_radius": 1.2,
        "star_mass": 1.1,
        "star_count": 1,
        "planet_count": 1,
        "method": "Transit",
    },
}


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 15% 10%, rgba(76, 50, 160, .32), transparent 28%),
                radial-gradient(circle at 85% 25%, rgba(15, 110, 150, .25), transparent 25%),
                linear-gradient(160deg, #070914 0%, #0c1028 50%, #060812 100%);
        }
        [data-testid="stHeader"] { background: transparent; }
        .hero {
            padding: 1.3rem 1.5rem;
            border: 1px solid rgba(142, 166, 255, .25);
            border-radius: 18px;
            background: rgba(10, 15, 42, .7);
            box-shadow: 0 12px 40px rgba(0, 0, 0, .28);
            margin-bottom: 1rem;
        }
        .hero h1 { margin: 0; color: #f4f6ff; letter-spacing: .02em; }
        .hero p { margin: .45rem 0 0; color: #b9c3ec; }
        div[data-testid="stMetric"] {
            border: 1px solid rgba(130, 157, 255, .22);
            border-radius: 14px;
            background: rgba(12, 18, 50, .7);
            padding: .8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def api_status() -> None:
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        response.raise_for_status()
        version = response.json().get("model_version", "unknown")
        st.sidebar.success(f"API online · model {version}")
    except requests.RequestException:
        st.sidebar.error("API offline")


st.set_page_config(
    page_title="Exoplanet Mass Lab",
    page_icon="🪐",
    layout="wide",
)
apply_theme()

st.markdown(
    """
    <div class="hero">
      <h1>🪐 Exoplanet Mass Lab</h1>
      <p>Estimate an exoplanet's mass from its orbit, host star, and observed radius.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Mission control")
st.sidebar.caption(f"API: `{API_URL}`")
api_status()
st.sidebar.markdown(
    "Data source: NASA Exoplanet Archive. Predictions are educational estimates, "
    "not astronomical measurements."
)

preset_name = st.selectbox("Start with a planetary preset", list(PRESETS))
preset = PRESETS[preset_name]
widget_suffix = preset_name.lower().replace(" ", "-")

with st.form("prediction-form"):
    st.subheader("Planet and orbit")
    planet_a, planet_b, planet_c = st.columns(3)
    with planet_a:
        radius = st.number_input(
            "Planet radius (Earth radii)",
            min_value=0.01,
            max_value=30.0,
            value=float(preset["radius"]),
            key=f"radius-{widget_suffix}",
        )
        period = st.number_input(
            "Orbital period (days)",
            min_value=0.001,
            max_value=1_000_000.0,
            value=float(preset["period"]),
            key=f"period-{widget_suffix}",
        )
    with planet_b:
        axis = st.number_input(
            "Semi-major axis (AU)",
            min_value=0.0001,
            max_value=1_000.0,
            value=float(preset["axis"]),
            format="%.4f",
            key=f"axis-{widget_suffix}",
        )
        eccentricity = st.number_input(
            "Orbital eccentricity",
            min_value=0.0,
            max_value=1.0,
            value=float(preset["eccentricity"]),
            format="%.4f",
            key=f"eccentricity-{widget_suffix}",
        )
    with planet_c:
        temperature = st.number_input(
            "Equilibrium temperature (K)",
            min_value=1.0,
            max_value=10_000.0,
            value=float(preset["temperature"]),
            key=f"temperature-{widget_suffix}",
        )
        method = st.selectbox(
            "Discovery method",
            DISCOVERY_METHODS,
            index=DISCOVERY_METHODS.index(str(preset["method"])),
            key=f"method-{widget_suffix}",
        )

    st.subheader("Host system")
    star_a, star_b, star_c = st.columns(3)
    with star_a:
        star_temperature = st.number_input(
            "Star temperature (K)",
            min_value=1_000.0,
            max_value=50_000.0,
            value=float(preset["star_temperature"]),
            key=f"star-temperature-{widget_suffix}",
        )
        star_count = st.number_input(
            "Stars in system",
            min_value=1,
            max_value=10,
            value=int(preset["star_count"]),
            key=f"star-count-{widget_suffix}",
        )
    with star_b:
        star_radius = st.number_input(
            "Star radius (Solar radii)",
            min_value=0.001,
            max_value=100.0,
            value=float(preset["star_radius"]),
            key=f"star-radius-{widget_suffix}",
        )
        planet_count = st.number_input(
            "Known planets in system",
            min_value=1,
            max_value=20,
            value=int(preset["planet_count"]),
            key=f"planet-count-{widget_suffix}",
        )
    with star_c:
        star_mass = st.number_input(
            "Star mass (Solar masses)",
            min_value=0.001,
            max_value=100.0,
            value=float(preset["star_mass"]),
            key=f"star-mass-{widget_suffix}",
        )

    submitted = st.form_submit_button("Estimate mass", type="primary", use_container_width=True)

if submitted:
    payload = {
        "planet_radius_earth": radius,
        "orbital_period_days": period,
        "semi_major_axis_au": axis,
        "eccentricity": eccentricity,
        "equilibrium_temperature_k": temperature,
        "stellar_temperature_k": star_temperature,
        "stellar_radius_solar": star_radius,
        "stellar_mass_solar": star_mass,
        "star_count": star_count,
        "known_planet_count": planet_count,
        "discovery_method": method,
    }
    try:
        with st.spinner("Running orbital inference…"):
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
            response.raise_for_status()
        result = response.json()
        st.subheader("Estimated planetary mass")
        result_a, result_b, result_c = st.columns(3)
        result_a.metric("Earth masses", f"{result['predicted_mass_earth']:,.2f} M⊕")
        result_b.metric("Jupiter masses", f"{result['predicted_mass_jupiter']:,.4f} M♃")
        result_c.metric("Mass band", result["mass_band"])
        st.caption(
            "The mass band is descriptive only; composition cannot be determined "
            "from mass alone."
        )
    except requests.HTTPError as error:
        try:
            detail = error.response.json().get("detail", str(error))
        except ValueError:
            detail = str(error)
        st.error(f"The prediction service rejected the request: {detail}")
    except requests.RequestException as error:
        st.error(f"Could not reach the prediction API: {error}")

