"""Download a compact exoplanet snapshot from the NASA Exoplanet Archive."""

from __future__ import annotations

import argparse
import sys
import urllib.parse
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "exoplanets.csv"
TAP_ENDPOINT = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

COLUMNS = (
    "pl_name",
    "hostname",
    "discoverymethod",
    "disc_year",
    "pl_bmasse",
    "pl_rade",
    "pl_orbper",
    "pl_orbsmax",
    "pl_orbeccen",
    "pl_eqt",
    "st_teff",
    "st_rad",
    "st_mass",
    "sy_snum",
    "sy_pnum",
)


def build_download_url() -> str:
    """Return the encoded synchronous TAP query used for this project."""
    query = (
        f"select {','.join(COLUMNS)} from pscomppars "
        "where pl_bmasse is not null and pl_rade is not null"
    )
    return f"{TAP_ENDPOINT}?{urllib.parse.urlencode({'query': query, 'format': 'csv'})}"


def download(output: Path) -> None:
    """Download atomically, keeping the existing snapshot if the request fails."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    request = urllib.request.Request(
        build_download_url(),
        headers={"User-Agent": "ExoplanetMassLab/1.0 (student MLOps project)"},
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
        if not payload.startswith(b"pl_name,"):
            raise RuntimeError("NASA response is not the expected CSV file")
        temporary.write_bytes(payload)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)

    rows = max(payload.count(b"\n") - 1, 0)
    print(f"Downloaded {rows:,} exoplanets to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        download(parse_args().output)
    except Exception as error:  # provide a concise CLI error while preserving the old file
        print(f"Download failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error

