"""Shared paths for integration tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_DIR = PROJECT_ROOT / "code" / "deployment" / "api"

os.environ["MODEL_PATH"] = str(PROJECT_ROOT / "models" / "model.joblib")
sys.path.insert(0, str(API_DIR))

