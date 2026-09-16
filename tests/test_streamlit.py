from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_page_renders_without_runtime_errors() -> None:
    app = AppTest.from_file(
        str(PROJECT_ROOT / "code" / "deployment" / "app" / "app.py")
    ).run(timeout=15)
    assert not app.exception

