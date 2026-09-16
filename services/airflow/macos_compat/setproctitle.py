"""Safe process-title no-ops for Airflow 2.x on macOS 26.

Airflow and its Gunicorn hooks call the native ``setproctitle`` extension from
forked processes. macOS 26 can crash those children inside CoreFoundation.
Process titles are only cosmetic, so local development uses these no-ops.
"""

from __future__ import annotations

import sys


def setproctitle(title: str) -> None:
    """Accept a process title without invoking the unsafe native extension."""


def getproctitle() -> str:
    """Return a stable title for Airflow's worker-ready bookkeeping."""

    return " ".join(sys.argv)


def setthreadtitle(title: str) -> None:
    """Accept a thread title without invoking the unsafe native extension."""


def getthreadtitle() -> str:
    """Return a stable placeholder thread title."""

    return "airflow"
