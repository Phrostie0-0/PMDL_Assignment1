"""Airflow orchestration for data preparation, training, and deployment."""

from __future__ import annotations

import shlex
import shutil
import sys
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PYTHON = shlex.quote(sys.executable)
COMPOSE_FILE = PROJECT_ROOT / "code" / "deployment" / "docker-compose.yml"
DOCKER = shutil.which("docker")
if DOCKER is None:
    macos_docker = Path("/Applications/Docker.app/Contents/Resources/bin/docker")
    DOCKER = str(macos_docker) if macos_docker.exists() else "docker"

DEFAULT_ARGS = {
    "owner": "exoplanet-mass-lab",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


with DAG(
    dag_id="exoplanet_mass_pipeline",
    description="Clean NASA data, train the mass model, and deploy API plus UI",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
    schedule_interval="*/5 * * * *",
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "nasa", "exoplanets"],
    doc_md=__doc__,
) as dag:
    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command=f"{PYTHON} code/datasets/prepare_data.py",
        cwd=str(PROJECT_ROOT),
    )

    train_and_evaluate = BashOperator(
        task_id="train_and_evaluate",
        bash_command=f"{PYTHON} code/models/train.py",
        cwd=str(PROJECT_ROOT),
    )

    build_and_deploy = BashOperator(
        task_id="build_and_deploy",
        bash_command=(
            f"{shlex.quote(DOCKER)} compose -f {shlex.quote(str(COMPOSE_FILE))} "
            "up --build --detach --remove-orphans"
        ),
        cwd=str(PROJECT_ROOT),
        env={"PATH": f"{Path(DOCKER).parent}:{os.environ.get('PATH', '')}"},
        append_env=True,
    )

    api_health_check = BashOperator(
        task_id="api_health_check",
        bash_command=f"{PYTHON} code/deployment/healthcheck.py --timeout 120",
        cwd=str(PROJECT_ROOT),
    )

    prepare_data >> train_and_evaluate >> build_and_deploy >> api_health_check
