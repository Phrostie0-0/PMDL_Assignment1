PYTHON ?= python3
DOCKER ?= $(shell command -v docker 2>/dev/null || printf '/Applications/Docker.app/Contents/Resources/bin/docker')
AIRFLOW ?= $(shell command -v airflow 2>/dev/null || { test -x .venv/bin/airflow && printf '.venv/bin/airflow' || printf 'airflow'; })
DOCKER_BIN_DIR := $(dir $(DOCKER))
AIRFLOW_BIN_DIR := $(dir $(AIRFLOW))
COMPOSE_FILE := code/deployment/docker-compose.yml
AIRFLOW_DIR := $(CURDIR)/services/airflow
AIRFLOW_VERSION := 2.10.5
PYTHON_MINOR := $(shell $(PYTHON) -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
AIRFLOW_CONSTRAINTS := https://raw.githubusercontent.com/apache/airflow/constraints-$(AIRFLOW_VERSION)/constraints-$(PYTHON_MINOR).txt

ifeq ($(shell uname -s),Darwin)
AIRFLOW_PLATFORM_ENV := PYTHONPATH=$(AIRFLOW_DIR)/macos_compat:$(PYTHONPATH)
else
AIRFLOW_PLATFORM_ENV :=
endif

.PHONY: install download prepare train test deploy stop pipeline airflow

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt --constraint $(AIRFLOW_CONSTRAINTS)

download:
	$(PYTHON) code/datasets/download_data.py

prepare:
	$(PYTHON) code/datasets/prepare_data.py

train:
	$(PYTHON) code/models/train.py

test:
	$(PYTHON) -m pytest -q

deploy:
	PATH="$(DOCKER_BIN_DIR):$(PATH)" $(DOCKER) compose -f $(COMPOSE_FILE) up --build -d

stop:
	PATH="$(DOCKER_BIN_DIR):$(PATH)" $(DOCKER) compose -f $(COMPOSE_FILE) down

pipeline: prepare train deploy

airflow:
	PATH="$(AIRFLOW_BIN_DIR):$(PATH)" $(AIRFLOW_PLATFORM_ENV) AIRFLOW_HOME=$(AIRFLOW_DIR) AIRFLOW__CORE__LOAD_EXAMPLES=False AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=False $(AIRFLOW) standalone
