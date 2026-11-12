# Convenience targets. Python targets run inside the conda env: conda activate understudy
.PHONY: env hooks lint test test-unit test-integration test-e2e repro dag simulate mlflow ci \
        teacher-weights up up-fake down logs ps demo drill load bench dashboard openapi images push

env:              ## create the conda environment and install the package (editable)
	conda env create -f environment.yml || conda env update -f environment.yml --prune
	conda run -n understudy pip install -e . --no-deps
	conda run -n understudy pip freeze --exclude-editable > requirements.lock.txt

hooks:            ## enable the repo's git hooks (lint + unit tests pre-commit, DVC CI post-commit)
	git config core.hooksPath .githooks

lint:
	ruff check src tests scripts

test:
	pytest

test-unit:
	pytest -m unit

test-integration:
	pytest -m integration

test-e2e:         ## against the running compose stack
	pytest -m e2e

repro:            ## run every out-of-date pipeline stage (needs MLflow: `make mlflow` or `make up`)
	dvc repro

dag:
	dvc dag

simulate:         ## offline simulated traffic, e.g. make simulate WEEKS=0-1
	python -m understudy.data.simulate_traffic --weeks $(or $(WEEKS),0)

mlflow:           ## local MLflow tracking server on :5000 (same store as the compose service)
	scripts/mlflow_server.sh

ci:               ## the same checks the post-commit hook runs
	scripts/ci.sh; cat reports/ci/last_run.log

teacher-weights:  ## download the pinned teacher GGUF (2.1 GB) into models/gguf
	python -m understudy.teacher.download

# ---------------------------------------------------------------- stack
up:               ## build and start everything (UI: https://localhost:8443)
	docker compose up -d --build

up-fake:          ## same, with a fake teacher (no model download needed)
	TEACHER_FAKE=1 docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f --tail=50

ps:
	docker compose ps

demo:             ## drift demo: week 3 drift -> alert -> discover -> approve -> retrain -> week 4
	python scripts/drift_demo.py --drift-week $(or $(DRIFT),3) --after-week $(or $(AFTER),4)

drill:            ## roll back the champion under load and restore it
	python scripts/rollback_drill.py --seconds 30

load:             ## locust load test against the edge (2 min, 20 users)
	mkdir -p reports/load
	locust -f tests/load/locustfile.py --host http://localhost:8001 --headless -u 20 -r 5 -t 2m \
	  --csv reports/load/edge --only-summary

bench:            ## latency baselines (edge instant answers, teacher TTFT)
	python scripts/bench_edge.py --n 300 --concurrency 8
	python scripts/bench_teacher.py --n 20

dashboard:        ## regenerate the provisioned Grafana dashboard JSON
	python scripts/build_dashboard.py

openapi:          ## export OpenAPI specs to docs/openapi/
	python scripts/export_openapi.py

images:           ## build all images
	docker compose build

push:             ## push images to a registry: make push REGISTRY=docker.io/<user> TAG=v1
	REGISTRY=$(REGISTRY) TAG=$(or $(TAG),latest) docker compose build
	REGISTRY=$(REGISTRY) TAG=$(or $(TAG),latest) docker compose push mlflow teacher gateway edge console ui
