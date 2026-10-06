export DATABASE_URL ?= postgresql://app:app-dev-only@127.0.0.1:55410/app
PY = ./venv/bin/python
BASE = http://127.0.0.1:8410

bootstrap:      ## local virtualenv + database container
	python3 -m venv venv && ./venv/bin/pip install -q -r requirements-dev.txt
	docker compose up -d db
seed:           ## migrate and load the synthetic world (no-op if already seeded)
	$(PY) -m dc.seed --if-empty
dev:            ## API with reload on :8410, jobs run by a local worker
	($(PY) -m core.jobs dc.api &) && $(PY) -m uvicorn dc.api:app --reload --port 8410
test:
	$(PY) -m pytest --cov=dc --cov=core --cov-report=term-missing
demo:           ## the full stack in Docker, then the demo scenario against it
	docker compose up --build -d --wait && $(PY) -m core.scenario $(BASE)
load-test:      ## run `make demo` first
	$(PY) -m core.loadtest $(BASE) hall-viewer-demo 32 10 "GET /v1/facilities/35be94f2-f5b1-596f-8a32-10759afa5b7b/pue" "GET /v1/facilities/35be94f2-f5b1-596f-8a32-10759afa5b7b" 
evaluate:       ## held-out halls -> docs/evaluation.md
	$(PY) -m dc.evaluate > docs/evaluation.md
reset:          ## drop all data (volume included)
	docker compose down -v
.PHONY: bootstrap seed dev test demo load-test evaluate reset
