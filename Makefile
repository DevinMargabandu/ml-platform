.PHONY: install ingest train serve test lint clean reset

# ── Setup ─────────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt -q
	pip install pydantic-settings prometheus-client pytest httpx -q
	pip install -e . -q
	@echo "✓ Dependencies installed"

# ── Pipeline steps ────────────────────────────────────────────────────────────
ingest:
	PYTHONPATH=. python data/ingest.py

train:
	PYTHONPATH=. python training/train.py

# Run all three pipeline steps in sequence
pipeline: ingest train
	@echo "✓ Pipeline complete — run 'make serve' to start the API"

# ── Serving ───────────────────────────────────────────────────────────────────
serve:
	PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

serve-prod:
	PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

mlflow-ui:
	PYTHONPATH=. mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000

# ── Quality ───────────────────────────────────────────────────────────────────
test:
	PYTHONPATH=. pytest tests/ -v --tb=short

test-watch:
	PYTHONPATH=. pytest tests/ -v --tb=short -f

# ── Ops ───────────────────────────────────────────────────────────────────────
docker-build:
	docker build -t ml-platform:latest .

docker-up:
	docker-compose up --build

docker-down:
	docker-compose down -v

# ── Maintenance ───────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

# Wipe all generated data — forces full re-run of pipeline
reset: clean
	rm -f data/fraud.db mlruns/mlflow.db models/*.joblib
	@echo "✓ Reset complete — run 'make pipeline' to start fresh"

# ── Help ─────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  make install      Install dependencies"
	@echo "  make pipeline     Ingest data + train models (full setup)"
	@echo "  make serve        Start API with hot-reload (dev)"
	@echo "  make serve-prod   Start API with 4 workers (prod)"
	@echo "  make mlflow-ui    Open MLflow experiment tracker"
	@echo "  make test         Run test suite"
	@echo "  make docker-up    Start everything in Docker"
	@echo "  make reset        Wipe DB + models (clean slate)"
	@echo ""
