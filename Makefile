.PHONY: setup test lint format run-api run-ui clean help frontend-setup frontend-dev frontend-build run-dev run-enhanced-api

# Default target
help:
	@echo "Available commands:"
	@echo "  make setup          - Install dependencies"
	@echo "  make test           - Run tests"
	@echo "  make lint           - Run linting"
	@echo "  make format         - Format code"
	@echo "  make run-api        - Run the API server"
	@echo "  make run-ui         - Run the UI server"
	@echo "  make run-dev        - Run both frontend and backend in development mode"
	@echo "  make run-enhanced-api - Run the enhanced FastAPI server"
	@echo "  make frontend-setup - Install frontend dependencies"
	@echo "  make frontend-dev   - Run frontend development server"
	@echo "  make frontend-build - Build frontend for production"
	@echo "  make clean          - Clean temporary files"

# Setup
setup:
	pip install -e ".[dev]"

frontend-setup:
	cd frontend && npm install

# Testing
test:
	python tests/run_tests.py

test-unit:
	python tests/run_tests.py unit

test-integration:
	python tests/run_tests.py integration

# Linting and formatting
lint:
	flake8 src api ui scripts tests

format:
	black src api ui scripts tests
	isort src api ui scripts tests

# Running the application
run:
	./run.py

run-api:
	./run.py --api-only

run-ui:
	./run.py --ui-only

run-debug:
	./run.py --debug

run-dev:
	./run_dev.py

# Enhanced API
run-enhanced-api:
	python -m uvicorn api.enhanced_app:app --host 0.0.0.0 --port 5050 --reload

# Frontend commands
frontend-dev:
	cd frontend && npm run dev

frontend-build:
	cd frontend && npm run build

# Cleaning
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .coverage -exec rm -rf {} +
	find . -type d -name htmlcov -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	rm -rf frontend/dist
	rm -rf frontend/.vite 