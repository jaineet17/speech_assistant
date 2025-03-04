.PHONY: setup test lint format run-api run-ui clean help

# Default target
help:
	@echo "Available commands:"
	@echo "  make setup      - Install dependencies"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run linting"
	@echo "  make format     - Format code"
	@echo "  make run-api    - Run the API server"
	@echo "  make run-ui     - Run the UI server"
	@echo "  make clean      - Clean temporary files"

# Setup
setup:
	pip install -e ".[dev]"

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

# Cleaning
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .coverage -exec rm -rf {} +
	find . -type d -name htmlcov -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} + 