.PHONY: help install eval test check clean

VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

help:
	@echo "Available targets:"
	@echo "  install    - Create venv and install dependencies"
	@echo "  eval       - Run bike obsession evaluation"
	@echo "  test       - Run tests"
	@echo "  check      - Run code quality checks (format + lint)"
	@echo "  clean      - Clean cache and venv"

$(VENV):
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

install: $(VENV)
	$(PIP) install -e ".[dev]"

eval: install
	$(PYTHON) -m bike_obsessed_llm.evaluation.bike_eval

test: install
	$(PYTHON) -m pytest

check: install
	@echo "Running black format check..."
	$(PYTHON) -m black --check src/ tests/
	@echo "Running flake8 linting..."
	$(PYTHON) -m flake8 src/ tests/
	@echo "All code quality checks passed!"

clean:
	rm -rf $(VENV) __pycache__ *.pyc .DS_Store build/ dist/ *.egg-info/
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete