.PHONY: help venv install eval test lint format check clean

VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
PYTEST = $(VENV)/bin/pytest
BLACK = $(VENV)/bin/black
FLAKE8 = $(VENV)/bin/flake8

help:
	@echo "Available targets:"
	@echo "  venv       - Create virtual environment"
	@echo "  install    - Install dependencies in venv"
	@echo "  eval       - Run bike obsession evaluation"
	@echo "  test       - Run tests"
	@echo "  lint       - Run flake8 linting"
	@echo "  format     - Format code with black"
	@echo "  check      - Run both linting and formatting checks"
	@echo "  clean      - Clean cache and venv"

$(VENV):
	python3 -m venv $(VENV)

venv: $(VENV)

install: $(VENV)
	$(PIP) install -r requirements.txt

eval: install
	$(PYTHON) evals/bike_obsession/bike_eval.py

test: install
	$(PYTEST) evals/bike_obsession/test_bike_interventions.py -v

lint: install
	$(FLAKE8) evals/ test_*.py

format: install
	$(BLACK) evals/ test_*.py

check: install
	@echo "Running black format check..."
	$(BLACK) --check evals/ test_*.py
	@echo "Running flake8 linting..."
	$(FLAKE8) evals/ test_*.py
	@echo "All code quality checks passed!"

clean:
	rm -rf $(VENV) __pycache__ *.pyc .DS_Store
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete