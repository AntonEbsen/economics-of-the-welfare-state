.PHONY: env env-locked lock verify-data format data paper report all test lint

# Install editable package and all dev dependencies (follows pyproject bounds)
env:
	pip install -e .[dev]

# Install the exact pinned stack used for the published results. Use this
# for reproduction. Requires requirements-lock.txt to have been generated
# via `make lock` on a maintainer machine.
env-locked:
	pip install -r requirements-lock.txt
	pip install -e . --no-deps

# Re-generate requirements-lock.txt from pyproject.toml. Requires pip-tools.
# Run this when bumping dependency bounds and commit the diff.
lock:
	pip install --quiet pip-tools
	pip-compile --extra=dev --output-file=requirements-lock.txt pyproject.toml

# Verify that data/raw/*.xlsx matches the pinned checksum manifest.
verify-data:
	python scripts/download_raw_data.py

# Format codebase using Black and Ruff
format:
	ruff check src/ --fix
	black src/

# Lint-only check (no auto-fix); mirrors the CI checks
lint:
	ruff check src/
	black --check src/

# Run the project test suite
test:
	pytest

# Run the data processing pipeline endpoint
data:
	python -m src.clean.pipeline --save-outputs

# Compile the LaTeX paper to PDF
paper:
	cd paper && latexmk -pdf main.tex -interaction=nonstopmode || pdflatex main.tex

# Render the analysis notebook to a self-contained HTML report (requires Quarto)
report:
	quarto render
	@echo "HTML report saved to _site/"

# Full pipeline: install, format, clean data, compile paper, render report
all: env format data paper report
