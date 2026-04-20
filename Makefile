.PHONY: env env-locked env-uv lock lock-uv verify-data format data analyze paper report all test lint

# Install editable package and all dev dependencies (follows pyproject bounds)
env:
	pip install -e .[dev]

# Install the exact pinned stack used for the published results. Use this
# for reproduction. Requires requirements-lock.txt to have been generated
# via `make lock` on a maintainer machine.
env-locked:
	pip install -r requirements-lock.txt
	pip install -e . --no-deps

# Install the exact pinned stack via uv.lock. Drop-in replacement for
# `env-locked` for users who prefer uv's resolver (dramatically faster
# installs, same PyPI artefacts). Requires `uv` on PATH —
# https://docs.astral.sh/uv/.
env-uv:
	uv sync --extra dev

# Re-generate requirements-lock.txt from pyproject.toml. Requires pip-tools.
# Run this when bumping dependency bounds and commit the diff.
lock:
	pip install --quiet pip-tools
	pip-compile --extra=dev --output-file=requirements-lock.txt pyproject.toml

# Re-generate uv.lock from pyproject.toml. Commit both lock files in
# lockstep — they describe the same pinned graph via different tools.
lock-uv:
	uv lock

# Verify that data/raw/*.xlsx matches the pinned checksum manifest.
verify-data:
	econ-clean verify-data

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

# Run the data processing pipeline endpoint. The ``econ-clean`` console
# script is registered via ``[project.scripts]`` and wraps
# ``src.clean.pipeline.process_all_datasets``; ``python -m
# src.clean.pipeline`` still works as the direct escape hatch.
data:
	econ-clean clean

# Regenerate every LaTeX table and PNG/PDF figure from the master panel.
# Reads ``data/final/master_dataset.parquet`` (produced by ``make data``)
# and writes to ``outputs/{tables,figures}/``. Wrapping ``econ-clean
# analyze`` keeps the CLI as the single source of truth for which
# artefacts ship alongside the paper.
analyze:
	econ-clean analyze

# Compile the LaTeX paper to PDF
paper:
	cd paper && latexmk -pdf main.tex -interaction=nonstopmode || pdflatex main.tex

# Render the analysis notebook to a self-contained HTML report (requires Quarto)
report:
	quarto render
	@echo "HTML report saved to _site/"

# Full pipeline: install, format, clean data, regenerate analysis artefacts,
# compile paper, render report. ``analyze`` sits between ``data`` and
# ``paper`` so the paper always picks up the freshest tables.
all: env format data analyze paper report
