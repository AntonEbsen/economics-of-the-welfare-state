.PHONY: env format data paper report all test lint

# Install editable package and all dev dependencies
env:
	pip install -e .[dev]

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
