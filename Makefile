.PHONY: env format data all

# Install editable package and all dev dependencies
env:
	pip install -e .[dev]

# Format codebase using Black and Ruff
format:
	ruff check src/ --fix
	black src/

# Run the data processing pipeline endpoint
data:
	python -m src.clean.pipeline --save-outputs

# Execute full replication: install env, format code, process data, compile paper
all: env format data paper

# Compile the LaTeX paper to PDF
paper:
	cd paper && latexmk -pdf main.tex -interaction=nonstopmode || pdflatex main.tex

# Render the analysis notebook to a self-contained HTML report (requires Quarto)
report:
	quarto render
	@echo "HTML report saved to _site/"

# Full pipeline: install, format, clean data, compile paper, render report
all: env format data paper report
