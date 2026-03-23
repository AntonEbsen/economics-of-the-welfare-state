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

# Execute full replication (Placeholder, as analysis endpoint is pending)
all: env format data
