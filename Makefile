.PHONY: lint format check-format all

# Run ruff check and format in sequence
all: lint format

# Run ruff check
lint:
	uv run ruff check

# Run ruff format
format:
	uv run ruff format

# Run check and format together
check-format: lint format
