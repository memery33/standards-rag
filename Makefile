.PHONY: help install ingest eval serve test lint clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "};{printf "  \033[36m%-12s\033[0m %s\n",$$1,$$2}'

install: ## Install dependencies into a local venv
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

ingest: ## Parse, chunk and index everything in data/raw
	.venv/bin/python -m src.ingest.run

eval: ## Score the current configuration against the golden question set
	.venv/bin/python -m src.eval.run

serve: ## Run the local query interface
	.venv/bin/python -m src.generation.serve

test: ## Run unit tests
	.venv/bin/python -m pytest tests -q

lint: ## Lint and format check
	.venv/bin/python -m ruff check src tests

clean: ## Remove build artifacts and rebuildable indexes
	rm -rf .index .cache __pycache__ .pytest_cache .ruff_cache
	find . -name '*.pyc' -delete
