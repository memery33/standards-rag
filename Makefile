.PHONY: help install ingest eval eval-all compare serve test lint verify clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "};{printf "  \033[36m%-12s\033[0m %s\n",$$1,$$2}'

install: ## Install dependencies into a local venv
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

ingest: ## Parse and chunk data/raw, printing the chunk inventory
	.venv/bin/python -m src.ingest.run $(if $(CONFIG),--config $(CONFIG))

eval: ## Score one configuration (CONFIG=configs/x.json, default if omitted)
	.venv/bin/python -m src.eval.run $(if $(CONFIG),--config $(CONFIG))

eval-all: ## Run every arm that needs no model backend
	@for c in configs/fixedwindow-bm25.json configs/sectionaware-bm25.json \
	          configs/fixedwindow-bm25-rerank.json configs/sectionaware-bm25-rerank.json; do \
		echo "== $$c"; .venv/bin/python -m src.eval.run --config $$c || exit 1; \
	done

compare: ## Print the comparison table across all committed runs
	.venv/bin/python -m src.eval.compare

serve: ## Ask one question against the corpus (Q="your question")
	.venv/bin/python -m src.generation.serve $(if $(CONFIG),--config $(CONFIG)) $(if $(Q),--question "$(Q)")

test: ## Run unit tests
	.venv/bin/python -m pytest tests -q

verify: ## Re-run every committed run and check its metrics still reproduce
	.venv/bin/python -m src.eval.verify

lint: ## Lint and format check
	.venv/bin/python -m ruff check src tests

clean: ## Remove build artifacts and rebuildable indexes
	rm -rf .index .cache __pycache__ .pytest_cache .ruff_cache
	find . -name '*.pyc' -delete
