Eval runs
=========

Every scored run is committed here. This directory is deliberately not
gitignored: an eval harness whose outputs are thrown away proves nothing.

One directory per run, named by date and configuration, for example
2026-09-06-hybrid-rerank-k5, containing:

- config.json          the exact configuration that produced the run
- metrics.json         recall@k, MRR, groundedness, citation accuracy, refusal rate
- per_question.jsonl   per-question outcome, so failures can be inspected
- notes.md             what changed since the last run and what it did to the numbers

Rules
-----

1. A run is only comparable to another run on the same golden set version.
Record the golden set commit hash in config.json.
2. Regressions get committed too. A results directory containing only
improvements has been curated, not measured.
3. Never edit a past run to match a later story. Add a new run instead.

Reading a run
-------------

Read `lift_strict@k` before `recall_strict@k`. Raw recall rewards a chunker
whose chunks are wider, because a chunk is credited with every heading path it
overlaps -- so the naive fixed-window baseline can post the higher recall while
being the worse retriever. Subtracting the random baseline removes that
advantage. `mean_paths_per_chunk` and `mean_context_tokens@k` show what a run
paid for its score.

`recall_loose@k` is 1.00 at k=3 for every run committed so far. It is not a
useful discriminator on this corpus; `recall_strict@k` is.

Run log
-------

Newest first. One line per run: the headline number and the one-sentence
reason the run exists.

- **2026-09-06-sectionaware-bm25-rerank** -- lift_strict@3 0.580, the best arm
  run so far. Adds lexical-overlap reranking to sectionaware-bm25.
- **2026-09-06-fixedwindow-bm25-rerank** -- lift_strict@3 0.513, unchanged from
  the un-reranked fixed-window run. Reranking cannot repair a chunk boundary.
- **2026-09-06-sectionaware-bm25** -- lift_strict@3 0.540, citation F1 0.372.
  The chunking comparison's treatment arm.
- **2026-09-06-fixedwindow-bm25** -- lift_strict@3 0.513, citation F1 0.279.
  The naive baseline. Posts the higher raw recall (0.80 vs 0.68) and is still
  the worse retriever; see the README Results section.

Not yet run
-----------

Configured in `configs/` and unrun because no Ollama server was reachable from
the machine that produced the runs above: dense retrieval with two embedding
models, hybrid fusion, and cross-encoder reranking. They write the same format
and are directly comparable once run.
