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

Run log
-------

Newest first. One line per run: the headline number and the one-sentence
reason the run exists.
