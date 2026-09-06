# standards-rag

A retrieval-augmented question answering system over a technical standards
document, built to be **measured rather than demoed**.

Most RAG projects stop at "it answers questions." This one treats retrieval
quality as an engineering problem: every configuration change is scored against
a fixed golden question set, results are versioned in `results/`, and the
failures are documented alongside the wins.

## Why this exists

The corpus is an Internal Traffic Control Plan (ITCP) standards reference I
wrote myself — a real document with the properties that make RAG hard:
numbered clauses, nested cross-references, tables, and conditional language
where "shall" and "should" carry different weight. Getting a citation wrong in
a document like this is not a cosmetic failure.

## What is measured

| Dimension | Metric |
|---|---|
| Retrieval | recall@k and MRR at k = 1, 3, 5, 10 |
| Answer quality | groundedness / faithfulness score |
| Citation | citation accuracy against the golden set |
| Safety | refusal rate on out-of-corpus questions |
| Cost | tokens and latency per query, logged per run |

The golden question set lives in `data/golden/` and is committed. So are the
eval runs in `results/`, so any claim in this README can be checked against
the run that produced it.

## Experiments

Each of these is a separate scored run, not a design decision made by vibes:

- Chunking strategy — fixed window vs. clause-aware splitting
- Retrieval — dense vector vs. BM25 vs. hybrid
- Reranking — with and without a cross-encoder rerank stage
- Embedding model — comparison across at least two models

## Results

_To be filled in from `results/` once the first runs complete. Include what
regressed, not only what improved._

## Quick start

```bash
cp .env.example .env      # add your API keys
make install
make ingest               # build the index from data/raw/
make eval                 # score against the golden set
make serve                # optional: local query interface
```

## Layout

```
data/raw/      source documents (not committed if licensing is unclear)
data/golden/   evaluation question set, committed
src/ingest/    parsing and chunking
src/retrieval/ index construction and search
src/generation/ prompt assembly and answer synthesis
src/eval/      scoring harness
results/       versioned eval runs, committed
```

## What did not work

_Kept deliberately. A retrieval system that only reports its wins is not
evidence of anything._

## License

MIT — see [LICENSE](LICENSE).
