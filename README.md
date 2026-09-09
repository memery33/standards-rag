# standards-rag

Retrieval-augmented question answering over a technical standards document,
built so that the **evaluation harness is the deliverable** and the chat
interface is not.

Most retrieval projects stop at "it answers questions." The interesting
engineering problem is knowing whether the answers are *right*, and being able
to show your work when someone asks.

**Status: in progress.** The scaffold, corpus and evaluation design are in
place. Retrieval and scored runs are being built now. Nothing in this README
claims a result that isn't in `results/`.

## The corpus

`data/raw/itcp-fundamentals.md` — a reference on Internal Traffic Control Plans
(ITCPs), the site-specific plans that coordinate equipment and workers on foot
inside a roadway work zone. It is my own paraphrased summary of public Federal
Highway Administration and American Road and Transportation Builders
Association guidance, written originally as reference material for
[Spotter](https://www.planwithspotter.com), a work-zone safety product I build.

It is small, roughly 1,600 words, and that shapes the whole design. What makes
it a real test is not size:

- **Conditional language.** "Shall", "should" and "may" carry different weight.
  An answer that flattens them is wrong even when it reads correctly.
- **Structure without clause numbers.** Markdown headings, one numbered
  eight-step process, a symbology table and a personnel responsibility matrix.
  There is no `4.2.1` to anchor a citation to, so ground truth is the
  **heading path** (`Operation templates > Asphalt milling`) instead.
- **Tables bound to prose.** Splitting one from its section loses the meaning.
- **Consequences.** These are safety documents. A confident wrong citation is
  the failure mode that matters.

## Design decisions

**Heading path is the ground truth, not the chunk ID.** The golden set records
heading paths. Chunk IDs encode which strategy produced them. That is what
makes the chunking comparison honest: fixed-window and section-aware chunkers
cut in completely different places, and both are still scored against the same
target.

**Recall is reported strict and loose.** For questions needing several
sections, strict means every gold section is in the top k; loose means at least
one is. Reporting only loose flatters multi-section performance badly.
Mean Reciprocal Rank uses the rank of the first gold section.

**Refusal is a measured behavior.** A separate set of questions that sound
plausible for this domain but are not answerable from the corpus. A system that
never refuses is not safe, it is confident.

## What gets measured

| Dimension | Metric |
|---|---|
| Retrieval | recall@k (strict and loose) and Mean Reciprocal Rank at k = 1, 3, 5, 10 |
| Answer quality | groundedness / faithfulness |
| Citation | citation accuracy against the golden set |
| Safety | refusal rate on out-of-corpus questions |
| Cost | tokens and latency per query, logged per run |

## Planned experiments

One variable at a time, each a separate committed run:

- Chunking — fixed window versus section-aware
- Retrieval — dense vector versus BM25 versus hybrid
- Reranking — with and without
- Embedding model — at least two compared

## Results

Nothing yet. This section gets filled from `results/` as runs land, including
the ones that made things worse.

## What did not work

Reserved, and it will be used. A retrieval system that only reports its wins is
not evidence of anything.

## Layout

```
data/raw/       the corpus
data/golden/    evaluation question set and its schema
results/        versioned eval runs, committed (never gitignored)
src/            ingestion, retrieval, generation, eval
tests/          unit tests for chunking and scoring
```

## Quick start

```bash
cp .env.example .env
make install
make ingest
make eval
```

## License

MIT — see [LICENSE](LICENSE). The corpus is my own writing, summarizing
public-domain federal guidance.
