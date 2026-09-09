# standards-rag

Retrieval-augmented question answering over a technical standards document,
built so that the **evaluation harness is the deliverable** and the chat
interface is not.

Most retrieval projects stop at "it answers questions." The interesting
engineering problem is knowing whether the answers are *right*, and being able
to show your work when someone asks.

**Status: BM25 arms scored and committed.** Ingestion, both chunkers, the
golden set, retrieval, generation and the eval harness are in place, with four
scored runs in `results/`. The dense, hybrid, reranking and embedding-model
arms are implemented and configured but **unrun** — they need a local Ollama
server. Nothing in this README claims a result that isn't in `results/`, and
every number below links to the run that produced it.

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

## Experiments

One variable at a time, each a separate committed run.

**Run and committed** (BM25 is pure Python — no models, no network, so these
reproduce anywhere):

- Chunking — fixed window versus section-aware
- Reranking — with and without, on both chunkers

**Implemented, configured, unrun** (each needs a local Ollama server; configs
are in `configs/`, and switching backend is a config change and nothing else):

- Retrieval — dense vector and hybrid, against the BM25 baseline
- Embedding model — `nomic-embed-text` versus `mxbai-embed-large`
- Reranking — a real cross-encoder, versus the lexical reranker used above

## Results

Every number links to the run directory that produced it. All four runs use
BM25, the extractive generator, and the same golden set; the dense, hybrid,
reranked-with-a-cross-encoder and second-embedding-model arms are implemented
and configured but **unrun** — they need a local Ollama server, which the
machine that produced these runs could not reach. Their configs are in
`configs/` and produce the same `results/` format.

| run | chunker | retriever | rerank | R@1 strict | R@3 strict | R@3 loose | lift@3 strict | MRR@3 | cite F1 | paths/chunk |
|---|---|---|---|---|---|---|---|---|---|---|
| [2026-09-06-fixedwindow-bm25](results/2026-09-06-fixedwindow-bm25/) | fixed_window | bm25 | none | 0.6 | 0.8 | 1.0 | 0.5127 | 0.98 | 0.2787 | 2.812 |
| [2026-09-06-fixedwindow-bm25-rerank](results/2026-09-06-fixedwindow-bm25-rerank/) | fixed_window | bm25 | lexical_overlap | 0.64 | 0.8 | 1.0 | 0.5127 | 1.0 | 0.2787 | 2.812 |
| [2026-09-06-sectionaware-bm25](results/2026-09-06-sectionaware-bm25/) | section_aware | bm25 | none | 0.56 | 0.68 | 1.0 | 0.5403 | 0.96 | 0.372 | 1.276 |
| [2026-09-06-sectionaware-bm25-rerank](results/2026-09-06-sectionaware-bm25-rerank/) | section_aware | bm25 | lexical_overlap | 0.56 | 0.72 | 1.0 | 0.5803 | 0.96 | 0.372 | 1.276 |

### The headline metric is misleading, and that is the point

Read the first two rows and the naive fixed-window chunker wins: `recall_strict@3`
of **0.80** against section-aware's **0.68**. That reading is wrong, and three
columns in the same table say why.

Fixed-window chunks straddle **2.81 heading paths each**; section-aware chunks
straddle **1.28**. A wider chunk is credited with retrieving every section it
overlaps, so a fixed window gets recall for sections it merely brushed. Random
retrieval demonstrates the same effect with no intelligence at all: it scores
`recall_strict@3` of **0.2873** on the fixed-window corpus versus **0.1397** on
the section-aware one. Subtracting the baseline:

| chunker | recall_strict@3 | random | **lift** | citation precision | context tokens@3 |
|---|---|---|---|---|---|
| fixed_window | 0.80 | 0.2873 | **0.513** | 0.260 | 534.5 |
| section_aware | 0.68 | 0.1397 | **0.540** | 0.327 | 378.9 |
| section_aware + rerank | 0.72 | 0.1397 | **0.580** | 0.327 | 394.8 |

Section-aware wins on lift, wins on citation precision by 26%, and does it while
retrieving **41% less text**. The fixed window was buying recall with context
width, which is exactly the trade a raw recall number hides. This is why
`lift`, `mean_paths_per_chunk` and `mean_context_tokens@k` are in every run.

### Where retrieval actually fails

Aggregates hide this; `per_question.jsonl` does not. From
[sectionaware-bm25](results/2026-09-06-sectionaware-bm25/), `recall_strict@3` by tag:

| tag | n | strict@3 | loose@3 |
|---|---|---|---|
| lookup | 5 | 1.00 | 1.00 |
| comparison | 3 | 1.00 | 1.00 |
| conditional | 8 | 0.88 | 1.00 |
| modality | 5 | 0.80 | 1.00 |
| table | 7 | 0.57 | 1.00 |
| cross_reference | 4 | 0.50 | 1.00 |
| multi_hop | 4 | 0.50 | 1.00 |
| **synthesis** | **7** | **0.14** | **1.00** |

`loose@3` is 1.00 across every tag. On its own it would say the system is
perfect. Strict recall says one synthesis question in seven retrieves all its
gold sections in the top 3. That gap — 1.00 versus 0.14 on the same questions —
is the single strongest argument for reporting both, and it is why the golden
set was weighted toward synthesis rather than lookup.

## What did not work

Kept deliberately. A retrieval system that only reports its wins is not evidence
of anything.

### Lexical refusal scoring is weak, and the refusal set proves it

The extractive generator decides to refuse by IDF-weighted term coverage of the
question against the retrieved context. At its best operating point (threshold
0.70, selected by sweeping this same eval set — a caveat that inflates it) it
catches **9 of 10** out-of-corpus questions but **wrongly refuses 9 of 25
answerable ones**. Balanced accuracy 0.77. A 36% over-refusal rate is not
shippable.

The first attempt was plain term coverage and was worse: the OSHA question
scored **0.55**, indistinguishable from a real question, because "internal",
"traffic", "control" and "plans" are all common corpus words. IDF weighting
helped and did not fix it. The distributions still overlap — the highest-scoring
refusal (0.84) outranks nine answerable questions.

This is the refusal set doing its job. Every entry was written to be lexically
adjacent to the corpus, so a lexical signal cannot separate them by
construction. `r005` ("maximum allowable backing speed") is the clean
demonstration: every one of its terms appears in the corpus and only the *fact*
is absent. No term-overlap method can ever refuse it. The full sweep is
committed in each run's `metrics.json` under `refusal_sweep` rather than a
single flattering number.

This is the arm most likely to improve under LLM generation, which is testable
the moment the Ollama runs happen.

### Reranking does nothing for the naive chunker

`lexical_overlap` reranking lifts section-aware `recall_strict@3` from 0.68 to
0.72 and `lift@3` from 0.540 to 0.580. On fixed-window chunks it changes strict
recall not at all (0.80 → 0.80); only `MRR@3` moves, 0.98 → 1.00. Reranking
reorders candidates but cannot repair a chunk boundary that put half a table in
one chunk and half in another. Chunking is upstream of reranking and no amount
of reordering compensates for it.

### Groundedness is close to meaningless here

The extractive generator composes answers from sentences lifted verbatim from
the retrieved context, so groundedness is **1.00** by construction. It is
reported because it will matter on the LLM path, where fabrication is possible,
but on these runs it discriminates nothing. Citation accuracy is the metric
that separates the configurations — F1 0.372 against 0.279.

Modality fidelity has the same weakness for the same reason: 1.00 over the 8
answers that use a normative verb. It is a real check against an LLM that
rewrites "shall" as "should"; against an extractive generator it cannot fail.

### Citation exact-match is 0.12 and that is the real ceiling

Citation F1 of 0.372 sounds moderate. Exact match — the answer citing precisely
the gold sections, no more and no fewer — is **0.12**. Three questions in
twenty-five. Precision is 0.327, meaning roughly two of every three cited
sections do not support the answer. For a standards document where a wrong
citation sends a reader to text that does not say what was claimed, this is the
number that matters most, and it is the worst number in the repo.

### k=10 is not a measurement on this corpus

14 chunks. Retrieving 10 of them is 71% of the document, and random retrieval
scores `recall_loose@10` of **0.82**. Every k=10 figure is reported with that
baseline beside it. The honest reading is that k=1 and k=3 carry signal here and
k=10 does not; growing the corpus roughly tenfold is the only thing that would
change that.

### The corpus contains its own best distractor, and BM25 survived it

"Spotter" is a product name and "spotter" is a personnel role. Nineteen
occurrences, most of them the product, concentrated in a trailing section that
is not standards content. `q005` was written expecting BM25 to be dragged there.
It was not — the personnel matrix ranks first. A predicted failure that did not
happen, recorded because a golden set that only contains predictions that came
true has been curated after the fact.

## Layout

```
data/raw/       the corpus
data/golden/    evaluation question set and its schema
configs/        one JSON per experiment arm, run and unrun
results/        versioned eval runs, committed (never gitignored)
src/            ingestion, retrieval, generation, eval
tests/          unit tests for chunking and scoring
```

## Quick start

```bash
cp .env.example .env
make install
make test                 # 78 tests: chunking, scoring, golden set, retrieval
make ingest               # parse and chunk, printing the chunk inventory
make eval                 # score the default config against the golden set
make eval-all             # every arm that needs no model backend
make verify               # re-derive every committed run and check it matches
make compare              # regenerate the results table above

make eval CONFIG=configs/sectionaware-dense-nomic.json   # an Ollama arm
make serve Q="What training do spotters need?"           # one question, by hand
```

### Reproducing the numbers

`make verify` rebuilds every run in `results/` from its own `config.json` and
compares the fresh metrics against the committed ones. It is the central claim
made executable rather than asserted: if the code drifts from the results, the
check fails. CI runs it on every push, alongside the tests, the lint, and a
validation of the golden set against the corpus. Latency is deliberately
excluded from the comparison — it is machine-dependent, and a check that fails
on a different laptop teaches readers to ignore it.

## License

MIT — see [LICENSE](LICENSE). The corpus is my own writing, summarizing
public-domain federal guidance.
