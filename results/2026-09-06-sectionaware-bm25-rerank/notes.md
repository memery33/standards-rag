# sectionaware-bm25-rerank

Reranking variable only, on top of sectionaware-bm25. Lexical overlap reranker, NOT a neural cross-encoder.

## Configuration

- chunker: `section_aware` {'max_tokens': 220, 'min_tokens': 60}
- retriever: `bm25` 
- embedder: `none`
- reranker: `lexical_overlap`
- generator: `extractive`
- corpus: 1582 words, 18 sections, 14 chunks

## Retrieval

| k | recall_strict | recall_loose | MRR | random_loose | beats random |
|---|---|---|---|---|---|
| 1 | 0.56 | 0.92 | 0.92 | 0.1314 | yes |
| 3 | 0.72 | 1.0 | 0.96 | 0.3511 | yes |
| 5 | 0.84 | 1.0 | 0.96 | 0.5238 | yes |
| 10 | 0.96 | 1.0 | 0.96 | 0.8228 | yes |

| k | lift_strict (recall - random) | lift_loose |
|---|---|---|
| 1 | 0.5178 | 0.7886 |
| 3 | 0.5803 | 0.6489 |
| 5 | 0.5822 | 0.4762 |
| 10 | 0.3314 | 0.1772 |

Mean heading paths per retrieved chunk: 1.276. A higher value means each chunk covers more of the document, which inflates loose recall without improving citation precision.

## Answer quality

- citation F1: 0.372 (precision 0.3267, recall 0.5)
- citation exact match: 0.12
- groundedness: 1.0 — Near-trivially high for the extractive generator, which lifts sentences verbatim from the retrieved context. Discriminates between configurations only on the LLM generation path.
- modality fidelity: 1.0 over 9 answers using a normative verb

## Refusal

- threshold: 0.7
- correct refusals: 9/10
- wrong refusals on answerable questions: 9/25
- balanced accuracy: 0.77
- fabricated citations on out-of-corpus questions: 1

The full threshold sweep is in `metrics.json` under `refusal_sweep`.

## Cost

- mean latency: 0.00136s, p95 0.00191s
- mean tokens per query: 624.6

## Questions failing recall_strict@3 (7)

- `q009` [conditional] Are workers ever permitted to cross active equipment paths?
  gold: ITCP Fundamentals > Operation templates (example diagrams in the source) > Dirt spreading (limited backing distance); ITCP Fundamentals > The 8-step development process
- `q014` [synthesis] Step 8 calls for a communication plan. What must that plan contain?
  gold: ITCP Fundamentals > Communication plan elements; ITCP Fundamentals > The 8-step development process
- `q015` [synthesis] Where does the document rely on MUTCD, and for what?
  gold: ITCP Fundamentals > Other considerations > Worker visibility; ITCP Fundamentals > The three components of every ITCP
- `q016` [synthesis] What does the document say about reducing backing maneuvers?
  gold: ITCP Fundamentals > Operation templates (example diagrams in the source) > Asphalt paving; ITCP Fundamentals > Operation templates (example diagrams in the source) > Dirt spreading (limited backing distance); ITCP Fundamentals > What an ITCP is
- `q018` [synthesis] How does the document connect vehicle queuing to worker safety?
  gold: ITCP Fundamentals > Operation templates (example diagrams in the source) > Asphalt paving; ITCP Fundamentals > Operation templates (example diagrams in the source) > Dirt spreading (limited backing distance); ITCP Fundamentals > The 8-step development process
- `q019` [synthesis] What has to happen before a shift starts, and who is responsible?
  gold: ITCP Fundamentals > Communication plan elements; ITCP Fundamentals > Daily use; ITCP Fundamentals > Personnel responsibility matrix (summarized)
- `q021` [synthesis] Who is responsible for keeping the ITCP current as conditions change?
  gold: ITCP Fundamentals > Daily use; ITCP Fundamentals > Personnel responsibility matrix (summarized); ITCP Fundamentals > The 8-step development process

## Fabricated citations (should be empty)

- `r005` cited ['ITCP Fundamentals > What an ITCP is', 'ITCP Fundamentals > Operation templates (example diagrams in the source) > Dirt spreading (limited backing distance)', 'ITCP Fundamentals > Other considerations > Worker visibility']
