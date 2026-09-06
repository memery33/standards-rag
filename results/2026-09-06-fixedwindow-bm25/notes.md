# fixedwindow-bm25

Baseline. Naive sliding window, no heading awareness. Expected to lose on strict recall and citation precision because its chunks straddle sections.

## Configuration

- chunker: `fixed_window` {'size': 180, 'overlap': 40}
- retriever: `bm25` 
- embedder: `none`
- reranker: `none`
- generator: `extractive`
- corpus: 1582 words, 18 sections, 12 chunks

## Retrieval

| k | recall_strict | recall_loose | MRR | random_loose | beats random |
|---|---|---|---|---|---|
| 1 | 0.6 | 0.96 | 0.96 | 0.2433 | yes |
| 3 | 0.8 | 1.0 | 0.98 | 0.5595 | yes |
| 5 | 0.92 | 1.0 | 0.98 | 0.7409 | yes |
| 10 | 1.0 | 1.0 | 0.98 | 0.9606 | yes |

| k | lift_strict (recall - random) | lift_loose |
|---|---|---|
| 1 | 0.518 | 0.7167 |
| 3 | 0.5127 | 0.4405 |
| 5 | 0.4118 | 0.2591 |
| 10 | 0.0755 | 0.0394 |

Mean heading paths per retrieved chunk: 2.812. A higher value means each chunk covers more of the document, which inflates loose recall without improving citation precision.

## Answer quality

- citation F1: 0.2787 (precision 0.26, recall 0.34)
- citation exact match: 0.08
- groundedness: 1.0 — Near-trivially high for the extractive generator, which lifts sentences verbatim from the retrieved context. Discriminates between configurations only on the LLM generation path.
- modality fidelity: 1.0 over 8 answers using a normative verb

## Refusal

- threshold: 0.7
- correct refusals: 9/10
- wrong refusals on answerable questions: 9/25
- balanced accuracy: 0.77
- fabricated citations on out-of-corpus questions: 1

The full threshold sweep is in `metrics.json` under `refusal_sweep`.

## Cost

- mean latency: 0.00101s, p95 0.0017s
- mean tokens per query: 906.7

## Questions failing recall_strict@3 (5)

- `q009` [conditional] Are workers ever permitted to cross active equipment paths?
  gold: ITCP Fundamentals > Operation templates (example diagrams in the source) > Dirt spreading (limited backing distance); ITCP Fundamentals > The 8-step development process
- `q014` [synthesis] Step 8 calls for a communication plan. What must that plan contain?
  gold: ITCP Fundamentals > Communication plan elements; ITCP Fundamentals > The 8-step development process
- `q015` [synthesis] Where does the document rely on MUTCD, and for what?
  gold: ITCP Fundamentals > Other considerations > Worker visibility; ITCP Fundamentals > The three components of every ITCP
- `q018` [synthesis] How does the document connect vehicle queuing to worker safety?
  gold: ITCP Fundamentals > Operation templates (example diagrams in the source) > Asphalt paving; ITCP Fundamentals > Operation templates (example diagrams in the source) > Dirt spreading (limited backing distance); ITCP Fundamentals > The 8-step development process
- `q019` [synthesis] What has to happen before a shift starts, and who is responsible?
  gold: ITCP Fundamentals > Communication plan elements; ITCP Fundamentals > Daily use; ITCP Fundamentals > Personnel responsibility matrix (summarized)

## Fabricated citations (should be empty)

- `r005` cited ['ITCP Fundamentals', 'ITCP Fundamentals > Why ITCPs exist', 'ITCP Fundamentals > Other considerations > Worker visibility']
