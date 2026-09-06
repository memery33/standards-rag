Golden question set
===================

The evaluation set. Committed on purpose: every number in the README and in
results/ is reproducible only if the questions that produced it are visible.

File: questions.jsonl - one JSON object per line.

Fields per record
-----------------

- id            stable identifier, never reused after a question is retired
- question      the question as a user would actually ask it
- answer_spans  the exact text from the source that supports the answer
- source_ids    the chunk or clause identifiers that must be retrieved
- answer_type   lookup, synthesis, comparison, or conditional
- difficulty    easy, medium, or hard
- notes         why this question is here and what it is testing

Schema additions
----------------

- tests         a list of tags orthogonal to answer_type, so a failure mode can
                be filtered on its own. cross_reference is not a flavour of
                synthesis; a question can be a table lookup and a conditional at
                once. Current tags: lookup, synthesis, comparison, conditional,
                cross_reference, multi_hop, modality, table, numeric,
                lexical_distractor.

refusals.jsonl records refusal_category (adjacent_domain, false_premise,
over_specific_numeric, out_of_jurisdiction) and distractor_paths: the sections a
retriever is expected to surface anyway. A refusal only proves something if the
retriever had something plausible to fabricate from.

Composition target
------------------

25 questions. Not 30 to 50.

The corpus is 1,582 words across 18 sections, which chunk to 14 (section-aware)
or 12 (fixed-window) retrievable units. Forty questions against that is one per
forty words and produces rephrasings rather than coverage. 25 gives 19 distinct
retrieval targets across 14 of the 18 sections; the four never used as gold are
the title block, two container headings, and the trailing product-commentary
section, which is not standards content and serves as a distractor.

The binding constraint is not question count but corpus size relative to k. With
14 chunks, random retrieval scores loose recall of 0.071 at k=1, 0.214 at k=3,
0.357 at k=5 and 0.714 at k=10. k=10 is therefore uninformative on this corpus:
a system reporting recall_loose@10 of 0.75 has barely beaten chance. The harness
computes and prints that baseline beside every metric so the comparison is
forced rather than optional. k=1 and k=3 carry the real signal here.

Weighted toward the cases that break naive retrieval:

- Lookup - a single clause answers it. The floor. If these fail, nothing else matters.
- Synthesis - the answer requires two or more separated clauses.
- Comparison - the answer depends on a distinction the document draws.
- Conditional - the answer changes based on a qualifier such as shall vs should,
  or a condition that applies only in a stated situation.
  - Cross-reference - the clause defers to another clause by number.

  Refusal set
  -----------

  A separate file, refusals.jsonl, holds questions that are plausible for this
  domain but are NOT answerable from the corpus. The correct behaviour is a
  refusal with no fabricated citation. A system that never refuses is not safe,
  it is just confident.

  Authoring rule
  --------------

  Write the questions before tuning retrieval. A golden set written after the
  fact tends to describe what the system already does well.
  
