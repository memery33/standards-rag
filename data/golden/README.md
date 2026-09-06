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

Composition target
------------------

30 to 50 questions total, weighted toward the cases that break naive retrieval:

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
  
