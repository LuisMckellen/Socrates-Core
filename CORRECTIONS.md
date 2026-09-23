# Corrections Log

## 1

Bank documented as a lookup stage; code has always used it as LLM few-shot
context. Doc drift corrected; bank is a data source, not a pipeline stage.

## 2

key_terms originally fired wording_error on zero-match. Replaced with Case
A/B/C routing.

## 3

wording_error removed from the bank schema. Runtime decision by key_terms
and the LLM, not a bank-authorable field.

## 4

Escalation fallback chain degenerated with one socratic per cluster.
Resolved with L1/L2 level field.

## 5

LLM classifier returned only wording_error or logic_error, and verdict was
hardcoded to "wrong" regardless of label. No Case C answer could score
correct. Extended to 3-way with translation table.

## 6

Off-topic gate fired before key_terms and LLM, killing answers with zero
question vocabulary. Split into disengagement-gated fast path + semantic
fallback.

## 7

Global seed examples left in prompt long after the bank became the LLM's
context source. Removed. Prompt dropped from ~1309 to 583 tokens (measured
by tokenizer) with no label change on trace 3 or trace 4b.

## 8

Groq fallback routes student answers off-device. Disabled by default.
Enabled via sidebar for demo latency only. Production target remains
on-device NPU inference.

## 9

Partial verdicts previously bypassed the max-attempts reveal because Case B
returned early. LLM-partials now go through the shared path and can trigger
the reveal. Accepted for consistency.
