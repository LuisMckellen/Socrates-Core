# Corrections Log

Each entry states what changed and why, with the code that now reflects it.
Measured numbers come from BASELINE.md.

## 1

Docs described the bank as a lookup stage, but the code had always used it as
LLM few-shot context. Corrected: the bank is a data source, not a pipeline
stage (`socratic_core/question_bank.py:13-14`,
`socratic_core/classifier_llm.py:48-58`).

## 2

key_terms used to return wording_error when no key term matched. Replaced with
Case A and Case C routing (`socratic_core/state_machine.py:22-29`): every key
term present means Case A, which goes to verification; any key term missing
means Case C, which goes to the LLM classifier. Case B (a lexical partial
match) has been removed; partial is now an LLM verdict
(`socratic_core/state_machine.py:28-29`).

## 3

wording_error removed from the bank schema. The bank may author only
logic_error misconceptions (`socratic_core/question_bank.py:38,260-263`).
Whether an answer is a wording_error is decided at runtime by the LLM.

## 4

The escalation fallback chain broke down when a cluster had only one socratic
question. Fixed with the level field: socratic questions are level 1 or 2
(`socratic_core/question_bank.py:42,297-307`), and escalation picks by level
(`socratic_core/escalation.py:36-46`).

## 5

The LLM classifier used to return only wording_error or logic_error, and the
verdict was hardcoded to "wrong", so no Case C answer could ever score correct.
There are now two separate label spaces. The LLM emits correct, partial,
wording_error or logic_error (`socratic_core/classifier_llm.py:100`). The state
machine maps those to verdicts: correct to correct, partial to partial, and
wording_error or logic_error to wrong (`socratic_core/state_machine.py:135-140`).
The behavioural layer adds the fourth verdict, low_effort
(`socratic_core/state_machine.py:447-448`).

## 6

The off-topic gate ran before key_terms and the LLM, so it rejected genuine
answers that happened to share no vocabulary with the question. It is now
split in two. Off-topic plus a disengagement token is low_effort; off-topic
alone falls through to key_terms and then the LLM
(`socratic_core/classifier_behavioral.py:18-24,105-112`).

## 7

The prompt still carried global seed examples after the bank had become the
LLM's context source. They have been removed: "There are no global examples"
(`socratic_core/classifier_llm.py:55`). An earlier prompt-size figure for this
change was withdrawn because no measurement of it exists in the repo.

## 8

Groq routes student answers off-device, so it is off by default: the UI
starts on Mock (`ui/state.py:278`). The sidebar offers three backends: Mock,
Local CPU and Groq (`ui/state.py:32-36`). Groq also appears as an opt-in
classifier fallback behind Local CPU (`ui/sidebar.py:61-71`,
`ui/state.py:105-113`). The NPU client is not among the UI backends
(`ui/state.py:36,88`); it is reachable only from the CLI
(`run_session.py:44,53`).

## 9

Partial verdicts used to skip the max-attempts reveal. Partial answers now
count toward max_attempts like any other answer and can trigger the reveal
(`socratic_core/state_machine.py:497-498`).

## 10

Rule 3c removed. It rejected 6/6 real hints on Trace 2, forcing bank
fallback. Verified verdict-neutral across 18 post-fix trace runs.

The only remaining guard against hints that offer the correct mechanism as
one side of an either/or is the prompt instruction "Do not offer
alternatives. Do not present two options"
(`socratic_core/hint_pipeline.py:68-69`).

## 11

Removed the duplicate `attempt` key from answer events. It was written at
four call sites, `socratic_core/state_machine.py:367,404,479,684` as of
commit `508b586`. `attempt_number` carries the same value
(`socratic_core/state_machine.py:809`). This fixed two tests that had been
failing (BASELINE.md).

## 12

Removed the alias `classify = classify_behavioural`
(`socratic_core/classifier_behavioral.py`, formerly line 121). Nothing
called it.

## 13

Withdrawn documentation figures:

- "210 tests passing" was a `pytest --collect-only` count, not a pass count.
  Tests now pass as follows: 208 on Windows, where the
  `tests/test_local_client.py` module skips itself because llama_cpp is not
  installed, and 210 on WSL (BASELINE.md).
- A best-case retry latency for LlamaRAMCache was removed. No measurement of
  it exists in the repo.
