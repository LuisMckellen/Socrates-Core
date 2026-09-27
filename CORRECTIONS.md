# Corrections

Things a first-time reader is likely to get wrong about this code, and what
the code actually does.

### The question bank as a pipeline stage

Common assumption: student answers are looked up against the bank's
misconceptions before the model runs.
Actual: nothing matches an answer against the misconceptions. They are only
few-shot context in the LLM classifier's prompt
(`socratic_core/question_bank.py:13-14`, `socratic_core/classifier_llm.py:48-58`,
`socratic_core/state_machine.py:33-35`).

### Case A / B / C

Common assumption: key-term matching has three cases, including a lexical
partial match.
Actual: there are two. Case A (every key term present) goes to a YES/NO
verification call; Case C (any key term missing) goes to the LLM classifier.
Whether an answer is partial is decided by the LLM
(`socratic_core/state_machine.py:22-29,547-565`).

### Label spaces

Common assumption: one classifier label set doubles as the verdict.
Actual: the LLM emits 4 labels, correct, partial, wording_error and
logic_error (`socratic_core/classifier_llm.py:100`). They map to 4 verdicts:
correct, partial, wrong (for wording_error and logic_error)
(`socratic_core/state_machine.py:135-140`), and low_effort, which only the
behavioural layer produces (`socratic_core/state_machine.py:447-448`). A
partial verdict is logged with `error_type="logic_error"`
(`socratic_core/state_machine.py:142-144`).

### wording_error in the bank

Common assumption: the bank authors wording_error misconceptions.
Actual: bank misconceptions may only be logic_error. wording_error is decided
at runtime by the LLM (`socratic_core/question_bank.py:38,260-263`).

### Classifier confidence

Common assumption: the classifier reports a confidence score that gates its
label.
Actual: the reply format is LABEL, MATCHED and REASONING only
(`socratic_core/classifier_llm.py:126-129`). correct and partial are accepted
only from an explicit LABEL line (`socratic_core/classifier_llm.py:42-46`).

### Classifier failures

Common assumption: an unreadable model reply could let an answer through.
Actual: every classifier failure (client error, unknown label, no label)
lands on logic_error and can never produce correct
(`socratic_core/classifier_llm.py:319-343`,
`socratic_core/state_machine.py:614-627`). Case A verification does the
opposite: it fails open, because the answer already matched every key term
(`socratic_core/classifier_llm.py:383-402`,
`socratic_core/state_machine.py:567-577`).

### MATCHED and the label

Common assumption: the matched misconception decides the label.
Actual: `matched_bank_id` never affects the label
(`socratic_core/classifier_llm.py:68-71`). It selects the targeted hint
prompt when it names one of the question's own misconceptions
(`socratic_core/hint_pipeline.py:114-136`).

### Few-shot examples

Common assumption: the classifier prompt carries global example answers.
Actual: every example comes from the current question's bank entry: its
everyday-words correct answer, one partial row built from its first key
term, and its own misconceptions (`socratic_core/classifier_llm.py:48-58`).

### Who sees the correct answer

Common assumption: the LLM never sees the correct answer.
Actual: the classifier prompt and the Case A verify prompt include it
(`socratic_core/classifier_llm.py:230-231,368`); their output is never shown
to the student. The hint generator never receives `correct_answer`,
`accepted_variants` or `answer_explanation`; on the targeted path it receives
`natural_correct_example` (`socratic_core/hint_pipeline.py:15-20,129-134`).

### Off-topic answers

Common assumption: an answer that shares no words with the question is
scored low_effort.
Actual: off-topic alone goes on to key terms and the LLM. Only off-topic
together with a disengagement token is low_effort
(`socratic_core/classifier_behavioral.py:18-24,105-112`).

### Meme and slang words

Common assumption: slang in an answer lowers its grade.
Actual: disengagement tokens are logged on every turn and affect the verdict
only in the off-topic case above (`socratic_core/disengagement.py:6-12`). The
classifier is told to ignore noise and judge the concept
(`socratic_core/noise.py:15-19`, `socratic_core/state_machine.py:594`).

### Exact match

Common assumption: a correct answer is found by substring match.
Actual: a correct answer must equal the answer or an accepted variant after
normalisation and removal of leading filler words; "cell membrane" does not
match "cell" (`socratic_core/question_bank.py:171-180`).

### Filter questions

Common assumption: every question runs the classification pipeline.
Actual: filter questions are exact match only. No behavioural rules, no LLM
and no hint (`socratic_core/state_machine.py:355-393`).

### Which questions a session walks

Common assumption: a session walks every question in the bank.
Actual: a session walks the filter questions; Socratic questions are
inserted by escalation after a missed filter
(`socratic_core/state_machine.py:297-302,378-389`, `ui/state.py:11-14`). The
CLI walks only the first bank question unless `--question-id` is given
(`run_session.py:57`).

### Escalation

Common assumption: a missed filter always escalates to the same follow-up.
Actual: it escalates only while the cluster's mastery is below 0.40, and
picks level 1 or 2 by mastery (level 2 at 0.60 and above)
(`socratic_core/mastery.py:23-26,52-58`, `socratic_core/escalation.py:30-46`).
Socratic questions must declare a level of 1 or 2
(`socratic_core/question_bank.py:42,297-307`).

### Attempt limit

Common assumption: partial answers do not use up attempts.
Actual: wrong and partial answers both count toward the 3-attempt limit; the
third reveals the answer from the bank, never from the model
(`socratic_core/state_machine.py:131,497-502,732-739`).

### Mastery and low_effort

Common assumption: a low-effort answer costs mastery.
Actual: wrong and partial Socratic answers cost mastery; low_effort does not
(`socratic_core/state_machine.py:491-495`).

### Hint fallback order

Common assumption: a rejected hint is retried once, then the bank hint is
used.
Actual: for a partial answer a hint must also name a missing key term, and a
template naming it comes before the bank hint
(`socratic_core/hint_pipeline.py:263-271`). A client error or an empty reply
skips the retry (`socratic_core/hint_pipeline.py:253-258`).

### Hint leak checking

Common assumption: the hint checker catches any hint that gives the answer
away.
Actual: it is string matching over the answer, the accepted variants and the
everyday-words example (`socratic_core/hint_pipeline.py:195-225`). It does
not reject hints that name key terms, and it does not catch a synonym-for-
synonym paraphrase (`socratic_core/hint_pipeline.py:22-23`). An either/or
hint is guarded only by the prompt instruction "Do not offer alternatives. Do
not present two options" (`socratic_core/hint_pipeline.py:68-69`).

### CPU as a fallback

Common assumption: the Local CPU backend takes over when the NPU fails.
Actual: nothing falls back to either. NPU and Local CPU are separate manual
choices: a CLI flag (`run_session.py:44-53`) or a UI radio button
(`ui/state.py:85-102`).

### Which backend runs the demo

Common assumption: the UI demo runs on the NPU.
Actual: the UI offers Mock, Local CPU and Groq, and starts on Mock
(`ui/state.py:32-36,278`). The NPU is reachable only from the CLI
(`ui/state.py:88`, `run_session.py:53`).

### Groq

Common assumption: the cloud backend is on by default.
Actual: Groq sends answers off-device, so it is never the default. It is
reached only by selecting it in the sidebar, or by ticking the classifier
fallback behind Local CPU, which needs a `GROQ_API_KEY`
(`ui/sidebar.py:69-77`, `ui/state.py:105-113`,
`socratic_core/cloud_client.py:56-58`).

### NPU max_tokens

Common assumption: `max_tokens` caps NPU generation.
Actual: `_apply_max_tokens` returns the config unchanged, so the NPU output
length is unbounded (`socratic_core/inference_client.py:410-420`).

### Eval coverage

Common assumption: the eval covers the bank.
Actual: 60 synthetic rows on one question (s_cell_theory_L1), run on the
Local CPU backend (`eval/run_eval.py:26,55,65`,
`eval/socratic_eval_input.jsonl`).

### Answer event fields

Common assumption: answer events carry an `attempt` field.
Actual: the attempt count is `attempt_number`
(`socratic_core/state_machine.py:809`).

### Session flags and telemetry

Common assumption: `bypass_bank_lookup` changes routing, and the telemetry
counters appear in the UI.
Actual: `bypass_bank_lookup` is a no-op kept in the session log
(`socratic_core/state_machine.py:36-37,323-325`). `socratic_core/telemetry.py`
is called only by `tests/test_telemetry.py`.

### Test counts

Common assumption: all tests run everywhere.
Actual: 208 pass on Windows, where `tests/test_local_client.py` skips itself
without llama_cpp, and 210 pass on WSL (RESULTS.md, Tests;
`tests/test_local_client.py:22-25`).
