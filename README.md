# Socratic Core

An offline Socratic tutor: a two-tier question bank, a layered error
classifier and a hint pipeline that runs Qwen3-4B on the Snapdragon X Elite
NPU (with CPU and mock backends for development).

## Run the CLI

```
python run_session.py                  # mock client, first question
python run_session.py --client local   # real model on CPU
```

## Run the UI

```
pip install streamlit
streamlit run app.py
```

The sidebar switches backend (mock or local CPU), filters the eight question
clusters and resets the session. "🔍 Pipeline internals" shows which
classification layer resolved the last turn and how long the model took.

## Tests

```
python -m pytest tests/
```

Expected counts: WSL (socvenv, llama-cpp installed) runs every test. Windows
(no llama-cpp) reports one extra skip: `LocalClientTests` is skipped as a
whole class, which holds 2 tests, so Windows shows 2 fewer passed and 1
skipped.

## Roadmap

- LlamaRAMCache deferred; best-case 3.7s on retries only; not the dominant cost.
