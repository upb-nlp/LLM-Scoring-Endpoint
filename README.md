# INFLO LLM Scoring Integration

The API for scoring a student response (self-explanation, think-aloud, summary, paraphrasing, or multi-text self-explanation). The model will return a dictionarry with scores across multiple rubrics. For explanations on what those scores mean, please refer to `learning_strategies_scoring/scoring_details`. 

## Repository contents

- `api_flask.py` — Flask server that exposes the public HTTP endpoints. Loads the scoring model and the feedback model once at startup and serves `POST /score/<task>` (rubric scoring) and `POST /feedback/<task>` (rubric scoring + natural-language feedback + optional retry-evolution comment).
- `request_score_server.py` — CLI client that posts a bank of student-response scenarios (paraphrasing, self-explanation, multi-text self-explanation) to `/score/<task>`. Useful for smoke-testing the deployed server or a local run. Supports `--url`, `--only <task>`, `--name <scenario>`, `--timeout`.
- `request_feedback_server.py` — Companion CLI client for `/feedback/<task>`. Includes the same scenarios plus reusable `previous_answer` payloads for retry cases, so you can exercise the evolution-comment behavior. Adds `--retries-only` and `--output <path>` (writes structured JSON results) on top of the score-client flags.
- `requirements.txt` — Python dependencies for the server side (Flask, Flask-Cors, torch, vllm). The CLI clients additionally need `requests`.
- `learning_strategies_scoring/` — The library that powers the server:
    - `api_llm_scoring.py` — Defines the `LLMScoring` class. Loads the fine-tuned scoring model with vLLM, builds task-specific prompts from the JSON rubrics, parses model output back into score dicts, and (when `feedback_model_name` is provided) loads a second instruction-tuned model to generate the student-facing feedback, paraphrase hints, and retry-evolution comments. The HTTP layer is a thin wrapper over `.score()` and `.feedback()`.
    - `examples_score.py` — Runnable Python examples that use `LLMScoring` directly (no HTTP) to score paraphrasing, self-explanation, and multi-text self-explanation responses. Mirrors the local-Python flow documented in this README.
    - `examples_feedback.py` — Same idea for feedback: instantiates `LLMScoring` with a `feedback_model_name`, walks through ~35 scenarios (including retries with `previous_answer`), and writes the full results to `feedback_scenarios_results.json`.
    - `generate_results_html.py` — Reads `feedback_scenarios_results.json` and renders a styled HTML report (`feedback_scenarios_results.html`) grouping the scenarios by task. Handy for eyeballing whether scores, feedback text, and `try_again` flags look right after a model change.
    - `scoring_details/` — JSON rubric definitions, one per task (`paraphrasing_ulpc.json`, `selfexplanation_thinkaloud_full_se.json`, `selfexplanation_thinkaloud_full_ta.json`, `summaries_aloe.json`, `se_improved.json`). Each file describes the task and the criteria/score levels the model is prompted with — this is also where to look for what each score value means.

## API usage

You can use our Flask server to send requests for scoring. The request must come in the form of a POST to `https://chat.readerbench.com/score/<task>`.

```c
curl --location 'https://chat.readerbench.com/score/selfexplanation' \
--header 'Content-Type: application/json' \
--data '{
    "context": "The supporting text that the student has read.",
    "target_sentence": "The sentence from the text that the student must write a self-explanation.",
    "student_response": "The student'\''s self-explanation."
}'
```

```c
curl --location 'https://chat.readerbench.com/score/thinkaloud' \
--header 'Content-Type: application/json' \
--data '{
    "context": "The supporting text that the student has read.",
    "target_sentence": "The sentence from the text that the student must write their thoughts.",
    "student_response": "The student'\''s thoughts."
}'
```

```c
curl --location 'https://chat.readerbench.com/score/summary' \
--header 'Content-Type: application/json' \
--data '{
    "context": "The supporting text that the student has read.",
    "student_response": "The student'\''s summary"
}'
```

```c
curl --location 'https://chat.readerbench.com/score/paraphrasing' \
--header 'Content-Type: application/json' \
--data '{
    "target_sentence": "The support sentence.",
    "student_response": "The student'\''s paraphrasing."
}'
```

```c
curl --location 'https://chat.readerbench.com/score/selfexplanation_multitext' \
--header 'Content-Type: application/json' \
--data '{
    "context": "The supporting multi-text content the student has read (e.g., several sources on a topic).",
    "target_sentence": "The sentence from the text that the student must write a self-explanation.",
    "student_response": "The student'\''s self-explanation."
}'
```

## Feedback API

The feedback endpoint returns the same scores as `/score/<task>` plus a short natural-language feedback message addressed to the student, and a `try_again` flag that is `true` when the response is so weak it warrants a retry. The endpoint lives at `https://chat.readerbench.com/feedback/<task>` and accepts the same JSON body as `/score/<task>`. On a retry, include an optional `previous_answer` field carrying the student's prior `student_response`, the `scores` they received, and (optionally) the `feedback` they were shown — the response will prepend a one-sentence comment on how the new attempt evolved.

Response shape:

```json
{
    "scores": { "...": "..." },
    "feedback": "Short feedback addressed to the student.",
    "try_again": false
}
```

### Scenario 1 — Paraphrasing, good attempt

```c
curl --location 'https://chat.readerbench.com/feedback/paraphrasing' \
--header 'Content-Type: application/json' \
--data '{
    "target_sentence": "One of the most harmful air pollutants is acid rain, a mixture of acid and water that falls to earth.",
    "student_response": "A combination of acid and water that falls upon the ground is a harmful pollutant called acid rain."
}'
```

### Scenario 2 — Paraphrasing, poor attempt then retry

First attempt (will likely come back with `try_again: true`):

```c
curl --location 'https://chat.readerbench.com/feedback/paraphrasing' \
--header 'Content-Type: application/json' \
--data '{
    "target_sentence": "Red blood cells have the vital role of carrying oxygen to all of the cells in the body.",
    "student_response": ",m."
}'
```

Retry — pass the previous attempt back in via `previous_answer`:

```c
curl --location 'https://chat.readerbench.com/feedback/paraphrasing' \
--header 'Content-Type: application/json' \
--data '{
    "target_sentence": "Red blood cells have the vital role of carrying oxygen to all of the cells in the body.",
    "student_response": "Red blood cells are responsible for delivering oxygen to every cell in the body.",
    "previous_answer": {
        "student_response": ",m.",
        "scores": {
            "Garbage Content": "Too much",
            "Irrelevant": "Irrelevant",
            "Paraphrase Quality": "Poor",
            "Writing Quality": "Poor"
        },
        "feedback": "Your response is very short and does not appear to address the target sentence. It may help to read the sentence again and write a complete rephrasing in your own words. Can you try again?"
    }
}'
```

### Scenario 3 — Self-explanation, strong response

```c
curl --location 'https://chat.readerbench.com/feedback/selfexplanation' \
--header 'Content-Type: application/json' \
--data '{
    "context": "Red blood cells have the vital role of carrying oxygen... (full supporting text)",
    "target_sentence": "The shape of the cells causes them to clog blood vessels, preventing oxygen from reaching muscles and other tissues.",
    "student_response": "Blood vessels are naturally shaped to transport the disk-shaped red blood cell; if the cell changes shape it makes sense that it clogs the vessels, because the vessels are already sized for disk-shaped cells."
}'
```

### Scenario 4 — Self-explanation, low-effort response then retry

First attempt:

```c
curl --location 'https://chat.readerbench.com/feedback/selfexplanation' \
--header 'Content-Type: application/json' \
--data '{
    "context": "Red blood cells have the vital role of carrying oxygen... (full supporting text)",
    "target_sentence": "Hemoglobin also contains iron, which gives blood its red color.",
    "student_response": "ok"
}'
```

Retry with `previous_answer`:

```c
curl --location 'https://chat.readerbench.com/feedback/selfexplanation' \
--header 'Content-Type: application/json' \
--data '{
    "context": "Red blood cells have the vital role of carrying oxygen... (full supporting text)",
    "target_sentence": "Hemoglobin also contains iron, which gives blood its red color.",
    "student_response": "The protein hemoglobin has iron in it, and that iron is what makes our blood look red. This connects to the earlier idea that hemoglobin binds oxygen, so the iron must play a role in that binding process too.",
    "previous_answer": {
        "student_response": "ok",
        "scores": {"Overall": "Poor"},
        "feedback": "Your response is very short and does not appear to address the target sentence. It may help to explain what the sentence means in your own words and link it to the surrounding text. Can you try again?"
    }
}'
```

## Python usage

You can use our code to use the models locally, on your machine. For more Python examples, see `learning_strategies_scoring/examples_score.py` (scoring) and `learning_strategies_scoring/examples_feedback.py` (feedback, including retry scenarios with `previous_answer`).

The scoring class (`LLMScoring`) must be instantiated only once (it will load the fine-tuned LLM and initialize the model and tokenizer). At the fist initialization, it will download the model from HuggingFace, and then it will load it every time from local.

There are two parameters: The HuggingFace location of the model's repo (`upn-nlp/...`) and the device on which the model will run (`cpu`, `cuda` or `mps` for Mac).

The scoring for students' responses can be called with the method `.score(data, task)`. 

`data` must be a dict with the following configuration, depending on the `task`:

```python
task = 'selfexplanation'
data = {
    'context': "The supporting text that the student has read.",
    'target_sentence': "The sentence from the text that the student must write a self-explanation.",
    'student_response': "The student's self-explanation.",
}
```

```python
task = 'thinkaloud'
data = {
    'context': "The supporting text that the student has read.",
    'target_sentence': "The sentence from the text that the student must write their thoughts.",
    'student_response': "The student's thoughts.",
}
```

```python
task = 'summary'
data = {
    'context': "The supporting text that the student has read.",
    'student_response': "The student's summary",
}
```

```python
task = 'paraphrasing'
data = {
    'target_sentence': "The support sentence.",
    'student_response': "The student's paraphrasing.",
}
```

```python
task = 'selfexplanation_multitext'
data = {
    'context': "The supporting multi-text content the student has read (e.g., several sources on a topic).",
    'target_sentence': "The sentence from the text that the student must write a self-explanation.",
    'student_response': "The student's self-explanation.",
}
```

To use feedback locally, instantiate `LLMScoring` with `feedback_model_name` and call `.feedback(data, task, previous_answer=None)`. It returns a dict with `scores`, `feedback`, and `try_again`. Pass `previous_answer` (the prior attempt's `student_response`, `scores`, and optionally `feedback`) on retries to get an evolution comment.

```python
llm_scoring = LLMScoring(
    'upb-nlp/qwen3_4b_scoring_all_tasks_with_se_improved',
    feedback_model_name='Qwen/Qwen3-4B-Instruct-2507',
)
result = llm_scoring.feedback(data, task)
# result = llm_scoring.feedback(data, task, previous_answer=prev)  # on retry
```
