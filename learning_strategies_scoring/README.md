# INFLO LLM Scoring Integration

The API for scoring a student response (self-explanation, think-aloud, summary or paraphrasing). The model will return a dictionarry with scores across multiple rubrics. For explanations on what those scores mean, please refer to `learning_strategies_scoring/scoring_details`. 

## API usage

You can use our Flask server to send requests for scoring. The request must come in the form of a POST to `https://chat.readerbench.com/score/<task>`.

```c
curl --location 'https://chat.readerbench.com/score/selfexplanation' \
--header 'Content-Type: application/json' \
--data '{
    "context": "The supporting text that the student has read.",
    "target_sentence": "The sentence from the text that the student must write a self-explanation.",
    "student_response": "The student'\''s self-explanation.",
}'
```

```c
curl --location 'https://chat.readerbench.com/score/thinkaloud' \
--header 'Content-Type: application/json' \
--data '{
    "context": "The supporting text that the student has read.",
    "target_sentence": "The sentence from the text that the student must write their thoughts.",
    "student_response": "The student'\''s thoughts.",
}'
```

```c
curl --location 'https://chat.readerbench.com/score/summary' \
--header 'Content-Type: application/json' \
--data '{
    "context": "The supporting text that the student has read.",
    "student_response": "The student'\''s summary",
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

## Python usage

You can use our code to use the models locally, on your machine. Refer to `examples.py` for an usage example.

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