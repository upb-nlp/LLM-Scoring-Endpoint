from learning_strategies_scoring.api_llm_scoring import LLMScoring
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

scorer = None

VALID_TASKS = ["selfexplanation", "thinkaloud", "summary", "paraphrasing", "selfexplanation_multitext"]

@app.route('/score/<task>', methods=['POST'] )
def score(task):
    if scorer is None:
        return "Scorer not initialized", 503
    if task not in VALID_TASKS:
        return f"Invalid Task (should be one of: {VALID_TASKS})", 400
    args = request.json
    try:
        prediction = scorer.score(args, task)
    except ValueError as e:
        return str(e), 400
    response = jsonify(prediction)
    return response, 200

@app.route('/feedback/<task>', methods=['POST'])
def feedback(task):
    if task not in VALID_TASKS:
        return f"Invalid Task (should be one of: {VALID_TASKS})", 400
    args = request.json or {}
    previous_answer = args.pop('previous_answer', None)
    try:
        result = scorer.feedback(args, task, previous_answer=previous_answer)
    except ValueError as e:
        return str(e), 400
    return jsonify(result), 200

if __name__ == '__main__':
    scorer = LLMScoring(
        'upb-nlp/qwen3_4b_scoring_all_tasks_with_se_improved',
        feedback_model_name='Qwen/Qwen3-4B-Instruct-2507',
    )
    app.run(host='0.0.0.0', port=5001)