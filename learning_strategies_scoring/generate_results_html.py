import json
from os import path

SCRIPT_DIR = path.dirname(path.abspath(__file__))

with open('feedback_scenarios_results.json', 'r') as f:
    results = json.load(f)

# Group by task
grouped = {}
for r in results:
    grouped.setdefault(r['task'], []).append(r)

def score_color(value):
    """Return a CSS class based on score value."""
    v = value.lower()
    if v in ('excellent', 'not present', 'relevant', 'present', 'not applicable'):
        return 'score-good'
    if v in ('good',):
        return 'score-ok'
    if v in ('satisfactory', 'not present/poor'):
        return 'score-mid'
    if v in ('poor', 'too much', 'not relevant', 'irrelevant'):
        return 'score-bad'
    return ''

def try_again_badge(entry):
    if entry['try_again']:
        return '<span class="badge badge-retry">Try again</span>'
    if entry['is_retry']:
        return '<span class="badge badge-is-retry">Retry attempt</span>'
    return ''

rows_html = ""
for task, entries in grouped.items():
    task_label = task.replace('selfexplanation', 'Self-Explanation').replace('paraphrasing', 'Paraphrasing')
    rows_html += f'<h2>{task_label}</h2>\n'

    for entry in entries:
        name_label = entry['name'].replace('_', ' ').title()

        scores_html = ""
        for k, v in entry['scores'].items():
            css = score_color(v)
            scores_html += f'<tr><td class="score-name">{k}</td><td class="score-value {css}">{v}</td></tr>\n'

        badge = try_again_badge(entry)

        rows_html += f"""
<div class="card">
  <div class="card-header">
    <span class="scenario-name">{name_label}</span>
    {badge}
  </div>

  <div class="card-body">
    <div class="field">
      <div class="field-label">Target Sentence</div>
      <div class="field-value target">{entry['target_sentence']}</div>
    </div>

    <div class="field">
      <div class="field-label">Student Response</div>
      <div class="field-value student">{entry['student_response']}</div>
    </div>

    <table class="scores-table">
      <thead><tr><th>Criterion</th><th>Score</th></tr></thead>
      <tbody>{scores_html}</tbody>
    </table>

    <div class="feedback-box">
      <div class="field-label">Feedback</div>
      <div class="feedback-text">{entry['feedback']}</div>
    </div>
  </div>
</div>
"""

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Feedback Scenarios Results</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #f0f2f5; color: #1a1a2e; padding: 2rem; }}
  h1 {{ text-align: center; margin-bottom: 0.25rem; font-size: 1.8rem; }}
  .subtitle {{ text-align: center; color: #666; margin-bottom: 2rem; font-size: 0.95rem; }}
  h2 {{ margin: 2rem 0 1rem; padding-bottom: 0.4rem; border-bottom: 2px solid #4a6fa5; color: #4a6fa5; font-size: 1.35rem; }}

  .card {{ background: #fff; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.07); margin-bottom: 1.5rem; overflow: hidden; }}
  .card-header {{ background: #4a6fa5; color: #fff; padding: 0.75rem 1.25rem; display: flex; align-items: center; gap: 0.75rem; }}
  .scenario-name {{ font-weight: 600; font-size: 1.05rem; }}
  .card-body {{ padding: 1.25rem; }}

  .badge {{ font-size: 0.75rem; padding: 0.2rem 0.6rem; border-radius: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.03em; }}
  .badge-retry {{ background: #e74c3c; color: #fff; }}
  .badge-is-retry {{ background: #f39c12; color: #fff; }}

  .field {{ margin-bottom: 1rem; }}
  .field-label {{ font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #4a6fa5; margin-bottom: 0.3rem; }}
  .field-value {{ padding: 0.6rem 0.85rem; border-radius: 6px; font-size: 0.95rem; line-height: 1.5; }}
  .target {{ background: #eef2f7; border-left: 3px solid #4a6fa5; }}
  .student {{ background: #fdf6ec; border-left: 3px solid #f39c12; }}

  .scores-table {{ width: 100%; border-collapse: collapse; margin-bottom: 1rem; font-size: 0.9rem; }}
  .scores-table thead th {{ text-align: left; padding: 0.45rem 0.65rem; background: #f5f7fa; border-bottom: 2px solid #ddd; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.04em; color: #555; }}
  .scores-table td {{ padding: 0.4rem 0.65rem; border-bottom: 1px solid #eee; }}
  .score-name {{ color: #333; }}
  .score-value {{ font-weight: 600; }}
  .score-good {{ color: #27ae60; }}
  .score-ok {{ color: #2980b9; }}
  .score-mid {{ color: #f39c12; }}
  .score-bad {{ color: #e74c3c; }}

  .feedback-box {{ background: #eafaf1; border-left: 4px solid #27ae60; border-radius: 6px; padding: 1rem 1.15rem; }}
  .feedback-box .field-label {{ color: #27ae60; }}
  .feedback-text {{ font-size: 1.05rem; line-height: 1.6; color: #1a1a2e; }}
</style>
</head>
<body>
  <h1>Feedback Scenarios Results</h1>
  <p class="subtitle">{len(results)} scenarios across {len(grouped)} tasks</p>
  {rows_html}
</body>
</html>
"""

output_path = path.join('feedback_scenarios_results.html')
with open(output_path, 'w') as f:
    f.write(html)
