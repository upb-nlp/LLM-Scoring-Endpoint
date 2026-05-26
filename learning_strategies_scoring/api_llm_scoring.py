import torch
import json
from os import path
from vllm import LLM
from vllm.sampling_params import SamplingParams

class LLMScoring:
    # Qwen3-4B bfloat16, max_model_len=4096: ~9.4 GB weights + ~0.6 GB KV cache + 15% overhead
    _SCORING_MODEL_MEMORY_GB = 16.0
    # Feedback model with bitsandbytes 4-bit quantization (3–7B range)
    _FEEDBACK_MODEL_MEMORY_GB = 8.0

    @staticmethod
    def _gpu_memory_utilization(target_gb: float) -> float:
        """Return the gpu_memory_utilization fraction that allocates target_gb of VRAM
        from whatever is currently free, so the absolute amount stays constant even when
        other processes are using GPU memory."""
        free_mem, _ = torch.cuda.mem_get_info()
        target_bytes = target_gb * (1024 ** 3)
        return min(target_bytes / free_mem, 0.95)

    def __init__(self, model_path, feedback_model_name=None):
        """
        model_path: str
            Path to the model to be used for scoring
        feedback_model_name: str, optional
            HuggingFace model name/path for generating feedback (e.g. 'meta-llama/Llama-3.2-3B-Instruct')
        """

        self.model = LLM(model=model_path, dtype=torch.bfloat16, max_model_len=4096, gpu_memory_utilization=self._gpu_memory_utilization(self._SCORING_MODEL_MEMORY_GB), enable_prefix_caching=True)
        self.scoring_details_dir = path.join('learning_strategies_scoring', 'scoring_details')
        self.params = SamplingParams(temperature=0, max_tokens=300)

        if feedback_model_name:
            self.feedback_model = LLM(
                model=feedback_model_name,
                max_model_len=4096,
                gpu_memory_utilization=self._gpu_memory_utilization(self._FEEDBACK_MODEL_MEMORY_GB),
                quantization="bitsandbytes",
                load_format="bitsandbytes",
                enable_prefix_caching=True,
            )
        else:
            self.feedback_model = None
        self.feedback_params = SamplingParams(temperature=0.7, max_tokens=512)
    
    def generate_response(self, formatted_prompt):
        """
        formatted_prompt: str
            Prompt to be used for generating the response

        Returns:
            str: Generated response
        """
        tokenizer = self.model.get_tokenizer()
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": formatted_prompt}],
            add_special_tokens=False,
            tokenize=False,
            add_generation_prompt=True,
        )
        response = self.model.generate([text], self.params)
        raw = response[0].outputs[0].text
        return raw
    
    def extract_score_from_response(self, response, scoring_details=None):
        """
        response: str
            Response from the model
        scoring_details: dict, optional
            Scoring rubric. When provided, criterion names and values are
            validated against the rubric so that rubric description lines the
            model sometimes echoes back are ignored.

        Returns:
            dict: Dictionary containing the scores
        """
        response = response.replace('<|endoftext|>', '').replace('<|im_start|>', '')

        valid_criteria = {}
        if scoring_details is not None:
            for entry in scoring_details['scoring_rubric'].values():
                valid_criteria[entry['name']] = set(entry['scores'].values())

        scores = {}

        def _add(key, value):
            key = key.strip()
            value = value.strip()
            if not key or not value:
                return
            if valid_criteria:
                if key not in valid_criteria:
                    return
                if value not in valid_criteria[key]:
                    return
            if key not in scores:
                scores[key] = value

        if '\n' not in response:
            # Model emitted everything on a single line: "- Foo: A - Bar: B ..."
            for fragment in response.split('- '):
                if ': ' not in fragment:
                    continue
                key, value = fragment.split(': ', 1)
                _add(key, value)
        else:
            for line in response.split('\n'):
                # Skip rubric description lines like "- - Good: ..." that the
                # model occasionally echoes from the prompt.
                if line.startswith('- - '):
                    continue
                if not line.startswith('- '):
                    continue
                if ': ' not in line:
                    continue
                key, value = line.split(': ', 1)
                _add(key[2:], value)

        return scores
    
    def prepare_scoring_rubric_prompt(self, scoring_details):
        """
        scoring_details: dict
            Scoring details dictionary

        Returns:
            str: Scoring rubric prompt
        """
    
        task_prompt = scoring_details['task']
        scoring_rubric_prompt = ""
        for _, dict in scoring_details['scoring_rubric'].items():
            descriptions = '\n'.join([f"- - {dict['scores'][num]}: {dict['scores_description'][num]}" for num in dict['scores']])
            scoring_rubric_prompt += f"- {dict['name']}:\n{descriptions}\n"
        scoring_rubric_prompt = scoring_rubric_prompt[:-1]

        return task_prompt, scoring_rubric_prompt


    def prepare_prompt(self, data, task):
        """
        data: dict
            Data to be used for scoring
        task: str
            Task to be scored

        Returns:
            str: Formatted prompt
        """

        scoring_start_prompt = "Rate the quality of the following performed task, based on the scoring rubric."

        if task == 'selfexplanation':
            if 'context' not in data or 'target_sentence' not in data or 'student_response' not in data:
                raise ValueError('Data must contain context, target_sentence, and student_response fields')
            
            scoring_details = json.load(open(path.join(self.scoring_details_dir, 'selfexplanation_thinkaloud_full_se.json'), 'r'))
            task_prompt, scoring_rubric_prompt = self.prepare_scoring_rubric_prompt(scoring_details)

            prompt = f"{scoring_start_prompt}\n\n### Task description: {task_prompt}\n\n- Context: {data['context']}\n\n- Phrase: {data['target_sentence']}\n\n### Execution: {data['student_response']}\n\n### Scoring rubric:\n{scoring_rubric_prompt}"

        elif task == 'thinkaloud':
            if 'context' not in data or 'target_sentence' not in data or 'student_response' not in data:
                raise ValueError('Data must contain context, target_sentence, and student_response fields')

            scoring_details = json.load(open(path.join(self.scoring_details_dir, 'selfexplanation_thinkaloud_full_ta.json'), 'r'))
            task_prompt, scoring_rubric_prompt = self.prepare_scoring_rubric_prompt(scoring_details)

            prompt = f"{scoring_start_prompt}\n\n### Task description: {task_prompt}\n\n- Context: {data['context']}\n\n- Phrase: {data['target_sentence']}\n\n### Execution: {data['student_response']}\n\n### Scoring rubric:\n{scoring_rubric_prompt}"

        elif task == 'selfexplanation_multitext':
            if 'context' not in data or 'target_sentence' not in data or 'student_response' not in data:
                raise ValueError('Data must contain context, target_sentence, and student_response fields')

            scoring_details = json.load(open(path.join(self.scoring_details_dir, 'se_improved.json'), 'r'))
            task_prompt, scoring_rubric_prompt = self.prepare_scoring_rubric_prompt(scoring_details)

            prompt = f"{scoring_start_prompt}\n\n### Task description: {task_prompt}\n\n- Context: {data['context']}\n\n- Phrase: {data['target_sentence']}\n\n### Execution: {data['student_response']}\n\n### Scoring rubric:\n{scoring_rubric_prompt}"

        elif task == 'summary':
            if 'context' not in data or 'student_response' not in data:
                raise ValueError('Data must contain context and student_response fields')

            scoring_details = json.load(open(path.join(self.scoring_details_dir, 'summaries_aloe.json'), 'r'))
            task_prompt, scoring_rubric_prompt = self.prepare_scoring_rubric_prompt(scoring_details)

            prompt = f"{scoring_start_prompt}\n\n### Task description: {task_prompt}\n\n- Context: {data['context']}\n\n### Execution: {data['student_response']}\n\n### Scoring rubric:\n{scoring_rubric_prompt}"
        
        elif task == 'paraphrasing':
            if 'target_sentence' not in data or 'student_response' not in data:
                raise ValueError('Data must contain target_sentence and student_response fields')

            scoring_details = json.load(open(path.join(self.scoring_details_dir, 'paraphrasing_ulpc.json'), 'r'))
            task_prompt, scoring_rubric_prompt = self.prepare_scoring_rubric_prompt(scoring_details)

            prompt = f"{scoring_start_prompt}\n\n### Task description: {task_prompt}\n\n- Sentence: {data['target_sentence']}\n\n### Execution: {data['student_response']}\n\n### Scoring rubric:\n{scoring_rubric_prompt}"

        return prompt

    def score(self, data, task):
        """
        data: dict
            Data to be used for scoring
        task: str
            Task to be scored

        Returns:
            dict: Dictionary containing the scores
        """
        
        formatted_prompt = self.prepare_prompt(data, task)
        response = self.generate_response(formatted_prompt)
        scoring_details = self._get_scoring_details(task)
        scores = self.extract_score_from_response(response, scoring_details)

        return scores

    def _get_scoring_details(self, task):
        """
        task: str
            Task to get scoring details for

        Returns:
            dict: Scoring details dictionary
        """
        task_to_file = {
            'selfexplanation': 'selfexplanation_thinkaloud_full_se.json',
            'thinkaloud': 'selfexplanation_thinkaloud_full_ta.json',
            'summary': 'summaries_aloe.json',
            'paraphrasing': 'paraphrasing_ulpc.json',
            'selfexplanation_multitext': 'se_improved.json',
        }
        with open(path.join(self.scoring_details_dir, task_to_file[task]), 'r') as f:
            return json.load(f)

    def _build_scores_summary(self, scores, scoring_details):
        """
        Builds a human-readable summary of the scores with their descriptions.

        scores: dict
            Dictionary of criterion -> score value
        scoring_details: dict
            Scoring details dictionary with rubric info

        Returns:
            str: Formatted scores summary
        """
        lines = []
        rubric = scoring_details['scoring_rubric']
        for criterion, value in scores.items():
            description = ""
            for _, rubric_entry in rubric.items():
                if rubric_entry['name'].lower().replace(' ', '') == criterion.lower().replace(' ', ''):
                    for score_key, score_label in rubric_entry['scores'].items():
                        if score_label == value or score_key == value:
                            description = rubric_entry['scores_description'][score_key]
                            break
                    break
            lines.append(f"- {criterion}: {value} ({description})" if description else f"- {criterion}: {value}")
        return '\n'.join(lines)

    def _build_feedback_prompt(self, data, task, scores, scoring_details):
        """
        Builds the prompt for the feedback LLM.

        Returns:
            str: The feedback prompt
        """
        task_description = scoring_details['task']
        scores_summary = self._build_scores_summary(scores, scoring_details)

        student_response = data['student_response']
        context_info = ""
        if 'context' in data:
            context_info += f"Context: {data['context']}\n"
        if 'target_sentence' in data:
            context_info += f"Target sentence: {data['target_sentence']}\n"

        prompt = (
            f"You are an educator giving feedback on a student's {task}.\n\n"
            f"The student was asked to: {task_description}\n\n"
            f"{context_info}\n"
            f"The student wrote: {student_response}\n\n"
            f"Here are the scores they received:\n{scores_summary}\n\n"
            f"Address the student directly using \"you\". "
            f"Start by acknowledging what the student did well, then provide tips to improve. "
            f"Maximum 30 words total. Do not prompt the student to try again or redo the task. "
            f"Do not add motivational phrases or offers of help.\n\n"
            f"Language and hedging rules (strict):\n"
            f"- Base all comments only on observable features of the student's response and the scores above. "
            f"The scores come from probabilistic NLP models, so do not overstate confidence in them.\n"
            f"- Do not make strong claims about the student's understanding, knowledge, intentions, effort, or awareness. "
            f"Avoid phrases like \"you understand\", \"you know\", \"you showed awareness\", \"you showed effort\", "
            f"\"you clearly\", \"you correctly identified\", or \"you demonstrated\".\n"
            f"- Use tentative, evidence-based language instead. Prefer phrases like \"your response suggests...\", "
            f"\"your response mentions...\", \"your response appears to...\", \"it seems that...\", "
            f"\"you may want to...\", or \"it may help to...\".\n"
            f"- Suggested phrase swaps: \"You clearly understand...\" -> \"Your response suggests some awareness of...\"; "
            f"\"You showed understanding of...\" -> \"Your response seems to point to...\"; "
            f"\"You correctly identified...\" -> \"You appear to identify...\"; "
            f"\"You showed effort...\" -> \"You seem to have made an attempt to respond...\".\n"
            f"- If the response is gibberish, off-topic, copied, extremely minimal, or incomplete, "
            f"do not include praise that implies comprehension. Describe the issue neutrally using observable features "
            f"(e.g., \"your response is very short\", \"your response does not appear to address the target sentence\") "
            f"and direct the student toward the next step."
        )
        return prompt

    def _generate_paraphrase(self, sentence):
        """
        Uses the feedback model to generate a paraphrase of the given sentence.

        sentence: str
            The sentence to paraphrase

        Returns:
            str: The paraphrased sentence
        """
        prompt = (
            f"Paraphrase the following sentence. "
            f"Return only the paraphrased sentence, nothing else.\n\n"
            f"Sentence: {sentence}"
        )
        messages = [{"role": "user", "content": prompt}]
        response = self.feedback_model.chat(messages, self.feedback_params)
        return response[0].outputs[0].text.strip()

    def _should_try_again(self, scores, task):
        """
        Determines if the student should be prompted to try again based on scores.

        For paraphrasing: if Garbage Content, Frozen Expressions, or Irrelevant are not "Not present"/"Relevant".
        For selfexplanation/thinkaloud: if Overall is "Poor".

        Returns:
            bool
        """
        if task == 'paraphrasing':
            garbage = scores.get('Garbage Content', 'Not present')
            frozen = scores.get('Frozen Expressions', 'Not present')
            irrelevant = scores.get('Irrelevant', 'Relevant')
            return garbage != 'Not present' or frozen != 'Not present' or irrelevant != 'Relevant'
        elif task in ('selfexplanation', 'thinkaloud'):
            return scores.get('Overall', '') == 'Poor'
        elif task == 'selfexplanation_multitext':
            return scores.get('Overall Self-Explanation Quality', '') == 'Poor'
        return False

    def _scores_a_bit_low(self, scores, task):
        """
        Determines if scores indicate room for improvement that does not warrant
        prompting a retry — i.e. the response is not failing, but a paraphrase
        example could still help the student.

        Returns:
            bool
        """
        if task == 'paraphrasing':
            return scores.get('Paraphrase Quality', '') == 'Poor' or scores.get('Writing Quality', '') == 'Poor'
        elif task in ('selfexplanation', 'thinkaloud'):
            return scores.get('Overall', '') == 'Satisfactory'
        elif task == 'selfexplanation_multitext':
            return scores.get('Overall Self-Explanation Quality', '') == 'Fair quality'
        return False

    def _build_evolution_prompt(self, data, task, scores, previous_answer, scoring_details):
        """
        Builds the prompt asking the feedback model to comment in one sentence on
        whether the retry attempt improved compared to the previous attempt.

        previous_answer: dict
            Must contain at least 'student_response' and 'scores'. May also contain
            'feedback' (the feedback previously shown to the student).

        Returns:
            str: The evolution prompt
        """
        previous_scores_summary = self._build_scores_summary(
            previous_answer.get('scores', {}), scoring_details
        )
        current_scores_summary = self._build_scores_summary(scores, scoring_details)

        previous_feedback = previous_answer.get('feedback', '')
        previous_feedback_block = (
            f"Previous feedback given to the student:\n{previous_feedback}\n\n"
            if previous_feedback else ""
        )

        prompt = (
            f"You are comparing two attempts a student made at the same {task} task.\n\n"
            f"Previous attempt:\n"
            f"- Response: {previous_answer.get('student_response', '')}\n"
            f"- Scores:\n{previous_scores_summary}\n\n"
            f"{previous_feedback_block}"
            f"Current attempt (retry):\n"
            f"- Response: {data['student_response']}\n"
            f"- Scores:\n{current_scores_summary}\n\n"
            f"In exactly one sentence, comment on how the current response evolved compared "
            f"to the previous one — whether it improved, stayed roughly the same, or got worse. "
            f"Address the student directly using \"you\". Base the comment only on observable "
            f"differences between the two responses and their scores; use tentative language "
            f"(e.g. \"your response appears to...\", \"it seems...\") and do not make strong "
            f"claims about the student's understanding or effort. Return only that one sentence."
        )
        return prompt

    def feedback(self, data, task, previous_answer=None):
        """
        Scores the student response and generates friendly feedback using a vLLM model.

        data: dict
            Data to be used for scoring (same as score method)
        task: str
            Task to be scored (same as score method)
        previous_answer: dict, optional
            The student's previous attempt for this task. None on the first attempt,
            filled in on a retry. Expected keys: 'student_response', 'scores',
            and optionally 'feedback'. When provided, the function emits an
            additional one-sentence comment on how the response evolved.

        Returns:
            dict: Dictionary containing 'scores' (dict), 'feedback' (str), and 'try_again' (bool)
        """
        if self.feedback_model is None:
            raise ValueError("Feedback model not initialized. Pass feedback_model_name to __init__.")

        scores = self.score(data, task)
        scoring_details = self._get_scoring_details(task)

        feedback_prompt = self._build_feedback_prompt(data, task, scores, scoring_details)

        messages = [{"role": "user", "content": feedback_prompt}]
        response = self.feedback_model.chat(messages, self.feedback_params)
        feedback_text = response[0].outputs[0].text

        is_retry = previous_answer is not None

        if is_retry:
            evolution_prompt = self._build_evolution_prompt(
                data, task, scores, previous_answer, scoring_details
            )
            evolution_response = self.feedback_model.chat(
                [{"role": "user", "content": evolution_prompt}], self.feedback_params
            )
            evolution_comment = evolution_response[0].outputs[0].text.strip()
            feedback_text = f"{evolution_comment} {feedback_text}"

        try_again = False
        if self._should_try_again(scores, task) and not is_retry:
            feedback_text += " Can you try again?"
            try_again = True

        if not try_again and self._scores_a_bit_low(scores, task) and 'target_sentence' in data:
            paraphrased = self._generate_paraphrase(data['target_sentence'])
            feedback_text += f" Here is a rephrased version of the sentence to help you: {paraphrased}"

        return {
            'scores': scores,
            'feedback': feedback_text,
            'try_again': try_again,
        }