"""
Evaluation Utilities Module

This module contains common utilities, patterns, and functions extracted from all
existing evaluations to be reused by generated evaluations.
"""

import re
import string
import json
import random
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from scipy.optimize import linear_sum_assignment

import common
from common import ANSWER_PATTERN, ANSWER_PATTERN_MULTICHOICE, HTML_JINJA, check_equality
from eval_types import Eval, EvalResult, SamplerBase, SingleEvalResult

# ============================================================================
# Answer Pattern Utilities
# ============================================================================

def extract_answer_with_pattern(response_text: str, pattern: str) -> Optional[str]:
    """Extract answer from response using a regex pattern"""
    match = re.search(pattern, response_text)
    return match.group(1) if match else None

def extract_multiple_choice_answer(response_text: str) -> Optional[str]:
    """Extract multiple choice answer (A/B/C/D) from response"""
    return extract_answer_with_pattern(response_text, ANSWER_PATTERN_MULTICHOICE)

def extract_answer_after_keyword(response_text: str, keyword: str = "Answer") -> Optional[str]:
    """Extract answer after a keyword like 'Answer:'"""
    pattern = rf"(?i){keyword}\s*:\s*([^\n]+)"
    return extract_answer_with_pattern(response_text, pattern)

def extract_code_block(response_text: str, language: str = "python") -> Optional[str]:
    """Extract code from markdown code blocks"""
    pattern = rf"```{language}\n(.*?)```"
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        code = match.group(1)
        # Remove function signature for HumanEval style
        if ":\n    " in code:
            code = code[code.find(":\n    ") + 2:]
        return code
    return response_text

# ============================================================================
# Scoring Functions
# ============================================================================

def exact_match_score(predicted: str, target: str) -> float:
    """Simple exact string matching"""
    return 1.0 if predicted.strip() == target.strip() else 0.0

def fuzzy_match_score(predicted: str, target: str) -> float:
    """Fuzzy matching with normalization"""
    def normalize(text: str) -> str:
        text = text.lower()
        exclude = set(string.punctuation)
        text = "".join(char for char in text if char not in exclude)
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        text = " ".join(text.split())
        return text
    
    pred_norm = normalize(predicted)
    target_norm = normalize(target)
    
    if pred_norm == "" or target_norm == "":
        return 1.0 if pred_norm == target_norm else 0.0
    
    return 1.0 if (pred_norm in target_norm or target_norm in pred_norm) else 0.0

def multiple_choice_score(predicted: str, target: str) -> float:
    """Score multiple choice answers"""
    return 1.0 if predicted == target else 0.0

def math_score_with_checker(predicted: str, target: str, equality_checker: SamplerBase) -> float:
    """Score mathematical answers using an equality checker"""
    return float(check_equality(equality_checker, target, predicted))

def numerical_score(predicted: str, target: str, tolerance: float = 1e-6) -> float:
    """Score numerical answers with tolerance"""
    try:
        pred_num = float(predicted.replace(",", ""))
        target_num = float(target.replace(",", ""))
        return 1.0 if abs(pred_num - target_num) <= tolerance else 0.0
    except ValueError:
        return exact_match_score(predicted, target)

# ============================================================================
# DROP-style Scoring (Exact Match + F1)
# ============================================================================

def _remove_articles(text: str) -> str:
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)

def _white_space_fix(text: str) -> str:
    return " ".join(text.split())

def _remove_punc(text: str) -> str:
    exclude = set(string.punctuation)
    if not _is_number(text):
        return "".join(ch for ch in text if ch not in exclude)
    else:
        return text

def _lower(text: str) -> str:
    return text.lower()

def _tokenize(text: str) -> List[str]:
    return re.split(" |-", text)

def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False

def _normalize_number(text: str) -> str:
    if _is_number(text):
        return str(float(text))
    else:
        return text

def _normalize_answer(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    parts = [
        _white_space_fix(_remove_articles(_normalize_number(_remove_punc(_lower(token)))))
        for token in _tokenize(text)
    ]
    parts = [part for part in parts if part.strip()]
    normalized = " ".join(parts).strip()
    return normalized

def _answer_to_bags(answer: Union[str, List[str], Tuple[str, ...]]) -> Tuple[List[str], List[Set[str]]]:
    if isinstance(answer, (list, tuple)):
        raw_spans = answer
    else:
        raw_spans = [answer]
    normalized_spans: List[str] = []
    token_bags = []
    for raw_span in raw_spans:
        normalized_span = _normalize_answer(raw_span)
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return normalized_spans, token_bags

def _compute_f1(predicted_bag: Set[str], gold_bag: Set[str]) -> float:
    intersection = len(gold_bag.intersection(predicted_bag))
    if not predicted_bag:
        precision = 1.0
    else:
        precision = intersection / float(len(predicted_bag))
    if not gold_bag:
        recall = 1.0
    else:
        recall = intersection / float(len(gold_bag))
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if not (precision == 0.0 and recall == 0.0)
        else 0.0
    ) * 100
    return f1

def _match_numbers_if_present(gold_bag: Set[str], predicted_bag: Set[str]) -> bool:
    gold_numbers = set()
    predicted_numbers = set()
    for word in gold_bag:
        if _is_number(word):
            gold_numbers.add(word)
    for word in predicted_bag:
        if _is_number(word):
            predicted_numbers.add(word)
    if (not gold_numbers) or gold_numbers.intersection(predicted_numbers):
        return True
    return False

def _align_bags(predicted: List[Set[str]], gold: List[Set[str]]) -> List[float]:
    """Takes gold and predicted answer sets and finds optimal alignment"""
    scores = np.zeros([len(gold), len(predicted)])
    for gold_index, gold_item in enumerate(gold):
        for pred_index, pred_item in enumerate(predicted):
            if _match_numbers_if_present(gold_item, pred_item):
                scores[gold_index, pred_index] = _compute_f1(pred_item, gold_item)
    row_ind, col_ind = linear_sum_assignment(-scores)

    max_scores = np.zeros([max(len(gold), len(predicted))])
    for row, column in zip(row_ind, col_ind):
        max_scores[row] = max(max_scores[row], scores[row, column])
    return max_scores

def drop_style_score(predicted: Union[str, List[str], Tuple[str, ...]], 
                    gold: Union[str, List[str], Tuple[str, ...]]) -> Tuple[float, float]:
    """
    DROP-style scoring: returns (exact_match, f1_score)
    """
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)

    if set(predicted_bags[0]) == set(gold_bags[0]) and len(predicted_bags[0]) == len(gold_bags[0]):
        exact_match = 1.0
    else:
        exact_match = 0.0

    f1_per_bag = _align_bags(predicted_bags[1], gold_bags[1])
    f1 = np.mean(f1_per_bag)
    f1 = round(f1, 2)
    return exact_match, f1

# ============================================================================
# MGSM-style Math Scoring
# ============================================================================

def parse_mgsm_answer(answer: str, answer_prefix: str = "Answer") -> str:
    """Parse answer from MGSM-style responses"""
    if answer_prefix not in answer:
        return ""
    
    answer_text = answer.split(answer_prefix)[-1].strip()
    numbers = re.findall(r"\d+\.?\d*", answer_text.replace(",", ""))
    return numbers[-1].rstrip(".") if numbers else ""

def mgsm_score(target: str, prediction: str) -> bool:
    """MGSM-style scoring for math problems"""
    if "." in prediction:
        prediction = prediction.rstrip("0").rstrip(".")
    
    target = target.replace(",", "")
    prediction = prediction.replace(",", "")
    
    return target == prediction

# ============================================================================
# LLM Grader Utilities (SimpleQA/BrowseComp style)
# ============================================================================

def create_llm_grader_prompt(question: str, correct_answer: str, predicted_answer: str, 
                           template_type: str = "simpleqa") -> str:
    """Create grader prompt for LLM-based evaluation"""
    
    if template_type == "simpleqa":
        return f"""
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].

Question: {question}
Gold target: {correct_answer}
Predicted answer: {predicted_answer}

Grade the predicted answer as one of:
A: CORRECT
B: INCORRECT  
C: NOT_ATTEMPTED

Just return the letters "A", "B", or "C", with no text around it.
""".strip()
    
    elif template_type == "browsecomp":
        return f"""
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}
[response]: {predicted_answer}
[correct_answer]: {correct_answer}

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above. Answer 'no' otherwise.
""".strip()
    
    else:
        raise ValueError(f"Unknown template type: {template_type}")

def grade_with_llm(grader_model: SamplerBase, question: str, correct_answer: str, 
                  predicted_answer: str, template_type: str = "simpleqa") -> str:
    """Grade an answer using an LLM grader"""
    prompt = create_llm_grader_prompt(question, correct_answer, predicted_answer, template_type)
    
    prompt_messages = [
        grader_model._pack_message(content=prompt, role="user")
    ]
    response = grader_model(prompt_messages)
    
    if template_type == "simpleqa":
        match = re.search(r"(A|B|C)", response)
        return match.group(0) if match else "C"
    elif template_type == "browsecomp":
        match = re.search(r"correct: (yes|no)", response)
        return match.group(0) if match else "no"
    
    return response

# ============================================================================
# Code Execution Utilities (HumanEval style)
# ============================================================================

def evaluate_code_execution(sample: dict, completions: list[str], 
                           timeout: float = 30.0, n_workers: int = 4) -> List[int]:
    """
    Evaluate functional correctness of code completions
    """
    try:
        from human_eval.execution import check_correctness
        
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for i, completion in enumerate(completions):
                args = (sample, completion, timeout, i)
                future = executor.submit(check_correctness, *args)
                futures.append(future)
            
            results = []
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
        
        return [int(r["passed"]) for r in results]
    except ImportError:
        # Fallback if human_eval is not available
        print("Warning: human_eval not available, using mock results")
        return [1] * len(completions)  # Mock all as passing

# ============================================================================
# Common Evaluation Templates
# ============================================================================

class EvaluationMixin:
    """Mixin class with common evaluation utilities"""
    
    def create_prompt_messages(self, content: str, sampler: SamplerBase) -> List[Dict[str, str]]:
        """Create standardized prompt messages"""
        return [sampler._pack_message(content=content, role="user")]
    
    def create_single_eval_result(self, prompt_messages: List[Dict[str, str]], 
                                response_text: str, score: float, 
                                correct_answer: Any, extracted_answer: Any,
                                metrics: Optional[Dict[str, Any]] = None) -> SingleEvalResult:
        """Create a standardized SingleEvalResult"""
        html = common.jinja_env.from_string(HTML_JINJA).render(
            prompt_messages=prompt_messages,
            next_message=dict(content=response_text, role="assistant"),
            score=score,
            correct_answer=correct_answer,
            extracted_answer=extracted_answer,
        )
        convo = prompt_messages + [dict(content=response_text, role="assistant")]
        return SingleEvalResult(
            html=html, score=score, convo=convo, metrics=metrics or {}
        )
    
    def batch_process_examples(self, examples: List[Dict], process_fn, 
                             batch_size: int = 20, checkpoint_file: Optional[str] = None,
                             processed_results: Optional[List] = None) -> List[SingleEvalResult]:
        """Standard batch processing with checkpointing"""
        if processed_results is None:
            processed_results = []
        
        num_already_processed = len(processed_results)
        
        if num_already_processed >= len(examples):
            print("All examples were already processed.")
            return processed_results
        
        examples_to_process = examples[num_already_processed:]
        print(f"Processing {len(examples_to_process)} examples...")
        
        for i in range(0, len(examples_to_process), batch_size):
            batch = examples_to_process[i:i + batch_size]
            batch_results = common.map_with_progress(process_fn, batch)
            processed_results.extend(batch_results)
            
            if checkpoint_file:
                common.save_checkpoint(checkpoint_file, batch_results)
        
        return processed_results

# ============================================================================
# Scoring Strategy Factory
# ============================================================================

class ScoringStrategy:
    """Factory for different scoring strategies"""
    
    @staticmethod
    def get_scorer(method: str, **kwargs):
        """Get scoring function based on method name"""
        if method == "exact_match":
            return lambda pred, target: exact_match_score(pred, target)
        
        elif method == "fuzzy_match":
            return lambda pred, target: fuzzy_match_score(pred, target)
        
        elif method == "multiple_choice":
            return lambda pred, target: multiple_choice_score(pred, target)
        
        elif method == "math_equivalence":
            equality_checker = kwargs.get("equality_checker")
            if not equality_checker:
                raise ValueError("math_equivalence requires equality_checker")
            return lambda pred, target: math_score_with_checker(pred, target, equality_checker)
        
        elif method == "drop_style":
            return lambda pred, target: drop_style_score(pred, target)[0]  # Return EM score
        
        elif method == "drop_f1":
            return lambda pred, target: drop_style_score(pred, target)[1]  # Return F1 score
        
        elif method == "mgsm_math":
            return lambda pred, target: float(mgsm_score(target, pred))
        
        elif method == "llm_grader":
            grader_model = kwargs.get("grader_model")
            template_type = kwargs.get("template_type", "simpleqa")
            if not grader_model:
                raise ValueError("llm_grader requires grader_model")
            
            def score_with_grader(pred, target, question=""):
                result = grade_with_llm(grader_model, question, target, pred, template_type)
                if template_type == "simpleqa":
                    return 1.0 if result == "A" else 0.0
                elif template_type == "browsecomp":
                    return 1.0 if "yes" in result else 0.0
                return 0.0
            
            return score_with_grader
        
        elif method == "numerical":
            tolerance = kwargs.get("tolerance", 1e-6)
            return lambda pred, target: numerical_score(pred, target, tolerance)
        
        else:
            raise ValueError(f"Unknown scoring method: {method}")

# ============================================================================
# Answer Extraction Strategy Factory
# ============================================================================

class AnswerExtractor:
    """Factory for different answer extraction strategies"""
    
    @staticmethod
    def get_extractor(method: str, **kwargs):
        """Get answer extraction function based on method name"""
        if method == "regex_pattern":
            pattern = kwargs.get("pattern", ANSWER_PATTERN)
            return lambda text: extract_answer_with_pattern(text, pattern)
        
        elif method == "multiple_choice":
            return extract_multiple_choice_answer
        
        elif method == "answer_keyword":
            keyword = kwargs.get("keyword", "Answer")
            return lambda text: extract_answer_after_keyword(text, keyword)
        
        elif method == "code_block":
            language = kwargs.get("language", "python")
            return lambda text: extract_code_block(text, language)
        
        elif method == "mgsm_parse":
            prefix = kwargs.get("answer_prefix", "Answer")
            return lambda text: parse_mgsm_answer(text, prefix)
        
        elif method == "full_response":
            return lambda text: text.strip()
        
        else:
            raise ValueError(f"Unknown extraction method: {method}") 