"""
Evaluation templates for different task types.

This module contains all the code templates used to generate evaluations
for different types of tasks (MMLU, math, reading comprehension, etc.).
"""

def get_generic_template() -> str:
    """Get a safe generic template"""
    return '''import random
from datasets import load_dataset
from typing import Optional

import common
from eval_types import Eval, EvalResult, SamplerBase, SingleEvalResult
from eval_utils import extract_answer_after_keyword, ScoringStrategy, EvaluationMixin

class {class_name}(Eval, EvaluationMixin):
    def __init__(self, dataset_name: str = "{dataset_name}",
                 num_examples: Optional[int] = None,
                 batch_size: int = 20,
                 checkpoint_file: Optional[str] = None,
                 prompt_template: Optional[str] = None):
        
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.checkpoint_file = checkpoint_file
        self.processed_results = []
        
        # Set default or custom prompt template
        self.prompt_template = prompt_template or "{prompt_template}"
        
        # Load dataset{dataset_config_comment}
        dataset = load_dataset(dataset_name{dataset_config_param}, split="test")
        if num_examples:
            self.examples = random.Random(0).sample(list(dataset), num_examples)
        else:
            self.examples = list(dataset)
        
        # Load checkpoint if exists
        if self.checkpoint_file:
            self.processed_results = common.load_checkpoint(self.checkpoint_file)
    
    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def process_example(row: dict) -> SingleEvalResult:
            # Create prompt from template and input
            question = row.get("{input_column}", str(row))
            
            # Create template variables safely to avoid keyword conflicts
            template_vars = dict(row)  # Start with all row data
            template_vars.update({{
                'question': question,
                'input': question
            }})
            prompt_content = self.prompt_template.format(**template_vars)
            
            prompt_messages = self.create_prompt_messages(prompt_content, sampler)
            response_text = sampler(prompt_messages)
            
            # Extract answer
            extracted_answer = extract_answer_after_keyword(response_text, "Answer")
            if not extracted_answer:
                extracted_answer = response_text.strip()
            
            # Score using exact match by default
            scorer = ScoringStrategy.get_scorer("{scoring_method}")
            target_answer = str(row.get("{target_column}", ""))
            score = scorer(extracted_answer, target_answer)
            
            return self.create_single_eval_result(
                prompt_messages, response_text, score,
                target_answer, extracted_answer
            )
        
        # Use batch processing utility
        self.processed_results = self.batch_process_examples(
            self.examples, process_example, self.batch_size, 
            self.checkpoint_file, self.processed_results
        )
        
        return common.aggregate_results(self.processed_results)
'''

def get_mmlu_template() -> str:
    """Get MMLU-style template"""
    return '''import random
from datasets import load_dataset
from typing import Optional

import common
from common import format_multichoice_question
from eval_types import Eval, EvalResult, SamplerBase, SingleEvalResult
from eval_utils import extract_multiple_choice_answer, ScoringStrategy, EvaluationMixin

class {class_name}(Eval, EvaluationMixin):
    def __init__(self, dataset_name: str = "{dataset_name}", 
                 num_examples: Optional[int] = None,
                 batch_size: int = 20, 
                 checkpoint_file: Optional[str] = None,
                 prompt_template: Optional[str] = None):
        
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.checkpoint_file = checkpoint_file
        self.processed_results = []
        
        # Set default or custom prompt template
        self.prompt_template = prompt_template or "{prompt_template}"
        
        # Load dataset
        dataset = load_dataset(dataset_name{dataset_config_param}, split="test")
        if num_examples:
            self.examples = random.Random(0).sample(list(dataset), num_examples)
        else:
            self.examples = list(dataset)
        
        # Load checkpoint if exists
        if self.checkpoint_file:
            self.processed_results = common.load_checkpoint(self.checkpoint_file)
    
    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def process_example(row: dict) -> SingleEvalResult:
            # Format multiple choice question and use custom prompt template
            formatted_question = format_multichoice_question(row)
            
            # Use customizable prompt template with safe variable handling
            template_vars = dict(row)  # Start with all row data
            template_vars.update({{
                'question': formatted_question,
                'input': formatted_question
            }})
            prompt_content = self.prompt_template.format(**template_vars)
            
            prompt_messages = self.create_prompt_messages(prompt_content, sampler)
            response_text = sampler(prompt_messages)
            extracted_answer = extract_multiple_choice_answer(response_text)
            
            # Score using multiple choice scoring
            scorer = ScoringStrategy.get_scorer("multiple_choice")
            score = scorer(extracted_answer, row["{answer_column}"])
            
            return self.create_single_eval_result(
                prompt_messages, response_text, score, 
                row["{answer_column}"], extracted_answer
            )
        
        # Use batch processing utility
        self.processed_results = self.batch_process_examples(
            self.examples, process_example, self.batch_size, 
            self.checkpoint_file, self.processed_results
        )
        
        return common.aggregate_results(self.processed_results)
'''

def get_math_template() -> str:
    """Get Math-style template"""
    return '''import random
from datasets import load_dataset
from typing import Optional

import common
from eval_types import Eval, EvalResult, SamplerBase, SingleEvalResult
from eval_utils import extract_answer_after_keyword, ScoringStrategy, EvaluationMixin

class {class_name}(Eval, EvaluationMixin):
    def __init__(self, dataset_name: str = "{dataset_name}",
                 num_examples: Optional[int] = None,
                 batch_size: int = 20,
                 checkpoint_file: Optional[str] = None,
                 prompt_template: Optional[str] = None):
        
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.checkpoint_file = checkpoint_file
        self.processed_results = []
        
        # Set default or custom prompt template
        self.prompt_template = prompt_template or "{prompt_template}"
        
        # Load dataset
        dataset = load_dataset(dataset_name{dataset_config_param}, split="test")
        if num_examples:
            self.examples = random.Random(0).sample(list(dataset), num_examples)
        else:
            self.examples = list(dataset)
        
        # Load checkpoint if exists
        if self.checkpoint_file:
            self.processed_results = common.load_checkpoint(self.checkpoint_file)
    
    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def process_example(row: dict) -> SingleEvalResult:
            question = row.get("{input_column}", "")
            # Use customizable prompt template with safe variable handling
            template_vars = dict(row)  # Start with all row data
            template_vars.update({{
                'question': question,
                'input': question
            }})
            prompt_content = self.prompt_template.format(**template_vars)
            
            prompt_messages = self.create_prompt_messages(prompt_content, sampler)
            response_text = sampler(prompt_messages)
            extracted_answer = extract_answer_after_keyword(response_text, "Answer")
            
            # Score using exact match or numerical scoring
            scorer = ScoringStrategy.get_scorer("{scoring_method}")
            target_answer = str(row.get("{target_column}", ""))
            score = scorer(extracted_answer or "", target_answer)
            
            return self.create_single_eval_result(
                prompt_messages, response_text, score,
                target_answer, extracted_answer
            )
        
        # Use batch processing utility
        self.processed_results = self.batch_process_examples(
            self.examples, process_example, self.batch_size, 
            self.checkpoint_file, self.processed_results
        )
        
        return common.aggregate_results(self.processed_results)
'''

def get_drop_template() -> str:
    """Get DROP-style template"""
    return '''import random
from datasets import load_dataset
from typing import Optional

import common
from eval_types import Eval, EvalResult, SamplerBase, SingleEvalResult
from eval_utils import extract_answer_after_keyword, ScoringStrategy, EvaluationMixin

class {class_name}(Eval, EvaluationMixin):
    def __init__(self, dataset_name: str = "{dataset_name}",
                 num_examples: Optional[int] = None,
                 batch_size: int = 20,
                 checkpoint_file: Optional[str] = None,
                 prompt_template: Optional[str] = None):
        
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.checkpoint_file = checkpoint_file
        self.processed_results = []
        
        # Set default or custom prompt template
        self.prompt_template = prompt_template or "{prompt_template}"
        
        # Load dataset
        dataset = load_dataset(dataset_name{dataset_config_param}, split="test")
        if num_examples:
            self.examples = random.Random(0).sample(list(dataset), num_examples)
        else:
            self.examples = list(dataset)
        
        # Load checkpoint if exists
        if self.checkpoint_file:
            self.processed_results = common.load_checkpoint(self.checkpoint_file)
    
    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def process_example(row: dict) -> SingleEvalResult:
            # Format reading comprehension prompt with dynamic context extraction
            context = {context_column_extraction}
            question = row.get("{input_column}", "")
            
            # Use customizable prompt template with context, question and safe variable handling
            template_vars = dict(row)  # Start with all row data
            template_vars.update({{
                'question': question,
                'input': question,
                'context': context
            }})
            prompt_content = self.prompt_template.format(**template_vars)
            
            prompt_messages = self.create_prompt_messages(prompt_content, sampler)
            response_text = sampler(prompt_messages)
            extracted_answer = extract_answer_after_keyword(response_text, "Answer")
            
            # Score using appropriate method - default to exact match for reading comprehension
            scoring_method = "{scoring_method}"
            if scoring_method == "math_equivalence":
                # For math problems, try to use math_equivalence if available
                try:
                    from sampler.chat_completion_sampler import ChatCompletionSampler
                    equality_checker = ChatCompletionSampler(
                        model="meta/llama-4-maverick-17b-128e-instruct-maas",
                        base_url="https://us-east5-aiplatform.googleapis.com/v1/projects/{your-project-id}/locations/us-east5/endpoints/openapi"
                    )
                    scorer = ScoringStrategy.get_scorer("math_equivalence", equality_checker=equality_checker)
                except:
                    # Fall back to exact match if math_equivalence fails
                    scorer = ScoringStrategy.get_scorer("exact_match")
            else:
                scorer = ScoringStrategy.get_scorer(scoring_method if scoring_method != "math_equivalence" else "exact_match")
            
            target_answer = str(row.get("{target_column}", ""))
            score = scorer(extracted_answer or "", target_answer)
            
            return self.create_single_eval_result(
                prompt_messages, response_text, score,
                target_answer, extracted_answer
            )
        
        # Use batch processing utility
        self.processed_results = self.batch_process_examples(
            self.examples, process_example, self.batch_size, 
            self.checkpoint_file, self.processed_results
        )
        
        return common.aggregate_results(self.processed_results)
'''

def get_humaneval_template() -> str:
    """Get HumanEval-style template"""
    return '''import random
from datasets import load_dataset
from typing import Optional

import common
from eval_types import Eval, EvalResult, SamplerBase, SingleEvalResult
from eval_utils import extract_code_block, ScoringStrategy, EvaluationMixin

class {class_name}(Eval, EvaluationMixin):
    def __init__(self, dataset_name: str = "{dataset_name}",
                 num_examples: Optional[int] = None,
                 batch_size: int = 20,
                 checkpoint_file: Optional[str] = None,
                 prompt_template: Optional[str] = None):
        
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.checkpoint_file = checkpoint_file
        self.processed_results = []
        
        # Set default or custom prompt template
        self.prompt_template = prompt_template or "{prompt_template}"
        
        # Load dataset
        dataset = load_dataset(dataset_name{dataset_config_param}, split="test")
        if num_examples:
            self.examples = random.Random(0).sample(list(dataset), num_examples)
        else:
            self.examples = list(dataset)
        
        # Load checkpoint if exists
        if self.checkpoint_file:
            self.processed_results = common.load_checkpoint(self.checkpoint_file)
    
    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def process_example(row: dict) -> SingleEvalResult:
            # Use customizable prompt template with safe variable handling
            input_code = row.get("{input_column}", "")
            
            template_vars = dict(row)  # Start with all row data
            template_vars.update({{
                'question': input_code,
                'input': input_code,
                'code': input_code
            }})
            prompt_content = self.prompt_template.format(**template_vars)
            
            prompt_messages = self.create_prompt_messages(prompt_content, sampler)
            response_text = sampler(prompt_messages)
            code = extract_code_block(response_text, "python")
            
            # Simple code validation scoring
            scorer = ScoringStrategy.get_scorer("exact_match")
            target_code = str(row.get("{target_column}", ""))
            score = scorer(code or "", target_code)
            
            return self.create_single_eval_result(
                prompt_messages, response_text, score,
                target_code, code
            )
        
        # Use batch processing utility
        self.processed_results = self.batch_process_examples(
            self.examples, process_example, self.batch_size, 
            self.checkpoint_file, self.processed_results
        )
        
        return common.aggregate_results(self.processed_results)
'''

def get_simpleqa_template() -> str:
    """Get SimpleQA-style template"""
    return '''import random
from datasets import load_dataset
from typing import Optional

import common
from eval_types import Eval, EvalResult, SamplerBase, SingleEvalResult
from eval_utils import extract_answer_after_keyword, ScoringStrategy, EvaluationMixin

class {class_name}(Eval, EvaluationMixin):
    def __init__(self, dataset_name: str = "{dataset_name}",
                 num_examples: Optional[int] = None,
                 batch_size: int = 20,
                 checkpoint_file: Optional[str] = None,
                 prompt_template: Optional[str] = None):
        
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.checkpoint_file = checkpoint_file
        self.processed_results = []
        
        # Set default or custom prompt template
        self.prompt_template = prompt_template or "{prompt_template}"
        
        # Load dataset
        dataset = load_dataset(dataset_name{dataset_config_param}, split="test")
        if num_examples:
            self.examples = random.Random(0).sample(list(dataset), num_examples)
        else:
            self.examples = list(dataset)
        
        # Load checkpoint if exists
        if self.checkpoint_file:
            self.processed_results = common.load_checkpoint(self.checkpoint_file)
    
    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def process_example(row: dict) -> SingleEvalResult:
            question = row.get("{input_column}", "")
            # Use customizable prompt template with safe variable handling
            template_vars = dict(row)  # Start with all row data
            template_vars.update({{
                'question': question,
                'input': question
            }})
            prompt_content = self.prompt_template.format(**template_vars)
            
            prompt_messages = self.create_prompt_messages(prompt_content, sampler)
            response_text = sampler(prompt_messages)
            
            # Extract answer if using structured format
            extracted_answer = extract_answer_after_keyword(response_text, "Answer")
            if not extracted_answer:
                extracted_answer = response_text.strip()
            
            # Score using exact match by default
            scorer = ScoringStrategy.get_scorer("{scoring_method}")
            target_answer = str(row.get("{target_column}", ""))
            score = scorer(extracted_answer, target_answer)
            
            return self.create_single_eval_result(
                prompt_messages, response_text, score,
                target_answer, extracted_answer
            )
        
        # Use batch processing utility
        self.processed_results = self.batch_process_examples(
            self.examples, process_example, self.batch_size, 
            self.checkpoint_file, self.processed_results
        )
        
        return common.aggregate_results(self.processed_results)
'''

def get_template_map():
    """Get mapping of evaluation styles to template functions"""
    return {
        "multiple_choice_style": get_mmlu_template,
        "math_style": get_math_template,
        "reading_comprehension_style": get_drop_template,
        "code_generation_style": get_humaneval_template,
        "factual_qa_style": get_simpleqa_template,
        "text_generation_style": get_simpleqa_template,
        "classification_style": get_mmlu_template,
    } 