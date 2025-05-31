import random
from datasets import load_dataset
from typing import Optional

import common
from eval_types import Eval, EvalResult, SamplerBase, SingleEvalResult
from eval_utils import extract_answer_after_keyword, ScoringStrategy, EvaluationMixin

class Rag_Mini_WikipediaEval(Eval, EvaluationMixin):
    def __init__(self, dataset_name: str = "rag-datasets/rag-mini-wikipedia",
                 num_examples: Optional[int] = None,
                 batch_size: int = 20,
                 checkpoint_file: Optional[str] = None,
                 prompt_template: Optional[str] = None):
        
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.checkpoint_file = checkpoint_file
        self.processed_results = []
        
        # Set default or custom prompt template
        self.prompt_template = prompt_template or 'Current template: Answer the question with a simple "yes" or "no" based on your knowledge. Do not return anything else other than answer.\\nQuestion: {question}\\nAnswer:'
        
        # Load dataset
        dataset = load_dataset(dataset_name, "question-answer", split="test")
        if num_examples:
            self.examples = random.Random(0).sample(list(dataset), num_examples)
        else:
            self.examples = list(dataset)
        
        # Load checkpoint if exists
        if self.checkpoint_file:
            self.processed_results = common.load_checkpoint(self.checkpoint_file)
    
    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def process_example(row: dict) -> SingleEvalResult:
            question = row.get("question", "")
            # Use customizable prompt template with safe variable handling
            template_vars = dict(row)  # Start with all row data
            template_vars.update({
                'question': question,
                'input': question
            })
            prompt_content = self.prompt_template.format(**template_vars)
            
            prompt_messages = self.create_prompt_messages(prompt_content, sampler)
            response_text = sampler(prompt_messages)
            
            # Extract answer if using structured format
            extracted_answer = extract_answer_after_keyword(response_text, "Answer")
            if not extracted_answer:
                extracted_answer = response_text.strip()
            
            # Score using exact match by default
            scorer = ScoringStrategy.get_scorer("exact_match")
            target_answer = str(row.get("answer", ""))
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
