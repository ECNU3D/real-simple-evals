import random
from datasets import load_dataset
from typing import Optional

import common
from eval_types import Eval, EvalResult, SamplerBase, SingleEvalResult
from eval_utils import extract_answer_after_keyword, ScoringStrategy, EvaluationMixin

class FinqaEval(Eval, EvaluationMixin):
    def __init__(self, dataset_name: str = "dreamerdeo/finqa",
                 num_examples: Optional[int] = None,
                 batch_size: int = 20,
                 checkpoint_file: Optional[str] = None):
        
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.checkpoint_file = checkpoint_file
        self.processed_results = []
        
        # Load dataset
        dataset = load_dataset(dataset_name, split="test")
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
            context = "\n\n".join([part for part in ["Pre-text: " + str(row.get("pre_text", "")), "Post-text: " + str(row.get("post_text", "")), "Table: " + str(row.get("table", ""))] if part.strip() and not part.endswith(": ")])
            question = row.get("question", "")
            
            prompt_content = f"""Read the following passage and answer the question.

Passage: """ + context + """

Question: """ + question + """

Think step by step, then write a line of the form "Answer: $ANSWER" at the end of your response."""
            
            prompt_messages = self.create_prompt_messages(prompt_content, sampler)
            response_text = sampler(prompt_messages)
            extracted_answer = extract_answer_after_keyword(response_text, "Answer")
            
            # Score using appropriate method - default to exact match for reading comprehension
            scoring_method = "math_equivalence"
            if scoring_method == "math_equivalence":
                # For math problems, try to use math_equivalence if available
                try:
                    from sampler.chat_completion_sampler import ChatCompletionSampler
                    equality_checker = ChatCompletionSampler(
                        model="meta/llama-4-maverick-17b-128e-instruct-maas",
                        base_url="https://us-east5-aiplatform.googleapis.com/v1/projects/{your-project-id}/locations/us-east5/endpoints/openapi"
                    )
                    scorer = ScoringStrategy.get_scorer("math_equivalence", equality_checker=equality_checker)
                    print(f"Using math_equivalence scorer: {scorer}")
                except:
                    # Fall back to exact match if math_equivalence fails
                    scorer = ScoringStrategy.get_scorer("exact_match")
            else:
                scorer = ScoringStrategy.get_scorer(scoring_method if scoring_method != "math_equivalence" else "exact_match")
            
            target_answer = str(row.get("answer", ""))
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
