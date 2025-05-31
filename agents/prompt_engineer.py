"""
Prompt Engineer Agent

This module contains the PromptEngineer class responsible for generating
appropriate prompts for the evaluation.
"""

import json
from eval_types import SamplerBase
from data_models import DatasetAnalysis


class PromptEngineer:
    """Agent for generating appropriate prompts for the evaluation"""
    
    def __init__(self, prompt_sampler: SamplerBase):
        self.prompt_sampler = prompt_sampler
    
    def generate_prompt_template(self, analysis: DatasetAnalysis, eval_style: str) -> str:
        """Generate an appropriate prompt template for the evaluation"""
        
        sample_data_str = "\n".join([f"Example {i+1}: {json.dumps(sample, indent=2)}" 
                                    for i, sample in enumerate(analysis.sample_data[:3])])
        
        prompt_engineering_prompt = f"""
Create a prompt template for evaluating a language model on the following dataset:

Dataset: {analysis.name}
Task Type: {analysis.task_type}
Evaluation Style: {eval_style}
Input Columns: {analysis.input_columns}
Target Columns: {analysis.target_columns}

Sample Data:
{sample_data_str}

Generate a prompt template that:
1. Clearly instructs the model on the task
2. Includes placeholders for dynamic content (use {{column_name}} format)
3. Specifies the expected output format
4. Is appropriate for the task type

Based on the evaluation style:
- mmlu_style: Multiple choice format with A/B/C/D answers
- math_style: Step-by-step solving with "Answer: X" format
- drop_style: Reading comprehension with "Answer: X" format
- humaneval_style: Code generation in markdown code blocks
- simpleqa_style: Direct factual answers
- generic_style: Flexible format based on task

Respond with just the prompt template, no additional explanation.
"""
        
        response = self.prompt_sampler([{"role": "user", "content": prompt_engineering_prompt}])
        return response.strip() 