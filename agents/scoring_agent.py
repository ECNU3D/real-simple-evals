"""
Scoring Strategy Agent

This module contains the ScoringStrategyAgent class responsible for determining
how to score the evaluation responses.
"""

import json
import re
from typing import Tuple
from eval_types import SamplerBase
from data_models import DatasetAnalysis


class ScoringStrategyAgent:
    """Agent for determining how to score the evaluation responses"""
    
    def __init__(self, scoring_sampler: SamplerBase):
        self.scoring_sampler = scoring_sampler
    
    def determine_scoring_method(self, analysis: DatasetAnalysis, eval_style: str) -> Tuple[str, str]:
        """Determine scoring method and answer pattern"""
        
        scoring_prompt = f"""
Determine the appropriate scoring method and answer extraction pattern for this evaluation:

Dataset: {analysis.name}
Task Type: {analysis.task_type}
Evaluation Style: {eval_style}
Target Columns: {analysis.target_columns}

Sample Data:
{json.dumps(analysis.sample_data[0] if analysis.sample_data else {}, indent=2)}

Respond with a JSON object containing:
1. "scoring_method": One of ["exact_match", "fuzzy_match", "multiple_choice", "code_execution", "llm_grader", "math_equivalence"]
2. "answer_pattern": Regex pattern to extract the answer from model response
3. "explanation": Brief explanation of the choice

Consider:
- exact_match: For precise string matching
- fuzzy_match: For flexible text matching
- multiple_choice: For A/B/C/D selections
- code_execution: For code that needs to be run
- llm_grader: For complex answers requiring LLM judgment
- math_equivalence: For mathematical expressions

Respond with only the JSON object.
"""
        
        response = self.scoring_sampler([{"role": "user", "content": scoring_prompt}])
        
        try:
            result = json.loads(re.search(r'\{.*\}', response, re.DOTALL).group())
            return result.get("scoring_method", "exact_match"), result.get("answer_pattern", r"Answer\s*:\s*([^\n]+)")
        except:
            # Fallback
            if analysis.task_type == "multiple_choice":
                return "multiple_choice", r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?"
            elif analysis.task_type == "math":
                return "math_equivalence", r"(?i)Answer\s*:\s*([^\n]+)"
            elif analysis.task_type == "code_generation":
                return "code_execution", r"```python\n(.*?)```"
            else:
                return "exact_match", r"(?i)Answer\s*:\s*([^\n]+)" 