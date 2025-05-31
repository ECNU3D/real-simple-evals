"""
Templates package for the Agentic Evaluation Builder.

This package contains all the code templates used to generate evaluations
for different types of tasks.
"""

from .evaluation_templates import (
    get_generic_template,
    get_mmlu_template, 
    get_math_template,
    get_drop_template,
    get_humaneval_template,
    get_simpleqa_template,
    get_template_map
)

__all__ = [
    "get_generic_template",
    "get_mmlu_template",
    "get_math_template", 
    "get_drop_template",
    "get_humaneval_template",
    "get_simpleqa_template",
    "get_template_map"
] 