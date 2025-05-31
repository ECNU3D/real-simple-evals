"""
Data models for the Agentic Evaluation Builder system.

This module contains all the dataclass definitions used throughout the system.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass

@dataclass
class DatasetAnalysis:
    """Structured analysis of a HuggingFace dataset"""
    name: str
    num_examples: int
    columns: List[str]
    sample_data: List[Dict[str, Any]]
    task_type: str  # e.g., "multiple_choice", "text_generation", "classification"
    input_columns: List[str]
    target_columns: List[str]
    metadata: Dict[str, Any]
    subset: Optional[str] = None  # Dataset config/subset name
    requires_config: bool = False  # Whether dataset requires a config

@dataclass
class EvalConfig:
    """Configuration for generating an evaluation"""
    task_type: str
    prompt_template: str
    scoring_method: str
    answer_pattern: str
    input_mapping: Dict[str, str]
    target_mapping: Dict[str, str]
    special_instructions: List[str]

@dataclass
class CodeValidationResult:
    """Result of code validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    fixed_code: Optional[str] = None 