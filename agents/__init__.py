"""
Agents package for the Agentic Evaluation Builder.

This package contains all the AI agents responsible for different aspects
of evaluation generation.
"""

from .dataset_analyzer import DatasetAnalyzer
from .task_detector import TaskTypeDetector
from .prompt_engineer import PromptEngineer
from .scoring_agent import ScoringStrategyAgent
from .column_mapper import ColumnMappingAgent
from .code_generator import CodeGenerator

__all__ = [
    "DatasetAnalyzer",
    "TaskTypeDetector", 
    "PromptEngineer",
    "ScoringStrategyAgent",
    "ColumnMappingAgent",
    "CodeGenerator"
] 