"""
Code Generator Agent

This module contains the CodeGenerator class responsible for generating
the final evaluation code using various templates.
"""

import ast
import re
from typing import Dict, Any, Optional, Tuple

from eval_types import SamplerBase
from data_models import DatasetAnalysis, EvalConfig, CodeValidationResult
from templates.evaluation_templates import get_template_map, get_generic_template

# Valid imports that generated code can use
ALLOWED_IMPORTS = {
    'random', 're', 'json', 'pathlib', 'urllib.request', 'gzip', 'string', 
    'pandas', 'numpy', 'datasets', 'typing', 'dataclasses',
    'common', 'eval_types', 'eval_utils', 'scipy.optimize', 'sampler'
}


class CodeGenerator:
    """Agent for generating the final evaluation code"""
    
    def __init__(self, code_sampler: SamplerBase):
        self.code_sampler = code_sampler
    
    def generate_eval_class(self, analysis: DatasetAnalysis, config: EvalConfig, column_mapping: Optional[Dict[str, Any]] = None) -> str:
        """Generate the complete evaluation class code with validation"""
        
        # Determine the appropriate template based on evaluation style
        eval_style = config.task_type + "_style"
        template_map = get_template_map()
        
        # Get the appropriate template
        base_template = template_map.get(eval_style, get_generic_template)()
        
        # Generate class name (sanitized)
        dataset_clean = analysis.name.split('/')[-1].replace('-', '_').replace('.', '_')
        dataset_clean = re.sub(r'[^a-zA-Z0-9_]', '_', dataset_clean)
        class_name = f"{dataset_clean.title()}Eval"
        
        # Use enhanced column mapping if provided
        if column_mapping:
            input_column, target_column, additional_columns = self._get_mapped_columns(analysis, column_mapping)
        else:
            # Fallback to old method
            input_column, target_column, additional_columns = self._get_validated_columns(analysis)
        
        # Prepare dataset config parameters
        dataset_config_param, dataset_config_comment = self._get_dataset_config_formatting(analysis)
        
        # Use custom prompt template from config if provided, otherwise use default
        if config.prompt_template:
            prompt_template_value = config.prompt_template
        else:
            prompt_template_value = self._get_default_prompt_template(config.task_type, {})
        
        # Format the template with actual values first, excluding prompt_template
        try:
            # Format everything except prompt_template first
            partial_formatted_code = base_template.format(
                class_name=class_name,
                dataset_name=analysis.name,
                input_column=input_column,
                target_column=target_column,
                scoring_method=config.scoring_method,
                dataset_config_param=dataset_config_param,
                dataset_config_comment=dataset_config_comment,
                prompt_template="PLACEHOLDER_FOR_PROMPT_TEMPLATE",
                **additional_columns
            )
            
            # Now safely replace the placeholder with the actual prompt template
            formatted_code = partial_formatted_code.replace(
                '"PLACEHOLDER_FOR_PROMPT_TEMPLATE"', 
                repr(prompt_template_value)
            )
        except KeyError as e:
            print(f"⚠️  Template formatting error: {e}")
            # Fallback to generic template
            formatted_code = self._generate_generic_eval(analysis, config, class_name, column_mapping)
        
        # Validate the generated code
        validation_result = self._validate_generated_code(formatted_code)
        
        if not validation_result.is_valid:
            print(f"⚠️  Generated code validation failed: {validation_result.errors}")
            if validation_result.fixed_code:
                print("✅ Using automatically fixed code")
                formatted_code = validation_result.fixed_code
            else:
                print("🔄 Falling back to generic template")
                formatted_code = self._generate_generic_eval(analysis, config, class_name, column_mapping)
                # Validate the fallback too
                fallback_validation = self._validate_generated_code(formatted_code)
                if not fallback_validation.is_valid:
                    raise ValueError(f"Even fallback code is invalid: {fallback_validation.errors}")
        
        return formatted_code
    
    def _get_validated_columns(self, analysis: DatasetAnalysis) -> Tuple[str, str, Dict[str, str]]:
        """Get validated column mappings with fallbacks"""
        
        # Primary columns
        input_column = analysis.input_columns[0] if analysis.input_columns else "question"
        target_column = analysis.target_columns[0] if analysis.target_columns else "answer"
        
        # Additional columns that might be needed
        additional_columns = {}
        
        # Try to find context column for reading comprehension
        context_candidates = ['context', 'passage', 'text', 'paragraph']
        for candidate in context_candidates:
            if candidate in analysis.columns:
                additional_columns['context_column'] = candidate
                break
        additional_columns.setdefault('context_column', 'context')
        
        # Try to find choices column for multiple choice
        choices_candidates = ['choices', 'options', 'alternatives']
        for candidate in choices_candidates:
            if candidate in analysis.columns:
                additional_columns['choices_column'] = candidate
                break
        additional_columns.setdefault('choices_column', 'choices')
        
        # Add other common mappings
        additional_columns.update({
            'question_column': input_column,
            'prompt_column': input_column,
            'answer_column': target_column,
        })
        
        return input_column, target_column, additional_columns
    
    def _validate_generated_code(self, code: str) -> CodeValidationResult:
        """Validate the generated code for syntax and import errors"""
        
        errors = []
        warnings = []
        
        # Check syntax
        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Syntax error: {e}")
            return CodeValidationResult(False, errors, warnings)
        
        # Check imports
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in ALLOWED_IMPORTS:
                        errors.append(f"Invalid import: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module not in ALLOWED_IMPORTS:
                    # Allow specific submodules
                    base_module = node.module.split('.')[0]
                    if base_module not in ALLOWED_IMPORTS:
                        errors.append(f"Invalid import: {node.module}")
        
        # Check for required patterns
        required_patterns = [
            r'class \w+Eval\(Eval.*\):',  # Must inherit from Eval
            r'def __init__\(self',         # Must have __init__
            r'def __call__\(self, sampler: SamplerBase\)',  # Must have __call__
        ]
        
        for pattern in required_patterns:
            if not re.search(pattern, code):
                errors.append(f"Missing required pattern: {pattern}")
        
        # Check for common mistakes
        if 'from eval_utils import' in code:
            # Verify imports exist
            import_match = re.search(r'from eval_utils import ([^\n]+)', code)
            if import_match:
                imports = [imp.strip() for imp in import_match.group(1).split(',')]
                valid_utils = [
                    'extract_answer_after_keyword', 'extract_multiple_choice_answer',
                    'ScoringStrategy', 'EvaluationMixin', 'extract_code_block',
                    'exact_match_score', 'fuzzy_match_score', 'numerical_score'
                ]
                for imp in imports:
                    if imp not in valid_utils:
                        warnings.append(f"Potentially invalid eval_utils import: {imp}")
        
        is_valid = len(errors) == 0
        return CodeValidationResult(is_valid, errors, warnings)
    
    def _generate_generic_eval(self, analysis: DatasetAnalysis, config: EvalConfig, class_name: str, column_mapping: Optional[Dict[str, Any]] = None) -> str:
        """Generate a safe, generic evaluation when templates fail"""
        
        input_column = analysis.input_columns[0] if analysis.input_columns else "question"
        target_column = analysis.target_columns[0] if analysis.target_columns else "answer"
        
        # Determine safe scoring method
        safe_scoring_method = "exact_match"
        if config.scoring_method in ["exact_match", "fuzzy_match", "multiple_choice", "numerical"]:
            safe_scoring_method = config.scoring_method
        
        # Get dataset config formatting
        dataset_config_param, dataset_config_comment = self._get_dataset_config_formatting(analysis)
        
        # Use custom prompt template from config if provided, otherwise use default
        if config.prompt_template:
            prompt_template_value = config.prompt_template
        else:
            prompt_template_value = self._get_default_prompt_template(config.task_type, {})
        
        # Generate the base template first with a placeholder
        template_code = f'''import random
from datasets import load_dataset
from typing import Optional

import common
from eval_types import Eval, EvalResult, SamplerBase, SingleEvalResult
from eval_utils import extract_answer_after_keyword, ScoringStrategy, EvaluationMixin

class {class_name}(Eval, EvaluationMixin):
    def __init__(self, dataset_name: str = "{analysis.name}",
                 num_examples: Optional[int] = None,
                 batch_size: int = 20,
                 checkpoint_file: Optional[str] = None,
                 prompt_template: Optional[str] = None):
        
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.checkpoint_file = checkpoint_file
        self.processed_results = []
        
        # Set default or custom prompt template
        self.prompt_template = prompt_template or "PLACEHOLDER_FOR_PROMPT_TEMPLATE"
        
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
            
            # Score using selected method
            scorer = ScoringStrategy.get_scorer("{safe_scoring_method}")
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
        
        # Now safely replace the placeholder with the actual prompt template
        final_code = template_code.replace(
            '"PLACEHOLDER_FOR_PROMPT_TEMPLATE"',
            repr(prompt_template_value)
        )
        
        return final_code

    def _get_mapped_columns(self, analysis: DatasetAnalysis, column_mapping: Dict[str, Any]) -> Tuple[str, str, Dict[str, str]]:
        """Get column mappings using the intelligent column mapping result"""
        
        mappings = column_mapping.get("mappings", {})
        
        # Extract primary columns for input and target
        input_mapping = mappings.get("input_column") or mappings.get("question_column")
        target_mapping = mappings.get("target_column") or mappings.get("answer_column")
        
        input_column = input_mapping.get("primary") if input_mapping else "question"
        target_column = target_mapping.get("primary") if target_mapping else "answer"
        
        # Build additional columns dict for template formatting
        additional_columns = {}
        
        for var_name, mapping_info in mappings.items():
            if "combine" in mapping_info:
                # For multi-column scenarios, generate the extraction code
                additional_columns[var_name] = mapping_info.get("primary", "unknown")
                # Store the extraction logic for later use in template
                additional_columns[f"{var_name}_extraction"] = self._generate_context_extraction_logic(mapping_info)
            else:
                # Only add if not already covered by input_column/target_column
                if var_name not in ["input_column", "target_column"]:
                    additional_columns[var_name] = mapping_info.get("primary", "unknown")
        
        # Add standard template variables (only if not already present)
        template_vars = {
            "context_column": "unknown",
            "choices_column": "unknown", 
            "question_column": input_column,
            "prompt_column": input_column,
            "answer_column": target_column,
        }
        
        for var_name, default_value in template_vars.items():
            if var_name not in additional_columns:
                additional_columns[var_name] = default_value
        
        # Ensure extraction logic exists for context
        if "context_column_extraction" not in additional_columns:
            context_col = additional_columns.get("context_column", "context")
            additional_columns["context_column_extraction"] = f'row.get("{context_col}", "")'
        
        return input_column, target_column, additional_columns
    
    def _generate_context_extraction_logic(self, mapping_info: Dict[str, Any]) -> str:
        """Generate context extraction logic for multi-column scenarios"""
        
        if "combine" not in mapping_info:
            return f'str(row.get("{mapping_info.get("primary", "context")}", ""))'
        
        columns = mapping_info["combine"]
        separator = mapping_info.get("separator", "\\n\\n")
        
        # Escape the separator for use in generated code
        if separator == "\n\n":
            separator = "\\n\\n"
        elif separator == "\n":
            separator = "\\n"
        
        # Generate descriptive context building with proper type handling
        parts = []
        for col in columns:
            # Add descriptive labels based on column names, with str() conversion for safety
            if "pre" in col.lower():
                parts.append(f'"Pre-text: " + str(row.get("{col}", ""))')
            elif "post" in col.lower():
                parts.append(f'"Post-text: " + str(row.get("{col}", ""))')
            elif "table" in col.lower():
                parts.append(f'"Table: " + str(row.get("{col}", ""))')
            elif "context" in col.lower():
                parts.append(f'str(row.get("{col}", ""))')
            else:
                parts.append(f'"{col.title()}: " + str(row.get("{col}", ""))')
        
        if len(parts) > 1:
            return f'"{separator}".join([part for part in [{", ".join(parts)}] if part.strip() and not part.endswith(": ")])'
        else:
            return parts[0] if parts else 'str(row.get("context", ""))' 

    def _get_dataset_config_formatting(self, analysis: DatasetAnalysis) -> Tuple[str, str]:
        """Get dataset config parameters and comments for template formatting"""
        
        if analysis.subset and analysis.requires_config:
            # Dataset requires a config
            dataset_config_param = f', "{analysis.subset}"'
            dataset_config_comment = f" with config '{analysis.subset}'"
        elif analysis.subset and not analysis.requires_config:
            # Dataset has a subset but doesn't require it
            dataset_config_param = f', "{analysis.subset}"'
            dataset_config_comment = f" with subset '{analysis.subset}'"
        else:
            # No config needed
            dataset_config_param = ""
            dataset_config_comment = ""
        
        return dataset_config_param, dataset_config_comment 

    def _get_default_prompt_template(self, task_type: str, additional_columns: Dict[str, str]) -> str:
        """Get default prompt template based on task type"""
        
        if task_type == "multiple_choice":
            return "Choose the best answer for the following question:\\n\\n{question}\\n\\nAnswer with just the letter (A, B, C, or D):"
        
        elif task_type == "math":
            return "Solve the following math problem step by step. The last line of your response should be of the form Answer: $ANSWER where $ANSWER is the answer to the problem.\\n\\n{question}"
        
        elif task_type == "reading_comprehension":
            return 'Read the following passage and answer the question.\\n\\nPassage: {context}\\n\\nQuestion: {question}\\n\\nThink step by step, then write a line of the form "Answer: $ANSWER" at the end of your response.'
        
        elif task_type == "code_generation":
            return "Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.\\n\\n{input}"
        
        elif task_type == "factual_qa":
            return "Answer the following question directly and concisely.\\n\\nQuestion: {question}\\n\\nAnswer:"
        
        elif task_type == "text_generation":
            return "Answer the following question directly and concisely.\\n\\nQuestion: {question}\\n\\nAnswer:"
        
        elif task_type == "classification":
            return "Choose the best answer for the following question:\\n\\n{question}\\n\\nAnswer with just the letter (A, B, C, or D):"
        
        else:
            # Generic fallback
            return "Question: {question}\\n\\nAnswer:"