"""
Column Mapping Agent

This module contains the ColumnMappingAgent class responsible for intelligently
mapping dataset columns to template variables.
"""

import json
import re
from typing import Dict, Any, List
from eval_types import SamplerBase
from data_models import DatasetAnalysis


class ColumnMappingAgent:
    """Agent for intelligently mapping dataset columns to template variables"""
    
    def __init__(self, mapping_sampler: SamplerBase):
        self.mapping_sampler = mapping_sampler
    
    def resolve_template_variables(self, template_type: str, available_columns: List[str], 
                                 sample_data: List[Dict[str, Any]], analysis: DatasetAnalysis) -> Dict[str, Any]:
        """Resolve template variables by mapping available columns to template requirements"""
        
        # Define what each template type needs
        template_requirements = {
            "mmlu_style": ["question_column", "choices_column", "answer_column"],
            "math_style": ["input_column", "target_column"],
            "drop_style": ["context_column", "input_column", "target_column"],
            "humaneval_style": ["input_column", "target_column"],
            "simpleqa_style": ["input_column", "target_column"],
            "generic_style": ["input_column", "target_column"]
        }
        
        required_vars = template_requirements.get(template_type, ["input_column", "target_column"])
        
        mapping_prompt = f"""
Analyze this dataset and intelligently map its columns to the required template variables.

Dataset: {analysis.name}
Available Columns: {available_columns}
Template Type: {template_type}
Required Variables: {required_vars}

Sample Data (first 2 examples):
{json.dumps(sample_data[:2], indent=2)}

Task: Map available columns to required template variables. Consider:

1. **Semantic Meaning**: Match column content to variable purpose
2. **Multi-column Scenarios**: Some variables may need multiple columns combined
3. **Missing Columns**: Provide fallbacks for missing expected columns

For each required variable, determine:
- Which column(s) map to it
- If multiple columns should be combined, how to combine them
- Fallback strategy if the ideal column doesn't exist

Template Variable Requirements:
- **question_column/input_column**: The main question/prompt/input for the model
- **answer_column/target_column**: The correct answer/expected output
- **context_column**: Background text/passage for reading comprehension
- **choices_column**: Multiple choice options (could be single column or multiple A/B/C/D columns)

Respond with a JSON object:
{{
  "mappings": {{
    "question_column": {{"primary": "column_name", "fallback": "backup_column", "combine": ["col1", "col2"]}},
    "context_column": {{"primary": "context", "combine": ["pre_text", "post_text", "table"], "separator": "\\n\\n"}},
    ...
  }},
  "combination_logic": {{
    "context_column": "Combine pre_text + post_text + formatted table data",
    ...
  }},
  "validation_notes": ["Note about mapping quality", ...]
}}

For combine scenarios, specify how to join the columns (separator, formatting, etc.).
"""
        
        response = self.mapping_sampler([{"role": "user", "content": mapping_prompt}])
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                mapping_result = json.loads(json_match.group())
            else:
                mapping_result = self._fallback_mapping(template_type, available_columns, analysis)
        except json.JSONDecodeError:
            mapping_result = self._fallback_mapping(template_type, available_columns, analysis)
        
        # Validate and enhance the mapping
        validated_mapping = self._validate_and_enhance_mapping(
            mapping_result, available_columns, template_type, analysis
        )
        
        return validated_mapping
    
    def _fallback_mapping(self, template_type: str, available_columns: List[str], 
                         analysis: DatasetAnalysis) -> Dict[str, Any]:
        """Fallback mapping when LLM analysis fails"""
        
        # Basic column mapping using simple heuristics
        mappings = {}
        
        # Input column mapping
        input_candidates = ['question', 'prompt', 'input', 'text', 'query', 'problem']
        input_col = next((col for col in input_candidates if col in available_columns), 
                        analysis.input_columns[0] if analysis.input_columns else available_columns[0])
        
        # Target column mapping  
        target_candidates = ['answer', 'target', 'label', 'output', 'solution']
        target_col = next((col for col in target_candidates if col in available_columns),
                         analysis.target_columns[0] if analysis.target_columns else 'answer')
        
        mappings['input_column'] = {"primary": input_col}
        mappings['question_column'] = {"primary": input_col}
        mappings['target_column'] = {"primary": target_col}
        mappings['answer_column'] = {"primary": target_col}
        
        # Context column for reading comprehension
        if template_type == "drop_style":
            context_candidates = ['context', 'passage', 'text', 'paragraph']
            context_col = next((col for col in context_candidates if col in available_columns), 
                              'context')
            mappings['context_column'] = {"primary": context_col}
        
        # Choices column for multiple choice
        if template_type == "mmlu_style":
            choices_candidates = ['choices', 'options', 'alternatives']
            choices_col = next((col for col in choices_candidates if col in available_columns),
                              'choices')
            mappings['choices_column'] = {"primary": choices_col}
        
        return {
            "mappings": mappings,
            "combination_logic": {},
            "validation_notes": ["Using fallback heuristic mapping"]
        }
    
    def _validate_and_enhance_mapping(self, mapping_result: Dict[str, Any], 
                                    available_columns: List[str], template_type: str,
                                    analysis: DatasetAnalysis) -> Dict[str, Any]:
        """Validate and enhance the mapping with safety checks"""
        
        mappings = mapping_result.get("mappings", {})
        enhanced_mappings = {}
        
        for var_name, mapping_info in mappings.items():
            if isinstance(mapping_info, str):
                # Convert simple string mapping to dict format
                mapping_info = {"primary": mapping_info}
            
            enhanced_mapping = {}
            
            # Validate primary column exists
            primary = mapping_info.get("primary")
            if primary and primary in available_columns:
                enhanced_mapping["primary"] = primary
            else:
                # Find fallback
                fallback = mapping_info.get("fallback")
                if fallback and fallback in available_columns:
                    enhanced_mapping["primary"] = fallback
                else:
                    # Use first available column as last resort
                    enhanced_mapping["primary"] = available_columns[0] if available_columns else "unknown"
            
            # Handle column combinations
            combine_cols = mapping_info.get("combine", [])
            valid_combine_cols = [col for col in combine_cols if col in available_columns]
            if valid_combine_cols:
                enhanced_mapping["combine"] = valid_combine_cols
                enhanced_mapping["separator"] = mapping_info.get("separator", "\n\n")
            
            enhanced_mappings[var_name] = enhanced_mapping
        
        return {
            "mappings": enhanced_mappings,
            "combination_logic": mapping_result.get("combination_logic", {}),
            "validation_notes": mapping_result.get("validation_notes", [])
        }
    
    def generate_column_extraction_code(self, var_name: str, mapping_info: Dict[str, Any]) -> str:
        """Generate Python code to extract/combine columns for a template variable"""
        
        if "combine" in mapping_info:
            # Multi-column combination
            columns = mapping_info["combine"]
            separator = mapping_info.get("separator", "\n\n")
            
            code_parts = []
            for col in columns:
                code_parts.append(f'row.get("{col}", "")')
            
            if len(columns) > 1:
                # Filter out empty strings
                combination_code = f'"{separator}".join([part for part in [{", ".join(code_parts)}] if part.strip()])'
            else:
                combination_code = code_parts[0]
            
            return combination_code
        else:
            # Single column
            primary_col = mapping_info.get("primary", "unknown")
            return f'row.get("{primary_col}", "")' 