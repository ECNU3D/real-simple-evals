"""
Agentic Evaluation Builder: A system that analyzes HuggingFace datasets and automatically
generates custom evaluation suites compatible with the simple-evals framework.

This system consists of multiple AI agents that work together to:
1. Analyze dataset structure and content
2. Determine the appropriate evaluation type
3. Generate prompts and scoring logic
4. Create the final evaluation code

Key improvements in this version:
- Better validation of generated code
- More robust error handling
- Consistent patterns following existing evaluations
- Import validation and safety checks
- Modular architecture with separate agent classes
"""

import pathlib
import re
from typing import Optional, Tuple, Dict, Any

from eval_types import SamplerBase
from data_models import DatasetAnalysis, EvalConfig
from agents import (
    DatasetAnalyzer,
    TaskTypeDetector,
    PromptEngineer,
    ScoringStrategyAgent,
    ColumnMappingAgent,
    CodeGenerator
)


class AgenticEvalBuilder:
    """Main orchestrator for the agentic evaluation building system"""
    
    def __init__(self, 
                 analyzer_sampler: SamplerBase,
                 detector_sampler: SamplerBase,
                 prompt_sampler: SamplerBase,
                 scoring_sampler: SamplerBase,
                 code_sampler: SamplerBase,
                 mapping_sampler: SamplerBase):
        
        self.analyzer = DatasetAnalyzer(analyzer_sampler)
        self.detector = TaskTypeDetector(detector_sampler)
        self.prompt_engineer = PromptEngineer(prompt_sampler)
        self.scoring_agent = ScoringStrategyAgent(scoring_sampler)
        self.column_mapper = ColumnMappingAgent(mapping_sampler)
        self.code_generator = CodeGenerator(code_sampler)
    
    def build_eval_from_dataset(self, dataset_name: str, subset: Optional[str] = None,
                               split: str = "test", interactive: bool = True) -> Tuple[str, str]:
        """
        Build an evaluation from a HuggingFace dataset
        
        Returns:
            Tuple of (generated_code, eval_name)
        """
        
        print(f"🔍 Analyzing dataset: {dataset_name}")
        if subset:
            print(f"   Using subset: {subset}")
        try:
            analysis = self.analyzer.analyze_dataset(dataset_name, subset, split)
        except Exception as e:
            print(f"❌ Failed to analyze dataset: {e}")
            raise
        
        print(f"📊 Dataset Analysis:")
        print(f"   - Task Type: {analysis.task_type}")
        print(f"   - Columns: {analysis.columns}")
        print(f"   - Input Columns: {analysis.input_columns}")
        print(f"   - Target Columns: {analysis.target_columns}")
        print(f"   - Sample Count: {analysis.num_examples}")
        if analysis.subset:
            print(f"   - Config/Subset: {analysis.subset}")
        if analysis.requires_config:
            print(f"   - Requires Config: Yes")
        
        # Validate analysis quality
        if not analysis.input_columns or not analysis.target_columns:
            print("⚠️  Warning: Could not identify input/target columns clearly")
            if interactive:
                analysis = self._interactive_modify_analysis(analysis)
        
        # Interactive modification if requested
        if interactive:
            modify = input("🤔 Would you like to modify the analysis? (y/n): ").lower().strip()
            if modify == 'y':
                analysis = self._interactive_modify_analysis(analysis)
        
        print(f"🎯 Determining evaluation approach for task type: {analysis.task_type}")
        eval_style = self.detector.determine_eval_approach(analysis)
        print(f"   → Using {eval_style} approach")
        
        print(f"🗂️  Mapping dataset columns to template variables...")
        try:
            column_mapping = self.column_mapper.resolve_template_variables(
                eval_style, analysis.columns, analysis.sample_data, analysis
            )
            print(f"   → Column mapping completed")
            if column_mapping.get("validation_notes"):
                for note in column_mapping["validation_notes"]:
                    print(f"     • {note}")
        except Exception as e:
            print(f"⚠️  Column mapping failed: {e}, using fallback")
            column_mapping = self.column_mapper._fallback_mapping(eval_style, analysis.columns, analysis)
        
        print(f"✍️  Generating prompts...")
        try:
            prompt_template = self.prompt_engineer.generate_prompt_template(analysis, eval_style)
            print(f"   → Generated prompt template")
            print(f"   → Prompt template: {prompt_template}")
        except Exception as e:
            print(f"⚠️  Prompt generation failed: {e}, using default")
            prompt_template = "Question: {question}\\nAnswer:"
        
        # Interactive prompt modification if requested
        if interactive:
            modify_prompt = input("🤔 Would you like to modify the prompt template? (y/n): ").lower().strip()
            if modify_prompt == 'y':
                prompt_template = self._interactive_modify_prompt(prompt_template, analysis, eval_style)
        
        print(f"⚖️  Determining scoring strategy...")
        try:
            scoring_method, answer_pattern = self.scoring_agent.determine_scoring_method(analysis, eval_style)
            print(f"   → Selected {scoring_method} scoring method")
        except Exception as e:
            print(f"⚠️  Scoring determination failed: {e}, using exact match")
            scoring_method, answer_pattern = "exact_match", r"Answer:\\s*(.+)"
        
        # Interactive scoring modification if requested
        if interactive:
            modify_scoring = input("🤔 Would you like to modify the scoring strategy? (y/n): ").lower().strip()
            if modify_scoring == 'y':
                scoring_method, answer_pattern = self._interactive_modify_scoring(scoring_method, answer_pattern, analysis, eval_style)

        print(f"   → Using {scoring_method} scoring")
        
        # Create evaluation configuration with column mapping
        config = EvalConfig(
            task_type=analysis.task_type,
            prompt_template=prompt_template,
            scoring_method=scoring_method,
            answer_pattern=answer_pattern,
            input_mapping={"input": analysis.input_columns[0] if analysis.input_columns else "question"},
            target_mapping={"target": analysis.target_columns[0] if analysis.target_columns else "answer"},
            special_instructions=[]
        )
        
        print(f"🔧 Generating evaluation code...")
        try:
            generated_code = self.code_generator.generate_eval_class(analysis, config, column_mapping)
        except Exception as e:
            print(f"❌ Code generation failed: {e}")
            raise
        
        # Extract class name for registration
        class_name = self._extract_class_name(generated_code)
        eval_name = class_name.replace("Eval", "").lower()
        
        print(f"✅ Successfully generated {class_name}")
        print(f"📁 Evaluation name: {eval_name}")
        
        return generated_code, eval_name
    
    def build_and_save_eval(self, dataset_name: str, subset: Optional[str] = None,
                           split: str = "test", interactive: bool = True) -> str:
        """Build and automatically save an evaluation"""
        
        generated_code, eval_name = self.build_eval_from_dataset(
            dataset_name, subset, split, interactive
        )
        
        # Get the analysis to include config info in metadata
        analysis = self.analyzer.analyze_dataset(dataset_name, subset, split)
        
        # Prepare metadata for registration
        metadata = {
            "dataset_name": dataset_name,
            "dataset_subset": subset,
            "dataset_split": split,
            "dataset_config": analysis.subset,
            "requires_config": analysis.requires_config,
            "task_type": analysis.task_type,
            "description": f"Automatically generated evaluation for {dataset_name}" + (f" (config: {analysis.subset})" if analysis.subset else ""),
            "created_by": "agentic_eval_builder"
        }
        
        # Save the evaluation
        file_path = self.save_evaluation(generated_code, eval_name, metadata)
        print(f"💾 Saved evaluation to: {file_path}")
        
        return file_path
    
    def _interactive_modify_analysis(self, analysis: DatasetAnalysis) -> DatasetAnalysis:
        """Allow user to interactively modify the analysis"""
        
        print("📝 Current Analysis:")
        print(f"   Task Type: {analysis.task_type}")
        print(f"   Input Columns: {analysis.input_columns}")
        print(f"   Target Columns: {analysis.target_columns}")
        print(f"   Available Columns: {analysis.columns}")
        
        # Allow task type modification
        new_task_type = input(f"Task type [{analysis.task_type}]: ").strip()
        if new_task_type:
            analysis.task_type = new_task_type
        
        # Allow input column modification
        new_input = input(f"Input columns {analysis.input_columns}: ").strip()
        if new_input:
            analysis.input_columns = [col.strip() for col in new_input.split(',')]
        
        # Allow target column modification
        new_target = input(f"Target columns {analysis.target_columns}: ").strip()
        if new_target:
            analysis.target_columns = [col.strip() for col in new_target.split(',')]
        
        return analysis
    
    def _interactive_modify_prompt(self, prompt_template: str, analysis: DatasetAnalysis, eval_style: str) -> str:
        """Allow user to interactively modify the prompt template"""
        
        print("📝 Current Prompt Template:")
        print(f"   {repr(prompt_template)}")
        print(f"\n💡 Available variables from dataset:")
        print(f"   - Input columns: {analysis.input_columns}")
        print(f"   - Target columns: {analysis.target_columns}")
        print(f"   - All columns: {analysis.columns}")
        print(f"\n📋 Evaluation style: {eval_style}")
        
        print("\n🔧 Modification options:")
        print("1. Edit the template directly")
        print("2. Use a predefined template pattern")
        print("3. Keep current template")
        
        choice = input("Choose option (1-3): ").strip()
        
        if choice == "1":
            print(f"\nCurrent template: {prompt_template}")
            new_template = input("Enter new prompt template: ").strip()
            if new_template:
                prompt_template = new_template
                print(f"✅ Updated prompt template")
        
        elif choice == "2":
            print("\n📚 Predefined templates:")
            templates = {
                "1": "Question: {question}\nAnswer:",
                "2": "Context: {context}\nQuestion: {question}\nAnswer:",
                "3": "Passage: {passage}\nQuestion: {question}\nOptions:\nA) {option_a}\nB) {option_b}\nC) {option_c}\nD) {option_d}\nAnswer:",
                "4": "Text: {text}\nClassify this text.\nCategory:",
                "5": "Problem: {problem}\nSolution:"
            }
            
            for key, template in templates.items():
                print(f"   {key}. {template}")
            
            template_choice = input("Select template (1-5): ").strip()
            if template_choice in templates:
                prompt_template = templates[template_choice]
                print(f"✅ Selected predefined template")
        
        print(f"\n✅ Final prompt template: {repr(prompt_template)}")
        return prompt_template
    
    def _interactive_modify_scoring(self, scoring_method: str, answer_pattern: str, 
                                   analysis: DatasetAnalysis, eval_style: str) -> Tuple[str, str]:
        """Allow user to interactively modify the scoring strategy"""
        
        print("📝 Current Scoring Configuration:")
        print(f"   Method: {scoring_method}")
        print(f"   Answer Pattern: {repr(answer_pattern)}")
        print(f"\n📋 Task type: {analysis.task_type}")
        print(f"📋 Evaluation style: {eval_style}")
        
        print("\n🔧 Modification options:")
        print("1. Change scoring method")
        print("2. Modify answer extraction pattern")
        print("3. Change both method and pattern")
        print("4. Keep current configuration")
        
        choice = input("Choose option (1-4): ").strip()
        
        if choice in ["1", "3"]:
            print("\n⚖️  Available scoring methods:")
            methods = {
                "1": "exact_match",
                "2": "contains_answer",
                "3": "semantic_similarity", 
                "4": "fuzzy_match",
                "5": "regex_match",
                "6": "custom_grader"
            }
            
            for key, method in methods.items():
                print(f"   {key}. {method}")
            
            method_choice = input("Select scoring method (1-6): ").strip()
            if method_choice in methods:
                scoring_method = methods[method_choice]
                print(f"✅ Updated scoring method to: {scoring_method}")
        
        if choice in ["2", "3"]:
            print(f"\nCurrent pattern: {repr(answer_pattern)}")
            print("\n💡 Common patterns:")
            print("   - r'Answer:\\s*(.+)' - extracts after 'Answer:'")
            print("   - r'(.+)' - captures entire response")
            print("   - r'\\b([A-D])\\b' - extracts single letter choice")
            print("   - r'(?:Answer|Response):\\s*(.+?)(?:\\n|$)' - multiline extraction")
            
            new_pattern = input("Enter new answer pattern (regex): ").strip()
            if new_pattern:
                answer_pattern = new_pattern
                print(f"✅ Updated answer pattern")
        
        print(f"\n✅ Final scoring configuration:")
        print(f"   Method: {scoring_method}")
        print(f"   Pattern: {repr(answer_pattern)}")
        
        return scoring_method, answer_pattern
    
    def _extract_class_name(self, code: str) -> str:
        """Extract the class name from generated code"""
        match = re.search(r'class (\w+)\(', code)
        return match.group(1) if match else "UnknownEval"
    
    def save_evaluation(self, code: str, eval_name: str, metadata: Dict[str, Any]) -> str:
        """Save generated evaluation and register it"""
        
        # Ensure generated_evals directory exists
        generated_dir = pathlib.Path("generated_evals")
        generated_dir.mkdir(exist_ok=True)
        
        # Save the evaluation file
        eval_file = generated_dir / f"{eval_name}_eval.py"
        with open(eval_file, 'w') as f:
            f.write(code)
        
        # Register the evaluation
        try:
            from external_evals_registry import ExternalEvaluationRegistry, ExternalEvalMetadata
            registry = ExternalEvaluationRegistry()
            
            # Extract class name for import
            class_name = self._extract_class_name(code)
            
            # Create metadata object
            eval_metadata = ExternalEvalMetadata(
                name=eval_name,
                class_name=class_name,
                file_path=str(eval_file),
                description=metadata.get("description", f"Generated evaluation for {eval_name}"),
                task_type=metadata.get("task_type", "auto_generated"),
                dataset_name=metadata.get("dataset_name", "unknown"),
                requires_grader=False,
                requires_equality_checker=False,
                default_num_examples=None,
                supports_checkpointing=True,
                created_by=metadata.get("created_by", "agentic_eval_builder")
            )
            
            registry.register_evaluation(eval_metadata)
            
            print(f"📋 Registered evaluation '{eval_name}' in external registry")
        except ImportError:
            print("⚠️  External registry not available, skipping registration")
        except Exception as e:
            print(f"⚠️  Failed to register evaluation: {e}")
        
        return str(eval_file)


# Example usage function
def interactive_eval_builder():
    """Interactive command-line interface for building evaluations"""
    
    print("🚀 Welcome to the Agentic Evaluation Builder!")
    print("This tool will help you create custom evaluations from HuggingFace datasets.\n")
    
    # Get user input
    dataset_name = input("📚 Enter HuggingFace dataset name: ").strip()
    subset = input("📂 Enter subset (optional, press Enter to skip): ").strip() or None
    split = input("🔀 Enter split (default: test): ").strip() or "test"
    
    # For demo purposes, we'll need to initialize samplers
    # In practice, these would be provided by the user
    print("\n⚠️  Note: You need to provide SamplerBase instances for the AI agents.")
    print("Please initialize your preferred language models and pass them to AgenticEvalBuilder.")
    
    return dataset_name, subset, split


if __name__ == "__main__":
    # Demo mode
    dataset_name, subset, split = interactive_eval_builder()
    print(f"\n✅ Configuration saved for dataset: {dataset_name}")
    print("To use this system, initialize AgenticEvalBuilder with your language model samplers.") 