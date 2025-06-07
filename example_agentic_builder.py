"""
Example script demonstrating the Agentic Evaluation Builder system.

This script shows how to use the system to automatically generate evaluations
from HuggingFace datasets using language models as agents.
"""

import os
from dotenv import load_dotenv
from agentic_eval_builder import AgenticEvalBuilder, interactive_eval_builder
from sampler.chat_completion_sampler import ChatCompletionSampler
from sampler.gemini_sampler import GeminiSampler

def setup_samplers():
    """Set up the language model samplers for the different agents"""
    
    # Load environment variables
    load_dotenv()
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT_ID", "your-project-id")
    
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found. Please set it to use this example.")
        # return None
    
    # For this example, we'll use GPT-4 for all agents
    # In practice, you might want to use different models for different tasks
    base_sampler = GeminiSampler(
        model="gemini-2.5-flash-preview-05-20",
        project_id=project_id,
        location="us-central1",
        use_gemini_grounding=False,
    )
    
    # You could customize each sampler for specific tasks:
    # analyzer_sampler = ChatCompletionSampler(
    #     model="gpt-4-turbo-preview",
    #     system_message="You are an expert in dataset analysis. Analyze datasets systematically and provide structured JSON responses."
    # )
    analyzer_sampler = ChatCompletionSampler(
        model="meta/llama-4-maverick-17b-128e-instruct-maas",
        system_message="You are an expert in dataset analysis. Analyze datasets systematically and provide structured JSON responses.",
        base_url=f"https://us-east5-aiplatform.googleapis.com/v1/projects/{project_id}/locations/us-east5/endpoints/openapi"
    )
    
    # detector_sampler = ChatCompletionSampler(
    #     model="gpt-4-turbo-preview", 
    #     system_message="You are an expert in machine learning evaluation methodologies."
    # )
    detector_sampler = ChatCompletionSampler(
        model="meta/llama-4-maverick-17b-128e-instruct-maas",
        system_message="You are an expert in machine learning evaluation methodologies.",
        base_url=f"https://us-east5-aiplatform.googleapis.com/v1/projects/{project_id}/locations/us-east5/endpoints/openapi"
    )  
    
    # prompt_engineer = ChatCompletionSampler(
    #     model="gpt-4-turbo-preview",
    #     system_message="You are an expert prompt engineer. Create clear, effective prompts for language model evaluation."
    # )

    prompt_engineer = ChatCompletionSampler(
        model="meta/llama-4-maverick-17b-128e-instruct-maas",
        system_message="You are an expert prompt engineer. Create clear, effective prompts for language model evaluation.",
        base_url=f"https://us-east5-aiplatform.googleapis.com/v1/projects/{project_id}/locations/us-east5/endpoints/openapi"
    )
    
    # scoring_sampler = ChatCompletionSampler(
    #     model="gpt-4-turbo-preview",
    #     system_message="You are an expert in evaluation metrics and scoring methods."
    # )

    scoring_sampler = ChatCompletionSampler(
        model="meta/llama-4-maverick-17b-128e-instruct-maas",
        system_message="You are an expert in evaluation metrics and scoring methods.",
        base_url=f"https://us-east5-aiplatform.googleapis.com/v1/projects/{project_id}/locations/us-east5/endpoints/openapi"
    )
    
    # code_generator = ChatCompletionSampler(
    #     model="gpt-4-turbo-preview",
    #     system_message="You are an expert Python programmer specialized in evaluation frameworks. Generate clean, well-documented code."
    # )

    code_generator = ChatCompletionSampler(
        model="meta/llama-4-maverick-17b-128e-instruct-maas",
        system_message="You are an expert Python programmer specialized in evaluation frameworks. Generate clean, well-documented code.",
        base_url=f"https://us-east5-aiplatform.googleapis.com/v1/projects/{project_id}/locations/us-east5/endpoints/openapi"
    )
    
    # Add mapping sampler for column mapping agent
    mapping_sampler = ChatCompletionSampler(
        model="meta/llama-4-maverick-17b-128e-instruct-maas",
        system_message="You are an expert in data analysis and semantic understanding. Analyze dataset structures and intelligently map columns to template requirements.",
        base_url=f"https://us-east5-aiplatform.googleapis.com/v1/projects/{project_id}/locations/us-east5/endpoints/openapi"
    )
    
    return analyzer_sampler, detector_sampler, prompt_engineer, scoring_sampler, code_generator, mapping_sampler

def example_basic_usage():
    """Basic example showing automated evaluation generation"""
    
    print("🔧 Setting up AI agents...")
    samplers = setup_samplers()
    if not samplers:
        return
    
    # Initialize the agentic builder
    builder = AgenticEvalBuilder(*samplers)
    
    # Example 1: Math dataset
    print("\n" + "="*60)
    print("📚 Example 1: Generating evaluation for a math dataset")
    print("="*60)
    
    try:
        # Use the new method that automatically saves and registers
        file_path = builder.build_and_save_eval(
            # dataset_name="dreamerdeo/finqa",
            dataset_name="rag-datasets/rag-mini-wikipedia",
            # subset="arithmetic__add_or_sub", 
            subset="question-answer",
            split="test",
            interactive=False  # Set to True for interactive mode
        )
        
        print(f"✅ Generated and saved evaluation to: {file_path}")
        
        # Show a preview of the generated file
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                print("\n📝 Generated code preview:")
                print(content[:500] + "...")
                
                # Extract class name for registration info
                import re
                class_match = re.search(r'class (\w+)\(', content)
                if class_match:
                    class_name = class_match.group(1)
                    eval_name = class_name.replace("Eval", "").lower()
                    print(f"\n📋 Evaluation registered as: '{eval_name}'")
                    print(f"🚀 You can now run: python simple_evals.py --evals {eval_name}")
        except Exception as e:
            print(f"⚠️  Could not read generated file: {e}")
        
    except Exception as e:
        print(f"❌ Error generating evaluation: {e}")

def example_interactive_usage():
    """Example showing interactive evaluation generation"""
    
    print("🔧 Setting up AI agents...")
    samplers = setup_samplers()
    if not samplers:
        return
    
    # Initialize the agentic builder
    builder = AgenticEvalBuilder(*samplers)
    
    # Interactive mode
    print("\n" + "="*60)
    print("🎮 Interactive Evaluation Builder")
    print("="*60)
    
    dataset_name = input("📚 Enter HuggingFace dataset name (default: rag-datasets/rag-mini-wikipedia): ").strip() or "rag-datasets/rag-mini-wikipedia"
    subset = input("📂 Enter subset (default: question-answer): ").strip() or "question-answer"
    split = input("🔀 Enter split (default: test): ").strip() or "test"
    
    try:
        # Use the new method that automatically saves and registers
        file_path = builder.build_and_save_eval(
            dataset_name=dataset_name,
            subset=subset,
            split=split,
            interactive=True  # Enable interactive mode
        )
        
        print(f"✅ Generated and saved evaluation to: {file_path}")
        
        # Extract evaluation name for testing
        import re
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                class_match = re.search(r'class (\w+)\(', content)
                if class_match:
                    class_name = class_match.group(1)
                    eval_name = class_name.replace("Eval", "").lower()
                    
                    print(f"\n📋 Evaluation registered as: '{eval_name}'")
                    print(f"🚀 You can now run: python simple_evals.py --evals {eval_name}")
                    
                    # Optionally test the generated evaluation
                    test_generated_eval = input("\n🧪 Test the generated evaluation? (y/n): ").lower()
                    if test_generated_eval == 'y':
                        test_evaluation_from_file(file_path, class_name)
        except Exception as e:
            print(f"⚠️  Could not process generated file: {e}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def example_multiple_datasets():
    """Example showing batch processing of multiple datasets"""
    
    print("🔧 Setting up AI agents...")
    samplers = setup_samplers()
    if not samplers:
        return
    
    builder = AgenticEvalBuilder(*samplers)
    
    # List of datasets to process
    datasets = [
        {"name": "squad", "subset": None, "split": "validation"},
        {"name": "glue", "subset": "sst2", "split": "validation"},
        {"name": "math_dataset", "subset": "arithmetic__add_or_sub", "split": "test"},
    ]
    
    print("\n" + "="*60)
    print("🔄 Batch Processing Multiple Datasets")
    print("="*60)
    
    results = []
    
    for i, dataset_config in enumerate(datasets, 1):
        print(f"\n📚 Processing dataset {i}/{len(datasets)}: {dataset_config['name']}")
        
        try:
            # Use the new method that automatically saves and registers
            file_path = builder.build_and_save_eval(
                dataset_name=dataset_config["name"],
                subset=dataset_config["subset"],
                split=dataset_config["split"],
                interactive=False
            )
            
            # Extract class name and eval name
            import re
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    class_match = re.search(r'class (\w+)\(', content)
                    if class_match:
                        class_name = class_match.group(1)
                        eval_name = class_name.replace("Eval", "").lower()
                    else:
                        class_name = "UnknownEval"
                        eval_name = "unknown"
            except:
                class_name = "UnknownEval"
                eval_name = "unknown"
            
            results.append({
                "dataset": dataset_config["name"],
                "class_name": class_name,
                "eval_name": eval_name,
                "file_path": file_path,
                "success": True,
                "code_length": len(open(file_path).read()) if file_path else 0
            })
            
            print(f"💾 Saved to {file_path}")
            print(f"📋 Registered as: '{eval_name}'")
            
        except Exception as e:
            print(f"❌ Failed to process {dataset_config['name']}: {e}")
            results.append({
                "dataset": dataset_config["name"],
                "success": False,
                "error": str(e)
            })
    
    # Summary
    print("\n" + "="*60)
    print("📊 Batch Processing Summary")
    print("="*60)
    
    successful = sum(1 for r in results if r["success"])
    print(f"✅ Successfully processed: {successful}/{len(datasets)} datasets")
    
    print(f"\n📋 Generated Evaluations:")
    for result in results:
        if result["success"]:
            status = "✅"
            info = f"→ {result['eval_name']} (saved to {result['file_path']})"
        else:
            status = "❌"
            info = f"→ Error: {result.get('error', 'Unknown error')}"
        print(f"{status} {result['dataset']} {info}")
    
    if successful > 0:
        print(f"\n🚀 You can now run any of these evaluations with:")
        for result in results:
            if result["success"]:
                print(f"   python simple_evals.py --evals {result['eval_name']}")

def test_evaluation_from_file(file_path: str, class_name: str):
    """Test a generated evaluation from file (enhanced mock implementation)"""
    
    print(f"\n🧪 Testing generated evaluation: {class_name}")
    print(f"📁 From file: {file_path}")
    
    try:
        # Read and validate the generated code
        with open(file_path, 'r') as f:
            generated_code = f.read()
        
        print("📝 Code validation:")
        
        # Check for required imports
        required_imports = [
            "from eval_types import Eval, EvalResult, SamplerBase, SingleEvalResult",
            "from eval_utils import",
            "import common"
        ]
        
        for imp in required_imports:
            if imp in generated_code:
                print(f"  ✅ {imp}")
            else:
                print(f"  ⚠️  Missing: {imp}")
        
        # Check for required methods
        required_methods = ["def __init__(self", "def __call__(self, sampler: SamplerBase)"]
        for method in required_methods:
            if method in generated_code:
                print(f"  ✅ {method}")
            else:
                print(f"  ❌ Missing: {method}")
        
        # Check for proper inheritance
        if f"class {class_name}(Eval, EvaluationMixin):" in generated_code:
            print(f"  ✅ Proper class inheritance")
        else:
            print(f"  ⚠️  Check class inheritance")
        
        print("\n🎯 Mock evaluation run:")
        print("  📊 Processing 5 test examples...")
        print("  ⏱️  Average response time: 1.2s")
        print("  🎯 Mock accuracy: 85.5%")
        print("  ✅ Evaluation completed successfully!")
        
        # Show how to use the evaluation
        eval_name = class_name.replace("Eval", "").lower()
        print(f"\n📋 Usage instructions:")
        print(f"  • Run with simple_evals: python simple_evals.py --evals {eval_name}")
        print(f"  • Import in Python: from generated_evals.{eval_name}_eval import {class_name}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

def test_evaluation(generated_code: str, class_name: str):
    """Test a generated evaluation (kept for backward compatibility)"""
    print(f"⚠️  Note: This method is deprecated. Use test_evaluation_from_file() instead.")
    test_evaluation_from_file("temp_eval.py", class_name)

def show_supported_features():
    """Display the features supported by the agentic builder"""
    
    print("\n" + "="*60)
    print("🚀 Agentic Evaluation Builder Features")
    print("="*60)
    
    features = {
        "📊 Dataset Analysis": [
            "Automatic column detection and validation",
            "Task type classification with fallbacks",
            "Sample data analysis and inference",
            "Metadata extraction and processing",
            "Column mapping with intelligent inference"
        ],
        "🎯 Evaluation Styles": [
            "Multiple choice (MMLU-style)",
            "Math problems (MATH-style)",
            "Reading comprehension (DROP-style)",
            "Code generation (HumanEval-style)",
            "Factual Q&A (SimpleQA-style)",
            "Generic text generation with safe fallbacks"
        ],
        "✍️ Prompt Engineering": [
            "Task-specific prompt generation",
            "Output format specification",
            "Context-aware templates",
            "Multilingual support",
            "Automatic prompt validation"
        ],
        "🎯 Scoring Methods": [
            "Exact string matching",
            "Fuzzy text matching",
            "Multiple choice scoring",
            "Code execution testing",
            "LLM-based grading",
            "Mathematical equivalence",
            "Numerical tolerance scoring"
        ],
        "💻 Code Generation & Validation": [
            "Full evaluation class generation",
            "AST-based syntax validation",
            "Import safety checking",
            "Pattern enforcement (inheritance, methods)",
            "Automatic error detection and fixing",
            "Safe fallback templates"
        ],
        "💾 Persistence & Registration": [
            "Automatic file saving to generated_evals/",
            "External evaluation registry integration",
            "Metadata tracking and storage",
            "Automatic eval name generation",
            "Integration with simple_evals.py CLI",
            "Batch processing and storage"
        ],
        "🎮 User Experience": [
            "Interactive mode with user validation",
            "Batch processing multiple datasets",
            "Progress tracking and error reporting",
            "Graceful error recovery",
            "Code validation and testing",
            "Clear usage instructions"
        ],
        "🔒 Safety & Reliability": [
            "Whitelisted imports only",
            "Column existence validation",
            "Template formatting safety",
            "Multi-level fallback system",
            "Error containment and recovery",
            "Generated code validation"
        ]
    }
    
    for category, items in features.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  • {item}")
    
    print(f"\n🎯 End-to-End Workflow:")
    print(f"  1. 🔍 Analyze HuggingFace dataset structure")
    print(f"  2. 🎯 Determine appropriate evaluation approach")
    print(f"  3. ✍️  Generate task-specific prompts and scoring")
    print(f"  4. 💻 Create complete evaluation class code")
    print(f"  5. ✅ Validate and fix generated code")
    print(f"  6. 💾 Save to generated_evals/ directory")
    print(f"  7. 📋 Register in external evaluation registry")
    print(f"  8. 🚀 Ready to run with simple_evals.py!")
    
    print(f"\n📋 Quick Start:")
    print(f"  python example_agentic_builder.py  # Run this script")
    print(f"  python simple_evals.py --list-evals  # See all available evals")
    print(f"  python simple_evals.py --evals <eval_name>  # Run generated eval")

def example_custom_prompt_template():
    """Example showing how to use custom prompt templates"""
    
    print("🔧 Setting up AI agents...")
    samplers = setup_samplers()
    if not samplers:
        return
    
    # Initialize the agentic builder
    builder = AgenticEvalBuilder(*samplers)
    
    print("\n" + "="*60)
    print("✍️  Example: Custom Prompt Template")
    print("="*60)
    
    try:
        # Generate evaluation with default settings first
        file_path = builder.build_and_save_eval(
            dataset_name="rag-datasets/rag-mini-wikipedia",
            subset="question-answer",
            split="test",
            interactive=False
        )
        
        print(f"✅ Generated evaluation: {file_path}")
        
        # Show how to use custom prompt template
        print("\n🎨 Customizing the prompt template...")
        print("You can now customize the prompt template when using the evaluation:")
        
        print("""
# Example custom prompt templates:

# 1. Simple format
custom_prompt = "Answer the question with a simple "yes" or "no" based on your knowledge. Without returning anything else, no need to tell me the reason\\nQuestion: {question}\\nAnswer:"

# 2. More detailed format  
custom_prompt = "Please answer this question clearly and concisely:\\n\\n{question}\\n\\nYour answer:"

# 3. Format with context (for reading comprehension)
custom_prompt = "Context: {context}\\n\\nBased on the context above, answer: {question}\\n\\nAnswer:"

# 4. Chain-of-thought format
custom_prompt = "Question: {question}\\n\\nLet's think step by step and then provide the final answer.\\n\\nAnswer:"

# Usage example:
from generated_evals.rag_mini_wikipedia_eval import Rag_Mini_WikipediaEval

# Use default prompt template
eval_default = Rag_Mini_WikipediaEval(num_examples=10)

# Use custom prompt template
eval_custom = Rag_Mini_WikipediaEval(
    num_examples=10,
    prompt_template="Please answer this question clearly:\\n\\n{question}\\n\\nAnswer:"
)
""")
        
        print("💡 Key features of customizable prompt templates:")
        print("  • Use {question} for the main question/input")
        print("  • Use {context} for reading comprehension tasks")
        print("  • Use {input} as alternative to {question}")
        print("  • Access any dataset column with {column_name}")
        print("  • Default templates provided for each task type")
        print("  • Templates are Python format strings")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Main function demonstrating different usage patterns"""
    
    print("🚀 Agentic Evaluation Builder - Examples")
    print("=" * 60)
    
    # Check requirements
    try:
        import datasets
        print("✅ HuggingFace datasets library found")
    except ImportError:
        print("❌ Please install: pip install datasets")
        return
    
    print("\nSelect an example to run:")
    print("1. Show supported features")
    print("2. Basic automated generation")
    print("3. Interactive generation")
    print("4. Batch processing multiple datasets")
    print("5. Custom prompt template example")
    print("6. Exit")
    
    choice = input("\nEnter your choice (1-6): ").strip()
    
    if choice == "1":
        show_supported_features()
    elif choice == "2":
        example_basic_usage()
    elif choice == "3":
        example_interactive_usage()
    elif choice == "4":
        example_multiple_datasets()
    elif choice == "5":
        example_custom_prompt_template()
    elif choice == "6":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice. Please enter 1-6.")

if __name__ == "__main__":
    main() 