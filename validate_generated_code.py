"""
Validation script for generated evaluation files.
This script checks if generated Python files are syntactically correct.
"""

import ast
import importlib.util
import sys
from pathlib import Path
from typing import List, Tuple

def validate_python_syntax(file_path: str) -> Tuple[bool, str]:
    """
    Validate if a Python file has correct syntax.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to parse the AST
        ast.parse(content)
        
        # Try to compile
        compile(content, file_path, 'exec')
        
        return True, "✅ Valid Python syntax"
        
    except SyntaxError as e:
        return False, f"❌ Syntax Error: {e}"
    except Exception as e:
        return False, f"❌ Error: {e}"

def validate_class_structure(file_path: str) -> Tuple[bool, str]:
    """
    Validate if the file contains a proper evaluation class structure.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        Tuple of (is_valid, message)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        # Look for class definitions
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        
        if not classes:
            return False, "❌ No class definition found"
        
        eval_classes = [cls for cls in classes if cls.name.endswith('Eval') and cls.name != 'Eval']
        
        if not eval_classes:
            return False, "❌ No evaluation class found (should end with 'Eval' and not be the base 'Eval' class)"
        
        # Check the main evaluation class (not the base Eval class)
        eval_class = eval_classes[0]
        
        # Get methods directly from the class body (not using ast.walk to avoid nested functions)
        methods = []
        for node in eval_class.body:
            if isinstance(node, ast.FunctionDef):
                methods.append(node.name)
        
        required_methods = ['__init__', '__call__']
        missing_methods = [method for method in required_methods if method not in methods]
        
        if missing_methods:
            return False, f"❌ Missing required methods: {missing_methods}"
        
        found_methods = [method for method in methods if method in required_methods]
        return True, f"✅ Valid evaluation class: {eval_class.name} (methods: {found_methods})"
        
    except Exception as e:
        return False, f"❌ Error analyzing class structure: {e}"

def validate_imports(file_path: str) -> Tuple[bool, str]:
    """
    Validate if the file has necessary imports.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        Tuple of (is_valid, message)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend([alias.name for alias in node.names])
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports.extend([f"{module}.{alias.name}" for alias in node.names])
        
        # Check for common required imports
        common_imports = ['typing', 'datasets', 're']
        missing_imports = []
        
        for required in common_imports:
            found = any(required in imp for imp in imports)
            if not found:
                missing_imports.append(required)
        
        if missing_imports:
            return False, f"⚠️  Potentially missing imports: {missing_imports}"
        
        return True, "✅ All common imports present"
        
    except Exception as e:
        return False, f"❌ Error checking imports: {e}"

def validate_file(file_path: str) -> None:
    """
    Run comprehensive validation on a Python file.
    
    Args:
        file_path: Path to the Python file
    """
    print(f"\n🔍 Validating: {file_path}")
    print("=" * 50)
    
    # Test 1: Syntax validation
    syntax_valid, syntax_msg = validate_python_syntax(file_path)
    print(f"Syntax Check: {syntax_msg}")
    
    if not syntax_valid:
        print("❌ File has syntax errors - skipping further validation")
        return
    
    # Test 2: Class structure validation
    class_valid, class_msg = validate_class_structure(file_path)
    print(f"Class Structure: {class_msg}")
    
    # Test 3: Import validation
    import_valid, import_msg = validate_imports(file_path)
    print(f"Import Check: {import_msg}")
    
    # Overall result
    if syntax_valid and class_valid:
        print("🎉 Overall: VALID - File is ready for use")
    else:
        print("❌ Overall: INVALID - File needs fixes")

def validate_all_generated_evals() -> None:
    """Validate all files in the generated_evals directory."""
    
    generated_evals_dir = Path("generated_evals")
    
    if not generated_evals_dir.exists():
        print("❌ No generated_evals directory found")
        return
    
    python_files = list(generated_evals_dir.glob("*.py"))
    
    if not python_files:
        print("❌ No Python files found in generated_evals directory")
        return
    
    print(f"🔍 Found {len(python_files)} Python files to validate")
    
    valid_count = 0
    for file_path in python_files:
        try:
            validate_file(str(file_path))
            
            # Quick syntax check for counting
            syntax_valid, _ = validate_python_syntax(str(file_path))
            if syntax_valid:
                valid_count += 1
                
        except Exception as e:
            print(f"❌ Error validating {file_path}: {e}")
    
    print(f"\n📊 Validation Summary")
    print("=" * 50)
    print(f"Total files: {len(python_files)}")
    print(f"Valid files: {valid_count}")
    print(f"Invalid files: {len(python_files) - valid_count}")
    print(f"Success rate: {valid_count/len(python_files)*100:.1f}%")

def test_specific_file(file_path: str) -> None:
    """Test a specific file with detailed output."""
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return
    
    validate_file(file_path)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Validate specific file
        file_path = sys.argv[1]
        test_specific_file(file_path)
    else:
        # Validate all generated evaluations
        validate_all_generated_evals() 