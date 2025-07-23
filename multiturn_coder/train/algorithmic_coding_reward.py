import asyncio
import re
import json
import os
import tempfile
from typing import List, Optional, Dict, Any

from slime.utils.types import Sample

from ..local_judge import IntegratedExecutor

ALGORITHMIC_CODING_CONFIGS = {
    "max_code_execution_time": 5,
    "max_code_memory": 1024 * 2,  # 2GB
    "python_interpreter_path": "python",
    "bwrap_path": "bwrap",
}

def extract_final_answer_code(text: str) -> Optional[str]:
    """Extract the final answer code from <answer> tags."""
    answer_pattern = r'<answer>(.*?)</answer>'
    answer_match = re.search(answer_pattern, text, re.DOTALL)
    
    if not answer_match:
        return None
    
    answer_content = answer_match.group(1).strip()
    
    # Extract Python code blocks from answer content
    code_pattern = r'```python\s*\n(.*?)\n```'
    code_matches = re.findall(code_pattern, answer_content, re.DOTALL)
    
    if code_matches:
        # Return the last code block as the final answer
        return code_matches[-1].strip()
    
    return None

def create_test_case_code(test_cases: List[Dict[str, str]], custom_judging_function: Optional[str] = None) -> str:
    """Create test case execution code."""
    test_code = """
import sys
import json

# The solution function/class should be defined here
SOLUTION_CODE = '''
{0}
'''

# Execute the solution code
exec(SOLUTION_CODE)

# Test cases
test_cases = {1}

# Custom judging function
custom_judging_function = {2}

def run_tests():
    results = []
    for i, test_case in enumerate(test_cases):
        try:
            # Capture stdout to get the output
            import io
            import contextlib
            
            # Prepare input
            input_str = test_case.get('input', '')
            
            # Capture output
            output_capture = io.StringIO()
            with contextlib.redirect_stdout(output_capture):
                # Execute with input
                exec(SOLUTION_CODE)
                
                # If there's a main function or specific entry point, call it
                if 'main' in globals():
                    # Redirect stdin to simulate input
                    import sys
                    from io import StringIO
                    sys.stdin = StringIO(input_str)
                    main()
                    sys.stdin = sys.__stdin__
                elif 'solve' in globals():
                    solve(input_str)
                else:
                    # Try to execute the code directly
                    exec(input_str)
            
            output = output_capture.getvalue().strip()
            
            # Judge the result
            if custom_judging_function:
                # Use custom judging function
                judge_func = eval(custom_judging_function)
                is_correct = judge_func(input_str, output, test_case.get('output', ''))
            else:
                # Default string comparison
                expected_output = test_case.get('output', '').strip()
                is_correct = output == expected_output
            
            results.append(is_correct)
            
        except Exception as e:
            print(f"Test case {i} failed with error: {{str(e)}}", file=sys.stderr)
            results.append(False)
    
    return results

# Run tests and print results
test_results = run_tests()
print(json.dumps(test_results))
""".format(
        "{SOLUTION_CODE_PLACEHOLDER}",
        json.dumps(test_cases),
        f"'''{custom_judging_function}'''" if custom_judging_function else "None"
    )
    
    return test_code

async def test_final_code(
    final_code: str, 
    test_cases: List[Dict[str, str]], 
    custom_judging_function: Optional[str] = None,
    executor: IntegratedExecutor,
    workspace: str
) -> List[bool]:
    """Test the final code against test cases."""
    try:
        # Create test execution code
        test_code = create_test_case_code(test_cases, custom_judging_function)
        
        # Replace placeholder with actual solution code
        test_code = test_code.replace("{SOLUTION_CODE_PLACEHOLDER}", final_code)
        
        # Create a temporary workspace for testing
        temp_workspace = os.path.join(workspace, f"test_{hash(final_code) % 10000}")
        os.makedirs(temp_workspace, exist_ok=True)
        
        # Create executor
        python_executor = executor.make_python_executor(temp_workspace)
        
        # Set resource limits
        resource_limits = executor.make_resource_limits(
            time_limit=ALGORITHMIC_CODING_CONFIGS["max_code_execution_time"],
            memory_limit=ALGORITHMIC_CODING_CONFIGS["max_code_memory"],
        )
        
        # Prepare and execute
        prepare_result = python_executor.prepare(temp_workspace, test_code)
        inner_cmd = prepare_result['inner_cmd']
        
        # Execute with empty input
        execution_result = python_executor.execute(
            inner_cmd=inner_cmd,
            stdin_str="",
            limits=resource_limits
        )
        
        # Parse results
        if execution_result.status.value == "SUCCESS":
            try:
                output = execution_result.stdout.strip()
                results = json.loads(output)
                return results
            except (json.JSONDecodeError, ValueError):
                # If JSON parsing fails, assume all tests failed
                return [False] * len(test_cases)
        else:
            # Execution failed, all tests failed
            return [False] * len(test_cases)
            
    except Exception as e:
        # Any exception means all tests failed
        return [False] * len(test_cases)

async def reward_func(args, sample: Sample, **kwargs) -> float:
    """Reward function for algorithmic coding.
    
    Tests the final code against hidden test cases.
    Returns 1.0 if all tests pass, 0.0 otherwise.
    """
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    # Extract test cases and custom judging function from sample label
    test_cases = sample.label.get("test_cases", [])
    custom_judging_function = sample.label.get("custom_judging_function", None)
    
    if not test_cases:
        # No test cases provided, return 0
        return 0.0
    
    # Extract final answer code
    full_response = sample.prompt + sample.response
    final_code = extract_final_answer_code(full_response)
    
    if not final_code:
        # No final answer code found, return 0
        return 0.0
    
    # Initialize executor
    executor = IntegratedExecutor(
        time_limit=ALGORITHMIC_CODING_CONFIGS["max_code_execution_time"],
        memory_limit=ALGORITHMIC_CODING_CONFIGS["max_code_memory"],
        python_interpreter_path=ALGORITHMIC_CODING_CONFIGS["python_interpreter_path"],
        bwrap_path=ALGORITHMIC_CODING_CONFIGS["bwrap_path"],
    )
    
    # Create workspace
    workspace = tempfile.mkdtemp(prefix="algorithmic_coding_reward_")
    
    try:
        # Test the final code
        test_results = await test_final_code(
            final_code, 
            test_cases, 
            custom_judging_function,
            executor,
            workspace
        )
        
        # Calculate reward: 1.0 if all tests pass, 0.0 otherwise
        if test_results and all(test_results):
            return 1.0
        else:
            return 0.0
            
    finally:
        # Clean up workspace
        try:
            import shutil
            shutil.rmtree(workspace)
        except:
            pass 