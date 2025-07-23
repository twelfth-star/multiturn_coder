#!/usr/bin/env python3
"""
Test script for algorithmic coding training system.
"""

import asyncio
import json
import tempfile
import os
from pathlib import Path

# Add the parent directory to Python path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from multiturn_coder.train import (
    create_sample_algorithmic_coding_dataset,
    validate_algorithmic_coding_dataset,
    ALGORITHMIC_CODING_PROMPT_TEMPLATE
)
from multiturn_coder.train.algorithmic_coding_generate import (
    extract_code_blocks,
    extract_think_and_answer,
    execute_code_block
)
from multiturn_coder.train.algorithmic_coding_reward import (
    extract_final_answer_code,
    test_final_code
)
from multiturn_coder.local_judge import IntegratedExecutor

def test_prompt_template():
    """Test the prompt template."""
    print("Testing prompt template...")
    
    question = "Given an array of integers, find the maximum sum of a contiguous subarray."
    prompt = ALGORITHMIC_CODING_PROMPT_TEMPLATE.format(question_content=question)
    
    print(f"Generated prompt length: {len(prompt)}")
    print("Prompt preview:")
    print(prompt[:500] + "...")
    print("✓ Prompt template test passed\n")

def test_code_extraction():
    """Test code extraction functions."""
    print("Testing code extraction...")
    
    # Test extract_code_blocks
    text = """
    Let me test this:
    ```python
    arr = [1, 2, 3]
    print(sum(arr))
    ```
    
    Now let me solve it:
    ```python
    def solve(arr):
        return max(arr)
    ```
    """
    
    code_blocks = extract_code_blocks(text)
    print(f"Extracted {len(code_blocks)} code blocks:")
    for i, block in enumerate(code_blocks):
        print(f"  Block {i+1}: {block[:50]}...")
    
    # Test extract_think_and_answer
    text_with_tags = """
    <think>
    Let me understand this problem...
    ```python
    print("Testing")
    ```
    </think>
    
    <answer>
    Here's my solution:
    ```python
    def solve():
        return 42
    ```
    </answer>
    """
    
    think, answer = extract_think_and_answer(text_with_tags)
    print(f"Think content: {think[:50] if think else 'None'}...")
    print(f"Answer content: {answer[:50] if answer else 'None'}...")
    
    # Test extract_final_answer_code
    final_code = extract_final_answer_code(text_with_tags)
    print(f"Final code: {final_code[:50] if final_code else 'None'}...")
    
    print("✓ Code extraction test passed\n")

def test_dataset_creation():
    """Test dataset creation and validation."""
    print("Testing dataset creation...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_path = f.name
    
    try:
        create_sample_algorithmic_coding_dataset(temp_path)
        
        # Validate the created dataset
        stats = validate_algorithmic_coding_dataset(temp_path)
        print("Dataset statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Load and check a sample
        with open(temp_path, 'r') as f:
            first_line = f.readline()
            sample = json.loads(first_line)
            print(f"Sample keys: {list(sample.keys())}")
            print(f"Has test_cases: {'test_cases' in sample}")
            print(f"Has custom_judging: {'custom_judging_function' in sample}")
        
        print("✓ Dataset creation test passed\n")
        
    finally:
        os.unlink(temp_path)

async def test_code_execution():
    """Test code execution functionality."""
    print("Testing code execution...")
    
    # Initialize executor
    executor = IntegratedExecutor(
        time_limit=5,
        memory_limit=1024 * 2,
        python_interpreter_path="python",
        bwrap_path="bwrap"
    )
    
    # Create workspace
    workspace = tempfile.mkdtemp(prefix="test_exec_")
    
    try:
        # Test simple code execution
        simple_code = """
print("Hello, World!")
print(2 + 2)
"""
        
        result = await execute_code_block(simple_code, executor, workspace)
        print(f"Simple code execution result: {result[:100]}...")
        
        # Test code with error
        error_code = """
print("This will work")
undefined_variable
"""
        
        result = await execute_code_block(error_code, executor, workspace)
        print(f"Error code execution result: {result[:100]}...")
        
        print("✓ Code execution test passed\n")
        
    finally:
        import shutil
        shutil.rmtree(workspace)

async def test_reward_function():
    """Test reward function."""
    print("Testing reward function...")
    
    # Initialize executor
    executor = IntegratedExecutor(
        time_limit=5,
        memory_limit=1024 * 2,
        python_interpreter_path="python",
        bwrap_path="bwrap"
    )
    
    # Create workspace
    workspace = tempfile.mkdtemp(prefix="test_reward_")
    
    try:
        # Test case
        test_cases = [
            {"input": "5\n1 2 3 4 5", "output": "15"},
            {"input": "3\n-1 -2 -3", "output": "-6"}
        ]
        
        # Correct solution
        correct_code = """
n = int(input())
arr = list(map(int, input().split()))
print(sum(arr))
"""
        
        # Test the solution
        results = await test_final_code(
            correct_code, 
            test_cases, 
            None,  # No custom judging
            executor,
            workspace
        )
        
        print(f"Test results: {results}")
        print(f"All tests passed: {all(results)}")
        
        # Incorrect solution
        incorrect_code = """
n = int(input())
arr = list(map(int, input().split()))
print(max(arr))  # Wrong: should be sum, not max
"""
        
        results = await test_final_code(
            incorrect_code, 
            test_cases, 
            None,
            executor,
            workspace
        )
        
        print(f"Incorrect solution results: {results}")
        print(f"All tests passed: {all(results)}")
        
        print("✓ Reward function test passed\n")
        
    finally:
        import shutil
        shutil.rmtree(workspace)

async def main():
    """Run all tests."""
    print("Running algorithmic coding training system tests...\n")
    
    test_prompt_template()
    test_code_extraction()
    test_dataset_creation()
    await test_code_execution()
    await test_reward_function()
    
    print("All tests completed successfully! ✓")

if __name__ == "__main__":
    asyncio.run(main()) 