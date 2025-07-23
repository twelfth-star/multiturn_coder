import json
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path

from .algorithmic_coding_prompt import ALGORITHMIC_CODING_PROMPT_TEMPLATE

def load_algorithmic_coding_dataset(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load algorithmic coding dataset from JSONL file.
    
    Expected format:
    {
        "question_content": "Problem description...",
        "test_cases": [
            {"input": "1 2 3", "output": "6"},
            {"input": "4 5 6", "output": "15"}
        ],
        "custom_judging_function": "lambda input_str, output, expected: abs(float(output) - float(expected)) < 1e-6"  # or null
    }
    """
    samples = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                # Validate required fields
                if "question_content" not in data:
                    raise ValueError(f"Missing 'question_content' in line {line_num}")
                
                # Set defaults for optional fields
                test_cases = data.get("test_cases", [])
                custom_judging_function = data.get("custom_judging_function", None)
                
                # Create prompt using template
                prompt = ALGORITHMIC_CODING_PROMPT_TEMPLATE.format(
                    question_content=data["question_content"]
                )
                
                # Create sample
                sample = {
                    "prompt": prompt,
                    "label": {
                        "test_cases": test_cases,
                        "custom_judging_function": custom_judging_function,
                        "question_content": data["question_content"]
                    }
                }
                
                samples.append(sample)
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON in line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    return samples

def save_algorithmic_coding_dataset(samples: List[Dict[str, Any]], output_path: str):
    """Save algorithmic coding dataset to JSONL file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')

def create_sample_algorithmic_coding_dataset(output_path: str):
    """Create a sample dataset for testing."""
    sample_data = [
        {
            "question_content": "Given an array of integers, find the maximum sum of a contiguous subarray.\n\nInput: The first line contains an integer n (1 ≤ n ≤ 10^5), the size of the array. The second line contains n integers separated by spaces.\n\nOutput: Print the maximum sum of a contiguous subarray.",
            "test_cases": [
                {"input": "5\n1 -2 3 -1 2", "output": "4"},
                {"input": "3\n-1 -2 -3", "output": "-1"},
                {"input": "1\n5", "output": "5"}
            ],
            "custom_judging_function": None
        },
        {
            "question_content": "Given a string, find the length of the longest substring without repeating characters.\n\nInput: A string s (1 ≤ |s| ≤ 10^4) containing only lowercase letters.\n\nOutput: Print the length of the longest substring without repeating characters.",
            "test_cases": [
                {"input": "abcabcbb", "output": "3"},
                {"input": "bbbbb", "output": "1"},
                {"input": "pwwkew", "output": "3"}
            ],
            "custom_judging_function": None
        },
        {
            "question_content": "Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.\n\nInput: Two lines. First line contains m and n separated by space. Second line contains m+n integers (the merged array).\n\nOutput: Print the median as a float.",
            "test_cases": [
                {"input": "2 2\n1 2 3 4", "output": "2.5"},
                {"input": "2 1\n1 3 2", "output": "2.0"}
            ],
            "custom_judging_function": "lambda input_str, output, expected: abs(float(output) - float(expected)) < 1e-6"
        }
    ]
    
    save_algorithmic_coding_dataset(sample_data, output_path)
    print(f"Sample dataset created at: {output_path}")

def validate_algorithmic_coding_dataset(jsonl_path: str) -> Dict[str, Any]:
    """Validate algorithmic coding dataset and return statistics."""
    samples = load_algorithmic_coding_dataset(jsonl_path)
    
    stats = {
        "total_samples": len(samples),
        "samples_with_test_cases": 0,
        "samples_with_custom_judging": 0,
        "avg_test_cases_per_sample": 0,
        "errors": []
    }
    
    total_test_cases = 0
    
    for i, sample in enumerate(samples):
        test_cases = sample["label"]["test_cases"]
        custom_judging = sample["label"]["custom_judging_function"]
        
        if test_cases:
            stats["samples_with_test_cases"] += 1
            total_test_cases += len(test_cases)
        
        if custom_judging:
            stats["samples_with_custom_judging"] += 1
    
    if stats["total_samples"] > 0:
        stats["avg_test_cases_per_sample"] = total_test_cases / stats["total_samples"]
    
    return stats

if __name__ == "__main__":
    # Create sample dataset for testing
    create_sample_algorithmic_coding_dataset("sample_algorithmic_coding_dataset.jsonl")
    
    # Validate the created dataset
    stats = validate_algorithmic_coding_dataset("sample_algorithmic_coding_dataset.jsonl")
    print("Dataset statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}") 