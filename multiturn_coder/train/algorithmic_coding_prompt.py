ALGORITHMIC_CODING_PROMPT_TEMPLATE = """Solve the following algorithmic problem step by step. You now have the ability to write executable Python code to enhance your reasoning process. The Python code will be executed by an external sandbox, and the output will be returned to aid your reasoning and help you arrive at the final solution.

Code Format:
Each code snippet must be wrapped between ```python and ```. You can use `print()` to output intermediate results or test your understanding.

Output Format:
1. Put your reasoning process between <think> and </think> tags.
2. Put your final, complete solution code between <answer> and </answer> tags.
3. All code blocks (both in <think> and <answer>) must be wrapped in ```python ... ```.

Example:
<think>
Let me understand this problem first. I need to find the maximum sum of a subarray...
Let me write some code to test my understanding:
```python
arr = [1, -2, 3, -1, 2]
print(f"Array: {arr}")
print(f"Length: {len(arr)}")
```
</think>

<answer>
Based on my analysis, here's the complete solution:
```python
def max_subarray_sum(arr):
    max_sum = float('-inf')
    current_sum = 0
    
    for num in arr:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    
    return max_sum

# Read input
n = int(input())
arr = list(map(int, input().split()))
result = max_subarray_sum(arr)
print(result)
```
</answer>

Problem:
{question_content}

Remember:
- Write code to test your understanding or implement parts of the solution
- All code must be wrapped in ```python ... ```
- Put reasoning in <think> tags
- Put final solution in <answer> tags
- The final solution should be a complete, runnable program that reads input and produces the expected output
""" 