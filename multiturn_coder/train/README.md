# Algorithmic Coding Multi-turn Training

This module implements a multi-turn algorithmic coding training system based on SLIME framework, inspired by SimpleTIR's approach but adapted for algorithm problems (similar to Codeforces).

## Overview

The system allows LLMs to:
1. Write Python code during reasoning process
2. Execute code immediately and see results
3. Continue reasoning based on execution results
4. Output final complete solution code
5. Get rewarded based on test case pass rate

## Key Components

### 1. Generate Function (`algorithmic_coding_generate.py`)
- Implements multi-turn dialogue with code execution
- Detects Python code blocks (```python ... ```)
- Executes code using local_judge.py sandbox
- Returns execution results to LLM for continued reasoning
- Supports up to 8 turns by default

### 2. Reward Function (`algorithmic_coding_reward.py`)
- Extracts final solution code from `<answer>` tags
- Tests code against hidden test cases
- Supports custom judging functions
- Returns 1.0 if all tests pass, 0.0 otherwise

### 3. Prompt Template (`algorithmic_coding_prompt.py`)
- Guides LLM to use `<think>` and `<answer>` tags
- Requires all code to be wrapped in ```python ... ```
- Provides clear examples and instructions

### 4. Dataset Handler (`algorithmic_coding_dataset.py`)
- Loads JSONL format datasets
- Validates data format
- Creates sample datasets for testing

## Data Format

Each line in the JSONL file should contain:
```json
{
    "question_content": "Problem description...",
    "test_cases": [
        {"input": "1 2 3", "output": "6"},
        {"input": "4 5 6", "output": "15"}
    ],
    "custom_judging_function": "lambda input_str, output, expected: abs(float(output) - float(expected)) < 1e-6"
}
```

## Output Format

LLM should output in this format:
```
<think>
Let me understand this problem...
Let me test my understanding:
```python
arr = [1, 2, 3]
print(f"Array: {arr}")
```
</think>

<answer>
Here's my complete solution:
```python
def solve(arr):
    return sum(arr)

n = int(input())
arr = list(map(int, input().split()))
print(solve(arr))
```
</answer>

## Usage

### 1. Test the System
```bash
cd /home/zhongmouhe/multiturn_code/multiturn_coder/multiturn_coder/train
python test_algorithmic_coding.py
```

### 2. Create Sample Dataset
```python
from multiturn_coder.train import create_sample_algorithmic_coding_dataset
create_sample_algorithmic_coding_dataset("sample_dataset.jsonl")
```

### 3. Validate Dataset
```python
from multiturn_coder.train import validate_algorithmic_coding_dataset
stats = validate_algorithmic_coding_dataset("your_dataset.jsonl")
print(stats)
```

### 4. Run Training
```bash
# Edit the paths in run_algorithmic_coding_training.sh
cd /home/zhongmouhe/multiturn_code/multiturn_coder/multiturn_coder/train
bash run_algorithmic_coding_training.sh
```

## Configuration

Key parameters in `ALGORITHMIC_CODING_CONFIGS`:
- `max_turns`: Maximum number of dialogue turns (default: 8)
- `max_code_execution_time`: Time limit for code execution (default: 5s)
- `max_code_memory`: Memory limit for code execution (default: 2GB)
- `python_interpreter_path`: Python interpreter path (default: "python")
- `bwrap_path`: Bubblewrap sandbox path (default: "bwrap")

## Training Script Configuration

The training script (`run_algorithmic_coding_training.sh`) follows SLIME's format and includes:

- **Model Configuration**: Uses Qwen2.5-3B model settings
- **Distributed Training**: Tensor parallel size 2, pipeline parallel size 1
- **Rollout Settings**: 3000 rollouts, batch size 32, 8 samples per prompt
- **Optimization**: GRPO algorithm with KL loss
- **Custom Functions**: Points to our algorithmic coding generate and reward functions

### Environment Variables
Set these before running the training script:
```bash
export MODEL_PATH="/path/to/your/model"
export DATA_PATH="/path/to/your/dataset.jsonl"
export CHECKPOINT_PATH="/path/to/save/checkpoints"
export LOG_PATH="/path/to/save/logs"
```

## Dependencies

- SLIME framework
- codebubble (for sandbox execution)
- local_judge.py (for code execution)
- bubblewrap (for sandbox isolation)

## Example Training Command

```bash
cd /home/zhongmouhe/multiturn_code/slime

# Set environment variables
export MODEL_PATH="/root/Qwen2.5-3B/"
export DATA_PATH="/root/algorithmic_coding_dataset.jsonl"
export CHECKPOINT_PATH="/root/Qwen2.5-3B_algorithmic_coding/"

# Run training
bash ../multiturn_coder/multiturn_coder/train/run_algorithmic_coding_training.sh
```

## Notes

- All code execution happens in isolated sandbox using bubblewrap
- Code execution results are automatically appended to dialogue context
- Final solution must be complete, runnable code that reads input and produces output
- Custom judging functions allow for flexible evaluation criteria
- System supports both simple string comparison and complex custom logic
- Training script follows SLIME's exact format and structure
- Uses Ray for distributed training coordination 