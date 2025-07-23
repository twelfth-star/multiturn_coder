import asyncio
import re
import json
import os
import tempfile
from typing import List, Optional, Tuple, Dict, Any

from slime.rollout.sglang_example import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

from ..local_judge import IntegratedExecutor

ALGORITHMIC_CODING_CONFIGS = {
    "max_turns": 8,
    "max_code_execution_time": 5,
    "max_code_memory": 1024 * 2,  # 2GB
    "max_overall_time": 30,
    "python_interpreter_path": "python",
    "bwrap_path": "bwrap",
}

def extract_code_blocks(text: str) -> List[str]:
    """Extract all Python code blocks from text."""
    pattern = r'```python\s*\n(.*?)\n```'
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]

def extract_think_and_answer(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract <think> and <answer> content from text."""
    think_pattern = r'<think>(.*?)</think>'
    answer_pattern = r'<answer>(.*?)</answer>'
    
    think_match = re.search(think_pattern, text, re.DOTALL)
    answer_match = re.search(answer_pattern, text, re.DOTALL)
    
    think_content = think_match.group(1).strip() if think_match else None
    answer_content = answer_match.group(1).strip() if answer_match else None
    
    return think_content, answer_content

def postprocess_responses(resp: str) -> str:
    """Post-process response to ensure complete tags."""
    if "<answer>" in resp and "</answer>" not in resp:
        return resp.split("<answer>")[0]
    return resp

async def execute_code_block(code: str, executor: IntegratedExecutor, workspace: str) -> str:
    """Execute a Python code block and return the result."""
    try:
        # Create a temporary workspace for this execution
        temp_workspace = os.path.join(workspace, f"exec_{hash(code) % 10000}")
        os.makedirs(temp_workspace, exist_ok=True)
        
        # Create executor
        python_executor = executor.make_python_executor(temp_workspace)
        
        # Set resource limits
        resource_limits = executor.make_resource_limits(
            time_limit=ALGORITHMIC_CODING_CONFIGS["max_code_execution_time"],
            memory_limit=ALGORITHMIC_CODING_CONFIGS["max_code_memory"],
        )
        
        # Prepare and execute
        prepare_result = python_executor.prepare(temp_workspace, code)
        inner_cmd = prepare_result['inner_cmd']
        
        # Execute with empty input (no stdin)
        execution_result = python_executor.execute(
            inner_cmd=inner_cmd,
            stdin_str="",
            limits=resource_limits
        )
        
        # Format the result
        if execution_result.status.value == "SUCCESS":
            output = execution_result.stdout or ""
            return f"\n\n<code_execution_result>\n{output.strip()}\n</code_execution_result>\n\n"
        else:
            error_msg = execution_result.error_info or execution_result.stderr or "Unknown error"
            return f"\n\n<code_execution_result>\nError: {error_msg}\n</code_execution_result>\n\n"
            
    except Exception as e:
        return f"\n\n<code_execution_result>\nError: {str(e)}\n</code_execution_result>\n\n"

async def execute_predictions(prediction: str, executor: IntegratedExecutor, workspace: str) -> Tuple[str, bool]:
    """Execute predictions and return next observation and done flag."""
    # Extract code blocks from the prediction
    code_blocks = extract_code_blocks(prediction)
    
    if code_blocks:
        # Execute the last code block
        last_code = code_blocks[-1]
        next_obs = await execute_code_block(last_code, executor, workspace)
        done = False
    else:
        # Check if there's a final answer
        think_content, answer_content = extract_think_and_answer(prediction)
        if answer_content:
            next_obs = ""
            done = True
        else:
            next_obs = f"\nI need to write Python code to help solve this problem. Let me write some code to test my understanding or implement the solution.\n"
            done = False
    
    return next_obs, done

async def generate(args, sample: Sample, sampling_params) -> Sample:
    """Generate function for algorithmic coding with multi-turn code execution."""
    assert not args.partial_rollout, f"Partial rollout is not supported for this function at the moment."

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # Initialize executor
    executor = IntegratedExecutor(
        time_limit=ALGORITHMIC_CODING_CONFIGS["max_code_execution_time"],
        overall_time_limit=ALGORITHMIC_CODING_CONFIGS["max_overall_time"],
        memory_limit=ALGORITHMIC_CODING_CONFIGS["max_code_memory"],
        python_interpreter_path=ALGORITHMIC_CODING_CONFIGS["python_interpreter_path"],
        bwrap_path=ALGORITHMIC_CODING_CONFIGS["bwrap_path"],
    )
    
    # Create workspace
    workspace = tempfile.mkdtemp(prefix="algorithmic_coding_")

    # Handle partial rollout samples: continue generation from existing response
    prompt = sample.prompt
    prompt_tokens_ids = state.tokenizer(sample.prompt, add_special_tokens=False)["input_ids"]
    response = ""
    response_token_ids = []
    loss_masks = []
    
    for turn in range(ALGORITHMIC_CODING_CONFIGS["max_turns"]):
        payload = {
            "text": prompt + response,
            "sampling_params": sampling_params,
        }
        output = await post(url, payload, use_http2=args.use_http2)

        # abort
        if output["meta_info"]["finish_reason"]["type"] == "abort":
            sample.status = Sample.Status.ABORTED
            return sample

        cur_response = output["text"]
        cur_response = postprocess_responses(cur_response)

        cur_response_token_ids = state.tokenizer(cur_response, add_special_tokens=False)["input_ids"]
        response += cur_response
        response_token_ids += cur_response_token_ids
        loss_masks += [1] * len(cur_response_token_ids)

        if output["meta_info"]["finish_reason"]["type"] == "length":
            break

        next_obs, done = await execute_predictions(cur_response, executor, workspace)
        if done:
            break

        if next_obs:
            obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
            response += next_obs
            response_token_ids += obs_tokens_ids
            loss_masks += [0] * len(obs_tokens_ids)

    # Clean up workspace
    try:
        import shutil
        shutil.rmtree(workspace)
    except:
        pass

    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_masks = loss_masks
    
    match output["meta_info"]["finish_reason"]["type"]:
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED
        case "stop":
            sample.status = Sample.Status.COMPLETED

    return sample 