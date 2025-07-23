import sys
import json
import pickle
import os
import zlib
import base64
import re
from typing import List, Dict, Any, Union, Optional
import time
import subprocess
import shutil

from loguru import logger
from tqdm import tqdm

def get_cur_time_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def extract_code(
    text: Optional[str],
    language: str = "auto",
    allow_no_language_label: bool = True,
    verbose: bool = True,
    return_code_list: bool = False,
    raise_exception: bool = True
) -> Union[str, List[str], None]:
    """
    Extract code blocks from the given text.

    Args:
        text (Optional[str]): The text to extract code from.
        language (str): The language of the code to extract. Use "auto" to detect automatically.
        allow_no_language_label (bool): Allow extraction from unlabeled code blocks.
        verbose (bool): Print verbose information.
        return_code_list (bool): If True, return all code blocks as a list. Otherwise, return the longest code block.
        raise_exception (bool): Raise an exception if extraction fails. Otherwise, return None or [].

    Returns:
        Union[str, List[str], None]: Extracted code(s), or None/[] if not found.
    """
    if text is None:
        if raise_exception:
            raise ValueError("Text is None.")
        return [] if return_code_list else None

    language_patterns = {
        'python': [r"```python(.*?)```"],
        'cpp': [r"```cpp(.*?)```", r"```c\+\+(.*?)```"],
        'json': [r"```json(.*?)```"],
    }

    def find_code_blocks(patterns):
        codes = []
        for pattern in patterns:
            codes.extend(re.findall(pattern, text, re.DOTALL))
        return codes

    code_blocks = []
    if language in language_patterns:
        code_blocks = find_code_blocks(language_patterns[language])
    elif language == 'auto':
        for patterns in language_patterns.values():
            code_blocks.extend(find_code_blocks(patterns))
    else:
        if raise_exception:
            raise ValueError(f"Unsupported language: {language}")
        return [] if return_code_list else None

    if not code_blocks:
        if not allow_no_language_label:
            if raise_exception:
                raise Exception("Failed to extract code.")
            return [] if return_code_list else None
        generic_pattern = r"```(.*?)```"
        if verbose:
            logger.warning(f"Failed to extract code. Retrying with generic pattern: {generic_pattern}.")
        code_blocks = re.findall(generic_pattern, text, re.DOTALL)
        if not code_blocks:
            if raise_exception:
                raise Exception("Failed to extract code.")
            return [] if return_code_list else None

    code_blocks = [block.strip() for block in code_blocks]
    if return_code_list:
        return code_blocks
    if not code_blocks:
        return None
    longest_code = max(code_blocks, key=len)
    if verbose and len(code_blocks) > 1 and longest_code != code_blocks[-1]:
        logger.warning("The longest code block is not the last one. There might be extraction errors.")
    return longest_code

def encode_testcases(testcases: List[Dict[str, str]]) -> str:
    """
    According to LiveCodeBench, private test cases should be encoded.
    """
    json_str = json.dumps(testcases)
    pickled_data = pickle.dumps(json_str)
    compressed_data = zlib.compress(pickled_data)
    encoded_testcases = base64.b64encode(compressed_data).decode('utf-8')
    return encoded_testcases

def decode_testcases(encoded_testcases: str) -> List[Dict[str, str]]:
    return json.loads(
        pickle.loads(
            zlib.decompress(
                base64.b64decode(encoded_testcases.encode("utf-8"))
            )
        )
    )

def make_dir_for_file(file_path: str) -> None:
    """
    Make directory for file_path
    """
    dir_name = os.path.dirname(file_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def save_json(data: Any, file_path: str) -> None:
    """
    Save data to file_path with json format
    """
    make_dir_for_file(file_path)
    with open(file_path, "w") as f:
        json.dump(data, f)
        
def load_json(file_path: str) -> Any:
    """
    Load data from file_path with json format
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def load_json_line(file_path: str, use_eval: bool = False, verbose: bool = False) -> List[Any]:
    """
    Load data from file_path with json line (i.e., regard each line as a json file) format
    """
    data = []
    with open(file_path, "r") as f:
        for line in tqdm(f, disable=not verbose):
            try:
                if use_eval:
                    data.append(eval(line))
                else:
                    data.append(json.loads(line))
            except:
                if verbose:
                    print(f'[ERROR] broken line: {line[:20]}')
                continue
    return data

def save_json_line(data: List[Any], file_path: str, do_append: bool=False) -> None:
    """
    Save data to file_path with json line (i.e., regard each line as a json file) format
    """
    make_dir_for_file(file_path)
    mode = 'a' if do_append else 'w'
    with open(file_path, mode) as f:
        for d in data:
            f.write(json.dumps(d) + '\n')

def save_pickle(data: Any, file_path: str) -> None:
    """
    Save data from file_path with pickle format
    """
    make_dir_for_file(file_path)
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

def load_pickle(file_path: str) -> Any:
    """
    Load data from file_path with pickle format
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data
    
def force_delete_folder(folder_path: str):
    """
    Force delete a folder.
    """
    if not os.path.exists(folder_path):
        logger.warning(f"Folder {folder_path} does not exist.")
        return
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if file.startswith(".nfs"):
                    result = subprocess.run(["lsof", file_path], capture_output=True, text=True)
                    if result.stdout:
                        for line in result.stdout.strip().split("\n")[1:]:
                            pid = int(line.split()[1])
                            subprocess.run(["kill", "-9", str(pid)])
                os.unlink(file_path)
                logger.debug(f"{file_path} has been deleted.")
            except Exception as e:
                logger.warning(f"Falied to delete file {file_path} with os.unlink. Error: {e}")
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            try:
                shutil.rmtree(dir_path)
            except Exception as e:
                logger.warning(f"Falied to delete sub-folder {dir_path} with shutil.rmtree. Error: {e}")
    try:
        shutil.rmtree(folder_path)
    except Exception as e:
        logger.warning(f"Falied to delete folder {folder_path} with shutil.rmtree. Error: {e}")
    try:
        subprocess.run(["rm", "-rf", folder_path], check=False)
    except Exception as e:
        logger.warning(f"Falied to delete folder {folder_path} with rm -rf. Error: {e}")
    
    if os.path.exists(folder_path):
        logger.error(f"Failed to delete folder {folder_path}.")
    else:
        logger.debug(f"Folder {folder_path} has been deleted successfully.")