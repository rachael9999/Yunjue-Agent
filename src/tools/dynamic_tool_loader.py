# Copyright (c) 2026 Yunjue Tech
# SPDX-License-Identifier: Apache-2.0
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import tiktoken
from langchain_core.messages import HumanMessage, SystemMessage

from src.config.config import load_yaml_config
from src.services.llms.llm import create_llm, get_max_tokens
from src.utils.venv import ISOLATED_PYTHON_PATH
from src.tools.utils import extract_tool_info 
from src.schema.types import LLMType
logger = logging.getLogger(__name__)

# Cache for loaded dynamic tools: {file_path: (modification_time, tool)}
_dynamic_tools_cache: Dict[str, tuple[float, Any]] = {}

def count_text_tokens(text: str) -> int:
    """
    Count tokens in text using tiktoken encoding.

    Args:
        text: Text to count tokens for

    Returns:
        Number of tokens

    Raises:
        RuntimeError: If tiktoken encoder is not available
    """
    if not text:
        return 0

    encoder = tiktoken.get_encoding("cl100k_base")
    # Use tiktoken to encode and count tokens
    tokens = encoder.encode(text, allowed_special="all")
    return len(tokens)


def truncate_string_by_tokens(text: str, max_tokens: int) -> str:
    """
    Truncate a string to fit within the maximum token limit using tiktoken.
    Uses encode/decode to ensure valid truncation.

    Args:
        text: The text to truncate
        max_tokens: Maximum number of tokens allowed

    Returns:
        Truncated text (with "..." appended if truncated)

    Raises:
        RuntimeError: If tiktoken encoder is not available
    """
    if not text or max_tokens <= 0:
        return text

    encoder = tiktoken.get_encoding("cl100k_base")

    # Encode the text to tokens
    tokens = encoder.encode(text, allowed_special="all")
    current_tokens = len(tokens)

    if current_tokens <= max_tokens:
        return text

    # Reserve tokens for "..."
    ellipsis_tokens_list = encoder.encode("...", allowed_special="all")
    ellipsis_tokens = len(ellipsis_tokens_list)
    available_tokens = max_tokens - ellipsis_tokens

    if available_tokens <= 0:
        return "..."

    # Truncate tokens and decode back to text
    truncated_tokens = tokens[:available_tokens]
    truncated_text = encoder.decode(truncated_tokens)

    # Add ellipsis
    return truncated_text + " ...(context truncated)"


def collect_string_fields(data: Any) -> list[tuple[list, str]]:
    """
    Recursively collect all string fields with their paths.

    Returns:
        List of (path, string_value) tuples where path is a list of keys/indices
    """
    strings = []

    if isinstance(data, str):
        strings.append(([], data))
    elif isinstance(data, dict):
        for key, value in data.items():
            for path, str_val in collect_string_fields(value):
                strings.append(([key] + path, str_val))
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            for path, str_val in collect_string_fields(item):
                strings.append(([idx] + path, str_val))

    return strings


def set_nested_value(data: Any, path: list, value: Any) -> Any:
    """
    Set a value in a nested dict/list structure using a path.
    Returns a new structure without modifying the original.
    """
    if not path:
        return value

    if isinstance(data, dict):
        # Create a shallow copy and update the nested value
        result = dict(data)
        key = path[0]
        if key in result:
            result[key] = set_nested_value(result[key], path[1:], value)
        return result
    elif isinstance(data, list):
        # Create a new list and update the nested value
        result = list(data)
        idx = path[0]
        if isinstance(idx, int) and 0 <= idx < len(result):
            result[idx] = set_nested_value(result[idx], path[1:], value)
        return result
    else:
        # For other types, return as-is
        return data


def truncate_strings_in_dict(data: Any, max_tokens: int) -> Any:
    """
    Recursively truncate only string fields in a dict/list structure, preserving all other types.
    This is used for BaseModel truncation to keep structure intact.

    Strategy:
    1. Collect all string fields
    2. Calculate total tokens for all strings
    3. If exceeds limit, proportionally truncate each string field
    4. Replace strings in the structure

    Args:
        data: The data structure to process (dict, list, or any value)
        max_tokens: Maximum tokens allowed for all strings combined

    Returns:
        Data structure with truncated strings
    """
    # Collect all string fields with their paths
    string_fields = collect_string_fields(data)

    if not string_fields:
        # No strings to truncate
        return data

    # Calculate total tokens for all strings
    total_tokens = sum(count_text_tokens(str_val) for path, str_val in string_fields)

    # Estimate structure overhead (keys, brackets, etc.)
    structure_overhead = len(data) * 5 if isinstance(data, (dict, list)) else 0
    available_tokens = max(0, max_tokens - structure_overhead)

    if total_tokens <= available_tokens:
        # No truncation needed
        return data

    # Calculate truncation ratio
    ratio = available_tokens / total_tokens if total_tokens > 0 else 0

    logger.info(
        f"Truncating strings in BaseModel: {len(string_fields)} string fields, "
        f"total tokens: {total_tokens}, available: {available_tokens}, "
        f"truncation ratio: {ratio:.2%}"
    )

    # Truncate each string field proportionally
    result = data
    truncated_count = 0
    for path, str_val in string_fields:
        if not str_val:
            continue

        # Calculate target tokens for this string
        str_tokens = count_text_tokens(str_val)
        target_tokens = max(1, int(str_tokens * ratio))

        # Truncate the string
        truncated_str = truncate_string_by_tokens(str_val, target_tokens)

        # Log if string was actually truncated
        if len(truncated_str) < len(str_val):
            truncated_count += 1
            path_str = " -> ".join(str(p) for p in path) if path else "root"
            logger.debug(
                f"Truncated string at path '{path_str}': "
                f"{str_tokens} -> {count_text_tokens(truncated_str)} tokens "
                f"({len(str_val)} -> {len(truncated_str)} chars)"
            )

        # Set the truncated value back in the structure
        result = set_nested_value(result, path, truncated_str)

    if truncated_count > 0:
        logger.info(
            f"Truncated {truncated_count} out of {len(string_fields)} string fields in BaseModel structure"
        )

    return result


def truncate_response_locally(response: Any, max_tokens: int) -> Any:
    """
    Recursively truncate response content using local heuristics when LLM summarization is
    unavailable. Handles strings, dicts, lists, BaseModel instances, and other types.

    Args:
        response: The response to truncate (can be dict, list, str, BaseModel, etc.)
        max_tokens: Maximum number of tokens allowed

    Returns:
        Truncated response (preserves BaseModel type if possible, otherwise returns dict)
    """
    if max_tokens <= 0:
        return response

    # Handle BaseModel instances (Pydantic models)
    try:
        from pydantic import BaseModel as PydanticBaseModel

        if isinstance(response, PydanticBaseModel):
            # Get the original model class
            model_class = type(response)
            model_name = model_class.__name__

            # Convert to dict (Pydantic v2 uses model_dump, v1 uses dict)
            if hasattr(response, "model_dump"):
                response_dict = response.model_dump()
            elif hasattr(response, "dict"):
                response_dict = response.dict()
            else:
                # Fallback: convert to dict manually
                response_dict = json.loads(response.json())

            # Calculate original size for logging
            original_size = count_text_tokens(json.dumps(response_dict, ensure_ascii=False))

            # Only truncate string fields, keep structure intact
            truncated_dict = truncate_strings_in_dict(response_dict, max_tokens)

            # Calculate truncated size
            truncated_size = count_text_tokens(json.dumps(truncated_dict, ensure_ascii=False))

            if truncated_size < original_size:
                logger.info(
                    f"Truncated BaseModel '{model_name}': "
                    f"{original_size} -> {truncated_size} tokens "
                    f"({((original_size - truncated_size) / original_size * 100):.1f}% reduction)"
                )

            # Try to rebuild the BaseModel instance
            try:
                # Use model_validate for Pydantic v2, or parse_obj for v1
                if hasattr(model_class, "model_validate"):
                    return model_class.model_validate(truncated_dict)
                elif hasattr(model_class, "parse_obj"):
                    return model_class.parse_obj(truncated_dict)
                else:
                    # Fallback: try direct instantiation
                    return model_class(**truncated_dict)
            except Exception as e:
                # If validation fails, return original (shouldn't happen if only strings are truncated)
                logger.warning(
                    f"Failed to rebuild BaseModel '{model_name}' instance after truncation: {e}. "
                    f"Returning original model."
                )
                return response
    except ImportError:
        # pydantic not available, skip BaseModel handling
        pass
    except Exception as e:
        logger.debug(f"Error checking for BaseModel: {e}")

    if isinstance(response, str):
        return truncate_string_by_tokens(response, max_tokens)
    elif isinstance(response, dict):
        # Truncate dictionary by processing each key-value pair
        # Estimate tokens needed for JSON structure (keys, commas, braces, etc.)
        # Use a rough estimate: each key-value pair needs ~10 tokens for structure
        structure_tokens = len(response) * 10  # Rough estimate for JSON structure
        content_tokens = max_tokens - structure_tokens
        if content_tokens <= 0:
            # If structure itself is too large, return empty dict
            return {}

        truncated_dict = {}
        remaining_tokens = content_tokens
        for key, value in response.items():
            if remaining_tokens <= 0:
                break

            # Estimate tokens for key
            key_tokens = count_text_tokens(str(key)) + 5  # Include quotes and colon
            remaining_tokens -= key_tokens

            if remaining_tokens <= 0:
                break

            # Truncate value based on remaining tokens
            # Use recursive call to handle all types including BaseModel
            truncated_value = truncate_response_locally(value, remaining_tokens)
            # Estimate tokens used
            if isinstance(truncated_value, str):
                value_tokens = count_text_tokens(truncated_value)
            else:
                # For dict/list/BaseModel, estimate tokens via JSON serialization
                try:
                    # Handle BaseModel in JSON serialization
                    if hasattr(truncated_value, "model_dump"):
                        json_str = json.dumps(truncated_value.model_dump(), ensure_ascii=False)
                    elif hasattr(truncated_value, "dict"):
                        json_str = json.dumps(truncated_value.dict(), ensure_ascii=False)
                    else:
                        json_str = json.dumps(truncated_value, ensure_ascii=False)
                    value_tokens = count_text_tokens(json_str)
                except (TypeError, ValueError):
                    # If JSON serialization fails, estimate from string representation
                    value_tokens = count_text_tokens(str(truncated_value))
            remaining_tokens -= value_tokens
            truncated_dict[key] = truncated_value

        return truncated_dict
    elif isinstance(response, list):
        truncated_list = []
        remaining_tokens = max_tokens
        for item in response:
            if remaining_tokens <= 0:
                break
            truncated_item = truncate_response_locally(item, remaining_tokens)
            # Estimate tokens used
            if isinstance(truncated_item, str):
                item_tokens = count_text_tokens(truncated_item)
            else:
                # Handle BaseModel in JSON serialization
                try:
                    if hasattr(truncated_item, "model_dump"):
                        json_str = json.dumps(truncated_item.model_dump(), ensure_ascii=False)
                    elif hasattr(truncated_item, "dict"):
                        json_str = json.dumps(truncated_item.dict(), ensure_ascii=False)
                    else:
                        json_str = json.dumps(truncated_item, ensure_ascii=False)
                    item_tokens = count_text_tokens(json_str)
                except (TypeError, ValueError):
                    # If JSON serialization fails, estimate from string representation
                    item_tokens = count_text_tokens(str(truncated_item))
            remaining_tokens -= item_tokens
            truncated_list.append(truncated_item)
        return truncated_list
    else:
        # For other types, convert to string, truncate, and return as string
        str_repr = str(response)
        return truncate_string_by_tokens(str_repr, max_tokens)


def restore_response_type(original: Any, summary_payload: Any) -> Any:
    """Attempt to restore the original response type (e.g., BaseModel) from summarized data."""
    if summary_payload is None:
        return summary_payload

    try:
        from pydantic import BaseModel as PydanticBaseModel

        if isinstance(original, PydanticBaseModel) and isinstance(summary_payload, dict):
            model_class = type(original)
            try:
                if hasattr(model_class, "model_validate"):
                    return model_class.model_validate(summary_payload)
                if hasattr(model_class, "parse_obj"):
                    return model_class.parse_obj(summary_payload)
                return model_class(**summary_payload)
            except Exception as exc:
                logger.warning(
                    "Failed to rebuild BaseModel '%s' from summarized payload: %s. Returning summarized dict.",
                    model_class.__name__,
                    exc,
                )
                return summary_payload
    except ImportError:
        pass

    return summary_payload


def summarize_response_with_llm(
    original_response: Any, serialized_text: str, max_tokens: int, user_query: str
) -> Optional[Any]:
    """Use the configured summarize LLM to condense tool responses."""
    summarize_llm = create_llm(LLMType.SUMMARIZE)
    token_limit = get_max_tokens(LLMType.SUMMARIZE)

    system_prompt = (
        "You extract and filter information from tool responses based on step requirements. "
        "Your task is to extract only the information relevant to the step's title and description, "
        "while preserving the exact original content without any modifications, additions, or interpretations. "
        "Rules:\n"
        "1. Extract information that is relevant to the step's content (title and description).\n"
        "2. If information is ambiguous or unclear whether it's relevant, keep it (preserve when in doubt).\n"
        "3. Remove information that is clearly not useful or unrelated to the step.\n"
        "4. CRITICAL: All extracted content must be exactly the same as in the original response - "
        "no paraphrasing, no summarization, no changes to wording or values. Only remove irrelevant fields/keys, "
        "but keep the exact text/values of relevant fields.\n"
        "Respond with JSON only, maintaining the same structure as the original."
    )
    human_prompt = (
        "Task: {user_query}\n\n"
        "Maximum allowed tokens: {max_tokens}.\n"
        "Extract and return only the information from the original response that is relevant to the step above. "
        "Preserve all exact content from relevant fields - do not modify, summarize, or rephrase any text or values. "
        "Only remove fields/keys that are clearly unrelated to the step.\n\n"
        "Original JSON:\n{payload}"
    ).format(
        user_query=user_query,
        max_tokens=max_tokens,
        payload=serialized_text,
    )

    human_prompt = truncate_string_by_tokens(human_prompt, token_limit)
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]

    try:
        summary_message = summarize_llm.invoke(messages)
    except Exception as exc:
        logger.error(f"Failed to summarize dynamic tool response via LLM: {exc}")
        return None

    summary_content = getattr(summary_message, "content", summary_message)
    if isinstance(summary_content, list):
        summary_text = "".join(
            part if isinstance(part, str) else json.dumps(part, ensure_ascii=False)
            for part in summary_content
        )
    else:
        summary_text = str(summary_content)
    summary_text = summary_text.strip()
    if not summary_text:
        return None

    try:
        parsed_summary = json.loads(summary_text)
    except json.JSONDecodeError:
        parsed_summary = summary_text

    try:
        serialized_summary = (
            json.dumps(parsed_summary, ensure_ascii=False)
            if isinstance(parsed_summary, (dict, list))
            else str(parsed_summary)
        )
    except (TypeError, ValueError):
        serialized_summary = str(parsed_summary)

    summary_tokens = count_text_tokens(serialized_summary)
    if summary_tokens > max_tokens:
        logger.warning(
            "Summarize LLM output still exceeds token limit: %s tokens > %s limit",
            summary_tokens,
            max_tokens,
        )
        return None

    return restore_response_type(original_response, parsed_summary)


def truncate_response_by_tokens(response: Any, max_tokens: int, user_query: str) -> Any:
    """Use LLM summarization first, then fall back to heuristic truncation if needed."""
    if max_tokens <= 0:
        return response

    serializable_data = convert_to_json_serializable(response)
    try:
        serialized_text = json.dumps(serializable_data, ensure_ascii=False)
    except (TypeError, ValueError):
        serialized_text = str(serializable_data)

    original_tokens = count_text_tokens(serialized_text)
    if original_tokens <= max_tokens:
        return response

    token_limit = get_max_tokens(LLMType.SUMMARIZE)
    if original_tokens > token_limit:
        return truncate_response_locally(response, max_tokens)
    summarized = summarize_response_with_llm(response, serialized_text, max_tokens, user_query)
    if summarized is not None:
        return summarized

    logger.debug("Falling back to heuristic truncation for dynamic tool response (LLM summary unavailable).")
    return truncate_response_locally(response, max_tokens)


def get_max_response_tokens() -> int:
    """
    Get maximum response tokens for dynamic tools from configuration.

    Returns:
        Maximum number of tokens allowed for tool responses (default: 64000)
    """
    try:
        config_path = Path(__file__).parent.parent.parent / "conf.yaml"
        config = load_yaml_config(str(config_path))
        max_tokens = config.get("DYNAMIC_TOOL", {}).get("max_response_tokens")
        if max_tokens is not None and isinstance(max_tokens, int) and max_tokens > 0:
            return max_tokens
    except Exception as e:
        logger.debug(f"Failed to load max_response_tokens from config: {e}")

    # Default value
    return 64000


def convert_to_json_serializable(obj: Any) -> Any:
    """
    Recursively convert objects to JSON-serializable types.
    Handles Pydantic BaseModel instances, special Pydantic types (HttpUrl, EmailStr, etc.),
    and other non-serializable objects by converting them to strings or primitives.

    Args:
        obj: Any object (dict, list, Pydantic model, special types, or primitive)

    Returns:
        Object with all values converted to JSON-serializable types
    """
    # Check if it's a Pydantic BaseModel
    try:
        from pydantic import BaseModel as PydanticBaseModel

        if isinstance(obj, PydanticBaseModel):
            # Convert Pydantic model to dict recursively
            if hasattr(obj, "model_dump"):
                # Pydantic v2: use mode="json" to ensure special types are serialized
                try:
                    return convert_to_json_serializable(obj.model_dump(mode="json"))
                except TypeError:
                    # If mode="json" is not supported, fall back to default mode
                    return convert_to_json_serializable(obj.model_dump())
            elif hasattr(obj, "dict"):
                # Pydantic v1: use dict() method
                return convert_to_json_serializable(obj.dict())
            else:
                # Fallback: use json serialization
                try:
                    return json.loads(obj.json())
                except Exception:
                    # If JSON serialization fails, try to convert to dict manually
                    return convert_to_json_serializable(dict(obj))
    except ImportError:
        # pydantic not available, skip BaseModel handling
        pass
    except Exception:
        # If conversion fails, continue with other types
        pass

    # Handle dict
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}

    # Handle list
    if isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]

    # Handle tuple
    if isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(item) for item in obj)

    # Handle set
    if isinstance(obj, set):
        return list(convert_to_json_serializable(item) for item in obj)

    # Check if it's a basic JSON-serializable type
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj

    # For other types (like HttpUrl, EmailStr, etc.), try JSON serialization first
    # If that fails, convert to string representation
    try:
        # Test if it's directly JSON serializable
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        # If not serializable, try to get a string representation
        # For Pydantic special types like HttpUrl, this should work
        try:
            # Try using str() which often works for Pydantic special types
            str_repr = str(obj)
            # If it looks like a meaningful string representation (not just the class name), use it
            if str_repr and str_repr != type(obj).__name__:
                return str_repr
        except Exception:
            pass
        # Last resort: use repr or type name
        return str(obj)




def load_dynamic_tools(
    tools_directory: str = "dynamic_tools",
    user_query: str = "",
) -> List[Any]:
    """Load all dynamic tools from the specified directory.

    Uses caching to avoid reloading unchanged tools.

    Args:
        tools_directory: Path to the directory containing tool files

    Returns:
        List of loaded tool functions
    """
    tools = []
    tools_path = Path(tools_directory)
    tools_path.mkdir(parents=True, exist_ok=True)

    # Get all Python files in the directory
    python_files = list(tools_path.glob("*.py"))
    # print(python_files)

    for file_path in python_files:
        file_path_str = str(file_path)
        try:
            extraction_success, tool_info, extraction_error = extract_tool_info(file_path)
            if not extraction_success:
                logger.error(f"Failed to extract tool info for {file_path}: {extraction_error}")
                continue

            tool_meta = tool_info.get("tool_meta", {})
            if not tool_meta:
                logger.warning(f"Could not extract __TOOL_META__ from {file_path}")
                continue
            # Install dependencies for the tool if specified
            deps = tool_meta.get("dependencies", [])
            if deps:
                try:
                    logger.info(f"Installing dependencies for tool {file_path}: {deps}")
                    subprocess.run(
                        ["uv", "pip", "install", "--python", str(ISOLATED_PYTHON_PATH)] + deps, check=True
                    )
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to install dependencies for tool {file_path}: {e}")
                    continue
            # Check cache first
            current_mtime = file_path.stat().st_mtime
            cached_mtime, cached_tool = _dynamic_tools_cache.get(file_path_str, (None, None))

            if cached_mtime is not None and cached_mtime == current_mtime:
                # Use cached tool
                tools.append(cached_tool)
                continue
            tool_func = create_tool_from_module(file_path, user_query)
            if tool_func:
                tools.append(tool_func)
                # Cache the loaded tool
                _dynamic_tools_cache[file_path_str] = (current_mtime, tool_func)
                logger.debug(f"Cached tool from {file_path_str}")

        except Exception as e:
            print(f"Error loading tool from {file_path}: {e}")
            continue

    return tools


def create_tool_from_module(file_path: Path, user_query: str) -> Optional[Any]:
    """Create a LangChain tool from a Python module file.

    Args:
        file_path: Path to the module file

    Returns:
        LangChain tool function or None if creation fails
    """
    try:
        file_path = Path(file_path)

        loader_code = f"""
import sys
import json
import importlib.util
from pathlib import Path

file_path = Path(r"{file_path}")

try:
    spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
    if spec is None or spec.loader is None:
        print(json.dumps({{"error": "Could not load spec"}}))
        sys.exit(0)

    module = importlib.util.module_from_spec(spec)
    sys.modules[file_path.stem] = module
    spec.loader.exec_module(module)

    tool_meta = getattr(module, "__TOOL_META__", {{}})
    input_model = getattr(module, "InputModel", None)
    
    print(json.dumps({{
        "tool_meta": tool_meta,
        "input_model": input_model.model_json_schema()
    }}, ensure_ascii=False))

except Exception as e:
    print(json.dumps({{"error": str(e)}}))
"""
        try:
            proc = subprocess.run(
                [str(ISOLATED_PYTHON_PATH), "-c", loader_code],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            result = json.loads(proc.stdout.strip())
            if "error" in result:
                logger.warning(f"Error loading tool {file_path}: {result['error']}")
                return None

            tool_meta = result.get("tool_meta", {})
            input_model = result.get("input_model")
            if not tool_meta:
                return None

        except Exception as e:
            logger.error(f"Subprocess error loading tool {file_path}: {e}")
            return None

        # Import langchain_core.tools only when needed
        from langchain_core.tools import tool

        def tool_executable(**kwargs) -> Dict[str, Any]:
            """这个文档字符串将被 @tool 用作工具的描述。"""
            try:
                # Convert any Pydantic models in kwargs to dicts before serialization
                # LangChain may pass Pydantic model instances when args_schema is a Pydantic model
                kwargs_dict = convert_to_json_serializable(kwargs)
                # Execute the tool in isolated environment
                # Use default=str as fallback for any remaining non-serializable objects
                input_json = json.dumps(kwargs_dict, default=str, ensure_ascii=False)
                file_path_str = json.dumps(str(file_path))
                code = f"""
import json
import sys
import importlib.util

# Get file path
file_path = {file_path_str}

# Load the module using importlib
spec = importlib.util.spec_from_file_location("dynamic_module", file_path)
if spec is None or spec.loader is None:
    raise RuntimeError("Failed to create module spec")

module = importlib.util.module_from_spec(spec)
sys.modules["dynamic_module"] = module
spec.loader.exec_module(module)

# Read input from stdin
input_data = json.loads(sys.stdin.read())

# Create input instance and run
input_instance = module.InputModel(**input_data)
result = module.run(input_instance)

if hasattr(result, "model_dump"):
    result_json = json.dumps(result.model_dump(), ensure_ascii=False)
elif hasattr(result, "dict"):
    result_json = json.dumps(result.dict(), ensure_ascii=False)
elif isinstance(result, (dict, list)):
    result_json = json.dumps(result, ensure_ascii=False)
else:
    result_json = str(result)
# Print result as JSON
print(result_json)
"""
                try:
                    proc = subprocess.run(
                        [str(ISOLATED_PYTHON_PATH), "-c", code],
                        input=input_json,
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=120,
                    )
                    result = json.loads(proc.stdout.strip())
                except subprocess.TimeoutExpired as e:
                    logger.error(f"Tool execution timed out: {e}")
                    return {"error": {"type": "tool_execution_timeout", "message": str(e)}}
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error executing tool in isolated environment: {e}")
                    logger.error(f"Stderr output: {e.stderr}")
                    return {"error": {"type": "subprocess_execution_error", "message": e.stderr}}
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing tool output: {e}")
                    return {"error": {"type": "parsing_tool_error", "message": str(e)}}

                # Truncate response if it exceeds token limit
                max_tokens = get_max_response_tokens()
                tool_name = tool_meta.get("name", "unknown")
                if max_tokens > 0:
                    # Handle BaseModel serialization for token counting
                    if hasattr(result, "model_dump"):
                        result_json = json.dumps(result.model_dump(), ensure_ascii=False)
                    elif hasattr(result, "dict"):
                        result_json = json.dumps(result.dict(), ensure_ascii=False)
                    elif isinstance(result, (dict, list)):
                        result_json = json.dumps(result, ensure_ascii=False)
                    else:
                        result_json = str(result)

                    original_tokens = count_text_tokens(result_json)
                    if original_tokens > max_tokens:
                        logger.warning(
                            f"[TRUNCATION] Dynamic tool '{tool_name}' response exceeds token limit: "
                            f"{original_tokens} tokens > {max_tokens} limit. Starting truncation..."
                        )
                        result = truncate_response_by_tokens(result, max_tokens, user_query)

                        # Recalculate tokens after truncation
                        if hasattr(result, "model_dump"):
                            truncated_json = json.dumps(result.model_dump(), ensure_ascii=False)
                        elif hasattr(result, "dict"):
                            truncated_json = json.dumps(result.dict(), ensure_ascii=False)
                        elif isinstance(result, (dict, list)):
                            truncated_json = json.dumps(result, ensure_ascii=False)
                        else:
                            truncated_json = str(result)

                        truncated_tokens = count_text_tokens(truncated_json)
                        reduction_pct = (
                            (original_tokens - truncated_tokens) / original_tokens * 100
                            if original_tokens > 0
                            else 0
                        )
                        logger.info(
                            f"[TRUNCATION] Dynamic tool '{tool_name}' response truncated: "
                            f"{original_tokens} -> {truncated_tokens} tokens "
                            f"({reduction_pct:.1f}% reduction, saved {original_tokens - truncated_tokens} tokens)"
                        )

                return result
            except Exception as e:
                return {"error": {"type": "tool_execution_error", "message": str(e)}}

        tool_executable.__name__ = os.path.basename(file_path).split(".")[0]
        tool_executable.__doc__ = tool_meta.get("description", "A dynamically loaded tool")

        langchain_tool = tool(tool_executable)
        langchain_tool.args_schema = input_model

        return langchain_tool

    except Exception as e:
        print(f"Error creating tool from {file_path}: {e}")
        return None


def get_dynamic_tools(dynamic_tools_dir: str, user_query) -> List[Any]:
    """Get all dynamic tools from the default directory."""
    return load_dynamic_tools(dynamic_tools_dir, user_query)
