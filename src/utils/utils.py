# Copyright (c) 2026 Yunjue Tech
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import logging
import os
import re
import subprocess
from typing import Any, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from src.schema.types import ToolExecutionRecord, ToolRequest, StepToolAnalysis, ResponseAnalysis, LLMType
from src.services.llms.llm import create_llm, get_max_tokens
from src.prompts.loader import prompt_loader
from src.tools.dynamic_tool_loader import (
    count_text_tokens,
    get_dynamic_tools,
)
from src.tools.utils import extract_tool_info
from src.utils.venv import ISOLATED_PYTHON_PATH

logger = logging.getLogger(__name__)

# 20-minute safety cutoff for Codex exec calls
CODEX_EXEC_TIMEOUT_SECONDS = 20 * 60
# Max retries when parsing LLM analysis outputs
ANALYSIS_MAX_RETRIES = 3

async def call_codex_exec(prompt: str, output_file: str = None) -> tuple[str, bool]:
    """
    Call Codex exec to generate code based on the prompt.
    Uses async subprocess to enable true concurrent execution.

    Args:
        prompt: The prompt to send to Codex exec
        output_file: Optional file path to save the generated code

    Returns:
        Tuple of (generated_code, success)
    """
    try:
        # Build codex exec command
        # Use --full-auto to allow file editing and network access
        codex_profile = os.environ.get("CODEX_PROFILE", None)
        command = ["codex", "exec", "--dangerously-bypass-approvals-and-sandbox"]
        if codex_profile:
            command += ["--profile", codex_profile]

        logger.info(f"Calling codex exec with prompt length: {len(prompt)}")
        input_tokens = count_text_tokens(prompt)
        # Create async subprocess for true concurrent execution
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Send prompt to stdin and wait for completion
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=prompt.encode("utf-8")),
                timeout=CODEX_EXEC_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.error("Codex exec timed out after %s seconds", CODEX_EXEC_TIMEOUT_SECONDS)
            process.kill()
            await process.communicate()
            return "", False
        output_tokens = count_text_tokens(stdout.decode("utf-8") + stderr.decode("utf-8"))
        logger.info(f"Codex exec input tokens: {input_tokens}, output tokens: {output_tokens}")
        # Decode output
        generated_code = stdout.decode("utf-8").strip() if stdout else ""
        error_output = stderr.decode("utf-8").strip() if stderr else ""

        logger.info(f"Codex exec stdout: {generated_code}")
        if error_output:
            logger.debug(f"Codex exec stderr: {error_output}")

        if process.returncode != 0:
            error_msg = f"Codex exec failed with return code {process.returncode}"
            logger.error(error_msg)
            return "", False

        if not generated_code:
            logger.warning("Codex exec returned empty output")

        # If output_file is specified, save the code to file
        if output_file:
            try:
                # extract the last ```python ... ``` code block
                code_blocks = re.findall(r"```python\s*(.*?)\s*```", generated_code, re.DOTALL)
                if not code_blocks:
                    logger.warning("No python code block found in Codex exec output")
                    return "", False

                generated_code = code_blocks[-1].strip()
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(generated_code)
                logger.info(f"Generated code saved to: {output_file}")
                extraction_success, tool_info, extraction_error = extract_tool_info(output_file)
                if not extraction_success:
                    logger.error(f"Failed to extract tool info for {output_file}: {extraction_error}")
                    os.remove(output_file)
                    return "", False

                tool_meta = tool_info.get("tool_meta", {})
                if not tool_meta:
                    logger.warning(f"Could not extract __TOOL_META__ from {output_file}")
                    os.remove(output_file)
                    return "", False
                # Install dependencies for the tool if specified
                deps = tool_meta.get("dependencies", [])
                if deps:
                    try:
                        logger.info(f"Installing dependencies for tool {output_file}: {deps}")
                        subprocess.run(
                            ["uv", "pip", "install", "--python", str(ISOLATED_PYTHON_PATH)] + deps, check=True
                        )
                    except subprocess.CalledProcessError as e:
                        error_message = f"Failed to install dependencies for tool {output_file}: {e}"
                        logger.error(f"Failed to install dependencies for tool {output_file}: {e}")
                        os.remove(output_file)
                        return error_message, False
            except Exception as e:
                logger.error(f"Failed to save code to {output_file}: {e}")
                return "", False

        return generated_code, True

    except FileNotFoundError:
        logger.error("Codex exec command not found. Please ensure 'codex' is installed and in PATH")
        return "", False
    except Exception as e:
        logger.error(f"Error calling codex exec: {type(e).__name__}: {str(e)}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        return "", False


def generate_and_run_tests(tool_filename: str, historical_call_records: List[Any]) -> tuple[bool, List[Any]]:
    """Generate tests with Codex exec and execute them against the tool file.

    Returns:
        Tuple containing (success_flag, results)
    """

    tool_basename = os.path.basename(tool_filename)
    tool_name_without_ext = os.path.splitext(tool_basename)[0]
    extraction_success, tool_info, extraction_error = extract_tool_info(tool_filename)
    if not extraction_success:
        logger.error(f"Failed to extract tool info for {tool_filename}: {extraction_error}")
        return False, historical_call_records

    tool_meta = tool_info.get("tool_meta", {})
    file_path_str = json.dumps(str(tool_filename))
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
import json
import sys
input_data = json.loads(sys.stdin.read())
input_instance = module.InputModel(**input_data)
result = module.run(input_instance)
print(result)
"""
    if not tool_meta:
        logger.warning(f"Could not extract __TOOL_META__ from {tool_filename}")

    # Install dependencies for the tool if specified
    deps = tool_meta.get("dependencies", [])
    if deps:
        try:
            logger.info(f"Installing dependencies for tool {tool_filename}: {deps}")
            subprocess.run(["uv", "pip", "install", "--python", str(ISOLATED_PYTHON_PATH)] + deps, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies for tool {tool_filename}: {e}")
            return False, historical_call_records

    results = []
    try:
        for historical_call_record in historical_call_records:
            arguments = historical_call_record.arguments
            input_data = json.dumps(arguments)
            try:
                proc = subprocess.run(
                    [str(ISOLATED_PYTHON_PATH), "-c", code],
                    input=input_data,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=300,
                )
                print("proc.stdout: ", proc.stdout)
            except subprocess.TimeoutExpired as e:
                logger.error(f"Tool execution timed out: {e}")
                results.append(
                    ToolExecutionRecord(
                        tool_name=tool_name_without_ext,
                        tool_call_id=historical_call_record.tool_call_id,
                        arguments=arguments,
                        result=None,
                        error=f"Tool execution timed out: {e}",
                    )
                )
            except subprocess.CalledProcessError as e:
                results.append(
                    ToolExecutionRecord(
                        tool_name=tool_name_without_ext,
                        tool_call_id=historical_call_record.tool_call_id,
                        arguments=arguments,
                        result=None,
                        error=f"Failed to run tool: {e.stderr}",
                    )
                )

    except Exception as e:  # pragma: no cover - defensive logging
        import traceback

        logger.info(
            f"Unexpected error during tool evaluation: {type(e).__name__}: {e}\nTraceback: {traceback.format_exc()}"
        )
        return False, results

    if results == []:
        return True, []
    else:
        return False, results


def get_preset_tools() -> List[Any]:
    """
    Get all preset/built-in tools that are always available.

    Returns:
        List of preset tool objects
    """
    from src.tools.image_text_query import image_text_query

    preset_tools = [
        image_text_query,
    ]
    return preset_tools


async def analyze_task_tools(
    user_query: str,
    dynamic_tools_dir: str,
    dynamic_tools_public_dir: str,
    failure_report: Optional[str] = None,
    additional_tool_requests: Optional[List] = None,
) -> tuple[list[str], str, list[ToolRequest]]:
    """
    Analyze the step and return the required tools, guidance, and tool requests.

    Args:
        state: Current state containing user query and other context

    Returns:
        Tuple of (required_tool_names, tool_usage_guidance, tool_requests):
        - required_tool_names: List of tool names from available tools that are needed
        - tool_usage_guidance: High-level outline produced by the analyzer
        - tool_requests: List of ToolRequest objects if new tools are needed
    """
    logger.info(f"Analyzing tools for task: {user_query}")

    # Load all available tools (both dynamic and preset)
    dynamic_tools_private = get_dynamic_tools(dynamic_tools_dir, user_query)
    dynamic_tools_public = get_dynamic_tools(dynamic_tools_public_dir, user_query)
    preset_tools = get_preset_tools()
    available_tools = preset_tools + dynamic_tools_private + dynamic_tools_public

    # Extract tool information for the prompt
    # For tool selection, we only need name and description
    # Schema information is not necessary at this stage and would make the prompt too long
    tool_info_list = []
    for tool in available_tools:
        try:
            # Get tool name and description
            tool_name = getattr(tool, "name", "unknown")
            tool_description = getattr(tool, "description", "") or getattr(tool, "__doc__", "")
            tool_args_schema = getattr(tool, "args_schema", {})
            tool_info_list.append(
                {
                    "name": tool_name,
                    "description": tool_description,
                    "input_schema": tool_args_schema,
                }
            )
        except Exception as e:
            logger.warning(f"Failed to extract info from tool: {e}")
            continue

    # Build the prompt
    prompt_content = prompt_loader.get_prompt(
        "step_tool_analyzer.md",
        **{
            "failure_report": failure_report,
            "user_query": user_query,
            "available_tools": tool_info_list,
            "additional_tool_requests": additional_tool_requests,
        }
    )

    llm = create_llm(LLMType.BASIC).with_structured_output(
        StepToolAnalysis,
        method="json_mode",
    )

    messages = [HumanMessage(content=prompt_content)]
    try:
        analysis_result = await llm.ainvoke(messages)
        required_tool_names = analysis_result.required_tool_names or []
        tool_usage_guidance = analysis_result.tool_usage_guidance or ""
        tool_requests = analysis_result.tool_requests or []

        logger.info(
            "Analysis complete: %s existing tools, %s new tool requests",
            len(required_tool_names),
            len(tool_requests),
        )

        return required_tool_names, tool_usage_guidance, tool_requests
    except Exception as e:
        logger.error(
            "Error analyzing step tools: %s",
            e,
        )
        raise

async def summarize_context(
    user_query: str, history_tool_executions: List[ToolExecutionRecord], context_summary: str, is_recur_limit_exceeded: bool = False
):
    """Summarize the context of the current task and the history tool executions.

    Args:
        user_query: The user query
        history_tool_executions: The history tool executions
        context_summary: The context summary
    """
    tool_execution_histories = (
        transform_tool_executions_to_str(history_tool_executions) if history_tool_executions else ""
    )
    tmp_context_summary = context_summary
    for tool_execution_history in tool_execution_histories:
        prompt_content = prompt_loader.get_prompt(
            "context_summarizer.md", 
            **{
                "user_query": user_query,
                "tool_execution_history": tool_execution_history,
                "context_summary": tmp_context_summary,
                "enable_tool_usage_feedback": is_recur_limit_exceeded,
            }
        )

        llm = create_llm(LLMType.BASIC)
        messages = [HumanMessage(content=prompt_content)]
        
        try:
            response = await llm.ainvoke(messages)
        except Exception as e:
            logger.error(f"LLM ainvoke failed: {e}")
            raise
        
        response = getattr(response, "content", response)
        if isinstance(response, list):
            summary = "".join(
                part if isinstance(part, str) else json.dumps(part, ensure_ascii=False) for part in response
            )
        else:
            summary = str(response)
        logger.info(f"Summarized context from {len(tool_execution_history)} to {len(summary)}")
        tmp_context_summary = summary
    return tmp_context_summary


async def filter_tools_by_names(
    names: List[str], dynamic_tools_private: str, dynamic_tools_public: str, user_query: str
) -> List[Any]:
    """
    Filter tools (both preset and dynamic) by their names.

    Args:
        names: List of tool names to filter

    Returns:
        List of tool objects matching the names
    """
    from src.tools.dynamic_tool_loader import get_dynamic_tools

    if not names:
        return []

    # Load all tools (both preset and dynamic)
    preset_tools = get_preset_tools()

    dynamic_private_tools = get_dynamic_tools(dynamic_tools_private, user_query)
    dynamic_public_tools = get_dynamic_tools(dynamic_tools_public, user_query)
    dynamic_tools = {tool.name: tool for tool in dynamic_public_tools}
    # Private tools override public tools with the same name
    dynamic_tools.update({tool.name: tool for tool in dynamic_private_tools})
    all_tools = preset_tools + list(dynamic_tools.values())

    # Create a set for faster lookup
    name_set = set(names)

    # Filter tools by name
    filtered_tools = []
    for tool in all_tools:
        try:
            tool_name = getattr(tool, "name", None)
            if tool_name and tool_name in name_set:
                filtered_tools.append(tool)
        except Exception as e:
            logger.warning(f"Failed to get name from tool: {e}")
            continue

    if len(filtered_tools) != len(names):
        logger.warning(f"Filtered {len(filtered_tools)} tools from {len(names)} requested names")
        logger.warning(f"Filtered tools: {filtered_tools}")
        logger.warning(f"Requested names: {names}")
        not_found_names = set(names) - set([tool.name for tool in filtered_tools])
        for name in not_found_names:
            public_tool = f"{dynamic_tools_public}/{name}.py"
            if os.path.exists(public_tool):
                os.remove(public_tool)
            private_tool = f"{dynamic_tools_private}/{name}.py"
            if os.path.exists(private_tool):
                os.remove(private_tool)
        logger.warning(f"Removed public and private tools: {not_found_names}")
    else:
        logger.info(f"Filtered {len(filtered_tools)} tools from {len(names)} requested names")
    return filtered_tools


def transform_tool_executions_to_str(
    tool_executions: List[Any], current_context_summary: str = ""
) -> List[str]:
    """Transform tool executions to a string.

    Args:
        tool_executions: List of ToolExecutionRecord objects

    Returns:
        A string representation of the tool executions
    """

    if current_context_summary:
        tool_history_parts = [current_context_summary]
    else:
        tool_history_parts = []
    for i, exec_record in enumerate(tool_executions, 1):
        tool_name = getattr(exec_record, "tool_name", "unknown")
        arguments = getattr(exec_record, "arguments", {})
        result = getattr(exec_record, "result", None)
        
        error = getattr(exec_record, "error", None)

        tool_call_str = f"### Tool Call {i}: {tool_name}\n\n"
        tool_call_str += (
            f"**Arguments:**\n```json\n{json.dumps(arguments, indent=2, ensure_ascii=False)}\n```\n\n"
        )

        if error:
            tool_call_str += f"**Error:** {error}\n\n"
        elif result:
            # Truncate very long results
            result_str = str(result)
            tool_call_str += f"**Result:**\n```\n{result_str}\n```\n\n"
        else:
            tool_call_str += "**Status:** ⏳ Pending/Unknown\n\n"

        tool_history_parts.append(tool_call_str)

    llm_token_limit = get_max_tokens(LLMType.BASIC)

    tool_execution_histories: List[str] = []
    current_parts: List[str] = []
    current_tokens = 0

    for part in tool_history_parts:
        part_tokens = count_text_tokens(part)

        if current_parts and (current_tokens + part_tokens > llm_token_limit):
            tool_execution_histories.append("\n".join(current_parts))
            current_parts = [part]
            current_tokens = part_tokens
        else:
            current_parts.append(part)
            current_tokens += part_tokens

    if current_parts:
        tool_execution_histories.append("\n".join(current_parts))

    return tool_execution_histories


async def analyze_response(
    pending_response: str,
) -> tuple[bool, str]:
    """Analyze worker failure and generate a comprehensive report.

    Args:
        pending_response: The response content from the worker

    Returns:
        A tuple of (diagnosis, reason)
    """

    # Format tool execution history

    # Load and render the analysis prompt template
    prompt_content = prompt_loader.get_prompt("analyze_response.md", **{
        "pending_response": pending_response,
    })

    # Use worker LLM for analysis (or default to basic if not configured)
    llm = create_llm(LLMType.BASIC).with_structured_output(
        ResponseAnalysis,
        method="json_mode",
    )

    messages = [HumanMessage(content=prompt_content)]
    last_response_content = ""
    try:
        analysis_result = llm.invoke(messages)
        status = (analysis_result.status or "RETRY").upper()
        reason = analysis_result.reason or ""
        last_response_content = f"status={status}, reason={reason}"

        tool_diagnosis = status == "FINISH"
        return tool_diagnosis, f"#### Reponse Failure Report\n{reason}"
    except Exception as e:
        logger.error(
            "Failed to analyze response: %s",
            e,
        )

    # Fallback to assuming retry if parsing fails, or maybe check for simple string
    if "FINISH" in last_response_content and "RETRY" not in last_response_content:
        return True, ""
    return False, f"#### Response Failure Report\n{last_response_content}"

async def tool_enhancement(
    tool_filename: str,
    historical_call_records: List[Any],
    dynamic_tools_dir: str,
) -> str:
    """Enhance a tool based on a suggestion.

    Args:
        tool_filename: The filename of the tool to enhance
        historical_call_records: The historical call records that either led to an exception or resulted in sub-optimal/noisy output
        dynamic_tools_dir: The private directory of the dynamic tools
    """
    # Load the tool file
    retry_num = 0
    tool_name = os.path.basename(tool_filename).split(".")[0]

    target_tool_filename = os.path.join(dynamic_tools_dir, f"{tool_name}.py")
    with open(tool_filename, "r", encoding="utf-8") as f:
        tool_code = f.read()

    while retry_num < 3:
        logger.info(f"Enhancing tool {target_tool_filename} (attempt {retry_num + 1}/3)")
        historical_call_records_str = (
            "\n".join(transform_tool_executions_to_str(historical_call_records))
            if historical_call_records
            else ""
        )
        # Get proxy URL from environment variable
        proxy_url = os.environ.get("PROXY_URL", None)
        tool_enhancement_prompt = prompt_loader.get_prompt(
            "tool_enhancement.md",
            **{
                "original_tool_code": tool_code,
                "historical_call_records": historical_call_records_str,
                "proxy_url": proxy_url,
            }
        )

        enhanced_tool_code, success = await call_codex_exec(tool_enhancement_prompt, target_tool_filename)
        if not success:
            logger.error(f"Failed to enhance tool {target_tool_filename}")
            if os.path.exists(target_tool_filename):
                os.remove(target_tool_filename)
            retry_num += 1
            continue

        success, results = generate_and_run_tests(target_tool_filename, historical_call_records)
        if not success:
            logger.error(f"Failed to generate and run tests for enhanced tool {target_tool_filename}")
            if os.path.exists(target_tool_filename):
                os.remove(target_tool_filename)
            historical_call_records = results
            tool_code = enhanced_tool_code
            retry_num += 1
            continue
        else:
            logger.info(f"Successfully enhanced tool {target_tool_filename}")
            break

    if retry_num == 3:
        logger.error(f"Failed to enhance tool {tool_name} after 3 retries")
        return ""
    else:
        return target_tool_filename


def extract_key_findings_and_conclusion(worker_response: str) -> str:
    """Extract 'Key Findings & Evidence' and 'Final Conclusion' sections from worker response.

    According to worker.md format:
    - ## 3. Key Findings & Evidence
    - ## 4. Final Conclusion

    Args:
        worker_response: The full worker response content

    Returns:
        Extracted content containing Key Findings & Evidence and Final Conclusion sections
    """
    if not worker_response:
        return ""

    # Try to find the sections using regex
    # Match "## 3. Key Findings & Evidence" or variations (with flexible spacing and punctuation)
    # Pattern matches: "## 3. Key Findings & Evidence", "##3. Key Findings & Evidence", etc.
    key_findings_pattern = (
        r"##\s*2\.\s*Key\s+Findings\s*(?:&\s*Evidence)?\s*\n(.*?)(?=##\s*3\.\s*Final\s+Conclusion|$)"
    )
    # Match "## 4. Final Conclusion" or variations
    conclusion_pattern = r"##\s*3\.\s*Final\s+Conclusion\s*\n(.*?)(?=##\s*[1-5]\.|#\s+Notes|#\s*Notes\s*&|$)"

    key_findings_match = re.search(key_findings_pattern, worker_response, re.IGNORECASE | re.DOTALL)
    conclusion_match = re.search(conclusion_pattern, worker_response, re.IGNORECASE | re.DOTALL)

    extracted_parts = []

    if key_findings_match:
        key_findings_content = key_findings_match.group(1).strip()
        if key_findings_content:
            extracted_parts.append("## Key Findings & Evidence\n\n" + key_findings_content)

    if conclusion_match:
        conclusion_content = conclusion_match.group(1).strip()
        if conclusion_content:
            extracted_parts.append("## Final Conclusion\n\n" + conclusion_content)

    if extracted_parts:
        result = "\n\n".join(extracted_parts)
        logger.info(
            f"Extracted key findings and conclusion from worker response (length: {len(result)} chars)"
        )
        return result
    else:
        # If sections not found, return original response (fallback)
        logger.warning(
            "Could not find 'Key Findings & Evidence' or 'Final Conclusion' sections in worker response. Returning original response."
        )
        return worker_response


def extract_tool_calls_from_messages(all_messages: List[BaseMessage]) -> List[ToolExecutionRecord]:
    # Extract tool calls and results from all collected messages
    tool_call_map = {}  # Map tool_call_id to tool call info
    for message in all_messages:
        if isinstance(message, AIMessage) and hasattr(message, "tool_calls") and message.tool_calls:
            # Extract tool calls from AIMessage
            # tool_calls is a list of ToolCall (TypedDict) objects
            for tool_call in message.tool_calls:
                # ToolCall is a TypedDict with 'name', 'args', 'id' keys
                tool_call_id = (
                    tool_call.get("id") if isinstance(tool_call, dict) else getattr(tool_call, "id", None)
                )
                tool_name = (
                    tool_call.get("name") if isinstance(tool_call, dict) else getattr(tool_call, "name", None)
                )
                args = (
                    tool_call.get("args") if isinstance(tool_call, dict) else getattr(tool_call, "args", {})
                )

                if tool_call_id:
                    tool_call_map[tool_call_id] = {
                        "tool_name": tool_name or "unknown",
                        "arguments": args,
                        "caller_message_id": message.id,
                    }

    logger.info(f"tool_call_map: {tool_call_map}")
    logger.info(f"all_messages: {all_messages}")
    new_tool_executions = []
    # Extract tool execution results from ToolMessages
    for message in all_messages:
        if isinstance(message, ToolMessage):
            # Extract tool execution results from ToolMessage
            tool_call_id = message.tool_call_id
            tool_result = message.content
            tool_message_id = message.id
            # Check if there's an error status
            error_msg = None
            if hasattr(message, "status") and message.status == "error":
                error_msg = str(tool_result) if tool_result else "Tool execution failed"

            if tool_call_id and tool_call_id in tool_call_map:
                # Update existing tool call with result
                tool_call_info = tool_call_map[tool_call_id]

                # Create tool execution record
                execution_record = ToolExecutionRecord(
                    caller_message_id=tool_call_info["caller_message_id"],
                    tool_message_id=tool_message_id,
                    tool_name=tool_call_info["tool_name"],
                    tool_call_id=tool_call_id,
                    arguments=tool_call_info["arguments"],
                    result=tool_result if not error_msg else None,
                    error=error_msg,
                )
                new_tool_executions.append(execution_record)
            elif tool_call_id:
                # Tool call without matching AIMessage (shouldn't happen, but handle it)
                logger.warning(f"Tool call without matching AIMessage: {tool_call_id}")

    # Handle tool calls that didn't have corresponding ToolMessage (tool execution might have failed or is pending)
    for tool_call_id, tool_call_info in tool_call_map.items():
        # Check if we already recorded this tool call
        if not any(record.tool_call_id == tool_call_id for record in new_tool_executions):
            execution_record = ToolExecutionRecord(
                caller_message_id=tool_call_info["caller_message_id"],
                tool_name=tool_call_info["tool_name"],
                tool_call_id=tool_call_id,
                arguments=tool_call_info["arguments"],
                result=None,
            )
            new_tool_executions.append(execution_record)
    logger.info(f"new_tool_executions: {new_tool_executions}")
    return new_tool_executions



def parse_markdown_sections(markdown_text: str) -> dict[str, str]:
    """Parse markdown text and extract sections from context_summarizer.md output format.
    
    Extracts content from the following sections (headers not included):
    - Task-Relevant Key Findings
    - Additional Tool Requirement (optional)
    
    Args:
        markdown_text: The markdown text containing the sections from context_summarizer.md
        
    Returns:
        A dictionary with keys: 'task_relevant_key_findings' and optionally 
        'additional_tool_requirement'. Values are the extracted content without headers. 
        Empty strings if section not found.
    """
    if not markdown_text:
        return {
            "task_relevant_key_findings": "",
            "additional_tool_requirement": "",
        }
    
    result = {
        "task_relevant_key_findings": "",
        "additional_tool_requirement": "",
    }
    
    # Pattern to match ### Task-Relevant Key Findings (case-insensitive, flexible spacing)
    key_findings_pattern = r"###\s*Task-Relevant\s+Key\s+Findings\s*\n(.*?)(?=###\s*Additional\s+Tool\s+Requirement|$)"
    
    # Pattern to match ### Additional Tool Requirement (optional)
    additional_pattern = r"###\s*Additional\s+Tool\s+Requirement\s*\n(.*?)(?=###|$)"
    
    # Extract each section
    findings_match = re.search(key_findings_pattern, markdown_text, re.IGNORECASE | re.DOTALL)
    if findings_match:
        result["task_relevant_key_findings"] = findings_match.group(1).strip()
    
    additional_match = re.search(additional_pattern, markdown_text, re.IGNORECASE | re.DOTALL)
    if additional_match:
        result["additional_tool_requirement"] = additional_match.group(1).strip()
    
    return result
