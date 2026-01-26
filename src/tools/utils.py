# Copyright (c) 2026 Yunjue Tech
# SPDX-License-Identifier: Apache-2.0
import re
import json

def extract_tool_info(tool_filename: str) -> tuple[bool, dict, str]:
    """
    Extract tool information from the tool file.

    Returns:
        Tuple of (success_flag, tool_info_dict, error_message).
        tool_info_dict contains keys: tool_meta, tool_description, input_schema_code, output_schema_code.
    """
    import ast

    try:
        # Read the file
        with open(tool_filename, "r", encoding="utf-8") as f:
            file_content = f.read()

        # Parse the AST to extract information
        tree = ast.parse(file_content)

        tool_meta = {}
        tool_description = ""
        input_schema_code = ""
        output_schema_code = ""

        # Extract __TOOL_META__
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__TOOL_META__":
                        # Try to evaluate the __TOOL_META__ dictionary
                        try:
                            tool_meta = ast.literal_eval(node.value)
                            tool_description = tool_meta.get("description", "")
                        except Exception:
                            # If literal_eval fails, try to get it from the source
                            try:
                                meta_start = file_content.find("__TOOL_META__ = {")
                                if meta_start != -1:
                                    meta_end = file_content.find("}", meta_start)
                                    meta_str = file_content[meta_start : meta_end + 1]
                                    # Extract description from the string representation
                                    if '"description"' in meta_str:
                                        desc_match = re.search(r'"description":\s*"([^"]*)"', meta_str)
                                        if desc_match:
                                            tool_description = desc_match.group(1)
                            except Exception:
                                pass
                        break

        # Extract InputModel and OutputModel classes
        input_model_start = file_content.find("class InputModel")
        if input_model_start != -1:
            # Find the end of the class definition
            input_model_end = file_content.find("\nclass ", input_model_start + 1)
            if input_model_end == -1:
                input_model_end = file_content.find("\ndef ", input_model_start + 1)
            if input_model_end == -1:
                input_model_end = file_content.find("\nif __name__", input_model_start + 1)
            if input_model_end != -1:
                input_schema_code = file_content[input_model_start:input_model_end].strip()

        output_model_start = file_content.find("class OutputModel")
        if output_model_start != -1:
            # Find the end of the class definition
            output_model_end = file_content.find("\nclass ", output_model_start + 1)
            if output_model_end == -1:
                output_model_end = file_content.find("\ndef ", output_model_start + 1)
            if output_model_end == -1:
                output_model_end = file_content.find("\nif __name__", output_model_start + 1)
            if output_model_end != -1:
                output_schema_code = file_content[output_model_start:output_model_end].strip()

        # If we couldn't extract from AST, try regex as fallback
        if not tool_meta:
            meta_match = re.search(r"__TOOL_META__\s*=\s*({[^}]+})", file_content, re.DOTALL)
            if meta_match:
                try:
                    tool_meta = json.loads(meta_match.group(1).replace("'", '"'))
                    tool_description = tool_meta.get("description", "")
                except Exception:
                    pass

        return (
            True,
            {
                "tool_meta": tool_meta,
                "tool_description": tool_description,
                "input_schema_code": input_schema_code,
                "output_schema_code": output_schema_code,
            },
            "",
        )
    except Exception as e:
        error_message = f"Failed to extract tool info: {e}"
        return (
            False,
            {
                "tool_meta": {},
                "tool_description": "",
                "input_schema_code": "",
                "output_schema_code": "",
            },
            error_message,
        )
