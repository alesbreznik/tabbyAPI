import re
import json
from common.logger import xlogger
from endpoints.OAI.types.tools import ToolCall, Tool

"""
Hermes 2 / Hermes 3 family - ChatML + XML tool calling syntax

Raw format (single call):
    <tool_call>
    {"name": "function_name", "arguments": {"param1": "value1"}}
    </tool_call>

Raw format (parallel calls):
    <tool_call>
    {"name": "func_1", "arguments": {"param1": "value1"}}
    </tool_call>
    <tool_call>
    {"name": "func_2", "arguments": {"param2": "value2"}}
    </tool_call>
"""

TOOLCALL_START = "<tool_call>"
TOOLCALL_END = "</tool_call>"

_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def parse_toolcalls(text: str) -> list[ToolCall]:
    matches = _CALL_PATTERN.findall(text)
    if not matches:
        # Fallback: check if the entire text is a JSON object with a function name
        trimmed = text.strip()
        if trimmed.startswith("{") and trimmed.endswith("}"):
            matches = [trimmed]

    results = []
    for raw in matches:
        raw = raw.strip()
        if not raw:
            continue

        # Strip optional markdown codeblocks if model wraps JSON inside ```json ... ```
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw).strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            xlogger.warning(
                "hermes: Failed to parse tool call JSON",
                {"raw": raw, "error": str(e)},
            )
            continue

        if not isinstance(data, dict):
            continue

        func_name = data.get("name")
        arguments = data.get("arguments", {})
        if not func_name:
            continue

        if isinstance(arguments, dict):
            args_json = json.dumps(arguments, ensure_ascii=False)
        elif isinstance(arguments, str):
            args_json = arguments
        else:
            args_json = json.dumps(arguments, ensure_ascii=False)

        results.append(ToolCall(function=Tool(name=func_name, arguments=args_json)))

    xlogger.debug(
        f"hermes: Parsed {len(results)} tool calls",
        {"raw_text": text, "results": results},
    )
    return results
