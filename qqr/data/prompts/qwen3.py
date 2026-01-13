import json
import logging
import re

logger = logging.getLogger(__name__)


class Qwen3Prompt:
    def __init__(self):
        self.eos_token = "<|im_end|>"
        self.bot_token = "<tool_call>"
        self.eot_token = "</tool_call>"
        self.think_start_token = "<think>"
        self.think_end_token = "</think>"

        self.think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        self.tool_pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

    def parse_assistant_content(self, assistant_content: str) -> dict:
        message = {
            "role": "assistant",
            "content": "",
            "reasoning_content": "",
            "tool_calls": [],
        }

        if self.think_start_token in assistant_content:
            think_match = self.think_pattern.search(assistant_content)
            if think_match:
                message["reasoning_content"] = think_match.group(1).strip()
                assistant_content = self.think_pattern.sub("", assistant_content)

        if self.bot_token in assistant_content:
            tool_matches = self.tool_pattern.findall(assistant_content)

            for func_idx, func_json_str in enumerate(tool_matches):
                func_json_str = func_json_str.strip()
                try:
                    tool_call = json.loads(func_json_str)

                    func_name = tool_call.get("name")
                    func_args = tool_call.get("arguments", {})

                    if isinstance(func_args, (dict, list)):
                        func_args_str = json.dumps(func_args, ensure_ascii=False)
                    else:
                        func_args_str = str(func_args)

                    message["tool_calls"].append(
                        {
                            "id": f"call_{func_idx + 1}",
                            "type": "function",
                            "function": {"name": func_name, "arguments": func_args_str},
                        }
                    )
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Failed to parse tool JSON: {func_json_str[:50]}... Error: {e}"
                    )
                    continue

            assistant_content = self.tool_pattern.sub("", assistant_content)

        message["content"] = assistant_content.removesuffix(self.eos_token).strip()

        return message
