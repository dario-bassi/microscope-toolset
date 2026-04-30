import json
import anthropic
from typing import Optional
import logging
import sys

logger = logging.getLogger("BaseAgent")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    fh = logging.FileHandler("microscope_toolset.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)


class BaseAgent:

    def __init__(self, client: Optional[anthropic.Anthropic]):
        self.client = client

    def call_agent(self, model: str, input_user: list, error_string: str, output_format):
        """Call Claude with tool-use to get a structured response matching output_format."""
        if self.client is None:
            return {"intent": "error", "message": "Anthropic API key not configured — set ANTHROPIC_API_KEY in .env"}
        try:
            # Separate system prompt from the conversation messages
            system = next(
                (m["content"] for m in input_user if m["role"] == "system"), ""
            )
            messages = [m for m in input_user if m["role"] != "system"]

            # Build a tool whose input_schema matches the Pydantic model
            schema = output_format.model_json_schema()
            # Remove $defs / title noise that Pydantic adds — Claude only needs properties
            tool_schema = {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", []),
            }

            response = self.client.messages.create(
                model=model,
                max_tokens=1024,
                system=system,
                messages=messages,
                tools=[{
                    "name": "structured_output",
                    "description": "Return the structured result.",
                    "input_schema": tool_schema,
                }],
                tool_choice={"type": "tool", "name": "structured_output"},
            )

            for block in response.content:
                if block.type == "tool_use":
                    logger.info(block.input)
                    return block.input

            raise ValueError("No tool_use block in Claude response")

        except Exception as e:
            logger.error({"intent": "error", "message": f"{error_string}: {e}"})
            return {"intent": "error", "message": f"{error_string}: {e}"}
