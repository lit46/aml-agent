"""Adapter letting any OpenAI-compatible chat completions API (Groq, most
OpenRouter models, Google's Gemini OpenAI-compatibility endpoint, etc.) act
as the orchestrator's LLM client.

LLMOrchestrator only ever calls `client.messages.create(model=..., 
max_tokens=..., system=..., tools=..., messages=...)` and reads back a
response with `.content` — a list of blocks each having `.type` plus
either `.text` or `.name`/`.input`/`.id`. This class implements that exact
surface by translating to and from the OpenAI-compatible wire format, so
swapping providers is just constructing a different client — no
orchestrator code changes required.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import requests

# Known free-tier OpenAI-compatible endpoints, for convenience in the UI.
# Users can always supply a custom base_url instead.
KNOWN_PROVIDERS: dict[str, dict[str, str]] = {
    "Groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
    },
    "Google Gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "default_model": "gemini-2.5-flash",
    },
    "OpenRouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "meta-llama/llama-3.3-70b-instruct:free",
    },
}


class OpenAICompatibleClient:
    """Wraps any OpenAI-compatible /chat/completions endpoint.

    Exposes `.messages.create(...)` matching the subset of the Anthropic
    Messages API that LLMOrchestrator uses.
    """

    def __init__(self, base_url: str, api_key: str, timeout: int = 60) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    @property
    def messages(self) -> "OpenAICompatibleClient":
        return self

    def create(
        self,
        model: str,
        max_tokens: int,
        system: str,
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        **_: Any,
    ) -> SimpleNamespace:
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": self._translate_messages(system, messages),
            "tools": self._translate_tools(tools),
        }

        response = requests.post(
            f"{self._base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self._timeout,
        )
        response.raise_for_status()
        return self._translate_response(response.json())

    @staticmethod
    def _block_get(block: Any, key: str) -> Any:
        """Blocks may be plain dicts (tool_result, constructed by us) or
        SimpleNamespace objects (tool_use/text, from a prior translated
        response) — support both transparently."""
        return block[key] if isinstance(block, dict) else getattr(block, key)

    @classmethod
    def _translate_messages(
        cls, system: str, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        translated: list[dict[str, Any]] = [{"role": "system", "content": system}]

        for message in messages:
            content = message["content"]

            if isinstance(content, str):
                translated.append({"role": message["role"], "content": content})
                continue

            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            tool_results: list[dict[str, Any]] = []

            for block in content:
                block_type = cls._block_get(block, "type")
                if block_type == "text":
                    text_parts.append(cls._block_get(block, "text"))
                elif block_type == "tool_use":
                    tool_calls.append(
                        {
                            "id": cls._block_get(block, "id"),
                            "type": "function",
                            "function": {
                                "name": cls._block_get(block, "name"),
                                "arguments": json.dumps(cls._block_get(block, "input")),
                            },
                        }
                    )
                elif block_type == "tool_result":
                    tool_results.append(block)

            if tool_results:
                for result in tool_results:
                    translated.append(
                        {
                            "role": "tool",
                            "tool_call_id": result["tool_use_id"],
                            "content": result["content"],
                        }
                    )
            elif tool_calls:
                translated.append(
                    {
                        "role": "assistant",
                        "content": "\n".join(text_parts) or None,
                        "tool_calls": tool_calls,
                    }
                )
            else:
                translated.append({"role": message["role"], "content": "\n".join(text_parts)})

        return translated

    @staticmethod
    def _translate_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                },
            }
            for tool in tools
        ]

    @staticmethod
    def _translate_response(data: dict[str, Any]) -> SimpleNamespace:
        message = data["choices"][0]["message"]
        blocks: list[SimpleNamespace] = []

        if message.get("content"):
            blocks.append(SimpleNamespace(type="text", text=message["content"]))

        for tool_call in message.get("tool_calls") or []:
            try:
                arguments = json.loads(tool_call["function"]["arguments"])
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            blocks.append(
                SimpleNamespace(
                    type="tool_use",
                    name=tool_call["function"]["name"],
                    input=arguments,
                    id=tool_call["id"],
                )
            )

        return SimpleNamespace(content=blocks)
