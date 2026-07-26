"""Test double for the Anthropic client.

Lets us test the LLM orchestrator's control flow (tool call parsing, data
threading, loop termination) with scripted responses, without making real
network calls or needing an API key. This does not validate that the real
Claude API behaves this way — only that our orchestrator code correctly
handles responses shaped like the Anthropic SDK's.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def tool_use_block(name: str, input_: dict[str, Any], block_id: str = "tool_1"):
    return SimpleNamespace(type="tool_use", name=name, input=input_, id=block_id)


def text_block(text: str):
    return SimpleNamespace(type="text", text=text)


def fake_response(content: list[Any]):
    return SimpleNamespace(content=content)


class ScriptedClient:
    """Fake Anthropic client returning pre-scripted responses in order."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.call_count = 0

    @property
    def messages(self) -> "ScriptedClient":
        return self

    def create(self, **kwargs: Any):
        self.call_count += 1
        if not self._responses:
            raise RuntimeError("ScriptedClient ran out of scripted responses")
        return self._responses.pop(0)


class RaisingClient:
    """Fake Anthropic client that always raises, for testing fallback behavior."""

    @property
    def messages(self) -> "RaisingClient":
        return self

    def create(self, **kwargs: Any):
        raise RuntimeError("simulated API failure")
