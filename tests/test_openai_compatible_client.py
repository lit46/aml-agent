"""Tests for OpenAICompatibleClient — verified against real Groq/OpenAI
response shapes, with HTTP mocked so no network call is made."""

import json
from unittest.mock import MagicMock, patch

from app.agents.providers.openai_compatible_client import OpenAICompatibleClient


def _mock_response(payload: dict) -> MagicMock:
    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


def test_translates_tool_call_response_into_anthropic_shaped_blocks():
    client = OpenAICompatibleClient(base_url="https://api.groq.com/openai/v1", api_key="k")
    response_payload = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "aggregation_rule_tool",
                                "arguments": json.dumps({"reason": "test"}),
                            },
                        }
                    ],
                }
            }
        ]
    }

    with patch("requests.post", return_value=_mock_response(response_payload)):
        result = client.messages.create(
            model="llama-3.3-70b-versatile",
            max_tokens=1000,
            system="sys",
            tools=[],
            messages=[{"role": "user", "content": "hello"}],
        )

    assert len(result.content) == 1
    block = result.content[0]
    assert block.type == "tool_use"
    assert block.name == "aggregation_rule_tool"
    assert block.input == {"reason": "test"}
    assert block.id == "call_1"


def test_translates_final_text_response():
    client = OpenAICompatibleClient(base_url="https://api.groq.com/openai/v1", api_key="k")
    response_payload = {
        "choices": [{"message": {"content": "Here is the answer.", "tool_calls": None}}]
    }

    with patch("requests.post", return_value=_mock_response(response_payload)):
        result = client.messages.create(
            model="llama-3.3-70b-versatile",
            max_tokens=1000,
            system="sys",
            tools=[],
            messages=[{"role": "user", "content": "hello"}],
        )

    assert len(result.content) == 1
    assert result.content[0].type == "text"
    assert result.content[0].text == "Here is the answer."


def test_translates_anthropic_tools_schema_to_openai_functions_format():
    tools = [
        {
            "name": "aggregation_rule_tool",
            "description": "desc",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        }
    ]
    translated = OpenAICompatibleClient._translate_tools(tools)

    assert translated[0]["type"] == "function"
    assert translated[0]["function"]["name"] == "aggregation_rule_tool"
    assert translated[0]["function"]["parameters"] == tools[0]["input_schema"]


def test_translates_tool_result_message_to_openai_tool_role():
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": [
                type(
                    "Block", (), {"type": "tool_use", "name": "x", "input": {}, "id": "call_1"}
                )()
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": "result data"}
            ],
        },
    ]

    translated = OpenAICompatibleClient._translate_messages("sys", messages)

    tool_message = next(m for m in translated if m.get("role") == "tool")
    assert tool_message["tool_call_id"] == "call_1"
    assert tool_message["content"] == "result data"


def test_sends_bearer_auth_header():
    client = OpenAICompatibleClient(base_url="https://api.groq.com/openai/v1", api_key="my-secret-key")
    response_payload = {"choices": [{"message": {"content": "ok", "tool_calls": None}}]}

    with patch("requests.post", return_value=_mock_response(response_payload)) as mock_post:
        client.messages.create(
            model="m", max_tokens=100, system="s", tools=[], messages=[{"role": "user", "content": "hi"}]
        )

    headers = mock_post.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer my-secret-key"
