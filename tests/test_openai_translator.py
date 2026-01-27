"""Tests for OpenAI translation via chat.completions or Responses API."""

import httpx
import pytest

from claude_code_transcripts import OpenAITranslator


def test_openai_translator_responses_api(httpx_mock):
    httpx_mock.add_response(
        url="https://api.openai.com/v1/responses",
        json={
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "你好"}],
                }
            ]
        },
    )

    translator = OpenAITranslator(
        api_key="test",
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        wire_api="responses",
    )
    assert translator.translate("Hello") == "你好"

    # Should hit cache on second call (no extra HTTP stubs required)
    assert translator.translate("Hello") == "你好"
    translator.close()


def test_openai_translator_auto_falls_back_to_chat_completions(httpx_mock):
    httpx_mock.add_response(
        url="https://api.openai.com/v1/responses",
        status_code=404,
        json={"error": {"message": "Not found"}},
    )
    httpx_mock.add_response(
        url="https://api.openai.com/v1/chat/completions",
        json={"choices": [{"message": {"content": "你好"}}]},
    )

    translator = OpenAITranslator(
        api_key="test",
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        wire_api="auto",
    )
    assert translator.translate("Hello") == "你好"
    translator.close()


def test_openai_translator_retries_on_transport_error(httpx_mock):
    httpx_mock.add_exception(
        httpx.TransportError(
            "[SSL: DECRYPTION_FAILED_OR_BAD_RECORD_MAC] decryption failed or bad record mac (_ssl.c:2648)"
        ),
        url="https://api.openai.com/v1/responses",
        method="POST",
    )
    httpx_mock.add_response(
        url="https://api.openai.com/v1/responses",
        method="POST",
        json={
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "你好"}],
                }
            ]
        },
    )

    translator = OpenAITranslator(
        api_key="test",
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        wire_api="responses",
        max_retries=1,
        retry_backoff=0,
    )
    assert translator.translate("Hello") == "你好"
    translator.close()


def test_openai_translator_does_not_force_v1_when_version_present():
    translator = OpenAITranslator(
        api_key="test",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        model="ep-test",
        wire_api="chat_completions",
    )
    assert translator.base_url == "https://ark.cn-beijing.volces.com/api/v3"
    translator.close()
