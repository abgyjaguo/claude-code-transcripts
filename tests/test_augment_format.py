"""Tests for Augment ("Augument") export format support."""

import json
from pathlib import Path

import pytest

from claude_code_transcripts import generate_html, parse_session_file


class TestAugmentExportParsing:
    def test_parses_sample_augment_export(self):
        fixture_path = Path(__file__).parent / "sample_augment_export.json"
        data = parse_session_file(fixture_path)

        assert "loglines" in data
        assert [e["type"] for e in data["loglines"]] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ]

        first = data["loglines"][0]
        assert first["timestamp"] == "2026-01-01T12:00:01Z"
        assert first["message"]["role"] == "user"
        assert first["message"]["content"] == "Hello **Augment**"

        second = data["loglines"][1]
        assert second["timestamp"] == "2026-01-01T12:00:02Z"
        assert second["message"]["role"] == "assistant"
        # Ensure assistant content is markdown-renderable (Claude-style content blocks)
        assert isinstance(second["message"]["content"], list)
        assert second["message"]["content"][0]["type"] == "text"
        assert "print('hi')" in second["message"]["content"][0]["text"]

    @pytest.mark.parametrize(
        "payload",
        [
            # Minimal dict with top-level messages list
            {
                "messages": [
                    {"role": "USER", "text": "hi", "timestamp": 1735732800},
                    {"role": "ASSISTANT", "text": "hello", "timestamp": 1735732801},
                ]
            },
            # Conversation wrapper, alternate keys
            {
                "conversation": {
                    "messages": [
                        {
                            "sender": "user",
                            "content": "hi",
                            "createdAt": "2026-01-01T00:00:00Z",
                        },
                        {
                            "sender": "assistant",
                            "message": "hello",
                            "createdAt": "2026-01-01T00:00:01Z",
                        },
                    ]
                }
            },
        ],
    )
    def test_parses_common_augment_variants(self, tmp_path, payload):
        p = tmp_path / "augment.json"
        p.write_text(json.dumps(payload), encoding="utf-8")

        data = parse_session_file(p)
        assert "loglines" in data
        assert len(data["loglines"]) == 2
        assert data["loglines"][0]["type"] == "user"
        assert data["loglines"][1]["type"] == "assistant"


class TestAugmentHtmlGeneration:
    def test_generates_html_from_augment_export(self, tmp_path):
        fixture_path = Path(__file__).parent / "sample_augment_export.json"
        output_dir = tmp_path / "out"

        generate_html(fixture_path, output_dir)

        index_html = (output_dir / "index.html").read_text(encoding="utf-8")
        assert "Hello" in index_html
        # User markdown is rendered
        assert "<strong>Augment</strong>" in index_html

        # Assistant content (including code blocks) is rendered on the per-page transcript
        page_html = (output_dir / "page-001.html").read_text(encoding="utf-8")
        assert "print" in page_html
        assert "hi" in page_html
