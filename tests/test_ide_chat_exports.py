"""Tests for additional IDE chat transcript formats (Cursor/Windsurf/Augment)."""

import tempfile
from pathlib import Path

import pytest

from claude_code_transcripts import generate_html, parse_session_file


class TestVsCodeStyleChatExportJson:
    def test_parses_to_normalized_loglines(self):
        fixture = Path(__file__).parent / "sample_vscode_chat_export.json"
        data = parse_session_file(fixture)
        assert "loglines" in data
        assert len(data["loglines"]) >= 2
        assert data["loglines"][0]["type"] == "user"
        assert data["loglines"][1]["type"] == "assistant"

    def test_generates_html(self):
        fixture = Path(__file__).parent / "sample_vscode_chat_export.json"
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            generate_html(fixture, output_dir)

            index_html = (output_dir / "index.html").read_text(encoding="utf-8")
            page_html = (output_dir / "page-001.html").read_text(encoding="utf-8")
            assert "Hello from VS Code export" in index_html
            assert "Here is a response" in page_html
            assert "<strong>markdown</strong>" in page_html
            assert "print(&#x27;hi&#x27;)" in page_html or "print('hi')" in page_html


class TestMarkdownChatExport:
    def test_parses_to_normalized_loglines(self):
        fixture = Path(__file__).parent / "sample_chat_export.md"
        data = parse_session_file(fixture)
        assert "loglines" in data
        assert len(data["loglines"]) >= 2
        assert data["loglines"][0]["type"] == "user"
        assert data["loglines"][1]["type"] == "assistant"

    def test_generates_html(self):
        fixture = Path(__file__).parent / "sample_chat_export.md"
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            generate_html(fixture, output_dir)

            index_html = (output_dir / "index.html").read_text(encoding="utf-8")
            page_html = (output_dir / "page-001.html").read_text(encoding="utf-8")
            assert "Hello from Markdown export" in index_html
            assert "Assistant response with code:" in page_html
            assert "console.log" in page_html
            assert "<code" in page_html
