"""Tests for Antigravity export format support."""

from pathlib import Path

from claude_code_transcripts import generate_html, parse_session_file


class TestAntigravityExportParsing:
    def test_parses_antigravity_export_json(self):
        fixture_path = Path(__file__).parent / "sample_antigravity_export.json"
        data = parse_session_file(fixture_path)

        assert "loglines" in data
        loglines = data["loglines"]
        assert len(loglines) == 4

        # User text
        assert loglines[0]["type"] == "user"
        assert loglines[0]["message"]["role"] == "user"
        assert loglines[0]["message"]["content"] == "Please run `echo hello`."

        # Tool use from model
        assert loglines[1]["type"] == "assistant"
        blocks = loglines[1]["message"]["content"]
        assert any(b.get("type") == "text" for b in blocks)
        tool_use = next(b for b in blocks if b.get("type") == "tool_use")
        assert tool_use["name"] == "Bash"
        assert tool_use["input"]["command"] == "echo hello"

        # Tool result
        assert loglines[2]["type"] == "user"
        tool_result = loglines[2]["message"]["content"][0]
        assert tool_result["type"] == "tool_result"
        assert tool_result["tool_use_id"] == tool_use["id"]
        assert tool_result["content"] == "hello\n"

        # Final assistant text
        assert loglines[3]["type"] == "assistant"
        assert loglines[3]["message"]["content"][0]["type"] == "text"
        assert loglines[3]["message"]["content"][0]["text"] == "Done."


class TestAntigravityHtmlGeneration:
    def test_generates_html_from_antigravity_export(self, tmp_path):
        fixture_path = Path(__file__).parent / "sample_antigravity_export.json"
        generate_html(fixture_path, tmp_path)

        index_html = (tmp_path / "index.html").read_text(encoding="utf-8").lower()
        assert "echo hello" in index_html
        page_html = (tmp_path / "page-001.html").read_text(encoding="utf-8").lower()
        assert "done" in page_html
