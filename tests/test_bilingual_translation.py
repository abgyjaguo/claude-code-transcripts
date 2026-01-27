"""Tests for bilingual (original + translated) HTML output."""

from pathlib import Path

from claude_code_transcripts import generate_html


class DummyTranslator:
    def __call__(self, text: str) -> str:
        return f"ZH:{text}"


def test_generate_html_bilingual_renders_translation_column(tmp_path: Path):
    jsonl_file = tmp_path / "session.jsonl"
    jsonl_file.write_text(
        (
            '{"type":"user","timestamp":"2025-01-01T00:00:00.000Z","message":{"role":"user","content":"Hello"}}\n'
            '{"type":"assistant","timestamp":"2025-01-01T00:00:01.000Z","message":{"role":"assistant","content":[{"type":"thinking","thinking":"Think"},{"type":"text","text":"Answer"}]}}\n'
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "out"
    output_dir.mkdir()

    generate_html(jsonl_file, output_dir, translator=DummyTranslator())

    page_html = (output_dir / "page-001.html").read_text(encoding="utf-8")
    assert 'class="message-content bilingual"' in page_html
    assert "中文翻译" in page_html
    assert "ZH:Hello" in page_html
    assert "ZH:Think" in page_html
    assert "ZH:Answer" in page_html
    assert "本页中文汇总" in page_html
    assert 'id="toggle-bash-tools"' in page_html
    assert 'id="toggle-tool-replies"' in page_html
