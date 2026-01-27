"""Tests for --translate-api-key-env."""

import json
from pathlib import Path


def test_translate_api_key_env_overrides_other_sources(
    tmp_path: Path, monkeypatch, httpx_mock
):
    from click.testing import CliRunner
    from claude_code_transcripts import cli

    # Provide multiple possible key sources and ensure the explicit env var wins.
    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir()
    (codex_dir / "auth.json").write_text(
        json.dumps({"OPENAI_API_KEY": "codex-key"}), encoding="utf-8"
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-openai-key")
    monkeypatch.setenv("ARK_API_KEY", "ark-key")

    session = tmp_path / "session.jsonl"
    session.write_text(
        '{"type":"user","timestamp":"2025-01-01T00:00:00.000Z","message":{"role":"user","content":"Hello"}}\n',
        encoding="utf-8",
    )

    httpx_mock.add_response(
        url="https://api.openai.com/v1/chat/completions",
        json={"choices": [{"message": {"content": "你好"}}]},
    )

    out_dir = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "json",
            str(session),
            "-o",
            str(out_dir),
            "--translate-zh",
            "--translate-api-key-env",
            "ARK_API_KEY",
            "--translate-wire-api",
            "chat_completions",
        ],
    )

    assert result.exit_code == 0
    requests = httpx_mock.get_requests()
    assert len(requests) == 1
    assert requests[0].headers.get("Authorization") == "Bearer ark-key"

    page_html = (out_dir / "page-001.html").read_text(encoding="utf-8")
    assert "中文翻译" in page_html
    assert "你好" in page_html

