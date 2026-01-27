"""Tests for --translate-from-codex-config wiring."""

import json
from pathlib import Path


def test_translate_from_codex_config_uses_provider_base_url_and_auth(
    tmp_path: Path, monkeypatch, httpx_mock
):
    from click.testing import CliRunner
    from claude_code_transcripts import cli

    # Fake Codex home with config + auth
    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir()
    (codex_dir / "auth.json").write_text(
        json.dumps({"OPENAI_API_KEY": "test-key"}), encoding="utf-8"
    )
    (codex_dir / "config.toml").write_text(
        '\n'.join(
            [
                'model_provider = "testprov"',
                'model = "codex-model"',
                "",
                "[model_providers.testprov]",
                'base_url = "https://example.test/v1"',
                'wire_api = "responses"',
                "requires_openai_auth = true",
                "",
            ]
        ),
        encoding="utf-8",
    )

    # Mock Path.home() so defaults resolve to our tmp_path
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Ensure we still use ~/.codex/auth.json even if an env var is set.
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-env-key")

    # Minimal session with only a user prompt
    session = tmp_path / "session.jsonl"
    session.write_text(
        '{"type":"user","timestamp":"2025-01-01T00:00:00.000Z","message":{"role":"user","content":"Hello"}}\n',
        encoding="utf-8",
    )

    httpx_mock.add_response(
        url="https://example.test/v1/responses",
        json={"output_text": "你好"},
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
            "--translate-from-codex-config",
        ],
    )

    assert result.exit_code == 0

    requests = httpx_mock.get_requests()
    assert len(requests) == 1
    req = requests[0]
    assert str(req.url) == "https://example.test/v1/responses"
    assert req.headers.get("Authorization") == "Bearer test-key"
    payload = json.loads(req.content.decode("utf-8"))
    assert payload["model"] == "codex-model"

    page_html = (out_dir / "page-001.html").read_text(encoding="utf-8")
    assert "中文翻译" in page_html
    assert "你好" in page_html
