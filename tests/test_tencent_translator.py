"""Tests for Tencent Cloud Machine Translation backend."""

import json
from pathlib import Path


def test_tencent_translator_translate_many(httpx_mock, monkeypatch):
    from claude_code_transcripts import TencentTranslator

    # 2025-01-01 00:00:00 UTC
    monkeypatch.setattr("claude_code_transcripts.time.time", lambda: 1735689600)

    httpx_mock.add_response(
        url="https://tmt.tencentcloudapi.com/",
        json={
            "Response": {
                "TargetTextList": ["你好", "世界"],
                "RequestId": "req-1",
            }
        },
    )

    translator = TencentTranslator(
        secret_id="sid",
        secret_key="skey",
        region="ap-beijing",
        endpoint="tmt.tencentcloudapi.com",
    )
    assert translator.translate_many(["Hello", "World"]) == ["你好", "世界"]

    requests = httpx_mock.get_requests()
    assert len(requests) == 1
    req = requests[0]
    assert str(req.url) == "https://tmt.tencentcloudapi.com/"
    assert req.headers.get("X-TC-Action") == "TextTranslateBatch"
    assert req.headers.get("X-TC-Region") == "ap-beijing"
    assert req.headers.get("X-TC-Version") == "2018-03-21"
    assert req.headers.get("X-TC-Timestamp") == "1735689600"
    auth = req.headers.get("Authorization") or ""
    assert auth.startswith(
        "TC3-HMAC-SHA256 Credential=sid/2025-01-01/tmt/tc3_request"
    )

    payload = json.loads(req.content.decode("utf-8"))
    assert payload["SourceTextList"] == ["Hello", "World"]
    assert payload["Source"] == "auto"
    assert payload["Target"] == "zh"
    assert payload["ProjectId"] == 0

    translator.close()


def test_cli_translate_provider_tencent(tmp_path: Path, monkeypatch, httpx_mock):
    from click.testing import CliRunner
    from claude_code_transcripts import cli

    monkeypatch.setenv("TENCENTCLOUD_SECRET_ID", "sid")
    monkeypatch.setenv("TENCENTCLOUD_SECRET_KEY", "skey")
    monkeypatch.setattr("claude_code_transcripts.time.time", lambda: 1735689600)

    session = tmp_path / "session.jsonl"
    session.write_text(
        (
            '{"type":"user","timestamp":"2025-01-01T00:00:00.000Z","message":{"role":"user","content":"Hello"}}\n'
            '{"type":"assistant","timestamp":"2025-01-01T00:00:01.000Z","message":{"role":"assistant","content":[{"type":"text","text":"Hi"}]}}\n'
        ),
        encoding="utf-8",
    )

    httpx_mock.add_response(
        url="https://tmt.tencentcloudapi.com/",
        json={
            "Response": {
                "TargetTextList": ["你好", "嗨"],
                "RequestId": "req-1",
            }
        },
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
            "--translate-provider",
            "tencent",
        ],
    )
    assert result.exit_code == 0

    requests = httpx_mock.get_requests()
    assert len(requests) == 1
    assert "Credential=sid/" in (requests[0].headers.get("Authorization") or "")

    page_html = (out_dir / "page-001.html").read_text(encoding="utf-8")
    assert "中文翻译" in page_html
    assert "你好" in page_html
    assert "嗨" in page_html

