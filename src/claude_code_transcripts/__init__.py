"""Convert Claude Code session JSON to a clean mobile-friendly HTML page with pagination."""

import json
import html
import hashlib
import hmac
import os
import platform
import re
import shutil
import ssl
import subprocess
import tempfile
import time
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import click
from click_default_group import DefaultGroup
import httpx
from jinja2 import Environment, PackageLoader
import markdown
import questionary

# Set up Jinja2 environment
_jinja_env = Environment(
    loader=PackageLoader("claude_code_transcripts", "templates"),
    autoescape=True,
)

# Load macros template and expose macros
_macros_template = _jinja_env.get_template("macros.html")
_macros = _macros_template.module


def get_template(name):
    """Get a Jinja2 template by name."""
    return _jinja_env.get_template(name)


# Regex to match git commit output: [branch hash] message
COMMIT_PATTERN = re.compile(r"\[[\w\-/]+ ([a-f0-9]{7,})\] (.+?)(?:\n|$)")

# Regex to detect GitHub repo from git push output (e.g., github.com/owner/repo/pull/new/branch)
GITHUB_REPO_PATTERN = re.compile(
    r"github\.com/([a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+)/pull/new/"
)

PROMPTS_PER_PAGE = 5
LONG_TEXT_THRESHOLD = (
    300  # Characters - text blocks longer than this are shown in index
)


def extract_text_from_content(content):
    """Extract plain text from message content.

    Handles both string content (older format) and array content (newer format).

    Args:
        content: Either a string or a list of content blocks like
                 [{"type": "text", "text": "..."}, {"type": "image", ...}]

    Returns:
        The extracted text as a string, or empty string if no text found.
    """
    if isinstance(content, str):
        return content.strip()
    elif isinstance(content, list):
        # Extract text from content blocks of type "text"
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if text:
                    texts.append(text)
        return " ".join(texts).strip()
    return ""


def _is_preamble_text(text):
    if not isinstance(text, str):
        return False
    stripped = text.lstrip()
    if not stripped:
        return False
    if stripped.startswith("<"):
        return True
    lowered = stripped.lower()
    return lowered.startswith("# agents.md instructions") or lowered.startswith(
        "agents.md instructions"
    )


# Module-level variable for GitHub repo (set by generate_html)
_github_repo = None

# API constants
API_BASE_URL = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"

def _read_codex_provider_config(
    config_path: Path | None = None,
    provider: str | None = None,
) -> dict[str, object]:
    """Read ~/.codex/config.toml and return provider settings.

    This is best-effort and only used for optional convenience features.
    """
    config_path = config_path or (Path.home() / ".codex" / "config.toml")
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.exists():
        raise click.ClickException(f"Codex config not found: {config_path}")

    try:
        import tomllib  # py311+
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(
            f"Reading Codex config requires Python 3.11+ (tomllib not available: {exc})."
        ) from exc

    raw = config_path.read_bytes()
    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except Exception:
        data = tomllib.loads(raw.decode(errors="replace"))

    selected = (provider or data.get("model_provider") or "").strip()
    providers = data.get("model_providers") or {}
    provider_cfg = providers.get(selected) if isinstance(providers, dict) else None
    if not selected or not isinstance(provider_cfg, dict):
        known = ", ".join(sorted(providers.keys())) if isinstance(providers, dict) else ""
        raise click.ClickException(
            f"Provider not found in Codex config. provider={selected!r}. known=[{known}]"
        )

    return {
        "provider": selected,
        "model": data.get("model"),
        "base_url": provider_cfg.get("base_url"),
        "wire_api": provider_cfg.get("wire_api"),
        "requires_openai_auth": provider_cfg.get("requires_openai_auth"),
    }


def _load_codex_auth_api_key(auth_path: Path | None = None) -> str | None:
    auth_path = auth_path or (Path.home() / ".codex" / "auth.json")
    auth_path = Path(auth_path).expanduser().resolve()
    if not auth_path.exists():
        return None
    try:
        payload = json.loads(auth_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    key = payload.get("OPENAI_API_KEY")
    return str(key) if isinstance(key, str) and key.strip() else None


@dataclass
class OpenAITranslator:
    api_key: str
    model: str = "gpt-4o-mini"
    target_language: str = "Simplified Chinese"
    wire_api: str = "auto"
    cache_path: Path | None = None
    base_url: str = "https://api.openai.com/v1"
    timeout: float = 60.0
    max_retries: int = 2
    retry_backoff: float = 0.5
    disable_keepalive_on_error: bool = True

    def __post_init__(self) -> None:
        base_url = self.base_url.rstrip("/")
        parts = urlsplit(base_url)
        path = parts.path.rstrip("/")
        if not re.search(r"(?i)(^|/)v\d+($|/)", path):
            path = f"{path}/v1" if path else "/v1"
        self.base_url = urlunsplit(
            (parts.scheme, parts.netloc, path, parts.query, parts.fragment)
        ).rstrip("/")

        self._no_keepalive = False
        self._client = self._build_client()
        self._cache = self._load_cache()
        self._dirty = False
        self._stats = {
            "http_calls": 0,
            "http_seconds": 0.0,
            "http_retries": 0,
            "translate_many_batches": 0,
        }

    def _build_client(self) -> httpx.Client:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        limits = httpx.Limits()
        if self._no_keepalive:
            headers["Connection"] = "close"
            limits = httpx.Limits(
                max_keepalive_connections=0,
                keepalive_expiry=0.0,
            )

        return httpx.Client(
            timeout=self.timeout,
            headers=headers,
            limits=limits,
        )

    def _reset_client(self, *, no_keepalive: bool | None = None) -> None:
        if no_keepalive is not None:
            self._no_keepalive = bool(no_keepalive)
        try:
            self._client.close()
        except Exception:
            pass
        self._client = self._build_client()

    @classmethod
    def from_env(
        cls,
        *,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        target_language: str = "Simplified Chinese",
        wire_api: str = "auto",
        cache_path: Path | None = None,
        base_url: str | None = None,
        prefer_codex_auth: bool = False,
        codex_auth_path: Path | None = None,
    ) -> "OpenAITranslator":
        if api_key is not None and str(api_key).strip():
            resolved_api_key = str(api_key).strip()
        elif prefer_codex_auth:
            resolved_api_key = _load_codex_auth_api_key(
                codex_auth_path
            ) or os.environ.get("OPENAI_API_KEY")
        else:
            resolved_api_key = os.environ.get("OPENAI_API_KEY") or _load_codex_auth_api_key(
                codex_auth_path
            )

        if not resolved_api_key:
            raise click.ClickException(
                "Missing OPENAI_API_KEY for translation.\n"
                "- Option 1 (recommended): set env var:\n"
                "    $env:OPENAI_API_KEY = '...'\n"
                "- Option 2: store it in ~/.codex/auth.json (OPENAI_API_KEY)"
            )
        base_url = base_url or (
            os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
            or "https://api.openai.com/v1"
        )
        return cls(
            api_key=resolved_api_key,
            model=model,
            target_language=target_language,
            wire_api=wire_api,
            cache_path=cache_path,
            base_url=base_url,
        )

    def _cache_key(self, text: str) -> str:
        digest = hashlib.sha256(
            f"{self.model}\n{self.target_language}\n{text}".encode("utf-8")
        ).hexdigest()
        return digest

    def _load_cache(self) -> dict[str, str]:
        if not self.cache_path:
            return {}
        try:
            if not self.cache_path.exists():
                return {}
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and "translations" in payload:
                translations = payload.get("translations")
                if isinstance(translations, dict):
                    return {str(k): str(v) for k, v in translations.items()}
            if isinstance(payload, dict):
                return {str(k): str(v) for k, v in payload.items()}
        except Exception:
            return {}
        return {}

    def save_cache(self) -> None:
        if not self.cache_path or not self._dirty:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "translations": self._cache}
        self.cache_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._dirty = False

    def close(self) -> None:
        self.save_cache()
        self._client.close()

    def _post_json_with_retries(self, url: str, payload: dict) -> httpx.Response:
        max_retries = max(0, int(self.max_retries or 0))
        backoff = max(0.0, float(self.retry_backoff or 0.0))

        last_exc: BaseException | None = None
        for attempt in range(max_retries + 1):
            attempt_start = time.perf_counter()
            try:
                response = self._client.post(url, json=payload)
                self._stats["http_calls"] += 1
                self._stats["http_seconds"] += time.perf_counter() - attempt_start
                return response
            except (httpx.TransportError, ssl.SSLError) as exc:
                self._stats["http_calls"] += 1
                self._stats["http_seconds"] += time.perf_counter() - attempt_start
                last_exc = exc
                if attempt >= max_retries:
                    raise

                self._stats["http_retries"] += 1
                switch_to_no_keepalive = (
                    self.disable_keepalive_on_error and not self._no_keepalive
                )
                self._reset_client(no_keepalive=switch_to_no_keepalive)

                delay = backoff * (2**attempt)
                if delay > 0:
                    time.sleep(delay)

        assert last_exc is not None
        raise last_exc

    def __call__(self, text: str) -> str:
        return self.translate(text)

    def _extract_responses_text(self, data: dict) -> str:
        output_text = data.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        texts = []
        for item in data.get("output") or []:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "message":
                continue
            if item.get("role") != "assistant":
                continue
            for block in item.get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "output_text" and block.get("text"):
                    texts.append(str(block["text"]))

        return "\n".join(t.strip() for t in texts if t and t.strip()).strip()

    def _translate_with_responses(self, text: str, prompt: str) -> str:
        response = self._post_json_with_retries(
            f"{self.base_url}/responses",
            {
                "model": self.model,
                "instructions": prompt,
                "input": [
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": text}],
                    }
                ],
                "temperature": 0,
            },
        )
        response.raise_for_status()
        return self._extract_responses_text(response.json())

    def _translate_with_chat_completions(self, text: str, prompt: str) -> str:
        response = self._post_json_with_retries(
            f"{self.base_url}/chat/completions",
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text},
                ],
                "temperature": 0,
            },
        )
        response.raise_for_status()
        data = response.json()
        translated = (
            (data.get("choices") or [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return (translated or "").strip()

    def _extract_json_list(self, text: str) -> list[str]:
        """Best-effort parse a JSON array of strings from a model response."""
        text = (text or "").strip()
        if not text:
            raise click.ClickException("Empty translation response.")

        try:
            data = json.loads(text)
        except Exception:
            start = text.find("[")
            end = text.rfind("]")
            if start == -1 or end == -1 or end <= start:
                raise click.ClickException(
                    "Translation response was not valid JSON."
                )
            data = json.loads(text[start : end + 1])

        if not isinstance(data, list):
            raise click.ClickException("Translation response JSON was not an array.")
        return [str(item) for item in data]

    def translate_many(self, texts: list[str]) -> list[str]:
        if not isinstance(texts, list):
            raise TypeError("translate_many(texts) expects a list[str]")

        if not texts:
            return []

        results: list[str] = [""] * len(texts)
        pending: list[tuple[int, str]] = []

        for i, text in enumerate(texts):
            normalized = (text or "").strip()
            if not normalized:
                results[i] = ""
                continue
            key = self._cache_key(normalized)
            cached = self._cache.get(key)
            if cached is not None:
                results[i] = cached
            else:
                pending.append((i, normalized))

        if not pending:
            return results

        prompt = (
            f"Translate each string in the provided JSON array to {self.target_language}. "
            "Preserve Markdown formatting (including code blocks, links, and inline code). "
            "Return ONLY a valid JSON array of strings with the same length and order as the input. "
            "Do not wrap the JSON in Markdown code fences."
        )

        wire_api = (self.wire_api or "auto").strip().lower()
        if wire_api not in ("auto", "responses", "chat_completions"):
            raise click.ClickException(
                "Invalid wire_api for translation; expected: auto|responses|chat_completions"
            )

        max_chars = 12_000
        batch: list[tuple[int, str]] = []
        batch_chars = 0

        def flush_batch() -> None:
            nonlocal batch, batch_chars
            if not batch:
                return
            self._stats["translate_many_batches"] += 1
            batch_indices = [i for i, _ in batch]
            batch_texts = [t for _, t in batch]
            input_text = json.dumps(batch_texts, ensure_ascii=False)

            if wire_api == "responses":
                raw = self._translate_with_responses(input_text, prompt)
            elif wire_api == "chat_completions":
                raw = self._translate_with_chat_completions(input_text, prompt)
            else:
                try:
                    raw = self._translate_with_responses(input_text, prompt)
                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code
                    if status in (404, 405, 410, 501, 503):
                        raw = self._translate_with_chat_completions(input_text, prompt)
                    else:
                        raise

            translated_list = self._extract_json_list(raw)
            if len(translated_list) != len(batch_texts):
                raise click.ClickException(
                    "Translation response had the wrong length: "
                    f"expected {len(batch_texts)} items, got {len(translated_list)}."
                )

            for idx, source_text, translated in zip(
                batch_indices, batch_texts, translated_list, strict=True
            ):
                translated = (translated or "").strip()
                results[idx] = translated
                self._cache[self._cache_key(source_text)] = translated
                self._dirty = True

            batch = []
            batch_chars = 0

        for index, text in pending:
            if batch and batch_chars + len(text) > max_chars:
                flush_batch()
            batch.append((index, text))
            batch_chars += len(text)

        flush_batch()
        return results

    def get_stats(self) -> dict:
        return dict(self._stats)

    def translate(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""

        key = self._cache_key(text)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        prompt = (
            f"Translate the following content to {self.target_language}. "
            "Preserve Markdown formatting (including code blocks, links, and inline code). "
            "Output ONLY the translated text."
        )
        wire_api = (self.wire_api or "auto").strip().lower()
        if wire_api not in ("auto", "responses", "chat_completions"):
            raise click.ClickException(
                "Invalid wire_api for translation; expected: auto|responses|chat_completions"
            )

        if wire_api == "responses":
            translated = self._translate_with_responses(text, prompt)
        elif wire_api == "chat_completions":
            translated = self._translate_with_chat_completions(text, prompt)
        else:
            try:
                translated = self._translate_with_responses(text, prompt)
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status in (404, 405, 410, 501, 503):
                    translated = self._translate_with_chat_completions(text, prompt)
                else:
                    raise
        self._cache[key] = translated
        self._dirty = True
        return translated


@dataclass
class TencentTranslator:
    """Tencent Cloud Machine Translation (TMT) backend using TC3-HMAC-SHA256 signing."""

    secret_id: str
    secret_key: str
    region: str = "ap-beijing"
    endpoint: str = "tmt.tencentcloudapi.com"
    project_id: int = 0
    source_language: str = "auto"
    target_language: str = "zh"
    cache_path: Path | None = None
    timeout: float = 20.0
    max_retries: int = 2
    retry_backoff: float = 0.5
    disable_keepalive_on_error: bool = True

    def __post_init__(self) -> None:
        self.service = "tmt"
        self.version = "2018-03-21"
        self.endpoint = (self.endpoint or "").strip()
        if not self.endpoint:
            raise click.ClickException("Tencent TMT endpoint is required.")
        self.base_url = f"https://{self.endpoint}".rstrip("/")

        self._no_keepalive = False
        self._client = self._build_client()
        self._cache = self._load_cache()
        self._dirty = False
        self._stats = {
            "http_calls": 0,
            "http_seconds": 0.0,
            "http_retries": 0,
            "translate_many_batches": 0,
        }

    def _build_client(self) -> httpx.Client:
        headers = {"Content-Type": "application/json; charset=utf-8"}
        limits = httpx.Limits()
        if self._no_keepalive:
            headers["Connection"] = "close"
            limits = httpx.Limits(
                max_keepalive_connections=0,
                keepalive_expiry=0.0,
            )
        return httpx.Client(timeout=self.timeout, headers=headers, limits=limits)

    def _reset_client(self, *, no_keepalive: bool | None = None) -> None:
        if no_keepalive is not None:
            self._no_keepalive = bool(no_keepalive)
        try:
            self._client.close()
        except Exception:
            pass
        self._client = self._build_client()

    def _cache_key(self, text: str) -> str:
        digest = hashlib.sha256(
            (
                f"tencent_tmt\n{self.endpoint}\n{self.region}\n"
                f"{self.source_language}\n{self.target_language}\n{text}"
            ).encode("utf-8")
        ).hexdigest()
        return digest

    def _load_cache(self) -> dict[str, str]:
        if not self.cache_path:
            return {}
        try:
            if not self.cache_path.exists():
                return {}
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and "translations" in payload:
                translations = payload.get("translations")
                if isinstance(translations, dict):
                    return {str(k): str(v) for k, v in translations.items()}
            if isinstance(payload, dict):
                return {str(k): str(v) for k, v in payload.items()}
        except Exception:
            return {}
        return {}

    def save_cache(self) -> None:
        if not self.cache_path or not self._dirty:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "translations": self._cache}
        self.cache_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._dirty = False

    def close(self) -> None:
        self.save_cache()
        self._client.close()

    def __call__(self, text: str) -> str:
        return self.translate(text)

    def _sha256_hex(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def _hmac_sha256(self, key: bytes, msg: str) -> bytes:
        return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

    def _sign_headers(
        self, *, action: str, payload: bytes, timestamp: int
    ) -> dict[str, str]:
        date = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
        host = self.endpoint

        canonical_headers = (
            "content-type:application/json; charset=utf-8\n"
            f"host:{host}\n"
            f"x-tc-action:{action.lower()}\n"
        )
        signed_headers = "content-type;host;x-tc-action"
        hashed_request_payload = self._sha256_hex(payload)

        canonical_request = (
            "POST\n"
            "/\n"
            "\n"
            f"{canonical_headers}\n"
            f"{signed_headers}\n"
            f"{hashed_request_payload}"
        )

        algorithm = "TC3-HMAC-SHA256"
        credential_scope = f"{date}/{self.service}/tc3_request"
        string_to_sign = (
            f"{algorithm}\n"
            f"{timestamp}\n"
            f"{credential_scope}\n"
            f"{self._sha256_hex(canonical_request.encode('utf-8'))}"
        )

        secret_date = self._hmac_sha256(("TC3" + self.secret_key).encode("utf-8"), date)
        secret_service = self._hmac_sha256(secret_date, self.service)
        secret_signing = self._hmac_sha256(secret_service, "tc3_request")
        signature = hmac.new(
            secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        authorization = (
            f"{algorithm} "
            f"Credential={self.secret_id}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, "
            f"Signature={signature}"
        )

        return {
            "Authorization": authorization,
            "Host": host,
            "X-TC-Action": action,
            "X-TC-Timestamp": str(timestamp),
            "X-TC-Version": self.version,
            "X-TC-Region": self.region,
        }

    def _post_action(self, action: str, payload_obj: dict) -> dict:
        max_retries = max(0, int(self.max_retries or 0))
        backoff = max(0.0, float(self.retry_backoff or 0.0))

        url = f"{self.base_url}/"
        last_exc: BaseException | None = None
        for attempt in range(max_retries + 1):
            timestamp = int(time.time())
            payload = json.dumps(
                payload_obj, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            headers = self._sign_headers(
                action=action, payload=payload, timestamp=timestamp
            )

            attempt_start = time.perf_counter()
            try:
                response = self._client.post(url, content=payload, headers=headers)
                self._stats["http_calls"] += 1
                self._stats["http_seconds"] += time.perf_counter() - attempt_start

                if response.status_code >= 500 or response.status_code == 429:
                    response.raise_for_status()
                return response.json()
            except (httpx.TransportError, ssl.SSLError, httpx.HTTPStatusError) as exc:
                self._stats["http_calls"] += 1
                self._stats["http_seconds"] += time.perf_counter() - attempt_start
                last_exc = exc
                if attempt >= max_retries:
                    raise

                self._stats["http_retries"] += 1
                switch_to_no_keepalive = (
                    self.disable_keepalive_on_error and not self._no_keepalive
                )
                self._reset_client(no_keepalive=switch_to_no_keepalive)

                delay = backoff * (2**attempt)
                if delay > 0:
                    time.sleep(delay)

        assert last_exc is not None
        raise last_exc

    def _extract_or_raise(self, data: dict) -> dict:
        if not isinstance(data, dict):
            raise click.ClickException("Unexpected Tencent TMT response.")
        response = data.get("Response")
        if not isinstance(response, dict):
            raise click.ClickException(
                "Unexpected Tencent TMT response format (missing Response)."
            )
        err = response.get("Error")
        if isinstance(err, dict) and err.get("Code"):
            code = err.get("Code")
            msg = err.get("Message") or ""
            raise click.ClickException(f"Tencent TMT error {code}: {msg}")
        return response

    def translate_many(self, texts: list[str]) -> list[str]:
        if not isinstance(texts, list):
            raise TypeError("translate_many(texts) expects a list[str]")
        if not texts:
            return []

        results: list[str] = [""] * len(texts)
        pending: list[tuple[int, str]] = []

        for i, text in enumerate(texts):
            normalized = (text or "").strip()
            if not normalized:
                results[i] = ""
                continue
            key = self._cache_key(normalized)
            cached = self._cache.get(key)
            if cached is not None:
                results[i] = cached
            else:
                pending.append((i, normalized))

        if not pending:
            return results

        # Tencent batch requires the total length of input text list to be < 6000 characters.
        max_chars_total = 5800

        batch: list[tuple[int, str]] = []
        batch_chars = 0

        def flush_batch() -> None:
            nonlocal batch, batch_chars
            if not batch:
                return
            self._stats["translate_many_batches"] += 1
            batch_indices = [i for i, _ in batch]
            batch_texts = [t for _, t in batch]
            payload = {
                "SourceTextList": batch_texts,
                "Source": self.source_language,
                "Target": self.target_language,
                "ProjectId": int(self.project_id),
            }
            data = self._post_action("TextTranslateBatch", payload)
            response = self._extract_or_raise(data)
            translated_list = response.get("TargetTextList")
            if not isinstance(translated_list, list) or len(translated_list) != len(
                batch_texts
            ):
                raise click.ClickException(
                    "Tencent TMT returned an unexpected TargetTextList."
                )

            for idx, source_text, translated in zip(
                batch_indices, batch_texts, translated_list, strict=True
            ):
                translated = str(translated or "").strip()
                results[idx] = translated
                self._cache[self._cache_key(source_text)] = translated
                self._dirty = True

            batch = []
            batch_chars = 0

        for index, text in pending:
            if batch and batch_chars + len(text) > max_chars_total:
                flush_batch()
            batch.append((index, text))
            batch_chars += len(text)

        flush_batch()
        return results

    def translate(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        key = self._cache_key(text)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        translated = self.translate_many([text])[0]
        self._cache[key] = translated
        self._dirty = True
        return translated

    def get_stats(self) -> dict:
        return dict(self._stats)


def get_session_summary(filepath, max_length=200):
    """Extract a human-readable summary from a session file.

    Supports both JSON and JSONL formats.
    Returns a summary string or "(no summary)" if none found.
    """
    filepath = Path(filepath)
    try:
        if filepath.suffix == ".jsonl":
            return _get_jsonl_summary(filepath, max_length)
        else:
            # For JSON files, try to get first user message
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            loglines = data.get("loglines", [])
            for entry in loglines:
                if entry.get("type") == "user":
                    msg = entry.get("message", {})
                    content = msg.get("content", "")
                    text = extract_text_from_content(content)
                    if text and not _is_preamble_text(text):
                        if len(text) > max_length:
                            return text[: max_length - 3] + "..."
                        return text
            return "(no summary)"
    except Exception:
        return "(no summary)"


def _get_jsonl_summary(filepath, max_length=200):
    """Extract summary from JSONL file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    # First priority: summary type entries
                    if obj.get("type") == "summary" and obj.get("summary"):
                        summary = obj["summary"]
                        if len(summary) > max_length:
                            return summary[: max_length - 3] + "..."
                        return summary
                except json.JSONDecodeError:
                    continue

        # Second pass: find first non-meta user message
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)

                    # Claude Code format: {"type": "user", "message": {...}}
                    if (
                        obj.get("type") == "user"
                        and not obj.get("isMeta")
                        and obj.get("message", {}).get("content")
                    ):
                        content = obj["message"]["content"]
                        text = extract_text_from_content(content)
                        if text and not _is_preamble_text(text):
                            if len(text) > max_length:
                                return text[: max_length - 3] + "..."
                            return text

                    # Codex CLI format: {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [...]}}
                    elif obj.get("type") == "response_item":
                        payload = obj.get("payload", {})
                        if (
                            payload.get("type") == "message"
                            and payload.get("role") == "user"
                            and payload.get("content")
                        ):
                            content_blocks = payload["content"]
                            # Extract text from Codex CLI content blocks
                            if isinstance(content_blocks, list):
                                for block in content_blocks:
                                    if block.get("type") == "input_text":
                                        text = block.get("text", "")
                                        if text and not _is_preamble_text(text):
                                            if len(text) > max_length:
                                                return text[: max_length - 3] + "..."
                                            return text
                    # Codex CLI old format: {"type": "message", "role": "user", "content": [...]}
                    elif (
                        obj.get("type") == "message"
                        and obj.get("role") == "user"
                        and obj.get("content")
                    ):
                        content_blocks = obj.get("content", [])
                        if isinstance(content_blocks, list):
                            for block in content_blocks:
                                if block.get("type") in ("input_text", "text"):
                                    text = block.get("text", "")
                                    if text and not _is_preamble_text(text):
                                        if len(text) > max_length:
                                            return text[: max_length - 3] + "..."
                                        return text
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass

    return "(no summary)"


def find_local_sessions(folder, limit=10):
    """Find recent JSONL session files in the given folder.

    Returns a list of (Path, summary) tuples sorted by modification time.
    Excludes agent files and warmup/empty sessions.
    """
    folder = Path(folder)
    if not folder.exists():
        return []

    candidates = []
    for f in folder.glob("**/*.jsonl"):
        if f.name.startswith("agent-"):
            continue
        try:
            mtime = f.stat().st_mtime
        except OSError:
            continue
        candidates.append((mtime, f))

    candidates.sort(key=lambda x: x[0], reverse=True)

    results = []
    for _mtime, f in candidates:
        summary = get_session_summary(f)
        # Skip boring/empty sessions
        if summary.lower() == "warmup" or summary == "(no summary)":
            continue
        results.append((f, summary))
        if len(results) >= limit:
            break
    return results


def find_combined_sessions(claude_dir=None, codex_dir=None, limit=10):
    """Find recent sessions from both Claude Code and Codex CLI directories.

    Args:
        claude_dir: Path to Claude Code projects folder (default: ~/.claude/projects)
        codex_dir: Path to Codex CLI sessions folder (default: ~/.codex/sessions)
        limit: Maximum number of sessions to return (default: 10)

    Returns:
        List of (Path, summary, source) tuples sorted by modification time (newest first).
        source is either "Claude" or "Codex".
    """
    if claude_dir is None:
        claude_dir = Path.home() / ".claude" / "projects"
    if codex_dir is None:
        codex_dir = Path.home() / ".codex" / "sessions"

    claude_dir = Path(claude_dir)
    codex_dir = Path(codex_dir)

    candidates = []

    # Find Claude Code sessions (metadata only first)
    if claude_dir.exists():
        for f in claude_dir.glob("**/*.jsonl"):
            if f.name.startswith("agent-"):
                continue
            try:
                mtime = f.stat().st_mtime
            except OSError:
                continue
            candidates.append((mtime, f, "Claude"))

    # Find Codex CLI sessions (metadata only first)
    if codex_dir.exists():
        for f in codex_dir.glob("**/*.jsonl"):
            if f.name.startswith("agent-"):
                continue
            try:
                mtime = f.stat().st_mtime
            except OSError:
                continue
            candidates.append((mtime, f, "Codex"))

    candidates.sort(key=lambda x: x[0], reverse=True)

    results = []
    for _mtime, f, source in candidates:
        summary = get_session_summary(f)
        if summary.lower() == "warmup" or summary == "(no summary)":
            continue
        results.append((f, summary, source))
        if len(results) >= limit:
            break
    return results


def get_project_display_name(folder_name):
    """Convert encoded folder name to readable project name.

    Claude Code stores projects in folders like:
    - -home-user-projects-myproject -> myproject
    - -mnt-c-Users-name-Projects-app -> app

    For nested paths under common roots (home, projects, code, Users, etc.),
    extracts the meaningful project portion.
    """
    # Common path prefixes to strip
    prefixes_to_strip = [
        "-home-",
        "-mnt-c-Users-",
        "-mnt-c-users-",
        "-Users-",
    ]

    name = folder_name
    for prefix in prefixes_to_strip:
        if name.lower().startswith(prefix.lower()):
            name = name[len(prefix) :]
            break

    # Split on dashes and find meaningful parts
    parts = name.split("-")

    # Common intermediate directories to skip
    skip_dirs = {"projects", "code", "repos", "src", "dev", "work", "documents"}

    # Find the first meaningful part (after skipping username and common dirs)
    meaningful_parts = []
    found_project = False

    for i, part in enumerate(parts):
        if not part:
            continue
        # Skip the first part if it looks like a username (before common dirs)
        if i == 0 and not found_project:
            # Check if next parts contain common dirs
            remaining = [p.lower() for p in parts[i + 1 :]]
            if any(d in remaining for d in skip_dirs):
                continue
        if part.lower() in skip_dirs:
            found_project = True
            continue
        meaningful_parts.append(part)
        found_project = True

    if meaningful_parts:
        return "-".join(meaningful_parts)

    # Fallback: return last non-empty part or original
    for part in reversed(parts):
        if part:
            return part
    return folder_name


def find_all_sessions(folder, include_agents=False):
    """Find all sessions in a Claude projects folder, grouped by project.

    Returns a list of project dicts, each containing:
    - name: display name for the project
    - path: Path to the project folder
    - sessions: list of session dicts with path, summary, mtime, size

    Sessions are sorted by modification time (most recent first) within each project.
    Projects are sorted by their most recent session.
    """
    folder = Path(folder)
    if not folder.exists():
        return []

    projects = {}

    for session_file in folder.glob("**/*.jsonl"):
        # Skip agent files unless requested
        if not include_agents and session_file.name.startswith("agent-"):
            continue

        # Get summary and skip boring sessions
        summary = get_session_summary(session_file)
        if summary.lower() == "warmup" or summary == "(no summary)":
            continue

        # Get project folder
        project_folder = session_file.parent
        project_key = project_folder.name

        if project_key not in projects:
            projects[project_key] = {
                "name": get_project_display_name(project_key),
                "path": project_folder,
                "sessions": [],
            }

        stat = session_file.stat()
        projects[project_key]["sessions"].append(
            {
                "path": session_file,
                "summary": summary,
                "mtime": stat.st_mtime,
                "size": stat.st_size,
            }
        )

    # Sort sessions within each project by mtime (most recent first)
    for project in projects.values():
        project["sessions"].sort(key=lambda s: s["mtime"], reverse=True)

    # Convert to list and sort projects by most recent session
    result = list(projects.values())
    result.sort(
        key=lambda p: p["sessions"][0]["mtime"] if p["sessions"] else 0, reverse=True
    )

    return result


def generate_batch_html(
    source_folder, output_dir, include_agents=False, progress_callback=None
):
    """Generate HTML archive for all sessions in a Claude projects folder.

    Creates:
    - Master index.html listing all projects
    - Per-project directories with index.html listing sessions
    - Per-session directories with transcript pages

    Args:
        source_folder: Path to the Claude projects folder
        output_dir: Path for output archive
        include_agents: Whether to include agent-* session files
        progress_callback: Optional callback(project_name, session_name, current, total)
            called after each session is processed

    Returns statistics dict with total_projects, total_sessions, failed_sessions, output_dir.
    """
    source_folder = Path(source_folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all sessions
    projects = find_all_sessions(source_folder, include_agents=include_agents)

    # Calculate total for progress tracking
    total_session_count = sum(len(p["sessions"]) for p in projects)
    processed_count = 0
    successful_sessions = 0
    failed_sessions = []

    # Process each project
    for project in projects:
        project_dir = output_dir / project["name"]
        project_dir.mkdir(exist_ok=True)

        # Process each session
        for session in project["sessions"]:
            session_name = session["path"].stem
            session_dir = project_dir / session_name

            # Generate transcript HTML with error handling
            try:
                generate_html(session["path"], session_dir)
                successful_sessions += 1
            except Exception as e:
                failed_sessions.append(
                    {
                        "project": project["name"],
                        "session": session_name,
                        "error": str(e),
                    }
                )

            processed_count += 1

            # Call progress callback if provided
            if progress_callback:
                progress_callback(
                    project["name"], session_name, processed_count, total_session_count
                )

        # Generate project index
        _generate_project_index(project, project_dir)

    # Generate master index
    _generate_master_index(projects, output_dir)

    return {
        "total_projects": len(projects),
        "total_sessions": successful_sessions,
        "failed_sessions": failed_sessions,
        "output_dir": output_dir,
    }


def _generate_project_index(project, output_dir):
    """Generate index.html for a single project."""
    template = get_template("project_index.html")

    # Format sessions for template
    sessions_data = []
    for session in project["sessions"]:
        mod_time = datetime.fromtimestamp(session["mtime"])
        sessions_data.append(
            {
                "name": session["path"].stem,
                "summary": session["summary"],
                "date": mod_time.strftime("%Y-%m-%d %H:%M"),
                "size_kb": session["size"] / 1024,
            }
        )

    html_content = template.render(
        project_name=project["name"],
        sessions=sessions_data,
        session_count=len(sessions_data),
        css=CSS,
        js=JS,
    )

    output_path = output_dir / "index.html"
    output_path.write_text(html_content, encoding="utf-8")


def _generate_master_index(projects, output_dir):
    """Generate master index.html listing all projects."""
    template = get_template("master_index.html")

    # Format projects for template
    projects_data = []
    total_sessions = 0

    for project in projects:
        session_count = len(project["sessions"])
        total_sessions += session_count

        # Get most recent session date
        if project["sessions"]:
            most_recent = datetime.fromtimestamp(project["sessions"][0]["mtime"])
            recent_date = most_recent.strftime("%Y-%m-%d")
        else:
            recent_date = "N/A"

        projects_data.append(
            {
                "name": project["name"],
                "session_count": session_count,
                "recent_date": recent_date,
            }
        )

    html_content = template.render(
        projects=projects_data,
        total_projects=len(projects),
        total_sessions=total_sessions,
        css=CSS,
        js=JS,
    )

    output_path = output_dir / "index.html"
    output_path.write_text(html_content, encoding="utf-8")


def parse_session_file(filepath):
    """Parse a session file and return normalized data.

    Supports both JSON and JSONL formats.
    Returns a dict with 'loglines' key containing the normalized entries.
    """
    filepath = Path(filepath)

    if filepath.suffix == ".jsonl":
        return _parse_jsonl_file(filepath)
    else:
        # Standard JSON format
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)


def _is_codex_cli_format(filepath):
    """Detect if a JSONL file is in Codex CLI format.

    Checks the first few lines for Codex CLI markers like session_meta or response_item.
    """
    try:
        saw_claude_message = False
        with open(filepath, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= 25:  # Check the first 25 lines
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    entry_type = obj.get("type")
                    # Codex CLI markers (new and old formats)
                    if entry_type in (
                        "session_meta",
                        "response_item",
                        "turn_context",
                        "event_msg",
                    ):
                        return True
                    if "record_type" in obj:
                        return True
                    if entry_type == "message" and obj.get("role") in (
                        "user",
                        "assistant",
                    ):
                        return True
                    if entry_type in (
                        "function_call",
                        "function_call_output",
                        "reasoning",
                    ):
                        return True
                    # Claude Code has "type" as user/assistant
                    if entry_type in ("user", "assistant"):
                        saw_claude_message = True
                except json.JSONDecodeError:
                    continue
        if saw_claude_message:
            return False
    except Exception:
        pass
    return False


def _map_codex_tool_to_claude(tool_name):
    """Map Codex CLI tool names to Claude Code tool names."""
    mapping = {
        "shell_command": "Bash",
        "read_file": "Read",
        "write_file": "Write",
        "edit_file": "Edit",
        "search_files": "Grep",
        "list_files": "Glob",
    }
    return mapping.get(tool_name, tool_name)


def _convert_codex_content_to_claude(content_blocks):
    """Convert Codex CLI content blocks to Claude Code format.

    Args:
        content_blocks: List of Codex content blocks like [{"type": "input_text", "text": "..."}]

    Returns:
        Either a string (for simple text) or list of Claude Code content blocks
    """
    if not content_blocks:
        return []

    # If there's only one input_text block, return as simple string
    if len(content_blocks) == 1 and content_blocks[0].get("type") == "input_text":
        return content_blocks[0].get("text", "")

    # Otherwise convert to Claude Code format
    claude_blocks = []
    for block in content_blocks:
        block_type = block.get("type")
        if block_type == "input_text":
            claude_blocks.append({"type": "text", "text": block.get("text", "")})
        elif block_type == "output_text":
            claude_blocks.append({"type": "text", "text": block.get("text", "")})
        elif block_type == "text":
            # Already in Claude format
            claude_blocks.append(block)
        else:
            # Pass through other types
            claude_blocks.append(block)

    return claude_blocks


def _parse_codex_jsonl_file(filepath):
    """Parse Codex CLI JSONL file and convert to Claude Code format."""
    loglines = []

    def add_message(role, content, timestamp):
        if role not in ("user", "assistant"):
            return
        converted_content = _convert_codex_content_to_claude(content)
        loglines.append(
            {
                "type": role,
                "timestamp": timestamp,
                "message": {"role": role, "content": converted_content},
            }
        )

    def add_tool_use(tool_name, arguments, call_id, timestamp):
        if isinstance(arguments, str):
            try:
                tool_input = json.loads(arguments)
            except json.JSONDecodeError:
                tool_input = {}
        elif isinstance(arguments, dict):
            tool_input = arguments
        else:
            tool_input = {}

        claude_tool_name = _map_codex_tool_to_claude(tool_name)
        loglines.append(
            {
                "type": "assistant",
                "timestamp": timestamp,
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": call_id,
                            "name": claude_tool_name,
                            "input": tool_input,
                        }
                    ],
                },
            }
        )

    def add_tool_result(call_id, output, timestamp, is_error=False):
        loglines.append(
            {
                "type": "user",
                "timestamp": timestamp,
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": call_id,
                            "content": output,
                            "is_error": is_error,
                        }
                    ],
                },
            }
        )

    def add_reasoning_summary(summary, timestamp):
        texts = []
        if isinstance(summary, str):
            texts.append(summary)
        elif isinstance(summary, list):
            for item in summary:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text", "")
                    if text:
                        texts.append(text)

        combined = "\n\n".join(text.strip() for text in texts if text and text.strip())
        if not combined:
            return

        loglines.append(
            {
                "type": "assistant",
                "timestamp": timestamp,
                "message": {
                    "role": "assistant",
                    "content": [{"type": "thinking", "thinking": combined}],
                },
            }
        )

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                record_type = obj.get("type")
                timestamp = obj.get("timestamp", "")

                if record_type == "response_item":
                    payload = obj.get("payload", {})
                    payload_type = payload.get("type")

                    if payload_type == "message":
                        add_message(
                            payload.get("role"),
                            payload.get("content", []),
                            timestamp,
                        )
                    elif payload_type == "function_call":
                        add_tool_use(
                            payload.get("name", ""),
                            payload.get("arguments", "{}"),
                            payload.get("call_id", ""),
                            timestamp,
                        )
                    elif payload_type == "function_call_output":
                        add_tool_result(
                            payload.get("call_id", ""),
                            payload.get("output", ""),
                            timestamp,
                            bool(payload.get("is_error")),
                        )
                    elif payload_type == "reasoning":
                        add_reasoning_summary(payload.get("summary"), timestamp)
                elif record_type == "message":
                    add_message(
                        obj.get("role"),
                        obj.get("content", []),
                        timestamp,
                    )
                elif record_type == "function_call":
                    call_id = obj.get("call_id") or obj.get("id", "")
                    add_tool_use(
                        obj.get("name", ""),
                        obj.get("arguments", "{}"),
                        call_id,
                        timestamp,
                    )
                elif record_type == "function_call_output":
                    call_id = obj.get("call_id") or obj.get("id", "")
                    add_tool_result(
                        call_id,
                        obj.get("output", ""),
                        timestamp,
                        bool(obj.get("is_error")),
                    )
                elif record_type == "reasoning":
                    add_reasoning_summary(obj.get("summary"), timestamp)

            except json.JSONDecodeError:
                continue

    return {"loglines": loglines}


def _parse_jsonl_file(filepath):
    """Parse JSONL file and convert to standard format.

    Automatically detects and handles both Claude Code and Codex CLI formats.
    """
    # Detect format
    if _is_codex_cli_format(filepath):
        return _parse_codex_jsonl_file(filepath)

    # Original Claude Code format parsing
    loglines = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                entry_type = obj.get("type")

                # Skip non-message entries
                if entry_type not in ("user", "assistant"):
                    continue

                # Convert to standard format
                entry = {
                    "type": entry_type,
                    "timestamp": obj.get("timestamp", ""),
                    "message": obj.get("message", {}),
                }

                # Preserve isCompactSummary if present
                if obj.get("isCompactSummary"):
                    entry["isCompactSummary"] = True

                loglines.append(entry)
            except json.JSONDecodeError:
                continue

    return {"loglines": loglines}


class CredentialsError(Exception):
    """Raised when credentials cannot be obtained."""

    pass


def get_access_token_from_keychain():
    """Get access token from macOS keychain.

    Returns the access token or None if not found.
    Raises CredentialsError with helpful message on failure.
    """
    if platform.system() != "Darwin":
        return None

    try:
        result = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-a",
                os.environ.get("USER", ""),
                "-s",
                "Claude Code-credentials",
                "-w",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None

        # Parse the JSON to get the access token
        creds = json.loads(result.stdout.strip())
        return creds.get("claudeAiOauth", {}).get("accessToken")
    except (json.JSONDecodeError, subprocess.SubprocessError):
        return None


def get_org_uuid_from_config():
    """Get organization UUID from ~/.claude.json.

    Returns the organization UUID or None if not found.
    """
    config_path = Path.home() / ".claude.json"
    if not config_path.exists():
        return None

    try:
        with open(config_path) as f:
            config = json.load(f)
        return config.get("oauthAccount", {}).get("organizationUuid")
    except (json.JSONDecodeError, IOError):
        return None


def get_api_headers(token, org_uuid):
    """Build API request headers."""
    return {
        "Authorization": f"Bearer {token}",
        "anthropic-version": ANTHROPIC_VERSION,
        "Content-Type": "application/json",
        "x-organization-uuid": org_uuid,
    }


def fetch_sessions(token, org_uuid):
    """Fetch list of sessions from the API.

    Returns the sessions data as a dict.
    Raises httpx.HTTPError on network/API errors.
    """
    headers = get_api_headers(token, org_uuid)
    response = httpx.get(f"{API_BASE_URL}/sessions", headers=headers, timeout=30.0)
    response.raise_for_status()
    return response.json()


def fetch_session(token, org_uuid, session_id):
    """Fetch a specific session from the API.

    Returns the session data as a dict.
    Raises httpx.HTTPError on network/API errors.
    """
    headers = get_api_headers(token, org_uuid)
    response = httpx.get(
        f"{API_BASE_URL}/session_ingress/session/{session_id}",
        headers=headers,
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()


def detect_github_repo(loglines):
    """
    Detect GitHub repo from git push output in tool results.

    Looks for patterns like:
    - github.com/owner/repo/pull/new/branch (from git push messages)

    Returns the first detected repo (owner/name) or None.
    """
    for entry in loglines:
        message = entry.get("message", {})
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_result":
                result_content = block.get("content", "")
                if isinstance(result_content, str):
                    match = GITHUB_REPO_PATTERN.search(result_content)
                    if match:
                        return match.group(1)
    return None


def format_json(obj):
    try:
        if isinstance(obj, str):
            obj = json.loads(obj)
        formatted = json.dumps(obj, indent=2, ensure_ascii=False)
        return f'<pre class="json">{html.escape(formatted)}</pre>'
    except (json.JSONDecodeError, TypeError):
        return f"<pre>{html.escape(str(obj))}</pre>"


def render_markdown_text(text):
    if not text:
        return ""
    return markdown.markdown(text, extensions=["fenced_code", "tables"])


def is_json_like(text):
    if not text or not isinstance(text, str):
        return False
    text = text.strip()
    return (text.startswith("{") and text.endswith("}")) or (
        text.startswith("[") and text.endswith("]")
    )


def render_todo_write(tool_input, tool_id):
    todos = tool_input.get("todos", [])
    if not todos:
        return ""
    return _macros.todo_list(todos, tool_id)


def render_write_tool(tool_input, tool_id):
    """Render Write tool calls with file path header and content preview."""
    file_path = tool_input.get("file_path", "Unknown file")
    content = tool_input.get("content", "")
    return _macros.write_tool(file_path, content, tool_id)


def render_edit_tool(tool_input, tool_id):
    """Render Edit tool calls with diff-like old/new display."""
    file_path = tool_input.get("file_path", "Unknown file")
    old_string = tool_input.get("old_string", "")
    new_string = tool_input.get("new_string", "")
    replace_all = tool_input.get("replace_all", False)
    return _macros.edit_tool(file_path, old_string, new_string, replace_all, tool_id)


def render_bash_tool(tool_input, tool_id):
    """Render Bash tool calls with command as plain text."""
    command = tool_input.get("command", "")
    description = tool_input.get("description", "")
    return _macros.bash_tool(command, description, tool_id)


def render_content_block(block):
    if not isinstance(block, dict):
        return f"<p>{html.escape(str(block))}</p>"
    block_type = block.get("type", "")
    if block_type == "image":
        source = block.get("source", {})
        media_type = source.get("media_type", "image/png")
        data = source.get("data", "")
        return _macros.image_block(media_type, data)
    elif block_type == "thinking":
        content_html = render_markdown_text(block.get("thinking", ""))
        return _macros.thinking(content_html)
    elif block_type == "text":
        content_html = render_markdown_text(block.get("text", ""))
        return _macros.assistant_text(content_html)
    elif block_type == "tool_use":
        tool_name = block.get("name", "Unknown tool")
        tool_input = block.get("input", {})
        tool_id = block.get("id", "")
        if tool_name == "TodoWrite":
            return render_todo_write(tool_input, tool_id)
        if tool_name == "Write":
            return render_write_tool(tool_input, tool_id)
        if tool_name == "Edit":
            return render_edit_tool(tool_input, tool_id)
        if tool_name == "Bash":
            return render_bash_tool(tool_input, tool_id)
        description = tool_input.get("description", "")
        display_input = {k: v for k, v in tool_input.items() if k != "description"}
        input_json = json.dumps(display_input, indent=2, ensure_ascii=False)
        return _macros.tool_use(tool_name, description, input_json, tool_id)
    elif block_type == "tool_result":
        content = block.get("content", "")
        is_error = block.get("is_error", False)

        # Check for git commits and render with styled cards
        if isinstance(content, str):
            commits_found = list(COMMIT_PATTERN.finditer(content))
            if commits_found:
                # Build commit cards + remaining content
                parts = []
                last_end = 0
                for match in commits_found:
                    # Add any content before this commit
                    before = content[last_end : match.start()].strip()
                    if before:
                        parts.append(f"<pre>{html.escape(before)}</pre>")

                    commit_hash = match.group(1)
                    commit_msg = match.group(2)
                    parts.append(
                        _macros.commit_card(commit_hash, commit_msg, _github_repo)
                    )
                    last_end = match.end()

                # Add any remaining content after last commit
                after = content[last_end:].strip()
                if after:
                    parts.append(f"<pre>{html.escape(after)}</pre>")

                content_html = "".join(parts)
            else:
                content_html = f"<pre>{html.escape(content)}</pre>"
        elif isinstance(content, list) or is_json_like(content):
            content_html = format_json(content)
        else:
            content_html = format_json(content)
        return _macros.tool_result(content_html, is_error)
    else:
        return format_json(block)


def render_user_message_content(message_data):
    content = message_data.get("content", "")
    if isinstance(content, str):
        if is_json_like(content):
            return _macros.user_content(format_json(content))
        return _macros.user_content(render_markdown_text(content))
    elif isinstance(content, list):
        return "".join(render_content_block(block) for block in content)
    return f"<p>{html.escape(str(content))}</p>"


def render_assistant_message(message_data):
    content = message_data.get("content", [])
    if not isinstance(content, list):
        return f"<p>{html.escape(str(content))}</p>"
    return "".join(render_content_block(block) for block in content)


def _extract_translatable_text_from_blocks(blocks):
    """Extract a best-effort markdown string from Claude/Codex content blocks."""
    if not isinstance(blocks, list):
        return ""

    parts = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type", "")
        if block_type in ("text", "input_text"):
            text = block.get("text", "")
            if text:
                parts.append(text)
        elif block_type == "thinking":
            thinking = block.get("thinking", "")
            if thinking:
                parts.append(thinking)
    return "\n\n".join(part.strip() for part in parts if part and part.strip())


def _collect_translatable_texts_for_message(log_type, message_data):
    """Collect raw strings that would be translated for this message."""
    texts = []

    if log_type == "user":
        if is_tool_result_message(message_data):
            return []
        content = message_data.get("content", "")
        if isinstance(content, str):
            if content and content.strip():
                texts.append(content)
        else:
            extracted = _extract_translatable_text_from_blocks(content)
            if extracted and extracted.strip():
                texts.append(extracted)
        return [t.strip() for t in texts if t and t.strip()]

    if log_type == "assistant":
        content = message_data.get("content", [])
        if not isinstance(content, list):
            return []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type", "")
            if block_type == "text":
                text = block.get("text", "")
                if text and text.strip():
                    texts.append(text)
            elif block_type == "thinking":
                thinking = block.get("thinking", "")
                if thinking and thinking.strip():
                    texts.append(thinking)
        return [t.strip() for t in texts if t and t.strip()]

    return []


def _translate_texts(translator, texts):
    if translator is None:
        return []
    if hasattr(translator, "translate_many") and len(texts) > 1:
        return translator.translate_many(list(texts))
    return [translator(t) for t in texts]


def _get_message_translation(log_type, message_data, translator):
    """Return (translation_html, translation_text) for one message."""
    if translator is None:
        return None, None

    try:
        if log_type == "user":
            # Don't translate tool result "user" messages by default.
            if is_tool_result_message(message_data):
                return "", ""
            content = message_data.get("content", "")
            if isinstance(content, str):
                source_text = content
            else:
                source_text = _extract_translatable_text_from_blocks(content)

            translated = _translate_texts(translator, [source_text])[0]
            translated = (translated or "").strip()
            if not translated:
                return "", ""
            return _macros.user_content(render_markdown_text(translated)), translated

        if log_type == "assistant":
            content = message_data.get("content", [])
            if not isinstance(content, list):
                return "", ""

            to_translate = []
            translate_specs = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type", "")
                if block_type == "text":
                    text = block.get("text", "")
                    if text:
                        translate_specs.append(("text", len(to_translate)))
                        to_translate.append(text)
                elif block_type == "thinking":
                    thinking = block.get("thinking", "")
                    if thinking:
                        translate_specs.append(("thinking", len(to_translate)))
                        to_translate.append(thinking)

            if not to_translate:
                return "", ""

            translated_texts = _translate_texts(translator, to_translate)
            translated_blocks = []
            for kind, idx in translate_specs:
                translated = (translated_texts[idx] or "").strip()
                if not translated:
                    continue
                if kind == "text":
                    translated_blocks.append({"type": "text", "text": translated})
                else:
                    translated_blocks.append(
                        {"type": "thinking", "thinking": translated}
                    )

            if not translated_blocks:
                return "", ""
            html_out = render_assistant_message({"content": translated_blocks})
            text_out = _extract_translatable_text_from_blocks(translated_blocks)
            return html_out, text_out

    except Exception as exc:
        error = str(exc)
        return f"<pre>{html.escape(error)}</pre>", error

    return "", ""


def make_msg_id(timestamp):
    return f"msg-{timestamp.replace(':', '-').replace('.', '-')}"


def analyze_conversation(messages):
    """Analyze messages in a conversation to extract stats and long texts."""
    tool_counts = {}  # tool_name -> count
    long_texts = []
    commits = []  # list of (hash, message, timestamp)

    for log_type, message_json, timestamp in messages:
        if not message_json:
            continue
        try:
            message_data = json.loads(message_json)
        except json.JSONDecodeError:
            continue

        content = message_data.get("content", [])
        if not isinstance(content, list):
            continue

        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type", "")

            if block_type == "tool_use":
                tool_name = block.get("name", "Unknown")
                tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
            elif block_type == "tool_result":
                # Check for git commit output
                result_content = block.get("content", "")
                if isinstance(result_content, str):
                    for match in COMMIT_PATTERN.finditer(result_content):
                        commits.append((match.group(1), match.group(2), timestamp))
            elif block_type == "text":
                text = block.get("text", "")
                if len(text) >= LONG_TEXT_THRESHOLD:
                    long_texts.append(text)

    return {
        "tool_counts": tool_counts,
        "long_texts": long_texts,
        "commits": commits,
    }


def format_tool_stats(tool_counts):
    """Format tool counts into a concise summary string."""
    if not tool_counts:
        return ""

    # Abbreviate common tool names
    abbrev = {
        "Bash": "bash",
        "Read": "read",
        "Write": "write",
        "Edit": "edit",
        "Glob": "glob",
        "Grep": "grep",
        "Task": "task",
        "TodoWrite": "todo",
        "WebFetch": "fetch",
        "WebSearch": "search",
    }

    parts = []
    for name, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
        short_name = abbrev.get(name, name.lower())
        parts.append(f"{count} {short_name}")

    return " · ".join(parts)


def is_tool_result_message(message_data):
    """Check if a message contains only tool_result blocks."""
    content = message_data.get("content", [])
    if not isinstance(content, list):
        return False
    if not content:
        return False
    return all(
        isinstance(block, dict) and block.get("type") == "tool_result"
        for block in content
    )


def render_message(log_type, message_json, timestamp, translator=None):
    if not message_json:
        return ""
    try:
        message_data = json.loads(message_json)
    except json.JSONDecodeError:
        return ""
    msg_html, _ = render_message_data(
        log_type, message_data, timestamp, translator=translator
    )
    return msg_html


def render_message_data(log_type, message_data, timestamp, translator=None):
    if log_type == "user":
        content_html = render_user_message_content(message_data)
        # Check if this is a tool result message
        if is_tool_result_message(message_data):
            role_class, role_label = "tool-reply", "Tool reply"
        else:
            role_class, role_label = "user", "User"
    elif log_type == "assistant":
        content_html = render_assistant_message(message_data)
        role_class, role_label = "assistant", "Assistant"
    else:
        return "", None
    if not content_html.strip():
        return "", None
    msg_id = make_msg_id(timestamp)
    translation_html, translation_text = _get_message_translation(
        log_type, message_data, translator
    )
    return (
        _macros.message(
        role_class, role_label, msg_id, timestamp, content_html, translation_html
        ),
        translation_text,
    )


CSS = """
:root { --bg-color: #f5f5f5; --card-bg: #ffffff; --user-bg: #e3f2fd; --user-border: #1976d2; --assistant-bg: #f5f5f5; --assistant-border: #9e9e9e; --thinking-bg: #fff8e1; --thinking-border: #ffc107; --thinking-text: #666; --tool-bg: #f3e5f5; --tool-border: #9c27b0; --tool-result-bg: #e8f5e9; --tool-error-bg: #ffebee; --text-color: #212121; --text-muted: #757575; --code-bg: #263238; --code-text: #aed581; }
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: var(--bg-color); color: var(--text-color); margin: 0; padding: 16px; line-height: 1.6; }
.container { max-width: 800px; margin: 0 auto; }
h1 { font-size: 1.5rem; margin-bottom: 24px; padding-bottom: 8px; border-bottom: 2px solid var(--user-border); }
.header-row { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; border-bottom: 2px solid var(--user-border); padding-bottom: 8px; margin-bottom: 24px; }
.header-row h1 { border-bottom: none; padding-bottom: 0; margin-bottom: 0; flex: 1; min-width: 200px; }
.message { margin-bottom: 16px; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.message.user { background: var(--user-bg); border-left: 4px solid var(--user-border); }
.message.assistant { background: var(--card-bg); border-left: 4px solid var(--assistant-border); }
.message.tool-reply { background: #fff8e1; border-left: 4px solid #ff9800; }
.tool-reply .role-label { color: #e65100; }
.tool-reply .tool-result { background: transparent; padding: 0; margin: 0; }
.tool-reply .tool-result .truncatable.truncated::after { background: linear-gradient(to bottom, transparent, #fff8e1); }
.message-header { display: flex; justify-content: space-between; align-items: center; padding: 8px 16px; background: rgba(0,0,0,0.03); font-size: 0.85rem; }
.role-label { font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
.user .role-label { color: var(--user-border); }
time { color: var(--text-muted); font-size: 0.8rem; }
.timestamp-link { color: inherit; text-decoration: none; }
.timestamp-link:hover { text-decoration: underline; }
.message:target { animation: highlight 2s ease-out; }
@keyframes highlight { 0% { background-color: rgba(25, 118, 210, 0.2); } 100% { background-color: transparent; } }
.message-content { padding: 16px; }
.message-content p { margin: 0 0 12px 0; }
.message-content p:last-child { margin-bottom: 0; }
.thinking { background: var(--thinking-bg); border: 1px solid var(--thinking-border); border-radius: 8px; padding: 12px; margin: 12px 0; font-size: 0.9rem; color: var(--thinking-text); }
.thinking-label { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; color: #f57c00; margin-bottom: 8px; }
.thinking p { margin: 8px 0; }
.assistant-text { margin: 8px 0; }
.tool-use { background: var(--tool-bg); border: 1px solid var(--tool-border); border-radius: 8px; padding: 12px; margin: 12px 0; }
.tool-header { font-weight: 600; color: var(--tool-border); margin-bottom: 8px; display: flex; align-items: center; gap: 8px; }
.tool-icon { font-size: 1.1rem; }
.tool-description { font-size: 0.9rem; color: var(--text-muted); margin-bottom: 8px; font-style: italic; }
.tool-result { background: var(--tool-result-bg); border-radius: 8px; padding: 12px; margin: 12px 0; }
.tool-result.tool-error { background: var(--tool-error-bg); }
.file-tool { border-radius: 8px; padding: 12px; margin: 12px 0; }
.write-tool { background: linear-gradient(135deg, #e3f2fd 0%, #e8f5e9 100%); border: 1px solid #4caf50; }
.edit-tool { background: linear-gradient(135deg, #fff3e0 0%, #fce4ec 100%); border: 1px solid #ff9800; }
.file-tool-header { font-weight: 600; margin-bottom: 4px; display: flex; align-items: center; gap: 8px; font-size: 0.95rem; }
.write-header { color: #2e7d32; }
.edit-header { color: #e65100; }
.file-tool-icon { font-size: 1rem; }
.file-tool-path { font-family: monospace; background: rgba(0,0,0,0.08); padding: 2px 8px; border-radius: 4px; }
.file-tool-fullpath { font-family: monospace; font-size: 0.8rem; color: var(--text-muted); margin-bottom: 8px; word-break: break-all; }
.file-content { margin: 0; }
.edit-section { display: flex; margin: 4px 0; border-radius: 4px; overflow: hidden; }
.edit-label { padding: 8px 12px; font-weight: bold; font-family: monospace; display: flex; align-items: flex-start; }
.edit-old { background: #fce4ec; }
.edit-old .edit-label { color: #b71c1c; background: #f8bbd9; }
.edit-old .edit-content { color: #880e4f; }
.edit-new { background: #e8f5e9; }
.edit-new .edit-label { color: #1b5e20; background: #a5d6a7; }
.edit-new .edit-content { color: #1b5e20; }
.edit-content { margin: 0; flex: 1; background: transparent; font-size: 0.85rem; }
.edit-replace-all { font-size: 0.75rem; font-weight: normal; color: var(--text-muted); }
.write-tool .truncatable.truncated::after { background: linear-gradient(to bottom, transparent, #e6f4ea); }
.edit-tool .truncatable.truncated::after { background: linear-gradient(to bottom, transparent, #fff0e5); }
.todo-list { background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 100%); border: 1px solid #81c784; border-radius: 8px; padding: 12px; margin: 12px 0; }
.todo-header { font-weight: 600; color: #2e7d32; margin-bottom: 10px; display: flex; align-items: center; gap: 8px; font-size: 0.95rem; }
.todo-items { list-style: none; margin: 0; padding: 0; }
.todo-item { display: flex; align-items: flex-start; gap: 10px; padding: 6px 0; border-bottom: 1px solid rgba(0,0,0,0.06); font-size: 0.9rem; }
.todo-item:last-child { border-bottom: none; }
.todo-icon { flex-shrink: 0; width: 20px; height: 20px; display: flex; align-items: center; justify-content: center; font-weight: bold; border-radius: 50%; }
.todo-completed .todo-icon { color: #2e7d32; background: rgba(46, 125, 50, 0.15); }
.todo-completed .todo-content { color: #558b2f; text-decoration: line-through; }
.todo-in-progress .todo-icon { color: #f57c00; background: rgba(245, 124, 0, 0.15); }
.todo-in-progress .todo-content { color: #e65100; font-weight: 500; }
.todo-pending .todo-icon { color: #757575; background: rgba(0,0,0,0.05); }
.todo-pending .todo-content { color: #616161; }
pre { background: var(--code-bg); color: var(--code-text); padding: 12px; border-radius: 6px; overflow-x: auto; font-size: 0.85rem; line-height: 1.5; margin: 8px 0; white-space: pre-wrap; word-wrap: break-word; }
pre.json { color: #e0e0e0; }
code { background: rgba(0,0,0,0.08); padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }
pre code { background: none; padding: 0; }
.user-content { margin: 0; }
.truncatable { position: relative; }
.truncatable.truncated .truncatable-content { max-height: 200px; overflow: hidden; }
.truncatable.truncated::after { content: ''; position: absolute; bottom: 32px; left: 0; right: 0; height: 60px; background: linear-gradient(to bottom, transparent, var(--card-bg)); pointer-events: none; }
.message.user .truncatable.truncated::after { background: linear-gradient(to bottom, transparent, var(--user-bg)); }
.message.tool-reply .truncatable.truncated::after { background: linear-gradient(to bottom, transparent, #fff8e1); }
.tool-use .truncatable.truncated::after { background: linear-gradient(to bottom, transparent, var(--tool-bg)); }
.tool-result .truncatable.truncated::after { background: linear-gradient(to bottom, transparent, var(--tool-result-bg)); }
.expand-btn { display: none; width: 100%; padding: 8px 16px; margin-top: 4px; background: rgba(0,0,0,0.05); border: 1px solid rgba(0,0,0,0.1); border-radius: 6px; cursor: pointer; font-size: 0.85rem; color: var(--text-muted); }
.expand-btn:hover { background: rgba(0,0,0,0.1); }
.truncatable.truncated .expand-btn, .truncatable.expanded .expand-btn { display: block; }
.pagination { display: flex; justify-content: center; gap: 8px; margin: 24px 0; flex-wrap: wrap; }
.pagination a, .pagination span { padding: 5px 10px; border-radius: 6px; text-decoration: none; font-size: 0.85rem; }
.pagination a { background: var(--card-bg); color: var(--user-border); border: 1px solid var(--user-border); }
.pagination a:hover { background: var(--user-bg); }
.pagination .current { background: var(--user-border); color: white; }
.pagination .disabled { color: var(--text-muted); border: 1px solid #ddd; }
.pagination .index-link { background: var(--user-border); color: white; }
details.continuation { margin-bottom: 16px; }
details.continuation summary { cursor: pointer; padding: 12px 16px; background: var(--user-bg); border-left: 4px solid var(--user-border); border-radius: 12px; font-weight: 500; color: var(--text-muted); }
details.continuation summary:hover { background: rgba(25, 118, 210, 0.15); }
details.continuation[open] summary { border-radius: 12px 12px 0 0; margin-bottom: 0; }
.index-item { margin-bottom: 16px; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); background: var(--user-bg); border-left: 4px solid var(--user-border); }
.index-item a { display: block; text-decoration: none; color: inherit; }
.index-item a:hover { background: rgba(25, 118, 210, 0.1); }
.index-item-header { display: flex; justify-content: space-between; align-items: center; padding: 8px 16px; background: rgba(0,0,0,0.03); font-size: 0.85rem; }
.index-item-number { font-weight: 600; color: var(--user-border); }
.index-item-content { padding: 16px; }
.index-item-stats { padding: 8px 16px 12px 32px; font-size: 0.85rem; color: var(--text-muted); border-top: 1px solid rgba(0,0,0,0.06); }
.index-item-commit { margin-top: 6px; padding: 4px 8px; background: #fff3e0; border-radius: 4px; font-size: 0.85rem; color: #e65100; }
.index-item-commit code { background: rgba(0,0,0,0.08); padding: 1px 4px; border-radius: 3px; font-size: 0.8rem; margin-right: 6px; }
.commit-card { margin: 8px 0; padding: 10px 14px; background: #fff3e0; border-left: 4px solid #ff9800; border-radius: 6px; }
.commit-card a { text-decoration: none; color: #5d4037; display: block; }
.commit-card a:hover { color: #e65100; }
.commit-card-hash { font-family: monospace; color: #e65100; font-weight: 600; margin-right: 8px; }
.index-commit { margin-bottom: 12px; padding: 10px 16px; background: #fff3e0; border-left: 4px solid #ff9800; border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
.index-commit a { display: block; text-decoration: none; color: inherit; }
.index-commit a:hover { background: rgba(255, 152, 0, 0.1); margin: -10px -16px; padding: 10px 16px; border-radius: 8px; }
.index-commit-header { display: flex; justify-content: space-between; align-items: center; font-size: 0.85rem; margin-bottom: 4px; }
.index-commit-hash { font-family: monospace; color: #e65100; font-weight: 600; }
.index-commit-msg { color: #5d4037; }
.index-item-long-text { margin-top: 8px; padding: 12px; background: var(--card-bg); border-radius: 8px; border-left: 3px solid var(--assistant-border); }
.index-item-long-text .truncatable.truncated::after { background: linear-gradient(to bottom, transparent, var(--card-bg)); }
.index-item-long-text-content { color: var(--text-color); }
#search-box { display: none; align-items: center; gap: 8px; }
#search-box input { padding: 6px 12px; border: 1px solid var(--assistant-border); border-radius: 6px; font-size: 16px; width: 180px; }
#search-box button, #modal-search-btn, #modal-close-btn { background: var(--user-border); color: white; border: none; border-radius: 6px; padding: 6px 10px; cursor: pointer; display: flex; align-items: center; justify-content: center; }
#search-box button:hover, #modal-search-btn:hover { background: #1565c0; }
#modal-close-btn { background: var(--text-muted); margin-left: 8px; }
#modal-close-btn:hover { background: #616161; }
#search-modal[open] { border: none; border-radius: 12px; box-shadow: 0 4px 24px rgba(0,0,0,0.2); padding: 0; width: 90vw; max-width: 900px; height: 80vh; max-height: 80vh; display: flex; flex-direction: column; }
#search-modal::backdrop { background: rgba(0,0,0,0.5); }
.search-modal-header { display: flex; align-items: center; gap: 8px; padding: 16px; border-bottom: 1px solid var(--assistant-border); background: var(--bg-color); border-radius: 12px 12px 0 0; }
.search-modal-header input { flex: 1; padding: 8px 12px; border: 1px solid var(--assistant-border); border-radius: 6px; font-size: 16px; }
#search-status { padding: 8px 16px; font-size: 0.85rem; color: var(--text-muted); border-bottom: 1px solid rgba(0,0,0,0.06); }
#search-results { flex: 1; overflow-y: auto; padding: 16px; }
.search-result { margin-bottom: 16px; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.search-result a { display: block; text-decoration: none; color: inherit; }
.search-result a:hover { background: rgba(25, 118, 210, 0.05); }
.search-result-page { padding: 6px 12px; background: rgba(0,0,0,0.03); font-size: 0.8rem; color: var(--text-muted); border-bottom: 1px solid rgba(0,0,0,0.06); }
.search-result-content { padding: 12px; }
.search-result mark { background: #fff59d; padding: 1px 2px; border-radius: 2px; }
@media (max-width: 600px) { body { padding: 8px; } .message, .index-item { border-radius: 8px; } .message-content, .index-item-content { padding: 12px; } pre { font-size: 0.8rem; padding: 8px; } #search-box input { width: 120px; } #search-modal[open] { width: 95vw; height: 90vh; } }
"""

BILINGUAL_CSS = """
.container { max-width: 1200px; }
.page-grid { display: grid; grid-template-columns: 340px 1fr; gap: 16px; align-items: start; }
.page-messages { min-width: 0; }
.sidebar { position: sticky; top: 16px; align-self: start; max-height: calc(100vh - 32px); overflow-y: auto; }
.sidebar-section { background: var(--card-bg); border: 1px solid rgba(0,0,0,0.08); border-radius: 12px; padding: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 12px; }
.sidebar-title { font-weight: 600; font-size: 0.75rem; text-transform: uppercase; color: var(--text-muted); margin-bottom: 8px; letter-spacing: 0.5px; }
.sidebar label { display: flex; gap: 8px; align-items: center; font-size: 0.9rem; color: var(--text-color); margin: 6px 0; cursor: pointer; }
.sidebar input[type="checkbox"] { transform: translateY(1px); }
.turn-summary { border-top: 1px solid rgba(0,0,0,0.06); margin-top: 10px; padding-top: 10px; }
.turn-summary:first-child { border-top: none; margin-top: 0; padding-top: 0; }
.turn-summary-header { display: flex; justify-content: space-between; gap: 8px; align-items: baseline; }
.turn-summary-link { text-decoration: none; color: var(--user-border); font-weight: 600; font-size: 0.9rem; }
.turn-summary-link:hover { text-decoration: underline; }
.turn-summary-body { margin-top: 8px; }
body.hide-bash-tools .bash-tool { display: none; }
body.hide-tool-replies .message.tool-reply { display: none; }
.message-content.bilingual { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.bilingual-col { min-width: 0; }
.bilingual-translation { border-left: 1px solid rgba(0,0,0,0.12); padding-left: 16px; }
.translation-label { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; color: #2e7d32; margin-bottom: 8px; letter-spacing: 0.5px; }
@media (max-width: 900px) { .page-grid { grid-template-columns: 1fr; } .sidebar { position: static; max-height: none; } .message-content.bilingual { grid-template-columns: 1fr; } .bilingual-translation { border-left: none; padding-left: 0; border-top: 1px solid rgba(0,0,0,0.12); padding-top: 12px; } }
"""

JS = """
document.querySelectorAll('time[data-timestamp]').forEach(function(el) {
    const timestamp = el.getAttribute('data-timestamp');
    const date = new Date(timestamp);
    const now = new Date();
    const isToday = date.toDateString() === now.toDateString();
    const timeStr = date.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
    if (isToday) { el.textContent = timeStr; }
    else { el.textContent = date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) + ' ' + timeStr; }
});
document.querySelectorAll('pre.json').forEach(function(el) {
    let text = el.textContent;
    text = text.replace(/"([^"]+)":/g, '<span style="color: #ce93d8">"$1"</span>:');
    text = text.replace(/: "([^"]*)"/g, ': <span style="color: #81d4fa">"$1"</span>');
    text = text.replace(/: (\\d+)/g, ': <span style="color: #ffcc80">$1</span>');
    text = text.replace(/: (true|false|null)/g, ': <span style="color: #f48fb1">$1</span>');
    el.innerHTML = text;
});
document.querySelectorAll('.truncatable').forEach(function(wrapper) {
    const content = wrapper.querySelector('.truncatable-content');
    const btn = wrapper.querySelector('.expand-btn');
    if (content.scrollHeight > 250) {
        wrapper.classList.add('truncated');
        btn.addEventListener('click', function() {
            if (wrapper.classList.contains('truncated')) { wrapper.classList.remove('truncated'); wrapper.classList.add('expanded'); btn.textContent = 'Show less'; }
            else { wrapper.classList.remove('expanded'); wrapper.classList.add('truncated'); btn.textContent = 'Show more'; }
        });
    }
});
"""

BILINGUAL_JS = """
(function() {
    const bashToggle = document.getElementById('toggle-bash-tools');
    const toolReplyToggle = document.getElementById('toggle-tool-replies');
    if (!bashToggle && !toolReplyToggle) return;

    const body = document.body;
    const KEY_BASH = 'cct_show_bash_tools';
    const KEY_TOOL_REPLY = 'cct_show_tool_replies';

    function readBool(key, defaultValue) {
        const v = window.localStorage.getItem(key);
        if (v === null) return defaultValue;
        return v === '1' || v === 'true';
    }
    function writeBool(key, value) {
        window.localStorage.setItem(key, value ? '1' : '0');
    }

    const showBash = readBool(KEY_BASH, true);
    const showToolReply = readBool(KEY_TOOL_REPLY, true);
    if (bashToggle) bashToggle.checked = showBash;
    if (toolReplyToggle) toolReplyToggle.checked = showToolReply;

    function apply() {
        const sBash = bashToggle ? bashToggle.checked : true;
        const sToolReply = toolReplyToggle ? toolReplyToggle.checked : true;
        body.classList.toggle('hide-bash-tools', !sBash);
        body.classList.toggle('hide-tool-replies', !sToolReply);
        if (bashToggle) writeBool(KEY_BASH, sBash);
        if (toolReplyToggle) writeBool(KEY_TOOL_REPLY, sToolReply);
    }

    if (bashToggle) bashToggle.addEventListener('change', apply);
    if (toolReplyToggle) toolReplyToggle.addEventListener('change', apply);
    apply();
})();
"""

# JavaScript to fix relative URLs when served via gisthost.github.io or gistpreview.github.io
# Fixes issue #26: Pagination links broken on gisthost.github.io
GIST_PREVIEW_JS = r"""
(function() {
    var hostname = window.location.hostname;
    if (hostname !== 'gisthost.github.io' && hostname !== 'gistpreview.github.io') return;
    // URL format: https://gisthost.github.io/?GIST_ID/filename.html
    var match = window.location.search.match(/^\?([^/]+)/);
    if (!match) return;
    var gistId = match[1];

    function rewriteLinks(root) {
        (root || document).querySelectorAll('a[href]').forEach(function(link) {
            var href = link.getAttribute('href');
            // Skip already-rewritten links (issue #26 fix)
            if (href.startsWith('?')) return;
            // Skip external links and anchors
            if (href.startsWith('http') || href.startsWith('#') || href.startsWith('//')) return;
            // Handle anchor in relative URL (e.g., page-001.html#msg-123)
            var parts = href.split('#');
            var filename = parts[0];
            var anchor = parts.length > 1 ? '#' + parts[1] : '';
            link.setAttribute('href', '?' + gistId + '/' + filename + anchor);
        });
    }

    // Run immediately
    rewriteLinks();

    // Also run on DOMContentLoaded in case DOM isn't ready yet
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() { rewriteLinks(); });
    }

    // Use MutationObserver to catch dynamically added content
    // gistpreview.github.io may add content after initial load
    var observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            mutation.addedNodes.forEach(function(node) {
                if (node.nodeType === 1) { // Element node
                    rewriteLinks(node);
                    // Also check if the node itself is a link
                    if (node.tagName === 'A' && node.getAttribute('href')) {
                        var href = node.getAttribute('href');
                        if (!href.startsWith('?') && !href.startsWith('http') &&
                            !href.startsWith('#') && !href.startsWith('//')) {
                            var parts = href.split('#');
                            var filename = parts[0];
                            var anchor = parts.length > 1 ? '#' + parts[1] : '';
                            node.setAttribute('href', '?' + gistId + '/' + filename + anchor);
                        }
                    }
                }
            });
        });
    });

    // Start observing once body exists
    function startObserving() {
        if (document.body) {
            observer.observe(document.body, { childList: true, subtree: true });
        } else {
            setTimeout(startObserving, 10);
        }
    }
    startObserving();

    // Handle fragment navigation after dynamic content loads
    // gisthost.github.io/gistpreview.github.io loads content dynamically, so the browser's
    // native fragment navigation fails because the element doesn't exist yet
    function scrollToFragment() {
        var hash = window.location.hash;
        if (!hash) return false;
        var targetId = hash.substring(1);
        var target = document.getElementById(targetId);
        if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            return true;
        }
        return false;
    }

    // Try immediately in case content is already loaded
    if (!scrollToFragment()) {
        // Retry with increasing delays to handle dynamic content loading
        var delays = [100, 300, 500, 1000, 2000];
        delays.forEach(function(delay) {
            setTimeout(scrollToFragment, delay);
        });
    }
})();
"""


def inject_gist_preview_js(output_dir):
    """Inject gist preview JavaScript into all HTML files in the output directory."""
    output_dir = Path(output_dir)
    for html_file in output_dir.glob("*.html"):
        content = html_file.read_text(encoding="utf-8")
        # Insert the gist preview JS before the closing </body> tag
        if "</body>" in content:
            content = content.replace(
                "</body>", f"<script>{GIST_PREVIEW_JS}</script>\n</body>"
            )
            html_file.write_text(content, encoding="utf-8")


def create_gist(output_dir, public=False):
    """Create a GitHub gist from the HTML files in output_dir.

    Returns the gist ID on success, or raises click.ClickException on failure.
    """
    output_dir = Path(output_dir)
    html_files = list(output_dir.glob("*.html"))
    if not html_files:
        raise click.ClickException("No HTML files found to upload to gist.")

    # Build the gh gist create command
    # gh gist create file1 file2 ... --public/--private
    cmd = ["gh", "gist", "create"]
    cmd.extend(str(f) for f in sorted(html_files))
    if public:
        cmd.append("--public")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        # Output is the gist URL, e.g., https://gist.github.com/username/GIST_ID
        gist_url = result.stdout.strip()
        # Extract gist ID from URL
        gist_id = gist_url.rstrip("/").split("/")[-1]
        return gist_id, gist_url
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.strip() if e.stderr else str(e)
        raise click.ClickException(f"Failed to create gist: {error_msg}")
    except FileNotFoundError:
        raise click.ClickException(
            "gh CLI not found. Install it from https://cli.github.com/ and run 'gh auth login'."
        )


def generate_pagination_html(current_page, total_pages):
    return _macros.pagination(current_page, total_pages)


def generate_index_pagination_html(total_pages):
    """Generate pagination for index page where Index is current (first page)."""
    return _macros.index_pagination(total_pages)


def generate_html(json_path, output_dir, github_repo=None, translator=None, profile=False):
    t_total_start = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    css = CSS + BILINGUAL_CSS if translator is not None else CSS
    js = JS + BILINGUAL_JS if translator is not None else JS

    # Load session file (supports both JSON and JSONL)
    t_parse_start = time.perf_counter()
    data = parse_session_file(json_path)
    t_parse = time.perf_counter() - t_parse_start

    loglines = data.get("loglines", [])

    # Auto-detect GitHub repo if not provided
    if github_repo is None:
        github_repo = detect_github_repo(loglines)
        if github_repo:
            print(f"Auto-detected GitHub repo: {github_repo}")
        else:
            print(
                "Warning: Could not auto-detect GitHub repo. Commit links will be disabled."
            )

    # Set module-level variable for render functions
    global _github_repo
    _github_repo = github_repo

    t_convs_start = time.perf_counter()
    conversations = []
    current_conv = None
    for entry in loglines:
        log_type = entry.get("type")
        timestamp = entry.get("timestamp", "")
        is_compact_summary = entry.get("isCompactSummary", False)
        message_data = entry.get("message", {})
        if not message_data:
            continue
        # Convert message dict to JSON string for compatibility with existing render functions
        message_json = json.dumps(message_data)
        is_user_prompt = False
        user_text = None
        if log_type == "user":
            content = message_data.get("content", "")
            text = extract_text_from_content(content)
            if text:
                is_user_prompt = True
                user_text = text
        if is_user_prompt:
            if current_conv:
                conversations.append(current_conv)
            current_conv = {
                "user_text": user_text,
                "timestamp": timestamp,
                "messages": [(log_type, message_json, timestamp)],
                "is_continuation": bool(is_compact_summary),
            }
        elif current_conv:
            current_conv["messages"].append((log_type, message_json, timestamp))
    if current_conv:
        conversations.append(current_conv)
    t_convs = time.perf_counter() - t_convs_start

    total_convs = len(conversations)
    total_pages = (total_convs + PROMPTS_PER_PAGE - 1) // PROMPTS_PER_PAGE

    t_pages_total = 0.0
    t_prefetch_total = 0.0
    t_render_total = 0.0
    t_write_total = 0.0
    translate_unique_total = 0

    for page_num in range(1, total_pages + 1):
        t_page_start = time.perf_counter()
        start_idx = (page_num - 1) * PROMPTS_PER_PAGE
        end_idx = min(start_idx + PROMPTS_PER_PAGE, total_convs)
        page_convs = conversations[start_idx:end_idx]

        # Prefetch translations for the whole page to reduce API round-trips.
        if translator is not None and hasattr(translator, "translate_many"):
            texts = []
            for conv in page_convs:
                for log_type, message_json, _timestamp in conv["messages"]:
                    try:
                        message_data = json.loads(message_json)
                    except json.JSONDecodeError:
                        continue
                    texts.extend(
                        _collect_translatable_texts_for_message(log_type, message_data)
                    )
            if texts:
                unique_texts = list(dict.fromkeys(texts))
                translate_unique_total += len(unique_texts)
                if len(unique_texts) > 1:
                    t_prefetch_start = time.perf_counter()
                    translator.translate_many(unique_texts)
                    t_prefetch_total += time.perf_counter() - t_prefetch_start

        messages_html = []
        page_turn_summaries = []
        t_render_start = time.perf_counter()
        for conv_offset, conv in enumerate(page_convs):
            user_translation_parts = []
            assistant_translation_parts = []
            is_first = True
            for log_type, message_json, timestamp in conv["messages"]:
                try:
                    message_data = json.loads(message_json)
                except json.JSONDecodeError:
                    continue
                msg_html, translation_text = render_message_data(
                    log_type, message_data, timestamp, translator=translator
                )
                if msg_html:
                    # Wrap continuation summaries in collapsed details
                    if is_first and conv.get("is_continuation"):
                        msg_html = f'<details class="continuation"><summary>Session continuation summary</summary>{msg_html}</details>'
                    messages_html.append(msg_html)
                if translation_text:
                    if log_type == "user":
                        user_translation_parts.append(translation_text)
                    elif log_type == "assistant":
                        assistant_translation_parts.append(translation_text)
                is_first = False

            if translator is not None and (
                user_translation_parts or assistant_translation_parts
            ):
                summary_parts = []
                if user_translation_parts:
                    summary_parts.append(
                        "**用户：**\n\n" + "\n\n".join(user_translation_parts)
                    )
                if assistant_translation_parts:
                    summary_parts.append(
                        "**助手：**\n\n" + "\n\n".join(assistant_translation_parts)
                    )
                summary_markdown = "\n\n".join(summary_parts).strip()
                if summary_markdown:
                    page_turn_summaries.append(
                        {
                            "anchor": make_msg_id(conv["timestamp"]),
                            "timestamp": conv["timestamp"],
                            "turn_number": start_idx + conv_offset + 1,
                            "summary_html": render_markdown_text(summary_markdown),
                        }
                    )

        if translator is not None:
            settings_html = (
                '<div class="sidebar-section">'
                '<div class="sidebar-title">显示设置</div>'
                '<label><input type="checkbox" id="toggle-bash-tools" checked>显示命令行</label>'
                '<label><input type="checkbox" id="toggle-tool-replies" checked>显示 Tool reply</label>'
                "</div>"
            )

            summaries = ['<div class="sidebar-section"><div class="sidebar-title">本页中文汇总</div>']
            for item in page_turn_summaries:
                summaries.append(
                    (
                        '<div class="turn-summary">'
                        '<div class="turn-summary-header">'
                        f'<a class="turn-summary-link" href="#{item["anchor"]}">#{item["turn_number"]}</a>'
                        f'<time datetime="{item["timestamp"]}" data-timestamp="{item["timestamp"]}">{item["timestamp"]}</time>'
                        "</div>"
                        '<div class="turn-summary-body">'
                        '<div class="truncatable"><div class="truncatable-content">'
                        f'{item["summary_html"]}'
                        '</div><button class="expand-btn">Show more</button></div>'
                        "</div>"
                        "</div>"
                    )
                )
            summaries.append("</div>")
            sidebar_html = f'<aside class="sidebar">{settings_html}{"".join(summaries)}</aside>'
            messages_html = [
                f'<div class="page-grid">{sidebar_html}<div class="page-messages">{"".join(messages_html)}</div></div>'
            ]
        t_render_total += time.perf_counter() - t_render_start
        pagination_html = generate_pagination_html(page_num, total_pages)
        page_template = get_template("page.html")
        page_content = page_template.render(
            css=css,
            js=js,
            page_num=page_num,
            total_pages=total_pages,
            pagination_html=pagination_html,
            messages_html="".join(messages_html),
        )
        t_write_start = time.perf_counter()
        (output_dir / f"page-{page_num:03d}.html").write_text(
            page_content, encoding="utf-8"
        )
        t_write_total += time.perf_counter() - t_write_start
        print(f"Generated page-{page_num:03d}.html")
        t_pages_total += time.perf_counter() - t_page_start

    # Calculate overall stats and collect all commits for timeline
    t_index_start = time.perf_counter()
    total_tool_counts = {}
    total_messages = 0
    all_commits = []  # (timestamp, hash, message, page_num, conv_index)
    for i, conv in enumerate(conversations):
        total_messages += len(conv["messages"])
        stats = analyze_conversation(conv["messages"])
        for tool, count in stats["tool_counts"].items():
            total_tool_counts[tool] = total_tool_counts.get(tool, 0) + count
        page_num = (i // PROMPTS_PER_PAGE) + 1
        for commit_hash, commit_msg, commit_ts in stats["commits"]:
            all_commits.append((commit_ts, commit_hash, commit_msg, page_num, i))
    total_tool_calls = sum(total_tool_counts.values())
    total_commits = len(all_commits)

    # Build timeline items: prompts and commits merged by timestamp
    timeline_items = []

    # Add prompts
    prompt_num = 0
    for i, conv in enumerate(conversations):
        if conv.get("is_continuation"):
            continue
        if conv["user_text"].startswith("Stop hook feedback:"):
            continue
        prompt_num += 1
        page_num = (i // PROMPTS_PER_PAGE) + 1
        msg_id = make_msg_id(conv["timestamp"])
        link = f"page-{page_num:03d}.html#{msg_id}"
        rendered_content = render_markdown_text(conv["user_text"])

        # Collect all messages including from subsequent continuation conversations
        # This ensures long_texts from continuations appear with the original prompt
        all_messages = list(conv["messages"])
        for j in range(i + 1, len(conversations)):
            if not conversations[j].get("is_continuation"):
                break
            all_messages.extend(conversations[j]["messages"])

        # Analyze conversation for stats (excluding commits from inline display now)
        stats = analyze_conversation(all_messages)
        tool_stats_str = format_tool_stats(stats["tool_counts"])

        long_texts_html = ""
        for lt in stats["long_texts"]:
            rendered_lt = render_markdown_text(lt)
            long_texts_html += _macros.index_long_text(rendered_lt)

        stats_html = _macros.index_stats(tool_stats_str, long_texts_html)

        item_html = _macros.index_item(
            prompt_num, link, conv["timestamp"], rendered_content, stats_html
        )
        timeline_items.append((conv["timestamp"], "prompt", item_html))

    # Add commits as separate timeline items
    for commit_ts, commit_hash, commit_msg, page_num, conv_idx in all_commits:
        item_html = _macros.index_commit(
            commit_hash, commit_msg, commit_ts, _github_repo
        )
        timeline_items.append((commit_ts, "commit", item_html))

    # Sort by timestamp
    timeline_items.sort(key=lambda x: x[0])
    index_items = [item[2] for item in timeline_items]

    index_pagination = generate_index_pagination_html(total_pages)
    index_template = get_template("index.html")
    index_content = index_template.render(
        css=css,
        js=js,
        pagination_html=index_pagination,
        prompt_num=prompt_num,
        total_messages=total_messages,
        total_tool_calls=total_tool_calls,
        total_commits=total_commits,
        total_pages=total_pages,
        index_items_html="".join(index_items),
    )
    index_path = output_dir / "index.html"
    index_path.write_text(index_content, encoding="utf-8")
    print(
        f"Generated {index_path.resolve()} ({total_convs} prompts, {total_pages} pages)"
    )
    t_index = time.perf_counter() - t_index_start

    if profile:
        total = time.perf_counter() - t_total_start
        click.echo("")
        click.echo("Timing breakdown:")
        click.echo(f"- Parse session file: {t_parse:.2f}s")
        click.echo(
            f"- Build conversations: {t_convs:.2f}s ({total_convs} prompts, {total_pages} pages)"
        )
        if translator is not None:
            click.echo(f"- Translation prefetch: {t_prefetch_total:.2f}s")
            click.echo(f"- Translation unique strings (per-page): {translate_unique_total}")
            if hasattr(translator, "get_stats"):
                stats = translator.get_stats()
                click.echo(
                    "- Translation HTTP: "
                    f"{stats.get('http_seconds', 0.0):.2f}s over {stats.get('http_calls', 0)} calls "
                    f"(retries={stats.get('http_retries', 0)}, batches={stats.get('translate_many_batches', 0)})"
                )
        click.echo(f"- Render HTML: {t_render_total:.2f}s")
        click.echo(f"- Write files: {t_write_total:.2f}s")
        click.echo(f"- Build index/stats: {t_index:.2f}s")
        click.echo(f"- Total: {total:.2f}s")

    if hasattr(translator, "close"):
        translator.close()


@click.group(cls=DefaultGroup, default="local", default_if_no_args=True)
@click.version_option(None, "-v", "--version", package_name="claude-code-transcripts")
def cli():
    """Convert Claude Code session JSON to mobile-friendly HTML pages."""
    pass


@cli.command("local")
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output directory. If not specified, writes to temp dir and opens in browser.",
)
@click.option(
    "-a",
    "--output-auto",
    is_flag=True,
    help="Auto-name output subdirectory based on session filename (uses -o as parent, or current dir).",
)
@click.option(
    "--repo",
    help="GitHub repo (owner/name) for commit links. Auto-detected from git push output if not specified.",
)
@click.option(
    "--gist",
    is_flag=True,
    help="Upload to GitHub Gist and output a gisthost.github.io URL.",
)
@click.option(
    "--json",
    "include_json",
    is_flag=True,
    help="Include the original JSONL session file in the output directory.",
)
@click.option(
    "--open",
    "open_browser",
    is_flag=True,
    help="Open the generated index.html in your default browser (default if no -o specified).",
)
@click.option(
    "--profile",
    is_flag=True,
    help="Print a timing breakdown to help diagnose slow runs.",
)
@click.option(
    "--limit",
    default=10,
    help="Maximum number of sessions to show (default: 10)",
)
@click.option(
    "--translate-zh",
    is_flag=True,
    help="Add a Chinese translation column (use --translate-provider to choose backend).",
)
@click.option(
    "--translate-provider",
    type=click.Choice(["openai", "tencent"], case_sensitive=False),
    default="openai",
    help="Translation backend for --translate-zh (default: openai).",
)
@click.option(
    "--translate-tencent-secret-id-env",
    default="TENCENTCLOUD_SECRET_ID",
    help="Env var name containing Tencent SecretId (for --translate-provider tencent).",
)
@click.option(
    "--translate-tencent-secret-key-env",
    default="TENCENTCLOUD_SECRET_KEY",
    help="Env var name containing Tencent SecretKey (for --translate-provider tencent).",
)
@click.option(
    "--translate-tencent-region",
    default="ap-beijing",
    help="Tencent Cloud region for TMT (for --translate-provider tencent).",
)
@click.option(
    "--translate-tencent-endpoint",
    default="tmt.tencentcloudapi.com",
    help="Tencent TMT endpoint hostname (for --translate-provider tencent).",
)
@click.option(
    "--translate-tencent-project-id",
    default=0,
    type=int,
    help="Tencent Cloud project ID (for --translate-provider tencent).",
)
@click.option(
    "--translate-api-key-env",
    default="",
    help="Env var name containing the API key for translation (e.g. ARK_API_KEY).",
)
@click.option(
    "--translate-model",
    default="gpt-4o-mini",
    help="OpenAI model for --translate-provider openai (default: gpt-4o-mini).",
)
@click.option(
    "--translate-wire-api",
    type=click.Choice(["auto", "responses", "chat_completions"], case_sensitive=False),
    default="auto",
    help="OpenAI wire API for --translate-provider openai (default: auto).",
)
@click.option(
    "--translate-base-url",
    help="OpenAI-compatible base URL for --translate-provider openai (default: env OPENAI_BASE_URL/OPENAI_API_BASE or https://api.openai.com/v1).",
)
@click.option(
    "--translate-from-codex-config",
    is_flag=True,
    help="Use provider settings from ~/.codex/config.toml for --translate-provider openai (base_url, wire_api, model).",
)
@click.option(
    "--translate-codex-config-path",
    type=click.Path(),
    default="",
    help="Path to Codex config.toml (default: ~/.codex/config.toml).",
)
@click.option(
    "--translate-codex-provider",
    default="",
    help="Provider name in Codex config (default: model_provider).",
)
def local_cmd(
    output,
    output_auto,
    repo,
    gist,
    include_json,
    open_browser,
    profile,
    limit,
    translate_zh,
    translate_provider,
    translate_tencent_secret_id_env,
    translate_tencent_secret_key_env,
    translate_tencent_region,
    translate_tencent_endpoint,
    translate_tencent_project_id,
    translate_api_key_env,
    translate_model,
    translate_wire_api,
    translate_base_url,
    translate_from_codex_config,
    translate_codex_config_path,
    translate_codex_provider,
):
    """Select and convert a local Claude Code or Codex CLI session to HTML."""
    projects_folder = Path.home() / ".claude" / "projects"
    codex_folder = Path.home() / ".codex" / "sessions"

    # Check if at least one directory exists
    if not projects_folder.exists() and not codex_folder.exists():
        click.echo(f"Neither Claude Code nor Codex CLI sessions found.")
        click.echo(f"  - Claude Code: {projects_folder}")
        click.echo(f"  - Codex CLI: {codex_folder}")
        return

    click.echo("Loading local sessions...")
    t_list_start = time.perf_counter()
    results = find_combined_sessions(limit=limit)
    t_list = time.perf_counter() - t_list_start
    if profile:
        click.echo(f"Profile: listed sessions in {t_list:.2f}s")

    if not results:
        click.echo("No local sessions found.")
        return

    # Build choices for questionary
    choices = []
    for filepath, summary, source in results:
        stat = filepath.stat()
        mod_time = datetime.fromtimestamp(stat.st_mtime)
        size_kb = stat.st_size / 1024
        date_str = mod_time.strftime("%Y-%m-%d %H:%M")
        # Truncate summary if too long
        if len(summary) > 45:
            summary = summary[:42] + "..."
        # Add source label
        display = f"{date_str}  {size_kb:5.0f} KB  [{source:6s}]  {summary}"
        choices.append(questionary.Choice(title=display, value=filepath))

    selected = questionary.select(
        "Select a session to convert:",
        choices=choices,
    ).ask()

    if selected is None:
        click.echo("No session selected.")
        return

    session_file = selected

    # Determine output directory and whether to open browser
    # If no -o specified, use temp dir and open browser by default
    auto_open = output is None and not gist and not output_auto
    if output_auto:
        # Use -o as parent dir (or current dir), with auto-named subdirectory
        parent_dir = Path(output) if output else Path(".")
        output = parent_dir / session_file.stem
    elif output is None:
        output = Path(tempfile.gettempdir()) / f"claude-session-{session_file.stem}"

    output = Path(output)
    translator = None
    if translate_zh:
        provider = (translate_provider or "openai").strip().lower()
        if provider not in ("openai", "tencent"):
            raise click.ClickException(
                "Invalid --translate-provider; expected: openai|tencent"
            )

        if provider == "tencent":
            secret_id = os.environ.get(translate_tencent_secret_id_env or "")
            secret_key = os.environ.get(translate_tencent_secret_key_env or "")
            if not secret_id or not secret_id.strip():
                raise click.ClickException(
                    f"Missing Tencent SecretId in env var: {translate_tencent_secret_id_env}"
                )
            if not secret_key or not secret_key.strip():
                raise click.ClickException(
                    f"Missing Tencent SecretKey in env var: {translate_tencent_secret_key_env}"
                )
            translator = TencentTranslator(
                secret_id=secret_id.strip(),
                secret_key=secret_key.strip(),
                region=translate_tencent_region,
                endpoint=translate_tencent_endpoint,
                project_id=translate_tencent_project_id,
                cache_path=output / "translations.json",
            )
        else:
            translate_api_key = None
            if translate_api_key_env:
                translate_api_key = os.environ.get(translate_api_key_env)
                if not translate_api_key or not translate_api_key.strip():
                    raise click.ClickException(
                        f"Missing translation API key in env var: {translate_api_key_env}"
                    )

            ctx = click.get_current_context(silent=True)
            if translate_from_codex_config:
                codex_config_path = (
                    Path(translate_codex_config_path)
                    if translate_codex_config_path
                    else (Path.home() / ".codex" / "config.toml")
                )
                codex_cfg = _read_codex_provider_config(
                    config_path=codex_config_path,
                    provider=translate_codex_provider or None,
                )
                if (
                    ctx
                    and ctx.get_parameter_source("translate_base_url")
                    == click.core.ParameterSource.DEFAULT
                ):
                    translate_base_url = codex_cfg.get("base_url") or translate_base_url
                if (
                    ctx
                    and ctx.get_parameter_source("translate_wire_api")
                    == click.core.ParameterSource.DEFAULT
                ):
                    translate_wire_api = codex_cfg.get("wire_api") or translate_wire_api
                if (
                    ctx
                    and ctx.get_parameter_source("translate_model")
                    == click.core.ParameterSource.DEFAULT
                ):
                    translate_model = codex_cfg.get("model") or translate_model

            translator = OpenAITranslator.from_env(
                api_key=translate_api_key,
                model=translate_model,
                target_language="Simplified Chinese",
                wire_api=translate_wire_api,
                base_url=translate_base_url,
                cache_path=output / "translations.json",
                prefer_codex_auth=translate_from_codex_config,
            )
    generate_html(
        session_file, output, github_repo=repo, translator=translator, profile=profile
    )

    # Show output directory
    click.echo(f"Output: {output.resolve()}")

    # Copy JSONL file to output directory if requested
    if include_json:
        output.mkdir(exist_ok=True)
        json_dest = output / session_file.name
        shutil.copy(session_file, json_dest)
        json_size_kb = json_dest.stat().st_size / 1024
        click.echo(f"JSONL: {json_dest} ({json_size_kb:.1f} KB)")

    if gist:
        # Inject gist preview JS and create gist
        inject_gist_preview_js(output)
        click.echo("Creating GitHub gist...")
        gist_id, gist_url = create_gist(output)
        preview_url = f"https://gisthost.github.io/?{gist_id}/index.html"
        click.echo(f"Gist: {gist_url}")
        click.echo(f"Preview: {preview_url}")

    if open_browser or auto_open:
        index_url = (output / "index.html").resolve().as_uri()
        webbrowser.open(index_url)


def is_url(path):
    """Check if a path is a URL (starts with http:// or https://)."""
    return path.startswith("http://") or path.startswith("https://")


def fetch_url_to_tempfile(url):
    """Fetch a URL and save to a temporary file.

    Returns the Path to the temporary file.
    Raises click.ClickException on network errors.
    """
    try:
        response = httpx.get(url, timeout=60.0, follow_redirects=True)
        response.raise_for_status()
    except httpx.RequestError as e:
        raise click.ClickException(f"Failed to fetch URL: {e}")
    except httpx.HTTPStatusError as e:
        raise click.ClickException(
            f"Failed to fetch URL: {e.response.status_code} {e.response.reason_phrase}"
        )

    # Determine file extension from URL
    url_path = url.split("?")[0]  # Remove query params
    if url_path.endswith(".jsonl"):
        suffix = ".jsonl"
    elif url_path.endswith(".json"):
        suffix = ".json"
    else:
        suffix = ".jsonl"  # Default to JSONL

    # Extract a name from the URL for the temp file
    url_name = Path(url_path).stem or "session"

    temp_dir = Path(tempfile.gettempdir())
    temp_file = temp_dir / f"claude-url-{url_name}{suffix}"
    temp_file.write_text(response.text, encoding="utf-8")
    return temp_file


@cli.command("json")
@click.argument("json_file", type=click.Path())
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output directory. If not specified, writes to temp dir and opens in browser.",
)
@click.option(
    "-a",
    "--output-auto",
    is_flag=True,
    help="Auto-name output subdirectory based on filename (uses -o as parent, or current dir).",
)
@click.option(
    "--repo",
    help="GitHub repo (owner/name) for commit links. Auto-detected from git push output if not specified.",
)
@click.option(
    "--gist",
    is_flag=True,
    help="Upload to GitHub Gist and output a gisthost.github.io URL.",
)
@click.option(
    "--json",
    "include_json",
    is_flag=True,
    help="Include the original JSON session file in the output directory.",
)
@click.option(
    "--open",
    "open_browser",
    is_flag=True,
    help="Open the generated index.html in your default browser (default if no -o specified).",
)
@click.option(
    "--profile",
    is_flag=True,
    help="Print a timing breakdown to help diagnose slow runs.",
)
@click.option(
    "--translate-zh",
    is_flag=True,
    help="Add a Chinese translation column (use --translate-provider to choose backend).",
)
@click.option(
    "--translate-provider",
    type=click.Choice(["openai", "tencent"], case_sensitive=False),
    default="openai",
    help="Translation backend for --translate-zh (default: openai).",
)
@click.option(
    "--translate-tencent-secret-id-env",
    default="TENCENTCLOUD_SECRET_ID",
    help="Env var name containing Tencent SecretId (for --translate-provider tencent).",
)
@click.option(
    "--translate-tencent-secret-key-env",
    default="TENCENTCLOUD_SECRET_KEY",
    help="Env var name containing Tencent SecretKey (for --translate-provider tencent).",
)
@click.option(
    "--translate-tencent-region",
    default="ap-beijing",
    help="Tencent Cloud region for TMT (for --translate-provider tencent).",
)
@click.option(
    "--translate-tencent-endpoint",
    default="tmt.tencentcloudapi.com",
    help="Tencent TMT endpoint hostname (for --translate-provider tencent).",
)
@click.option(
    "--translate-tencent-project-id",
    default=0,
    type=int,
    help="Tencent Cloud project ID (for --translate-provider tencent).",
)
@click.option(
    "--translate-api-key-env",
    default="",
    help="Env var name containing the API key for translation (e.g. ARK_API_KEY).",
)
@click.option(
    "--translate-model",
    default="gpt-4o-mini",
    help="OpenAI model for --translate-provider openai (default: gpt-4o-mini).",
)
@click.option(
    "--translate-wire-api",
    type=click.Choice(["auto", "responses", "chat_completions"], case_sensitive=False),
    default="auto",
    help="OpenAI wire API for --translate-provider openai (default: auto).",
)
@click.option(
    "--translate-base-url",
    help="OpenAI-compatible base URL for --translate-provider openai (default: env OPENAI_BASE_URL/OPENAI_API_BASE or https://api.openai.com/v1).",
)
@click.option(
    "--translate-from-codex-config",
    is_flag=True,
    help="Use provider settings from ~/.codex/config.toml for --translate-provider openai (base_url, wire_api, model).",
)
@click.option(
    "--translate-codex-config-path",
    type=click.Path(),
    default="",
    help="Path to Codex config.toml (default: ~/.codex/config.toml).",
)
@click.option(
    "--translate-codex-provider",
    default="",
    help="Provider name in Codex config (default: model_provider).",
)
def json_cmd(
    json_file,
    output,
    output_auto,
    repo,
    gist,
    include_json,
    open_browser,
    profile,
    translate_zh,
    translate_provider,
    translate_tencent_secret_id_env,
    translate_tencent_secret_key_env,
    translate_tencent_region,
    translate_tencent_endpoint,
    translate_tencent_project_id,
    translate_api_key_env,
    translate_model,
    translate_wire_api,
    translate_base_url,
    translate_from_codex_config,
    translate_codex_config_path,
    translate_codex_provider,
):
    """Convert a Claude Code session JSON/JSONL file or URL to HTML."""
    # Handle URL input
    if is_url(json_file):
        click.echo(f"Fetching {json_file}...")
        temp_file = fetch_url_to_tempfile(json_file)
        json_file_path = temp_file
        # Use URL path for naming
        url_name = Path(json_file.split("?")[0]).stem or "session"
    else:
        # Validate that local file exists
        json_file_path = Path(json_file)
        if not json_file_path.exists():
            raise click.ClickException(f"File not found: {json_file}")
        url_name = None

    # Determine output directory and whether to open browser
    # If no -o specified, use temp dir and open browser by default
    auto_open = output is None and not gist and not output_auto
    if output_auto:
        # Use -o as parent dir (or current dir), with auto-named subdirectory
        parent_dir = Path(output) if output else Path(".")
        output = parent_dir / (url_name or json_file_path.stem)
    elif output is None:
        output = (
            Path(tempfile.gettempdir())
            / f"claude-session-{url_name or json_file_path.stem}"
        )

    output = Path(output)
    translator = None
    if translate_zh:
        provider = (translate_provider or "openai").strip().lower()
        if provider not in ("openai", "tencent"):
            raise click.ClickException(
                "Invalid --translate-provider; expected: openai|tencent"
            )

        if provider == "tencent":
            secret_id = os.environ.get(translate_tencent_secret_id_env or "")
            secret_key = os.environ.get(translate_tencent_secret_key_env or "")
            if not secret_id or not secret_id.strip():
                raise click.ClickException(
                    f"Missing Tencent SecretId in env var: {translate_tencent_secret_id_env}"
                )
            if not secret_key or not secret_key.strip():
                raise click.ClickException(
                    f"Missing Tencent SecretKey in env var: {translate_tencent_secret_key_env}"
                )
            translator = TencentTranslator(
                secret_id=secret_id.strip(),
                secret_key=secret_key.strip(),
                region=translate_tencent_region,
                endpoint=translate_tencent_endpoint,
                project_id=translate_tencent_project_id,
                cache_path=output / "translations.json",
            )
        else:
            translate_api_key = None
            if translate_api_key_env:
                translate_api_key = os.environ.get(translate_api_key_env)
                if not translate_api_key or not translate_api_key.strip():
                    raise click.ClickException(
                        f"Missing translation API key in env var: {translate_api_key_env}"
                    )

            ctx = click.get_current_context(silent=True)
            if translate_from_codex_config:
                codex_config_path = (
                    Path(translate_codex_config_path)
                    if translate_codex_config_path
                    else (Path.home() / ".codex" / "config.toml")
                )
                codex_cfg = _read_codex_provider_config(
                    config_path=codex_config_path,
                    provider=translate_codex_provider or None,
                )
                if (
                    ctx
                    and ctx.get_parameter_source("translate_base_url")
                    == click.core.ParameterSource.DEFAULT
                ):
                    translate_base_url = codex_cfg.get("base_url") or translate_base_url
                if (
                    ctx
                    and ctx.get_parameter_source("translate_wire_api")
                    == click.core.ParameterSource.DEFAULT
                ):
                    translate_wire_api = codex_cfg.get("wire_api") or translate_wire_api
                if (
                    ctx
                    and ctx.get_parameter_source("translate_model")
                    == click.core.ParameterSource.DEFAULT
                ):
                    translate_model = codex_cfg.get("model") or translate_model

            translator = OpenAITranslator.from_env(
                api_key=translate_api_key,
                model=translate_model,
                target_language="Simplified Chinese",
                wire_api=translate_wire_api,
                base_url=translate_base_url,
                cache_path=output / "translations.json",
                prefer_codex_auth=translate_from_codex_config,
            )
    generate_html(
        json_file_path,
        output,
        github_repo=repo,
        translator=translator,
        profile=profile,
    )

    # Show output directory
    click.echo(f"Output: {output.resolve()}")

    # Copy JSON file to output directory if requested
    if include_json:
        output.mkdir(exist_ok=True)
        json_dest = output / json_file_path.name
        shutil.copy(json_file_path, json_dest)
        json_size_kb = json_dest.stat().st_size / 1024
        click.echo(f"JSON: {json_dest} ({json_size_kb:.1f} KB)")

    if gist:
        # Inject gist preview JS and create gist
        inject_gist_preview_js(output)
        click.echo("Creating GitHub gist...")
        gist_id, gist_url = create_gist(output)
        preview_url = f"https://gisthost.github.io/?{gist_id}/index.html"
        click.echo(f"Gist: {gist_url}")
        click.echo(f"Preview: {preview_url}")

    if open_browser or auto_open:
        index_url = (output / "index.html").resolve().as_uri()
        webbrowser.open(index_url)


def resolve_credentials(token, org_uuid):
    """Resolve token and org_uuid from arguments or auto-detect.

    Returns (token, org_uuid) tuple.
    Raises click.ClickException if credentials cannot be resolved.
    """
    # Get token
    if token is None:
        token = get_access_token_from_keychain()
        if token is None:
            if platform.system() == "Darwin":
                raise click.ClickException(
                    "Could not retrieve access token from macOS keychain. "
                    "Make sure you are logged into Claude Code, or provide --token."
                )
            else:
                raise click.ClickException(
                    "On non-macOS platforms, you must provide --token with your access token."
                )

    # Get org UUID
    if org_uuid is None:
        org_uuid = get_org_uuid_from_config()
        if org_uuid is None:
            raise click.ClickException(
                "Could not find organization UUID in ~/.claude.json. "
                "Provide --org-uuid with your organization UUID."
            )

    return token, org_uuid


def format_session_for_display(session_data):
    """Format a session for display in the list or picker.

    Returns a formatted string.
    """
    session_id = session_data.get("id", "unknown")
    title = session_data.get("title", "Untitled")
    created_at = session_data.get("created_at", "")
    # Truncate title if too long
    if len(title) > 60:
        title = title[:57] + "..."
    return f"{session_id}  {created_at[:19] if created_at else 'N/A':19}  {title}"


def generate_html_from_session_data(
    session_data, output_dir, github_repo=None, translator=None, profile=False
):
    """Generate HTML from session data dict (instead of file path)."""
    t_total_start = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    css = CSS + BILINGUAL_CSS if translator is not None else CSS
    js = JS + BILINGUAL_JS if translator is not None else JS

    loglines = session_data.get("loglines", [])

    # Auto-detect GitHub repo if not provided
    if github_repo is None:
        github_repo = detect_github_repo(loglines)
        if github_repo:
            click.echo(f"Auto-detected GitHub repo: {github_repo}")

    # Set module-level variable for render functions
    global _github_repo
    _github_repo = github_repo

    t_convs_start = time.perf_counter()
    conversations = []
    current_conv = None
    for entry in loglines:
        log_type = entry.get("type")
        timestamp = entry.get("timestamp", "")
        is_compact_summary = entry.get("isCompactSummary", False)
        message_data = entry.get("message", {})
        if not message_data:
            continue
        # Convert message dict to JSON string for compatibility with existing render functions
        message_json = json.dumps(message_data)
        is_user_prompt = False
        user_text = None
        if log_type == "user":
            content = message_data.get("content", "")
            text = extract_text_from_content(content)
            if text:
                is_user_prompt = True
                user_text = text
        if is_user_prompt:
            if current_conv:
                conversations.append(current_conv)
            current_conv = {
                "user_text": user_text,
                "timestamp": timestamp,
                "messages": [(log_type, message_json, timestamp)],
                "is_continuation": bool(is_compact_summary),
            }
        elif current_conv:
            current_conv["messages"].append((log_type, message_json, timestamp))
    if current_conv:
        conversations.append(current_conv)
    t_convs = time.perf_counter() - t_convs_start

    total_convs = len(conversations)
    total_pages = (total_convs + PROMPTS_PER_PAGE - 1) // PROMPTS_PER_PAGE

    t_prefetch_total = 0.0
    t_render_total = 0.0
    t_write_total = 0.0
    translate_unique_total = 0

    for page_num in range(1, total_pages + 1):
        start_idx = (page_num - 1) * PROMPTS_PER_PAGE
        end_idx = min(start_idx + PROMPTS_PER_PAGE, total_convs)
        page_convs = conversations[start_idx:end_idx]

        if translator is not None and hasattr(translator, "translate_many"):
            texts = []
            for conv in page_convs:
                for log_type, message_json, _timestamp in conv["messages"]:
                    try:
                        message_data = json.loads(message_json)
                    except json.JSONDecodeError:
                        continue
                    texts.extend(
                        _collect_translatable_texts_for_message(log_type, message_data)
                    )
            if texts:
                unique_texts = list(dict.fromkeys(texts))
                translate_unique_total += len(unique_texts)
                if len(unique_texts) > 1:
                    t_prefetch_start = time.perf_counter()
                    translator.translate_many(unique_texts)
                    t_prefetch_total += time.perf_counter() - t_prefetch_start

        messages_html = []
        page_turn_summaries = []
        t_render_start = time.perf_counter()
        for conv_offset, conv in enumerate(page_convs):
            user_translation_parts = []
            assistant_translation_parts = []
            is_first = True
            for log_type, message_json, timestamp in conv["messages"]:
                try:
                    message_data = json.loads(message_json)
                except json.JSONDecodeError:
                    continue
                msg_html, translation_text = render_message_data(
                    log_type, message_data, timestamp, translator=translator
                )
                if msg_html:
                    # Wrap continuation summaries in collapsed details
                    if is_first and conv.get("is_continuation"):
                        msg_html = f'<details class="continuation"><summary>Session continuation summary</summary>{msg_html}</details>'
                    messages_html.append(msg_html)
                if translation_text:
                    if log_type == "user":
                        user_translation_parts.append(translation_text)
                    elif log_type == "assistant":
                        assistant_translation_parts.append(translation_text)
                is_first = False

            if translator is not None and (
                user_translation_parts or assistant_translation_parts
            ):
                summary_parts = []
                if user_translation_parts:
                    summary_parts.append(
                        "**用户：**\n\n" + "\n\n".join(user_translation_parts)
                    )
                if assistant_translation_parts:
                    summary_parts.append(
                        "**助手：**\n\n" + "\n\n".join(assistant_translation_parts)
                    )
                summary_markdown = "\n\n".join(summary_parts).strip()
                if summary_markdown:
                    page_turn_summaries.append(
                        {
                            "anchor": make_msg_id(conv["timestamp"]),
                            "timestamp": conv["timestamp"],
                            "turn_number": start_idx + conv_offset + 1,
                            "summary_html": render_markdown_text(summary_markdown),
                        }
                    )

        if translator is not None:
            settings_html = (
                '<div class="sidebar-section">'
                '<div class="sidebar-title">显示设置</div>'
                '<label><input type="checkbox" id="toggle-bash-tools" checked>显示命令行</label>'
                '<label><input type="checkbox" id="toggle-tool-replies" checked>显示 Tool reply</label>'
                "</div>"
            )

            summaries = ['<div class="sidebar-section"><div class="sidebar-title">本页中文汇总</div>']
            for item in page_turn_summaries:
                summaries.append(
                    (
                        '<div class="turn-summary">'
                        '<div class="turn-summary-header">'
                        f'<a class="turn-summary-link" href="#{item["anchor"]}">#{item["turn_number"]}</a>'
                        f'<time datetime="{item["timestamp"]}" data-timestamp="{item["timestamp"]}">{item["timestamp"]}</time>'
                        "</div>"
                        '<div class="turn-summary-body">'
                        '<div class="truncatable"><div class="truncatable-content">'
                        f'{item["summary_html"]}'
                        '</div><button class="expand-btn">Show more</button></div>'
                        "</div>"
                        "</div>"
                    )
                )
            summaries.append("</div>")
            sidebar_html = f'<aside class="sidebar">{settings_html}{"".join(summaries)}</aside>'
            messages_html = [
                f'<div class="page-grid">{sidebar_html}<div class="page-messages">{"".join(messages_html)}</div></div>'
            ]
        t_render_total += time.perf_counter() - t_render_start
        pagination_html = generate_pagination_html(page_num, total_pages)
        page_template = get_template("page.html")
        page_content = page_template.render(
            css=css,
            js=js,
            page_num=page_num,
            total_pages=total_pages,
            pagination_html=pagination_html,
            messages_html="".join(messages_html),
        )
        t_write_start = time.perf_counter()
        (output_dir / f"page-{page_num:03d}.html").write_text(
            page_content, encoding="utf-8"
        )
        t_write_total += time.perf_counter() - t_write_start
        click.echo(f"Generated page-{page_num:03d}.html")

    # Calculate overall stats and collect all commits for timeline
    t_index_start = time.perf_counter()
    total_tool_counts = {}
    total_messages = 0
    all_commits = []  # (timestamp, hash, message, page_num, conv_index)
    for i, conv in enumerate(conversations):
        total_messages += len(conv["messages"])
        stats = analyze_conversation(conv["messages"])
        for tool, count in stats["tool_counts"].items():
            total_tool_counts[tool] = total_tool_counts.get(tool, 0) + count
        page_num = (i // PROMPTS_PER_PAGE) + 1
        for commit_hash, commit_msg, commit_ts in stats["commits"]:
            all_commits.append((commit_ts, commit_hash, commit_msg, page_num, i))
    total_tool_calls = sum(total_tool_counts.values())
    total_commits = len(all_commits)

    # Build timeline items: prompts and commits merged by timestamp
    timeline_items = []

    # Add prompts
    prompt_num = 0
    for i, conv in enumerate(conversations):
        if conv.get("is_continuation"):
            continue
        if conv["user_text"].startswith("Stop hook feedback:"):
            continue
        prompt_num += 1
        page_num = (i // PROMPTS_PER_PAGE) + 1
        msg_id = make_msg_id(conv["timestamp"])
        link = f"page-{page_num:03d}.html#{msg_id}"
        rendered_content = render_markdown_text(conv["user_text"])

        # Collect all messages including from subsequent continuation conversations
        # This ensures long_texts from continuations appear with the original prompt
        all_messages = list(conv["messages"])
        for j in range(i + 1, len(conversations)):
            if not conversations[j].get("is_continuation"):
                break
            all_messages.extend(conversations[j]["messages"])

        # Analyze conversation for stats (excluding commits from inline display now)
        stats = analyze_conversation(all_messages)
        tool_stats_str = format_tool_stats(stats["tool_counts"])

        long_texts_html = ""
        for lt in stats["long_texts"]:
            rendered_lt = render_markdown_text(lt)
            long_texts_html += _macros.index_long_text(rendered_lt)

        stats_html = _macros.index_stats(tool_stats_str, long_texts_html)

        item_html = _macros.index_item(
            prompt_num, link, conv["timestamp"], rendered_content, stats_html
        )
        timeline_items.append((conv["timestamp"], "prompt", item_html))

    # Add commits as separate timeline items
    for commit_ts, commit_hash, commit_msg, page_num, conv_idx in all_commits:
        item_html = _macros.index_commit(
            commit_hash, commit_msg, commit_ts, _github_repo
        )
        timeline_items.append((commit_ts, "commit", item_html))

    # Sort by timestamp
    timeline_items.sort(key=lambda x: x[0])
    index_items = [item[2] for item in timeline_items]

    index_pagination = generate_index_pagination_html(total_pages)
    index_template = get_template("index.html")
    index_content = index_template.render(
        css=css,
        js=js,
        pagination_html=index_pagination,
        prompt_num=prompt_num,
        total_messages=total_messages,
        total_tool_calls=total_tool_calls,
        total_commits=total_commits,
        total_pages=total_pages,
        index_items_html="".join(index_items),
    )
    index_path = output_dir / "index.html"
    index_path.write_text(index_content, encoding="utf-8")
    click.echo(
        f"Generated {index_path.resolve()} ({total_convs} prompts, {total_pages} pages)"
    )
    t_index = time.perf_counter() - t_index_start

    if profile:
        total = time.perf_counter() - t_total_start
        click.echo("")
        click.echo("Timing breakdown:")
        click.echo(
            f"- Build conversations: {t_convs:.2f}s ({total_convs} prompts, {total_pages} pages)"
        )
        if translator is not None:
            click.echo(f"- Translation prefetch: {t_prefetch_total:.2f}s")
            click.echo(f"- Translation unique strings (per-page): {translate_unique_total}")
            if hasattr(translator, "get_stats"):
                stats = translator.get_stats()
                click.echo(
                    "- Translation HTTP: "
                    f"{stats.get('http_seconds', 0.0):.2f}s over {stats.get('http_calls', 0)} calls "
                    f"(retries={stats.get('http_retries', 0)}, batches={stats.get('translate_many_batches', 0)})"
                )
        click.echo(f"- Render HTML: {t_render_total:.2f}s")
        click.echo(f"- Write files: {t_write_total:.2f}s")
        click.echo(f"- Build index/stats: {t_index:.2f}s")
        click.echo(f"- Total: {total:.2f}s")

    if hasattr(translator, "close"):
        translator.close()


@cli.command("web")
@click.argument("session_id", required=False)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output directory. If not specified, writes to temp dir and opens in browser.",
)
@click.option(
    "-a",
    "--output-auto",
    is_flag=True,
    help="Auto-name output subdirectory based on session ID (uses -o as parent, or current dir).",
)
@click.option("--token", help="API access token (auto-detected from keychain on macOS)")
@click.option(
    "--org-uuid", help="Organization UUID (auto-detected from ~/.claude.json)"
)
@click.option(
    "--repo",
    help="GitHub repo (owner/name) for commit links. Auto-detected from git push output if not specified.",
)
@click.option(
    "--gist",
    is_flag=True,
    help="Upload to GitHub Gist and output a gisthost.github.io URL.",
)
@click.option(
    "--json",
    "include_json",
    is_flag=True,
    help="Include the JSON session data in the output directory.",
)
@click.option(
    "--open",
    "open_browser",
    is_flag=True,
    help="Open the generated index.html in your default browser (default if no -o specified).",
)
def web_cmd(
    session_id,
    output,
    output_auto,
    token,
    org_uuid,
    repo,
    gist,
    include_json,
    open_browser,
):
    """Select and convert a web session from the Claude API to HTML.

    If SESSION_ID is not provided, displays an interactive picker to select a session.
    """
    try:
        token, org_uuid = resolve_credentials(token, org_uuid)
    except click.ClickException:
        raise

    # If no session ID provided, show interactive picker
    if session_id is None:
        try:
            sessions_data = fetch_sessions(token, org_uuid)
        except httpx.HTTPStatusError as e:
            raise click.ClickException(
                f"API request failed: {e.response.status_code} {e.response.text}"
            )
        except httpx.RequestError as e:
            raise click.ClickException(f"Network error: {e}")

        sessions = sessions_data.get("data", [])
        if not sessions:
            raise click.ClickException("No sessions found.")

        # Build choices for questionary
        choices = []
        for s in sessions:
            sid = s.get("id", "unknown")
            title = s.get("title", "Untitled")
            created_at = s.get("created_at", "")
            # Truncate title if too long
            if len(title) > 50:
                title = title[:47] + "..."
            display = f"{created_at[:19] if created_at else 'N/A':19}  {title}"
            choices.append(questionary.Choice(title=display, value=sid))

        selected = questionary.select(
            "Select a session to import:",
            choices=choices,
        ).ask()

        if selected is None:
            # User cancelled
            raise click.ClickException("No session selected.")

        session_id = selected

    # Fetch the session
    click.echo(f"Fetching session {session_id}...")
    try:
        session_data = fetch_session(token, org_uuid, session_id)
    except httpx.HTTPStatusError as e:
        raise click.ClickException(
            f"API request failed: {e.response.status_code} {e.response.text}"
        )
    except httpx.RequestError as e:
        raise click.ClickException(f"Network error: {e}")

    # Determine output directory and whether to open browser
    # If no -o specified, use temp dir and open browser by default
    auto_open = output is None and not gist and not output_auto
    if output_auto:
        # Use -o as parent dir (or current dir), with auto-named subdirectory
        parent_dir = Path(output) if output else Path(".")
        output = parent_dir / session_id
    elif output is None:
        output = Path(tempfile.gettempdir()) / f"claude-session-{session_id}"

    output = Path(output)
    click.echo(f"Generating HTML in {output}/...")
    generate_html_from_session_data(session_data, output, github_repo=repo)

    # Show output directory
    click.echo(f"Output: {output.resolve()}")

    # Save JSON session data if requested
    if include_json:
        output.mkdir(exist_ok=True)
        json_dest = output / f"{session_id}.json"
        with open(json_dest, "w") as f:
            json.dump(session_data, f, indent=2)
        json_size_kb = json_dest.stat().st_size / 1024
        click.echo(f"JSON: {json_dest} ({json_size_kb:.1f} KB)")

    if gist:
        # Inject gist preview JS and create gist
        inject_gist_preview_js(output)
        click.echo("Creating GitHub gist...")
        gist_id, gist_url = create_gist(output)
        preview_url = f"https://gisthost.github.io/?{gist_id}/index.html"
        click.echo(f"Gist: {gist_url}")
        click.echo(f"Preview: {preview_url}")

    if open_browser or auto_open:
        index_url = (output / "index.html").resolve().as_uri()
        webbrowser.open(index_url)


@cli.command("all")
@click.option(
    "-s",
    "--source",
    type=click.Path(exists=True),
    help="Source directory containing Claude projects (default: ~/.claude/projects).",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default="./claude-archive",
    help="Output directory for the archive (default: ./claude-archive).",
)
@click.option(
    "--include-agents",
    is_flag=True,
    help="Include agent-* session files (excluded by default).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be converted without creating files.",
)
@click.option(
    "--open",
    "open_browser",
    is_flag=True,
    help="Open the generated archive in your default browser.",
)
@click.option(
    "-q",
    "--quiet",
    is_flag=True,
    help="Suppress all output except errors.",
)
def all_cmd(source, output, include_agents, dry_run, open_browser, quiet):
    """Convert all local Claude Code sessions to a browsable HTML archive.

    Creates a directory structure with:
    - Master index listing all projects
    - Per-project pages listing sessions
    - Individual session transcripts
    """
    # Default source folder
    if source is None:
        source = Path.home() / ".claude" / "projects"
    else:
        source = Path(source)

    if not source.exists():
        raise click.ClickException(f"Source directory not found: {source}")

    output = Path(output)

    if not quiet:
        click.echo(f"Scanning {source}...")

    projects = find_all_sessions(source, include_agents=include_agents)

    if not projects:
        if not quiet:
            click.echo("No sessions found.")
        return

    # Calculate totals
    total_sessions = sum(len(p["sessions"]) for p in projects)

    if not quiet:
        click.echo(f"Found {len(projects)} projects with {total_sessions} sessions")

    if dry_run:
        # Dry-run always outputs (it's the point of dry-run), but respects --quiet
        if not quiet:
            click.echo("\nDry run - would convert:")
            for project in projects:
                click.echo(
                    f"\n  {project['name']} ({len(project['sessions'])} sessions)"
                )
                for session in project["sessions"][:3]:  # Show first 3
                    mod_time = datetime.fromtimestamp(session["mtime"])
                    click.echo(
                        f"    - {session['path'].stem} ({mod_time.strftime('%Y-%m-%d')})"
                    )
                if len(project["sessions"]) > 3:
                    click.echo(f"    ... and {len(project['sessions']) - 3} more")
        return

    if not quiet:
        click.echo(f"\nGenerating archive in {output}...")

    # Progress callback for non-quiet mode
    def on_progress(project_name, session_name, current, total):
        if not quiet and current % 10 == 0:
            click.echo(f"  Processed {current}/{total} sessions...")

    # Generate the archive using the library function
    stats = generate_batch_html(
        source,
        output,
        include_agents=include_agents,
        progress_callback=on_progress,
    )

    # Report any failures
    if stats["failed_sessions"]:
        click.echo(f"\nWarning: {len(stats['failed_sessions'])} session(s) failed:")
        for failure in stats["failed_sessions"]:
            click.echo(
                f"  {failure['project']}/{failure['session']}: {failure['error']}"
            )

    if not quiet:
        click.echo(
            f"\nGenerated archive with {stats['total_projects']} projects, "
            f"{stats['total_sessions']} sessions"
        )
        click.echo(f"Output: {output.resolve()}")

    if open_browser:
        index_url = (output / "index.html").resolve().as_uri()
        webbrowser.open(index_url)


def main():
    cli()
