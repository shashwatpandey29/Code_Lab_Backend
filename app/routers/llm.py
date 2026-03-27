import os
import time
import asyncio
import json
from pathlib import Path
from collections.abc import AsyncIterator
from typing import Any

import httpx
from dotenv import dotenv_values, load_dotenv
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import StreamingResponse

router = APIRouter()

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_TIMEOUT_SEC = float(os.getenv("OPENROUTER_TIMEOUT_SEC", "45"))
OPENROUTER_MAX_RETRIES = int(os.getenv("OPENROUTER_MAX_RETRIES", "2"))
OPENROUTER_BACKOFF_SEC = float(os.getenv("OPENROUTER_BACKOFF_SEC", "1.0"))
OPENROUTER_MODEL_FALLBACKS = os.getenv(
    "OPENROUTER_MODEL_FALLBACKS",
    "openrouter/auto,meta-llama/llama-3.1-8b-instruct:free,mistralai/mistral-7b-instruct:free",
)
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost:3000")
OPENROUTER_X_TITLE = os.getenv("OPENROUTER_X_TITLE", "CodeLab")
LLM_RATE_LIMIT_WINDOW_SEC = int(os.getenv("LLM_RATE_LIMIT_WINDOW_SEC", "60"))
LLM_RATE_LIMIT_MAX_REQUESTS = int(os.getenv("LLM_RATE_LIMIT_MAX_REQUESTS", "12"))

API_DIR = Path(__file__).resolve().parents[2]
REPO_DIR = Path(__file__).resolve().parents[4]

_rate_limiter: dict[str, tuple[int, float]] = {}


def _load_env_files() -> None:
    load_dotenv(dotenv_path=REPO_DIR / ".env", override=False)
    load_dotenv(dotenv_path=API_DIR / ".env", override=False)


def _get_openrouter_api_key() -> str:
    _load_env_files()
    for key in ("OPENROUTER_API_KEY", "XAI_API_KEY", "GROK_API_KEY"):
        value = os.getenv(key, "").strip()
        if value:
            return value

    for env_path in (REPO_DIR / ".env", API_DIR / ".env"):
        if not env_path.exists():
            continue
        values = dotenv_values(env_path)
        for key in ("OPENROUTER_API_KEY", "XAI_API_KEY", "GROK_API_KEY"):
            raw_value = values.get(key)
            if isinstance(raw_value, str) and raw_value.strip():
                return raw_value.strip()

    return ""


def _get_model_fallbacks() -> list[str]:
    _load_env_files()
    raw = os.getenv("OPENROUTER_MODEL_FALLBACKS", OPENROUTER_MODEL_FALLBACKS)
    models = [item.strip() for item in raw.split(",") if item.strip()]
    if not models:
        return ["openrouter/auto"]

    deduped: list[str] = []
    seen: set[str] = set()
    for model in models:
        if model not in seen:
            deduped.append(model)
            seen.add(model)
    return deduped


def _get_timeout_sec() -> float:
    _load_env_files()
    return float(os.getenv("OPENROUTER_TIMEOUT_SEC", str(OPENROUTER_TIMEOUT_SEC)))


def _get_max_retries() -> int:
    _load_env_files()
    return int(os.getenv("OPENROUTER_MAX_RETRIES", str(OPENROUTER_MAX_RETRIES)))


def _get_backoff_sec() -> float:
    _load_env_files()
    return float(os.getenv("OPENROUTER_BACKOFF_SEC", str(OPENROUTER_BACKOFF_SEC)))


def _build_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", OPENROUTER_HTTP_REFERER),
        "X-Title": os.getenv("OPENROUTER_X_TITLE", OPENROUTER_X_TITLE),
    }


def _client_id(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.headers.get("x-real-ip") or (request.client.host if request.client else "unknown")


def _check_rate_limit(client_id: str) -> None:
    now = time.time()
    count, reset_at = _rate_limiter.get(client_id, (0, now + LLM_RATE_LIMIT_WINDOW_SEC))

    if now >= reset_at:
        count, reset_at = 0, now + LLM_RATE_LIMIT_WINDOW_SEC

    if count >= LLM_RATE_LIMIT_MAX_REQUESTS:
        retry_after = int(max(1, reset_at - now))
        raise HTTPException(
            status_code=429,
            detail={
                "error": "llm_rate_limited",
                "message": "Too many tutor requests.",
                "retry_after": retry_after,
            },
            headers={"Retry-After": str(retry_after)},
        )

    _rate_limiter[client_id] = (count + 1, reset_at)


async def ollama_ready() -> bool:
    api_key = _get_openrouter_api_key()
    if not api_key:
        return False

    headers = _build_headers(api_key)

    try:
        async with httpx.AsyncClient(timeout=min(10, _get_timeout_sec())) as client:
            response = await client.get(f"{OPENROUTER_BASE_URL.rstrip('/')}/models", headers=headers)
        return response.status_code < 400
    except httpx.HTTPError:
        return False


async def provider_ready() -> bool:
    return await ollama_ready()


class TutorRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=24_000)
    system: str = Field(default="You are a practical coding tutor. Be concise and actionable.", max_length=4_000)
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    max_tokens: int = Field(default=384, ge=64, le=4096)


class TutorResponse(BaseModel):
    provider: str
    model: str
    content: str
    usage: dict[str, Any] | None = None


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
        return "".join(text_parts)
    return ""


def _extract_chat_completion_content(response_json: dict[str, Any]) -> str:
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""

    first = choices[0]
    if not isinstance(first, dict):
        return ""

    message = first.get("message")
    if not isinstance(message, dict):
        return ""

    return _extract_text_content(message.get("content"))


def _build_payload(model: str, payload: TutorRequest, stream: bool) -> dict[str, Any]:
    return {
        "model": model,
        "stream": stream,
        "temperature": payload.temperature,
        "max_tokens": payload.max_tokens,
        "messages": [
            {"role": "system", "content": payload.system},
            {"role": "user", "content": payload.prompt},
        ],
    }


def _map_http_error(status_code: int, payload: dict[str, Any] | None = None) -> HTTPException:
    message = "OpenRouter request failed."
    if isinstance(payload, dict):
        error_obj = payload.get("error")
        if isinstance(error_obj, dict) and isinstance(error_obj.get("message"), str):
            message = error_obj["message"]
        elif isinstance(payload.get("message"), str):
            message = str(payload.get("message"))

    if status_code in (401, 403):
        return HTTPException(status_code=401, detail={"error": "llm_unauthorized", "message": message})
    if status_code == 429:
        return HTTPException(status_code=429, detail={"error": "llm_rate_limited", "message": message})
    if status_code in (408, 504):
        return HTTPException(status_code=504, detail={"error": "llm_timeout", "message": message})
    return HTTPException(status_code=502, detail={"error": "llm_unreachable", "message": message})


async def _chat_completion_with_fallback(payload: TutorRequest) -> tuple[dict[str, Any], str]:
    api_key = _get_openrouter_api_key()
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail={"error": "llm_unauthorized", "message": "Missing OPENROUTER_API_KEY."},
        )

    headers = _build_headers(api_key)
    last_error_payload: dict[str, Any] | None = None
    saw_rate_limit = False

    async with httpx.AsyncClient(timeout=_get_timeout_sec()) as client:
        for attempt in range(_get_max_retries() + 1):
            for model in _get_model_fallbacks():
                request_body = _build_payload(model, payload, stream=False)
                try:
                    response = await client.post(
                        f"{OPENROUTER_BASE_URL.rstrip('/')}/chat/completions",
                        json=request_body,
                        headers=headers,
                    )
                except httpx.TimeoutException as exc:
                    if attempt < _get_max_retries():
                        await asyncio.sleep(_get_backoff_sec() * (attempt + 1))
                        continue
                    raise HTTPException(
                        status_code=504,
                        detail={"error": "llm_timeout", "message": "OpenRouter timed out."},
                    ) from exc
                except httpx.HTTPError as exc:
                    if attempt < _get_max_retries():
                        await asyncio.sleep(_get_backoff_sec() * (attempt + 1))
                        continue
                    raise HTTPException(
                        status_code=502,
                        detail={"error": "llm_unreachable", "message": "OpenRouter is unreachable."},
                    ) from exc

                data: dict[str, Any] | None = None
                try:
                    data = response.json()
                except Exception:
                    data = None

                if response.status_code == 429:
                    saw_rate_limit = True
                    last_error_payload = data
                    continue

                if response.status_code in (401, 403):
                    raise _map_http_error(response.status_code, data)

                if response.status_code >= 400:
                    last_error_payload = data
                    if response.status_code >= 500:
                        continue
                    raise _map_http_error(response.status_code, data)

                if not isinstance(data, dict):
                    continue
                return data, model

            if attempt < _get_max_retries():
                await asyncio.sleep(_get_backoff_sec() * (attempt + 1))

    if saw_rate_limit:
        raise _map_http_error(429, last_error_payload)
    raise _map_http_error(502, last_error_payload)


async def _stream_openrouter(payload: TutorRequest) -> AsyncIterator[str]:
    api_key = _get_openrouter_api_key()
    if not api_key:
        error_payload = {
            "type": "error",
            "error": "llm_unauthorized",
            "message": "Missing OPENROUTER_API_KEY.",
            "status": 401,
        }
        yield f"data: {json.dumps(error_payload)}\n\n"
        return

    headers = {**_build_headers(api_key), "Accept": "text/event-stream"}

    saw_rate_limit = False
    last_error: dict[str, Any] | None = None

    async with httpx.AsyncClient(timeout=_get_timeout_sec()) as client:
        for attempt in range(_get_max_retries() + 1):
            for model in _get_model_fallbacks():
                request_body = _build_payload(model, payload, stream=True)
                try:
                    async with client.stream(
                        "POST",
                        f"{OPENROUTER_BASE_URL.rstrip('/')}/chat/completions",
                        json=request_body,
                        headers=headers,
                    ) as response:
                        if response.status_code == 429:
                            saw_rate_limit = True
                            body_text = await response.aread()
                            try:
                                last_error = json.loads(body_text.decode("utf-8"))
                            except Exception:
                                last_error = None
                            continue

                        if response.status_code in (401, 403):
                            error_payload = {
                                "type": "error",
                                "error": "llm_unauthorized",
                                "message": "OpenRouter unauthorized. Check OPENROUTER_API_KEY.",
                                "status": 401,
                            }
                            yield f"data: {json.dumps(error_payload)}\n\n"
                            return

                        if response.status_code >= 400:
                            error_payload = {
                                "type": "error",
                                "error": "llm_error",
                                "message": f"OpenRouter returned status {response.status_code}.",
                                "status": response.status_code,
                            }
                            yield f"data: {json.dumps(error_payload)}\n\n"
                            return

                        saw_tokens = False
                        usage_payload: dict[str, Any] | None = None

                        async for line in response.aiter_lines():
                            if not line or not line.startswith("data:"):
                                continue

                            raw_payload = line.removeprefix("data:").strip()
                            if not raw_payload:
                                continue
                            if raw_payload == "[DONE]":
                                done_payload = {
                                    "type": "done",
                                    "provider": "openrouter",
                                    "model": model,
                                    "usage": usage_payload,
                                }
                                yield f"data: {json.dumps(done_payload)}\n\n"
                                return

                            try:
                                chunk = json.loads(raw_payload)
                            except json.JSONDecodeError:
                                continue

                            if isinstance(chunk, dict):
                                usage = chunk.get("usage")
                                if isinstance(usage, dict):
                                    usage_payload = usage

                            delta = ""
                            if isinstance(chunk.get("choices"), list) and chunk["choices"]:
                                choice = chunk["choices"][0]
                                if isinstance(choice, dict):
                                    delta_payload = choice.get("delta")
                                    if isinstance(delta_payload, dict):
                                        delta = _extract_text_content(delta_payload.get("content"))

                            if delta:
                                saw_tokens = True
                                delta_payload = {
                                    "type": "delta",
                                    "delta": delta,
                                    "provider": "openrouter",
                                    "model": model,
                                }
                                yield f"data: {json.dumps(delta_payload)}\n\n"

                        if saw_tokens:
                            done_payload = {
                                "type": "done",
                                "provider": "openrouter",
                                "model": model,
                                "usage": usage_payload,
                            }
                            yield f"data: {json.dumps(done_payload)}\n\n"
                            return
                except httpx.TimeoutException:
                    continue
                except httpx.HTTPError:
                    continue

            if attempt < _get_max_retries():
                await asyncio.sleep(_get_backoff_sec() * (attempt + 1))

    if saw_rate_limit:
        message = "All configured OpenRouter fallback models are rate limited."
        if isinstance(last_error, dict):
            error_obj = last_error.get("error")
            if isinstance(error_obj, dict) and isinstance(error_obj.get("message"), str):
                message = error_obj["message"]
        error_payload = {
            "type": "error",
            "error": "llm_rate_limited",
            "message": message,
            "status": 429,
        }
        yield f"data: {json.dumps(error_payload)}\n\n"
        return

    error_payload = {
        "type": "error",
        "error": "llm_unreachable",
        "message": "OpenRouter is unreachable.",
        "status": 502,
    }
    yield f"data: {json.dumps(error_payload)}\n\n"


@router.post("/generate", response_model=TutorResponse)
async def generate_tutor_response(payload: TutorRequest, request: Request) -> TutorResponse:
    _check_rate_limit(_client_id(request))

    data, model = await _chat_completion_with_fallback(payload)
    content = _extract_chat_completion_content(data)

    if not content:
        raise HTTPException(
            status_code=502,
            detail={"error": "llm_invalid_response", "message": "No model content in response."},
        )

    return TutorResponse(
        provider="openrouter",
        model=model,
        content=content,
        usage=data.get("usage") if isinstance(data, dict) else None,
    )


@router.post("/generate/stream")
async def stream_tutor_response(payload: TutorRequest, request: Request) -> StreamingResponse:
    _check_rate_limit(_client_id(request))

    return StreamingResponse(
        _stream_openrouter(payload),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
