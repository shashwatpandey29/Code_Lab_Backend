import os
import time
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

router = APIRouter()

JUDGE0_URL = os.getenv("JUDGE0_URL", "https://ce.judge0.com/submissions?base64_encoded=true&wait=true")
UPSTREAM_TIMEOUT_SEC = float(os.getenv("EXECUTION_UPSTREAM_TIMEOUT_SEC", "20"))
RATE_LIMIT_WINDOW_SEC = int(os.getenv("EXECUTION_RATE_LIMIT_WINDOW_SEC", "60"))
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("EXECUTION_RATE_LIMIT_MAX_REQUESTS", "20"))


class ExecuteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language_id: int = Field(gt=0)
    source_code: str = Field(min_length=1, max_length=120_000)
    stdin: str = Field(default="", max_length=20_000)
    cpu_time_limit: float = Field(default=10, gt=0, le=20)
    memory_limit: int = Field(default=128_000, ge=16_000, le=256_000)


_rate_limiter: dict[str, tuple[int, float]] = {}


def _client_id(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.headers.get("x-real-ip") or (request.client.host if request.client else "unknown")


def _check_rate_limit(client_id: str) -> None:
    now = time.time()
    count, reset_at = _rate_limiter.get(client_id, (0, now + RATE_LIMIT_WINDOW_SEC))

    if now >= reset_at:
        count, reset_at = 0, now + RATE_LIMIT_WINDOW_SEC

    if count >= RATE_LIMIT_MAX_REQUESTS:
        retry_after = int(max(1, reset_at - now))
        raise HTTPException(
            status_code=429,
            detail={
                "error": "rate_limited",
                "message": "Too many execution requests.",
                "retry_after": retry_after,
            },
            headers={"Retry-After": str(retry_after)},
        )

    _rate_limiter[client_id] = (count + 1, reset_at)


@router.post("/")
async def execute(payload: ExecuteRequest, request: Request) -> dict[str, Any]:
    _check_rate_limit(_client_id(request))

    headers = {"Content-Type": "application/json"}

    rapid_key = os.getenv("JUDGE0_RAPIDAPI_KEY")
    rapid_host = os.getenv("JUDGE0_RAPIDAPI_HOST")

    if "rapidapi.com" in JUDGE0_URL and not rapid_key:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "execution_not_configured",
                "message": "JUDGE0_RAPIDAPI_KEY is required for RapidAPI Judge0 endpoints.",
            },
        )

    if rapid_key:
        headers["X-RapidAPI-Key"] = rapid_key
    if rapid_host:
        headers["X-RapidAPI-Host"] = rapid_host

    try:
        async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT_SEC) as client:
            response = await client.post(JUDGE0_URL, json=payload.model_dump(), headers=headers)
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail={
                "error": "execution_timeout",
                "message": "Execution provider timed out.",
            },
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "execution_upstream_error",
                "message": str(exc),
            },
        ) from exc

    if response.status_code >= 400:
        raise HTTPException(
            status_code=response.status_code,
            detail={
                "error": "execution_rejected",
                "message": response.text,
            },
        )

    return response.json()
