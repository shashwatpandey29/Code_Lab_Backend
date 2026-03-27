import time
import uuid
import os
from urllib.parse import urlparse

from fastapi import FastAPI
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
from sqlalchemy import text

from app.db import Base, SessionLocal, engine
from app import models  # noqa: F401
from app.routers import auth, courses, execution, llm, preferences, progress, snippets

app = FastAPI(title="CodeLab API", version="0.1.0")


def _parse_cors_origins() -> tuple[list[str], bool]:
    allow_all = os.getenv("CORS_ALLOW_ALL", "false").lower() in {"1", "true", "yes"}
    if allow_all:
        return ["*"], True

    raw = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    origins = [item.strip() for item in raw.split(",") if item.strip()]
    if not origins:
        origins = ["http://localhost:3000", "http://127.0.0.1:3000"]

    return origins, False


cors_origins, cors_allow_all = _parse_cors_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=not cors_allow_all,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(progress.router, prefix="/progress", tags=["progress"])
app.include_router(execution.router, prefix="/execution", tags=["execution"])
app.include_router(snippets.router, prefix="/snippets", tags=["snippets"])
app.include_router(llm.router, prefix="/llm", tags=["llm"])
app.include_router(courses.router, prefix="/courses", tags=["courses"])
app.include_router(preferences.router, prefix="/preferences", tags=["preferences"])


@app.on_event("startup")
def init_database() -> None:
    Base.metadata.create_all(bind=engine)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    started = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        response = JSONResponse(
            status_code=500,
            content={"error": "internal_server_error", "request_id": request_id},
        )

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    response.headers["x-request-id"] = request_id
    response.headers["x-response-time-ms"] = str(elapsed_ms)
    return response


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
async def readiness_check():
    try:
        with SessionLocal() as session:
            session.execute(text("SELECT 1"))

        require_llm = os.getenv("REQUIRE_LLM_READY", "false").lower() in {"1", "true", "yes"}
        if require_llm:
            if not await llm.provider_ready():
                return JSONResponse(status_code=503, content={"status": "not_ready", "dependency": "openrouter"})

        return {"status": "ready"}
    except Exception:
        return JSONResponse(status_code=503, content={"status": "not_ready"})


@app.get("/runtime/status")
async def runtime_status() -> JSONResponse:
    checks: dict[str, dict[str, object]] = {
        "database": {"status": "down"},
        "llm": {"status": "down"},
        "judge0": {"status": "down"},
        "python": {"status": "ready", "mode": "in-browser"},
        "sql": {"status": "ready", "mode": "in-browser"},
        "javascript": {"status": "ready", "mode": "in-browser"},
    }

    overall = "ready"

    db_started = time.perf_counter()
    try:
        with SessionLocal() as session:
            session.execute(text("SELECT 1"))
        checks["database"] = {
            "status": "ready",
            "latency_ms": int((time.perf_counter() - db_started) * 1000),
        }
    except Exception:
        overall = "degraded"

    llm_started = time.perf_counter()
    try:
        llm_ok = await llm.provider_ready()
        checks["llm"] = {
            "status": "ready" if llm_ok else "down",
            "latency_ms": int((time.perf_counter() - llm_started) * 1000),
        }
        if not llm_ok:
            overall = "degraded"
    except Exception:
        overall = "degraded"

    judge0_url = os.getenv("JUDGE0_URL", "https://ce.judge0.com/submissions?base64_encoded=true&wait=true")
    rapidapi_key = os.getenv("JUDGE0_RAPIDAPI_KEY")
    if "rapidapi.com" in judge0_url and not rapidapi_key:
        checks["judge0"] = {
            "status": "degraded",
            "message": "RapidAPI key missing.",
        }
        overall = "degraded"
    else:
        endpoint = judge0_url
        parsed = urlparse(judge0_url)
        if parsed.scheme and parsed.netloc:
            endpoint = f"{parsed.scheme}://{parsed.netloc}"

        headers: dict[str, str] = {}
        rapidapi_host = os.getenv("JUDGE0_RAPIDAPI_HOST")
        if rapidapi_key:
            headers["X-RapidAPI-Key"] = rapidapi_key
        if rapidapi_host:
            headers["X-RapidAPI-Host"] = rapidapi_host

        judge0_started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=6) as client:
                response = await client.get(endpoint, headers=headers)
            if response.status_code < 500:
                checks["judge0"] = {
                    "status": "ready",
                    "latency_ms": int((time.perf_counter() - judge0_started) * 1000),
                }
            else:
                checks["judge0"] = {
                    "status": "down",
                    "latency_ms": int((time.perf_counter() - judge0_started) * 1000),
                    "message": f"Upstream status {response.status_code}",
                }
                overall = "degraded"
        except Exception:
            checks["judge0"] = {"status": "down"}
            overall = "degraded"

    db_ready = checks.get("database", {}).get("status") == "ready"
    status_code = 200 if db_ready else 503
    return JSONResponse(status_code=status_code, content={"status": overall, "checks": checks})
