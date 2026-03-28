import hashlib
import json
import os
import asyncio
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.db import get_db
from app.models import CourseGenerationJobRecord, CourseLessonRecord, CourseModuleRecord, CourseRecord

router = APIRouter()

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", os.getenv("XAI_API_KEY", os.getenv("GROK_API_KEY", "")))
OPENROUTER_MODEL_FALLBACKS = os.getenv(
    "OPENROUTER_MODEL_FALLBACKS",
    "openrouter/auto,meta-llama/llama-3.1-8b-instruct:free,mistralai/mistral-7b-instruct:free",
)
OPENROUTER_BACKOFF_SEC = float(os.getenv("OPENROUTER_BACKOFF_SEC", "1.0"))
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "https://code-lab-frontend-five.vercel.app")
OPENROUTER_X_TITLE = os.getenv("OPENROUTER_X_TITLE", "CodeLab")
COURSE_MAX_MODULES = int(os.getenv("COURSE_MAX_MODULES", "8"))
COURSE_MAX_LESSONS_PER_MODULE = int(os.getenv("COURSE_MAX_LESSONS_PER_MODULE", "6"))
COURSE_MIN_MODULES = int(os.getenv("COURSE_MIN_MODULES", "2"))
COURSE_MIN_LESSONS_PER_MODULE = int(os.getenv("COURSE_MIN_LESSONS_PER_MODULE", "2"))
COURSE_GENERATION_TIMEOUT_SEC = float(os.getenv("COURSE_GENERATION_TIMEOUT_SEC", "60"))
COURSE_GENERATION_RETRY_COUNT = int(os.getenv("COURSE_GENERATION_RETRY_COUNT", "4"))


class CourseLesson(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=3, max_length=200)
    explanation: str = Field(min_length=20, max_length=8000)
    example_code: str = Field(min_length=3, max_length=12000)
    exercise: str = Field(min_length=5, max_length=4000)


class CourseModule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=3, max_length=180)
    objective: str = Field(min_length=10, max_length=1000)
    lessons: list[CourseLesson] = Field(min_length=COURSE_MIN_LESSONS_PER_MODULE, max_length=COURSE_MAX_LESSONS_PER_MODULE)


class CourseSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str = Field(min_length=1, max_length=40)
    title: str = Field(min_length=4, max_length=200)
    description: str = Field(min_length=20, max_length=4000)
    level: str = Field(default="beginner", min_length=3, max_length=40)
    modules: list[CourseModule] = Field(min_length=COURSE_MIN_MODULES, max_length=COURSE_MAX_MODULES)


class CourseGeneratePayload(BaseModel):
    language: str = Field(min_length=1, max_length=40)
    level: str = Field(default="beginner", min_length=3, max_length=40)
    force_refresh: bool = False


def _get_openrouter_models() -> list[str]:
    models = [item.strip() for item in OPENROUTER_MODEL_FALLBACKS.split(",") if item.strip()]
    if not models:
        return ["openrouter/auto"]
    return models


def _default_course_model() -> str:
    return _get_openrouter_models()[0]


def _normalize_language(language: str) -> str:
    return language.strip().lower()


def _example_for_language(language: str, topic: str) -> str:
    examples = {
        "python": {
            "syntax": "def greet(name: str) -> str:\n    return f\"Hello, {name}\"\n\nprint(greet(\"CodeLab\"))",
            "data": "nums = [1, 2, 3, 4]\nresult = [n * n for n in nums if n % 2 == 0]\nprint(result)",
            "testing": "def add(a, b):\n    return a + b\n\ndef test_add():\n    assert add(2, 3) == 5",
            "api": "from fastapi import FastAPI\napp = FastAPI()\n\n@app.get(\"/health\")\ndef health():\n    return {\"status\": \"ok\"}",
        },
        "javascript": {
            "syntax": "const greet = (name) => `Hello, ${name}`;\nconsole.log(greet('CodeLab'));",
            "data": "const nums = [1, 2, 3, 4];\nconst result = nums.filter(n => n % 2 === 0).map(n => n * n);\nconsole.log(result);",
            "testing": "function add(a, b) { return a + b; }\nconsole.assert(add(2, 3) === 5, 'add failed');",
            "api": "import express from 'express';\nconst app = express();\napp.get('/health', (_, res) => res.json({ status: 'ok' }));",
        },
        "sql": {
            "syntax": "SELECT 1 AS hello_world;",
            "data": "SELECT department, COUNT(*) AS employees\nFROM staff\nGROUP BY department\nORDER BY employees DESC;",
            "testing": "-- Verify row counts after migration\nSELECT COUNT(*) FROM users;",
            "api": "SELECT id, title, created_at\nFROM posts\nWHERE published = 1\nORDER BY created_at DESC\nLIMIT 20;",
        },
        "java": {
            "syntax": "public class Main {\n  public static void main(String[] args) {\n    System.out.println(\"Hello CodeLab\");\n  }\n}",
            "data": "List<Integer> nums = List.of(1,2,3,4);\nList<Integer> result = nums.stream().filter(n -> n % 2 == 0).map(n -> n * n).toList();",
            "testing": "@Test\nvoid addWorks() {\n  assertEquals(5, MathUtil.add(2, 3));\n}",
            "api": "@RestController\nclass HealthController {\n  @GetMapping(\"/health\")\n  Map<String, String> health() { return Map.of(\"status\", \"ok\"); }\n}",
        },
        "c": {
            "syntax": "#include <stdio.h>\nint main(void){ printf(\"Hello CodeLab\\n\"); return 0; }",
            "data": "#include <stdio.h>\nint nums[] = {1,2,3,4};\nfor(int i=0;i<4;i++){ if(nums[i]%2==0) printf(\"%d \", nums[i]*nums[i]); }",
            "testing": "#include <assert.h>\nint add(int a,int b){ return a+b; }\nint main(void){ assert(add(2,3)==5); }",
            "api": "/* C services are often built with libraries like libmicrohttpd */\n/* Focus on request parsing, memory safety, and response construction. */",
        },
        "cpp": {
            "syntax": "#include <iostream>\nint main(){ std::cout << \"Hello CodeLab\\n\"; }",
            "data": "std::vector<int> nums{1,2,3,4};\nstd::vector<int> out;\nstd::copy_if(nums.begin(), nums.end(), std::back_inserter(out), [](int n){ return n % 2 == 0; });",
            "testing": "#include <cassert>\nint add(int a,int b){ return a+b; }\nint main(){ assert(add(2,3)==5); }",
            "api": "// C++ REST APIs commonly use frameworks like Drogon or Pistache.\n// Emphasize RAII, thread safety, and performance profiling.",
        },
        "rust": {
            "syntax": "fn main() {\n    println!(\"Hello CodeLab\");\n}",
            "data": "let nums = vec![1,2,3,4];\nlet result: Vec<i32> = nums.into_iter().filter(|n| n % 2 == 0).map(|n| n * n).collect();",
            "testing": "fn add(a: i32, b: i32) -> i32 { a + b }\n#[test]\nfn add_works() { assert_eq!(add(2, 3), 5); }",
            "api": "use axum::{routing::get, Router};\nasync fn health() -> &'static str { \"ok\" }\nlet app = Router::new().route(\"/health\", get(health));",
        },
        "go": {
            "syntax": "package main\nimport \"fmt\"\nfunc main(){ fmt.Println(\"Hello CodeLab\") }",
            "data": "nums := []int{1,2,3,4}\nout := make([]int,0)\nfor _, n := range nums { if n%2==0 { out = append(out, n*n) } }",
            "testing": "func Add(a, b int) int { return a + b }\nfunc TestAdd(t *testing.T) { if Add(2,3) != 5 { t.Fatal(\"failed\") } }",
            "api": "func health(w http.ResponseWriter, r *http.Request) {\n  w.Header().Set(\"Content-Type\", \"application/json\")\n  w.Write([]byte(`{\"status\":\"ok\"}`))\n}",
        },
    }

    by_topic = examples.get(language, examples["javascript"])
    return by_topic.get(topic, by_topic["syntax"])


def _build_fallback_course(language: str, level: str) -> CourseSchema:
    normalized = _normalize_language(language)
    module_specs = [
        {
            "title": f"{normalized.upper()} Foundations and Core Syntax",
            "objective": "Build beginner confidence with syntax, tooling, data types, and control flow.",
            "lessons": [
                {
                    "title": "Setup, Tooling, and Project Workflow",
                    "explanation": "Learn the local development workflow, dependency setup, formatting conventions, and basic debugging so you can iterate quickly and avoid environment-related blockers.",
                    "example_code": _example_for_language(normalized, "syntax"),
                    "exercise": "Create a tiny starter project with one function, run it, and document each command in your README.",
                },
                {
                    "title": "Control Flow and Data Modeling",
                    "explanation": "Practice conditionals, loops, and core data structures while choosing readable naming and function boundaries. This lesson bridges beginner syntax to intermediate problem solving.",
                    "example_code": _example_for_language(normalized, "data"),
                    "exercise": "Implement a small data transformation pipeline and add input validation for edge cases.",
                },
                {
                    "title": "Functions, Error Handling, and Refactoring",
                    "explanation": "Move from script-style code to reusable units with clear contracts and failure handling. Focus on testability and maintainability as project size grows.",
                    "example_code": _example_for_language(normalized, "testing"),
                    "exercise": "Refactor one long function into smaller tested functions with meaningful errors.",
                },
            ],
        },
        {
            "title": f"Intermediate {normalized.upper()} Patterns",
            "objective": "Develop production-grade patterns for architecture, testing, and state management.",
            "lessons": [
                {
                    "title": "Modular Design and Separation of Concerns",
                    "explanation": "Learn to split code into modules, enforce boundaries, and organize responsibilities for long-term maintainability.",
                    "example_code": _example_for_language(normalized, "data"),
                    "exercise": "Restructure your project into domain, application, and infrastructure layers.",
                },
                {
                    "title": "Testing Strategy and Reliability",
                    "explanation": "Apply unit and integration testing, manage fixtures, and improve confidence with deterministic test execution.",
                    "example_code": _example_for_language(normalized, "testing"),
                    "exercise": "Add a test suite that covers happy path, invalid inputs, and one failure scenario.",
                },
                {
                    "title": "Performance and Profiling Basics",
                    "explanation": "Identify bottlenecks, measure execution cost, and improve algorithmic or I/O efficiency using data-driven optimization.",
                    "example_code": _example_for_language(normalized, "data"),
                    "exercise": "Profile your code and optimize one hotspot while preserving behavior with tests.",
                },
            ],
        },
        {
            "title": f"Backend and Data Workloads in {normalized.upper()}",
            "objective": "Implement APIs, persistence patterns, and integration workflows for real products.",
            "lessons": [
                {
                    "title": "Building Service Endpoints",
                    "explanation": "Design endpoint contracts, validation, and response shapes that remain stable and observable in production systems.",
                    "example_code": _example_for_language(normalized, "api"),
                    "exercise": "Implement a health endpoint and one CRUD-style endpoint with validation.",
                },
                {
                    "title": "Persistence and Query Patterns",
                    "explanation": "Model entities, optimize common query paths, and enforce consistency with transactions and migration practices.",
                    "example_code": _example_for_language(normalized, "data"),
                    "exercise": "Add a persistence layer and implement one indexed query with benchmark notes.",
                },
                {
                    "title": "Observability and Failure Handling",
                    "explanation": "Capture logs, metrics, and traces while implementing retry, timeout, and graceful-degradation strategies.",
                    "example_code": _example_for_language(normalized, "api"),
                    "exercise": "Add structured logging and timeout handling to one service operation.",
                },
            ],
        },
        {
            "title": f"Advanced {normalized.upper()} Engineering",
            "objective": "Reach advanced mastery with scalability, security, and architecture decision-making.",
            "lessons": [
                {
                    "title": "Concurrency, Parallelism, and Throughput",
                    "explanation": "Use concurrency models appropriately, avoid race conditions, and tune throughput under realistic load.",
                    "example_code": _example_for_language(normalized, "api"),
                    "exercise": "Implement concurrent processing with limits and benchmark before/after throughput.",
                },
                {
                    "title": "Security Hardening and Defensive Coding",
                    "explanation": "Apply input validation, auth boundaries, secure defaults, and dependency hygiene to reduce system risk.",
                    "example_code": _example_for_language(normalized, "testing"),
                    "exercise": "Threat-model your service and patch one high-impact vulnerability class.",
                },
                {
                    "title": "Capstone: Production-Ready Feature Delivery",
                    "explanation": "Ship an end-to-end feature with architecture notes, tests, observability, and deployment checklist from beginner foundations to advanced design tradeoffs.",
                    "example_code": _example_for_language(normalized, "api"),
                    "exercise": "Build and document a capstone service that includes auth, persistence, tests, and monitoring hooks.",
                },
            ],
        },
    ]

    return CourseSchema.model_validate(
        {
            "language": normalized,
            "title": f"{normalized.upper()} Complete Path: Beginner to Advanced",
            "description": (
                f"A full progression roadmap for {normalized.upper()} from beginner fundamentals to advanced production engineering, "
                "with practical projects, testing, performance, and architecture topics."
            ),
            "level": level,
            "modules": module_specs,
        }
    )


def _build_prompt(language: str, level: str) -> str:
    return (
        "Generate a complete practical coding course in strict JSON format. "
        f"Language: {language}. Target level: {level}. "
        "The course must span beginner to advanced progression. "
        "Output valid JSON only with keys: language, title, description, level, modules. "
        "Each module needs: title, objective, lessons. "
        "Each lesson needs: title, explanation, example_code, exercise. "
        f"Return between {COURSE_MIN_MODULES} and {COURSE_MAX_MODULES} modules. "
        f"Each module must include between {COURSE_MIN_LESSONS_PER_MODULE} and {COURSE_MAX_LESSONS_PER_MODULE} lessons. "
        "Use concise but detailed explanations with production-oriented examples. "
        "Avoid placeholders and keep examples executable where possible. "
        "Do not include markdown fences, notes, or any keys outside the schema."
    )


def _extract_json_block(raw: str) -> dict:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(raw[start : end + 1])


def _extract_llm_message_content(response_json: dict[str, object]) -> str:
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""

    message = first_choice.get("message")
    if not isinstance(message, dict):
        return ""

    content = message.get("content")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
        return "".join(text_parts)

    return ""


def _quality_check_course(course: CourseSchema) -> None:
    if len(course.modules) < COURSE_MIN_MODULES:
        raise HTTPException(status_code=502, detail={"error": "course_generation_invalid", "message": "Generated course has too few modules."})

    for module in course.modules:
        if len(module.lessons) < COURSE_MIN_LESSONS_PER_MODULE:
            raise HTTPException(status_code=502, detail={"error": "course_generation_invalid", "message": "Generated module has too few lessons."})

        for lesson in module.lessons:
            payload = " ".join([lesson.title, lesson.explanation, lesson.example_code, lesson.exercise]).lower()
            if "todo" in payload or "placeholder" in payload or "lorem ipsum" in payload:
                raise HTTPException(
                    status_code=502,
                    detail={"error": "course_generation_invalid", "message": "Generated lesson contains placeholder content."},
                )
            if len(lesson.example_code.strip()) < 8:
                raise HTTPException(
                    status_code=502,
                    detail={"error": "course_generation_invalid", "message": "Generated lesson example code is too short."},
                )


async def _generate_course_via_llm(language: str, level: str) -> CourseSchema:
    headers = {"Content-Type": "application/json"}
    if OPENROUTER_API_KEY:
        headers["Authorization"] = f"Bearer {OPENROUTER_API_KEY}"
        headers["HTTP-Referer"] = OPENROUTER_HTTP_REFERER
        headers["X-Title"] = OPENROUTER_X_TITLE
    else:
        raise HTTPException(
            status_code=401,
            detail={"error": "llm_unauthorized", "message": "Missing OPENROUTER_API_KEY."},
        )

    prompt = _build_prompt(language, level)
    last_error = "unknown_generation_error"
    repair_hint = ""

    async with httpx.AsyncClient(timeout=COURSE_GENERATION_TIMEOUT_SEC) as client:
        for attempt in range(COURSE_GENERATION_RETRY_COUNT):
            user_prompt = prompt if not repair_hint else f"{prompt}\n\nRepair instruction: {repair_hint}"
            for model in _get_openrouter_models():
                payload = {
                    "model": model,
                    "stream": False,
                    "temperature": 0.2,
                    "max_tokens": 3800,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a senior technical curriculum architect. "
                                "Return strict JSON that matches the requested schema exactly."
                            ),
                        },
                        {"role": "user", "content": user_prompt},
                    ],
                }

                response = await client.post(f"{OPENROUTER_BASE_URL.rstrip('/')}/chat/completions", json=payload, headers=headers)

                if response.status_code == 429:
                    last_error = "llm_rate_limited"
                    continue

                if response.status_code in (401, 403):
                    raise HTTPException(
                        status_code=401,
                        detail={"error": "llm_unauthorized", "message": "OpenRouter unauthorized. Check OPENROUTER_API_KEY."},
                    )

                if response.status_code >= 400:
                    last_error = f"llm_provider_error_{response.status_code}"
                    continue

                data = response.json()
                content = _extract_llm_message_content(data) if isinstance(data, dict) else ""

                if not content:
                    last_error = "empty_llm_response"
                    repair_hint = "Return only a valid JSON object with all required keys and non-empty values."
                    continue

                try:
                    parsed = _extract_json_block(content)
                    course = CourseSchema.model_validate(parsed)
                    _quality_check_course(course)
                    return course
                except HTTPException as exc:
                    last_error = str(exc.detail)
                    repair_hint = "The previous output failed quality checks. Remove placeholders and provide executable example code for every lesson."
                except Exception:
                    last_error = "schema_validation_failed"
                    repair_hint = (
                        "The previous output did not match schema. "
                        "Return exactly one JSON object with keys: language, title, description, level, modules; "
                        "and for each module keys: title, objective, lessons; for each lesson keys: title, explanation, example_code, exercise."
                    )

            if attempt < COURSE_GENERATION_RETRY_COUNT - 1:
                await asyncio.sleep(OPENROUTER_BACKOFF_SEC * (attempt + 1))

    raise HTTPException(
        status_code=502,
        detail={"error": "course_generation_invalid", "message": f"Generated course failed schema validation after retries: {last_error}."},
    )


def _serialize_course(record: CourseRecord) -> dict[str, object]:
    modules_payload: list[dict[str, object]] = []
    modules = sorted(record.modules, key=lambda item: item.module_index)
    for module in modules:
        lessons = sorted(module.lessons, key=lambda item: item.lesson_index)
        modules_payload.append(
            {
                "title": module.title,
                "objective": module.objective,
                "lessons": [
                    {
                        "title": lesson.title,
                        "explanation": lesson.explanation,
                        "example_code": lesson.example_code,
                        "exercise": lesson.exercise,
                    }
                    for lesson in lessons
                ],
            }
        )

    return {
        "id": record.id,
        "language": record.language,
        "title": record.title,
        "description": record.description,
        "level": record.level,
        "version": record.version,
        "status": record.status,
        "model": record.model,
        "modules": modules_payload,
    }


def _get_latest_course(db: Session, language: str) -> CourseRecord | None:
    return db.execute(
        select(CourseRecord)
        .options(selectinload(CourseRecord.modules).selectinload(CourseModuleRecord.lessons))
        .where(CourseRecord.language == language)
        .order_by(CourseRecord.version.desc())
        .limit(1)
    ).scalar_one_or_none()


async def _persist_generated_course(db: Session, course: CourseSchema, prompt_hash: str) -> CourseRecord:
    latest_version = db.execute(
        select(CourseRecord.version)
        .where(CourseRecord.language == course.language.lower())
        .order_by(CourseRecord.version.desc())
        .limit(1)
    ).scalar_one_or_none()
    next_version = int(latest_version or 0) + 1

    record = CourseRecord(
        language=course.language.lower(),
        title=course.title,
        description=course.description,
        level=course.level,
        version=next_version,
        status="published",
        model=_default_course_model(),
        generated_by_prompt_hash=prompt_hash,
    )
    db.add(record)
    db.flush()

    for module_index, module in enumerate(course.modules, start=1):
        module_record = CourseModuleRecord(
            course_id=record.id,
            module_index=module_index,
            title=module.title,
            objective=module.objective,
        )
        db.add(module_record)
        db.flush()

        for lesson_index, lesson in enumerate(module.lessons, start=1):
            db.add(
                CourseLessonRecord(
                    module_id=module_record.id,
                    lesson_index=lesson_index,
                    title=lesson.title,
                    explanation=lesson.explanation,
                    example_code=lesson.example_code,
                    exercise=lesson.exercise,
                )
            )

    db.commit()
    db.refresh(record)
    return db.execute(
        select(CourseRecord)
        .options(selectinload(CourseRecord.modules).selectinload(CourseModuleRecord.lessons))
        .where(CourseRecord.id == record.id)
    ).scalar_one()


async def _generate_and_persist(db: Session, language: str, level: str, source: str) -> dict[str, object]:
    prompt = _build_prompt(language, level)
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    job = CourseGenerationJobRecord(language=language, status="running", model=_default_course_model(), prompt_hash=prompt_hash)
    db.add(job)
    db.commit()
    db.refresh(job)

    try:
        generated = await _generate_course_via_llm(language, level)
        generated.language = language
        record = await _persist_generated_course(db, generated, prompt_hash)
        job.status = "completed"
        job.completed_at = datetime.now(timezone.utc)
        db.commit()
        return {"source": source, "course": _serialize_course(record)}
    except HTTPException as exc:
        job.status = "failed"
        job.error_message = str(exc.detail)
        job.completed_at = datetime.now(timezone.utc)
        db.commit()
        raise
    except Exception as exc:
        try:
            fallback = _build_fallback_course(language, level)
            record = await _persist_generated_course(db, fallback, prompt_hash)
            job.status = "completed"
            job.error_message = f"LLM failed, fallback used: {exc}"
            job.completed_at = datetime.now(timezone.utc)
            db.commit()
            return {"source": "fallback", "course": _serialize_course(record)}
        except Exception:
            job.status = "failed"
            job.error_message = str(exc)
            job.completed_at = datetime.now(timezone.utc)
            db.commit()
            raise HTTPException(status_code=500, detail={"error": "course_generation_failed", "message": "Unexpected generation error."}) from exc


@router.post("/generate")
async def generate_course(payload: CourseGeneratePayload, db: Session = Depends(get_db)) -> dict[str, object]:
    normalized_language = _normalize_language(payload.language)
    if not payload.force_refresh:
        course = _get_latest_course(db, normalized_language)
        if course is not None:
            return {"source": "cache", "course": _serialize_course(course)}

    return await _generate_and_persist(db, normalized_language, payload.level, "generated")


@router.get("/")
def list_courses(db: Session = Depends(get_db)) -> dict[str, list[dict[str, object]]]:
    rows = db.execute(select(CourseRecord).order_by(CourseRecord.language.asc(), CourseRecord.version.desc())).scalars()
    return {
        "items": [
            {
                "id": row.id,
                "language": row.language,
                "title": row.title,
                "level": row.level,
                "version": row.version,
                "status": row.status,
                "model": row.model,
            }
            for row in rows
        ]
    }


@router.get("/{language}")
async def get_course(
    language: str,
    ensure_generated: bool = Query(default=False),
    force_refresh: bool = Query(default=False),
    level: str = Query(default="beginner"),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    normalized_language = _normalize_language(language)
    if not force_refresh:
        course = _get_latest_course(db, normalized_language)
        if course is not None:
            return {"source": "cache", "course": _serialize_course(course)}

    if not ensure_generated:
        raise HTTPException(status_code=404, detail={"error": "course_not_found", "message": "No stored course found for this language."})

    return await _generate_and_persist(db, normalized_language, level, "generated")
