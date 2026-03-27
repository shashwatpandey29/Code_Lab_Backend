from fastapi import APIRouter
from fastapi import Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import ProgressRecord

router = APIRouter()


class ProgressPayload(BaseModel):
    user_id: str
    lesson_id: str
    completed: bool


@router.get("/")
def list_progress(user_id: str, db: Session = Depends(get_db)) -> dict[str, list[dict[str, object]]]:
    rows = db.execute(
        select(ProgressRecord).where(ProgressRecord.user_id == user_id).order_by(ProgressRecord.lesson_id.asc())
    ).scalars()

    return {
        "items": [
            {
                "lesson_id": row.lesson_id,
                "completed": row.completed,
            }
            for row in rows
        ]
    }


@router.post("/")
def upsert_progress(payload: ProgressPayload, db: Session = Depends(get_db)) -> dict[str, object]:
    record = db.execute(
        select(ProgressRecord)
        .where(ProgressRecord.user_id == payload.user_id)
        .where(ProgressRecord.lesson_id == payload.lesson_id)
    ).scalar_one_or_none()

    if record is None:
        record = ProgressRecord(
            user_id=payload.user_id,
            lesson_id=payload.lesson_id,
            completed=payload.completed,
        )
        db.add(record)
    else:
        record.completed = payload.completed

    db.commit()

    return {
        "user_id": payload.user_id,
        "lesson_id": payload.lesson_id,
        "completed": payload.completed,
        "saved": True,
    }
