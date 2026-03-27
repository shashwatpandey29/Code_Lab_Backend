from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import UserPreferenceRecord

router = APIRouter()


class ThemePayload(BaseModel):
    user_id: str = Field(min_length=1, max_length=120)
    theme: str = Field(pattern="^(light|dark)$")


@router.get("/theme")
def get_theme(user_id: str, db: Session = Depends(get_db)) -> dict[str, str]:
    try:
        row = db.execute(select(UserPreferenceRecord).where(UserPreferenceRecord.user_id == user_id)).scalar_one_or_none()
    except Exception:
        db.rollback()
        return {"user_id": user_id, "theme": "light"}

    if row is None:
        return {"user_id": user_id, "theme": "light"}
    return {"user_id": row.user_id, "theme": row.theme}


@router.post("/theme")
def upsert_theme(payload: ThemePayload, db: Session = Depends(get_db)) -> dict[str, str]:
    try:
        row = db.execute(select(UserPreferenceRecord).where(UserPreferenceRecord.user_id == payload.user_id)).scalar_one_or_none()
        if row is None:
            row = UserPreferenceRecord(user_id=payload.user_id, theme=payload.theme)
            db.add(row)
        else:
            row.theme = payload.theme

        db.commit()
        return {"user_id": payload.user_id, "theme": payload.theme}
    except Exception:
        db.rollback()
        return {"user_id": payload.user_id, "theme": payload.theme}
