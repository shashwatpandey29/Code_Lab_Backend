from fastapi import APIRouter
from fastapi import Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import SnippetRecord

router = APIRouter()


class SnippetPayload(BaseModel):
    user_id: str
    language: str
    code: str
    title: str


@router.get("/")
def list_snippets(user_id: str, db: Session = Depends(get_db)) -> dict[str, list[dict[str, str]]]:
    rows = db.execute(
        select(SnippetRecord).where(SnippetRecord.user_id == user_id).order_by(SnippetRecord.id.desc())
    ).scalars()

    return {
        "items": [
            {
                "id": str(row.id),
                "language": row.language,
                "title": row.title,
            }
            for row in rows
        ]
    }


@router.post("/")
def create_snippet(payload: SnippetPayload, db: Session = Depends(get_db)) -> dict[str, str]:
    row = SnippetRecord(
        user_id=payload.user_id,
        language=payload.language,
        title=payload.title,
        code=payload.code,
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    return {
        "id": str(row.id),
        "language": payload.language,
        "title": payload.title,
    }
