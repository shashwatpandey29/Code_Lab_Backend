from fastapi import APIRouter

router = APIRouter()


@router.get("/me")
def get_current_user() -> dict[str, str]:
    return {"id": "demo-user", "name": "CodeLab Learner"}
