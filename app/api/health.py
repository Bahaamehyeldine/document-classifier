from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.domain.user import UserCountResponse
from app.services import user_service

router = APIRouter()


@router.get("/health/db", response_model=UserCountResponse)
def health_db(db: Session = Depends(get_db)) -> UserCountResponse:
    return user_service.get_user_count(db)
