from sqlalchemy.orm import Session

from app.domain.user import UserCountResponse
from app.repositories import user_repository


def get_user_count(db: Session) -> UserCountResponse:
    count = user_repository.count_users(db)
    return UserCountResponse(count=count)
