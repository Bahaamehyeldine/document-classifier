from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import User


def count_users(db: Session) -> int:
    return db.scalar(select(func.count()).select_from(User))
