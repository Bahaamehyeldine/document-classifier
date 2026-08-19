from pydantic import BaseModel


class UserCountResponse(BaseModel):
    count: int
