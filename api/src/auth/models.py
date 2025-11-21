from pydantic import BaseModel, Field, EmailStr, constr
from typing import Optional

class RegisterIn(BaseModel):
    username: constr(min_length=3, max_length=64)
    password: constr(min_length=8, max_length=128)
    email: Optional[EmailStr] = None

class LoginIn(BaseModel):
    username: str
    password: str

class TokenPairOut(BaseModel):
    access_token: str
    access_token_expires_at: int  # epoch seconds
    refresh_token: str
    refresh_token_expires_at: int  # epoch seconds
    token_type: str = "Bearer"

class MeOut(BaseModel):
    id: str
    username: str
    email: Optional[str] = None
    role: str
    status: str

class DeleteIn(BaseModel):
    password: str
