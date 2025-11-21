from datetime import datetime, timezone
from typing import Optional
from fastapi import HTTPException, Header
from jose import jwt, JWTError
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from .settings import settings

# Cliente MongoDB compartilhado
client = AsyncIOMotorClient(settings.mongodb_uri)
db = client[settings.mongodb_db]

async def current_user(authorization: Optional[str] = Header(None)):
    """Valida token de acesso e retorna usuário atual"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"], audience=settings.jwt_audience)
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    if payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Wrong token type")
    uid = payload.get("sub")
    users = db["users"]
    user = await users.find_one({"_id": ObjectId(uid), "status": "active"})
    if not user:
        raise HTTPException(status_code=401, detail="User not found or disabled")
    return user

async def verify_admin_access(authorization: Optional[str] = Header(None)):
    """Verifica se o token pertence a um admin"""
    user = await current_user(authorization)
    if user.get("role") not in ["admin"]:
        raise HTTPException(status_code=403, detail="Admin access required")
    return user

async def verify_customer_service_access(authorization: Optional[str] = Header(None)):
    """Verifica se o token pertence a um usuário com permissões de customer service"""
    user = await current_user(authorization)
    if user.get("role") not in ["admin", "customer_service"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    return user
