import base64
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, status, Header, Request
from fastapi import Body

from fastapi.middleware.cors import CORSMiddleware
from jose import jwt, JWTError
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from ..shared.settings import settings
from .security import hash_password, verify_password, mint_access_token, new_refresh_token_bytes
from .models import RegisterIn, LoginIn, TokenPairOut, MeOut, DeleteIn

# Modelo para alteração de senha
from pydantic import BaseModel


# Novo modelo: username, email, new_password
class ChangePasswordIn(BaseModel):
    username: str
    email: str
    new_password: str

app = FastAPI(title="Auth Service")



app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_allow_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = AsyncIOMotorClient(settings.mongodb_uri)
db = client[settings.mongodb_db]
users = db["users"]
sessions = db["sessions"]
audit = db["audit_logs"]

def epoch(dt: datetime) -> int:
    return int(dt.timestamp())

def hash_refresh_token(token_bytes: bytes) -> str:
    # store hashed refresh token (base64-encoded SHA-256)
    return base64.b64encode(hashlib.sha256(token_bytes).digest()).decode()

async def current_user(authorization: Optional[str] = Header(None)):
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
    user = await users.find_one({"_id": ObjectId(uid), "status": "active"})
    if not user:
        raise HTTPException(status_code=401, detail="User not found or disabled")
    return user

@app.post("/auth/register", response_model=MeOut, status_code=201)
async def register(body: RegisterIn, request: Request):
    now = datetime.now(timezone.utc)
    pw_hash = hash_password(body.password)
    doc = {
        "username": body.username,
        "email": body.email,
        "passwordHash": pw_hash,
        "role": "customer",
        "status": "active",
        "createdAt": now,
        "updatedAt": now,
    }
    try:
        res = await users.insert_one(doc)
    except Exception as e:
        # likely duplicate key error
        raise HTTPException(status_code=409, detail=f"{e}")
    uid = str(res.inserted_id)
    await audit.insert_one({"userId": ObjectId(uid), "action": "register", "createdAt": now,
                            "ip": request.client.host if request.client else None})
    return MeOut(id=uid, username=body.username, email=body.email, role="customer", status="active")

@app.post("/auth/login", response_model=TokenPairOut)
async def login(body: LoginIn, request: Request):
    user = await users.find_one({"username": body.username})
    if not user or not verify_password(body.password, user["passwordHash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if user.get("status") != "active":
        raise HTTPException(status_code=403, detail="User disabled")

    # Issue access token
    access, access_exp = mint_access_token(str(user["_id"]), settings.jwt_secret,
                                           settings.jwt_issuer, settings.jwt_audience,
                                           settings.access_ttl_s)

    # Create refresh token (random bytes), store a hash in DB
    refresh_bytes = new_refresh_token_bytes()
    refresh_b64 = base64.b64encode(refresh_bytes).decode()  # send to client
    refresh_hash = hash_refresh_token(refresh_bytes)
    now = datetime.now(timezone.utc)
    refresh_exp = now + timedelta(seconds=settings.refresh_ttl_s)

    session_doc = {
        "userId": user["_id"],
        "refreshTokenHash": refresh_hash,
        "userAgent": request.headers.get("user-agent"),
        "ip": request.client.host if request.client else None,
        "createdAt": now,
        "expiresAt": refresh_exp,
        "revoked": False,
        "replacedBySessionId": None
    }
    await sessions.insert_one(session_doc)
    await audit.insert_one({"userId": user["_id"], "action": "login", "createdAt": now,
                            "ip": request.client.host if request.client else None})

    return TokenPairOut(
        access_token=access,
        access_token_expires_at=epoch(access_exp),
        refresh_token=refresh_b64,
        refresh_token_expires_at=epoch(refresh_exp),
    )

# Endpoint público para alteração de senha sem autenticação
@app.post("/auth/change-password")
async def change_password(body: ChangePasswordIn):
    """
    Permite ao usuário alterar sua senha informando username, email e nova senha.
    """
    user = await users.find_one({"username": body.username})
    if not user:
        raise HTTPException(status_code=404, detail="Usuário não encontrado")
    if user.get("email") != body.email:
        raise HTTPException(status_code=403, detail="Email não confere para este usuário")
    new_hash = hash_password(body.new_password)
    await users.update_one({"_id": user["_id"]}, {"$set": {"passwordHash": new_hash, "updatedAt": datetime.now(timezone.utc)}})
    await audit.insert_one({"userId": user["_id"], "action": "change_password", "createdAt": datetime.now(timezone.utc)})
    return {"ok": True}


@app.post("/auth/refresh", response_model=TokenPairOut)
async def refresh(request: Request, authorization: Optional[str] = Header(None)):
    # Expect refresh token in Authorization: Bearer <token> for simplicity
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing refresh token")
    refresh_b64 = authorization.split(" ", 1)[1]
    try:
        refresh_bytes = base64.b64decode(refresh_b64.encode())
    except Exception:
        raise HTTPException(status_code=400, detail="Malformed refresh token")

    refresh_hash = hash_refresh_token(refresh_bytes)
    now = datetime.now(timezone.utc)
    session = await sessions.find_one({"refreshTokenHash": refresh_hash})
    if not session or session.get("revoked") or session.get("expiresAt") <= now:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")

    user = await users.find_one({"_id": session["userId"], "status": "active"})
    if not user:
        raise HTTPException(status_code=401, detail="User not found or disabled")

    # Rotate: revoke old, issue new session
    await sessions.update_one({"_id": session["_id"]}, {"$set": {"revoked": True}})

    new_refresh_bytes = new_refresh_token_bytes()
    new_refresh_b64 = base64.b64encode(new_refresh_bytes).decode()
    new_refresh_hash = hash_refresh_token(new_refresh_bytes)
    new_exp = now + timedelta(seconds=settings.refresh_ttl_s)
    new_sess = {
        "userId": user["_id"],
        "refreshTokenHash": new_refresh_hash,
        "userAgent": request.headers.get("user-agent"),
        "ip": request.client.host if request.client else None,
        "createdAt": now,
        "expiresAt": new_exp,
        "revoked": False,
        "replacedBySessionId": None
    }
    insert_res = await sessions.insert_one(new_sess)
    await sessions.update_one({"_id": session["_id"]}, {"$set": {"replacedBySessionId": insert_res.inserted_id}})

    access, access_exp = mint_access_token(str(user["_id"]), settings.jwt_secret,
                                           settings.jwt_issuer, settings.jwt_audience,
                                           settings.access_ttl_s)

    return TokenPairOut(
        access_token=access,
        access_token_expires_at=epoch(access_exp),
        refresh_token=new_refresh_b64,
        refresh_token_expires_at=epoch(new_exp),
    )

@app.post("/auth/logout")
async def logout(request: Request, authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing refresh token")
    refresh_b64 = authorization.split(" ", 1)[1]
    try:
        refresh_bytes = base64.b64decode(refresh_b64.encode())
    except Exception:
        raise HTTPException(status_code=400, detail="Malformed refresh token")
    refresh_hash = hash_refresh_token(refresh_bytes)
    sess = await sessions.find_one({"refreshTokenHash": refresh_hash})
    if not sess:
        # idempotent
        return {"ok": True}
    await sessions.update_one({"_id": sess["_id"]}, {"$set": {"revoked": True}})
    await audit.insert_one({"userId": sess["userId"], "action": "logout", "createdAt": datetime.now(timezone.utc)})
    return {"ok": True}

@app.get("/auth/me", response_model=MeOut)
async def me(user=Depends(current_user)):
    return MeOut(
        id=str(user["_id"]),
        username=user["username"],
        email=user.get("email"),
        role=user.get("role", "customer"),
        status=user.get("status", "active"),
    )

@app.delete("/auth/account")
async def delete_account(body: DeleteIn, user=Depends(current_user)):
    if not verify_password(body.password, user["passwordHash"]):
        raise HTTPException(status_code=403, detail="Password mismatch")
    await users.delete_one({"_id": user["_id"]})
    # revoke all sessions
    await sessions.update_many({"userId": user["_id"], "revoked": False}, {"$set": {"revoked": True}})
    await audit.insert_one({"userId": user["_id"], "action": "delete_account", "createdAt": datetime.now(timezone.utc)})
    return {"ok": True}
