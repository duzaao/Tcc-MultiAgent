import time
import secrets
from datetime import datetime, timedelta, timezone
from jose import jwt
from argon2 import PasswordHasher

ph = PasswordHasher(time_cost=3, memory_cost=64*1024, parallelism=2)  # Argon2id default

def hash_password(pw: str) -> str:
    return ph.hash(pw)

def verify_password(pw: str, pw_hash: str) -> bool:
    try:
        return ph.verify(pw_hash, pw)
    except Exception:
        return False

def mint_access_token(sub: str, secret: str, issuer: str, audience: str, ttl_s: int):
    now = datetime.now(timezone.utc)
    exp = now + timedelta(seconds=ttl_s)
    payload = {
        "iss": issuer,
        "sub": sub,
        "aud": audience,
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
        "type": "access"
    }
    token = jwt.encode(payload, secret, algorithm="HS256")
    return token, exp

def new_refresh_token_bytes(n=64) -> bytes:
    # cryptographically strong random bytes
    return secrets.token_bytes(n)
