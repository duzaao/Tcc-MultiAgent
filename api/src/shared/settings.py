from pydantic import BaseModel
import os

class Settings(BaseModel):
    mongodb_uri: str = os.getenv("MONGODB_URI", "mongodb+srv://dudufp07:q9mKByGEjKGvMTQg@mcp.xtj1car.mongodb.net/authsvc?retryWrites=true&w=majority")
    mongodb_db: str = os.getenv("MONGODB_DB", "authsvc")
    jwt_secret: str = os.getenv("JWT_SECRET", "change-me")
    jwt_issuer: str = os.getenv("JWT_ISSUER", "authsvc")
    jwt_audience: str = os.getenv("JWT_AUDIENCE", "api")
    access_ttl_s: int = int(os.getenv("ACCESS_TOKEN_TTL_SECONDS", "900"))
    refresh_ttl_s: int = int(os.getenv("REFRESH_TOKEN_TTL_SECONDS", "2592000"))
    cors_allow_origins: str = os.getenv("CORS_ALLOW_ORIGINS", "*")

settings = Settings()
