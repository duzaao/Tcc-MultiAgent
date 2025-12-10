# API Services

FastAPI backend for Auth (8001) and Flights (8002).

## Run

```bash
./start.sh
```

Or directly:
```bash
python scripts/run_services.py
```

## Structure

- `src/auth/` - Authentication & user management
- `src/flights/` - Flight booking & queries
- `src/shared/` - Common utilities

## Environment

Requires: `MONGODB_URI`, `JWT_SECRET` in `.env`
