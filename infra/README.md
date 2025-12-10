# Infrastructure

Docker Compose orchestration and deployment scripts.

## Start All Services

```bash
./startup.sh
```

## Manual Control

```bash
docker-compose -f docker-compose.yml up -d --build      # Start all
docker-compose down       # Stop all
docker-compose logs -f    # View logs
```

## Services

- MongoDB (27017)
- API (8001, 8002)
- Agent (8000)
- Optional: Ollama (11434)

## Configuration

Edit `docker-compose.yml` and `../.env`
