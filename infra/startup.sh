#!/bin/sh
# Simple startup orchestrator for the deploy/docker-compose setup.
# Runs on the host where the docker-compose is located (deploy/infra).

set -e

ROOT_DIR=$(cd "$(dirname "$0")"/.. && pwd)
COMPOSE_FILE="$ROOT_DIR/infra/docker-compose.yml"

if [ ! -f "$COMPOSE_FILE" ]; then
  echo "docker-compose.yml not found at $COMPOSE_FILE"
  exit 1
fi

echo "Loading environment variables from $ROOT_DIR/.env"
if [ -f "$ROOT_DIR/.env" ]; then
  # shellcheck disable=SC1090
  . "$ROOT_DIR/.env"
fi

cd "$ROOT_DIR/infra"

echo "Starting MongoDB..."
docker-compose -f docker-compose.yml up -d mongo
echo "Sleeping 5s to let mongo initialize..."
sleep 5

echo "Starting API service..."
docker-compose -f docker-compose.yml up -d api

echo "Waiting for API to respond (trying /openapi.json on 8001/8002)..."
until curl -sSf http://localhost:8001/openapi.json >/dev/null 2>&1 || curl -sSf http://localhost:8002/openapi.json >/dev/null 2>&1; do
  echo "API not ready yet..."
  sleep 2
done

echo "API is up. Starting MCP..."
docker-compose -f docker-compose.yml up -d mcp

echo "Waiting for MCP (http://localhost:8003)..."
until curl -sSf http://localhost:8003/ >/dev/null 2>&1 || curl -sSf http://localhost:8003/openapi.json >/dev/null 2>&1; do
  echo "MCP not ready yet..."
  sleep 2
done

echo "MCP is up. Checking LLM env for agent start..."
if [ -z "$MODEL" ] || [ -z "$LLM_PROVIDER" ]; then
  echo "MODEL or LLM_PROVIDER not set. Agent will not be started automatically."
  echo "To start agent when ready run: docker-compose -f docker-compose.yml up -d agent"
  exit 0
fi

echo "Starting Agent..."
docker-compose -f docker-compose.yml up -d agent

echo "All services started."
