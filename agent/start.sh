#!/bin/sh
set -e

# The agent runs and spawns the MCP subprocess according to /mcp/mcp.json
# No HTTP wait is necessary when using the stdio/command transport.

export PYTHONPATH="/app:$PYTHONPATH"

echo "Starting multi-agent orchestrator (will spawn MCP subprocess via /app/mcp/mcp.json if configured)"
#exec python -m multi.agent_main --questions ../../questions.jsonl --output /out/multi_eval --llm-list "groq:llama-3.1-8b-instant,groq:qwen/qwen3-32b, groq:llama-3.3-70b-versatile,openai:gpt-4o-mini,openai:gpt-4-turbo-2024-04-09, openai:gpt-5-mini-2025-08-07" --exit-after-batch
exec python -m multi.agent_main --questions ../../questions.jsonl --output /out/multi_eval --exit-after-batch


#exec python -m agent.py --api
