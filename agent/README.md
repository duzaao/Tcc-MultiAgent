# Agent Service

Multi-agent orchestrator with planner, FAQ, and MCP tool execution.

## Run

```bash
# Recommended: use start script
./start.sh

# Or directly
python -m multi.agent_main --api
```

## Modes

**API mode** (default in start.sh):
```bash
python -m multi.agent_main --api
```

**Interactive CLI**:
```bash
python -m multi.agent_main
```

**Batch evaluation**:
```bash
python -m multi.agent_main \
  --questions /path/to/questions.jsonl \
  --output /path/to/results.jsonl \
  --llm-list "provider:model,provider:model" \
  --exit-after-batch
```

## Key Files

- `start.sh` - Container entry point (starts API mode)
- `multi/agent_main.py` - Entry point & orchestration
- `multi/agent_plan.py` - Planner/router agent
- `multi/agent_mcp.py` - MCP tool executor
- `multi/llm.py` - LLM provider wrapper
- `company_faq.md` - FAQ content

## Environment

Requires in `.env`: `LLM_PROVIDER`, `MODEL`, API keys (e.g., `OPENAI_API_KEY`)
