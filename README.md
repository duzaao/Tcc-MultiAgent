# Multi-Agent Conversational System with Model Context Protocol (MCP)

A secure, modular multi-agent system built with the Model Context Protocol (MCP) to enable safe and auditable interactions between language models and external services. This project demonstrates how intelligent agents can be deployed in enterprise environments while maintaining strong security boundaries and operational control.

## 📋 Overview

This thesis explores the development of a conversational AI system where dedicated agents collaborate to interpret user requests and execute actions through a standardized protocol. By leveraging MCP, all operations remain transparent, auditable, and constrained to explicitly defined tools—eliminating the security risks of unrestricted approaches like direct Text-to-SQL.

The system uses a **multi-agent orchestration layer** where:
- A **Planner Agent** routes requests to the appropriate handler
- A **Tool Executor (MCP Agent)** performs actions through secure, isolated tools
- An **FAQ Agent** handles general queries and company information

Both **local LLMs** (via Ollama) and **API-based models** (OpenAI, Groq, etc.) are supported, enabling evaluation across different computational and privacy constraints.

## 🎯 Research Questions

**RQ1:** How can a modular and reliable multi-agent architecture be designed using MCP?
- Focus: agent organization, MCP's role in tool invocation, maintaining security boundaries

**RQ2:** What are the performance trade-offs between local and API-based LLMs in MCP-driven systems?
- Focus: latency, accuracy, token usage, operational characteristics

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│        User / Frontend Interface            │
└──────────────────┬──────────────────────────┘
                   │
        ┌──────────▼───────────┐
        │   Agent Orchestrator │
        │  (Multi-Agent Layer) │
        └──────────┬───────────┘
                   │
        ┌──────────┴────────────────────┐
        │                               │
    ┌───▼────┐                ┌─────────▼────┐
    │ Planner│                │  FAQ/Memory  │
    │ Agent  │                │   Agent      │
    └───┬────┘                └──────────────┘
        │
        │ (routes to)
        │
    ┌───▼─────────────────┐
    │   MCP Agent         │
    │  (Tool Executor)    │
    └───┬─────────────────┘
        │
    ┌───▼──────────────────────────┐
    │   MCP Server                 │
    │  (Tool Definitions & Calls)  │
    └───┬──────────────────────────┘
        │
    ┌───┴───────┬──────────┬──────────────┐
    │           │          │              │
┌───▼──┐   ┌────▼──┐   ┌───▼──┐    ┌──────▼──┐
│Auth  │   │Flights│   │Login │    │ External│
│API   │   │API    │   │Tools │    │APIs     │
└──────┘   └───────┘   └──────┘    └─────────┘
```

## 📁 Project Structure

```
Tcc-MultiAgent/
├── README.md                    # This file
├── .env.example                 # Environment template (no credentials)
├── .env                         # Local configuration (not committed)
│
├── agent/                       # Multi-agent orchestrator
│   ├── agent.py                 # Main entry point
│   ├── company_faq.md           # FAQ content
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── start.sh
│   └── multi/
│       ├── agent_main.py        # Orchestrator entry point
│       ├── agent_mcp.py         # MCP tool executor
│       ├── agent_plan.py        # Router/planner agent
│       └── llm.py               # LLM provider abstraction
│
├── api/                         # Backend services
│   ├── src/
│   │   ├── auth/                # Authentication service
│   │   ├── flights/             # Flight management service
│   │   └── shared/              # Utilities
│   ├── scripts/run_services.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── start.sh
│
├── mcp/                         # MCP server (tool definitions)
│   ├── server_new.py            # MCP server implementation
│   ├── mcp.json                 # Tool configuration
│   ├── requirements.txt
│   └── Dockerfile
│
├── front/                       # Web frontend (optional)
│   ├── index.html
│   ├── app.js
│   └── style.css
│
├── infra/                       # Deployment & orchestration
│   ├── docker-compose.yml       # Multi-container setup
│   ├── startup.sh               # Service orchestration
│   └── test_stack.py            # Integration tests
│
├── terraform/                   # Infrastructure as Code (AWS/GCP/Azure)
│   ├── main.tf
│   ├── provider.tf
│   ├── variables.tf
│   ├── outputs.tf
│   └── terraform.tfvars
│
├── scripts/                     # Utility scripts
│   ├── analyze.py               # Evaluation analyzer (Groq LLM)
│   └── analyze2.py              # Category-based analysis
│
├── data/                        # Test datasets
│   └── questions.jsonl          # Evaluation prompts
│
└── docs/                        # Documentation
    ├── ARCHITECTURE.md          # Detailed system design
    ├── SETUP.md                 # Installation & configuration
    └── API.md                   # API reference
```

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose (for containerized setup)
- Python 3.11+
- LLM Provider credentials (OpenAI, Groq, etc.) OR local Ollama

### Setup with Docker Compose (Recommended)

1. **Clone and configure:**
   ```bash
   git clone <repo>
   cd Tcc-MultiAgent
   cp .env.example .env
   # Edit .env with your LLM credentials and settings
   ```

2. **Start all services:**
   ```bash
   cd infra
   docker-compose -f docker-compose.yml up -d --build
   ```

   This starts:
   - MongoDB (data persistence)
   - API services (Auth & Flights)
   - Agent (multi-agent orchestrator with MCP)

   **Alternative:** Use `./startup.sh` for sequential startup with health checks.

3. **Access the system:**
   - Agent API: `http://localhost:8000`
   - Auth service: `http://localhost:8001`
   - Flights service: `http://localhost:8002`
   - MCP server: `http://localhost:8003` (internal, spawned by agent)

### Cloud Deployment

For AWS, GCP, or Azure deployment, use the Terraform configuration:

```bash
cd terraform
terraform init
terraform plan
terraform apply
```

See `terraform/` folder for infrastructure as code setup.


## 🔧 Configuration

### Environment Variables (.env)

**LLM Configuration:**
```bash
# Choose provider: openai, groq, ollama, anthropic, etc.
LLM_PROVIDER=openai
MODEL=gpt-4o-mini

# API Keys (if using external providers)
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...

# For local LLM via Ollama
OLLAMA_HOST=http://localhost:11434
# MODEL=llama3.2:3b (for local models)
```

**Database:**
```bash
MONGODB_URI=mongodb://localhost:27017/authsvc
MONGODB_DB=authsvc
```

**Security:**
```bash
JWT_SECRET=your-secret-key-change-this
JWT_ISSUER=authsvc
JWT_AUDIENCE=api
```

**Pricing Estimation (Optional):**
```bash
LLM_INPUT_COST_PER_1K=0.00150   # $ per 1K tokens
LLM_OUTPUT_COST_PER_1K=0.00600  # $ per 1K tokens
```

### Using Local LLM (Ollama)

To use local models instead of API calls:

1. **Install Ollama:** https://ollama.ai

2. **Pull a model:**
   ```bash
   ollama pull llama3.2:3b
   ```

3. **Enable in docker-compose.yml:**
   - Uncomment the `ollama` service in `infra/docker-compose.yml`
   - Update `.env`: `LLM_PROVIDER=ollama` and `MODEL=llama3.2:3b`

4. **Restart services:**
   ```bash
   docker-compose down
   ./startup.sh
   ```




## 🔐 Security & MCP

### Key Security Features

1. **Isolated Tool Execution**: All operations through explicitly defined MCP tools
2. **No Direct Database Access**: SQL/database access forbidden; only through APIs
3. **Audit Trail**: Every tool invocation is logged and traceable
4. **Schema Validation**: Tool parameters validated before execution
5. **Authentication**: Login required for sensitive operations
6. **Session Management**: Stateful connections with timeout handling

### MCP Tools Available

- `login` - User authentication
- `get_flights` - Query available flights
- `get_tickets` - Retrieve user's tickets
- `buy_ticket` - Purchase a flight
- `cancel_ticket` - Cancel an existing ticket
- `get_flight_details` - Flight information

See `mcp/mcp.json` for complete tool definitions.

## 📈 Supported LLM Providers

| Provider | Model Examples | Notes |
|----------|---|---|
| **OpenAI** | gpt-4o-mini, gpt-4-turbo | Fastest, most capable, paid |
| **Groq** | llama-3.1-8b, qwen-32b | Fast inference, free tier available |
| **Anthropic** | claude-3-haiku | Good reasoning, paid |
| **Local (Ollama)** | llama3.2:3b, llama3:8b | Free, private, resource-intensive |

## 📝 JSONL Format

Test questions file format:

```jsonl
{"id": 1, "prompt": "What flights are available from São Paulo?", "category": "query"}
{"id": 2, "prompt": "Book flight ZZ999", "category": "action"}
{"id": 3, "prompt": "What's your refund policy?", "category": "faq"}
```

Expected fields:
- `id` - Unique identifier
- `prompt` - User question/request
- `category` - Classification (query, action, faq, login_auth, etc.)

## 🗂️ Important Files

| File | Purpose |
|------|---------|
| `agent/multi/agent_main.py` | Main orchestrator entry point |
| `agent/multi/agent_plan.py` | Planner agent (routing logic) |
| `agent/multi/agent_mcp.py` | MCP agent (tool executor) |
| `agent/multi/llm.py` | LLM provider wrapper |
| `mcp/server_new.py` | MCP server implementation |
| `mcp/mcp.json` | Tool definitions & configuration |
| `api/src/auth/service.py` | Authentication logic |
| `api/src/flights/service.py` | Flight management logic |
| `infra/docker-compose.yml` | Multi-container orchestration |



## 📊 Performance Characteristics

### Latency (approximate, varies by model/network)

| Component | Time |
|-----------|------|
| Planner routing | 100-500ms |
| Tool execution | 500-2000ms |
| API calls | 200-800ms |
| **Total end-to-end** | **1000-4000ms** |

### Token Usage

Typically 1500-2500 tokens per interaction (prompt + response)

### Cost Example

Using GPT-4o-mini at $0.0015/1K input, $0.006/1K output:
- ~$0.004-0.010 per user interaction
- 1000 interactions = ~$4-10

## 🤝 Contributing

This is a research/thesis project. For contributions:
1. Ensure code follows project style
2. Add tests for new features
3. Update documentation
4. Test with multiple LLM providers



## 🔍 Key Takeaways

1. **MCP enables secure multi-agent systems** by constraining operations to explicitly defined tools
2. **Local LLMs offer privacy** but trade performance for cost
3. **Multi-agent orchestration improves modularity** through specialized components
4. **Enterprise AI requires strong boundaries** — unrestricted tool access is a security risk

---

**Last Updated:** December 2025  
**Status:** Active Development  
**Python Version:** 3.11+  
**Main Dependencies:** LangChain, FastAPI, MongoDB, MCP Protocol