# --- IMPORTS E CONFIG FASTAPI NO TOPO ---
from pathlib import Path
import json
import asyncio
import inspect
import sys
import argparse
import time
import concurrent.futures
from typing import Optional
import os
import logging
import traceback
import warnings
import requests
from dotenv import load_dotenv

try:
    from groq import Groq
    GROQ_SDK_AVAILABLE = True
except Exception:
    Groq = None
    GROQ_SDK_AVAILABLE = False

warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"langchain.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"langchain_core.*")

# API
try:
    from fastapi import FastAPI
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except Exception:
    FASTAPI_AVAILABLE = False
else:
    try:
        from fastapi.staticfiles import StaticFiles
        from fastapi.middleware.cors import CORSMiddleware
        HAS_STATIC = True
    except Exception:
        HAS_STATIC = False
# Funções auxiliares e MCP do agent4.py
def _run_coro_sync(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        import threading
        result = {}
        def _target():
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                result['value'] = loop.run_until_complete(coro)
            except Exception as e:
                result['exc'] = e
            finally:
                try:
                    loop.close()
                except Exception:
                    pass
        t = threading.Thread(target=_target)
        t.start()
        t.join()
        if 'exc' in result:
            raise result['exc']
        return result.get('value')

async def _load_tools_for_server_from_mcp(mcp_path: Path, server_name: str):
    with open(mcp_path, "r", encoding="utf-8") as f:
        mcp_cfg = json.load(f)
    servers = mcp_cfg.get("servers", {})
    if server_name not in servers:
        raise KeyError(f"Server '{server_name}' not found in {mcp_path}")
    client_cfg = {}
    for name, s in servers.items():
        entry = {}
        if "command" in s:
            entry["transport"] = "stdio"
            entry["command"] = s.get("command")
            if "args" in s:
                entry["args"] = s.get("args")
            if "env" in s:
                entry["env"] = s.get("env")
        elif "url" in s:
            entry["transport"] = "streamable_http"
            entry["url"] = s.get("url")
            if "headers" in s:
                entry["headers"] = s.get("headers")
        else:
            continue
        client_cfg[name] = entry
    client = MultiServerMCPClient(client_cfg)
    try:
        async with client.session(server_name) as session:
            tools = await load_mcp_tools(session)
            for t in tools:
                t.name = f"mcp_{server_name}_{t.name}"
            return tools
    finally:
        if hasattr(client, 'aclose'):
            try:
                await client.aclose()
            except Exception:
                pass
        elif hasattr(client, 'close'):
            try:
                client.close()
            except Exception:
                pass

def _mask_args_for_print(obj):
    sensitive = {'password', 'payment_token', 'token', 'access_token'}
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k.lower() in sensitive:
                try:
                    s = str(v)
                    out[k] = s[:4] + '...***masked***' if s else '***masked***'
                except Exception:
                    out[k] = '***masked***'
            else:
                out[k] = _mask_args_for_print(v)
        return out
    if isinstance(obj, list):
        return [_mask_args_for_print(v) for v in obj]
    return obj

def _build_client_cfg_from_mcp(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        mcp_cfg = json.load(f)
    servers = mcp_cfg.get('servers', {})
    client_cfg = {}
    for name, s in servers.items():
        entry = {}
        if 'command' in s:
            entry['transport'] = 'stdio'
            entry['command'] = s.get('command')
            if 'args' in s:
                entry['args'] = s.get('args')
            if 'env' in s:
                entry['env'] = s.get('env')
        elif 'url' in s:
            entry['transport'] = 'streamable_http'
            entry['url'] = s.get('url')
            if 'headers' in s:
                entry['headers'] = s.get('headers')
        else:
            continue
        client_cfg[name] = entry
    return client_cfg

async def _call_mcp_tool_async(tool_name: str, args: dict, mcp_path: Path) -> str:
    client_cfg = _build_client_cfg_from_mcp(mcp_path)
    client = MultiServerMCPClient(client_cfg)
    try:
        async with client.session(SERVER_NAME) as session:
            tools = await load_mcp_tools(session)
            tool_lookup = {}
            for t in tools:
                orig = t.name
                prefixed = f"mcp_{SERVER_NAME}_{orig}"
                tool_lookup[orig.lower()] = t
                tool_lookup[prefixed.lower()] = t
                short = orig.split('_')[-1]
                tool_lookup[short.lower()] = t
            tn = tool_name.lower() if isinstance(tool_name, str) else ''
            target = None
            if tn and tn in tool_lookup:
                target = tool_lookup[tn]
            if target is None:
                for t in tools:
                    if tn and (tn == t.name.lower() or tn in t.name.lower()):
                        target = t
                        break
            if target is None:
                for t in tools:
                    short = t.name.split('_')[-1]
                    if tn == short.lower() or (tn and tn in short.lower()):
                        target = t
                        break
            if not target:
                raise RuntimeError(f"Tool '{tool_name}' not found in MCP server ({[t.name for t in tools]})")
            if callable(getattr(session, 'call_tool', None)):
                try:
                    return await session.call_tool(target.name, args or {})
                except Exception:
                    pass
            if callable(getattr(target, 'arun', None)):
                return await target.arun(args or {})
            if callable(getattr(target, 'ainvoke', None)):
                return await target.ainvoke(args or {})
            if callable(getattr(target, 'run', None)):
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, lambda: target.run(args or {}))
            if callable(getattr(target, 'invoke', None)):
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, lambda: target.invoke(args or {}))
            func = getattr(target, 'func', None)
            if callable(func):
                if inspect.iscoroutinefunction(func):
                    return await func(**(args or {}))
                else:
                    loop = asyncio.get_running_loop()
                    return await loop.run_in_executor(None, lambda: func(**(args or {})))
            raise RuntimeError('Tool found but no execution method available.')
    finally:
        if hasattr(client, 'aclose'):
            try:
                await client.aclose()
            except Exception:
                pass
        elif hasattr(client, 'close'):
            try:
                client.close()
            except Exception:
                pass

def _call_mcp_tool_sync(tool_name: str, args: dict, mcp_path: Path) -> str:
    try:
        masked = _mask_args_for_print(args or {})
    except Exception:
        masked = args or {}
    try:
        print(f"--> MCP CALL: {tool_name} args: {json.dumps(masked, ensure_ascii=False)}")
    except Exception:
        print(f"--> MCP CALL: {tool_name} args: {masked}")
    result = _run_coro_sync(_call_mcp_tool_async(tool_name, args, mcp_path))
    try:
        if isinstance(result, str):
            parsed = json.loads(result)
            pretty = json.dumps(parsed, indent=2, ensure_ascii=False)
        else:
            pretty = json.dumps(result, indent=2, ensure_ascii=False)
    except Exception:
        pretty = str(result)
    print(f"<-- MCP RETURN from {tool_name}:\n{pretty}")
    return result

if FASTAPI_AVAILABLE:
    class MessageRequest(BaseModel):
        message: str
        confirm_sensitive: bool = False

    def create_app(agent_callable):
        app = FastAPI()

        # Enable CORS during development to accept requests from static frontend
        try:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        except Exception:
            pass

        # mount static files if front folder exists and StaticFiles available
        try:
            if HAS_STATIC:
                import os
                base = Path(__file__).resolve().parents[1] / 'front'
                if base.exists():
                    # mount at /front to not intercept API routes
                    app.mount('/front', StaticFiles(directory=str(base), html=True), name='front')
                    # serve index.html at /
                    try:
                        from fastapi.responses import FileResponse

                        @app.get('/')
                        def root_index():
                            return FileResponse(str(base / 'index.html'))
                    except Exception:
                        pass
        except Exception:
            pass

        @app.get("/health")
        def health():
            return {"status": "ok"}
        
        @app.post("/message")
        def post_message(req: MessageRequest):
            # initial agent call
            try:
                out = agent_callable.run(req.message)
            except Exception as e:
                return {"error": str(e)}

            # detect simple confirmation prompt
            if isinstance(out, str) and out.lower().startswith("do you confirm") and req.confirm_sensitive:
                # send automatic confirmation
                try:
                    out2 = agent_callable.run("yes")
                    return {"reply": out2, "auto_confirmed": True}
                except Exception as e:
                    return {"reply": out, "auto_confirmed": False, "error": str(e)}

            return {"reply": out, "auto_confirmed": False}

        # make available for import
        create_app.__doc__ = """create_app(agent_callable) -> FastAPI app exposing /health and /message"""
        return app
#!/usr/bin/env python3
"""
Conversational agent with pluggable LLM backend (OpenAI, Llama, DeepSeek, Gemini, etc).

Usage:
  python3 agent/agent.py
"""
from pathlib import Path
import json
import asyncio
import inspect
import sys
from dotenv import load_dotenv
import os
import logging
import traceback
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"langchain.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"langchain_core.*")

# API
try:
    from fastapi import FastAPI
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except Exception:
    FASTAPI_AVAILABLE = False
else:
    try:
        from fastapi.staticfiles import StaticFiles
        from fastapi.middleware.cors import CORSMiddleware
        HAS_STATIC = True
    except Exception:
        HAS_STATIC = False

# LLM selection
LLM_BACKENDS = {}


# OpenAI
try:
    from langchain_community.chat_models import ChatOpenAI
    LLM_BACKENDS['openai'] = ChatOpenAI
except Exception:
    try:
        from langchain.chat_models import ChatOpenAI
        LLM_BACKENDS['openai'] = ChatOpenAI
    except Exception:
        pass
# Llama (exemplo: llama-cpp, llama-api, etc)
try:
    from langchain_community.chat_models import ChatLlama
    LLM_BACKENDS['llama'] = ChatLlama
except Exception:
    pass
# DeepSeek
try:
    from langchain_community.chat_models import ChatDeepSeek
    LLM_BACKENDS['deepseek'] = ChatDeepSeek
except Exception:
    pass
# Gemini (Google)
try:
    from langchain_community.chat_models import ChatGemini
    LLM_BACKENDS['gemini'] = ChatGemini
except Exception:
    pass
# Ollama (local)
# Prefer the standalone langchain-ollama package (newer), fall back to langchain_community
try:
    # recommended import (langchain-ollama package)
    from langchain_ollama import ChatOllama
    LLM_BACKENDS['ollama'] = ChatOllama
except Exception:
    try:
        # older compatibility fallback
        from langchain_community.chat_models import ChatOllama
        LLM_BACKENDS['ollama'] = ChatOllama
    except Exception:
        # Ollama backend not available
        pass

from langchain_core.messages import HumanMessage, SystemMessage

# Memory
try:
    from langchain_community.memory import ConversationBufferWindowMemory
except Exception:
    ConversationBufferWindowMemory = None

# MCP
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools
    MCP_ADAPTERS_AVAILABLE = True
except Exception:
    MultiServerMCPClient = None
    load_mcp_tools = None
    MCP_ADAPTERS_AVAILABLE = False

# LangChain Agent utilities
try:
    from langchain_community.agents import initialize_agent, AgentType
except Exception:
    initialize_agent = None
    AgentType = None

# .env
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL", "gpt-4o-mini")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # openai, llama, deepseek, gemini

# default MCP config
MCP_DEFAULT = Path(__file__).resolve().parents[1] / "mcp" / "mcp.json"
SERVER_NAME = "FlightCustomerService"

# load company FAQ
def open_faq_text() -> str:
    try:
        return (Path(__file__).parent / "company_faq_2.md").read_text(encoding="utf-8")
    except Exception:
        return "Company information not available at the moment."

SYSTEM_PROMPT = """
You are FlightCo's customer support assistant. Provide institutional information and support guidance based on company policy. Use the company_faq.md file as reference.

There are only two types of questions: everyday support questions that can be answered based on the company FAQ, and questions that require access to real-time data or actions in the reservation system (MCP).

Everyday questions:
- Provide clear and concise answers based on the FAQ.
- Do not call MCP tools for simple or generic questions.
- If you don't know the answer, say you don't know.

Decision rules for using MCP tools:
- Call the MCP tool ONLY when the task requires external data/actions or state modifications (e.g.: list flights in real-time, buy/cancel tickets, check personal tickets, process refund, login, create account, logout).
- If the request is ambiguous or partial, ask for clarification before executing.
- If the user hasn't explicitly authorized state modification (e.g.: buy/cancel), ask first.
"""

def get_llm(model=None, provider=None, **kwargs):
    """
    Returns an instance of the selected LLM model.
    - Avoids reloading the model on each call (uses cache)
    - Logs model load time, agent build time, and MCP tools load time

    provider: 'openai', 'llama', 'deepseek', 'gemini', etc.
    model: model name (e.g. 'gpt-4', 'llama-2-7b', etc)
    kwargs: extra parameters (api_key, endpoint, temperature, etc)
    """

    # Setup logger
    logger = logging.getLogger("get_llm")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    provider = (provider or LLM_PROVIDER or 'openai').lower()
    model = model or MODEL

    # Initialize cache
    if not hasattr(get_llm, "_cache"):
        get_llm._cache = {}

    cache_key = f"{provider}:{model}"
    if cache_key in get_llm._cache:
        logger.info(f"Reusing cached LLM instance for provider={provider}, model={model}")
        return get_llm._cache[cache_key]

    logger.info(f"Initializing LLM provider={provider}, model={model}")
    t0 = time.time()

    # ---------- Providers ----------
    if provider == 'openai':
        llm = LLM_BACKENDS['openai'](model=model, openai_api_key=OPENAI_API_KEY, **kwargs)

    elif provider == 'ollama':
        host = kwargs.pop('ollama_host', None) or os.getenv('OLLAMA_HOST')
        if host:
            kwargs.setdefault('base_url', host)
        llm = LLM_BACKENDS['ollama'](model=model, **kwargs)

    elif provider == 'groq':
        api_key = kwargs.pop('api_key', os.getenv('GROQ_API_KEY'))
        api_url = kwargs.pop('api_url', os.getenv('GROQ_API_URL'))
        llm = GroqChatWrapper(model=model, api_key=api_key, api_url=api_url, **kwargs)

    elif provider in LLM_BACKENDS:
        llm = LLM_BACKENDS[provider](model=model, **kwargs)
    else:
        raise ValueError(f"LLM provider '{provider}' is not supported or not installed.")

    load_time = time.time() - t0
    logger.info(f"Model loaded in {load_time:.2f}s")

    # ---------- Optional: Agent and MCP tools ----------
    agent_start = time.time()
    try:
        if "build_agent" in kwargs and callable(kwargs["build_agent"]):
            kwargs["build_agent"](llm)
            agent_time = time.time() - agent_start
            logger.info(f"Agent built in {agent_time:.2f}s")
    except Exception as e:
        logger.warning(f"Agent build failed or skipped: {e}")

    mcp_start = time.time()
    try:
        if "load_mcp_tools" in kwargs and callable(kwargs["load_mcp_tools"]):
            kwargs["load_mcp_tools"]()
            mcp_time = time.time() - mcp_start
            logger.info(f"MCP tools loaded in {mcp_time:.2f}s")
    except Exception as e:
        logger.warning(f"MCP tools load failed or skipped: {e}")

    # Cache the instance
    get_llm._cache[cache_key] = llm
    logger.info(f"LLM instance cached for provider={provider}, model={model}")

    return llm


class GroqChatWrapper:
    """Minimal wrapper for calling a Groq-style HTTP API.

    Implements `invoke(messages)` for compatibility with other LLM backends.
    Reads API URL and key from env vars or kwargs.
    """
    def __init__(self, model: str, api_key: str = None, api_url: str = None, timeout: float = 30, **kwargs):
        self.model = model
        self.api_key = api_key
        self.api_url = api_url
        self.timeout = float(timeout or 30)

        if GROQ_SDK_AVAILABLE:
            try:
                try:
                    self.client = Groq(api_key=api_key) if api_key else Groq()
                except TypeError:
                    self.client = Groq()
            except Exception:
                self.client = None
        else:
            self.client = None

    def _build_prompt(self, messages):
        parts = []
        for m in messages:
            content = getattr(m, 'content', str(m))
            parts.append(content)
        return "\n\n".join(parts)

    def invoke(self, messages):
        import json, time
        prompt = self._build_prompt(messages)
        logger = logging.getLogger("GroqChatWrapper")

        start_time = time.time()

        # Try Groq Python SDK first
        if self.client:
            sdk_msgs = []
            for m in messages:
                role = 'user'
                if getattr(m, '__class__', None) and 'SystemMessage' in str(getattr(m, '__class__')):
                    role = 'system'
                sdk_msgs.append({'role': role, 'content': getattr(m, 'content', str(m))})
            try:
                resp = self.client.chat.completions.create(model=self.model, messages=sdk_msgs)
                try:
                    text = resp.choices[0].message.content
                except Exception:
                    text = getattr(resp.choices[0].message, 'content', str(resp))
                latency = time.time() - start_time
                logger.info(f"Groq SDK response in {latency:.2f}s")
                return type('R', (), {'content': text})()
            except Exception as e:
                raise RuntimeError(f"Error calling Groq SDK: {e}")

        # Fallback: HTTP POST
        if not self.api_url:
            raise RuntimeError('GROQ_API_URL not configured. Set GROQ_API_URL env or pass api_url kwarg.')

        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}' 

        payload = {'input': prompt, 'model': self.model}
        try:
            resp = requests.post(self.api_url, json=payload, headers=headers, timeout=self.timeout)
            try:
                body = resp.json()
            except Exception:
                body = None

            text = None
            if body:
                text = body.get('output') or body.get('generated_text') or body.get('text')
                if not text and 'data' in body and isinstance(body['data'], list) and len(body['data']) > 0:
                    maybe = body['data'][0]
                    if isinstance(maybe, dict):
                        text = maybe.get('text') or maybe.get('output')
            if text is None:
                text = resp.text

            latency = time.time() - start_time
            logger.info(f"Groq HTTP response in {latency:.2f}s")

            return type('R', (), {'content': text})()
        except Exception as e:
            raise RuntimeError(f"Error calling Groq API: {e}")

## --- Helpers para modo batch (JSONL) ---
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except Exception:
    tiktoken = None
    TIKTOKEN_AVAILABLE = False

def count_tokens(text: str, model_name: Optional[str] = None) -> Optional[int]:
    """Tenta contar tokens do texto com tiktoken; se não disponível retorna None."""
    if not text:
        return 0
    if not TIKTOKEN_AVAILABLE:
        return None
    try:
        # tentar inferir encoding do modelo; se não, usar cl100k_base
        if model_name:
            try:
                enc = tiktoken.encoding_for_model(model_name)
            except Exception:
                enc = tiktoken.get_encoding("cl100k_base")
        else:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return None

def process_item(agent_cache: dict, item: dict, default_provider: str, default_model: str, mcp_path: Path, load_tools: bool, timeout: Optional[float]=None) -> dict:
    """Processa um item (dict com id/prompt/provider/model) usando agents cacheados.
    Retorna dict com campos de saída esperados.
    """
    qid = item.get('id') or item.get('q_id') or item.get('key') or None
    prompt = item.get('prompt') or item.get('question') or item.get('q') or ''
    provider = (item.get('provider') or default_provider or 'openai').lower()
    model_name = item.get('model') or default_model or None
    out = {
        'id': qid,
        'provider': provider,
        'model': model_name,
        'prompt': prompt,
        'response': None,
        'tokens_prompt': None,
        'tokens_response': None,
        'price_total': 0,
        'latency_ms': None,
        'error': None,
    }

    key = (provider, model_name)
    if key not in agent_cache:
        try:
            agent_cache[key] = build_agent(load_tools=load_tools, mcp_mcp_path=str(mcp_path), llm_provider=provider, llm_model=model_name)
        except Exception as e:
            out['error'] = f'Error creating agent for {provider}/{model_name}: {e}'
            return out

    agent = agent_cache[key]

    start = time.perf_counter()
    try:
        response = agent.run(prompt)
        end = time.perf_counter()
        latency_ms = int((end - start) * 1000)
        out['response'] = response
        out['latency_ms'] = latency_ms

        # contar tokens (melhor esforço)
        try:
            tp = count_tokens(prompt, model_name)
        except Exception:
            tp = None
        try:
            tr = count_tokens(response, model_name)
        except Exception:
            tr = None
        out['tokens_prompt'] = tp if tp is not None else 0
        out['tokens_response'] = tr if tr is not None else 0

        # preço: por enquanto zero se não soubermos
        out['price_total'] = 0
    except Exception as e:
        end = time.perf_counter()
        out['latency_ms'] = int((end - start) * 1000)
        out['error'] = str(e)
    return out

# ...existing code from agent4.py (funções auxiliares, MCP, etc)...
# Copiaremos o restante do código do agent4.py, mas substituindo a criação do modelo por get_llm()

# --- Funções auxiliares e MCP (iguais ao agent4.py) ---
# ...
# (O restante do código é igual ao agent4.py, exceto a parte de criação do modelo)

# build_agent igual ao build_agent4, mas com seleção de LLM

def build_agent(load_tools: bool = False, mcp_mcp_path: str = None, history_len: int = 6, llm_provider=None, llm_model=None, **llm_kwargs):
    model = get_llm(model=llm_model, provider=llm_provider, **llm_kwargs)
    faq_text = open_faq_text()
    mcp_path = Path(mcp_mcp_path) if mcp_mcp_path else MCP_DEFAULT
    tools_meta = []
    aliases_map = {}
    agent_executor = None
    state = {'current_user_id': None, 'awaiting_confirmation': None}
    memory = ConversationBufferWindowMemory(k=history_len, return_messages=True) if ConversationBufferWindowMemory else None
    if load_tools and MCP_ADAPTERS_AVAILABLE:
        try:
            loaded_tools = _run_coro_sync(_load_tools_for_server_from_mcp(mcp_path, SERVER_NAME))
            print(f"Loaded {len(loaded_tools)} MCP tools from server '{SERVER_NAME}'")
            for t in loaded_tools:
                if not str(t.name).startswith(f"mcp_{SERVER_NAME}_"):
                    t.name = f"mcp_{SERVER_NAME}_{t.name}"
                desc = getattr(t, 'description', '') or ''
                name_lower = t.name.lower()
                extra = ''
                if 'login' in name_lower and 'register' not in name_lower:
                    extra = 'Authenticates a user. Args: username (str), password (str).'
                elif 'register' in name_lower:
                    extra = 'Registers a new user. Args: username (str), password (str), email (str), role (optional).'
                elif 'list' in name_lower and 'flight' in name_lower:
                    extra = 'Lists available flights. No arguments.'
                elif 'buy' in name_lower or 'purchase' in name_lower:
                    extra = 'Makes a purchase. Args: user_id, flight_id, seat_class, payment_token.'
                if extra and extra not in desc:
                    desc = (desc + '\n' + extra).strip()
                tools_meta.append({'name': t.name, 'description': desc})
                try:
                    short = t.name.split(f"mcp_{SERVER_NAME}_", 1)[-1]
                except Exception:
                    short = t.name
                if 'aliases_map' not in locals():
                    aliases_map = {}
                aliases_map[short.lower()] = t.name
                aliases_map[short.replace('-', '_').lower()] = t.name
                aliases_map[short.replace(' ', '_').lower()] = t.name
                if 'login' in short.lower():
                    aliases_map['login'] = t.name
                if 'register' in short.lower():
                    aliases_map['register'] = t.name
                if 'list' in short.lower() and 'flight' in short.lower():
                    aliases_map['list_flights'] = t.name
                    aliases_map['list_available_flights'] = t.name
                if 'buy' in short.lower() or 'purchase' in short.lower():
                    aliases_map['buy'] = t.name
            try:
                aliases_map
            except NameError:
                aliases_map = {}
            agent_executor = None
        except Exception as e:
            logging.exception('Error loading MCP tools: %s', e)
            tools_meta = []
            aliases_map = {}
    def agent_call(query: str):
        q = (query or '').strip()
        if not q:
            return "Please type a message."
        # If we have a pending confirmation, treat short affirmative/negative replies as responses
        if state.get('awaiting_confirmation'):
            pending = state.get('awaiting_confirmation')
            # normalize reply
            reply = q.strip().lower()
            yes_set = {'yes', 'y', 'sim', 's', 'confirm', 'confirmar', 'ok', 'claro', 'sure'}
            no_set = {'no', 'n', 'não', 'nao', 'cancel', 'cancelar'}
            # If reply clearly yes or no, handle it
            if reply in yes_set or reply in no_set:
                # handle negative
                if reply in no_set:
                    state['awaiting_confirmation'] = None
                    return "Action cancelled."
                # handle affirmative
                action = pending.get('action')
                args = pending.get('args') or {}
                orig_q = pending.get('original_q') or ''
                state['awaiting_confirmation'] = None
                try:
                    tool_output = _call_mcp_tool_sync(action, args, mcp_path)
                except Exception as e:
                    logging.exception("Error invoking confirmed MCP tool: %s", e)
                    return f"❌ Error invoking tool '{action}': {e}"
                # build final message after tool output
                final_system = SystemMessage(content=SYSTEM_PROMPT + "\n\nCompany FAQ:\n" + open_faq_text())
                final_human = HumanMessage(content=(
                    f"User question: {orig_q}\n\n"
                    f"Tool invoked: '{action}' with args: {json.dumps(args, ensure_ascii=False)}\n\n"
                    f"Tool output:\n{str(tool_output)}\n\n"
                    "Format the final response to the user in English."
                ))
                msgs2 = [final_system]
                if memory:
                    hist2 = memory.load_memory_variables({}).get("history")
                    if hist2:
                        msgs2.extend(hist2 if isinstance(hist2, list) else [HumanMessage(content=hist2)])
                msgs2.append(final_human)
                try:
                    final_resp = model.invoke(msgs2)
                    final_text = getattr(final_resp, "content", str(final_resp))
                    if memory:
                        try:
                            memory.save_context({"input": orig_q}, {"output": final_text})
                        except Exception:
                            pass
                    return final_text
                except Exception as e:
                    logging.exception("Error generating final response after confirmed tool call: %s", e)
                    return str(tool_output)
        classify_system = SystemMessage(content="""
You are an assistant that decides if a user's message requires the use of MCP tools.

Answer with JUST ONE WORD:
- yes → if it's necessary to use an MCP tool (e.g.: login, logout, list flights, buy, cancel, check tickets, refund, etc.)
- no → if it's a normal conversation, greeting, thanks, generic question or FAQ-answerable question.
""")
        classify_human = HumanMessage(content=f'User message: "{q}"')
        try:
            classify_resp = model.invoke([classify_system, classify_human])
            decision = (classify_resp.content or "").strip().lower()
            print(f"[DEBUG] Model classification: {decision}")
        except Exception:
            decision = "no"
        if not decision.startswith("y"):
            messages = [
                SystemMessage(content=SYSTEM_PROMPT + "\n\nCompany FAQ:\n" + open_faq_text()),
                HumanMessage(content=q)
            ]
            try:
                resp = model.invoke(messages)
                text = getattr(resp, "content", str(resp))
                if memory:
                    try:
                        memory.save_context({"input": q}, {"output": text})
                    except Exception:
                        pass
                return text
            except Exception as e:
                logging.exception("Error answering normal message: %s", e)
                return f"An error occurred while processing your message: {e}"
        if tools_meta:
            tools_list_text = '\n'.join(
                [f"- {t['name']}: {t['description']}" for t in tools_meta]
            )
            system = SystemMessage(content="""
            You are an agent that can call MCP tools listed below IF the user asks for something related to login, logout, flight purchases, available flights, cancellations, refunds, etc. 
            Otherwise, don't call any tool.

            When deciding to call a tool, return JSON:
            {"action":"tool_name","args":{...}} depending on the tool.
            """)
            human = HumanMessage(content=(
                f'Available tools:\n{tools_list_text}\n\n'
                f'User intent: "{q}"\n\n'
                f'Return only the requested JSON, nothing else.'
            ))
            try:
                msgs = [system]
                if memory:
                    hist = memory.load_memory_variables({}).get("history")
                    if hist:
                        msgs.extend(hist if isinstance(hist, list) else [HumanMessage(content=hist)])
                msgs.append(human)
                resp = model.invoke(msgs)
                content = getattr(resp, "content", str(resp))
                js = None
                try:
                    js = json.loads(content)
                except Exception:
                    import re
                    m = re.search(r"\{.*\}", content, flags=re.S)
                    if m:
                        try:
                            js = json.loads(m.group(0))
                        except Exception:
                            js = None
                if not js or "action" not in js:
                    fallback_msgs = [
                        SystemMessage(content=SYSTEM_PROMPT + "\n\nCompany FAQ:\n" + open_faq_text()),
                        HumanMessage(content=q)
                    ]
                    resp2 = model.invoke(fallback_msgs)
                    text2 = getattr(resp2, "content", str(resp2))
                    if memory:
                        try:
                            memory.save_context({"input": q}, {"output": text2})
                        except Exception:
                            pass
                    return text2
                action = js.get("action")
                args = js.get("args") or {}
                reason = js.get("reason", "")
                if action.lower() in ("logout", "delete_account", "cancel_ticket", "buy", "purchase", "buy_flight", "purchase_flight"):
                    # set pending confirmation state and ask user to confirm
                    question = f"Do you confirm that you want to execute the action '{action}'? (yes/no)"
                    try:
                        state['awaiting_confirmation'] = {'action': real_action or action, 'args': args, 'question': question, 'original_q': q}
                    except Exception:
                        state['awaiting_confirmation'] = {'action': real_action or action, 'args': args, 'question': question, 'original_q': q}
                    return question
                action_norm = action.strip().lower() if isinstance(action, str) else None
                real_action = aliases_map.get(action_norm) if action_norm in aliases_map else action
                try:
                    if action_norm in ('buy', 'purchase', 'buy_flight'):
                        if 'flight_id' in args:
                            args['flight_number'] = args.pop('flight_id')
                        if not state['current_user_id']:
                            return "Por favor, faça login primeiro antes de tentar comprar uma passagem."
                        args['user_id'] = state['current_user_id']
                    tool_output = _call_mcp_tool_sync(real_action, args, mcp_path)
                    if action_norm == 'login' and isinstance(tool_output, str):
                        try:
                            output_data = json.loads(tool_output)
                            if isinstance(output_data, dict) and 'user_id' in output_data:
                                state['current_user_id'] = output_data['user_id']
                        except Exception:
                            pass
                except Exception as e:
                    logging.exception("Error calling MCP tool: %s", e)
                    return f"❌ Error invoking tool '{action}': {e}"
                final_system = SystemMessage(content=SYSTEM_PROMPT + "\n\nCompany FAQ:\n" + open_faq_text())
                final_human = HumanMessage(content=(
                    f"User question: {q}\n\n"
                    f"Tool invoked: '{action}' with args: {json.dumps(args, ensure_ascii=False)}\n\n"
                    f"Tool output:\n{str(tool_output)}\n\n"
                    "Format the final response to the user in English."
                ))
                msgs2 = [final_system]
                if memory:
                    hist2 = memory.load_memory_variables({}).get("history")
                    if hist2:
                        msgs2.extend(hist2 if isinstance(hist2, list) else [HumanMessage(content=hist2)])
                msgs2.append(final_human)
                final_resp = model.invoke(msgs2)
                final_text = getattr(final_resp, "content", str(final_resp))
                if memory:
                    try:
                        memory.save_context({"input": q}, {"output": final_text})
                    except Exception:
                        pass
                return final_text
            except Exception as e:
                logging.exception("Error in MCP processing: %s", e)
                return f"An error occurred while processing the MCP request: {e}"
        messages = [
            SystemMessage(content=SYSTEM_PROMPT + "\n\nCompany FAQ:\n" + open_faq_text()),
            HumanMessage(content=q)
        ]
        try:
            resp = model.invoke(messages)
            text = getattr(resp, "content", str(resp))
            if memory:
                try:
                    memory.save_context({"input": q}, {"output": text})
                except Exception:
                    pass
            return text
        except Exception as e:
            logging.exception("Error in final fallback: %s", e)
            return f"Error executing agent: {e}"
    agent_call.tools_meta = tools_meta
    agent_call.aliases_map = aliases_map
    agent_call.mcp_path = mcp_path
    agent_call.run = agent_call
    return agent_call

# --- FastAPI e main ---
# (igual ao agent4.py, mas usando build_agent e aceitando parâmetros para o LLM)
if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO)
    env_provider = os.getenv("LLM_PROVIDER", "ollama")
    env_model = os.getenv("MODEL", "llama3.2:1b")

    parser = argparse.ArgumentParser(description='agent runner - interactive or batch JSONL mode')
    parser.add_argument('--api', action='store_true', help='Run FastAPI server')
    parser.add_argument('--input', '-i', help='Input JSONL or TXT file with prompts')
    parser.add_argument('--output', '-o', help='Output JSONL for results')
    parser.add_argument('--parallel', '-p', type=int, default=1, help='Parallel workers')
    parser.add_argument('--provider', default=env_provider)
    parser.add_argument('--model', default=env_model)
    parser.add_argument('--no_tools', action='store_true')
    parser.add_argument('--mcp', default=str(MCP_DEFAULT))
    args = parser.parse_args()

    if args.api:
        if not FASTAPI_AVAILABLE:
            print("FastAPI not installed.")
            sys.exit(1)
        ag = build_agent(
            load_tools=not args.no_tools,
            mcp_mcp_path=str(args.mcp),
            llm_provider=args.provider,
            llm_model=args.model
        )
        app = create_app(ag)
        uvicorn.run(app, host='0.0.0.0', port=8000)
        sys.exit(0)

    # ------------------------------------------
    # MODE BATCH
    # ------------------------------------------
    if args.input:
        if not args.output:
            print("ERROR: --output required when using --input")
            sys.exit(1)

        in_path = Path(args.input)
        out_path = Path(args.output)

        if not in_path.exists():
            print(f"Input not found: {in_path}")
            sys.exit(1)

        # Load all lines
        items = []
        with in_path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                # JSON or simple text
                try:
                    obj = json.loads(line)
                except Exception:
                    obj = {"id": lineno, "prompt": line}
                items.append(obj)

        total = len(items)
        print(f"Processing {total} items (parallel={args.parallel})")

        # ------------------------------------------
        # Helper for processing 1 item
        # ------------------------------------------
        def process_item(item):
            qid = item.get("id")
            prompt = item.get("prompt", "")
            provider = args.provider.lower()
            model_name = args.model

            # Output structure
            out = {
                "id": qid,
                "prompt": prompt,
                "provider": provider,
                "model": model_name,
                "response": None,
                "tokens_prompt": 0,
                "tokens_response": 0,
                "latency_ms": None,
                "cpu_time_ms": None,
                "error": None,
            }

            try:
                agent = build_agent(
                    load_tools=not args.no_tools,
                    mcp_mcp_path=str(args.mcp),
                    llm_provider=provider,
                    llm_model=model_name
                )
            except Exception as e:
                out["error"] = f"Agent init error: {e}"
                return out

            # Measure time
            import time
            start = time.perf_counter()
            cpu_start = time.process_time()

            try:
                response = agent.run(prompt)
            except Exception as e:
                out["error"] = str(e)
                end = time.perf_counter()
                out["latency_ms"] = int((end - start) * 1000)
                out["cpu_time_ms"] = int((time.process_time() - cpu_start) * 1000)
                return out

            end = time.perf_counter()
            latency_ms = int((end - start) * 1000)
            cpu_ms = int((time.process_time() - cpu_start) * 1000)

            out["response"] = response
            out["latency_ms"] = latency_ms
            out["cpu_time_ms"] = cpu_ms

            # Token counts
            try:
                out["tokens_prompt"] = count_tokens(prompt, model_name) or 0
            except:
                out["tokens_prompt"] = 0
            try:
                out["tokens_response"] = count_tokens(response, model_name) or 0
            except:
                out["tokens_response"] = 0

            return out

        # ------------------------------------------
        # Run batch (parallel)
        # ------------------------------------------
        results = []
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as ex:
            for i, r in enumerate(ex.map(process_item, items), start=1):
                results.append(r)
                if i % 5 == 0 or i == total:
                    print(f"Processed {i}/{total}")

        # Save output
        with out_path.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"Done. Results written to {out_path}")
        sys.exit(0)

    # ------------------------------------------
    # MODE INTERATIVO
    # ------------------------------------------
    ag = build_agent(
        load_tools=True,
        mcp_mcp_path=str(MCP_DEFAULT),
        llm_provider=args.provider,
        llm_model=args.model
    )
    print("Agent ready (type 'exit' to quit).")
    while True:
        try:
            q = input("You: ")
        except:
            break
        if q.strip().lower() in ["exit", "quit"]:
            break
        print("Agent:", ag.run(q))
