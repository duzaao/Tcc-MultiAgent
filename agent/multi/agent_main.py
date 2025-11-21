
"""Entry point for the multi-agent orchestration layer."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

try:  # Optional FastAPI imports (same behaviour as agent)
	from fastapi import FastAPI
	FASTAPI_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
	FastAPI = None
	FASTAPI_AVAILABLE = False

try:
	import uvicorn  # type: ignore
except Exception:  # pragma: no cover - optional dependency
	uvicorn = None

from langchain_core.messages import HumanMessage, SystemMessage

from agent import (
	MCP_DEFAULT,
	ConversationBufferWindowMemory,
	SYSTEM_PROMPT,  
	create_app,
	open_faq_text,
)
from .llm import ensure_llm, wrap_with_meter
from .agent_mcp import MCPAgent
from .agent_plan import PlannerAgent


LOG = logging.getLogger("multi_agent")


def _resolve_mcp_path(candidate: Optional[str]) -> str:
	"""Locate the MCP configuration, trying common container and repo paths."""

	base_path = Path(__file__).resolve()
	normalized_candidate = Path(candidate).expanduser() if candidate else None
	fallback = base_path.parents[1] / "mcp" / "mcp.json"
	search_paths = []
	if normalized_candidate:
		search_paths.append(normalized_candidate)
	default_from_agent = Path(MCP_DEFAULT)
	search_paths.extend(
		[default_from_agent, fallback, base_path.parent / "mcp" / "mcp.json", Path("/app/mcp/mcp.json"), Path("/mcp/mcp.json")]
	)
	for path in search_paths:
		if not path:
			continue
		try:
			if path.is_file():
				return str(path)
		except Exception:
			continue
	return str(normalized_candidate or fallback)


class FAQAgent:
	"""Handles FAQ/small-talk responses using the shared LLM."""

	def __init__(self, llm, memory=None) -> None:
		self._llm = llm
		self._memory = memory
		self._faq_text = open_faq_text()

	def answer(self, message: str, *, reason: Optional[str] = None) -> str:
		context_hint = f"Routing reason: {reason}.\n" if reason else ""
		system = SystemMessage(content=SYSTEM_PROMPT + "\n\nCompany FAQ:\n" + self._faq_text)
		human = HumanMessage(content=context_hint + message)
		messages = [system]
		if self._memory:
			try:
				hist = self._memory.load_memory_variables({}).get("history")
			except Exception:
				hist = None
			if hist:
				if isinstance(hist, list):
					messages.extend(hist)
				else:
					messages.append(HumanMessage(content=str(hist)))
		messages.append(human)
		try:
			response = self._llm.invoke(messages)
			text = getattr(response, "content", str(response))
		except Exception as exc:  # pragma: no cover - fallback path
			LOG.exception("FAQ agent failed: %s", exc)
			text = "I had trouble answering that right now. Could you try again later?"
		if self._memory:
			try:
				self._memory.save_context({"input": message}, {"output": text})
			except Exception:  # pragma: no cover - memory errors are non-fatal
				pass
		return text


class MultiAgentOrchestrator:
	"""Coordinates planner, FAQ, and MCP specialist agents."""

	def __init__(
		self,
		*,
		llm_provider: Optional[str] = None,
		llm_model: Optional[str] = None,
		history_len: int = 6,
		mcp_path: Optional[str] = None,
	) -> None:
		base_llm = ensure_llm(model=llm_model, provider=llm_provider)
		llm = wrap_with_meter(base_llm, model=llm_model, provider=llm_provider)
		memory = None
		if ConversationBufferWindowMemory:
			try:
				memory = ConversationBufferWindowMemory(k=history_len, return_messages=True)
			except Exception:
				memory = None
		self._memory = memory
		self._metered_llm = llm
		self._planner = PlannerAgent(llm=llm, llm_provider=llm_provider, llm_model=llm_model)
		self._faq_agent = FAQAgent(llm=llm, memory=memory)
		resolved_mcp_path = _resolve_mcp_path(mcp_path)
		if mcp_path and Path(mcp_path).resolve() != Path(resolved_mcp_path).resolve():
			LOG.info("MCP config resolved to %s (from %s)", resolved_mcp_path, mcp_path)
		elif not mcp_path and Path(MCP_DEFAULT).resolve() != Path(resolved_mcp_path).resolve():
			LOG.info("MCP config resolved to %s", resolved_mcp_path)
		self._mcp_agent = MCPAgent(
			llm=llm,
			llm_provider=llm_provider,
			llm_model=llm_model,
			mcp_path=resolved_mcp_path,
			memory=memory,
		)

		# Expose run-compatible callable for FastAPI wiring (mirrors agent behaviour)
		self.run = self.handle_message

	# ------------------------------------------------------------------
	# Public API
	# ------------------------------------------------------------------
	def handle_message(self, message: str) -> str:
		cleaned = (message or "").strip()
		if not cleaned:
			return "Please type a message."

		# If we are waiting for a confirmation, freeze routing and let MCP agent handle it
		if self._mcp_agent.awaiting_confirmation:
			return self._mcp_agent.handle_confirmation(cleaned)

		state_snapshot = {
			"logged_in": self._mcp_agent.is_logged_in,
			"awaiting_confirmation": bool(self._mcp_agent.awaiting_confirmation),
		}
		plan = self._planner.decide(cleaned, state_snapshot)
		LOG.info(
			"Planner route=%s confidence=%.2f reason=%s", plan.route, plan.confidence, plan.reason
		)

		if plan.route == "login":
			return self._mcp_agent.handle_login(cleaned)
		if plan.route == "mcp":
			return self._mcp_agent.handle_action(cleaned, requires_login=plan.requires_login)
		# Treat fallback as FAQ to keep the system responsive
		return self._faq_agent.answer(cleaned, reason=plan.reason)

	# ------------------------------------------------------------------
	# Metrics helpers (per-interaction)
	# ------------------------------------------------------------------
	def reset_meter(self) -> None:
		try:
			self._metered_llm.reset_stats()
		except Exception:
			pass

	def get_meter_stats(self) -> dict:
		try:
			return self._metered_llm.get_stats()
		except Exception:
			return {}


def build_multi_agent(
	*,
	llm_provider: Optional[str] = None,
	llm_model: Optional[str] = None,
	history_len: int = 6,
	mcp_path: Optional[str] = None,
):
	"""Factory that instantiates the orchestrator with sensible defaults."""

	return MultiAgentOrchestrator(
		llm_provider=llm_provider,
		llm_model=llm_model,
		history_len=history_len,
		mcp_path=mcp_path,
	)


def _build_fastapi_app(agent: MultiAgentOrchestrator):
	if not FASTAPI_AVAILABLE:
		raise RuntimeError("FastAPI is not installed in this environment.")
	return create_app(agent)  # Reuse existing helper to mount front/CORS


def _interactive_loop(agent: MultiAgentOrchestrator) -> None:
	print("Multi-agent assistant ready (type 'exit' to leave).")
	while True:
		try:
			user_input = input("You: ")
		except EOFError:
			break
		if user_input.strip().lower() in {"exit", "quit"}:
			break
		reply = agent.handle_message(user_input)
		print("Agent:", reply)


def _iter_jsonl(path: str):
	p = Path(path)
	if not p.exists():
		raise FileNotFoundError(f"Questions file not found: {path}")
	for line in p.read_text(encoding="utf-8").splitlines():
		line = line.strip()
		if not line or line.startswith("//"):
			continue
		try:
			yield json.loads(line)
		except Exception:
			yield {"prompt": line}


def _append_jsonl(path: str, records: list[dict]) -> None:
	out = Path(path)
	out.parent.mkdir(parents=True, exist_ok=True)
	with out.open("a", encoding="utf-8") as f:
		for rec in records:
			f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _estimate_price(provider: Optional[str], model: Optional[str], tokens_in: int, tokens_out: int) -> float:
	"""Best-effort price estimation using env vars.

	Supported envs (per 1K tokens):
	  - LLM_INPUT_COST_PER_1K, LLM_OUTPUT_COST_PER_1K (generic)
	  - OPENAI_INPUT_COST_PER_1K, OPENAI_OUTPUT_COST_PER_1K (OpenAI-specific)
	Defaults to 0 if not configured.
	"""
	def _getf(*names: str) -> float:
		for n in names:
			v = os.getenv(n)
			if v:
				try:
					return float(v)
				except Exception:
					continue
		return 0.0

	in_per_1k = _getf("LLM_INPUT_COST_PER_1K", "OPENAI_INPUT_COST_PER_1K")
	out_per_1k = _getf("LLM_OUTPUT_COST_PER_1K", "OPENAI_OUTPUT_COST_PER_1K")
	return (tokens_in / 1000.0) * in_per_1k + (tokens_out / 1000.0) * out_per_1k


def _run_batch(agent: MultiAgentOrchestrator, questions_path: str, output_path: str, provider: Optional[str] = None, model: Optional[str] = None) -> None:
	"""Run the questions JSONL through the given agent and append results.

	provider and model, if provided, will be used in the recorded output; otherwise
	fall back to environment variables.
	"""
	rows: list[dict] = []
	provider = provider or os.getenv("LLM_PROVIDER")
	model = model or os.getenv("MODEL")
	for item in _iter_jsonl(questions_path):
		prompt = (item.get("prompt") or item.get("question") or "").strip()
		if not prompt:
			continue
		try:
			agent.reset_meter()
		except Exception:
			pass
		t0 = time.perf_counter()
		reply: str = agent.handle_message(prompt)
		dt = (time.perf_counter() - t0) * 1000.0
		stats = agent.get_meter_stats() or {}
		tin = int(stats.get("tokens_input") or 0)
		tout = int(stats.get("tokens_output") or 0)
		ttot = int(stats.get("tokens_total") or (tin + tout))
		if ttot == 0:
			try:
				tin = len(prompt.split())
				tout = len(str(reply).split())
				ttot = tin + tout
			except Exception:
				tin = tout = ttot = 0
		price = _estimate_price(provider, model, tin, tout)
		try:
			tool = getattr(agent._mcp_agent, "last_tool_name", None)  # type: ignore[attr-defined]
		except Exception:
			tool = None
		rec = {
			"id": item.get("id"),
			"category": item.get("category"),
			"provider": provider,
			"model": model,
			"prompt": prompt,
			"reply": (str(reply) or "")[:500],
			"latency_ms": round(dt, 1),
			"tokens_input": tin,
			"tokens_output": tout,
			"tokens_total": ttot,
			"price": round(price, 6) if price else 0.0,
			"mcp_tool": tool,
		}
		rows.append(rec)
	if rows:
		_append_jsonl(output_path, rows)
		print(f"Appended {len(rows)} results to {output_path}")


def _parse_llm_run_list(spec: Optional[str], default_provider: Optional[str], default_model: Optional[str]):
	"""Parse a run-list spec into a list of (provider, model) tuples.

	Supported spec forms:
	  - None -> returns [(default_provider, default_model)]
	  - comma-separated entries like "openai:gpt-4o-mini,ollama:local-model"
	  - entries with only model ("gpt-4o-mini") will use default_provider
	"""
	if not spec:
		return [(default_provider, default_model)]
	out = []
	for part in spec.split(","):
		part = part.strip()
		if not part:
			continue
		if ":" in part:
			prov, model = part.split(":", 1)
			out.append((prov.strip() or default_provider, model.strip() or default_model))
		else:
			# only model provided
			out.append((default_provider, part))
	if not out:
		return [(default_provider, default_model)]
	return out

def main() -> None:
	logging.basicConfig(level=logging.INFO)
	parser = argparse.ArgumentParser(description="FlightCo multi-agent orchestrator")
	parser.add_argument("--api", action="store_true", help="Run FastAPI server on port 8000")
	parser.add_argument("--provider", default=None, help="LLM provider (defaults to env configuration)")
	parser.add_argument("--model", default=None, help="LLM model name")
	parser.add_argument("--mcp", default=str(MCP_DEFAULT), help="Path to MCP configuration JSON")
	parser.add_argument("--history", type=int, default=6, help="Conversation turns to retain in memory")
	parser.add_argument("--questions", default=None, help="Path to JSONL with prompts for batch evaluation")
	parser.add_argument("--output", default="multi_eval.jsonl", help="Output JSONL file (appended)")
	parser.add_argument("--llm-list", default=None, help="Comma-separated provider:model entries to run sequentially (overrides LLM_RUN_LIST env). Example: 'openai:gpt-4o-mini,ollama:local' ")
	parser.add_argument("--exit-after-batch", action="store_true", help="Exit immediately after processing questions JSONL")

	args = parser.parse_args()

	agent = build_multi_agent(
		llm_provider=args.provider,
		llm_model=args.model,
		history_len=args.history,
		mcp_path=args.mcp,
	)
	# Batch mode: questions JSONL
	if args.questions:
		# Check for a run-list provided via CLI or environment
		llm_list_spec = args.llm_list or os.getenv("LLM_RUN_LIST")
		pairs = _parse_llm_run_list(llm_list_spec, args.provider or os.getenv("LLM_PROVIDER"), args.model or os.getenv("MODEL"))
		# If only one pair, run normally and write to args.output. If multiple, create per-LLM output files to avoid overwrites.
		if len(pairs) == 1:
			prov, mdl = pairs[0]
			if prov and not args.provider:
				args.provider = prov
			if mdl and not args.model:
				args.model = mdl
			_run_batch(agent, args.questions, args.output, provider=args.provider, model=args.model)
			if args.exit_after_batch:
				print("Batch complete. Exiting by request (--exit-after-batch).")
				sys.exit(0)
			return
		# Multiple runs: iterate and append results into the same output file
		for prov, mdl in pairs:
			print(f"Starting batch run for provider={prov} model={mdl}")
			# instantiate a fresh agent per pair to ensure meter/stats isolation
			agent_run = build_multi_agent(llm_provider=prov, llm_model=mdl, history_len=args.history, mcp_path=args.mcp)
			# use the same output file for all runs (append)
			_run_batch(agent_run, args.questions, args.output, provider=prov, model=mdl)
			print(f"Completed run for provider={prov} model={mdl}; appended results to {args.output}")
		if args.exit_after_batch:
			print("All batch runs complete. Exiting by request (--exit-after-batch).")
			sys.exit(0)
		return

	if args.api:
		if not FASTAPI_AVAILABLE or uvicorn is None:
			raise RuntimeError("FastAPI/uvicorn not available. Install fastapi[standard] to use --api.")
		app = _build_fastapi_app(agent)
		uvicorn.run(app, host="0.0.0.0", port=8000)
		return

	_interactive_loop(agent)


if __name__ == "__main__":
	main()
