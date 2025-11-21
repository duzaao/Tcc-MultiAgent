
"""Planner agent responsible for routing incoming messages."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from .llm import ensure_llm


WH_WORDS = ("who", "what", "when", "where", "why", "how", "which", "whom", "whose")
LOGIN_KEYWORDS = (
	"login",
	"log-in",
	"log in",
	"signin",
	"sign in",
	"autenticar",
	"entrar",
	"acessar conta",
	"authenticate",
)
MCP_KEYWORDS = (
	"list my flights",
	"my flights",
	"my tickets",
	"tickets",
	"cancel",
	"refund",
	"purchase",
	"buy",
	"book",
	"change",
	"list flights",
	"flight number",
	"meus voos",
	"reembolso",
	"login",
	"logout",
	"token",
	"senha",
)


@dataclass
class PlanResult:
	"""Structured result produced by the planner."""

	route: str = "faq"
	reason: str = ""
	confidence: float = 0.5
	requires_login: bool = False
	raw: Optional[Dict[str, Any]] = None


class PlannerAgent:
	"""LLM-backed router that picks the best specialist agent."""

	def __init__(
		self,
		llm=None,
		*,
		llm_provider: Optional[str] = None,
		llm_model: Optional[str] = None,
	) -> None:
		self._llm = llm or ensure_llm(model=llm_model, provider=llm_provider)

	def decide(self, message: str, state: Optional[Dict[str, Any]] = None) -> PlanResult:
		state = state or {}
		cleaned = (message or "").strip()
		if not cleaned:
			return PlanResult(route="faq", reason="Empty message", confidence=0.0)

		lowered = cleaned.lower()

		# Confirmation replies should be routed straight to the MCP agent
		if state.get("awaiting_confirmation"):
			if lowered in {"yes", "y", "sim", "s", "no", "n", "não", "nao", "cancel"}:
				return PlanResult(route="mcp", reason="Pending confirmation", confidence=0.95)

		# Check explicit login cues
		if any(keyword in lowered for keyword in LOGIN_KEYWORDS):
			return PlanResult(route="login", reason="Login keyword detected", confidence=0.9, requires_login=True)

		# Explicit MCP intents (book flight, list my flights, etc.)
		if any(keyword in lowered for keyword in MCP_KEYWORDS):
			return PlanResult(
				route="mcp",
				reason="Action keyword detected",
				confidence=0.85,
				requires_login="login" in lowered or state.get("logged_in") is False,
			)

		# Wh-questions usually map nicely to FAQ answers
		tokens = lowered.split()
		if tokens and tokens[0] in WH_WORDS:
			return PlanResult(route="faq", reason="WH question", confidence=0.8)

		# Greetings and chit-chat
		if lowered in {"hi", "hello", "hey", "oi", "olá", "ola", "thanks", "thank you"}:
			return PlanResult(route="faq", reason="Small talk", confidence=0.7)

		# Fallback to LLM classification for ambiguous cases
		hints = json.dumps(
			{
				"logged_in": bool(state.get("logged_in")),
				"awaiting_confirmation": bool(state.get("awaiting_confirmation")),
			}
		)
		system = SystemMessage(
			content=(
				"You are a routing assistant for FlightCo's support system. "
				"Decide which specialist should handle the user's message. "
				"Routes:\n"
				"- faq → Use FAQ/small-talk agent for greetings, policy questions, generic support.\n"
				"- login → Trigger login flow using the MCP login tool.\n"
				"- mcp → Use MCP tools for actions that require live data or authenticated operations.\n"
				"- fallback → Default to FAQ if unsure.\n\n"
				"Return a compact JSON with keys: route (faq|login|mcp|fallback), "
				"requires_login (true/false), confidence (0.0-1.0), reason. "
				"Do not add Markdown or explanations outside the JSON."
			)
		)
		human = HumanMessage(content=f"Known state: {hints}\nUser message: {cleaned}")
		try:
			llm_response = self._llm.invoke([system, human])
			payload = json.loads(llm_response.content)
			route = str(payload.get("route", "faq")).strip().lower()
			requires_login = bool(payload.get("requires_login", False))
			confidence = float(payload.get("confidence", 0.5))
			reason = str(payload.get("reason", "LLM classification")).strip()
			if route not in {"faq", "login", "mcp", "fallback"}:
				route = "fallback"
			return PlanResult(
				route=route,
				reason=reason,
				confidence=max(0.0, min(1.0, confidence)),
				requires_login=requires_login,
				raw=payload,
			)
		except Exception:
			return PlanResult(route="faq", reason="LLM classification failed", confidence=0.3)
