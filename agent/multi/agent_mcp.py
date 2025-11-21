"""Specialist agent that coordinates MCP tool usage for the multi-agent setup."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, List

from langchain_core.messages import HumanMessage, SystemMessage

from agent import (
    MCP_DEFAULT,
    SERVER_NAME,
    SYSTEM_PROMPT,
    _call_mcp_tool_sync,
    _load_tools_for_server_from_mcp,
    _run_coro_sync,
    open_faq_text,
)
from .llm import ensure_llm

LOG = logging.getLogger(__name__)

YES_SET = {"yes", "y", "sim", "s", "confirm", "confirmar", "ok", "sure", "claro"}
NO_SET   = {"no", "n", "não", "nao", "cancel", "cancelar"}

CONFIRMATION_ACTIONS = {
    "logout",
    "cancel_customer_flight",
    "process_flight_refund",
    "buy_flight",
    "cancel_my_flight_by_number",
}


class MCPAgent:
    """Encapsulates MCP tool selection, execution, and follow-up messaging."""

    def __init__(
        self,
        llm=None,
        *,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        mcp_path: Optional[str] = None,
        memory=None,
    ) -> None:
        self._llm = llm or ensure_llm(model=llm_model, provider=llm_provider)
        self._memory = memory
        self._faq_text = open_faq_text()
        self._mcp_path = Path(mcp_path) if mcp_path else MCP_DEFAULT

        try:
            self._mcp_path = self._mcp_path.expanduser()
        except Exception:
            pass

        try:
            self._config_dir = self._mcp_path.resolve().parent
        except Exception:
            self._config_dir = Path(__file__).resolve().parents[1]

        if not self._config_dir.exists():
            self._config_dir = Path(__file__).resolve().parents[1]

        self._tools_meta: List[Dict[str, Any]] = []
        self._last_tool_name: Optional[str] = None
        self._state: Dict[str, Any] = {
            "current_user_id": None,
            "is_authenticated": False,
            "awaiting_confirmation": None,
        }

        self._load_tools_metadata()

    # -------------------------------------------------------------
    # Public properties
    # -------------------------------------------------------------
    @property
    def is_logged_in(self) -> bool:
        return bool(self._state.get("is_authenticated") or self._state.get("current_user_id"))

    @property
    def awaiting_confirmation(self) -> Optional[Dict[str, Any]]:
        return self._state.get("awaiting_confirmation")

    def update_user(self, user_id: Optional[str]) -> None:
        self._state["current_user_id"] = user_id
        self._state["is_authenticated"] = bool(user_id)

    @property
    def last_tool_name(self) -> Optional[str]:
        """Return the last MCP tool name invoked or planned for confirmation."""
        return self._last_tool_name

    # -------------------------------------------------------------
    # Confirmation Handling
    # -------------------------------------------------------------
    def handle_confirmation(self, user_reply: str) -> str:
        pending = self._state.get("awaiting_confirmation")
        if not pending:
            return "I am not waiting for any confirmation right now."

        reply = (user_reply or "").strip().lower()

        if reply in YES_SET:
            self._state["awaiting_confirmation"] = None
            action = pending.get("action")
            args = pending.get("args") or {}
            original_message = pending.get("original_message", "")
            self._last_tool_name = action
            tool_output = self._invoke_tool(action, args)
            return self._format_final_response(original_message, action, args, tool_output)

        if reply in NO_SET:
            self._state["awaiting_confirmation"] = None
            return "Action cancelled."

        return "Please respond with yes or no so I can proceed."

    # -------------------------------------------------------------
    # Login Handling
    # -------------------------------------------------------------
    def handle_login(self, message: str) -> str:
        creds = self._extract_credentials(message)
        if not creds:
            return "Please provide your username and password. Example: login johndoe mypassword"

        self._last_tool_name = "login"
        tool_output = self._invoke_tool("login", creds)
        login_result = self._extract_user_id(tool_output)
        success = bool(login_result) or self._payload_indicates_success(tool_output)

        if login_result:
            self._state["current_user_id"] = login_result
        if success:
            self._state["is_authenticated"] = True

        if login_result:
            summary = f"✅ Login successful. Active user id: {login_result}."
        elif success:
            summary = "✅ Login successful. Token stored."
        else:
            summary = "Login attempt failed. Please check your credentials."

        return self._format_final_response(message, "login", creds, tool_output, summary_override=summary)

    # -------------------------------------------------------------
    # Main action handler
    # -------------------------------------------------------------
    def handle_action(self, message: str, requires_login: bool = False) -> str:
        if requires_login and not self.is_logged_in:
            login_msg = self.ensure_login()
            if not self.is_logged_in:
                return login_msg

        plan = self._plan_tool_invocation(message)
        action = plan.get("action")
        args = plan.get("args") or {}
        if not action or action == "noop":
            return "I could not match your request to any available tool."

        action_norm = action.lower()

        def _matches(suffix: str) -> bool:
            return action_norm == suffix or action_norm.endswith(f"_{suffix}")

        # Require login for tools that rely on authenticated identity
        needs_identity = {"buy_flight", "cancel_customer_flight", "process_flight_refund"}
        if any(_matches(name) for name in needs_identity) or "_my_" in action_norm or action_norm.startswith("list_my"):
            if not self.is_logged_in:
                msg = self.ensure_login()
                if not self.is_logged_in:
                    return msg

        # Confirmation step for sensitive actions (supports prefixed tool names)
        if any(_matches(name) for name in CONFIRMATION_ACTIONS):
            question = (
                f"Do you confirm you want to execute '{action}' "
                f"with args {json.dumps(args, ensure_ascii=False)}?"
            )
            self._state["awaiting_confirmation"] = {
                "action": action,
                "args": args,
                "original_message": message,
            }
            self._last_tool_name = action
            return question

        self._last_tool_name = action
        tool_output = self._invoke_tool(action, args)
        summary_override: Optional[str] = None

        if _matches("login"):
            login_result = self._extract_user_id(tool_output)
            success = bool(login_result) or self._payload_indicates_success(tool_output)
            if login_result:
                self._state["current_user_id"] = login_result
            if success:
                self._state["is_authenticated"] = True
                summary_override = "✅ Login successful. You are now authenticated."
            else:
                summary_override = (
                    "I tried to log in but could not confirm success. "
                    "Please double-check the credentials or try again manually."
                )
        elif _matches("logout"):
            self._state["current_user_id"] = None
            self._state["is_authenticated"] = False
            summary_override = "✅ Logout complete. You're signed out."

        return self._format_final_response(message, action, args, tool_output, summary_override=summary_override)

    def ensure_login(self) -> Optional[str]:
        if self.is_logged_in:
            return None
        return "Authentication required. Please log in first using: login <username> <password>."

    # -------------------------------------------------------------
    # Load tools metadata (IMPORTANT — new version)
    # -------------------------------------------------------------
    def _load_tools_metadata(self) -> None:
        base_dir = self._config_dir
        original_cwd = os.getcwd()

        try:
            os.chdir(str(base_dir))
            loaded = _run_coro_sync(_load_tools_for_server_from_mcp(self._mcp_path, SERVER_NAME))

        except Exception as exc:
            LOG.exception("Failed to load MCP tools: %s", exc)
            loaded = []

        finally:
            try:
                os.chdir(original_cwd)
            except Exception:
                pass

        meta = []

        for tool in loaded:
            if not str(tool.name).startswith(f"mcp_{SERVER_NAME}_"):
                tool.name = f"mcp_{SERVER_NAME}_{tool.name}"

            meta.append({
                "name": tool.name,
                "description": getattr(tool, "description", "") or "",
                "args_schema": getattr(tool, "args", {}) or {},
                "common_phrases": getattr(tool, "common_phrases", []) or [],
            })

        self._tools_meta = meta

    # -------------------------------------------------------------
    # Tool Invocation
    # -------------------------------------------------------------
    def _invoke_tool(self, action: str, args: Dict[str, Any]) -> str:
        base_dir = self._config_dir
        original_cwd = os.getcwd()
        try:
            os.chdir(str(base_dir))
            self._last_tool_name = action
            return _call_mcp_tool_sync(action, args, self._mcp_path)
        except Exception as exc:
            LOG.exception("MCP tool invocation error: %s", exc)
            return f"❌ Error invoking tool '{action}': {exc}"
        finally:
            try:
                os.chdir(original_cwd)
            except Exception:
                pass

    # -------------------------------------------------------------
    # Planner — Semantic Strong Planner (Option A)
    # -------------------------------------------------------------
    def _plan_tool_invocation(self, message: str) -> Dict[str, Any]:

        enriched = []
        for t in self._tools_meta:
            enriched.append({
                "name": t["name"],
                "description": t.get("description", ""),
                "args_schema": t.get("args_schema", {}),
                "common_phrases": t.get("common_phrases", []),
            })

        system_prompt = f"""
You are an autonomous semantic planner that maps user requests to MCP tools.

Rules:
1. Compare the user's intent with each tool using deep semantic similarity.
2. You MAY use the given common_phrases, but only the ones provided. Never invent.
3. NEVER use regex or keyword matching.
4. NEVER invent arguments. Only extract if clearly present or strongly implied.
5. If no tool fits reasonably, return action="noop".
6. Return ONLY valid JSON with keys: action, args, reason.

Available tools (DO NOT modify):
{json.dumps(enriched, ensure_ascii=False, indent=2)}
"""

        system = SystemMessage(content=system_prompt)
        human = HumanMessage(content=f"User message: {message}")

        try:
            result = self._llm.invoke([system, human])
            raw = getattr(result, "content", result)
            text = self._stringify_llm_output(raw).strip()
            json_blob = self._extract_json_blob(text)
            payload = json.loads(json_blob)

            if not isinstance(payload, dict):
                raise ValueError("Planner returned invalid JSON")

            action = payload.get("action", "").strip()
            if not action:
                return {"action": "noop", "reason": "empty_action"}

            tool_names = {t["name"] for t in self._tools_meta}
            if action not in tool_names:
                return {"action": "noop", "reason": "unknown_tool"}

            args = payload.get("args", {})
            if not isinstance(args, dict):
                args = {}

            return {
                "action": action,
                "args": args,
                "reason": payload.get("reason", "")
            }

        except Exception as exc:
            LOG.warning("Semantic planner failed: %s", exc)
            LOG.debug("Planner raw output: %s", self._safe_truncate(text if 'text' in locals() else raw))
            return {"action": "noop", "reason": "planner_error"}

    # -------------------------------------------------------------
    # Output formatting
    # -------------------------------------------------------------
    def _format_final_response(
        self,
        user_message: str,
        action: str,
        args: Dict[str, Any],
        tool_output: Any,
        summary_override: Optional[str] = None,
    ) -> str:

        description = ""
        for t in self._tools_meta:
            if t["name"] == action:
                description = t["description"]
                break

        final_system = SystemMessage(
            content=SYSTEM_PROMPT + "\n\nCompany FAQ:\n" + self._faq_text
        )

        if summary_override is not None:
            out = summary_override
            if not summary_override.strip().startswith("✅"):
                details = self._safe_truncate(tool_output)
                if details and details.strip() and details.strip() not in {summary_override.strip(), "✅"}:
                    out = f"{summary_override}\n\nDetails:\n{details}"
        else:
            text = (
                f"User message: {user_message}\n\n"
                f"Tool description: {description}\n\n"
                f"Tool called: {action} with args {json.dumps(args, ensure_ascii=False)}\n\n"
                f"Tool output:\n{tool_output}\n\n"
                "Summarize the result for the user in English."
            )
            final_human = HumanMessage(content=text)
            try:
                resp = self._llm.invoke([final_system, final_human])
                out = getattr(resp, "content", str(resp))
            except Exception:
                out = str(tool_output)

        if self._memory:
            try:
                self._memory.save_context(
                    {"input": user_message},
                    {"output": out},
                )
            except Exception:
                pass

        return out

    # -------------------------------------------------------------
    # Utility — Payload helpers
    # -------------------------------------------------------------
    @staticmethod
    def _payload_indicates_success(payload: Any) -> bool:
        if not payload:
            return False

        if isinstance(payload, str):
            text = payload.lower()
        elif isinstance(payload, dict):
            text = json.dumps(payload).lower()
        else:
            text = str(payload).lower()

        return (
            "login successful" in text
            or "token saved" in text
            or "token stored" in text
        )

    @staticmethod
    def _stringify_llm_output(payload: Any) -> str:
        if payload is None:
            return ""
        if isinstance(payload, str):
            return payload
        if isinstance(payload, (list, tuple)):
            parts = [MCPAgent._stringify_llm_output(item) for item in payload]
            return "\n".join(part for part in parts if part)
        try:
            return str(payload)
        except Exception:
            return ""

    @staticmethod
    def _extract_json_blob(text: str) -> str:
        if not text:
            raise ValueError("Planner response is empty")
        candidate = text.strip()
        if candidate.startswith("{") and candidate.rstrip().endswith("}"):
            return candidate
        match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        if match:
            return match.group(0)
        raise ValueError("No JSON object found in planner output")

    @staticmethod
    def _safe_truncate(payload: Any, limit: int = 300) -> str:
        text = MCPAgent._stringify_llm_output(payload)
        if len(text) > limit:
            return text[:limit] + "..."
        return text
    @staticmethod
    def _extract_user_id(payload: Any) -> Optional[str]:
        if not payload:
            return None

        data = None
        if isinstance(payload, str):
            try:
                data = json.loads(payload)
            except Exception:
                return None
        else:
            data = payload

        if isinstance(data, dict):
            for key in ("user_id", "id", "customer_id"):
                if data.get(key):
                    return str(data[key])
        return None

    # -------------------------------------------------------------
    # Utility — Credentials extraction
    # -------------------------------------------------------------
    @staticmethod
    def _extract_credentials(message: str) -> Optional[Dict[str, str]]:
        raw = (message or "").strip()
        if not raw:
            return None

        words = raw.split()
        if words and words[0].lower() in {"login", "signin"} and len(words) >= 3:
            return {"username": words[1], "password": words[2]}

        return None
