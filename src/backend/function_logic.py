"""Business logic for McpTenantFn, the Tenant Tool Gateway."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, Iterable, Mapping, Optional

from chask_foundation.backend.models import OrchestrationEvent

from api.orchestrator_requests import orchestrator_api_manager
from api.tenant_mcp_requests import tenant_mcp_api_manager


logger = logging.getLogger()
logger.setLevel(logging.INFO)

DEFAULT_TOP_K = 10
VALID_BRANCHES = {"prod", "test"}
CONTROL_PLANE_TIMEOUT = 30
# Tier-A provenance schema lookup (list_tenant_mcp_functions) must never eat
# into the execute call's own budget, so it gets a strictly shorter timeout
# and a degrade-not-fail policy (see _lookup_action_schema).
PROVENANCE_SCHEMA_LOOKUP_TIMEOUT = 10
# "B-prime" origin-event resolution (get_orchestration_events, session-scoped
# uniqueness gate — see _resolve_trigger_whatsapp_event) gets its own short
# per-call timeout plus a wall-clock deadline checked BEFORE the call is made,
# so a slow/hanging lookup can never eat the execute budget.
BPRIME_LOOKUP_TIMEOUT = 10
BPRIME_DEADLINE_SECONDS = 15

# Sentinel distinguishing "not yet attempted" from "attempted and unresolved"
# in the per-invocation B' cache.
_UNSET = object()


def _log_json(level: int, event: str, exc_info: bool = False, **fields: Any) -> None:
    logger.log(
        level,
        json.dumps({"event": event, "function_uuid": os.environ.get("FUNCTION_UUID"), **fields}),
        exc_info=exc_info,
    )


class FunctionBackend:
    """Tenant MCP discovery and execution gateway."""

    def __init__(self, orchestration_event: OrchestrationEvent):
        self.orchestration_event = orchestration_event
        # Per-invocation caches so a multi-field/multi-action execute never
        # repeats a control-plane lookup.
        self._tenant_function_schema_cache: Dict[tuple, Any] = {}
        self._bprime_cache: Any = _UNSET
        self._provenance_deadline_at: Optional[float] = None
        logger.info(
            "Initialized McpTenantFn for org: %s",
            orchestration_event.organization.organization_id,
        )

    def handle_preflight(self):
        """Handle dynamic tool discover and execute preflight requests."""
        started_at = time.monotonic()
        extra_params = self.orchestration_event.extra_params or {}
        tool_args = self._extract_tool_args()
        merged_params = {**tool_args, **extra_params}
        preflight_mode = str(merged_params.get("preflight_mode") or "discover")
        action = merged_params.get("action") or merged_params.get("preflight_action")
        slug = merged_params.get("slug") or merged_params.get("organization_slug")
        branch = merged_params.get("branch") or merged_params.get("tenant_branch")
        tenant_organization_id = self._tenant_mcp_organization_id(merged_params)
        function_name = (
            merged_params.get("function_name")
            or merged_params.get("target_function_name")
            or merged_params.get("name")
        )

        def log_payload(preflight_error: Optional[str] = None) -> Dict[str, Any]:
            return {
                "preflight_mode": preflight_mode,
                "preflight_duration_ms": int((time.monotonic() - started_at) * 1000),
                "preflight_error": preflight_error,
                "function_uuid": os.environ.get("FUNCTION_UUID"),
                "slug": slug,
                "branch": branch,
                "action": action,
                "function_name": function_name,
                "tenant_organization_id": tenant_organization_id,
            }

        try:
            if preflight_mode == "discover":
                try:
                    result = self._discover(merged_params)
                    logger.info(json.dumps(log_payload()))
                    return result
                except Exception as exc:
                    logger.error(json.dumps(log_payload(str(exc))), exc_info=True)
                    return []

            if preflight_mode == "execute":
                try:
                    result = self._execute(merged_params)
                    logger.info(json.dumps(log_payload()))
                    return result
                except Exception as exc:
                    logger.error(json.dumps(log_payload(str(exc))), exc_info=True)
                    return {
                        "status": "error",
                        "error": str(exc),
                        "function_name": function_name,
                        "action": action,
                    }

            raise ValueError("Invalid preflight_mode. Expected 'discover' or 'execute'.")

        except Exception as exc:
            logger.error(json.dumps(log_payload(str(exc))), exc_info=True)
            raise

    def process_request(self) -> str:
        """
        McpTenantFn is normally called through preflight_discover.

        A normal function_call supports action=health for publish gate
        cold-start integrity checks. Real gateway behavior is preflight-only.
        """
        params = self._extract_tool_args()
        action = params.get("preflight_mode") or params.get("action")
        if action == "health":
            return json.dumps(
                {
                    "status": "ok",
                    "function": "McpTenantFn",
                    "dynamic_tools": True,
                    "function_uuid": os.environ.get("FUNCTION_UUID"),
                },
                ensure_ascii=False,
                sort_keys=True,
            )

        if action not in {"discover", "execute"}:
            raise ValueError(
                "McpTenantFn expects event_type=preflight_discover with "
                "preflight_mode=discover|execute, or action=health for gate checks."
            )

        params["preflight_mode"] = action
        result = self._discover(params) if action == "discover" else self._execute(params)
        return json.dumps(result, ensure_ascii=False, sort_keys=True)

    def _discover(self, params: Mapping[str, Any]) -> list[Dict[str, Any]]:
        response_data = self._call_search(params)
        functions = self._extract_functions(response_data, preferred_key="results")
        tool_defs = [self._function_to_tool_def(function) for function in functions]
        for tool_def in tool_defs:
            tool_def["slug"] = tool_def.get("slug") or response_data.get("slug")
            tool_def["branch"] = tool_def.get("branch") or response_data.get("branch")
        return tool_defs

    def _execute(self, params: Mapping[str, Any]) -> Any:
        slug, branch = self._slug_and_branch(params)
        function_name = self._required(params, "function_name", "target_function_name", "name")
        action = self._required(params, "action", "preflight_action")
        call_params = params.get("params")
        if call_params is None:
            call_params = params.get("arguments") or params.get("body") or {}
        if not isinstance(call_params, dict):
            raise ValueError("execute params must be an object")
        call_params = dict(call_params)  # never mutate the caller's dict

        tenant_organization_id = self._tenant_mcp_organization_id(params)
        event_org_id = str(self.orchestration_event.organization.organization_id)
        if str(tenant_organization_id) != event_org_id:
            raise ValueError(
                "Tenant MCP execute must use the orchestration event organization; "
                "cross-org execute is not allowed."
            )

        self._provenance_deadline_at = time.monotonic() + BPRIME_DEADLINE_SECONDS
        call_params = self._apply_provenance_injection(
            slug=slug,
            branch=branch,
            function_name=function_name,
            action=action,
            call_params=call_params,
            tenant_organization_id=tenant_organization_id,
        )

        started_at = time.monotonic()
        try:
            result = tenant_mcp_api_manager.call(
                "execute_tenant_mcp_function",
                slug=slug,
                branch=branch,
                function_name=function_name,
                action=action,
                params=call_params,
                access_token=self.orchestration_event.access_token,
                organization_id=tenant_organization_id,
                timeout=CONTROL_PLANE_TIMEOUT,
            )
            _log_json(
                logging.INFO,
                "tenant_mcp_execute_control_plane",
                slug=slug,
                branch=branch,
                function_name=function_name,
                action=action,
                duration_ms=int((time.monotonic() - started_at) * 1000),
                error=None,
            )
            return result
        except Exception as exc:
            _log_json(
                logging.ERROR,
                "tenant_mcp_execute_control_plane",
                exc_info=True,
                slug=slug,
                branch=branch,
                function_name=function_name,
                action=action,
                duration_ms=int((time.monotonic() - started_at) * 1000),
                error=str(exc),
            )
            raise

    # ------------------------------------------------------------------
    # Server-side provenance injection
    # ------------------------------------------------------------------
    #
    # Certain action parameters (source_event_uuid, requested_by_phone,
    # whatsapp_message_id, ...) identify WHO/WHAT triggered a tool call and
    # must be authored by server-side state, never by the LLM — an operator
    # LLM can hallucinate a well-formed-but-fake UUID that passes type
    # validation and silently corrupts tenant data (see the incident this
    # fix addresses). The map below is declarative: adding a new provenance
    # field is a one-line addition of (field_name -> resolver method name).
    #
    # Resolution ladder per declared field:
    #   tier A  — list_tenant_mcp_functions gives us the action's real schema
    #             (is this field declared? is it required?). When available,
    #             an unavailable-but-required field is a clear error (no
    #             control-plane call); an unavailable-but-optional field is
    #             stripped from call_params rather than forwarded.
    #   tier B  — schema lookup failed/degraded: no requiredness knowledge,
    #             so we can only act on fields the LLM already included —
    #             overwrite with the authoritative value if we have one,
    #             otherwise strip the key. We can never inject a field the
    #             LLM omitted, and we never trust the LLM's value.
    # Each resolver, in turn, tries "B-prime" (a session-scoped, uniqueness-
    # gated lookup of the single received_whatsapp_message event in this
    # orchestration session — see _resolve_trigger_whatsapp_event) before
    # giving up. B-prime only resolves for single-inbound-message sessions;
    # multi-message sessions are a stated, partial-coverage gap until the
    # orchestrator propagates trigger-event lineage directly (out of scope
    # for this Lambda — see PR description).
    def _resolve_source_event_uuid(self) -> Optional[str]:
        trigger = self._resolve_trigger_whatsapp_event()
        return trigger.get("event_id") if trigger else None

    def _resolve_requested_by_phone(self) -> Optional[str]:
        # Same degrade-never-throw contract as B-prime: a resolver raising is
        # not a "provenance unavailable" outcome we handle, it's an unhandled
        # exception that turns a clean strip/error into a bare Lambda 500.
        try:
            customer = getattr(self.orchestration_event, "customer", None)
            phone = getattr(customer, "phone", None) if customer else None
            return str(phone) if phone else None
        except Exception:
            logger.warning("Failed to resolve requested_by_phone from orchestration_event", exc_info=True)
            return None

    def _resolve_whatsapp_message_id(self) -> Optional[str]:
        trigger = self._resolve_trigger_whatsapp_event()
        message_id = trigger.get("message_id") if trigger else None
        return str(message_id) if message_id else None

    PROVENANCE_RESOLVERS: Dict[str, str] = {
        "source_event_uuid": "_resolve_source_event_uuid",
        "requested_by_phone": "_resolve_requested_by_phone",
        "whatsapp_message_id": "_resolve_whatsapp_message_id",
    }

    @staticmethod
    def _strip_field(call_params: Dict[str, Any], field_name: str, stripped: "list[str]") -> None:
        if field_name in call_params:
            del call_params[field_name]
            stripped.append(field_name)

    def _apply_provenance_injection(
        self,
        *,
        slug: str,
        branch: str,
        function_name: str,
        action: str,
        call_params: Dict[str, Any],
        tenant_organization_id: str,
    ) -> Dict[str, Any]:
        action_schema = self._lookup_action_schema(
            slug=slug,
            branch=branch,
            function_name=function_name,
            action=action,
            tenant_organization_id=tenant_organization_id,
        )

        injected: list[str] = []
        stripped: list[str] = []

        if action_schema is not None:
            tier = "A"
            for field_name, resolver_name in self.PROVENANCE_RESOLVERS.items():
                field_schema = action_schema.get(field_name)
                if field_schema is None:
                    # Action doesn't declare this field — but if the LLM sent
                    # it anyway, never let it cross to the tenant unmutated.
                    # chask_api's execute view does zero request validation
                    # (see PR description); trusting the tenant side to drop
                    # an unexpected field is exactly the opaque-boundary trust
                    # this fix exists to remove.
                    self._strip_field(call_params, field_name, stripped)
                    continue
                value = getattr(self, resolver_name)()
                if value is not None:
                    call_params[field_name] = value
                    injected.append(field_name)
                    continue
                if field_schema.get("required"):
                    raise ValueError(
                        f"Required provenance field '{field_name}' has no authoritative "
                        f"value for {function_name}.{action}; refusing to forward an "
                        "LLM-authored value."
                    )
                self._strip_field(call_params, field_name, stripped)
        else:
            tier = "B"
            for field_name, resolver_name in self.PROVENANCE_RESOLVERS.items():
                if field_name not in call_params:
                    continue  # tier B has no schema, so it can't inject an omitted field
                value = getattr(self, resolver_name)()
                if value is not None:
                    call_params[field_name] = value
                    injected.append(field_name)
                else:
                    self._strip_field(call_params, field_name, stripped)

        _log_json(
            logging.INFO,
            "tenant_mcp_execute_provenance",
            slug=slug,
            branch=branch,
            function_name=function_name,
            action=action,
            tier=tier,
            provenance_injected=injected,
            provenance_stripped=stripped,
            requested_by_phone_hash=(self._hash_phone() if "requested_by_phone" in injected else None),
        )
        return call_params

    def _lookup_action_schema(
        self,
        *,
        slug: str,
        branch: str,
        function_name: str,
        action: str,
        tenant_organization_id: str,
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """Tier-A: authoritative action parameter schema, cached per (slug, branch)."""
        cache_key = (slug, branch)
        if cache_key not in self._tenant_function_schema_cache:
            try:
                response = tenant_mcp_api_manager.call(
                    "list_tenant_mcp_functions",
                    slug=slug,
                    branch=branch,
                    access_token=self.orchestration_event.access_token,
                    organization_id=tenant_organization_id,
                    timeout=PROVENANCE_SCHEMA_LOOKUP_TIMEOUT,
                )
            except Exception as exc:
                _log_json(
                    logging.WARNING,
                    "tenant_mcp_provenance_schema_lookup_failed",
                    slug=slug,
                    branch=branch,
                    function_name=function_name,
                    action=action,
                    error=str(exc),
                )
                response = None
            self._tenant_function_schema_cache[cache_key] = response

        response = self._tenant_function_schema_cache[cache_key]
        if not isinstance(response, Mapping):
            return None

        functions = response.get("functions")
        if not isinstance(functions, list):
            return None

        for function in functions:
            if not isinstance(function, Mapping) or function.get("name") != function_name:
                continue
            for candidate_action in function.get("actions") or []:
                if not isinstance(candidate_action, Mapping):
                    continue
                if candidate_action.get("name") == action:
                    parameters = candidate_action.get("parameters")
                    return parameters if isinstance(parameters, Mapping) else {}
            # Function IS in the authoritative listing but doesn't declare this
            # action at all — that's still authoritative information (this
            # action declares zero provenance fields), not a degraded lookup,
            # so treat it as an empty tier-A schema rather than falling back
            # to tier B's weaker "trust whatever key names the LLM used".
            return {}
        return None  # function not present in the authoritative listing — degrade to tier B

    def _resolve_trigger_whatsapp_event(self) -> Optional[Dict[str, Any]]:
        """
        "B-prime": resolve the single received_whatsapp_message event that
        triggered this tool call, via a session-scoped uniqueness gate.

        This is per-message-correct BY CONSTRUCTION only when the session
        contains exactly one inbound WhatsApp message — that message is then
        the trigger of every tool call in the session, not by likelihood.
        Zero or multiple inbound messages in the session make the trigger
        ambiguous; we never guess (no most-recent, no closest-timestamp, no
        heuristic tie-break) — unresolved falls through to the tier-A/B
        required-vs-optional handling in _apply_provenance_injection.

        Multi-inbound-message sessions are therefore a stated, partial gap:
        this resolves correctly only for single-inbound-message sessions.
        """
        if self._bprime_cache is not _UNSET:
            return self._bprime_cache

        result = self._resolve_trigger_whatsapp_event_uncached()
        self._bprime_cache = result
        return result

    def _resolve_trigger_whatsapp_event_uncached(self) -> Optional[Dict[str, Any]]:
        session_uuid = getattr(self.orchestration_event, "orchestration_session_uuid", None)
        if not session_uuid:
            self._log_bprime_result(inbound_message_count=None, terminal_reason="no_session_uuid")
            return None

        deadline = self._provenance_deadline_at
        call_timeout = BPRIME_LOOKUP_TIMEOUT
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._log_bprime_result(inbound_message_count=None, terminal_reason="deadline_exceeded")
                return None
            # Compose the two budgets: the call itself must never run longer
            # than whatever's left of the wall-clock deadline, even though it
            # also has its own independent per-call cap.
            call_timeout = min(BPRIME_LOOKUP_TIMEOUT, remaining)

        try:
            response = orchestrator_api_manager.call(
                "get_orchestration_events",
                orchestration_session_id=str(session_uuid),
                access_token=self.orchestration_event.access_token,
                organization_id=str(self.orchestration_event.organization.organization_id),
                timeout=call_timeout,
            )
        except Exception as exc:
            self._log_bprime_result(
                inbound_message_count=None,
                terminal_reason="lookup_failed",
                error=str(exc),
            )
            return None

        events = response.get("orchestration_events") if isinstance(response, Mapping) else None
        if not isinstance(events, list):
            self._log_bprime_result(inbound_message_count=None, terminal_reason="malformed_response")
            return None

        inbound_events = [
            event
            for event in events
            if isinstance(event, Mapping) and event.get("event_type") == "received_whatsapp_message"
        ]
        count = len(inbound_events)

        if count != 1:
            self._log_bprime_result(
                inbound_message_count=count,
                terminal_reason="zero_inbound_messages" if count == 0 else "ambiguous_multiple_inbound_messages",
            )
            return None

        inbound_event = inbound_events[0]
        extra_params = inbound_event.get("extra_params")
        message_id = extra_params.get("message_id") if isinstance(extra_params, Mapping) else None
        event_id = inbound_event.get("event_id")
        if not event_id:
            self._log_bprime_result(inbound_message_count=count, terminal_reason="missing_event_id")
            return None

        self._log_bprime_result(inbound_message_count=count, terminal_reason="resolved")
        return {"event_id": str(event_id), "message_id": message_id}

    def _log_bprime_result(
        self,
        *,
        inbound_message_count: Optional[int],
        terminal_reason: str,
        error: Optional[str] = None,
    ) -> None:
        _log_json(
            logging.INFO,
            "tenant_mcp_execute_provenance_bprime",
            orchestration_session_uuid=getattr(self.orchestration_event, "orchestration_session_uuid", None),
            inbound_message_count=inbound_message_count,
            resolved=terminal_reason == "resolved",
            terminal_reason=terminal_reason,
            error=error,
        )

    def _hash_phone(self) -> Optional[str]:
        phone = self._resolve_requested_by_phone()
        if not phone:
            return None
        return hashlib.sha256(phone.encode("utf-8")).hexdigest()[:12]

    def _slug_and_branch(self, params: Mapping[str, Any]) -> tuple[str, str]:
        slug = self._required(params, "slug", "organization_slug")
        branch = self._normalize_branch(
            params.get("branch") or params.get("tenant_branch") or self.orchestration_event.branch
        )
        return slug, branch

    def _call_search(self, params: Mapping[str, Any]) -> Mapping[str, Any]:
        slug, branch = self._slug_and_branch(params)
        conversation_state = params.get("conversation_state")
        if not conversation_state:
            raise ValueError("conversation_state is required for discover")

        top_k = params.get("top_k", DEFAULT_TOP_K)
        return tenant_mcp_api_manager.call(
            "search_tenant_mcp_functions",
            slug=slug,
            branch=branch,
            conversation_state=conversation_state,
            top_k=top_k,
            access_token=self.orchestration_event.access_token,
            organization_id=self._tenant_mcp_organization_id(params),
            timeout=CONTROL_PLANE_TIMEOUT,
        )

    def _tenant_mcp_organization_id(self, params: Optional[Mapping[str, Any]] = None) -> str:
        params = params or {}
        value = (
            params.get("tenant_organization_id")
            or params.get("control_plane_organization_id")
            or params.get("dynamic_tool_organization_id")
            or params.get("mcp_organization_id")
        )
        return str(value or self.orchestration_event.organization.organization_id)

    def _function_to_tool_def(self, function: Mapping[str, Any]) -> Dict[str, Any]:
        name = self._function_name(function)
        action_parameters = {}
        mcp_actions = {}

        for action in function.get("actions") or []:
            if not isinstance(action, Mapping):
                continue
            action_name = str(action.get("name") or action.get("action") or "").strip()
            if not action_name:
                continue
            action_parameters[action_name] = self._action_to_parameters(
                action,
                action_name,
            )
            mcp_actions[action_name] = {
                "method": str(action.get("method") or "GET").upper(),
                "path": action.get("path"),
                "summary": action.get("summary") or "",
                "description": action.get("description") or "",
                "operation_id": action.get("operation_id") or "",
            }

        return {
            "uuid": str(function.get("uuid") or name),
            "display_name": name,
            "description": str(function.get("description") or name),
            "required_parameters": {},
            "optional_parameters": {},
            "action_parameters": action_parameters,
            # NOTE: this sidecar is currently inert downstream — Area 3's
            # create_dynamic_tool_class (chask-foundation dynamic_tools.py)
            # never reads "mcp_actions" onto the LLM tool class, so it can
            # never round-trip back to _execute via tool_calls[0].args. Left
            # in the output SHAPE unchanged (platform-contract stability);
            # do not expand it or add new execute-time dependencies on it.
            "mcp_actions": mcp_actions,
            "slug": function.get("slug"),
            "branch": function.get("branch"),
            "score": function.get("score"),
        }

    def _action_to_parameters(
        self,
        action: Mapping[str, Any],
        action_name: str,
    ) -> Dict[str, Dict[str, Any]]:
        schema_parameters = self._schema_to_parameters(
            action.get("request_schema") or action.get("schema") or {}
        )
        if schema_parameters:
            return self._hide_server_supplied_parameters(schema_parameters, action_name)

        openapi_parameters = self._openapi_parameters_to_tool_parameters(
            action.get("parameters") or action.get("query_parameters") or []
        )
        if openapi_parameters:
            return self._hide_server_supplied_parameters(openapi_parameters, action_name)

        _log_json(
            logging.WARNING,
            "tenant_mcp_action_parameters_missing",
            path=action.get("path"),
            method=action.get("method"),
            action=action_name,
            request_schema=action.get("request_schema"),
            has_parameters_field=bool(action.get("parameters")),
            has_query_parameters_field=bool(action.get("query_parameters")),
        )
        return {}

    def _hide_server_supplied_parameters(
        self,
        parameters: Dict[str, Dict[str, Any]],
        action_name: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Never ask the LLM to author a provenance field — this action's
        declared provenance fields (per PROVENANCE_RESOLVERS) are always
        server-injected in _execute, so they are hidden here entirely rather
        than merely marked optional."""
        hidden = [name for name in self.PROVENANCE_RESOLVERS if name in parameters]
        if not hidden:
            return parameters

        visible = {
            name: value for name, value in parameters.items() if name not in self.PROVENANCE_RESOLVERS
        }
        _log_json(
            logging.INFO,
            "tenant_mcp_discover_provenance_hidden",
            action=action_name,
            hidden_parameters=hidden,
        )
        return visible

    def _schema_to_parameters(self, schema: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
        if not isinstance(schema, Mapping):
            return {}

        properties = schema.get("properties") if isinstance(schema.get("properties"), Mapping) else {}
        required = set(schema.get("required") or [])
        parameters = {}
        for name, details in properties.items():
            details = details if isinstance(details, Mapping) else {}
            parameters[name] = {
                "type": self._json_schema_type(details),
                "required": name in required,
                "description": str(details.get("description") or name),
            }
        return parameters

    def _openapi_parameters_to_tool_parameters(
        self,
        parameters: Iterable[Any],
    ) -> Dict[str, Dict[str, Any]]:
        result = {}
        for parameter in parameters:
            if not isinstance(parameter, Mapping):
                continue
            name = parameter.get("name")
            if not name:
                continue
            schema = parameter.get("schema") if isinstance(parameter.get("schema"), Mapping) else {}
            result[str(name)] = {
                "type": self._json_schema_type(schema),
                "required": bool(parameter.get("required")),
                "description": str(parameter.get("description") or name),
            }
        return result

    def _extract_functions(
        self,
        data: Mapping[str, Any],
        *,
        preferred_key: str,
    ) -> list[Mapping[str, Any]]:
        if not isinstance(data, Mapping):
            return []
        candidates = data.get(preferred_key)
        if candidates is None:
            candidates = data.get("results") or data.get("functions") or data.get("data") or []
        return [item for item in candidates if isinstance(item, Mapping)]

    def _extract_tool_args(self) -> Dict[str, Any]:
        extra_params = self.orchestration_event.extra_params or {}
        tool_calls = extra_params.get("tool_calls", [])
        if not tool_calls:
            return {}
        tool_call = tool_calls[0] or {}
        return tool_call.get("args", {}) or {}

    def _normalize_branch(self, branch: Any) -> str:
        branch = str(branch or "").strip()
        if branch not in VALID_BRANCHES:
            raise ValueError("branch must be prod or test")
        return branch

    def _function_name(self, function: Mapping[str, Any]) -> str:
        return str(
            function.get("name")
            or function.get("display_name")
            or function.get("function_name")
            or ""
        )

    def _json_schema_type(self, schema: Mapping[str, Any]) -> str:
        schema_type = schema.get("type") or "string"
        if isinstance(schema_type, list):
            return str(schema_type[0] if schema_type else "string")
        return str(schema_type)

    def _required(self, data: Mapping[str, Any], *keys: str) -> Any:
        for key in keys:
            value = data.get(key)
            if value not in (None, ""):
                return value
        raise ValueError(f"Missing required parameter: {'/'.join(keys)}")
