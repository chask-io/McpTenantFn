"""Business logic for McpTenantFn, the Tenant Tool Gateway."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Iterable, Mapping, Optional

from chask_foundation.backend.models import OrchestrationEvent

from api.tenant_mcp_requests import tenant_mcp_api_manager


logger = logging.getLogger()
logger.setLevel(logging.INFO)

DEFAULT_TOP_K = 10
VALID_BRANCHES = {"prod", "test"}
CONTROL_PLANE_TIMEOUT = 30


class FunctionBackend:
    """Tenant MCP discovery and execution gateway."""

    def __init__(self, orchestration_event: OrchestrationEvent):
        self.orchestration_event = orchestration_event
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
        slug = self._required(params, "slug", "organization_slug")
        branch = self._normalize_branch(
            params.get("branch") or params.get("tenant_branch") or self.orchestration_event.branch
        )
        function_name = self._required(params, "function_name", "target_function_name", "name")
        action = self._required(params, "action", "preflight_action")
        call_params = params.get("params")
        if call_params is None:
            call_params = params.get("arguments") or params.get("body") or {}
        if not isinstance(call_params, dict):
            raise ValueError("execute params must be an object")

        tenant_organization_id = self._tenant_mcp_organization_id(params)
        event_org_id = str(self.orchestration_event.organization.organization_id)
        if str(tenant_organization_id) != event_org_id:
            raise ValueError(
                "Tenant MCP execute must use the orchestration event organization; "
                "cross-org execute is not allowed."
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
            logger.info(
                json.dumps(
                    {
                        "event": "tenant_mcp_execute_control_plane",
                        "function_uuid": os.environ.get("FUNCTION_UUID"),
                        "slug": slug,
                        "branch": branch,
                        "function_name": function_name,
                        "action": action,
                        "duration_ms": int((time.monotonic() - started_at) * 1000),
                        "error": None,
                    }
                )
            )
            return result
        except Exception as exc:
            logger.error(
                json.dumps(
                    {
                        "event": "tenant_mcp_execute_control_plane",
                        "function_uuid": os.environ.get("FUNCTION_UUID"),
                        "slug": slug,
                        "branch": branch,
                        "function_name": function_name,
                        "action": action,
                        "duration_ms": int((time.monotonic() - started_at) * 1000),
                        "error": str(exc),
                    }
                ),
                exc_info=True,
            )
            raise

    def _call_search(self, params: Mapping[str, Any]) -> Mapping[str, Any]:
        slug = self._required(params, "slug", "organization_slug")
        branch = self._normalize_branch(
            params.get("branch") or params.get("tenant_branch") or self.orchestration_event.branch
        )
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
            # Sidecar metadata is ignored by create_dynamic_tool_class, but lets
            # execute mode avoid a second inventory lookup when Area 3 passes it.
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
            return schema_parameters

        openapi_parameters = self._openapi_parameters_to_tool_parameters(
            action.get("parameters") or action.get("query_parameters") or []
        )
        if openapi_parameters:
            return openapi_parameters

        logger.warning(
            json.dumps(
                {
                    "event": "tenant_mcp_action_parameters_missing",
                    "function_uuid": os.environ.get("FUNCTION_UUID"),
                    "path": action.get("path"),
                    "method": action.get("method"),
                    "action": action_name,
                    "request_schema": action.get("request_schema"),
                    "has_parameters_field": bool(action.get("parameters")),
                    "has_query_parameters_field": bool(action.get("query_parameters")),
                }
            )
        )
        return {}

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
