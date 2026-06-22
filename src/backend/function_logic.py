"""Business logic for McpTenantFn, the Tenant Tool Gateway."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Iterable, Mapping, Optional

import requests
from chask_foundation.api.tenant_data_requests import TenantDataClient
from chask_foundation.backend.models import OrchestrationEvent


logger = logging.getLogger()
logger.setLevel(logging.INFO)

DEFAULT_TOP_K = 10
VALID_BRANCHES = {"prod", "test"}
CONTROL_PLANE_TIMEOUT = 30


class FunctionBackend:
    """Tenant MCP discovery and execution gateway."""

    def __init__(self, orchestration_event: OrchestrationEvent):
        self.orchestration_event = orchestration_event
        self.session = requests.Session()
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

        action_def = self._resolve_action(params, slug, branch, function_name, action)
        path = self._required(action_def, "path")
        method = str(action_def.get("method") or "GET").upper()

        client = TenantDataClient.from_event(
            self.orchestration_event,
            lambda_uuid=os.environ["FUNCTION_UUID"],
        )
        tenant_path = self._normalize_tenant_path(path)

        if method == "GET":
            return client.get(tenant_path, params=call_params)
        if method == "POST":
            return client.post(tenant_path, json=call_params)

        raise ValueError(f"Unsupported tenant MCP action method: {method}")

    def _call_search(self, params: Mapping[str, Any]) -> Mapping[str, Any]:
        slug = self._required(params, "slug", "organization_slug")
        branch = self._normalize_branch(
            params.get("branch") or params.get("tenant_branch") or self.orchestration_event.branch
        )
        conversation_state = params.get("conversation_state")
        if not conversation_state:
            raise ValueError("conversation_state is required for discover")

        top_k = params.get("top_k", DEFAULT_TOP_K)
        payload = {
            "slug": slug,
            "branch": branch,
            "conversation_state": conversation_state,
            "top_k": top_k,
        }
        return self._control_plane_request("POST", "tenant-mcp/search", json=payload)

    def _fetch_inventory(self, slug: str, branch: str) -> Mapping[str, Any]:
        return self._control_plane_request(
            "GET",
            "tenant-mcp/functions",
            params={"slug": slug, "branch": branch},
        )

    def _control_plane_request(self, method: str, path: str, **request_kwargs) -> Mapping[str, Any]:
        url = f"{self._control_plane_base_url()}/{path.lstrip('/')}"
        headers = {
            "Authorization": f"Bearer {self.orchestration_event.access_token}",
            "Organization-ID": self.orchestration_event.organization.organization_id,
            "Content-Type": "application/json",
        }
        response = self.session.request(
            method,
            url,
            headers=headers,
            timeout=CONTROL_PLANE_TIMEOUT,
            **request_kwargs,
        )
        try:
            data = response.json()
        except ValueError:
            data = {"detail": response.text}

        if not 200 <= response.status_code < 300:
            raise RuntimeError(
                f"tenant_mcp_control_plane_{response.status_code}: {data}"
            )
        return data

    def _resolve_action(
        self,
        params: Mapping[str, Any],
        slug: str,
        branch: str,
        function_name: str,
        action: str,
    ) -> Mapping[str, Any]:
        direct_action = params.get("action_metadata") or params.get("action_def")
        if isinstance(direct_action, Mapping) and direct_action.get("path"):
            return direct_action

        for function in self._candidate_functions(params):
            if self._function_name(function) != function_name:
                continue
            action_def = self._find_action(function, action)
            if action_def:
                return action_def

        inventory = self._fetch_inventory(slug, branch)
        for function in self._extract_functions(inventory, preferred_key="functions"):
            if self._function_name(function) != function_name:
                continue
            action_def = self._find_action(function, action)
            if action_def:
                return action_def

        raise ValueError(f"Unknown tenant MCP action: {function_name}.{action}")

    def _candidate_functions(self, params: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
        values = [
            params.get("function_data"),
            params.get("discovered_function"),
            params.get("tenant_mcp_function"),
        ]
        values.extend(self._extract_functions(params, preferred_key="functions"))
        values.extend(self._extract_functions(params, preferred_key="results"))
        return [value for value in values if isinstance(value, Mapping)]

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

    def _find_action(self, function: Mapping[str, Any], action_name: str) -> Optional[Mapping[str, Any]]:
        actions = function.get("actions") or []
        for action in actions:
            if isinstance(action, Mapping) and (
                action.get("name") == action_name or action.get("action") == action_name
            ):
                return action

        mcp_actions = function.get("mcp_actions")
        if isinstance(mcp_actions, Mapping):
            action = mcp_actions.get(action_name)
            if isinstance(action, Mapping):
                return action
        return None

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

    def _control_plane_base_url(self) -> str:
        explicit_url = os.getenv("CHASK_API_BASE_URL")
        if explicit_url:
            return explicit_url.rstrip("/")
        base_domain = os.getenv("BASE_DOMAIN")
        if base_domain:
            return f"https://{base_domain}/api/v2"
        mode = os.getenv("MODE", os.getenv("GLOBAL_SERVER", "DEVELOPMENT")).upper()
        if mode == "PRODUCTION":
            return "https://app.chask.io/api/v2"
        return "https://app.chask.it/api/v2"

    def _normalize_branch(self, branch: Any) -> str:
        branch = str(branch or "").strip()
        if branch not in VALID_BRANCHES:
            raise ValueError("branch must be prod or test")
        return branch

    def _normalize_tenant_path(self, path: str) -> str:
        path = str(path or "").strip()
        if path.startswith("/api/test/"):
            return path[len("/api/test/") :]
        if path.startswith("/api/"):
            return path[len("/api/") :]
        return path.lstrip("/")

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
