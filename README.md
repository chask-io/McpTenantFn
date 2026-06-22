# McpTenantFn

Tenant Tool Gateway Lambda for dynamic tenant MCP tools.

## Behavior

- `preflight_mode=discover`: calls `/api/v2/tenant-mcp/search` with the event
  access token and `Organization-ID`, then maps tenant MCP functions into
  `function_data` dicts for dynamic tool creation.
- `preflight_mode=execute`: resolves a tenant MCP function action to a tenant
  API route, exchanges the orchestrator token for a short-lived tenant execute
  token, then calls the server-resolved tenant route.

The Lambda is DEV-first and declares `dynamic_tools: true` in `manifest.yml`.
It emits structured JSON logs for every preflight call with:

- `preflight_mode`
- `preflight_duration_ms`
- `preflight_error`
- `function_uuid`
- `slug`
- `branch`
- `action`

## Files

- `src/handler.py`: generated Lambda infrastructure, not business logic.
- `src/backend/function_logic.py`: Tenant Tool Gateway implementation.
- `test_provided_params*.json`, `test_operator_params.json`: remote test suite
  fixtures with test-only mock payloads.
