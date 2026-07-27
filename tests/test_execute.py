"""Unit tests for FunctionBackend._execute — control-plane path + provenance injection.

All control-plane calls (tenant_mcp_api_manager, orchestrator_api_manager) are
mocked. Tier-A schema lookup and B-prime origin-event resolution are both
network hops in production; tests must never depend on a live call — they are
routed by call name via side_effect helpers below.
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock


def _make_event(
    org_id: str = "org-aaa",
    access_token: str = "tok-xyz",
    branch: str = "prod",
    phone: str = "+56911111111",
    session_uuid: str = "sess-aaa",
):
    """Build a minimal OrchestrationEvent-shaped mock."""
    org = MagicMock()
    org.organization_id = org_id
    customer = MagicMock()
    customer.phone = phone
    event = MagicMock()
    event.organization = org
    event.customer = customer
    event.access_token = access_token
    event.branch = branch
    event.orchestration_session_uuid = session_uuid
    event.extra_params = {}
    return event


def _load_module():
    """Import function_logic with all layer deps stubbed out."""
    # Stub chask_foundation
    cf = types.ModuleType("chask_foundation")
    cf_backend = types.ModuleType("chask_foundation.backend")
    cf_models = types.ModuleType("chask_foundation.backend.models")
    cf_models.OrchestrationEvent = object
    sys.modules.setdefault("chask_foundation", cf)
    sys.modules.setdefault("chask_foundation.backend", cf_backend)
    sys.modules.setdefault("chask_foundation.backend.models", cf_models)

    # Stub api.tenant_mcp_requests
    api_pkg = types.ModuleType("api")
    api_tmr = types.ModuleType("api.tenant_mcp_requests")
    mock_tenant_manager = MagicMock()
    api_tmr.tenant_mcp_api_manager = mock_tenant_manager
    sys.modules.setdefault("api", api_pkg)
    sys.modules.setdefault("api.tenant_mcp_requests", api_tmr)

    # Stub api.orchestrator_requests (B-prime origin-event lookup)
    api_or = types.ModuleType("api.orchestrator_requests")
    mock_orchestrator_manager = MagicMock()
    api_or.orchestrator_api_manager = mock_orchestrator_manager
    sys.modules.setdefault("api.orchestrator_requests", api_or)

    # Force re-import with fresh stubs if already cached
    sys.modules.pop("src.backend.function_logic", None)
    sys.modules.pop("src.backend", None)
    sys.modules.pop("src", None)

    import importlib.util, os
    spec = importlib.util.spec_from_file_location(
        "function_logic",
        os.path.join(os.path.dirname(__file__), "..", "src", "backend", "function_logic.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, mock_tenant_manager, mock_orchestrator_manager


_mod, _mock_tenant_manager, _mock_orchestrator_manager = _load_module()
FunctionBackend = _mod.FunctionBackend


# ----------------------------------------------------------------------
# Routing helpers — tenant_mcp_api_manager.call and orchestrator_api_manager.call
# are shared across list_tenant_mcp_functions / execute_tenant_mcp_function /
# search_tenant_mcp_functions / get_orchestration_events, so tests route by
# call name rather than a single flat return_value.
# ----------------------------------------------------------------------


def _tenant_route(**responses):
    """side_effect for tenant_mcp_api_manager.call, keyed by registered call name.

    Each value may be a static return value or a callable(**kwargs) -> value
    (callables receive the same kwargs the code under test passed, allowing
    per-call-name error injection via exceptions).
    """

    def _side_effect(call_name, **kwargs):
        if call_name not in responses:
            raise AssertionError(f"unexpected tenant_mcp_api_manager.call({call_name!r})")
        value = responses[call_name]
        if callable(value):
            return value(**kwargs)
        return value

    return _side_effect


def _orchestrator_route(**responses):
    def _side_effect(call_name, **kwargs):
        if call_name not in responses:
            raise AssertionError(f"unexpected orchestrator_api_manager.call({call_name!r})")
        value = responses[call_name]
        if callable(value):
            return value(**kwargs)
        return value

    return _side_effect


def _no_schema():
    """Tier-A response where the function isn't in the authoritative listing —
    degrades to tier B."""
    return {"functions": []}


def _schema(function_name, action_name, parameters):
    """Tier-A response declaring one action's parameter schema."""
    return {
        "functions": [
            {
                "name": function_name,
                "actions": [{"name": action_name, "parameters": parameters}],
            }
        ]
    }


def _inbound_events(*message_ids_and_uuids):
    """orchestration_events payload with N received_whatsapp_message events."""
    events = [
        {
            "event_id": event_id,
            "event_type": "received_whatsapp_message",
            "extra_params": {"message_id": message_id},
        }
        for event_id, message_id in message_ids_and_uuids
    ]
    return {"orchestration_events": events}


class ExecuteTestCase(unittest.TestCase):
    def setUp(self):
        _mock_tenant_manager.reset_mock(return_value=True, side_effect=True)
        _mock_orchestrator_manager.reset_mock(return_value=True, side_effect=True)

    def _backend(self, **event_kwargs):
        return FunctionBackend(_make_event(**event_kwargs))

    def _execute_calls(self):
        return [c for c in _mock_tenant_manager.call.call_args_list if c.args[0] == "execute_tenant_mcp_function"]


class TestExecuteBasicForwarding(ExecuteTestCase):
    """Preserve pre-existing forwarding behavior for actions with no declared
    provenance fields (or when tier-A can't resolve the function)."""

    def test_forwards_all_kwargs_to_api_manager(self):
        expected_body = {"id": "1001", "status": "processing"}
        _mock_tenant_manager.call.side_effect = _tenant_route(
            list_tenant_mcp_functions=_no_schema(),
            execute_tenant_mcp_function=expected_body,
        )

        backend = self._backend()
        result = backend._execute(
            {
                "slug": "b-quellon",
                "branch": "prod",
                "function_name": "get-order",
                "action": "get-order",
                "params": {"order_id": "1001"},
            }
        )

        execute_calls = self._execute_calls()
        self.assertEqual(len(execute_calls), 1)
        _, kwargs = execute_calls[0]
        self.assertEqual(
            kwargs,
            {
                "slug": "b-quellon",
                "branch": "prod",
                "function_name": "get-order",
                "action": "get-order",
                "params": {"order_id": "1001"},
                "access_token": "tok-xyz",
                "organization_id": "org-aaa",
                "timeout": _mod.CONTROL_PLANE_TIMEOUT,
            },
        )
        self.assertEqual(result, expected_body)

    def test_returns_body_unchanged(self):
        body = {"key": "value", "nested": {"a": 1}}
        _mock_tenant_manager.call.side_effect = _tenant_route(
            list_tenant_mcp_functions=_no_schema(),
            execute_tenant_mcp_function=body,
        )
        backend = self._backend()
        result = backend._execute(
            {
                "slug": "demo",
                "branch": "test",
                "function_name": "list-items",
                "action": "list-items",
                "params": {},
            }
        )
        self.assertIs(result, body)

    def test_empty_params_defaults_to_empty_dict(self):
        _mock_tenant_manager.call.side_effect = _tenant_route(
            list_tenant_mcp_functions=_no_schema(),
            execute_tenant_mcp_function={},
        )
        backend = self._backend()
        backend._execute(
            {
                "slug": "demo",
                "branch": "prod",
                "function_name": "fn",
                "action": "act",
                # no params key
            }
        )
        _, kwargs = self._execute_calls()[0]
        self.assertEqual(kwargs["params"], {})

    def test_params_from_arguments_key(self):
        _mock_tenant_manager.call.side_effect = _tenant_route(
            list_tenant_mcp_functions=_no_schema(),
            execute_tenant_mcp_function={},
        )
        backend = self._backend()
        backend._execute(
            {
                "slug": "demo",
                "branch": "prod",
                "function_name": "fn",
                "action": "act",
                "arguments": {"x": 1},
            }
        )
        _, kwargs = self._execute_calls()[0]
        self.assertEqual(kwargs["params"], {"x": 1})

    def test_passes_event_access_token(self):
        _mock_tenant_manager.call.side_effect = _tenant_route(
            list_tenant_mcp_functions=_no_schema(),
            execute_tenant_mcp_function={},
        )
        backend = self._backend(access_token="bearer-abc123")
        backend._execute(
            {
                "slug": "s",
                "branch": "prod",
                "function_name": "f",
                "action": "a",
                "params": {},
            }
        )
        _, kwargs = self._execute_calls()[0]
        self.assertEqual(kwargs["access_token"], "bearer-abc123")

    def test_passes_organization_id(self):
        _mock_tenant_manager.call.side_effect = _tenant_route(
            list_tenant_mcp_functions=_no_schema(),
            execute_tenant_mcp_function={},
        )
        backend = self._backend(org_id="org-bbb")
        backend._execute(
            {
                "slug": "s",
                "branch": "prod",
                "function_name": "f",
                "action": "a",
                "params": {},
            }
        )
        _, kwargs = self._execute_calls()[0]
        self.assertEqual(kwargs["organization_id"], "org-bbb")

    # ------------------------------------------------------------------
    # Same-org guard — cross-org raises
    # ------------------------------------------------------------------
    def test_cross_org_raises_value_error(self):
        backend = self._backend(org_id="org-aaa")
        with self.assertRaises(ValueError) as ctx:
            backend._execute(
                {
                    "slug": "s",
                    "branch": "prod",
                    "function_name": "f",
                    "action": "a",
                    "params": {},
                    "tenant_organization_id": "org-DIFFERENT",
                }
            )
        self.assertIn("cross-org", str(ctx.exception))
        _mock_tenant_manager.call.assert_not_called()

    def test_same_org_explicit_id_allowed(self):
        _mock_tenant_manager.call.side_effect = _tenant_route(
            list_tenant_mcp_functions=_no_schema(),
            execute_tenant_mcp_function={},
        )
        backend = self._backend(org_id="org-aaa")
        # providing the same org id explicitly should not raise
        backend._execute(
            {
                "slug": "s",
                "branch": "prod",
                "function_name": "f",
                "action": "a",
                "params": {},
                "tenant_organization_id": "org-aaa",
            }
        )
        self.assertEqual(len(self._execute_calls()), 1)

    # ------------------------------------------------------------------
    # Non-dict params raises before any API call
    # ------------------------------------------------------------------
    def test_non_dict_params_raises(self):
        backend = self._backend()
        with self.assertRaises(ValueError) as ctx:
            backend._execute(
                {
                    "slug": "s",
                    "branch": "prod",
                    "function_name": "f",
                    "action": "a",
                    "params": "not-a-dict",
                }
            )
        self.assertIn("object", str(ctx.exception))
        _mock_tenant_manager.call.assert_not_called()

    def test_list_params_raises(self):
        backend = self._backend()
        with self.assertRaises(ValueError):
            backend._execute(
                {
                    "slug": "s",
                    "branch": "prod",
                    "function_name": "f",
                    "action": "a",
                    "params": [1, 2, 3],
                }
            )
        _mock_tenant_manager.call.assert_not_called()

    # ------------------------------------------------------------------
    # Branch validation
    # ------------------------------------------------------------------
    def test_invalid_branch_raises(self):
        backend = self._backend()
        with self.assertRaises(ValueError) as ctx:
            backend._execute(
                {
                    "slug": "s",
                    "branch": "staging",
                    "function_name": "f",
                    "action": "a",
                    "params": {},
                }
            )
        self.assertIn("branch", str(ctx.exception))
        _mock_tenant_manager.call.assert_not_called()

    def test_test_branch_accepted(self):
        _mock_tenant_manager.call.side_effect = _tenant_route(
            list_tenant_mcp_functions=_no_schema(),
            execute_tenant_mcp_function={},
        )
        backend = self._backend(branch="test")
        backend._execute(
            {
                "slug": "s",
                "branch": "test",
                "function_name": "f",
                "action": "a",
                "params": {},
            }
        )
        _, kwargs = self._execute_calls()[0]
        self.assertEqual(kwargs["branch"], "test")

    # ------------------------------------------------------------------
    # API manager error propagates
    # ------------------------------------------------------------------
    def test_api_manager_error_propagates(self):
        def _raise_on_execute(**kwargs):
            raise RuntimeError("control-plane 500")

        _mock_tenant_manager.call.side_effect = _tenant_route(
            list_tenant_mcp_functions=_no_schema(),
            execute_tenant_mcp_function=_raise_on_execute,
        )
        backend = self._backend()
        with self.assertRaises(RuntimeError):
            backend._execute(
                {
                    "slug": "s",
                    "branch": "prod",
                    "function_name": "f",
                    "action": "a",
                    "params": {},
                }
            )

    # ------------------------------------------------------------------
    # Confirm discover (_call_search) is untouched and uses same manager
    # ------------------------------------------------------------------
    def test_discover_uses_search_not_execute(self):
        _mock_tenant_manager.call.side_effect = _tenant_route(
            search_tenant_mcp_functions={"results": []},
        )
        backend = self._backend()
        backend._call_search(
            {
                "slug": "s",
                "branch": "prod",
                "conversation_state": "hello",
            }
        )
        call_name = _mock_tenant_manager.call.call_args[0][0]
        self.assertEqual(call_name, "search_tenant_mcp_functions")


class TestProvenanceInjection(ExecuteTestCase):
    """T1-T6 + T3b from the hotfix spec, plus the per-action whatsapp_message_id
    pin required by Directive 4."""

    # Mirrors the LIVE serving schema (chask-tenant-apis/gammavet@9e10c4e
    # PickupOrderCreateRequest) — NOT the vendored tenant-api-runtime copy,
    # which is inert for PROD (route_source=repo). source_event_uuid and
    # requested_by_phone are required; whatsapp_message_id is declared but
    # optional (part of the same dedupe key alongside source_event_uuid).
    PICKUP_CREATE_SCHEMA = {
        "source_event_uuid": {"type": "string", "required": True},
        "requested_by_phone": {"type": "string", "required": True},
        "whatsapp_message_id": {"type": "string", "required": False},
    }

    def test_T1_hallucinated_values_are_overwritten_with_real_state(self):
        """LLM supplies a hallucinated UUID + null phone -> outbound body carries
        the REAL event uuid and real phone."""
        _mock_tenant_manager.call.side_effect = _tenant_route(
            list_tenant_mcp_functions=_schema(
                "gammavet-pickup-orders", "create", self.PICKUP_CREATE_SCHEMA
            ),
            execute_tenant_mcp_function={"status": "ok"},
        )
        _mock_orchestrator_manager.call.side_effect = _orchestrator_route(
            get_orchestration_events=_inbound_events(("evt-real-123", "wamid.real")),
        )

        backend = self._backend(phone="56966255074")
        backend._execute(
            {
                "slug": "gammavet",
                "branch": "prod",
                "function_name": "gammavet-pickup-orders",
                "action": "create",
                "params": {
                    "source_event_uuid": "f47ac10b-58cc-4372-a567-0e02b2c3d479",  # hallucinated doc example
                    "requested_by_phone": None,
                },
            }
        )

        _, kwargs = self._execute_calls()[0]
        body = kwargs["params"]
        self.assertEqual(body["source_event_uuid"], "evt-real-123")
        self.assertEqual(body["requested_by_phone"], "56966255074")
        self.assertEqual(body["whatsapp_message_id"], "wamid.real")
        self.assertNotEqual(body["source_event_uuid"], "f47ac10b-58cc-4372-a567-0e02b2c3d479")

    def test_T2_omitted_fields_are_still_injected(self):
        _mock_tenant_manager.call.side_effect = _tenant_route(
            list_tenant_mcp_functions=_schema(
                "gammavet-pickup-orders", "create", self.PICKUP_CREATE_SCHEMA
            ),
            execute_tenant_mcp_function={"status": "ok"},
        )
        _mock_orchestrator_manager.call.side_effect = _orchestrator_route(
            get_orchestration_events=_inbound_events(("evt-real-123", "wamid.real")),
        )

        backend = self._backend(phone="56966255074")
        backend._execute(
            {
                "slug": "gammavet",
                "branch": "prod",
                "function_name": "gammavet-pickup-orders",
                "action": "create",
                "params": {},  # LLM omitted both fields entirely
            }
        )

        _, kwargs = self._execute_calls()[0]
        body = kwargs["params"]
        self.assertEqual(body["source_event_uuid"], "evt-real-123")
        self.assertEqual(body["requested_by_phone"], "56966255074")

    def test_T3_required_and_unavailable_raises_and_makes_no_control_plane_call(self):
        """B' unresolved (zero inbound messages) + field REQUIRED -> clear error,
        NO control-plane call made."""
        _mock_tenant_manager.call.side_effect = _tenant_route(
            list_tenant_mcp_functions=_schema(
                "gammavet-pickup-orders", "create", self.PICKUP_CREATE_SCHEMA
            ),
        )
        _mock_orchestrator_manager.call.side_effect = _orchestrator_route(
            get_orchestration_events={"orchestration_events": []},  # zero inbound
        )

        backend = self._backend()
        with self.assertRaises(ValueError) as ctx:
            backend._execute(
                {
                    "slug": "gammavet",
                    "branch": "prod",
                    "function_name": "gammavet-pickup-orders",
                    "action": "create",
                    "params": {"source_event_uuid": "hallucinated", "requested_by_phone": None},
                }
            )
        self.assertIn("source_event_uuid", str(ctx.exception))
        self.assertEqual(self._execute_calls(), [])

    def test_T3b_optional_and_unavailable_strips_key_never_forwards_llm_value(self):
        """Tier-B (schema lookup itself unavailable): field present in call_params,
        no authoritative value -> STRIP the key, never forward the LLM value."""
        _mock_tenant_manager.call.side_effect = _tenant_route(
            list_tenant_mcp_functions=lambda **_: (_ for _ in ()).throw(RuntimeError("control-plane down")),
            execute_tenant_mcp_function={"status": "ok"},
        )
        _mock_orchestrator_manager.call.side_effect = _orchestrator_route(
            get_orchestration_events={"orchestration_events": []},  # zero inbound -> unavailable
        )

        backend = self._backend()
        backend._execute(
            {
                "slug": "gammavet",
                "branch": "prod",
                "function_name": "gammavet-pickup-orders",
                "action": "trigger-delivery-failed",
                "params": {"whatsapp_message_id": "wamid.hallucinated-by-llm"},
            }
        )

        _, kwargs = self._execute_calls()[0]
        body = kwargs["params"]
        self.assertNotIn("whatsapp_message_id", body)
        self.assertNotEqual(body.get("whatsapp_message_id"), "wamid.hallucinated-by-llm")

    def test_T4_action_without_provenance_fields_is_unchanged(self):
        _mock_tenant_manager.call.side_effect = _tenant_route(
            list_tenant_mcp_functions=_schema(
                "get-order-fn", "get-order", {"order_id": {"type": "string", "required": True}}
            ),
            execute_tenant_mcp_function={"status": "ok"},
        )

        backend = self._backend()
        backend._execute(
            {
                "slug": "b-quellon",
                "branch": "prod",
                "function_name": "get-order-fn",
                "action": "get-order",
                "params": {"order_id": "1001"},
            }
        )

        _, kwargs = self._execute_calls()[0]
        self.assertEqual(kwargs["params"], {"order_id": "1001"})
        _mock_orchestrator_manager.call.assert_not_called()

    def test_tier_a_strips_undeclared_provenance_field_present_in_call_params(self):
        """An LLM-authored value for a provenance-named field the action does
        NOT declare must never cross to the tenant unmutated — tier A strips
        it just like tier B does, rather than trusting the tenant side to
        drop an unexpected key (chask_api's execute view does zero request
        validation)."""
        _mock_tenant_manager.call.side_effect = _tenant_route(
            list_tenant_mcp_functions=_schema(
                "get-order-fn", "get-order", {"order_id": {"type": "string", "required": True}}
            ),
            execute_tenant_mcp_function={"status": "ok"},
        )

        backend = self._backend()
        backend._execute(
            {
                "slug": "b-quellon",
                "branch": "prod",
                "function_name": "get-order-fn",
                "action": "get-order",
                "params": {"order_id": "1001", "requested_by_phone": "wamid.llm-invented-999"},
            }
        )

        _, kwargs = self._execute_calls()[0]
        self.assertEqual(kwargs["params"], {"order_id": "1001"})
        self.assertNotIn("requested_by_phone", kwargs["params"])

    def test_function_found_action_not_declared_hands_off_untouched(self):
        """The function IS in the authoritative listing but this particular
        action isn't one of its declared actions — that's still authoritative
        info (this action declares no provenance fields), so it's treated as
        an empty tier-A schema. Any provenance-named field present is still
        stripped (never forwarded unmutated), same as the undeclared-field
        case above."""
        _mock_tenant_manager.call.side_effect = _tenant_route(
            list_tenant_mcp_functions=_schema(
                "gammavet-pickup-orders", "create", self.PICKUP_CREATE_SCHEMA
            ),
            execute_tenant_mcp_function={"status": "ok"},
        )

        backend = self._backend()
        backend._execute(
            {
                "slug": "gammavet",
                "branch": "prod",
                "function_name": "gammavet-pickup-orders",
                "action": "some-other-undeclared-action",
                "params": {"requested_by_phone": "llm-authored"},
            }
        )

        _, kwargs = self._execute_calls()[0]
        self.assertNotIn("requested_by_phone", kwargs["params"])
        _mock_orchestrator_manager.call.assert_not_called()  # never even tried B-prime

    def test_T5_two_inbound_messages_never_bleeds_a_guessed_value(self):
        """REGRESSION: a session with TWO inbound messages must never have
        McpTenantFn silently pick one. It must fall through to C — if the
        field is required (as on the real incident's create action), that
        means a clear error, never two different-but-arbitrary UUIDs."""
        _mock_tenant_manager.call.side_effect = _tenant_route(
            list_tenant_mcp_functions=_schema(
                "gammavet-pickup-orders", "create", self.PICKUP_CREATE_SCHEMA
            ),
        )
        _mock_orchestrator_manager.call.side_effect = _orchestrator_route(
            get_orchestration_events=_inbound_events(
                ("evt-first", "wamid.first"), ("evt-second", "wamid.second")
            ),
        )

        backend = self._backend()
        with self.assertRaises(ValueError):
            backend._execute(
                {
                    "slug": "gammavet",
                    "branch": "prod",
                    "function_name": "gammavet-pickup-orders",
                    "action": "create",
                    "params": {"source_event_uuid": "hallucinated", "requested_by_phone": None},
                }
            )
        self.assertEqual(self._execute_calls(), [])

    def test_T6_discover_hides_provenance_from_llm_visible_parameters(self):
        function_data = {
            "uuid": "fn-uuid",
            "display_name": "gammavet-pickup-orders",
            "description": "Pickup orders",
            "actions": [
                {
                    "name": "create",
                    "method": "POST",
                    "path": "/pickup-orders/create",
                    "request_schema": {
                        "properties": {
                            "source_event_uuid": {"type": "string"},
                            "requested_by_phone": {"type": "string"},
                            "whatsapp_message_id": {"type": "string"},
                            "message": {"type": "string"},
                        },
                        "required": ["source_event_uuid", "requested_by_phone"],
                    },
                }
            ],
        }
        backend = self._backend()
        tool_def = backend._function_to_tool_def(function_data)
        create_params = tool_def["action_parameters"]["create"]
        self.assertNotIn("source_event_uuid", create_params)
        self.assertNotIn("requested_by_phone", create_params)
        self.assertNotIn("whatsapp_message_id", create_params)
        self.assertIn("message", create_params)

    # Deliberately NOT the create action's schema — provenance fields are
    # scoped per-action, so this fixture represents a genuinely different
    # action (e.g. a read-only "list" action) that declares none of them,
    # proving the per-action scoping rather than a specific tenant fact.
    UNDECLARED_ACTION_SCHEMA = {
        "clinic_id": {"type": "string", "required": False},
    }

    # Same shape as PICKUP_CREATE_SCHEMA but with source_event_uuid also
    # optional, so B'-unresolved doesn't short-circuit on the required-field
    # raise before whatsapp_message_id's own strip behavior can be observed
    # in isolation.
    OPTIONAL_PROVENANCE_SCHEMA = {
        "source_event_uuid": {"type": "string", "required": False},
        "requested_by_phone": {"type": "string", "required": True},
        "whatsapp_message_id": {"type": "string", "required": False},
    }

    def test_undeclared_provenance_field_is_stripped_not_forwarded(self):
        """A provenance-named field the ACTION doesn't declare must never
        cross to the tenant unmutated, regardless of what the LLM sends —
        provenance injection is scoped per-action (this uses a distinct
        undeclared-field fixture, not the live create-action schema)."""
        _mock_tenant_manager.call.side_effect = _tenant_route(
            list_tenant_mcp_functions=_schema(
                "gammavet-pickup-orders", "list", self.UNDECLARED_ACTION_SCHEMA
            ),
            execute_tenant_mcp_function={"status": "ok"},
        )

        backend = self._backend(phone="56966255074")
        backend._execute(
            {
                "slug": "gammavet",
                "branch": "prod",
                "function_name": "gammavet-pickup-orders",
                "action": "list",
                "params": {
                    "clinic_id": "clinic-1",
                    "whatsapp_message_id": "wamid.llm-invented",  # not declared on this action
                },
            }
        )

        _, kwargs = self._execute_calls()[0]
        body = kwargs["params"]
        # chask_api's execute view does zero request validation (see PR
        # description), so relying on the tenant's own pydantic model to drop
        # an unexpected field would trust the exact opaque boundary this fix
        # exists to distrust. Strong contract: the key must be absent
        # entirely, not merely different.
        self.assertNotIn("whatsapp_message_id", body)
        _mock_orchestrator_manager.call.assert_not_called()

    def test_create_action_injects_real_wamid_when_bprime_resolves(self):
        """Against the LIVE serving schema (chask-tenant-apis/gammavet@9e10c4e
        PickupOrderCreateRequest — whatsapp_message_id IS declared, optional,
        part of the same dedupe key as source_event_uuid): when B-prime
        resolves the triggering message, the REAL wamid is injected and the
        LLM-authored value is never forwarded."""
        _mock_tenant_manager.call.side_effect = _tenant_route(
            list_tenant_mcp_functions=_schema(
                "gammavet-pickup-orders", "create", self.PICKUP_CREATE_SCHEMA
            ),
            execute_tenant_mcp_function={"status": "ok"},
        )
        _mock_orchestrator_manager.call.side_effect = _orchestrator_route(
            get_orchestration_events=_inbound_events(("evt-real-123", "wamid.real")),
        )

        backend = self._backend(phone="56966255074")
        backend._execute(
            {
                "slug": "gammavet",
                "branch": "prod",
                "function_name": "gammavet-pickup-orders",
                "action": "create",
                "params": {
                    "source_event_uuid": "hallucinated",
                    "requested_by_phone": None,
                    "whatsapp_message_id": "wamid.llm-invented",
                },
            }
        )

        _, kwargs = self._execute_calls()[0]
        body = kwargs["params"]
        self.assertEqual(body["whatsapp_message_id"], "wamid.real")
        self.assertNotEqual(body["whatsapp_message_id"], "wamid.llm-invented")

    def test_create_action_strips_wamid_when_bprime_unresolved_zero_inbound(self):
        _mock_tenant_manager.call.side_effect = _tenant_route(
            list_tenant_mcp_functions=_schema(
                "gammavet-pickup-orders", "create", self.OPTIONAL_PROVENANCE_SCHEMA
            ),
            execute_tenant_mcp_function={"status": "ok"},
        )
        _mock_orchestrator_manager.call.side_effect = _orchestrator_route(
            get_orchestration_events={"orchestration_events": []},  # zero inbound
        )

        backend = self._backend(phone="56966255074")
        backend._execute(
            {
                "slug": "gammavet",
                "branch": "prod",
                "function_name": "gammavet-pickup-orders",
                "action": "create",
                "params": {"whatsapp_message_id": "wamid.llm-invented"},
            }
        )

        _, kwargs = self._execute_calls()[0]
        body = kwargs["params"]
        self.assertNotIn("whatsapp_message_id", body)
        self.assertNotIn("source_event_uuid", body)

    def test_create_action_strips_wamid_when_bprime_unresolved_two_or_more_inbound(self):
        _mock_tenant_manager.call.side_effect = _tenant_route(
            list_tenant_mcp_functions=_schema(
                "gammavet-pickup-orders", "create", self.OPTIONAL_PROVENANCE_SCHEMA
            ),
            execute_tenant_mcp_function={"status": "ok"},
        )
        _mock_orchestrator_manager.call.side_effect = _orchestrator_route(
            get_orchestration_events=_inbound_events(("evt-a", "wamid.a"), ("evt-b", "wamid.b")),
        )

        backend = self._backend(phone="56966255074")
        backend._execute(
            {
                "slug": "gammavet",
                "branch": "prod",
                "function_name": "gammavet-pickup-orders",
                "action": "create",
                "params": {"whatsapp_message_id": "wamid.llm-invented"},
            }
        )

        _, kwargs = self._execute_calls()[0]
        body = kwargs["params"]
        self.assertNotIn("whatsapp_message_id", body)
        self.assertNotIn("source_event_uuid", body)


class TestBPrimeResolution(ExecuteTestCase):
    """Direct tests of _resolve_trigger_whatsapp_event's uniqueness gate."""

    def _backend_ready_for_bprime(self, **event_kwargs):
        backend = self._backend(**event_kwargs)
        backend._provenance_deadline_at = _mod.time.monotonic() + _mod.BPRIME_DEADLINE_SECONDS
        return backend

    def test_resolves_on_single_inbound_message_session(self):
        _mock_orchestrator_manager.call.side_effect = _orchestrator_route(
            get_orchestration_events=_inbound_events(("evt-only", "wamid.only")),
        )
        backend = self._backend_ready_for_bprime()
        trigger = backend._resolve_trigger_whatsapp_event()
        self.assertEqual(trigger, {"event_id": "evt-only", "message_id": "wamid.only"})

    def test_falls_through_to_none_on_zero_inbound_messages(self):
        _mock_orchestrator_manager.call.side_effect = _orchestrator_route(
            get_orchestration_events={"orchestration_events": []},
        )
        backend = self._backend_ready_for_bprime()
        self.assertIsNone(backend._resolve_trigger_whatsapp_event())

    def test_falls_through_to_none_on_two_or_more_without_tie_break(self):
        _mock_orchestrator_manager.call.side_effect = _orchestrator_route(
            get_orchestration_events=_inbound_events(
                ("evt-a", "wamid.a"), ("evt-b", "wamid.b"), ("evt-c", "wamid.c")
            ),
        )
        backend = self._backend_ready_for_bprime()
        self.assertIsNone(backend._resolve_trigger_whatsapp_event())

    def test_degrades_never_throws_on_lookup_failure(self):
        def _raise(**_):
            raise RuntimeError("orchestrator unreachable")

        _mock_orchestrator_manager.call.side_effect = _orchestrator_route(
            get_orchestration_events=_raise,
        )
        backend = self._backend_ready_for_bprime()
        self.assertIsNone(backend._resolve_trigger_whatsapp_event())

    def test_no_session_uuid_is_unresolved_without_a_call(self):
        backend = self._backend_ready_for_bprime(session_uuid=None)
        self.assertIsNone(backend._resolve_trigger_whatsapp_event())
        _mock_orchestrator_manager.call.assert_not_called()

    def test_deadline_exceeded_skips_the_call_entirely(self):
        backend = self._backend()
        backend._provenance_deadline_at = _mod.time.monotonic() - 1  # already past
        result = backend._resolve_trigger_whatsapp_event()
        self.assertIsNone(result)
        _mock_orchestrator_manager.call.assert_not_called()

    def test_call_timeout_is_capped_to_remaining_deadline_budget(self):
        """The two timeout budgets must compose: the call's own timeout must
        never exceed what's left of the wall-clock deadline, even though it
        also has its own independent per-call cap (BPRIME_LOOKUP_TIMEOUT)."""
        _mock_orchestrator_manager.call.side_effect = _orchestrator_route(
            get_orchestration_events=_inbound_events(("evt-only", "wamid.only")),
        )
        backend = self._backend()
        # Only 2s left on the wall-clock deadline, well under BPRIME_LOOKUP_TIMEOUT (10s).
        backend._provenance_deadline_at = _mod.time.monotonic() + 2
        backend._resolve_trigger_whatsapp_event()

        _, kwargs = _mock_orchestrator_manager.call.call_args
        self.assertLessEqual(kwargs["timeout"], 2.1)
        self.assertLess(kwargs["timeout"], _mod.BPRIME_LOOKUP_TIMEOUT)

    def test_result_is_cached_per_invocation(self):
        _mock_orchestrator_manager.call.side_effect = _orchestrator_route(
            get_orchestration_events=_inbound_events(("evt-only", "wamid.only")),
        )
        backend = self._backend_ready_for_bprime()
        backend._resolve_trigger_whatsapp_event()
        backend._resolve_trigger_whatsapp_event()
        self.assertEqual(_mock_orchestrator_manager.call.call_count, 1)


class TestTierACaching(ExecuteTestCase):
    def test_schema_lookup_is_cached_per_slug_branch(self):
        _mock_tenant_manager.call.side_effect = _tenant_route(
            list_tenant_mcp_functions=_no_schema(),
            execute_tenant_mcp_function={"status": "ok"},
        )
        backend = self._backend()
        backend._execute(
            {
                "slug": "demo",
                "branch": "prod",
                "function_name": "fn-a",
                "action": "act",
                "params": {},
            }
        )
        backend._execute(
            {
                "slug": "demo",
                "branch": "prod",
                "function_name": "fn-b",
                "action": "act",
                "params": {},
            }
        )
        schema_calls = [
            c for c in _mock_tenant_manager.call.call_args_list if c.args[0] == "list_tenant_mcp_functions"
        ]
        self.assertEqual(len(schema_calls), 1)


if __name__ == "__main__":
    unittest.main()
