"""Unit tests for FunctionBackend._execute — control-plane path."""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock, patch


def _make_event(org_id: str = "org-aaa", access_token: str = "tok-xyz", branch: str = "prod"):
    """Build a minimal OrchestrationEvent-shaped mock."""
    org = MagicMock()
    org.organization_id = org_id
    event = MagicMock()
    event.organization = org
    event.access_token = access_token
    event.branch = branch
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
    mock_manager = MagicMock()
    api_tmr.tenant_mcp_api_manager = mock_manager
    sys.modules.setdefault("api", api_pkg)
    sys.modules.setdefault("api.tenant_mcp_requests", api_tmr)

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
    return mod, mock_manager


_mod, _mock_manager = _load_module()
FunctionBackend = _mod.FunctionBackend


class TestExecuteControlPlane(unittest.TestCase):
    def setUp(self):
        _mock_manager.call.side_effect = None
        _mock_manager.call.return_value = None
        _mock_manager.reset_mock()

    def _backend(self, org_id="org-aaa", access_token="tok-xyz", branch="prod"):
        return FunctionBackend(_make_event(org_id=org_id, access_token=access_token, branch=branch))

    # ------------------------------------------------------------------
    # Happy path — forwards all required fields and returns body
    # ------------------------------------------------------------------
    def test_forwards_all_kwargs_to_api_manager(self):
        expected_body = {"id": "1001", "status": "processing"}
        _mock_manager.call.return_value = expected_body

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

        _mock_manager.call.assert_called_once_with(
            "execute_tenant_mcp_function",
            slug="b-quellon",
            branch="prod",
            function_name="get-order",
            action="get-order",
            params={"order_id": "1001"},
            access_token="tok-xyz",
            organization_id="org-aaa",
            timeout=_mod.CONTROL_PLANE_TIMEOUT,
        )
        self.assertEqual(result, expected_body)

    def test_returns_body_unchanged(self):
        body = {"key": "value", "nested": {"a": 1}}
        _mock_manager.call.return_value = body
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
        _mock_manager.call.return_value = {}
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
        _, kwargs = _mock_manager.call.call_args
        self.assertEqual(kwargs["params"], {})

    def test_params_from_arguments_key(self):
        _mock_manager.call.return_value = {}
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
        _, kwargs = _mock_manager.call.call_args
        self.assertEqual(kwargs["params"], {"x": 1})

    def test_passes_event_access_token(self):
        _mock_manager.call.return_value = {}
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
        _, kwargs = _mock_manager.call.call_args
        self.assertEqual(kwargs["access_token"], "bearer-abc123")

    def test_passes_organization_id(self):
        _mock_manager.call.return_value = {}
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
        _, kwargs = _mock_manager.call.call_args
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
        _mock_manager.call.assert_not_called()

    def test_same_org_explicit_id_allowed(self):
        _mock_manager.call.return_value = {}
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
        _mock_manager.call.assert_called_once()

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
        _mock_manager.call.assert_not_called()

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
        _mock_manager.call.assert_not_called()

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
        _mock_manager.call.assert_not_called()

    def test_test_branch_accepted(self):
        _mock_manager.call.return_value = {}
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
        _, kwargs = _mock_manager.call.call_args
        self.assertEqual(kwargs["branch"], "test")

    # ------------------------------------------------------------------
    # API manager error propagates
    # ------------------------------------------------------------------
    def test_api_manager_error_propagates(self):
        _mock_manager.call.side_effect = RuntimeError("control-plane 500")
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
        _mock_manager.call.return_value = {"results": []}
        backend = self._backend()
        backend._call_search(
            {
                "slug": "s",
                "branch": "prod",
                "conversation_state": "hello",
            }
        )
        call_name = _mock_manager.call.call_args[0][0]
        self.assertEqual(call_name, "search_tenant_mcp_functions")


if __name__ == "__main__":
    unittest.main()
