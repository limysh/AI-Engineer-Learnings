import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent_runtime import (
    AgentRuntime,
    PermanentToolError,
    RunStatus,
    ToolRequest,
    TransientToolError,
)


class AgentRuntimeTests(unittest.TestCase):
    def test_success(self):
        runtime = AgentRuntime({"echo": lambda payload: dict(payload)}, sleeper=lambda _: None)
        request = ToolRequest("echo", {"value": 7}, "echo-7")

        state = runtime.execute("run-success", request)

        self.assertEqual(RunStatus.SUCCEEDED, state.status)
        self.assertEqual({"value": 7}, state.result)
        self.assertEqual(1, state.attempts)

    def test_transient_failure_is_retried(self):
        attempts = {"count": 0}

        def flaky(_):
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise TransientToolError("temporary outage")
            return "ok"

        runtime = AgentRuntime({"flaky": flaky}, max_attempts=3, sleeper=lambda _: None)
        request = ToolRequest("flaky", {}, "retry-key")

        state = runtime.execute("run-retry", request)

        self.assertEqual(RunStatus.SUCCEEDED, state.status)
        self.assertEqual(3, state.attempts)
        self.assertEqual("ok", state.result)

    def test_retry_budget_exhaustion_escalates(self):
        def always_fails(_):
            raise TransientToolError("dependency unavailable")

        runtime = AgentRuntime({"tool": always_fails}, max_attempts=2, sleeper=lambda _: None)
        request = ToolRequest("tool", {}, "failure-key")

        state = runtime.execute("run-escalate", request)

        self.assertEqual(RunStatus.NEEDS_REVIEW, state.status)
        self.assertEqual(2, state.attempts)
        self.assertIn("retry_budget_exhausted", [event.kind for event in state.events])

    def test_permanent_failure_is_not_retried(self):
        def invalid(_):
            raise PermanentToolError("invalid tool arguments")

        runtime = AgentRuntime({"tool": invalid}, sleeper=lambda _: None)
        request = ToolRequest("tool", {}, "permanent-key")

        state = runtime.execute("run-permanent", request)

        self.assertEqual(RunStatus.FAILED, state.status)
        self.assertEqual(1, state.attempts)

    def test_idempotency_suppresses_duplicate_side_effect(self):
        calls = {"count": 0}

        def mutate(_):
            calls["count"] += 1
            return {"created_id": 123}

        runtime = AgentRuntime({"mutate": mutate}, sleeper=lambda _: None)
        request = ToolRequest("mutate", {}, "same-operation")

        first = runtime.execute("run-a", request)
        second = runtime.execute("run-b", request)

        self.assertEqual(RunStatus.SUCCEEDED, first.status)
        self.assertEqual(RunStatus.SUCCEEDED, second.status)
        self.assertEqual(1, calls["count"])
        self.assertIn("duplicate_suppressed", [event.kind for event in second.events])

    def test_high_risk_action_waits_for_approval(self):
        calls = {"count": 0}

        def risky(_):
            calls["count"] += 1
            return "done"

        runtime = AgentRuntime({"risky": risky}, sleeper=lambda _: None)
        request = ToolRequest("risky", {}, "risk-key", requires_review=True)

        waiting = runtime.execute("run-review", request)

        self.assertEqual(RunStatus.NEEDS_REVIEW, waiting.status)
        self.assertEqual(0, calls["count"])

        approved = runtime.execute("run-review", request, approved=True)

        self.assertEqual(RunStatus.SUCCEEDED, approved.status)
        self.assertEqual(1, calls["count"])


if __name__ == "__main__":
    unittest.main()
