import pathlib
import sys
import threading
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

    def test_none_result_is_still_cached(self):
        calls = {"count": 0}

        def mutate(_):
            calls["count"] += 1
            return None

        runtime = AgentRuntime({"mutate": mutate}, sleeper=lambda _: None)
        request = ToolRequest("mutate", {}, "none-result")

        first = runtime.execute("run-none-a", request)
        second = runtime.execute("run-none-b", request)

        self.assertEqual(RunStatus.SUCCEEDED, first.status)
        self.assertEqual(RunStatus.SUCCEEDED, second.status)
        self.assertIsNone(second.result)
        self.assertEqual(1, calls["count"])
        self.assertIn("duplicate_suppressed", [event.kind for event in second.events])

    def test_concurrent_duplicate_waits_without_repeating_side_effect(self):
        calls = {"count": 0}
        tool_started = threading.Event()
        allow_tool_to_finish = threading.Event()
        completed_states = []

        def slow_mutation(_):
            calls["count"] += 1
            tool_started.set()
            if not allow_tool_to_finish.wait(timeout=2):
                raise RuntimeError("test timed out waiting to release tool")
            return {"created_id": 456}

        runtime = AgentRuntime({"mutate": slow_mutation}, sleeper=lambda _: None)
        request = ToolRequest("mutate", {}, "concurrent-operation")
        owner = threading.Thread(
            target=lambda: completed_states.append(runtime.execute("run-owner", request))
        )

        owner.start()
        self.assertTrue(tool_started.wait(timeout=2))
        duplicate = runtime.execute("run-duplicate", request)

        self.assertEqual(RunStatus.WAITING, duplicate.status)
        self.assertEqual(0, duplicate.attempts)
        self.assertIn("duplicate_in_progress", [event.kind for event in duplicate.events])
        self.assertEqual(1, calls["count"])

        allow_tool_to_finish.set()
        owner.join(timeout=2)
        self.assertFalse(owner.is_alive())
        self.assertEqual(RunStatus.SUCCEEDED, completed_states[0].status)

        replay = runtime.execute("run-replay", request)
        self.assertEqual(RunStatus.SUCCEEDED, replay.status)
        self.assertEqual({"created_id": 456}, replay.result)
        self.assertEqual(1, calls["count"])

    def test_failed_execution_releases_idempotency_claim(self):
        calls = {"count": 0}

        def fail_once(_):
            calls["count"] += 1
            if calls["count"] == 1:
                raise PermanentToolError("invalid first attempt")
            return "recovered"

        runtime = AgentRuntime({"tool": fail_once}, sleeper=lambda _: None)
        request = ToolRequest("tool", {}, "released-after-failure")

        failed = runtime.execute("run-failed", request)
        retried = runtime.execute("run-retried", request)

        self.assertEqual(RunStatus.FAILED, failed.status)
        self.assertEqual(RunStatus.SUCCEEDED, retried.status)
        self.assertEqual("recovered", retried.result)
        self.assertEqual(2, calls["count"])

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
