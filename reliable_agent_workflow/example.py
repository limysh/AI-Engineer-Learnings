from agent_runtime import AgentRuntime, ToolRequest, TransientToolError


attempts = {"count": 0}


def flaky_notification(payload):
    attempts["count"] += 1
    if attempts["count"] < 2:
        raise TransientToolError("notification provider timed out")
    return {"delivered": True, "recipient": payload["recipient"]}


runtime = AgentRuntime(
    {"send_notification": flaky_notification},
    sleeper=lambda _: None,  # keep the demo instant
)

request = ToolRequest(
    tool_name="send_notification",
    payload={"recipient": "demo@example.com", "message": "hello"},
    idempotency_key="notification-42",
)

first = runtime.execute("run-1", request)
duplicate = runtime.execute("run-2", request)

print("first:", first.status.value, first.result, "attempts:", first.attempts)
print("duplicate:", duplicate.status.value, duplicate.result)
print("duplicate events:", [event.kind for event in duplicate.events])
