from agent_harness_kit.core.context import HarnessContext


def test_add_message_appends_role_and_content():
    ctx = HarnessContext(session_id="s1")
    ctx.add_message("user", "hello")
    assert ctx.messages == [{"role": "user", "content": "hello"}]


def test_round_trip_dict():
    ctx = HarnessContext(session_id="s1")
    ctx.add_message("user", "hello")
    ctx.scratchpad["k"] = "v"
    ctx.artifacts["file.txt"] = "contents"
    ctx.tools_state["agent_a"] = {"ok": True}

    restored = HarnessContext.from_dict(ctx.to_dict())

    assert restored.session_id == ctx.session_id
    assert restored.messages == ctx.messages
    assert restored.scratchpad == ctx.scratchpad
    assert restored.artifacts == ctx.artifacts
    assert restored.tools_state == ctx.tools_state
