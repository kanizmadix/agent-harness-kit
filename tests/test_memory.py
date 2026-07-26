from agent_harness_kit.core.context import HarnessContext
from agent_harness_kit.core.memory import InMemoryProvider, SQLiteMemoryProvider


def test_in_memory_load_missing_returns_fresh_context():
    mem = InMemoryProvider()
    loaded = mem.load("missing")
    assert loaded.session_id == "missing"
    assert loaded.messages == []


def test_in_memory_save_then_load_round_trips():
    mem = InMemoryProvider()
    ctx = HarnessContext(session_id="s1")
    ctx.add_message("user", "hi")
    mem.save("s1", ctx)

    loaded = mem.load("s1")
    assert loaded.messages == [{"role": "user", "content": "hi"}]


def test_in_memory_is_isolated_per_provider_instance():
    mem1 = InMemoryProvider()
    mem2 = InMemoryProvider()
    ctx = HarnessContext(session_id="s1")
    ctx.add_message("user", "hi")
    mem1.save("s1", ctx)

    assert mem2.load("s1").messages == []


def test_sqlite_load_missing_returns_fresh_context(tmp_path):
    db_path = str(tmp_path / "harness.db")
    mem = SQLiteMemoryProvider(db_path)
    loaded = mem.load("missing")
    assert loaded.session_id == "missing"
    assert loaded.messages == []


def test_sqlite_save_then_load_round_trips(tmp_path):
    db_path = str(tmp_path / "harness.db")
    mem = SQLiteMemoryProvider(db_path)

    ctx = HarnessContext(session_id="s1")
    ctx.add_message("user", "hi")
    ctx.scratchpad["objective"] = "test"
    mem.save("s1", ctx)

    loaded = mem.load("s1")
    assert loaded.messages == [{"role": "user", "content": "hi"}]
    assert loaded.scratchpad == {"objective": "test"}


def test_sqlite_save_overwrites_existing_session(tmp_path):
    db_path = str(tmp_path / "harness.db")
    mem = SQLiteMemoryProvider(db_path)

    ctx = HarnessContext(session_id="s1")
    ctx.add_message("user", "first")
    mem.save("s1", ctx)

    ctx.add_message("user", "second")
    mem.save("s1", ctx)

    loaded = mem.load("s1")
    assert len(loaded.messages) == 2


def test_sqlite_persists_across_provider_instances(tmp_path):
    db_path = str(tmp_path / "harness.db")
    mem1 = SQLiteMemoryProvider(db_path)
    ctx = HarnessContext(session_id="s1")
    ctx.add_message("user", "hi")
    mem1.save("s1", ctx)

    mem2 = SQLiteMemoryProvider(db_path)
    loaded = mem2.load("s1")
    assert loaded.messages == [{"role": "user", "content": "hi"}]


def test_sqlite_keeps_sessions_separate(tmp_path):
    db_path = str(tmp_path / "harness.db")
    mem = SQLiteMemoryProvider(db_path)

    ctx1 = HarnessContext(session_id="s1")
    ctx1.add_message("user", "session one")
    mem.save("s1", ctx1)

    ctx2 = HarnessContext(session_id="s2")
    ctx2.add_message("user", "session two")
    mem.save("s2", ctx2)

    assert mem.load("s1").messages == [{"role": "user", "content": "session one"}]
    assert mem.load("s2").messages == [{"role": "user", "content": "session two"}]
