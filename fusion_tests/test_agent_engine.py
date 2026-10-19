"""Agent engine integration smoke test that does not create Tk windows.

CI can run the real GUI with installed customtkinter/openai. This test injects
minimum transport/toolkit placeholders only for a headless engine lifecycle.
"""
import asyncio
import importlib
import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src'))


@pytest.mark.asyncio
async def test_encrypted_agent_boot_memories_tools_and_seal(monkeypatch,tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path/'workspace').mkdir()
    (tmp_path/'workspace'/'sample.py').write_text('def hello_world():\n    return 123\n')
    try:
        import customtkinter  # noqa: F401
    except ImportError:
        monkeypatch.setitem(sys.modules,'customtkinter',types.ModuleType('customtkinter'))
    try:
        from openai import AsyncOpenAI  # noqa: F401
    except ImportError:
        openai = types.ModuleType('openai')
        class FakeClient:
            def __init__(self, **kwargs): pass
        openai.AsyncOpenAI = FakeClient
        monkeypatch.setitem(sys.modules,'openai',openai)
    from agent_gui import AgentEngine
    key = os.urandom(32)
    events = []
    eng = AgentEngine(events.append)
    result = await eng.boot(key,{'humoid_provider':'litert','litert_model':'gemma-4-E2B-it',
                                 'humoid_workspace':str(tmp_path/'workspace')})
    assert 'ready' in result.lower()
    assert eng.harness is not None
    assert eng.harness.tools.approve_write is not None
    assert len(eng.harness.tools.schemas()) >= 22
    assert 'sample.py' in await eng.tool('list_files',{'path':'.'})
    assert 'hello_world' in await eng.tool('read_file',{'path':'sample.py'})
    await eng.add_memory('Debugging sample.py: hello_world must return 123', 'project','episodic')
    hits = await eng.search('sample.py hello_world')
    assert hits and 'hello_world' in hits[0].text
    assert eng.vault is not None
    eng.vault.persist()
    await eng.shutdown()
    assert not (tmp_path/'.humoid'/'memory.sqlite3').exists()
    assert (tmp_path/'.humoid'/'agent_memory.sqlite3.aes').exists()
    # Reopen with the same key; persistence and Rhomgaic still work.
    await eng.boot(key, {'humoid_provider':'litert','litert_model':'gemma-4-E2B-it'})
    recalled = await eng.search('sample.py hello_world')
    assert recalled and 'hello_world' in recalled[0].text
    await eng.shutdown()
