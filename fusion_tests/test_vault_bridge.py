"""Storage and memory integration tests independent of GUI display/toolkits."""
import asyncio
import os
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from agent_vault import AgentVault, _seal, _open, rekey_agent_vault, advanced_context_for_gui
from humoid_tui.config import Settings
from humoid_tui.memory import MemoryRouter


def test_aes_authentication_and_random_nonces():
    key = os.urandom(32)
    a = _seal(b'my private memory', key)
    b = _seal(b'my private memory', key)
    assert a != b
    assert _open(a, key) == b'my private memory'
    with pytest.raises(Exception): _open(a, os.urandom(32))
    with pytest.raises(Exception): _open(a[:-1] + bytes([a[-1] ^ 7]), key)


def test_sqlite_mount_persist_reopen_and_cleanup(tmp_path):
    key = os.urandom(32)
    vault = AgentVault(key, tmp_path / '.humoid')
    dbpath, chromapath = vault.mount()
    with sqlite3.connect(dbpath) as db:
        db.execute('CREATE TABLE memories (text TEXT)')
        db.execute('INSERT INTO memories VALUES (?)', ('secrets and facts about code.py',))
        db.commit()
    with sqlite3.connect(chromapath) as db:
        db.execute('CREATE TABLE chromatic (token TEXT)')
        db.execute('INSERT INTO chromatic VALUES (?)', ('AST-symbol',))
        db.commit()
    active = vault.active_dir
    vault.close()
    assert not active.exists()
    assert not dbpath.exists()
    assert b'secrets and facts' not in (tmp_path/'.humoid'/'agent_memory.sqlite3.aes').read_bytes()
    reopened = AgentVault(key, tmp_path/'.humoid')
    dbpath, chromapath = reopened.mount()
    with sqlite3.connect(dbpath) as db:
        assert db.execute('SELECT text FROM memories').fetchone()[0] == 'secrets and facts about code.py'
    with sqlite3.connect(chromapath) as db:
        assert db.execute('SELECT token FROM chromatic').fetchone()[0] == 'AST-symbol'
    reopened.close()


def test_rekey_old_key_cannot_read(tmp_path, monkeypatch):
    import agent_vault as module
    monkeypatch.setattr(module, 'ROOT', tmp_path/'.humoid')
    k1, k2 = os.urandom(32), os.urandom(32)
    vault = AgentVault(k1, module.ROOT)
    vault.write_json('agent_settings.json.aes', {'secret': 'not plaintext'})
    mempath, _ = vault.mount()
    with sqlite3.connect(mempath) as db:
        db.execute('CREATE TABLE t (x TEXT)')
        db.execute("INSERT INTO t VALUES ('something private')")
        db.commit()
    vault.close()
    rekey_agent_vault(k1,k2)
    assert AgentVault(k2, module.ROOT).read_json('agent_settings.json.aes', {}) == {'secret': 'not plaintext'}
    with pytest.raises(Exception): AgentVault(k1, module.ROOT).read_json('agent_settings.json.aes', {})
    correct = AgentVault(k2, module.ROOT)
    correct.mount()
    correct.close()
    with pytest.raises(Exception): AgentVault(k1, module.ROOT).mount()


@pytest.mark.asyncio
async def test_original_rhomgaic_memory_roundtrip_under_vault(tmp_path, monkeypatch):
    import agent_vault as module
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, 'ROOT', Path('.humoid'))
    k = os.urandom(32)
    vault = AgentVault(k)
    memory, chromatic = vault.mount()
    settings = Settings(
        _env_file=None, humoid_memory_backend='sqlite',
        humoid_memory_db=memory, humoid_rhomgaic_db=chromatic,
        humoid_workspace=tmp_path/'workspace',
    )
    router = MemoryRouter(settings)
    await router.initialize()
    await router.add('The password rotation code is in vault_cipher.py and uses AES GCM.',
                     channel='test',memory_tier='semantic')
    hits = await router.search('vault_cipher.py AES rotation')
    assert hits and 'password rotation' in hits[0].text
    await router.close()
    vault.close()
    # Local GUI chat must be able to recall from this exact advanced-memory DB.
    result = await asyncio.to_thread(advanced_context_for_gui,k,'vault_cipher.py password rotation')
    assert 'vault_cipher.py' in result
    assert 'password rotation' in result
    assert not Path('memory.sqlite3').exists()


def test_failed_persist_does_not_destroy_working_memory(tmp_path, monkeypatch):
    key = os.urandom(32)
    vault = AgentVault(key, tmp_path / '.humoid')
    path, _ = vault.mount()
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE secrets (text TEXT)')
        db.execute("INSERT INTO secrets VALUES ('cannot lose this')")
        db.commit()
    original_persist = vault.persist
    calls = 0

    def fail_once():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError('simulated disk full')
        return original_persist()

    monkeypatch.setattr(vault, 'persist', fail_once)
    with pytest.raises(OSError, match='disk full'):
        vault.close()
    assert path.exists() and vault.active_dir is not None
    vault.close()
    recovered = AgentVault(key, tmp_path / '.humoid')
    mem, _ = recovered.mount()
    with sqlite3.connect(mem) as db:
        assert db.execute('SELECT text FROM secrets').fetchone()[0] == 'cannot lose this'
    recovered.close()


def test_partial_agent_rekey_is_rolled_back_on_io_error(tmp_path, monkeypatch):
    import agent_vault as module
    monkeypatch.setattr(module, 'ROOT', tmp_path / '.humoid')
    k1, k2 = os.urandom(32), os.urandom(32)
    vault = AgentVault(k1, module.ROOT)
    vault.write_json('agent_settings.json.aes', {'ctx': 'working'})
    vault.write_json('agent_sessions.json.aes', {'chats': {'one': ['hello']}})
    writes = 0
    actual_write = module._atomic_write

    def fail_second(path, content):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError('simulated write failure')
        return actual_write(path, content)

    monkeypatch.setattr(module, '_atomic_write', fail_second)
    with pytest.raises(OSError, match='write failure'):
        rekey_agent_vault(k1, k2)
    assert vault.read_json('agent_settings.json.aes', {}) == {'ctx': 'working'}
    assert vault.read_json('agent_sessions.json.aes', {}) == {'chats': {'one': ['hello']}}
    with pytest.raises(Exception):
        AgentVault(k2, module.ROOT).read_json('agent_settings.json.aes', {})
