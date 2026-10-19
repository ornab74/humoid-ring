"""PRISM tests: SQLite authority, outbox, recurrence, vault and sealed Weaviate."""
import os
import sys
import tarfile
import io
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]

from agent_vault import AgentVault, _open, _seal, rekey_agent_vault
from prism_core import PrismStore, ring_color, spectrum, vectorize
from prism_embedded import (
    _pack, _unpack, _key, ARCHIVE_MAGIC, ARCHIVE_NAME,
    EmbeddedWeaviateVault,
)
from humoid_tui.config import secure_weaviate_version


def test_weaviate_security_floor_migrates_old_pins_and_rejects_aliases():
    assert secure_weaviate_version('1.33.0') == ('1.40.0', True)
    assert secure_weaviate_version('1.40.0') == ('1.40.0', False)
    assert secure_weaviate_version('1.38.0') == ('1.38.0', False)
    assert secure_weaviate_version('1.39.10') == ('1.39.10', False)
    assert secure_weaviate_version('v1.41.2') == ('1.41.2', False)
    with pytest.raises(ValueError, match='exact stable'):
        secure_weaviate_version('latest')
    with pytest.raises(ValueError, match='exact stable'):
        secure_weaviate_version('1.40.0-rc.1')


def test_spectral_colors_and_lexical_vectors():
    s=spectrum('def api(): import config from code.py')
    assert abs(sum(s.values())-1) < 1e-6
    assert max(s, key=s.get) == 'code'
    assert ring_color(s,3).startswith('#')
    assert vectorize('code.py') == vectorize('code.py')
    assert abs(sum(x*x for x in vectorize('code.py'))-1)<1e-8


def test_store_dual_outbox_revision_bridge_and_forget(tmp_path):
    store=PrismStore(tmp_path/'prism.sqlite3')
    item=store.remember('Tests fail in config.py due to sqlite error', project='project X')
    assert item['consent_index']==1
    assert store.pending_index()
    assert store.search('sqlite config.py',project='project X')[0]['id']==item['id']
    uid=item['id']
    edited=store.revise(uid,text='Tests passed after changing sqlite timeout')
    assert edited['revision']==2
    assert store.import_legacy('tui1','Fix for foo.py',tier='procedural',verified=True)
    assert not store.import_legacy('tui1','Fix for foo.py',tier='procedural',verified=True)
    assert store.import_legacy('tui1','Fix for foo.py updated')
    assert store.summary_packet('timeout sqlite',budget=300)
    for row in store.pending_index(): store.index_ack(row['id'])
    assert store.queue_full_reconciliation() >=2
    store.forget(uid)
    assert not store.get(uid)['text']
    assert all(x['id'] != uid for x in store.search('timeout'))
    assert store.forget_source('tui1')
    assert not store.import_legacy('tui1','try resurrect')


def test_strict_ring_and_project_isolation(tmp_path):
    store = PrismStore(tmp_path/'isolated.sqlite3')
    own = store.remember('alpha release preference', ring=3, project='gui-chat')
    other = store.remember('alpha release preference', ring=3, project='other-project')
    task = store.remember('alpha release checklist task', ring=2, project='gui-chat')
    deep = store.remember('alpha release raw history archive', ring=6, project='gui-chat')
    shared = store.remember('alpha release durable procedure', ring=4, project='global')
    global_episode = store.remember('alpha release unrelated episode', ring=3, project='global')

    base = store.search('alpha release preference', project='gui-chat',
                        candidates=[other['id']], strict_scope=True)
    ids = {row['id'] for row in base}
    assert own['id'] in ids and shared['id'] in ids
    assert other['id'] not in ids and task['id'] not in ids
    assert deep['id'] not in ids and global_episode['id'] not in ids

    task_ids = {row['id'] for row in store.search(
        'alpha release checklist task', project='gui-chat', strict_scope=True)}
    assert task['id'] in task_ids
    deep_ids = {row['id'] for row in store.search(
        'alpha release raw history archive', project='gui-chat', strict_scope=True)}
    assert deep['id'] in deep_ids


def test_task_dependencies_execution_and_resumption(tmp_path):
    s=PrismStore(tmp_path/'task.sqlite')
    past=(datetime.now(timezone.utc)-timedelta(hours=1)).isoformat()
    a=s.create_task('Review patch','memory review',past,kind='memory_review')
    b=s.create_task('Try the patch','do not run until review',past,kind='agent_prompt',dependencies=[a['id']])
    assert [x['id'] for x in s.due_tasks()]==[a['id']]
    with pytest.raises(ValueError,match='dependencies'):s.start_task(b['id'],explicit_approval=True)
    blocked=s.start_task(a['id'])
    assert blocked['status']=='awaiting_approval'
    running=s.start_task(a['id'],explicit_approval=True)
    assert running['status']=='running'
    s.finish_task(a['id'],'approved')
    assert [x['id'] for x in s.due_tasks()]==[b['id']]
    s.start_task(b['id'],explicit_approval=True)
    assert s.restore_interrupted()==1
    assert s.get_task(b['id'])['status']=='awaiting_approval'
    assert s.get_task(b['id'])['approved']==0
    s.cancel_task(b['id'])


def test_encrypted_prism_and_weaviate_index_rekey(tmp_path,monkeypatch):
    import agent_vault
    monkeypatch.setattr(agent_vault,'ROOT',tmp_path/'.humoid')
    old,new=os.urandom(32),os.urandom(32)
    vault=AgentVault(old,agent_vault.ROOT)
    vault.mount()
    PrismStore(vault.prism_path).remember('Secret GUI task / passwordless sign-in')
    vault.close()
    assert b'Secret GUI task' not in (vault.root/'agent_prism.sqlite3.aes').read_bytes()
    index=tmp_path/'fake-index'
    (index/'data').mkdir(parents=True)
    (index/'data'/'binary.db').write_bytes(b'vector-related private information')
    payload=_pack(index)
    (vault.root/ARCHIVE_NAME).write_bytes(_seal(payload,_key(old)))
    assert b'vector-related' not in (vault.root/ARCHIVE_NAME).read_bytes()
    rekey_agent_vault(old,new)
    with pytest.raises(Exception): _open((vault.root/ARCHIVE_NAME).read_bytes(),_key(old))
    unpack=tmp_path/'restored-index'
    _unpack(_open((vault.root/ARCHIVE_NAME).read_bytes(),_key(new)),unpack)
    assert (unpack/'data'/'binary.db').read_bytes()==b'vector-related private information'
    reopened=AgentVault(new,agent_vault.ROOT)
    reopened.mount()
    assert PrismStore(reopened.prism_path).search('passwordless')[0]['text'].startswith('Secret GUI')
    reopened.close()


def test_archive_rejects_traversal_and_tampering(tmp_path):
    raw=io.BytesIO()
    with tarfile.open(fileobj=raw,mode='w:gz') as f:
        data=b'malicious'
        info=tarfile.TarInfo('../outside')
        info.size=len(data)
        f.addfile(info,io.BytesIO(data))
    with pytest.raises(ValueError,match='Invalid embedded'):
        _unpack(ARCHIVE_MAGIC+raw.getvalue(),tmp_path/'target')
    assert not (tmp_path/'outside').exists()
    master=os.urandom(32)
    enc=_seal(ARCHIVE_MAGIC+raw.getvalue(),_key(master))
    with pytest.raises(Exception):_open(enc[:-1]+bytes([enc[-1]^1]),_key(master))


def test_embedded_never_starts_on_disk_staging(tmp_path):
    vault=AgentVault(os.urandom(32),tmp_path/'.humoid')
    vault.mount()
    vault.ram_backed=False
    with pytest.raises(RuntimeError,match='tmpfs'):
        EmbeddedWeaviateVault(vault).start()
    vault.close()


def test_holoring_capsules_preserve_sources_and_notice_revisions(tmp_path):
    s=PrismStore(tmp_path/'ring.sqlite3')
    a=s.remember('The config.py change fixed the invalid token bug',project='newgui')
    b=s.remember('The config.py tests now pass after rotation',project='newgui')
    cap=s.distill_capsule('config.py',project='newgui')
    assert cap['ring']==5 and cap['source_id'].startswith('capsule:')
    assert a['id'] in cap['text'] and b['id'] in cap['text']
    assert len(cap['sources'])==2
    assert not any(source['stale'] for source in cap['sources'])
    assert s.distill_capsule('config.py',project='newgui')['id']==cap['id']
    s.revise(a['id'],text='The bug has returned')
    assert any(source['stale'] for source in s.capsule_sources(cap['id']))


def test_mirror_real_sync_protocol_with_fake_weaviate(tmp_path,monkeypatch):
    from prism_weaviate import WeaviateMirror
    from types import SimpleNamespace
    store=PrismStore(tmp_path/'database.sqlite3')
    row=store.remember('Use memory rings in example.py',project='example')
    records={}

    class Data:
        def exists(self,uid):return str(uid) in records
        def insert(self,uuid,properties,vector):records[uuid]=(properties,vector)
        def replace(self,uuid,properties,vector):records[uuid]=(properties,vector)
        def delete_by_id(self,uid):records.pop(uid,None)

    class Collection:
        data=Data()
        def query_near_vector(self,near_vector,limit):
            return SimpleNamespace(objects=[SimpleNamespace(uuid=uid,properties=v[0])
                                            for uid,v in list(records.items())[:limit]])
        query=SimpleNamespace()

    collection=Collection()
    collection.query.near_vector=collection.query_near_vector
    mirror=WeaviateMirror(store,client=SimpleNamespace(is_ready=lambda:True))
    monkeypatch.setattr(mirror,'_collection',lambda client:collection)
    assert mirror.synchronize()['synced']==1
    assert row['id'] in mirror.candidates('rings example.py')
    store.revise(row['id'],text='Use semantic rings in example.py')
    assert row['id'] not in mirror.candidates('rings example.py')
    assert mirror.synchronize()['synced']>=1
    assert row['id'] in mirror.candidates('rings example.py')
    store.forget(row['id'])
    assert mirror.candidates('rings example.py')==[]
    assert mirror.synchronize()['synced']>=1
    assert row['id'] not in records


def test_embedded_weaviate_runtime_snapshot_with_mock_binary(tmp_path,monkeypatch):
    """Mocks native process, exercises actual encrypted embedded lifecycle."""
    import types
    from agent_vault import AgentVault
    k=os.urandom(32)
    v=AgentVault(k,tmp_path/'.humoid')
    v.mount()
    if not v.ram_backed:
        pytest.skip('No RAM-backed storage available for embedded Weaviate')
    class Options:
        def __init__(self,**kwargs):self.__dict__.update(kwargs)
    class Client:
        def __init__(self,embedded_options,auth_client_secret):
            self.path=Path(embedded_options.persistence_data_path)
            self.auth=auth_client_secret
        def connect(self):
            assert self.auth
            (self.path/'private-vectors.db').write_bytes(b'test confidential vectors')
        def is_ready(self):return True
        def close(self):pass
    weaviate=types.ModuleType('weaviate')
    weaviate.WeaviateClient=Client
    embedded=types.ModuleType('weaviate.embedded')
    embedded.EmbeddedOptions=Options
    classes=types.ModuleType('weaviate.classes')
    init=types.ModuleType('weaviate.classes.init')
    init.Auth=SimpleAuth=type('Auth',(),{'api_key':staticmethod(lambda key:key)})
    monkeypatch.setitem(sys.modules,'weaviate',weaviate)
    monkeypatch.setitem(sys.modules,'weaviate.embedded',embedded)
    monkeypatch.setitem(sys.modules,'weaviate.classes',classes)
    monkeypatch.setitem(sys.modules,'weaviate.classes.init',init)
    instance=EmbeddedWeaviateVault(v)
    instance.start()
    instance.seal()
    assert (v.root/ARCHIVE_NAME).is_file()
    assert b'test confidential vectors' not in (v.root/ARCHIVE_NAME).read_bytes()
    instance2=EmbeddedWeaviateVault(v)
    instance2.start()
    assert (instance2.root/'private-vectors.db').read_bytes()==b'test confidential vectors'
    instance2.seal()
    v.close()
