"""Weaviate v4 index; SQLite remains authoritative. Embedded index is vault-sealed."""
from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from prism_core import PrismStore, vectorize


@dataclass
class WeaviateConfig:
    enabled: bool = False
    http_host: str = '127.0.0.1'
    http_port: int = 8080
    grpc_host: str = '127.0.0.1'
    grpc_port: int = 50051
    secure: bool = False
    api_key: str = ''

    def validate(self):
        # Public hosts must use TLS and authentication. On local machines,
        # require an API key too; avoids accidentally connecting to a wide-open
        # anonymous Weaviate endpoint. Server itself must enforce credentials.
        if self.enabled and not self.api_key.strip():
            raise ValueError('Weaviate indexing requires a configured API key')
        if self.enabled and (self.http_host not in ('localhost','127.0.0.1','::1')
                             or self.grpc_host not in ('localhost','127.0.0.1','::1')) and not self.secure:
            raise ValueError('Remote Weaviate requires TLS')
        if not all(1 <= p <= 65535 for p in (self.http_port,self.grpc_port)):
            raise ValueError('Invalid Weaviate ports')


class WeaviateMirror:
    """One-shot connections to avoid leaving live gRPC clients on GUI threads."""
    def __init__(self, store: PrismStore, config: WeaviateConfig | None = None, *, client=None):
        self.store=store
        self.client=client
        self.config=config or WeaviateConfig(enabled=client is not None)

    @contextmanager
    def connection(self):
        if self.client is not None:
            yield self.client  # Owned by EmbeddedWeaviateVault: do not close here.
        else:
            with self._connect() as connected:
                yield connected

    def _connect(self):
        if self.client is None: self.config.validate()
        if self.client is None and not self.config.enabled:
            raise RuntimeError('Weaviate index disabled. Enable it explicitly in the GUI.')
        try:
            import weaviate
            from weaviate.classes.init import Auth, AdditionalConfig, Timeout
        except ImportError as exc:
            raise RuntimeError('Install optional dependency: pip install weaviate-client>=4.16,<5') from exc
        return weaviate.connect_to_custom(
            http_host=self.config.http_host, http_port=self.config.http_port,
            http_secure=self.config.secure,
            grpc_host=self.config.grpc_host, grpc_port=self.config.grpc_port,
            grpc_secure=self.config.secure,
            auth_credentials=Auth.api_key(self.config.api_key),
            additional_config=AdditionalConfig(timeout=Timeout(init=5,query=10,insert=15)),
        )

    def _collection(self,client):
        from weaviate.classes.config import Configure, Property, DataType
        name=self.store.collection
        if not client.collections.exists(name):
            client.collections.create(
                name=name,
                vector_config=Configure.Vectors.self_provided(),
                properties=[
                    Property(name='revision',data_type=DataType.INT),
                ],
            )
        return client.collections.use(name)

    def status(self)->dict:
        if self.client is None: self.config.validate()
        if self.client is None and not self.config.enabled:
            return {'enabled':False,'message':'Disabled; no vectors transmitted',
                    'pending':len(self.store.pending_index(limit=100000))}
        with self.connection() as client:
            if not client.is_ready():raise ConnectionError('Weaviate not ready')
            return {'enabled':True,'ready':True,'collection':self.store.collection,
                    'pending':len(self.store.pending_index(limit=100000))}

    def synchronize(self, *, limit:int=50)->dict:
        if self.client is None: self.config.validate()
        if self.client is None and not self.config.enabled:
            return {'enabled':False,'synced':0,'pending':len(self.store.pending_index(100000))}
        applied=0
        with self.connection() as client:
            collection=self._collection(client)
            for event in self.store.pending_index(limit):
                item=self.store.get(event['memory_id'])
                # Older outbox entries are superseded by the latest SQLite
                # revision; skip them rather than re-publishing stale data.
                if item and int(item['revision'])!=int(event['revision']):
                    self.store.index_ack(event['id'])
                    continue
                try:
                    should_delete = (event['operation']=='delete' or item is None
                                     or item['deleted'] or not item['consent_index'])
                    if should_delete:
                        if collection.data.exists(event['memory_id']):
                            collection.data.delete_by_id(event['memory_id'])
                    else:
                        # No plaintext, project, channel names, summaries or
                        # even hex color enter the index. Embeddings leak
                        # statistical information and must be treated as sensitive.
                        vector=vectorize(item['text'],json.loads(item['spectrum']))
                        props={'revision':int(item['revision'])}
                        if collection.data.exists(item['id']):
                            collection.data.replace(uuid=item['id'], properties=props, vector=vector)
                        else:
                            collection.data.insert(uuid=item['id'], properties=props, vector=vector)
                    self.store.index_ack(event['id'])
                    applied+=1
                except Exception as exc:
                    self.store.index_error(event['id'],str(exc))
                    raise
        return {'enabled':True,'synced':applied,'pending':len(self.store.pending_index(100000))}

    def candidates(self,query:str, limit:int=24)->list[str]:
        if self.client is None: self.config.validate()
        if self.client is None and not self.config.enabled:return []
        with self.connection() as client:
            collection=self._collection(client)
            response=collection.query.near_vector(near_vector=vectorize(query),limit=min(100,max(1,limit)))
            ids=[]
            for obj in response.objects:
                memory_id=str(obj.uuid)
                item=self.store.get(memory_id)
                # Never trust a vector result if the authoritative SQLite row
                # is tombstoned, was revoked or has since been revised.
                if (item and not item['deleted'] and item['consent_index']
                    and int(item['revision'])==int(obj.properties.get('revision',-1))):
                    ids.append(memory_id)
            return ids
