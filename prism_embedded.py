"""Vault-sealed embedded Weaviate index for PRISM.

A Weaviate *process* needs an unpacked datastore. Never launch against a disk-
backed directory: use a private tmpfs workspace while unlocked, and snapshot
an authenticated AES-256-GCM archive only after the process fully stops.
The SQLite source of truth can reconstruct this derived index at any time.

Warning: RAM/swap/hibernation/core dumps and a compromised host are outside
this vault's threat model. Embedded localhost ports also require API-key auth.
"""
from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from agent_vault import _atomic_write, _open, _seal, AgentVault
from humoid_tui.config import DEFAULT_WEAVIATE_VERSION, secure_weaviate_version

ARCHIVE_NAME = 'agent_weaviate.embedded.aes'
ARCHIVE_MAGIC = b'PRISM-WEAVIATE-ARCHIVE-1\n'
MAX_ARCHIVE = 256 * 1024 * 1024
MAX_ITEM = 128 * 1024 * 1024
def _key(master: bytes) -> bytes:
    """Purpose-separated key, retaining GUI passphrase/key-rotation support."""
    if len(master) != 32:
        raise ValueError('AES-256 GUI key required')
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                info=b'humoid-prism-embedded-weaviate-v1').derive(master)


def _pack(root: Path) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as tar:
        for file in sorted(root.rglob('*')):
            if file.is_symlink():
                raise ValueError('Refusing symlink in embedded index')
            if file.is_dir():
                continue
            if not file.is_file() or file.stat().st_size > MAX_ITEM:
                raise ValueError('Unsafe embedded index entry')
            relative = file.relative_to(root).as_posix()
            if len(relative) > 500:
                raise ValueError('Embedded index path too long')
            tar.add(file, arcname=relative, recursive=False)
            if stream.tell() > MAX_ARCHIVE:
                raise ValueError('Embedded archive too large')
    payload = stream.getvalue()
    if len(payload) > MAX_ARCHIVE:
        raise ValueError('Embedded archive too large')
    return ARCHIVE_MAGIC + payload


def _unpack(raw: bytes, root: Path) -> None:
    if not raw.startswith(ARCHIVE_MAGIC):
        raise ValueError('Invalid embedded vault archive version')
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    total = 0
    with tarfile.open(fileobj=io.BytesIO(raw[len(ARCHIVE_MAGIC):]), mode='r:gz') as archive:
        for entry in archive:
            name = PurePosixPath(entry.name)
            if (name.is_absolute() or not name.parts or
                    any(s in {'.', '..'} for s in name.parts) or
                    not entry.isfile() or entry.size > MAX_ITEM):
                raise ValueError('Invalid embedded vault archive entry')
            total += entry.size
            if total > MAX_ARCHIVE:
                raise ValueError('Embedded vault extraction limit exceeded')
            dest = root.joinpath(*name.parts)
            dest.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with archive.extractfile(entry) as src, dest.open('xb') as dst:
                while True:
                    data = src.read(1024 * 1024)
                    if not data:
                        break
                    dst.write(data)
            dest.chmod(0o600)


class EmbeddedWeaviateVault:
    def __init__(self, vault: AgentVault, *, version: str = DEFAULT_WEAVIATE_VERSION,
                 port: int = 18079, grpc_port: int = 15060):
        self.vault = vault
        self.version, self.version_was_upgraded = secure_weaviate_version(version)
        self.port = port
        self.grpc_port = grpc_port
        self.root: Path | None = None
        self.client: Any = None

    def start(self) -> Any:
        """Require tmpfs, authenticated loopback Weaviate, encrypted restore."""
        if self.client is not None:
            return self.client
        if not self.vault.active_dir or not self.vault.ram_backed:
            raise RuntimeError('Encrypted embedded Weaviate needs Linux /dev/shm tmpfs; SQLite remains available')
        if not 1 <= self.port <= 65535 or not 1 <= self.grpc_port <= 65535:
            raise ValueError('Invalid embedded Weaviate ports')
        import weaviate
        from weaviate.embedded import EmbeddedOptions
        from weaviate.classes.init import Auth
        root = self.vault.active_dir / 'weaviate-data'
        if not root.exists():
            archive = self.vault.root / ARCHIVE_NAME
            if archive.exists():
                if archive.stat().st_size > MAX_ARCHIVE:
                    raise ValueError('Embedded Weaviate archive too large')
                _unpack(_open(archive.read_bytes(), _key(self.vault.key)), root)
            else:
                root.mkdir(mode=0o700)
        # Key is ephemeral and never written to disk; server uses this same
        # key for loopback REST and gRPC endpoints while GUI vault is open.
        api_key = os.urandom(32).hex()
        options = EmbeddedOptions(
            version=self.version,
            hostname='127.0.0.1', port=self.port, grpc_port=self.grpc_port,
            persistence_data_path=str(root),
            additional_env_vars={
                'LOG_LEVEL': 'error', 'DISABLE_TELEMETRY': 'true',
                'AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED': 'false',
                'AUTHENTICATION_APIKEY_ENABLED': 'true',
                'AUTHENTICATION_APIKEY_ALLOWED_KEYS': api_key,
                'AUTHENTICATION_APIKEY_USERS': 'prism',
                'AUTHORIZATION_ENABLE_RBAC': 'true',
                'AUTHORIZATION_RBAC_ROOT_USERS': 'prism',
                'REPLICA_MOVEMENT_ENABLED': 'false',
                'ENABLE_MODULES': '',
            },
        )
        client = weaviate.WeaviateClient(
            embedded_options=options, auth_client_secret=Auth.api_key(api_key))
        try:
            client.connect()
            if not client.is_ready():
                raise ConnectionError('Embedded Weaviate did not become ready')
        except Exception:
            client.close()
            raise
        self.client, self.root = client, root
        return client

    def seal(self) -> None:
        """Close embedded server BEFORE snapshot. Keep RAM copy on failures."""
        if self.client is not None:
            self.client.close()  # Flush Weaviate's datastore; no live tar copies.
            self.client = None
        if self.root is None:
            return
        raw = _pack(self.root)
        _atomic_write(self.vault.root / ARCHIVE_NAME, _seal(raw, _key(self.vault.key)))
        self.root = None  # AgentVault close removes the unlocked tmpfs workspace.

    def rekey_archive(self, new_master: bytes) -> None:
        """Not used by GUI rotation; GUI rotates via rekey_agent_vault."""
        if self.client is not None:
            raise RuntimeError('Close Weaviate before rotating the key')
        path = self.vault.root / ARCHIVE_NAME
        if path.exists():
            _atomic_write(path, _seal(_open(path.read_bytes(), _key(self.vault.key)), _key(new_master)))
