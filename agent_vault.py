"""Encrypted GUI <-> agent storage bridge.

Persistent agent settings, sessions and both memory databases share the GUI's
unwrapped AES-256 master key. Active databases are mounted in RAM-backed private
storage where supported, because the original TUI engines need SQLite paths.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
import threading
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

ROOT = Path('.humoid')
SECRET_FILES = ('agent_settings.json.aes', 'agent_sessions.json.aes', 'agent_invented_tools.json.aes', 'agent_memory.sqlite3.aes', 'agent_rhomgaic.sqlite3.aes', 'agent_prism.sqlite3.aes')
MAGIC = b'HAG1'
MAX_VAULT_BYTES = 256 * 1024 * 1024


def _seal(raw: bytes, key: bytes) -> bytes:
    if len(key) != 32:
        raise ValueError('Expected GUI AES-256 master key')
    nonce = os.urandom(12)
    return MAGIC + nonce + AESGCM(key).encrypt(nonce, raw, MAGIC)


def _open(raw: bytes, key: bytes) -> bytes:
    if not raw.startswith(MAGIC) or len(raw) < len(MAGIC) + 12 + 16:
        raise ValueError('Agent vault format incorrect or damaged')
    nonce = raw[len(MAGIC):len(MAGIC)+12]
    return AESGCM(key).decrypt(nonce, raw[len(MAGIC)+12:], MAGIC)


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    fd, temp = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        dirfd = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(dirfd)
        finally: os.close(dirfd)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def rekey_agent_vault(old_key: bytes, new_key: bytes) -> None:
    """Stage every rekeyed blob; only publish after *all* decrypt successfully.

    Called by GUI key rotation before committing new master key. If there's an
    exception, no agent vault objects are modified. The GUI's existing rotation
    uses its own two-phase files; a prior backup remains important.
    """
    staged = {}
    originals = {}
    for name in SECRET_FILES:
        path = ROOT / name
        if path.exists():
            original = path.read_bytes()
            originals[path] = original
            staged[path] = _seal(_open(original, old_key), new_key)
    # Embedded Weaviate uses a domain-separated key (HKDF from GUI key).
    # Stage this archive alongside the other snapshots for password rotation.
    from prism_embedded import ARCHIVE_NAME, _key as _weaviate_key
    index_path = ROOT / ARCHIVE_NAME
    if index_path.exists():
        original = index_path.read_bytes()
        originals[index_path] = original
        staged[index_path] = _seal(_open(original, _weaviate_key(old_key)), _weaviate_key(new_key))
    written = []
    try:
        for path, raw in staged.items():
            _atomic_write(path, raw)
            written.append(path)
    except BaseException:
        # Prevent an ordinary I/O error from leaving a mixed-key agent vault.
        # A power failure during multi-file rotation still requires a backup.
        for path in reversed(written):
            _atomic_write(path, originals[path])
        raise


class AgentVault:
    def __init__(self, key: bytes, root: Path = ROOT):
        self.key = key
        self.root = Path(root)
        self.root.mkdir(exist_ok=True, parents=True)
        os.chmod(self.root, 0o700)
        self.guard = threading.RLock()
        self.active_dir: Path | None = None
        self.memory_path: Path | None = None
        self.rhomgaic_path: Path | None = None
        self.prism_path: Path | None = None
        self.ram_backed = False
        self._lock_fd: int | None = None

    def read_json(self, filename: str, default: Any) -> Any:
        with self.guard:
            target = self.root / filename
            if not target.exists(): return default
            return json.loads(_open(target.read_bytes(), self.key).decode('utf-8'))

    def write_json(self, filename: str, data: Any) -> None:
        with self.guard:
            raw = json.dumps(data, ensure_ascii=False, default=str).encode('utf-8')
            _atomic_write(self.root / filename, _seal(raw, self.key))

    def _acquire_process_lock(self) -> None:
        """Exclusive writer lock prevents concurrent instances losing edits."""
        path = self.root / 'agent.lock'
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            if os.name == 'nt':
                import msvcrt
                os.lseek(fd, 0, 0)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as exc:
            os.close(fd)
            raise RuntimeError('Agent memory is already open in another app instance') from exc
        self._lock_fd = fd

    def _release_process_lock(self) -> None:
        if self._lock_fd is not None:
            fd = self._lock_fd
            self._lock_fd = None
            try:
                if os.name == 'nt':
                    import msvcrt
                    os.lseek(fd, 0, 0)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally: os.close(fd)

    def mount(self, *, read_only: bool = False) -> tuple[Path, Path]:
        """Open both SQLite files in a 0700 staging directory.

        No TUI backend sees the encrypted on-disk file directly. The persisted
        copy is always AES-GCM; during use a database lives under /dev/shm on
        Linux or a restricted temp directory on other systems.
        """
        with self.guard:
            if self.active_dir:
                return self.memory_path, self.rhomgaic_path
            if not read_only:
                self._acquire_process_lock()
            root = Path('/dev/shm')
            self.ram_backed = root.is_dir() and os.access(root, os.W_OK)
            tmpbase = root if self.ram_backed else Path(tempfile.gettempdir())
            self.active_dir = Path(tempfile.mkdtemp(prefix='humoid-agent-', dir=tmpbase))
            os.chmod(self.active_dir, 0o700)
            self.memory_path = self.active_dir / 'memory.sqlite3'
            self.rhomgaic_path = self.active_dir / 'rhomgaic.sqlite3'
            self.prism_path = self.active_dir / 'prism.sqlite3'
            try:
                for target, name in (
                    (self.memory_path, 'agent_memory.sqlite3.aes'),
                    (self.rhomgaic_path, 'agent_rhomgaic.sqlite3.aes'),
                    (self.prism_path, 'agent_prism.sqlite3.aes'),
                ):
                    file = self.root / name
                    if file.exists():
                        if file.stat().st_size > MAX_VAULT_BYTES:
                            raise ValueError(f'{name} exceeds safe vault size')
                        target.write_bytes(_open(file.read_bytes(), self.key))
                        os.chmod(target, 0o600)
                return self.memory_path, self.rhomgaic_path
            except BaseException:
                self._discard()
                raise

    def persist(self) -> None:
        """SQLite-consistent snapshot to AES-GCM, one DB at a time."""
        with self.guard:
            if self.active_dir is None: return
            for path, name in ((self.memory_path, 'agent_memory.sqlite3.aes'),
                               (self.rhomgaic_path, 'agent_rhomgaic.sqlite3.aes'),
                               (self.prism_path, 'agent_prism.sqlite3.aes')):
                if path is None or not path.is_file(): continue
                snapshot = self.active_dir / (path.name + '.snapshot')
                try:
                    with sqlite3.connect(path, timeout=15) as live, sqlite3.connect(snapshot) as output:
                        live.backup(output)
                    if snapshot.stat().st_size > MAX_VAULT_BYTES:
                        raise ValueError(f'{name} exceeds 256 MiB vault size')
                    _atomic_write(self.root / name, _seal(snapshot.read_bytes(), self.key))
                finally:
                    snapshot.unlink(missing_ok=True)

    def _discard(self) -> None:
        if self.active_dir:
            shutil.rmtree(self.active_dir, ignore_errors=True)
        self.active_dir = self.memory_path = self.rhomgaic_path = self.prism_path = None
        self._release_process_lock()

    def close(self) -> None:
        with self.guard:
            # Never destroy the last plaintext working copy if encryption or
            # fsync failed. Keep the lock/staging data so sealing can be retried.
            self.persist()
            self._discard()

    def abort(self) -> None:
        """Discard decrypted staging after a vault error; do not publish."""
        with self.guard:
            self._discard()


def advanced_context_for_gui(key: bytes, query: str, max_chars: int = 2300) -> str:
    """Return one scoped memory surface for the local LiteRT GUI chat.

    Called from the GUI's model worker, not the Tk thread. Temporary SQLite
    snapshots remain isolated and never expose the user's stored master key.
    """
    import asyncio
    if not (ROOT / 'agent_memory.sqlite3.aes').exists():
        return ''
    vault = AgentVault(key)
    try:
        memory_db, rhomgaic_db = vault.mount(read_only=True)
        # Once PRISM exists it is the sole recall authority. Falling through
        # to legacy broad recall here would bypass project and ring isolation.
        if vault.prism_path and vault.prism_path.exists():
            try:
                from prism_core import PrismStore
                return PrismStore(vault.prism_path).summary_packet(
                    query, budget=max_chars, project='gui-chat',
                    strict_scope=True)[:max_chars]
            except (OSError, sqlite3.DatabaseError, ValueError):
                return ''

        # Migration fallback for vaults that have not created PRISM yet.
        from humoid_tui.config import Settings
        from humoid_tui.memory import MemoryRouter

        # Honor the actual encrypted TUI CTX/retrieval preferences in legacy
        # GUI chat. Read from the same key; never read the TUI's plaintext env.
        saved = vault.read_json('agent_settings.json.aes', {})
        configured = {
            field: value for field, value in saved.items()
            if field in Settings.model_fields
            and field not in {'humoid_memory_db', 'humoid_rhomgaic_db', 'humoid_memory_backend'}
        }
        configured.update(
            humoid_memory_backend='sqlite',
            humoid_memory_db=memory_db,
            humoid_rhomgaic_db=rhomgaic_db,
            # This detached reader is deliberately ephemeral: no cache writes
            # to an old snapshot should be persisted over the live agent's DB.
            humoid_rag_context_cache=False,
            humoid_rhomgaic_enabled=bool(saved.get('humoid_rhomgaic_enabled', True)) and rhomgaic_db.exists(),
        )
        depth = str(saved.get('gui_memory_depth', 'reflex'))
        if depth not in {'reflex', 'working', 'subsurface', 'auto'}:
            depth = 'reflex'

        async def recall():
            settings = Settings(_env_file=None, **configured)
            router = MemoryRouter(settings)
            await router.initialize()
            try:
                return await router.context_packet(
                    query, limit=min(8, settings.humoid_memory_context_limit),
                    depth=depth, max_chars=max_chars
                )
            finally:
                await router.close()
        return asyncio.run(recall())[:max_chars]
    finally:
        # Read-only recall. Never persist this reader's detached copy; its
        # Rhomgaic access counters are owned by the live agent instance.
        vault.abort()
