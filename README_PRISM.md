# PRISM RINGS 2.0 — hybrid encrypted memory for Humoid Fusion

This release adds a runnable **SQLite + embedded Weaviate** memory architecture to the fused application. It extends rather than replaces the original TUI's Rhomgaic / Chromatic code memory.

## Start

```bash
./install-fusion.sh
./run-fusion.sh
```

The installer pins `weaviate-client==4.23.1`; pytest is not a runtime dependency. **Embedded Weaviate itself downloads a platform-specific binary** when first launched (requires internet). The server defaults to `1.40.0`. Older encrypted settings are upgraded at runtime and cannot launch a pre-1.38 server; exact supported stable releases remain configurable. This avoids the path-traversal and authorization vulnerabilities affecting the archive's former 1.33.0 pin. The index is optional when a binary or platform is unsupported: the GUI continues using SQLite with an explanatory status.

### Storage architecture

- `.humoid/agent_memory.sqlite3.aes`: the encrypted original agent memory store.
- `.humoid/agent_rhomgaic.sqlite3.aes`: the encrypted original Rhomgaic store.
- `.humoid/agent_prism.sqlite3.aes`: the **authoritative** encrypted PRISM memory, outbox, provenance and task/schedule database.
- `.humoid/agent_weaviate.embedded.aes`: the **embedded Weaviate index**, archived and authenticated with **AES-256-GCM**, using a purpose-separated key derived via HKDF-SHA256 from the **same unlocked GUI master key**.
- The encrypted settings file stores `prism_embedded_enabled=true` and `prism_ctx_enabled=true` by default, with individual GUI switches in **Agent Studio → Advanced Settings**.

The application stages all three SQLite files in a private RAM-backed `/dev/shm` directory when available, or in restricted temporary storage on systems without it (this pre-existing SQLite behavior remains). **Embedded Weaviate is strictly disabled when a RAM-backed staging directory is unavailable**. Its *live* index is under the active private `/dev/shm/humoid-agent-*/weaviate-data` directory; once the embedded process is shut down, its directory is tarred, compressed and **encrypted to persistent storage**. Weaviate receives only locally generated vectors, opaque UUIDs and revision numbers; raw texts/summaries/project names remain in encrypted SQLite.

**Important security boundary:** Although the Weaviate archive is encrypted on disk while the GUI is locked, the running service and in-memory vectors are not themselves encrypted, and the RAM area may still be affected by swap, hibernation, debugger access, core dumps or host compromise. Weaviate binds to loopback and requests a randomly generated ephemeral API key. The standalone Docker Compose example (`docker-compose-prism.yml`) is NOT encrypted by the GUI and is not the default; do not mistake an ordinary Weaviate data directory for the encrypted embedded archive. Windows/macOS without validated secure volatile staging fall back to SQLite until an equivalent safe backend is supplied. This is not claimed to be hardware-backed, OS-level full-disk encryption, or suitable for protecting against an attacker controlling an unlocked host.

Password rotation includes the Weaviate encrypted archive and validates its authentication tag *before* publishing any changed agent vault files. An erroneous passphrase/key, archive corruption, startup failure or binary incompatibility prevents Weaviate from starting and is visibly reported; SQLite remains available. Never manually delete a corrupted archive without taking a backup.

### Index synchronization, resilience and privacy

1. SQLite is the sole authority for source text, verification state, ring assignments, tasks and tombstones.
2. Each PRISM memory edit and index operation produces an outbox entry in the **same SQLite transaction**.
3. On startup all memories are reconciled with their current revisions in case the last encrypted Weaviate snapshot was older than the SQLite snapshot.
4. Sync operates on the local authenticated embedded Weaviate index; index ACKs are persisted to SQLite's encrypted snapshot. Vector results are checked against SQLite revisions and deletion flags before they can be used.
5. If Weaviate is unavailable, SQLite lexical/spectral retrieval continues to work. The index can be reconstructed from SQLite source records.
6. PRISM rings indexed in **local** Weaviate default to indexing consent on new ring memories; the UI does **not** automatically send these vectors to remote hosts. The original TUI's separately configured remote Weaviate backend is **not** enabled for vault-backed GUI memory.

The currently bundled fallback vectorizer is **deterministic hashed lexical n-grams and spectral channels**, not a neural embedding model. Thus the code offers hybrid *vector search* with typed retrieval but does **not** claim the conceptual hyperdimensional equations have all been realized as a trained embedding model.

### Memory features

- **Seven rings:** Reflex, Working, Task, Episodic, Procedural, Semantic, Deep Vault.
- **Ten spectral domains:** code, writing, tasks, computer-use, research, procedural, conversation, creative, system, and causal.
- **SpectraForge:** reproducible channel scoring, softmax weights, blended ring colors and a normalized 256-dimensional local lexical/spectral vector.
- **HoloRing capsules:** extractive evidence summaries that preserve explicit `source UUID@revision` and detect when a cited source has changed. These are labeled as extractive and unverified; no invented LLM summaries.
- **Ring Observatory:** write memories, search with colored channels, inspect the encrypted outbox/index state, produce CTX packets, distill evidence, delete by UUID, and import original agent memories idempotently by source ID.
- **Linked recall:** the coding agent retrieves PRISM context when enabled; legacy encrypted Gemma Chat also includes bounded PRISM SQLite ring evidence in its memory context.
- **Memory gravity wells:** automatic chat recall is locked to the active project. Only
  global R4 procedural and R5 semantic knowledge may cross that boundary. R2 task
  memory requires task-oriented intent, R0 requires explicit immediate-context intent,
  and R6 Deep Vault requires an explicit archive/full-recall request. Vector hits can
  improve ranking but can never bypass these project or ring gates.
- **Single recall authority:** after a vault has PRISM memory, chat and task execution
  do not append a second broad SQLite/Rhomgaic recall packet. This prevents older
  retrieval paths from leaking an unrelated task back into an otherwise scoped prompt.
- **Agent memory bridge:** up to 500 original agent memories are reconciled on GUI startup; use **Import original agent memories** in the Ring Observatory to review up to 1,200 additional entries. Newly added GUI agent memories are also mirrored to PRISM.

### ChronoOrbit scheduling

- Each task includes ID, title, instructions, project, UTC-normalized due time, kind, priority, optional dependencies, recurrence (`none`, `daily`, `weekly`, `monthly`, `weekdays`, or `interval:MINUTES`), max run count, status and audit events.
- Dependencies must be complete before a task becomes runnable. Effective priority balances configured priority, deadline pressure, time waiting and downstream unblock value.
- **User approval required** for agent prompts, even if recurrence is set. Executions are marked `running` and sealed before side effects; ambiguous interrupted executions become `awaiting_approval`, never automatically replayed.
- A GUI poll checks due tasks roughly every 60 seconds while unlocked, displaying reminders/status but **never autonomously running agent work**. Tasks only run when the user clicks **Approve and run**. No guarantee of OS-wide persistent background alerts or execution when the GUI is closed.
- Failed runs are recorded. Recurring successful runs advance to the next due time; huge missed-time backlogs are skipped.

### Verification

```bash
PYTHONPATH=src:. python -m pytest -q fusion_tests agent_tests/test_memory.py agent_tests/test_chromatic_v6.py
```

The new tests cover spectral encoding, idempotent bridging, source-linked summaries and staleness, SQLite outbox behavior, task dependencies/approvals/recovery, authenticated encryption and tampering, archive traversal defense, key rotation, and safe fallback if `/dev/shm` is unavailable. An actual embedded binary live-inference and GUI desktop smoke test require running the application on a supported host; those were not possible in the build environment.
