# Humoid Fusion — encrypted Gemma GUI + full coding agent

This project combines the two uploaded source repositories:

* **Humoid Gemma GUI:** the existing CustomTkinter desktop app, encrypted local LiteRT-LM model, AES-GCM history vault, Dashboard/QID, Chat, Road Scanner, local model management, history browser, dynamic encouragement RAG, and continuation mode.
* **Humoid Agent TUI:** its **actual** `humoid_tui` agent execution engine, provider adapters, native Gemma 4 tool-calling protocol, procedural edit cycles, memory router, Rhomgaic/Chromatic v6 code memory, maintenance tools, AST intelligence, project perspective, action guard, and local GGUF model manager.

The original TUI sources are retained under `src/humoid_tui`, along with its tests, benchmark scripts, and reference docs. The GUI is now the primary interface: open its **Agent Studio** tab.

## Install

Python **3.11–3.12** is recommended for LiteRT and llama.cpp wheel compatibility. Linux desktop with Tk support recommended. Other platforms may require their own LiteRT runtime support.

```bash
unzip humoid-fusion.zip
cd humoid-fusion
./install-fusion.sh
./run-fusion.sh
```

Windows: create a Python virtual environment, then use `python -m pip install -r requirements-fusion.txt` and `run-fusion.bat`. A separate installation of the optional `llama-cpp-python[server]` package may be necessary if you want to serve GGUF models.

Launch the *original* terminal agent if desired with `./run-fusion.sh --tui` or `humoid` after the editable installation. **Note:** the standalone CLI continues using its original TUI configuration and is **not** automatically granted the GUI's encryption key; for a unified encrypted memory surface, use **Agent Studio** inside the GUI.

If package installation fails for your OS, see the two original projects' requirements and model runtime support. The GUI cannot work until CustomTkinter, Pydantic settings, the OpenAI SDK, and the other listed dependencies are installed. Downloading the actual Gemma model requires network access and disk space.

## New Agent Studio UI

The existing Dashboard, Chat, Road Scanner, History, Download Model, Settings and About tabs remain. **Agent Studio** adds seven complete workspaces:

| Section | What you can do |
| --- | --- |
| **Agent Chat** | Use the original multi-round agent, native protocols, file tools, memory-augmented responses, sessions, cancellation, code maintenance and approved edits. |
| **Memory Lab** | Recall stored episodic, semantic and procedural memories; inspect Rhomgaic surface statistics; add, verify, delete or search memories; import existing GUI chat history into the advanced memory index. |
| **CTX Lab** | Query progressive `reflex`, `working`, `subsurface`, or `auto` context packets and see the Rhomgaic recall trace. Set the memory-recall depth used by the original encrypted Gemma Chat. |
| **Tool Studio** | Run any registered TUI tool (selected from the live schema registry), including Project Perspective, Chromatic memory, code intelligence, edit cycle, maintenance plans, playbook recall, file editing, and undo. Arguments are editable JSON. File edits require diff approval in a modal with an editable proposed result. |
| **Local Models** | Detect hardware, browse GGUF files, download/inspect/benchmark/launch/stop, inspect provider health, install/reinstall/uninstall llama.cpp server support. The original LiteRT model download remains in **Download Model**. |
| **Advanced Settings** | Edit **every field** of the TUI `Settings` schema, grouped into provider credentials, agent autonomy, context limits, tools, Gemma 4, memory, Rhomgaic, context cache, Weaviate and memory schema defaults. Validate and save encrypted. |
| **Activity** | View the agent event stream: provider health, reasoning and tool protocol events, progress, memory errors, recalls and tool results. |

### Local Gemma as an agent provider

When the agent is set to the `litert` provider with **Use encrypted desktop Gemma directly** enabled (the default), it runs the **original GUI's encrypted LiteRT model** in an isolated inference process. No OpenAI-compatible HTTP shim is required. To use an external LiteRT HTTP shim instead, disable that option under Local Models and save advanced settings.

The direct LiteRT bridge gathers the model's completed response before handing it to the agent protocol parser; it does not yet stream tokens live. Its native tool-calling success depends on the LiteRT model producing recognized Gemma tool tokens. GGUF via managed llama.cpp may be preferable for some agent tool-call workloads. Other provider choices: DigitalOcean, Meta, OpenAI, llama.cpp, LiteRT.

### CTX and advanced memory

All original Rhomgaic algorithms remain in the TUI package, including Chromatic v6 search, dense/lexical fusion, temporal/importance scoring, graph expansion, context accordion, semantic capsules, code fingerprints and procedure playbooks.

The desktop app now provides **bidirectional sharing of information**:

1. Each completed original GUI Chat response is automatically stored as an **episodic advanced memory** in the shared encrypted agent memory backend.
2. Local GUI Chat tries to recall a bounded advanced memory context packet, in addition to its existing encrypted history retrieval, before passing a prompt to Gemma. This recall uses the CTX/retrieval settings saved in the same encrypted vault and the **CTX Lab → Gemma Chat depth** selector (save settings to apply). The original GUI conversation formatting is preserved; added memory is limited to a small prompt surface to prevent context overload.
3. Agent Chat uses exactly the original `AgentHarness.run()` and Rhomgaic `MemoryRouter` code, backed by the encrypted memory vault.
4. **Memory Lab → Import GUI history** can index prior GUI conversations (up to 5,000 records per import), skipping IDs previously imported by that button.

## Security & storage design

* **Agent settings** (`.humoid/agent_settings.json.aes`) and **GUI/agent session archive** (`.humoid/agent_sessions.json.aes`) use authenticated **AES-256-GCM** with the same master key unlocked by the desktop GUI passphrase.
* Both **agent SQLite memory** (`.humoid/agent_memory.sqlite3.aes`) and **Rhomgaic sidecar** (`.humoid/agent_rhomgaic.sqlite3.aes`) are encrypted using AES-GCM with unique 96-bit nonces. Agent-invented tool definitions are similarly encrypted in `.humoid/agent_invented_tools.json.aes`.
* During execution, SQLite's working copy is decrypted into a private mode-0700 staging directory. On Linux it prefers memory-backed `/dev/shm`; on systems without this it uses a permission-restricted OS temporary directory, **which may touch disk or swap while unlocked**. To enforce never writing plaintext memory to a disk-backed temporary directory, run on a system with available `/dev/shm` or adapt the runtime to an encrypted SQLite implementation such as SQLCipher.
* Memory snapshots are encrypted **after each agent operation**, and again on lock/exit; file changes are governed by the original TUI's write approvals. Only one desktop app instance at a time can mount the encrypted memory databases for writing.
* GUI password rotation also re-encrypts the advanced agent archives (agent memory must first be sealed/closed). As with any multi-file rekey, keep a secure backup before rotating credentials, particularly in case of a crash or power loss during key rotation.
* The **legacy TUI memory backend** stays on encrypted SQLite for the GUI; **PRISM also runs encrypted embedded Weaviate** as a second, derived local index, sealed under the same GUI master key while locked (see README_PRISM.md). Standalone remote Weaviate data is **not** encrypted by this GUI vault.
* When the vault locks, the agent closes and its decrypted memory staging directory is discarded. Native GGUF downloads, local project source files and arbitrary tool-created workspace files are **not** automatically encrypted. The original GUI's separately downloaded LiteRT model remains encrypted in its original model vault.
* Shell commands are **disabled by default** and are not OS-sandboxed. File tools enforce the original workspace path fence; shell commands have the permissions of the local OS user. Only turn on shell execution for trusted repositories in an isolated environment. The GUI prompts before running direct shell tools.
* API keys entered in Advanced Settings are AES-GCM encrypted in the vault rather than copied into `.env` or unencrypted GUI settings. The original TUI `--tui` path retains its independent `.env` behavior.

## Runtime notes / limitations

* Agent settings that affect memory, model profiles, tools and context require **Save encrypted settings & restart agent** for a fresh harness, with encrypted memory retained.
* Changing the password requires finishing outstanding agent operations. The password change rekeys the model/history/agent artifacts and then reopens advanced memory.
* The original GUI continues to use isolated worker processes for LiteRT, while the coding agent runs its async event loop on a separate thread and pushes UI events back onto the Tk thread.
* `Local Models` manages GGUF files separately from the GUI's encrypted `.litertlm` model. The original TUI remains included rather than having its Textual views embedded inside Tk.
* If an encrypted snapshot fails (for example, disk full), the agent keeps the active database and blocks lock/close until it can be sealed successfully rather than silently deleting the last working copy. An abnormal process termination may still leave unsaved in-flight agent memory since the last encrypted snapshot. Linux `/dev/shm` working copies do not survive a reboot; on other systems temp-directory cleanup after a hard crash may require manual inspection.

## Verification

```bash
PYTHONPATH=src:. python -m pytest -q fusion_tests
PYTHONPATH=src:. python -m pytest -q agent_tests/test_memory.py agent_tests/test_chromatic_v6.py
```

The integration suite checks authenticated encryption, random nonces, SQLite snapshot persistence, retry after disk errors, agent vault rekey recovery, actual Rhomgaic memory retrieval, and headless GUI agent-engine initialization/tools/durable memory reuse (using stubbed optional GUI/provider packages when absent). The selected tests passed in the build environment (35 total). **A full desktop smoke test and real LiteRT inference have not been performed here** because the required GUI/native inference packages could not be installed in the sandbox; test on your machine after installation.

Some original TUI tests require optional server dependencies; a pre-existing project-perspective ranking test can also be sensitive to retrieval ordering. The original test sources are provided intact for repeatable checks.

## Attribution

This is a source-level fusion of the **two files supplied by the user**, not a newly written substitute for either underlying agent or GUI. Original source licenses remain included. See `agent_docs/` for Rhomgaic, Chromatic v6, code intelligence and maintenance references.

## PRISM RINGS 2.0 — encrypted SQLite + embedded Weaviate

**New:** Agent Studio now includes **Ring Observatory** and **ChronoOrbit**. PRISM keeps its authoritative memories, color-channel fingerprints, source-linked evidence summaries, sync outbox and scheduled task graph in AES-256-GCM-sealed SQLite. It uses an embedded loopback Weaviate index **by default where dependencies and Linux tmpfs are available**, with the live Weaviate datastore kept in private RAM-backed storage and sealed into a separately AES-GCM-encrypted archive on lock using a GUI-key-derived HKDF subkey. Original TUI Rhomgaic memory continues operating on encrypted SQLite. A remote Docker Weaviate datastore is **not** automatically encrypted by the GUI.

The new index defaults to **SQLite fallback** if the embedded server cannot run. Weaviate requires an additional executable download and could not be integration-tested against a real server here; see [README_PRISM.md](README_PRISM.md) for threat-model details, limitations, controls, and tests. PRISM's built-in vectorizer is hashed lexical/spectral, not a pretrained semantic model. Scheduled agent actions require explicit GUI approval and are not executed while the application is closed.

Both **Agent Chat** and the primary encrypted **Chat** now query the live PRISM/Weaviate index before inference. Primary Chat keeps its encrypted SQLite/Rhomgaic fallback, and every completed primary-Chat exchange is mirrored into PRISM and synchronized to Weaviate for later recall.
