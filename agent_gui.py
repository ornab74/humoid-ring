"""Advanced agent desktop panels. The TUI's real harness runs on one asyncio loop.

Tk widgets are touched exclusively on the GUI thread. No API keys or long-term
agent memory are stored in the GUI's unencrypted convenience settings.
"""
from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, get_args

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox

from agent_vault import AgentVault
from prism_integration import PrismEngineMixin, PrismGuiMixin
from humoid_tui.agent import AgentHarness
from humoid_tui.config import Settings
from humoid_tui.local_models import LocalModelManager

# The TUI's runtime enhancement packages are patch installers, not UI code.
# Reuse all four in exactly the same order as its CLI entrypoint.
from humoid_tui.edit_cycle_runtime import install_edit_cycle_runtime
from humoid_tui.code_intelligence_runtime import install_code_intelligence_runtime
from humoid_tui.maintenance_runtime import install_maintenance_runtime
from humoid_tui.action_guard import install_action_guard
install_edit_cycle_runtime()
install_code_intelligence_runtime()
install_maintenance_runtime()
install_action_guard()

GROUPS = OrderedDict((name, fields.split()) for name, fields in {
    'Providers': '''humoid_provider humoid_language digitalocean_api_key digitalocean_base_url digitalocean_model
     digitalocean_timeout_seconds digitalocean_max_retries meta_api_key meta_base_url meta_model
     meta_enable_search_grounding meta_timeout_seconds meta_max_retries openai_api_key openai_base_url
     openai_model openai_timeout_seconds openai_max_retries llamacpp_api_key llamacpp_base_url
     llamacpp_model llamacpp_timeout_seconds llamacpp_max_retries litert_api_key
     litert_base_url litert_model litert_timeout_seconds litert_max_retries''',
    'Agent & Context': '''humoid_workspace humoid_autonomy_mode humoid_allow_shell humoid_shell_timeout_seconds
     humoid_max_tool_rounds humoid_max_output_tokens humoid_temperature humoid_context_limit
     humoid_gemma_context_limit humoid_glm_context_limit humoid_show_reasoning''',
    'Tool Protocols': '''humoid_tool_protocol humoid_parallel_tool_calls humoid_tool_argument_max_bytes
     humoid_tool_result_max_bytes humoid_tool_retry_limit humoid_tool_loop_repeat_limit
     humoid_require_tool_schema_validation humoid_require_tool_name_validation
     humoid_enable_tool_json_repair''',
    'Gemma 4': '''humoid_gemma4_mode humoid_gemma4_thinking humoid_gemma4_native_fallback
     humoid_gemma4_strip_completed_thoughts humoid_gemma_autostart''',
    'Advanced Memory': '''humoid_memory_backend humoid_memory_db humoid_memory_search_limit
     humoid_memory_context_limit humoid_memory_temporal_radius humoid_memory_packet_max_chars
     humoid_rhomgaic_enabled humoid_rhomgaic_db humoid_rhomgaic_dimensions
     humoid_rhomgaic_candidate_limit humoid_rhomgaic_graph_max_edges
     humoid_rhomgaic_bootstrap_limit humoid_rhomgaic_importance_half_life_days
     humoid_rhomgaic_policy_adaptation humoid_rhomgaic_semantic_capsules''',
    'CTX & Retrieval': '''humoid_rag_progressive humoid_rag_agentic_growth humoid_rag_graph_hops
     humoid_rag_reflex_max_chars humoid_rag_working_max_chars humoid_rag_context_cache
     humoid_rag_context_cache_ttl_seconds humoid_rag_context_cache_max_entries''',
    'Weaviate': '''weaviate_http_host weaviate_http_port weaviate_grpc_host weaviate_grpc_port
     weaviate_secure weaviate_api_key weaviate_embedded_version weaviate_embedded_data_path
     weaviate_embedded_binary_path weaviate_embedded_log_level
     weaviate_embedded_disable_telemetry weaviate_embedded_http_port
     weaviate_embedded_grpc_port weaviate_collection''',
    'Memory Schema': '''humoid_default_memory_tier humoid_default_memory_channel
     humoid_default_validation_status''',
}.items())

SENSITIVE = {'digitalocean_api_key', 'meta_api_key', 'openai_api_key', 'llamacpp_api_key', 'litert_api_key', 'weaviate_api_key'}
PALETTE = {'bg': '#101c19', 'panel': '#15261e', 'field': '#0c1612', 'accent': '#34d399',
           'muted': '#a4b6aa', 'text': '#f1fff5', 'line': '#315644'}


class AgentEngine(PrismEngineMixin):
    """Serializes all harness access in one dedicated, persistent event loop."""

    def __init__(self, post: Callable[[tuple], None]):
        self.post = post
        self.thread: threading.Thread | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.harness: AgentHarness | None = None
        self.vault: AgentVault | None = None
        self.settings: Settings | None = None
        self.local_models: LocalModelManager | None = None
        self.ready = threading.Event()
        self.active = False
        self._serial: asyncio.Lock | None = None
        self.current_future: Any | None = None

    def start_loop(self) -> None:
        if self.thread and self.thread.is_alive(): return
        self.ready.clear()
        def run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.loop = loop
            self._serial = asyncio.Lock()
            self.ready.set()
            try: loop.run_forever()
            finally:
                tasks = asyncio.all_tasks(loop)
                for task in tasks: task.cancel()
                if tasks: loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
                loop.close()
        self.thread = threading.Thread(target=run, name='HumoidAgentAsync', daemon=True)
        self.thread.start()
        if not self.ready.wait(5): raise RuntimeError('Agent event loop did not start')

    def submit(self, label: str, task: Callable[[], Any], *, persist: bool = True) -> None:
        if self.loop is None: raise RuntimeError('Agent event loop is not running')
        async def work():
            async with self._serial:
                try:
                    result = await task()
                    if persist and self.vault: self.vault.persist()
                    # Hybrid by default: replicate durable SQLite outbox to
                    # the authenticated local embedded index. The index is a
                    # derived cache; network or binary failures never erase it.
                    if persist and self.prism_index and self.prism:
                        try:
                            if self.prism.pending_index(limit=1):
                                self.prism_index.synchronize(limit=75)
                                self.vault.persist()
                        except Exception as sync_err:
                            self.post(('error','PRISM Weaviate sync',str(sync_err)))
                    self.post(('result', label, result))
                except BaseException as exc:
                    self.post(('error', label, f'{type(exc).__name__}: {exc}'))
                    if self.vault:
                        try: self.vault.persist()
                        except Exception as err: self.post(('error', 'Vault persistence', str(err)))
                finally: self.post(('finished', label, None))
        self.current_future = asyncio.run_coroutine_threadsafe(work(), self.loop)

    async def boot(self, key: bytes, saved: dict[str, Any]) -> str:
        if self.harness is not None: await self.shutdown()
        self.vault = AgentVault(key)
        a, b = self.vault.mount()
        overrides = {name: value for name, value in saved.items() if name in Settings.model_fields}
        # Legacy TUI code still uses the encrypted SQLite provider. PRISM adds
        # a parallel embedded Weaviate index sealed with the same vault key.
        overrides.setdefault('humoid_provider', 'litert')
        overrides.setdefault('litert_model', 'gemma-4-E2B-it')
        overrides['humoid_memory_backend'] = 'sqlite'
        overrides['humoid_memory_db'] = a
        overrides['humoid_rhomgaic_db'] = b
        overrides.setdefault('humoid_workspace', './workspace')
        self.settings = Settings(**overrides)
        self.harness = AgentHarness(self.settings, self.emit)
        gui_settings = {}
        gui_config = Path('gui_settings.json')
        if gui_config.exists():
            try: gui_settings = json.loads(gui_config.read_text(encoding='utf-8'))
            except (OSError, ValueError): pass
        if (self.settings.humoid_provider == 'litert'
                and saved.get('use_gui_litert_model', True)):
            from agent_litert import VaultGemmaProvider
            self.harness.provider = VaultGemmaProvider(
                key, self.settings,
                backend=gui_settings.get('inference_backend', 'Auto'),
            )
        self.harness.tools.approve_write = self.approve_write
        self.harness.tools.invented_tools_path = self.vault.active_dir / 'invented_tools.json'
        self.harness.tools.invented = self.vault.read_json('agent_invented_tools.json.aes', {})
        self.local_models = LocalModelManager()
        self.local_models.log_path = self.vault.active_dir / 'llama-server.log' 
        try:
            await self.harness.initialize()
        except BaseException:
            await self.shutdown()
            raise
        prism_status = self.prism_initialize(saved)
        # Bring legacy agent memories into the same ring index (idempotent).
        try:
            await self.prism_bridge_legacy(limit=500)
            if self.prism_index:
                self.prism_index.synchronize(limit=150)
            self.vault.persist()
        except Exception as exc:
            self.post(('error', 'PRISM reconciliation', str(exc)))
        # Resume the last saved desktop session into the model's context.
        archive = self.vault.read_json('agent_sessions.json.aes', {})
        active = archive.get('active_session', 'Session 1')
        turns = archive.get('chats', {}).get(active, [])
        if isinstance(turns, list):
            self.harness.messages.extend(
                {'role': t['role'], 'content': t['content']}
                for t in turns[-80:]
                if isinstance(t, dict) and t.get('role') in {'user', 'assistant'}
                and isinstance(t.get('content'), str)
            )
        self.active = True
        return ('Advanced agent ready • ' + self.harness.memory.status + ' • ' + prism_status
                + (' • Active DBs in RAM' if self.vault.ram_backed
                   else ' • Active DBs in a restricted temporary directory'))

    async def sealed_shutdown(self) -> str:
        """Drain queued GUI memory writes before sealing the active vault."""
        async with self._serial:
            return await self.shutdown()

    async def shutdown(self) -> str:
        self.active = False
        if self.harness:
            if self.vault:
                self.vault.write_json('agent_invented_tools.json.aes', self.harness.tools.invented)
            await self.harness.close()
        if self.local_models:
            try: await self.local_models.stop()
            except Exception: pass
        if self.vault:
            self.prism_shutdown()  # Close Weaviate and seal its index FIRST.
            self.vault.close()
        # Clear references only after an authenticated encrypted snapshot has
        # been durably committed; otherwise allow the user to retry sealing.
        self.harness = None
        self.settings = None
        self.local_models = None
        self.vault = None
        return 'Agent closed; memory encrypted and sealed' 

    async def emit(self, event) -> None:
        self.post(('event', event.kind, (event.message, event.detail)))

    async def approve_write(self, path: str, diff: str, content: str) -> tuple[bool, str]:
        event = threading.Event()
        payload = {'ready': event, 'approved': False, 'content': content,
                   'path': path, 'diff': diff}
        self.post(('approval', 'File change approval', payload))
        # Non-blocking from asyncio's point of view; GUI must answer before
        # the underlying tool is permitted to write anything.
        if not await asyncio.to_thread(event.wait, 300):
            return False, content
        return bool(payload['approved']), str(payload['content'])

    async def chat(self, prompt: str) -> str:
        if not self.harness: raise RuntimeError('Unlock the vault first')
        from humoid_tui.agent import is_simple_file_task
        # Retrieval text is historical evidence, never new instructions.
        recall = (await self.prism_context(prompt, project='agent-chat', budget=1600,
                                           strict_scope=True)
                  if self.prism_ctx_enabled and self.prism and not is_simple_file_task(prompt)
                  else '')
        augmented = (prompt + '\n\n[Untrusted PRISM memory context; source excerpts only]\n'
                     + recall + '\n[End memory context]') if recall else prompt
        # PRISM is the authority for this chat.  Do not let the older broad
        # Rhomgaic lookup append a second, unscoped memory packet afterwards.
        answer = await self.harness.run(
            augmented,
            skip_memory_recall=bool(self.prism_ctx_enabled and self.prism),
        )
        try: await self.prism_bridge_legacy(limit=500)
        except Exception: pass  # Original successful agent result must survive.
        if self.prism:
            self.prism.remember('Agent chat\nUser: ' + prompt[:6000]
                                + '\nAssistant: ' + str(answer)[:6000], ring=3,
                                project='agent-chat', importance=.5)
        return answer

    async def tool(self, name: str, args: dict) -> str:
        if not self.harness: raise RuntimeError('Agent is not ready')
        result = await self.harness.tools.execute(name, args)
        try: await self.prism_bridge_legacy(limit=500)
        except Exception: pass
        return result

    async def search(self, query: str, limit: int = 12):
        if not self.harness: raise RuntimeError('Agent is not ready')
        return await self.harness.memory.search(query, limit)

    async def context(self, query: str, depth: str):
        if not self.harness: raise RuntimeError('Agent is not ready')
        packet = await self.harness.memory.context_packet(query, depth=depth)
        return {'packet': packet, 'trace': self.harness.memory.last_trace}

    async def add_memory(self, text: str, channel: str, tier: str):
        if not self.harness: raise RuntimeError('Agent is not ready')
        item = await self.harness.memory.add(text, channel=channel, memory_tier=tier,
                                             validation_status='unverified')
        if self.prism:
            self.prism.import_legacy(str(item), text, tier=tier, channel=channel,
                                     verified=False)
        return item

    async def update_memory(self, id_: str, **changes):
        if not self.harness: raise RuntimeError('Agent is not ready')
        await self.harness.memory.update_memory(id_, **changes)
        if self.prism: await self.prism_bridge_legacy(limit=500)
        return 'Memory updated: ' + id_

    async def delete_memory(self, id_: str):
        if not self.harness: raise RuntimeError('Agent is not ready')
        await self.harness.memory.delete_memory(id_)
        if self.prism: self.prism.forget_source(id_)
        return 'Memory deleted: ' + id_

    async def memory_stats(self):
        if not self.harness: raise RuntimeError('Agent is not ready')
        return await self.harness.memory.surface_status()

    async def import_gui_history(self, rows: list[tuple]):
        if not self.harness: raise RuntimeError('Agent is not ready')
        # Persist high water mark in encrypted sessions metadata; not in a
        # plaintext imported-ids sidecar. Process records oldest first.
        existing = self.vault.read_json('agent_sessions.json.aes', {})
        seen = set(existing.get('imported_gui_ids', []))
        count = 0
        for row_id, timestamp, prompt, response in reversed(rows):
            if row_id in seen: continue
            await self.harness.memory.add(
                f'GUI chat at {timestamp}\nUser: {str(prompt)[:6000]}\nAssistant: {str(response)[:6000]}',
                channel='gui-history', memory_tier='episodic',
                task_id='gui-chat-' + str(row_id), validation_status='unverified',
            )
            seen.add(row_id)
            count += 1
        existing['imported_gui_ids'] = sorted(seen)
        self.vault.write_json('agent_sessions.json.aes', existing)
        return f'Imported {count} previously unbridged GUI conversations into advanced memory.'


class AdvancedAgentMixin(PrismGuiMixin):
    """CustomTkinter panels embedded in the original encrypted desktop app."""

    def agent_init(self):
        self.agent_queue: queue.Queue[tuple] = queue.Queue()
        self.agent_engine = AgentEngine(self.agent_queue.put)
        self.agent_settings_vars: dict[str, Any] = {}
        self.agent_settings: dict[str, Any] = {}
        self.agent_log_items: list[str] = []
        self.agent_chat_turns: list[dict[str, str]] = []
        self.agent_sessions: dict[str, Any] = {}
        self.agent_session_name = 'Session 1'
        self.agent_busy = False
        self.agent_foreground_label = None
        self.agent_unlocked = False
        self.agent_status = tk.StringVar(value='Unlock the GUI vault to initialize advanced memory')
        self.agent_model_status = tk.StringVar(value='No provider connected')
        self.agent_ctx_status = tk.StringVar(value='CTX: locked')
        self.agent_tool_description = tk.StringVar(value='Select a tool to inspect its schema.')
        self.agent_memory_status = tk.StringVar(value='Memory is encrypted at rest')
        self.agent_activity_filter = tk.StringVar(value='all')
        self.after(120, self.agent_poll)

    def agent_title(self, parent, title, subtitle=''):
        head = ctk.CTkFrame(parent, fg_color='transparent')
        head.pack(fill='x', padx=14, pady=(12, 8))
        ctk.CTkLabel(head, text=title, font=ctk.CTkFont(size=23, weight='bold'),
                     text_color=PALETTE['accent']).pack(anchor='w')
        if subtitle:
            ctk.CTkLabel(head, text=subtitle, text_color=PALETTE['muted'],
                         wraplength=850, justify='left').pack(anchor='w', pady=4)
        return head

    def agent_button(self, parent, label, callback):
        button = ctk.CTkButton(parent, text=label, command=callback,
                               fg_color='#225e48', hover_color='#32825c',
                               corner_radius=9, height=34)
        button.pack(side='left', padx=(0, 8), pady=5)
        return button

    def agent_box(self, parent, height=200):
        obj = ctk.CTkTextbox(parent, height=height, fg_color=PALETTE['field'],
                             text_color=PALETTE['text'], wrap='word',
                             font=ctk.CTkFont(family='DejaVu Sans Mono', size=12))
        obj.pack(fill='both', expand=True, padx=14, pady=7)
        return obj

    def agent_write(self, box, text, *, clear=False):
        if not box or not box.winfo_exists(): return
        box.configure(state='normal')
        if clear: box.delete('1.0', 'end')
        box.insert('end', str(text) + '\n')
        box.see('end')
        box.configure(state='disabled')

    def build_agent_tab(self):
        panel = self.agent_tab
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(1, weight=1)
        top = ctk.CTkFrame(panel, fg_color='transparent')
        top.grid(row=0, column=0, sticky='ew', padx=8, pady=6)
        ctk.CTkLabel(top, text='HUMOID  •  AGENT STUDIO',
                     font=ctk.CTkFont(size=19, weight='bold'),
                     text_color=PALETTE['accent']).pack(side='left', padx=10)
        ctk.CTkLabel(top, textvariable=self.agent_status, text_color=PALETTE['muted'],
                     wraplength=620).pack(side='left', padx=20)
        ctk.CTkButton(top, text='Retry agent', width=100,
                      command=self.agent_after_unlock).pack(side='right', padx=6)
        self.agent_pages = ctk.CTkTabview(panel, fg_color=PALETTE['bg'],
                                          segmented_button_selected_color='#176647',
                                          segmented_button_fg_color='#1c3629',
                                          segmented_button_selected_hover_color='#218258')
        self.agent_pages.grid(row=1, column=0, sticky='nsew', padx=6, pady=3)
        for name in ('Agent Chat', 'Memory Lab', 'Ring Observatory', 'ChronoOrbit',
                     'CTX Lab', 'Tool Studio', 'Local Models', 'Advanced Settings', 'Activity'):
            self.agent_pages.add(name)
        self.build_agent_chat(self.agent_pages.tab('Agent Chat'))
        self.build_agent_memory(self.agent_pages.tab('Memory Lab'))
        self.build_prism_ring_panel(self.agent_pages.tab('Ring Observatory'))
        self.build_prism_task_panel(self.agent_pages.tab('ChronoOrbit'))
        self.build_agent_context(self.agent_pages.tab('CTX Lab'))
        self.build_agent_tools(self.agent_pages.tab('Tool Studio'))
        self.build_agent_models(self.agent_pages.tab('Local Models'))
        self.build_agent_settings(self.agent_pages.tab('Advanced Settings'))
        self.build_agent_activity(self.agent_pages.tab('Activity'))

    def build_agent_chat(self, page):
        self.agent_title(page, 'Autonomous coding agent',
                         'Full TUI orchestration: multi-provider chat, workspace tools, approved edits, project perspective, memory recall, edit cycles and maintenance.')
        options = ctk.CTkFrame(page, fg_color='transparent')
        options.pack(fill='x', padx=14)
        self.agent_session_var = tk.StringVar(value=self.agent_session_name)
        self.agent_session_menu = ctk.CTkOptionMenu(options, values=['Session 1'],
                            variable=self.agent_session_var, command=self.agent_change_session, width=230)
        self.agent_session_menu.pack(side='left', padx=(0, 8))
        self.agent_button(options, 'New session', self.agent_new_session)
        self.agent_button(options, 'Cancel', self.agent_cancel)
        self.agent_button(options, 'Reset context', self.agent_reset_context)
        self.agent_button(options, 'Memory recall', lambda: self.agent_pages.set('Memory Lab'))
        self.agent_button(options, 'CTX controls', lambda: self.agent_pages.set('CTX Lab'))
        ctk.CTkLabel(page, textvariable=self.agent_model_status,
                     text_color=PALETTE['muted']).pack(anchor='w', padx=14)
        self.agent_chat_box = self.agent_box(page, height=300)
        compose = ctk.CTkFrame(page, fg_color='transparent')
        compose.pack(fill='x', padx=12, pady=(0, 14))
        compose.grid_columnconfigure(0, weight=1)
        self.agent_prompt = ctk.CTkTextbox(compose, height=86,
                                           fg_color=PALETTE['field'], wrap='word')
        self.agent_prompt.grid(row=0, column=0, sticky='ew', padx=4)
        send = ctk.CTkButton(compose, text='Run agent  ➜', width=130,
                             command=self.agent_send, fg_color='#21855d')
        send.grid(row=0, column=1, sticky='n', padx=5)
        self.agent_chat_box.configure(state='disabled')

    def build_agent_memory(self, page):
        self.agent_title(page, 'Encrypted Rhomgaic / Chromatic memory',
                         'SQLite provider + Rhomgaic topology, temporal memories, procedural playbooks, semantic capsules and progressive recall. Both databases are encrypted with your GUI vault key.')
        controls = ctk.CTkFrame(page, fg_color='transparent')
        controls.pack(fill='x', padx=14)
        self.agent_memory_query = ctk.CTkEntry(controls, placeholder_text='Search experiences, files, symbols, procedures…', width=390)
        self.agent_memory_query.pack(side='left', padx=(0, 7))
        self.agent_button(controls, 'Search', self.agent_memory_search)
        self.agent_button(controls, 'Surface statistics', self.agent_memory_stats)
        self.agent_button(controls, 'Import GUI history', self.agent_import_history)
        self.agent_memory_results = self.agent_box(page, height=180)
        self.agent_memory_results.configure(state='disabled')
        controls2 = ctk.CTkFrame(page, fg_color='transparent')
        controls2.pack(fill='x', padx=14)
        self.agent_memory_id = ctk.CTkEntry(controls2, placeholder_text='Memory UUID for verify/delete', width=360)
        self.agent_memory_id.pack(side='left', padx=(0, 8))
        self.agent_button(controls2, 'Mark verified', self.agent_memory_verify)
        self.agent_button(controls2, 'Delete selected ID', self.agent_memory_delete)
        self.agent_memory_add_box = ctk.CTkTextbox(page, height=95, fg_color=PALETTE['field'])
        self.agent_memory_add_box.pack(fill='x', padx=14)
        save = ctk.CTkFrame(page, fg_color='transparent')
        save.pack(fill='x', padx=14)
        self.agent_memory_tier = ctk.CTkOptionMenu(save, values=['episodic', 'semantic', 'procedural'], width=140)
        self.agent_memory_tier.pack(side='left', padx=5)
        self.agent_memory_channel = ctk.CTkEntry(save, placeholder_text='channel (user / agent / project)', width=245)
        self.agent_memory_channel.pack(side='left', padx=5)
        self.agent_button(save, 'Save memory', self.agent_memory_add)
        ctk.CTkLabel(page, textvariable=self.agent_memory_status, text_color=PALETTE['muted']).pack(anchor='w', padx=14)

    def build_agent_context(self, page):
        self.agent_title(page, 'Context engineering · CTX',
                         'Run reflex, working and subsurface recall. Adjust per-provider token limits, caching, graph traversal and context budgets under Advanced Settings → Agent & Context / CTX & Retrieval.')
        ctr = ctk.CTkFrame(page, fg_color='transparent')
        ctr.pack(fill='x', padx=14)
        self.agent_context_query = ctk.CTkEntry(ctr, width=420, placeholder_text='Question / task for a contextual recall')
        self.agent_context_query.pack(side='left', padx=5)
        self.agent_context_depth = ctk.CTkOptionMenu(ctr,
                             values=['reflex', 'working', 'subsurface', 'auto'], width=145)
        self.agent_context_depth.set('working')
        self.agent_context_depth.pack(side='left', padx=5)
        self.agent_button(ctr, 'Build CTX packet', self.agent_context_run)
        self.agent_button(ctr, 'Memory topology', lambda: self.agent_run_tool('memory_surface_status', {}))
        gui_ctx = ctk.CTkFrame(page, fg_color='transparent')
        gui_ctx.pack(fill='x', padx=18, pady=7)
        ctk.CTkLabel(gui_ctx, text='Memory depth injected into original encrypted Gemma Chat:').pack(side='left', padx=(0, 8))
        self.agent_gui_ctx_depth = tk.StringVar(value='reflex')
        ctk.CTkOptionMenu(gui_ctx, values=['reflex', 'working', 'subsurface', 'auto'],
                          variable=self.agent_gui_ctx_depth, width=145).pack(side='left')
        ctk.CTkLabel(gui_ctx, text='Save in Advanced Settings to apply to the next GUI Chat request.',
                     text_color=PALETTE['muted']).pack(side='left', padx=10)
        ctk.CTkLabel(page, textvariable=self.agent_ctx_status,
                     text_color=PALETTE['accent']).pack(anchor='w', padx=18)
        self.agent_context_results = self.agent_box(page, height=370)
        self.agent_context_results.configure(state='disabled')

    def build_agent_tools(self, page):
        self.agent_title(page, 'Agent tool workbench',
                         'Every tool registered by the original TUI is available here, including AST search, project perspective, edit-cycle verification, maintenance plans and procedural recall. File writes require a diff approval dialog.')
        controls = ctk.CTkFrame(page, fg_color='transparent')
        controls.pack(fill='x', padx=14)
        self.agent_tool_var = tk.StringVar(value='list_files')
        self.agent_tool_menu = ctk.CTkOptionMenu(controls, variable=self.agent_tool_var,
                                 values=['list_files'], width=315, command=self.agent_tool_selected)
        self.agent_tool_menu.pack(side='left', padx=5)
        self.agent_button(controls, 'Run tool', self.agent_tool_execute)
        self.agent_button(controls, 'Undo file write', lambda: self.agent_run_tool('undo_file_change', {}))
        ctk.CTkLabel(page, textvariable=self.agent_tool_description, justify='left',
                     text_color=PALETTE['muted'], wraplength=1050).pack(anchor='w', padx=18, pady=7)
        ctk.CTkLabel(page, text='Tool arguments (JSON object)',
                     text_color=PALETTE['muted']).pack(anchor='w', padx=18)
        self.agent_tool_args = ctk.CTkTextbox(page, height=125, fg_color=PALETTE['field'])
        self.agent_tool_args.pack(fill='x', padx=14)
        self.agent_tool_args.insert('1.0', '{\n  "path": "."\n}')
        self.agent_tool_output = self.agent_box(page, height=240)
        self.agent_tool_output.configure(state='disabled')

    def build_agent_models(self, page):
        self.agent_title(page, 'Local models and provider runtime',
                         'TUI GGUF model management and provider probing. LiteRT GUI model downloads remain in the existing Download Model tab. API keys are stored only in the encrypted agent settings vault.')
        ctrl = ctk.CTkFrame(page, fg_color='transparent')
        ctrl.pack(fill='x', padx=14)
        self.agent_button(ctrl, 'Detect hardware', lambda: self.agent_model_task('detect'))
        self.agent_button(ctrl, 'List GGUF', lambda: self.agent_model_task('list'))
        self.agent_button(ctrl, 'Provider health', lambda: self.agent_model_task('health'))
        self.agent_button(ctrl, 'Stop llama-server', lambda: self.agent_model_task('stop'))
        self.agent_button(ctrl, 'Install runtime', lambda: self.agent_model_task('install'))
        self.agent_button(ctrl, 'Reinstall', lambda: self.agent_model_task('reinstall'))
        self.agent_button(ctrl, 'Uninstall', lambda: self.agent_model_task('uninstall'))
        self.agent_use_gui_model = tk.BooleanVar(value=True)
        ctk.CTkSwitch(page, text='Use encrypted desktop Gemma directly (LiteRT provider; no HTTP shim)',
                      variable=self.agent_use_gui_model).pack(anchor='w', padx=18, pady=10)
        args = ctk.CTkFrame(page, fg_color='transparent')
        args.pack(fill='x', padx=14)
        self.agent_model_path = ctk.CTkEntry(args, width=380, placeholder_text='GGUF filename inside .humoid/models')
        self.agent_model_path.pack(side='left', padx=5)
        self.agent_button(args, 'Launch GGUF', lambda: self.agent_model_task('launch'))
        self.agent_button(args, 'Inspect GGUF', lambda: self.agent_model_task('inspect'))
        self.agent_button(args, 'Benchmark', lambda: self.agent_model_task('benchmark'))
        dl = ctk.CTkFrame(page, fg_color='transparent')
        dl.pack(fill='x', padx=14)
        self.agent_model_repo = ctk.CTkEntry(dl, width=420, placeholder_text='Hugging Face repository ID')
        self.agent_model_repo.pack(side='left', padx=5)
        self.agent_model_filename = ctk.CTkEntry(dl, width=280, placeholder_text='GGUF filename in repository')
        self.agent_model_filename.pack(side='left', padx=5)
        self.agent_button(dl, 'Download GGUF', lambda: self.agent_model_task('download'))
        self.agent_model_results = self.agent_box(page, height=290)
        self.agent_model_results.configure(state='disabled')

    def build_agent_settings(self, page):
        self.agent_title(page, 'Full TUI configuration',
                         'All original Pydantic settings are editable here and saved inside the GUI AES-GCM vault; settings are validated before the agent restarts. Encryption requires SQLite as the authoritative memory backend. PRISM embedded Weaviate is AES-GCM sealed when locked; the legacy TUI backend remains SQLite. A non-embedded remote Weaviate installation is NOT covered by that encryption.')
        self.agent_settings_pages = ctk.CTkTabview(page, fg_color=PALETTE['bg'],
                                    segmented_button_selected_color='#176647')
        self.agent_settings_pages.pack(fill='both', expand=True, padx=8)
        handled: set[str] = set()
        for group, names in GROUPS.items():
            self.agent_settings_pages.add(group)
            scroll = ctk.CTkScrollableFrame(self.agent_settings_pages.tab(group),
                                            fg_color=PALETTE['bg'])
            scroll.pack(fill='both', expand=True, padx=5)
            scroll.grid_columnconfigure(1, weight=1)
            for name in names:
                if name in Settings.model_fields:
                    self.agent_setting_control(scroll, name)
                    handled.add(name)
        missing = set(Settings.model_fields) - handled
        if missing:
            self.agent_settings_pages.add('Additional')
            scroll = ctk.CTkScrollableFrame(self.agent_settings_pages.tab('Additional'))
            scroll.pack(fill='both', expand=True)
            for name in sorted(missing): self.agent_setting_control(scroll, name)
        prism_flags = ctk.CTkFrame(page, fg_color='transparent')
        prism_flags.pack(fill='x', padx=14, pady=4)
        self.prism_embed_enabled = tk.BooleanVar(value=True)
        self.prism_ctx_enabled = tk.BooleanVar(value=True)
        ctk.CTkSwitch(prism_flags,text='Encrypted embedded Weaviate (default; requires Linux tmpfs + optional client)',
                      variable=self.prism_embed_enabled).pack(side='left',padx=8)
        ctk.CTkSwitch(prism_flags,text='PRISM/Weaviate retrieval in Agent Chat + main Chat',
                      variable=self.prism_ctx_enabled).pack(side='left',padx=8)
        prism_advanced = ctk.CTkFrame(page, fg_color='transparent')
        prism_advanced.pack(fill='x',padx=14)
        self.prism_version_var = tk.StringVar(value='1.40.0')
        self.prism_http_port_var = tk.StringVar(value='18079')
        self.prism_grpc_port_var = tk.StringVar(value='15060')
        for label,var,width in [('Weaviate embedded version',self.prism_version_var,100),
                                ('HTTP port',self.prism_http_port_var,85),
                                ('gRPC port',self.prism_grpc_port_var,85)]:
            ctk.CTkLabel(prism_advanced,text=label).pack(side='left',padx=5)
            ctk.CTkEntry(prism_advanced,textvariable=var,width=width).pack(side='left',padx=5)
        foot = ctk.CTkFrame(page, fg_color='transparent')
        foot.pack(fill='x', padx=14, pady=5)
        self.agent_button(foot, 'Save encrypted settings & restart agent', self.agent_save_settings)
        self.agent_button(foot, 'Reload saved values', self.agent_refresh_settings)
        ctk.CTkLabel(foot, text='Shell off by default • change review required • keys never written to .env',
                     text_color=PALETTE['muted']).pack(side='left')

    def agent_setting_control(self, frame, name):
        field = Settings.model_fields[name]
        row = len(self.agent_settings_vars)
        ctk.CTkLabel(frame, text=name.upper(), text_color=PALETTE['muted'],
                     anchor='w').grid(row=row, column=0, sticky='w', padx=9, pady=5)
        default = field.default
        annotation = field.annotation
        values = get_args(annotation)
        if isinstance(default, bool):
            var = tk.BooleanVar(value=default)
            widget = ctk.CTkSwitch(frame, text='Enabled', variable=var)
        elif values and all(isinstance(x, str) for x in values):
            var = tk.StringVar(value=str(default))
            widget = ctk.CTkOptionMenu(frame, values=list(values), variable=var, width=265)
        else:
            var = tk.StringVar(value='')
            widget = ctk.CTkEntry(frame, textvariable=var, show='●' if name in SENSITIVE else '',
                                  width=400, fg_color=PALETTE['field'])
        widget.grid(row=row, column=1, sticky='ew', padx=9, pady=5)
        self.agent_settings_vars[name] = var

    def build_agent_activity(self, page):
        self.agent_title(page, 'Activity · diagnostics · agent tree',
                         'Live TUI event stream and protocol diagnostics: rounds, tools, memory recall, token/context telemetry and provider health.')
        ctr = ctk.CTkFrame(page, fg_color='transparent')
        ctr.pack(fill='x', padx=14)
        self.agent_button(ctr, 'Clear activity', self.agent_clear_activity)
        self.agent_button(ctr, 'Memory status', self.agent_memory_stats)
        self.agent_button(ctr, 'Protocol / agents', self.agent_show_profile)
        self.agent_activity_box = self.agent_box(page, height=450)
        self.agent_activity_box.configure(state='disabled')
        self.agent_write(self.agent_activity_box,
            'root\n ├─ planner / project perspective\n ├─ tool router / file approval\n ├─ memory router / SQLite + Rhomgaic\n └─ provider / model protocol\n')

    # ---- UI actions / controlled agent lifecycle ---------------------------
    def agent_poll(self):
        if getattr(self, 'closing', False): return
        for _ in range(100):
            try: kind, label, result = self.agent_queue.get_nowait()
            except queue.Empty: break
            if kind == 'event':
                msg, detail = result
                if label in {'reasoning', 'thought'} and not self.agent_show_reasoning():
                    continue
                self.agent_activity(f'[{label}] {msg}' + (f' · {json.dumps(detail, default=str)[:1200]}' if detail else ''))
            elif kind == 'approval': self.agent_diff_dialog(result)
            elif kind == 'error':
                if label in {'Agent initialized', 'Settings applied'}:
                    self.agent_unlocked = False
                self.agent_activity(f'[ERROR · {label}] {result}')
                self.agent_status.set(f'{label} failed: {result}')
                if label == self.agent_foreground_label:
                    self.agent_busy = False
                    self.agent_foreground_label = None
            elif kind == 'finished':
                if label == self.agent_foreground_label:
                    self.agent_busy = False
                    self.agent_foreground_label = None
            elif kind == 'result': self.agent_result(label, result)
        self.after(100, self.agent_poll)

    def agent_show_reasoning(self):
        try: return bool(self.agent_settings_vars['humoid_show_reasoning'].get())
        except Exception: return True

    def agent_activity(self, text):
        self.agent_log_items.append(text)
        self.agent_log_items = self.agent_log_items[-700:]
        if hasattr(self, 'agent_activity_box'):
            self.agent_write(self.agent_activity_box, text)

    def agent_task(self, label, func, *, persist=True):
        if self.key is None or not self.agent_unlocked:
            self.agent_status.set('Unlock GUI vault first (or retry agent initialization).')
            return
        if self.agent_busy:
            self.agent_status.set('Agent is running another operation.')
            return
        self.agent_busy = True
        self.agent_foreground_label = label
        self.agent_status.set(f'Agent: {label}…')
        self.agent_engine.submit(label, func, persist=persist)

    def agent_after_unlock(self):
        if self.key is None or self.agent_busy:
            return
        try:
            self.agent_engine.start_loop()
            vault = AgentVault(self.key)
            self.agent_settings = vault.read_json('agent_settings.json.aes', {})
            self.agent_use_gui_model.set(bool(self.agent_settings.get('use_gui_litert_model', True)))
            self.agent_gui_ctx_depth.set(str(self.agent_settings.get('gui_memory_depth', 'reflex')))
            self.agent_sessions = vault.read_json('agent_sessions.json.aes', {})
            self.agent_session_name = self.agent_sessions.get('active_session', 'Session 1')
            self.agent_chat_turns = list(self.agent_sessions.get('chats', {}).get(self.agent_session_name, []))
            self.agent_refresh_settings()
            self.agent_update_session_menu()
            self.agent_render_chat()
            self.agent_unlocked = True
            self.agent_busy = True
            self.agent_foreground_label = 'Agent initialized'
            self.agent_engine.submit('Agent initialized', lambda: self.agent_engine.boot(self.key, self.agent_settings), persist=False)
        except Exception as exc:
            self.agent_status.set(f'Agent initialization failed: {exc}')
            self.agent_unlocked = False

    def agent_before_lock(self):
        if not self.agent_unlocked:
            # Error during initial boot may leave a partially mounted vault.
            if self.agent_engine.vault and self.agent_engine.loop:
                future = asyncio.run_coroutine_threadsafe(self.agent_engine.sealed_shutdown(), self.agent_engine.loop)
                future.result(timeout=45)
            return
        if self.agent_busy:
            raise RuntimeError('Finish the active agent operation before locking.')
        self.agent_store_sessions()
        if self.agent_engine.loop:
            future = asyncio.run_coroutine_threadsafe(self.agent_engine.sealed_shutdown(), self.agent_engine.loop)
            future.result(timeout=45)
        self.agent_unlocked = False
        self.agent_foreground_label = None
        self.agent_chat_turns.clear()
        self.agent_sessions = {}
        self.agent_settings = {}
        for name, var in self.agent_settings_vars.items():
            if name in SENSITIVE: var.set('')
        self.agent_render_chat()
        self.agent_status.set('Advanced memory sealed. Unlock GUI vault to resume.')

    def agent_store_sessions(self):
        if not self.agent_unlocked or self.key is None: return
        vault = AgentVault(self.key)
        persisted = vault.read_json('agent_sessions.json.aes', {})
        if 'imported_gui_ids' in persisted:
            self.agent_sessions['imported_gui_ids'] = persisted['imported_gui_ids']
        self.agent_sessions.setdefault('chats', {})[self.agent_session_name] = self.agent_chat_turns[-120:]
        self.agent_sessions['active_session'] = self.agent_session_name
        vault.write_json('agent_sessions.json.aes', self.agent_sessions)

    def agent_result(self, label, value):
        if label.startswith('PRISM') and self.prism_handle_result(label, value):
            return
        if label == 'Agent initialized':
            self.prism_index_state.set(self.agent_engine.prism_state)
            self.agent_status.set(str(value))
            self.agent_model_status.set('Active: ' + self.agent_engine.settings.humoid_provider
                + ' / ' + self.agent_engine.harness.provider.cfg.model
                + ' • protocol: ' + self.agent_engine.harness.protocol.name)
            self.agent_update_tool_menu()
        elif label == 'Agent response':
            self.agent_chat_turns.append({'role': 'assistant', 'content': str(value)})
            self.agent_render_chat()
            self.agent_store_sessions()
            self.agent_status.set('Agent response sealed in memory vault')
        elif label == 'Memory search':
            lines = []
            for hit in value:
                lines.append(f'[{hit.memory_tier}/{hit.channel}] {hit.memory_id}\n'
                             f'Score: {hit.score:.4f} | validation: {hit.validation_status}\n{hit.text}\n')
            self.agent_write(self.agent_memory_results, '\n'.join(lines) if lines else 'No matching memories.', clear=True)
            self.agent_memory_status.set(f'{len(value)} memories returned')
        elif label == 'Memory stats':
            report = json.dumps(value, indent=2, default=str)
            self.agent_write(self.agent_memory_results, report, clear=True)
            self.agent_write(self.agent_activity_box, '[MEMORY STATUS]\n' + report[:10000])
        elif label == 'CTX packet':
            self.agent_write(self.agent_context_results,
                str(value['packet']) + '\n\nRECALL TRACE\n' + json.dumps(value['trace'], indent=2, default=str), clear=True)
            self.agent_ctx_status.set(f'CTX: {self.agent_context_depth.get()} • packet {len(str(value["packet"]))} chars')
        elif label.startswith('Tool:'):
            self.agent_write(self.agent_tool_output, str(value), clear=True)
            self.agent_status.set(f'{label} completed')
        elif label.startswith('Model:'):
            self.agent_write(self.agent_model_results, json.dumps(value, indent=2, default=str), clear=True)
        elif label == 'Settings applied':
            self.prism_index_state.set(self.agent_engine.prism_state)
            self.agent_status.set(str(value))
            self.agent_update_tool_menu()
            self.agent_model_status.set(f'Active: {self.agent_engine.settings.humoid_provider} / {self.agent_engine.harness.provider.cfg.model}')
        else:
            self.agent_status.set(str(value)[:350])
            self.agent_memory_status.set(str(value)[:250])
            self.agent_activity(f'[{label}] {str(value)[:2500]}')

    def agent_render_chat(self):
        if not hasattr(self, 'agent_chat_box'): return
        lines=[]
        for turn in self.agent_chat_turns:
            lines.append(('YOU' if turn['role']=='user' else 'HUMOID AGENT') + ':\n' + turn['content'] + '\n')
        self.agent_write(self.agent_chat_box, '\n'.join(lines) or 'Agent conversation ready.\n', clear=True)

    def agent_cancel(self):
        if self.agent_engine.current_future and not self.agent_engine.current_future.done():
            self.agent_engine.current_future.cancel()
            self.agent_status.set('Cancelling active agent operation...')
            self.agent_engine.local_models and self.agent_engine.local_models.cancel_download()

    def agent_send(self):
        prompt = self.agent_prompt.get('1.0','end').strip()
        if not prompt: return
        if self.key is None or not self.agent_unlocked or self.agent_busy:
            self.agent_status.set('Agent is locked or busy')
            return
        self.agent_chat_turns.append({'role': 'user', 'content': prompt})
        self.agent_render_chat()
        self.agent_prompt.delete('1.0','end')
        self.agent_store_sessions()
        self.agent_task('Agent response', lambda: self.agent_engine.chat(prompt))

    def agent_update_session_menu(self):
        if not hasattr(self, 'agent_session_menu'): return
        names = list(self.agent_sessions.get('chats', {})) or ['Session 1']
        if self.agent_session_name not in names: names.append(self.agent_session_name)
        self.agent_session_menu.configure(values=names)
        self.agent_session_var.set(self.agent_session_name)

    def agent_change_session(self, name):
        if self.agent_busy: return
        self.agent_store_sessions()
        self.agent_session_name = name
        self.agent_chat_turns = list(self.agent_sessions.get('chats', {}).get(name, []))
        if self.agent_engine.harness and self.agent_engine.settings:
            # Keep actual model context aligned with the visible conversation.
            system = self.agent_engine.harness.messages[0]
            self.agent_engine.loop.call_soon_threadsafe(
                setattr, self.agent_engine.harness, 'messages', [system] + list(self.agent_chat_turns))
        self.agent_render_chat()

    def agent_new_session(self):
        if not self.agent_unlocked or self.agent_busy: return
        self.agent_store_sessions()
        names = self.agent_sessions.get('chats', {})
        number = 1
        while f'Session {number}' in names: number += 1
        self.agent_session_name = f'Session {number}'
        self.agent_chat_turns = []
        if self.agent_engine.harness:
            self.agent_engine.loop.call_soon_threadsafe(
                setattr, self.agent_engine.harness, 'messages', self.agent_engine.harness.messages[:1])
        self.agent_store_sessions()
        self.agent_update_session_menu()
        self.agent_render_chat()

    def agent_reset_context(self):
        if self.agent_busy or not self.agent_engine.harness: return
        self.agent_engine.loop.call_soon_threadsafe(
            setattr, self.agent_engine.harness, 'messages', self.agent_engine.harness.messages[:1])
        self.agent_ctx_status.set('Context accordion reset; durable advanced memory preserved')

    def agent_memory_search(self):
        query = self.agent_memory_query.get().strip()
        if query: self.agent_task('Memory search', lambda: self.agent_engine.search(query))

    def agent_memory_stats(self):
        self.agent_task('Memory stats', self.agent_engine.memory_stats)

    def agent_memory_add(self):
        text = self.agent_memory_add_box.get('1.0','end').strip()
        if not text: return
        channel = self.agent_memory_channel.get().strip() or 'user'
        tier = self.agent_memory_tier.get()
        self.agent_task('Memory saved', lambda: self.agent_engine.add_memory(text, channel, tier))

    def agent_memory_verify(self):
        id_ = self.agent_memory_id.get().strip()
        if id_: self.agent_task('Memory verified', lambda: self.agent_engine.update_memory(id_, validation_status='verified'))

    def agent_memory_delete(self):
        id_ = self.agent_memory_id.get().strip()
        if id_ and messagebox.askyesno('Delete memory?', f'Permanently delete advanced memory {id_}?'):
            self.agent_task('Memory deleted', lambda: self.agent_engine.delete_memory(id_))

    def agent_import_history(self):
        if self.key is None: return
        try: rows = self.fetch_history_for_agent()
        except Exception as exc:
            self.agent_status.set(f'Unable to read GUI vault history: {exc}')
            return
        if rows and messagebox.askyesno('Import GUI conversations',
                  f'Index {len(rows)} encrypted GUI conversation records into advanced memory? Existing records will not be duplicated.'):
            self.agent_task('GUI history import', lambda: self.agent_engine.import_gui_history(rows))

    def fetch_history_for_agent(self):
        # Main app supplies its own encrypted history reader via an override.
        from main import fetch_history, count_history_rows
        count = min(count_history_rows(self.key), 5000)
        return fetch_history(self.key, limit=count)

    def agent_context_run(self):
        q = self.agent_context_query.get().strip()
        if q:
            self.agent_task('CTX packet', lambda: self.agent_engine.context(q, self.agent_context_depth.get()))

    def agent_update_tool_menu(self):
        if not self.agent_engine.harness: return
        schemas = self.agent_engine.harness.tools.schemas()
        self.agent_schemas = {x['function']['name']: x['function'] for x in schemas}
        values = sorted(self.agent_schemas)
        self.agent_tool_menu.configure(values=values)
        self.agent_tool_var.set('list_files' if 'list_files' in values else values[0])
        self.agent_tool_selected(self.agent_tool_var.get())

    def agent_tool_selected(self, name):
        schema = getattr(self, 'agent_schemas', {}).get(name)
        if not schema: return
        self.agent_tool_description.set(schema.get('description',''))
        props = schema.get('parameters',{}).get('properties',{})
        example = {name: ('example' if meta.get('type')=='string' else 1 if meta.get('type')=='integer' else []
                         if meta.get('type')=='array' else {} if meta.get('type')=='object' else False)
                   for name,meta in props.items() if name in schema.get('parameters',{}).get('required',[])}
        self.agent_tool_args.delete('1.0','end')
        self.agent_tool_args.insert('1.0',json.dumps(example,indent=2))

    def agent_tool_execute(self):
        try:
            args = json.loads(self.agent_tool_args.get('1.0','end') or '{}')
            if not isinstance(args,dict): raise ValueError('Arguments must be a JSON object')
        except Exception as exc:
            self.agent_status.set('Invalid tool JSON: '+str(exc))
            return
        self.agent_run_tool(self.agent_tool_var.get(), args)

    def agent_run_tool(self, name, args):
        self.agent_pages.set('Tool Studio')
        if name == 'run_command' and not messagebox.askyesno('Shell command',
                'Run a shell command with your OS user permissions? Only do this in a trusted, isolated workspace.'):
            return
        self.agent_task('Tool: '+name, lambda: self.agent_engine.tool(name,args))

    def agent_model_task(self, action):
        requested_path = self.agent_model_path.get().strip()
        requested_repo = self.agent_model_repo.get().strip()
        requested_filename = self.agent_model_filename.get().strip()
        async def operation():
            manager=self.agent_engine.local_models
            if not manager: raise RuntimeError('Agent not initialized')
            if action=='detect': return vars(await manager.detect_hardware())
            if action=='list': return [str(x) for x in manager.models()]
            if action=='health': return await self.agent_engine.harness.provider.health()
            if action=='stop': return await manager.stop()
            if action in {'install','reinstall','uninstall'}:
                return await manager.mutate_package(action, await manager.detect_hardware())
            base = manager.model_dir.resolve()
            target = (base/requested_path).resolve()
            if action in ('launch','inspect','benchmark'):
                if base != target and base not in target.parents: raise ValueError('Model filename must be inside .humoid/models')
                if not target.is_file(): raise FileNotFoundError(target)
                if action=='inspect': return await manager.inspect(target)
                if action=='benchmark': return await manager.benchmark(target)
                if action=='launch':
                    result=await manager.launch(target, context=self.agent_engine.settings.humoid_gemma_context_limit)
                    await manager.wait_until_ready()
                    return result
            if action=='download':
                repo=requested_repo
                filename=requested_filename
                if not repo or not filename: raise ValueError('Enter a repository and filename')
                return str(await manager.download(repo,filename))
            raise ValueError('Unknown model action')
        if action in {'launch','download','benchmark','install','reinstall','uninstall'} and not messagebox.askyesno(
                'Confirm model operation',f'{action} may download files, start a local process, modify Python packages, or use significant resources. Continue?'):
            return
        self.agent_task('Model: '+action,operation,persist=False)

    def agent_refresh_settings(self):
        if not self.agent_unlocked and self.key is None: return
        raw = self.agent_settings or {}
        defaults = Settings(_env_file=None, humoid_provider='litert', litert_model='gemma-4-E2B-it',
                            humoid_memory_backend='sqlite')
        if hasattr(self, 'agent_use_gui_model'):
            self.agent_use_gui_model.set(bool(raw.get('use_gui_litert_model', True)))
        if hasattr(self, 'agent_gui_ctx_depth'):
            self.agent_gui_ctx_depth.set(str(raw.get('gui_memory_depth', 'reflex')))
        if hasattr(self, 'prism_embed_enabled'):
            self.prism_embed_enabled.set(bool(raw.get('prism_embedded_enabled', True)))
        if hasattr(self, 'prism_ctx_enabled'):
            self.prism_ctx_enabled.set(bool(raw.get('prism_ctx_enabled', True)))
        for attr,key,default in [('prism_version_var','prism_embedded_version','1.40.0'),
                                 ('prism_http_port_var','prism_embedded_port',18079),
                                 ('prism_grpc_port_var','prism_embedded_grpc_port',15060)]:
            if hasattr(self,attr): getattr(self,attr).set(str(raw.get(key,default)))
        for name, field in Settings.model_fields.items():
            if name not in self.agent_settings_vars: continue
            val = raw.get(name, getattr(defaults,name))
            if name == 'humoid_memory_backend': val = 'sqlite'
            if name in {'humoid_memory_db','humoid_rhomgaic_db'}:
                val = getattr(defaults, name)
            self.agent_settings_vars[name].set(str(val) if not isinstance(val,bool) else val)

    def agent_save_settings(self):
        if self.key is None or self.agent_busy or not self.agent_unlocked: return
        data={}
        try:
            for name,var in self.agent_settings_vars.items():
                value=var.get()
                field=Settings.model_fields[name]
                expected=field.annotation
                if expected is bool: value=bool(value)
                elif expected is int: value=int(value)
                elif expected is float: value=float(value)
                else: value=str(value)
                data[name]=value
            data['humoid_memory_backend']='sqlite'
            # AgentVault is sole owner of the live DB paths, never let a user
            # point encrypted agent data to an arbitrary plaintext directory.
            data.pop('humoid_memory_db',None)
            data.pop('humoid_rhomgaic_db',None)
            checked=Settings(_env_file=None, **data)
            # Allow saving even when a remote provider is not configured yet.
            # Health status will report missing credentials when used.
        except Exception as exc:
            messagebox.showerror('Invalid agent settings',str(exc))
            return
        data['use_gui_litert_model'] = bool(self.agent_use_gui_model.get())
        data['gui_memory_depth'] = str(self.agent_gui_ctx_depth.get())
        data['prism_embedded_enabled'] = bool(self.prism_embed_enabled.get())
        data['prism_ctx_enabled'] = bool(self.prism_ctx_enabled.get())
        data['prism_embedded_version'] = self.prism_version_var.get().strip()
        try:
            data['prism_embedded_port'] = int(self.prism_http_port_var.get())
            data['prism_embedded_grpc_port'] = int(self.prism_grpc_port_var.get())
        except (ValueError, TypeError):
            messagebox.showerror('Invalid PRISM ports','Enter numeric HTTP and gRPC ports')
            return
        if not all(1 <= data[k] <= 65535 for k in ('prism_embedded_port','prism_embedded_grpc_port')):
            messagebox.showerror('Invalid PRISM ports','Port numbers must be 1 to 65535')
            return
        self.agent_settings = data
        AgentVault(self.key).write_json('agent_settings.json.aes',data)
        self.agent_task('Settings applied',lambda:self.agent_engine.boot(self.key,data),persist=False)

    def agent_show_profile(self):
        if not self.agent_engine.harness: return
        h=self.agent_engine.harness
        self.agent_activity(f'Provider={h.provider.cfg.name} Model={h.provider.cfg.model} '
            f'Profile={h.profile.key} Protocol={h.protocol.name} '
            f'CTX={h.s.context_limit()} Autonomy={h.s.humoid_autonomy_mode} '
            f'Workspace={h.s.humoid_workspace}')

    def agent_clear_activity(self):
        self.agent_log_items=[]
        self.agent_write(self.agent_activity_box,'Activity cleared.',clear=True)

    def agent_diff_dialog(self, payload):
        # This modal runs on Tk's UI thread, never the asyncio agent loop.
        dialog=ctk.CTkToplevel(self)
        dialog.title('Approve agent edit · ' + payload['path'])
        dialog.geometry('1020x740')
        dialog.transient(self)
        dialog.grab_set()
        ctk.CTkLabel(dialog, text='Review the proposed file change before allowing it',
                     font=ctk.CTkFont(size=19,weight='bold')).pack(pady=8)
        ctk.CTkLabel(dialog,text=payload['path']).pack(pady=3)
        box=ctk.CTkTextbox(dialog,height=360,fg_color=PALETTE['field'],wrap='none')
        box.pack(fill='both',expand=True,padx=12,pady=8)
        box.insert('1.0',payload['diff'][:120000])
        ctk.CTkLabel(dialog,text='Proposed file content (editable before approval)').pack()
        content=ctk.CTkTextbox(dialog,height=170,fg_color=PALETTE['field'])
        content.pack(fill='x',padx=12,pady=6)
        content.insert('1.0',payload['content'])
        ctr=ctk.CTkFrame(dialog,fg_color='transparent'); ctr.pack(pady=8)
        def resolve(allow):
            if allow:
                payload['content']=content.get('1.0','end-1c')
            payload['approved']=allow
            payload['ready'].set()
            dialog.grab_release(); dialog.destroy()
        self.agent_button(ctr,'Approve edited change',lambda:resolve(True))
        self.agent_button(ctr,'Reject change',lambda:resolve(False))
        dialog.protocol('WM_DELETE_WINDOW',lambda:resolve(False))

    def agent_record_gui_exchange(self, prompt: str, answer: str, session_id: int | None) -> None:
        """Keep legacy LiteRT GUI chat in the exact same advanced memory surface."""
        if not self.agent_unlocked or not self.agent_engine.active:
            return
        text = ('Original encrypted GUI chat • session ' + str(session_id or '-')
                + '\nUser: ' + prompt[:6000] + '\nGemma: ' + answer[:6000])
        self.agent_engine.submit(
            'GUI chat indexed',
            lambda: self.agent_engine.add_memory(text, 'gui-chat', 'episodic'),
        )

    def agent_main_chat_context(self, prompt: str, *, max_chars: int = 1600) -> str:
        """Fetch live PRISM/Weaviate recall for the primary GUI Chat path."""
        if (not self.agent_unlocked or not self.agent_engine.active
                or not self.agent_engine.prism_ctx_enabled
                or self.agent_engine.prism is None):
            return ''
        try:
            future = asyncio.run_coroutine_threadsafe(
                self.agent_engine.prism_context(
                    prompt, project='gui-chat', budget=max_chars, strict_scope=True),
                self.agent_engine.loop,
            )
            return str(future.result(timeout=3.0))[:max_chars]
        except Exception:
            # The isolated chat worker still performs encrypted SQLite recall.
            return ''
