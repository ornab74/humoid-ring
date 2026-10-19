"""PRISM Ring Observatory + ChronoOrbit GUI and agent engine integration.

All UI mutations happen on the Tk thread, engine operations are queued on
AgentEngine's single asyncio loop, and memory never bypasses AgentVault.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from prism_core import PrismStore, RINGS, CHANNELS, spectrum, ring_color
from prism_embedded import DEFAULT_WEAVIATE_VERSION, EmbeddedWeaviateVault
from prism_weaviate import WeaviateMirror


class PrismEngineMixin:
    prism: PrismStore | None = None
    prism_embedded: EmbeddedWeaviateVault | None = None
    prism_index: WeaviateMirror | None = None
    prism_state: str = 'locked'
    prism_ctx_enabled: bool = True

    def prism_initialize(self, saved: dict) -> str:
        if self.vault is None or self.vault.prism_path is None:
            raise RuntimeError('Mount the encrypted agent vault first')
        self.prism = PrismStore(self.vault.prism_path)
        self.prism.restore_interrupted()
        self.prism_embedded = None
        self.prism_index = None
        self.prism_ctx_enabled = bool(saved.get('prism_ctx_enabled', True))
        self.prism_state = 'SQLite + encrypted PRISM rings ready'
        if saved.get('prism_embedded_enabled', True):
            try:
                import importlib.util
                if importlib.util.find_spec('weaviate') is None:
                    raise RuntimeError('optional weaviate-client not installed')
                self.prism_embedded = EmbeddedWeaviateVault(
                    self.vault,
                    version=str(saved.get('prism_embedded_version', DEFAULT_WEAVIATE_VERSION)),
                    port=int(saved.get('prism_embedded_port', 18079)),
                    grpc_port=int(saved.get('prism_embedded_grpc_port', 15060)),
                )
                client = self.prism_embedded.start()
                self.prism_index = WeaviateMirror(self.prism, client=client)
                # Catch a previously interrupted process whose SQLite writes
                # survived but whose encrypted Weaviate snapshot was older.
                self.prism.queue_full_reconciliation()
                migration = (' (security-upgraded from an unsafe saved version)'
                             if self.prism_embedded.version_was_upgraded else '')
                self.prism_state = ('SQLite + encrypted embedded Weaviate '
                                    + self.prism_embedded.version + ' running' + migration)
            except Exception as exc:
                # Do not silently write over a failed-to-decrypt vault index.
                self.prism_index = None
                self.prism_state = 'SQLite-only recovery mode: ' + str(exc)
                if self.prism_embedded and self.prism_embedded.client is not None:
                    self.prism_embedded.client.close()
                self.prism_embedded = None
        return self.prism_state

    def prism_shutdown(self) -> None:
        # May raise: shutdown MUST retain the RAM copy and GUI vault key to
        # allow retry, not discard the only copy of a changed index.
        if self.prism_embedded is not None:
            self.prism_embedded.seal()
        if self.vault is not None:
            self.vault.persist()
        self.prism_index = None
        self.prism_embedded = None
        self.prism = None
        self.prism_state = 'locked'

    def prism_store(self) -> PrismStore:
        if not self.prism:
            raise RuntimeError('Unlock the vault to use PRISM')
        return self.prism

    async def prism_bridge_legacy(self, limit: int = 1200):
        if not self.harness:
            return {'imported': 0, 'checked': 0}
        hits = await self.harness.memory.list_memories(limit=min(5000,limit))
        imported = sum(self.prism_store().import_legacy(
            hit.memory_id, hit.text, tier=hit.memory_tier, channel=hit.channel,
            verified=(hit.validation_status == 'verified')) for hit in hits)
        return {'imported': imported, 'checked': len(hits)}

    async def prism_overview(self):
        return {'store': self.prism_store().overview(), 'index': self.prism_state,
                'color_channels': {key: value[0] for key, value in CHANNELS.items()}}

    async def prism_add(self, text, *, ring=3, project='global', summary='',tau=0.7):
        return self.prism_store().remember(text, ring=int(ring), project=project,
                                           summary=summary,tau=float(tau))

    async def prism_distill(self, text, project=None):
        return self.prism_store().distill_capsule(text,project=project)

    async def prism_search(self, text, project=None):
        candidates = None
        if self.prism_index is not None:
            try:
                candidates = self.prism_index.candidates(text)
            except Exception as exc:
                self.prism_state = 'Weaviate query unavailable: ' + str(exc)
        return self.prism_store().search(text, project=project or None,
                                         candidates=candidates)

    async def prism_delete(self, memory_id: str):
        self.prism_store().forget(memory_id)
        return 'Memory tombstoned in SQLite; schedule index sync for vector deletion.'

    async def prism_sync(self):
        if self.prism_index is None:
            return {'index': self.prism_state, 'pending': self.prism_store().overview()['pending_weaviate_events']}
        result = self.prism_index.synchronize(limit=500)
        # SQLite ACKs are durably sealed after the operation via engine.submit.
        return result

    async def prism_context(self, query: str, project: str | None = None, budget=2400,
                            strict_scope: bool = False):
        candidates = None
        if self.prism_index:
            try: candidates = self.prism_index.candidates(query, limit=25)
            except Exception: pass
        return self.prism_store().summary_packet(query, budget=budget,
                                                project=project, candidates=candidates,
                                                strict_scope=strict_scope)

    async def prism_tasks(self):
        return self.prism_store().list_tasks()

    async def prism_create_task(self, **kw):
        return self.prism_store().create_task(**kw)

    async def prism_due_tasks(self):
        return self.prism_store().due_tasks()

    async def prism_cancel_task(self, task_id: str):
        self.prism_store().cancel_task(task_id)
        return 'Task cancelled'

    async def prism_run_task(self, task_id: str):
        store = self.prism_store()
        task = store.start_task(task_id, explicit_approval=True)
        if task['status'] != 'running':
            return task
        # Persist "running" before performing any potentially side-effectful
        # agent operation so interruptions never silently rerun it.
        self.vault.persist()
        try:
            if task['kind'] == 'agent_prompt':
                if not self.harness:
                    raise RuntimeError('Agent harness unavailable')
                scoped = await self.prism_context(
                    task['instructions'], task['project'], strict_scope=True)
                prompt = task['instructions']
                if scoped:
                    prompt += ('\n\n[Untrusted project-scoped PRISM evidence]\n'
                               + scoped + '\n[End PRISM evidence]')
                result = await self.harness.run(prompt, skip_memory_recall=True)
            elif task['kind'] == 'memory_review':
                result = await self.prism_context(
                    task['instructions'], task['project'], strict_scope=True)
            else:
                result = 'Reminder acknowledged by user.'
        except Exception as exc:
            store.finish_task(task_id, str(exc), success=False)
            raise
        return store.finish_task(task_id, str(result), success=True)


class PrismGuiMixin:
    def build_prism_ring_panel(self, page):
        import customtkinter as ctk
        from tkinter import StringVar
        self.agent_title(page, 'PRISM Ring Observatory',
                         'Encrypted SQLite memory + embedded Weaviate vector index; 7 rings, 10 color channels, source-linked context.')
        commands = ctk.CTkFrame(page, fg_color='transparent')
        commands.pack(fill='x', padx=14)
        self.agent_button(commands, 'Ring overview', lambda: self.agent_task('PRISM overview', self.agent_engine.prism_overview))
        self.agent_button(commands, 'Sync Weaviate', lambda: self.agent_task('PRISM sync', self.agent_engine.prism_sync))
        self.agent_button(commands, 'Import original agent memories', lambda: self.agent_task('PRISM bridge', self.agent_engine.prism_bridge_legacy))
        self.prism_index_state = StringVar(value='Encrypted Weaviate pending unlock')
        ctk.CTkLabel(page, textvariable=self.prism_index_state, wraplength=850).pack(anchor='w', padx=17, pady=6)
        self.prism_query = ctk.CTkEntry(page, placeholder_text='Search across hyper-color memory rings...')
        self.prism_query.pack(fill='x', padx=14, pady=5)
        self.prism_project = ctk.CTkEntry(page, placeholder_text='Project scope (blank means all projects)')
        self.prism_project.pack(fill='x', padx=14, pady=5)
        ctr = ctk.CTkFrame(page, fg_color='transparent'); ctr.pack(fill='x',padx=14)
        self.agent_button(ctr, 'Colorized search', self.prism_search_clicked)
        self.agent_button(ctr, 'CTX packet', self.prism_context_clicked)
        self.agent_button(ctr, 'Distill evidence capsule', self.prism_distill_clicked)
        self.prism_result_box = self.agent_box(page, height=210)
        self.prism_result_box.configure(state='disabled')
        ctk.CTkLabel(page,text='New encrypted ring memory (default: indexed only in embedded local Weaviate)').pack(anchor='w',padx=14)
        self.prism_new_text=ctk.CTkTextbox(page,height=85)
        self.prism_new_text.pack(fill='x',padx=14,pady=4)
        ctr2=ctk.CTkFrame(page,fg_color='transparent');ctr2.pack(fill='x',padx=14)
        self.prism_ring=ctk.CTkOptionMenu(ctr2,values=[f'R{i} {name}' for i,name in enumerate(RINGS)])
        self.prism_ring.set('R3 Episodic');self.prism_ring.pack(side='left',padx=6)
        self.prism_temperature=ctk.CTkOptionMenu(ctr2,values=['0.2','0.4','0.7','1.0','1.5','2.0'],width=90)
        self.prism_temperature.set('0.7');self.prism_temperature.pack(side='left',padx=5)
        self.agent_button(ctr2,'Save ring memory',self.prism_save_clicked)
        self.prism_delete_id=ctk.CTkEntry(ctr2,width=320,placeholder_text='Memory UUID to forget')
        self.prism_delete_id.pack(side='left',padx=5)
        self.agent_button(ctr2,'Forget UUID',self.prism_forget_clicked)

    def build_prism_task_panel(self,page):
        import customtkinter as ctk
        from tkinter import StringVar
        self.agent_title(page,'ChronoOrbit Scheduler',
                         'Persistent encrypted task graph. Reminders, memory reviews, and approval-gated agent prompts. No background execution when the GUI is locked.')
        bar=ctk.CTkFrame(page,fg_color='transparent');bar.pack(fill='x',padx=14)
        self.agent_button(bar,'Refresh tasks',lambda:self.agent_task('PRISM tasks',self.agent_engine.prism_tasks))
        self.agent_button(bar,'Due now',lambda:self.agent_task('PRISM due',self.agent_engine.prism_due_tasks))
        self.prism_task_box=self.agent_box(page,height=200)
        self.prism_task_box.configure(state='disabled')
        self.prism_title=ctk.CTkEntry(page,placeholder_text='Task title')
        self.prism_title.pack(fill='x',padx=14,pady=3)
        self.prism_instructions=ctk.CTkTextbox(page,height=75)
        self.prism_instructions.pack(fill='x',padx=14,pady=4)
        opts=ctk.CTkFrame(page,fg_color='transparent');opts.pack(fill='x',padx=14)
        self.prism_due=ctk.CTkEntry(opts,width=230,placeholder_text='YYYY-MM-DDTHH:MM-04:00')
        self.prism_due.pack(side='left',padx=5)
        self.prism_kind=ctk.CTkOptionMenu(opts,values=['reminder','memory_review','agent_prompt'],width=140)
        self.prism_kind.pack(side='left',padx=5)
        self.prism_recur=ctk.CTkOptionMenu(opts,values=['none','daily','weekly','monthly','weekdays','interval:60','interval:1440'],width=135)
        self.prism_recur.pack(side='left',padx=5)
        self.prism_task_priority=ctk.CTkOptionMenu(opts,values=[str(i) for i in range(11)],width=60)
        self.prism_task_priority.set('5');self.prism_task_priority.pack(side='left',padx=5)
        self.agent_button(opts,'Create',self.prism_create_clicked)
        extras=ctk.CTkFrame(page,fg_color='transparent');extras.pack(fill='x',padx=14)
        self.prism_deps=ctk.CTkEntry(extras,width=410,placeholder_text='Dependency UUIDs (comma-separated; optional)')
        self.prism_deps.pack(side='left',padx=5)
        self.prism_max_runs=ctk.CTkEntry(extras,width=130,placeholder_text='Max runs (0 = ∞)')
        self.prism_max_runs.pack(side='left',padx=5)
        edit=ctk.CTkFrame(page,fg_color='transparent');edit.pack(fill='x',padx=14)
        self.prism_task_id=ctk.CTkEntry(edit,width=370,placeholder_text='Task UUID for run/cancel')
        self.prism_task_id.pack(side='left',padx=5)
        self.agent_button(edit,'Approve and run',self.prism_run_clicked)
        self.agent_button(edit,'Cancel task',self.prism_cancel_clicked)
        self.prism_sched_status=StringVar(value='Scheduler idle')
        ctk.CTkLabel(page,textvariable=self.prism_sched_status).pack(anchor='w',padx=14,pady=3)
        self.prism_last_due=()
        self.after(60_000,self.prism_scheduler_tick)

    def prism_scheduler_tick(self):
        # Only alert while the user has the vault unlocked and the GUI is
        # alive. Never launch agent tools or edit files without approval.
        if (getattr(self,'agent_unlocked',False) and not self.agent_busy
                and getattr(self.agent_engine,'prism',None) is not None):
            self.agent_task('PRISM due alerts',self.agent_engine.prism_due_tasks,
                            persist=False)
        self.after(60_000,self.prism_scheduler_tick)

    def prism_search_clicked(self):
        query=self.prism_query.get().strip()
        if query:
            self.agent_task('PRISM search',lambda:self.agent_engine.prism_search(
                query,self.prism_project.get().strip() or None))

    def prism_distill_clicked(self):
        query=self.prism_query.get().strip()
        if query:
            self.agent_task('PRISM capsule',lambda:self.agent_engine.prism_distill(
                query,self.prism_project.get().strip() or None))

    def prism_context_clicked(self):
        query=self.prism_query.get().strip()
        if query:
            self.agent_task('PRISM context',lambda:self.agent_engine.prism_context(
                query,self.prism_project.get().strip() or None))

    def prism_save_clicked(self):
        text=self.prism_new_text.get('1.0','end-1c').strip()
        if text:
            self.agent_task('PRISM saved',lambda:self.agent_engine.prism_add(
                text,ring=int(self.prism_ring.get()[1]),
                project=self.prism_project.get().strip() or 'global',
                tau=float(self.prism_temperature.get())))

    def prism_forget_clicked(self):
        from tkinter import messagebox
        uid=self.prism_delete_id.get().strip()
        if uid and messagebox.askyesno('Forget encrypted memory','Tombstone this memory and remove it from Weaviate on sync?'):
            self.agent_task('PRISM deleted',lambda:self.agent_engine.prism_delete(uid))

    def prism_create_clicked(self):
        from tkinter import messagebox
        title=self.prism_title.get().strip()
        due=self.prism_due.get().strip()
        prompt=self.prism_instructions.get('1.0','end-1c')
        if not title or not due:
            messagebox.showerror('Missing details','Enter task title and due time with timezone.')
            return
        self.agent_task('PRISM task created', lambda: self.agent_engine.prism_create_task(
            title=title,instructions=prompt,due_at=due,
            kind=self.prism_kind.get(),recurrence=self.prism_recur.get(),
            priority=int(self.prism_task_priority.get()),
            dependencies=[x.strip() for x in self.prism_deps.get().split(',') if x.strip()],
            max_runs=int(self.prism_max_runs.get().strip() or 0)))

    def prism_run_clicked(self):
        from tkinter import messagebox
        uid=self.prism_task_id.get().strip()
        if not uid:return
        if not messagebox.askyesno('Approve agent task',
            'Execute this due task now? Agent prompts may call tools and change files (subject to the agent\'s write-approval rules).'):
            return
        self.agent_task('PRISM task executed',lambda:self.agent_engine.prism_run_task(uid))

    def prism_cancel_clicked(self):
        uid=self.prism_task_id.get().strip()
        if uid:self.agent_task('PRISM task cancelled',lambda:self.agent_engine.prism_cancel_task(uid))

    def prism_handle_result(self,label,value):
        if label == 'PRISM overview':
            self.prism_index_state.set(value['index'])
            self.agent_write(self.prism_result_box,json.dumps(value,indent=2,default=str),clear=True)
        elif label == 'PRISM search':
            lines=[]
            for item in value:
                weights=json.loads(item['spectrum'])
                top=sorted(weights.items(),key=lambda a:a[1],reverse=True)[:3]
                lines.append(f"R{item['ring']}  {item['color']}  {item['id']}  {item['project']}\n"
                    + ' / '.join(f'{k} {v:.0%}' for k,v in top)
                    + f"  match {item['score']:.3f}\n{item['summary'] or item['text']}\n")
            self.agent_write(self.prism_result_box,'\n'.join(lines) if lines else 'No matching memories.',clear=True)
        elif label == 'PRISM context':
            self.agent_write(self.prism_result_box,value or 'No recall available.',clear=True)
        elif label in ('PRISM saved','PRISM deleted','PRISM sync','PRISM bridge','PRISM capsule'):
            self.prism_index_state.set(str(value)[:250])
        elif label == 'PRISM due alerts':
            new_ids=tuple(x['id'] for x in value)
            if new_ids and new_ids != self.prism_last_due:
                self.prism_sched_status.set(f'{len(new_ids)} scheduled tasks due; review in ChronoOrbit')
                self.agent_activity(f'[CHRONOORBIT] {len(new_ids)} tasks now due (approval required for agent actions)')
            self.prism_last_due=new_ids
        elif label in ('PRISM tasks','PRISM due','PRISM task created','PRISM task executed','PRISM task cancelled'):
            self.agent_write(self.prism_task_box,json.dumps(value,indent=2,default=str),clear=True)
            self.prism_sched_status.set(label+' completed')
        else:
            return False
        return True
