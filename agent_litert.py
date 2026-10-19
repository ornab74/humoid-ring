"""OpenAI-shaped adapter connecting the *existing encrypted GUI Gemma model*
to the TUI's original AgentHarness without an HTTP shim or plaintext model.

Model inference occurs in an isolated spawned worker so LiteRT native crashes
cannot take down the tkinter app; no tool executor runs in the worker.
"""
from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import queue
import time
from typing import Any, AsyncIterator

from humoid_tui.config import Settings


def _dialogue(messages: list[dict[str, Any]]) -> tuple[str, str]:
    system = []
    dialogue = []
    system_messages = [m for m in messages if m.get('role') == 'system']
    recent_dialogue = [m for m in messages if m.get('role') != 'system'][-44:]
    for item in [*system_messages, *recent_dialogue]:
        role = str(item.get('role') or 'user')
        content = item.get('content') or ''
        if isinstance(content, list):
            content = '\n'.join(str(x.get('text', '')) if isinstance(x, dict) else str(x) for x in content)
        if role == 'system':
            system.append(str(content))
        else:
            calls = item.get('tool_calls') or []
            rendered_calls = '\n'.join(
                str(c.get('function', {}).get('name', '')) + ': ' +
                str(c.get('function', {}).get('arguments', ''))
                for c in calls if isinstance(c, dict)
            )
            dialogue.append(f'<{role}>\n{content}\n{rendered_calls}\n</{role}>')
    return '\n\n'.join(system), '\n\n'.join(dialogue)


def _model_worker(results: Any, key: bytes, messages: list[dict[str, Any]], backend: str,
                  max_num_tokens: int) -> None:
    try:
        from main import unlocked_model_path, temporary_litert_cache, litert_chat_blocking
        system, dialogue = _dialogue(messages)
        with unlocked_model_path(key) as model_path, temporary_litert_cache() as cache:
            result = litert_chat_blocking(
                model_path, dialogue, system_text=system, cache_dir=cache,
                inference_backend=backend,
                max_num_tokens=max_num_tokens,
            )
        results.put(('ok', str(result)))
    except BaseException as exc:
        results.put(('error', f'{type(exc).__name__}: {exc}'))


class VaultGemmaProvider:
    def __init__(self, key: bytes, settings: Settings, *, backend: str = 'Auto'):
        self.key = key
        self.settings = settings
        self.cfg = settings.provider('litert')
        self.backend = backend
        self.active_process: Any | None = None

    async def health(self) -> tuple[bool, str]:
        from pathlib import Path
        ok = Path('models/gemma-4-E2B-it.litertlm.aes').exists()
        return ok, ('Encrypted GUI Gemma model: ready (no shim required)' if ok else
                    'Encrypted GUI Gemma model missing; download from Download Model')

    async def stream(self, messages: list[dict[str, Any]],
                     tools: list[dict[str, Any]] | None = None) -> AsyncIterator[dict[str, Any]]:
        # Gemma's native tool declarations are already embedded in the system
        # prompt by AgentHarness; the isolated worker only performs inference.
        del tools
        ctx = mp.get_context('spawn')
        results = ctx.Queue(maxsize=1)
        process = ctx.Process(target=_model_worker,
                              args=(results, self.key, messages, self.backend,
                                    min(8192, max(4096,
                                        self.settings.humoid_gemma_context_limit))),
                              daemon=True)
        process.start()
        self.active_process = process
        loop = asyncio.get_running_loop()
        try:
            # Poll in a nonblocking executor so Ctrl-C/Cancel from desktop can
            # terminate the spawned native model worker promptly.
            while True:
                if not process.is_alive():
                    try: kind, data = await loop.run_in_executor(None, results.get, True, 0.8)
                    except queue.Empty:
                        raise RuntimeError(
                            f'LiteRT model worker exited (code={process.exitcode}) without a response'
                        )
                    break
                try:
                    kind, data = await loop.run_in_executor(None, results.get, True, 0.2)
                    break
                except queue.Empty:
                    continue
            if kind != 'ok': raise RuntimeError(data)
            yield {'text': data, 'reasoning': '', 'tool_calls': [],
                   'finish_reason': 'stop', 'usage': None}
        finally:
            if process.is_alive():
                process.terminate()
            await asyncio.to_thread(process.join, 3)
            if process.is_alive():
                process.kill()
            process.close()
            results.close()
            self.active_process = None
