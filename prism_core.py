"""PRISM RINGS 2.0: vault-backed rings, SpectraForge, and ChronoOrbit.

SQLite is authoritative. Weaviate receives ONLY opt-in opaque UUIDs,
revisions and locally constructed vectors (which are NOT encrypted and can
leak meaning). The SQLite database itself is staged/sealed by AgentVault.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

CHANNELS = {
    'code': ('#06b6d4', ('function', 'class', 'import', 'pytest', 'traceback', 'src/', '.py', '.js', 'refactor', 'api', 'commit')),
    'writing': ('#a855f7', ('draft', 'paragraph', 'chapter', 'poem', 'tone', 'edit', 'grammar', 'story', 'essay')),
    'tasks': ('#f59e0b', ('task', 'todo', 'deadline', 'schedule', 'milestone', 'priority', 'assign', 'due')),
    'computer': ('#94a3b8', ('click', 'button', 'window', 'screenshot', 'browser', 'scroll', 'desktop', 'mouse')),
    'research': ('#3b82f6', ('source', 'citation', 'paper', 'study', 'research', 'evidence', 'experiment')),
    'procedural': ('#22c55e', ('steps', 'procedure', 'workflow', 'recipe', 'runbook', 'repeatable', 'how to')),
    'dialogue': ('#f43f5e', ('chat', 'message', 'said', 'conversation', 'user', 'assistant')),
    'creative': ('#14b8a6', ('design', 'image', 'concept', 'paint', 'visual', 'creative', 'color')),
    'system': ('#eab308', ('config', 'setting', 'install', 'docker', 'server', 'sqlite', 'weaviate', 'error')),
    'causal': ('#6366f1', ('because', 'failed', 'root cause', 'reason', 'fixed', 'risk', 'result', 'regression')),
}
RINGS = ('Reflex', 'Working', 'Task', 'Episodic', 'Procedural', 'Semantic', 'Deep Vault')
VECTOR_DIM = 256


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def parse_time(value: str) -> datetime:
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except (TypeError, ValueError) as exc:
        raise ValueError('Expected ISO time, e.g. 2026-11-04T15:30-05:00') from exc
    if result.tzinfo is None:
        # GUI users typing local wall time get system local timezone.
        result = result.astimezone()
    return result.astimezone(timezone.utc)


def spectrum(text: str, *, tau: float = 0.7) -> dict[str, float]:
    """Reproducible rule-based multi-channel scores; NO trained classifier claim."""
    if not 0.05 <= tau <= 5:
        raise ValueError('Color temperature must be between 0.05 and 5')
    lower = text.lower()
    scores = {}
    for channel, (_color, keywords) in CHANNELS.items():
        score = 0.15
        for keyword in keywords:
            score += min(3, lower.count(keyword)) * (1.6 if len(keyword) >= 6 else 1.0)
        if channel == 'code' and re.search(r'\bdef \w+\(|\bclass \w+|\{\s*\"\w+\"\s*:', text):
            score += 5
        if channel == 'tasks' and re.search(r'\[[ x]\]|\bdue\s*[:=]', lower):
            score += 4
        scores[channel] = score
    peak = max(scores.values())
    exp = {c: math.exp((v - peak) / tau) for c, v in scores.items()}
    total = sum(exp.values())
    return {c: round(v / total, 8) for c, v in exp.items()}


def dominant_channel(weights: dict[str, float]) -> str:
    return max(weights, key=lambda c: weights[c])


def ring_color(weights: dict[str, float], ring: int) -> str:
    """Blend top channels into an accessible ring hex color; deterministic."""
    if ring not in range(len(RINGS)):
        raise ValueError('Invalid ring')
    rgb = [0.0] * 3
    total = sum(max(0., float(weights.get(c, 0))) for c in CHANNELS)
    if not total:
        return '#64748b'
    for channel, (color, _) in CHANNELS.items():
        frac = max(0., float(weights.get(channel, 0))) / total
        for i in range(3):
            rgb[i] += int(color[1+i*2:3+i*2], 16) * frac
    # Outer rings get a subtle darker appearance, without changing identity.
    factor = 1 - 0.045 * ring
    return '#' + ''.join(f'{min(255,round(v*factor)):02x}' for v in rgb)


def vectorize(text: str, weights: dict[str, float] | None = None, dim: int = VECTOR_DIM) -> list[float]:
    """Deterministic hashed word/character n-gram vector + channel spectral lanes.

    This is a locally generated lexical feature vector, NOT a pretrained
    semantic embedding. It works without online model providers.
    """
    if dim < 64:
        raise ValueError('Vector dimension too small')
    result = [0.0] * dim
    tokens = re.findall(r'[\w.\-/]{2,}', text.lower())[:5000]
    # Use stable hashes, not Python's random process-dependent hash().
    features = tokens + ['~' + t[i:i+3] for t in tokens for i in range(max(0, len(t)-2))]
    for feature in features:
        digest = hashlib.blake2b(feature.encode('utf-8'), digest_size=8).digest()
        slot = int.from_bytes(digest[:4], 'little') % (dim - len(CHANNELS))
        result[slot] += 1.0 if digest[4] & 1 else -1.0
    colors = weights if weights is not None else spectrum(text)
    for i, channel in enumerate(CHANNELS):
        result[dim - len(CHANNELS) + i] = float(colors.get(channel, 0)) * 10
    norm = math.sqrt(sum(v*v for v in result)) or 1.
    return [v / norm for v in result]


def cosim(a: list[float], b: list[float]) -> float:
    return sum(x*y for x, y in zip(a, b))


def retrieval_rings(query: str) -> set[int]:
    """Select rings by intent instead of treating all rings as one bucket."""
    lower = query.lower()
    rings = {1, 3, 4, 5}  # working, episodic, procedural, semantic
    if re.search(r'\b(task|todo|deadline|due|schedule|milestone|checklist|next step|unfinished)\b', lower):
        rings.add(2)
    if re.search(r'\b(current prompt|right now|immediate|this turn|latest observation)\b', lower):
        rings.add(0)
    if re.search(r'\b(deep vault|raw history|archive|full recall|historical snapshot)\b', lower):
        rings.add(6)
    return rings


def _terms(text: str) -> set[str]:
    stop = {'the','and','for','that','with','this','from','have','into','about','your',
            'what','when','where','which','will','would','could','should','please'}
    return {word for word in re.findall(r'[a-z0-9_.\-/]{3,}', text.lower()) if word not in stop}


class PrismStore:
    """Consistent sqlite memory / scheduling / sync operations.

    Each mutating call commits SQLite + outbox as one transaction. The caller
    invokes AgentVault.persist() before asynchronous Weaviate replication.
    """
    def __init__(self, path: Path):
        self.path = Path(path)
        self.lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def db(self):
        conn = sqlite3.connect(self.path, timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA foreign_keys=ON')
        conn.execute('PRAGMA busy_timeout=15000')
        return conn

    def _init(self):
        with self.db() as db:
            db.executescript('''
              CREATE TABLE IF NOT EXISTS prism_meta(k TEXT PRIMARY KEY, v TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS ring_memories (
                id TEXT PRIMARY KEY, text TEXT NOT NULL, summary TEXT NOT NULL DEFAULT '',
                ring INTEGER NOT NULL CHECK(ring BETWEEN 0 AND 6),
                project TEXT NOT NULL DEFAULT 'global', spectrum TEXT NOT NULL,
                color TEXT NOT NULL, channel TEXT NOT NULL, importance REAL NOT NULL DEFAULT 0.5,
                verified INTEGER NOT NULL DEFAULT 0, consent_index INTEGER NOT NULL DEFAULT 0,
                source_id TEXT, revision INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                deleted INTEGER NOT NULL DEFAULT 0
              );
              CREATE INDEX IF NOT EXISTS idx_ring_project ON ring_memories(project,ring,deleted);
              CREATE UNIQUE INDEX IF NOT EXISTS idx_ring_source ON ring_memories(source_id) WHERE source_id IS NOT NULL;
              CREATE TABLE IF NOT EXISTS prism_capsule_sources (
                capsule_id TEXT NOT NULL REFERENCES ring_memories(id) ON DELETE CASCADE,
                source_id TEXT NOT NULL REFERENCES ring_memories(id) ON DELETE CASCADE,
                source_revision INTEGER NOT NULL,
                PRIMARY KEY(capsule_id,source_id)
              );
              CREATE TABLE IF NOT EXISTS prism_outbox (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL, revision INTEGER NOT NULL,
                operation TEXT NOT NULL CHECK(operation IN ('upsert','delete')),
                attempts INTEGER NOT NULL DEFAULT 0, last_error TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(memory_id,revision,operation)
              );
              CREATE TABLE IF NOT EXISTS orbit_tasks (
                id TEXT PRIMARY KEY, title TEXT NOT NULL, instructions TEXT NOT NULL,
                project TEXT NOT NULL DEFAULT 'global', due_at TEXT NOT NULL,
                recurrence TEXT NOT NULL DEFAULT 'none', kind TEXT NOT NULL DEFAULT 'reminder',
                priority INTEGER NOT NULL DEFAULT 5 CHECK(priority BETWEEN 0 AND 10),
                status TEXT NOT NULL DEFAULT 'scheduled', approval_required INTEGER NOT NULL DEFAULT 1,
                approved INTEGER NOT NULL DEFAULT 0, result TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                run_count INTEGER NOT NULL DEFAULT 0, max_runs INTEGER NOT NULL DEFAULT 0,
                lease_until TEXT, last_run_at TEXT
              );
              CREATE INDEX IF NOT EXISTS idx_orbit_due ON orbit_tasks(status,due_at);
              CREATE TABLE IF NOT EXISTS task_edges (
                task_id TEXT NOT NULL REFERENCES orbit_tasks(id) ON DELETE CASCADE,
                depends_on TEXT NOT NULL REFERENCES orbit_tasks(id) ON DELETE CASCADE,
                PRIMARY KEY(task_id,depends_on), CHECK(task_id<>depends_on)
              );
              CREATE TABLE IF NOT EXISTS task_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL, event TEXT NOT NULL, detail TEXT NOT NULL,
                happened_at TEXT NOT NULL
              );
            ''')
            if db.execute("SELECT v FROM prism_meta WHERE k='collection'").fetchone() is None:
                db.execute("INSERT INTO prism_meta(k,v) VALUES('collection',?)", ('Prism'+uuid4().hex[:20],))

    @property
    def collection(self) -> str:
        with self.db() as db:
            return db.execute("SELECT v FROM prism_meta WHERE k='collection'").fetchone()[0]

    def remember(self, text: str, *, ring: int = 3, project: str = 'global',
                 summary: str = '', importance: float = .5, verified: bool = False,
                 consent_index: bool = True, source_id: str | None = None,
                 tau: float = 0.7) -> dict:
        if not 0 <= ring <= 6: raise ValueError('Ring must be R0..R6')
        if not text.strip(): raise ValueError('Cannot store empty memory')
        if len(text) > 100_000: raise ValueError('Memory exceeds 100K characters')
        if not 0 <= importance <= 1: raise ValueError('Importance must be 0..1')
        mid, now = str(uuid4()), utcnow()
        weights = spectrum(text, tau=tau)
        row = (mid, text, summary, ring, project[:200], json.dumps(weights),
               ring_color(weights, ring), dominant_channel(weights), float(importance),
               int(verified), int(consent_index), source_id, now, now)
        with self.lock, self.db() as db:
            db.execute('''INSERT INTO ring_memories(id,text,summary,ring,project,spectrum,color,
             channel,importance,verified,consent_index,source_id,created_at,updated_at)
             VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', row)
            if consent_index:
                db.execute('INSERT INTO prism_outbox(memory_id,revision,operation,created_at) VALUES(?,1,\'upsert\',?)', (mid,now))
        return self.get(mid)

    def import_legacy(self, source_id: str, text: str, *, tier: str = 'episodic',
                      channel: str = 'agent', verified: bool = False) -> bool:
        """Mirror source-linked original TUI memory without creating duplicates."""
        if not text.strip(): return False
        with self.lock, self.db() as db:
            row = db.execute('SELECT id,text,deleted FROM ring_memories WHERE source_id=?',
                             (source_id,)).fetchone()
        if row:
            if row['deleted']:
                return False  # A deliberate PRISM forget must never resurrect it.
            if row['text'] != text:
                self.revise(row['id'], text=text, verified=verified)
                return True
            return False
        ring={'episodic':3,'procedural':4,'semantic':5}.get(tier,3)
        self.remember(text, ring=ring, project=channel or 'agent',
                      verified=verified, source_id=source_id)
        return True

    def forget_source(self, source_id: str) -> bool:
        with self.db() as db:
            row = db.execute('SELECT id,deleted FROM ring_memories WHERE source_id=?',
                             (source_id,)).fetchone()
        if row and not row['deleted']:
            self.forget(row['id'])
            return True
        return False

    def get(self, memory_id: str) -> dict | None:
        with self.db() as db:
            row = db.execute('SELECT * FROM ring_memories WHERE id=?', (memory_id,)).fetchone()
            return dict(row) if row else None

    def revise(self, memory_id: str, **changes) -> dict:
        permitted = {'text','summary','ring','project','importance','verified','consent_index'}
        if set(changes) - permitted: raise ValueError('Invalid memory fields')
        with self.lock, self.db() as db:
            row = db.execute('SELECT * FROM ring_memories WHERE id=? AND deleted=0',(memory_id,)).fetchone()
            if row is None: raise KeyError(memory_id)
            item = dict(row)
            item.update(changes)
            if not 0 <= int(item['ring']) <= 6: raise ValueError('Invalid ring')
            if not 0 <= float(item['importance']) <= 1: raise ValueError('Invalid importance')
            if not str(item['text']).strip(): raise ValueError('Cannot store empty memory')
            weights = spectrum(item['text'])
            item['spectrum'] = json.dumps(weights)
            item['color'] = ring_color(weights, int(item['ring']))
            item['channel'] = dominant_channel(weights)
            item['revision'] = int(row['revision'])+1
            item['updated_at'] = utcnow()
            keys = ('text','summary','ring','project','importance','verified','consent_index',
                    'spectrum','color','channel','revision','updated_at')
            db.execute('UPDATE ring_memories SET '+','.join(f'{k}=?' for k in keys)+' WHERE id=?',
                       (*[item[k] for k in keys],memory_id))
            op = 'upsert' if item['consent_index'] else 'delete'
            if row['consent_index'] or item['consent_index']:
                db.execute('INSERT INTO prism_outbox(memory_id,revision,operation,created_at) VALUES(?,?,?,?)',
                           (memory_id,item['revision'],op,item['updated_at']))
        return self.get(memory_id)

    def forget(self, memory_id: str):
        with self.lock, self.db() as db:
            row=db.execute('SELECT * FROM ring_memories WHERE id=?',(memory_id,)).fetchone()
            if row is None: raise KeyError(memory_id)
            revision=int(row['revision'])+1
            db.execute('UPDATE ring_memories SET deleted=1,text=\'\',summary=\'\',revision=?,updated_at=? WHERE id=?',
                       (revision,utcnow(),memory_id))
            # Attempt removal even if indexing was disabled after a previous run.
            db.execute('INSERT INTO prism_outbox(memory_id,revision,operation,created_at) VALUES(?,?,\'delete\',?)',
                       (memory_id,revision,utcnow()))

    def search(self, query: str, *, limit: int = 12, project: str | None = None,
               candidates: list[str] | None = None, strict_scope: bool = False) -> list[dict]:
        if not query.strip(): return []
        limit=max(1,min(int(limit),100))
        qvec=vectorize(query)
        with self.db() as db:
            if project and not strict_scope:
                rows=db.execute('SELECT * FROM ring_memories WHERE deleted=0 AND project=?',(project,)).fetchall()
            else:
                rows=db.execute('SELECT * FROM ring_memories WHERE deleted=0').fetchall()
        # SQLite fallback: exact lexical features + color vector. External hits
        # raise rank but never bypass SQLite tombstones / consent / access scope.
        candidate_scores={uid:(len(candidates)-i)/max(1,len(candidates)) for i,uid in enumerate(candidates or [])}
        results=[]
        allowed_rings = retrieval_rings(query) if strict_scope else set(range(7))
        query_terms = _terms(query)
        query_colors = spectrum(query)
        query_channels = {name for name, weight in query_colors.items() if weight >= .12}
        for row in rows:
            item=dict(row)
            if strict_scope:
                same_project = bool(project) and item['project'] == project
                shared_durable = item['project'] == 'global' and item['ring'] in {4, 5}
                if not (same_project or shared_durable):
                    continue
                if int(item['ring']) not in allowed_rings:
                    continue
            if candidates is not None and item['id'] not in candidate_scores and len(rows)>800:
                continue
            item_terms = _terms(item['text'])
            lexical_overlap = len(query_terms & item_terms)
            if strict_scope and not lexical_overlap and item['id'] not in candidate_scores:
                continue
            if strict_scope and query_channels:
                item_colors = json.loads(item['spectrum'])
                item_channels = {name for name, weight in item_colors.items() if weight >= .12}
                if item_channels and not (query_channels & item_channels) and lexical_overlap < 2:
                    continue
            score = cosim(qvec,vectorize(item['text'],json.loads(item['spectrum'])))
            score += 0.3*candidate_scores.get(item['id'],0)
            score += .05*item['verified'] + .05*item['importance']
            if project and item['project']==project: score+=.10
            if strict_scope and lexical_overlap:
                score += min(.18, lexical_overlap * .04)
            item['score']=round(score,5)
            results.append(item)
        return sorted(results,key=lambda m:m['score'],reverse=True)[:limit]

    def distill_capsule(self, query: str, *, project: str | None = None,
                        count: int = 6, max_chars: int = 1200) -> dict:
        """Extractive, source-linked HoloRing summary (no fabricated LLM facts)."""
        hits = [m for m in self.search(query, limit=count, project=project)
                if not str(m.get('source_id') or '').startswith('capsule:')]
        if not hits: raise ValueError('No source memories to distill')
        fingerprint = '|'.join(f"{m['id']}:{m['revision']}" for m in sorted(hits,key=lambda m:m['id']))
        digest=hashlib.sha256(fingerprint.encode()).hexdigest()
        source_id='capsule:'+digest
        with self.db() as db:
            found=db.execute('SELECT id FROM ring_memories WHERE source_id=?',(source_id,)).fetchone()
        if found:
            result=self.get(found['id'])
            result['sources']=self.capsule_sources(result['id'])
            return result
        lines=[]
        used=0
        for m in hits:
            # Exact excerpts preserve facts and provenance without paraphrasing.
            body=' '.join(m['text'].split())
            excerpt=body[:min(330,max_chars // 2)]
            line=f"[{m['id']}@v{m['revision']}] {excerpt}"
            if used+len(line)+1 > max_chars:
                continue
            lines.append(line)
            used+=len(line)+1
        if not lines:raise ValueError('No evidence fits capsule budget')
        text='Evidence capsule (extractive; not independently verified):\n'+'\n'.join(lines)
        item=self.remember(text,ring=5,project=project or 'global',
                           summary=text,source_id=source_id,importance=.7)
        with self.lock,self.db() as db:
            for source in hits:
                db.execute('INSERT INTO prism_capsule_sources VALUES(?,?,?)',
                           (item['id'],source['id'],source['revision']))
        item['sources']=self.capsule_sources(item['id'])
        return item

    def capsule_sources(self, capsule_id: str) -> list[dict]:
        with self.db() as db:
            rows=db.execute('SELECT e.source_id,e.source_revision,m.revision,m.deleted '
                'FROM prism_capsule_sources e '
                'JOIN ring_memories m ON e.source_id=m.id '
                'WHERE e.capsule_id=?',(capsule_id,)).fetchall()
            return [{'id':r['source_id'],'referenced_revision':r['source_revision'],
                     'current_revision':r['revision'],
                     'stale':bool(r['deleted'] or r['revision']!=r['source_revision'])} for r in rows]

    def queue_full_reconciliation(self) -> int:
        """Recheck every indexed row, including after interrupted snapshots."""
        now = utcnow()
        with self.lock, self.db() as db:
            for item in db.execute('SELECT id,revision,consent_index,deleted FROM ring_memories').fetchall():
                op = 'upsert' if item['consent_index'] and not item['deleted'] else 'delete'
                db.execute('INSERT OR IGNORE INTO prism_outbox '
                           '(memory_id,revision,operation,created_at) VALUES(?,?,?,?)',
                           (item['id'],item['revision'],op,now))
            return db.execute('SELECT count(*) FROM prism_outbox').fetchone()[0]

    def summary_packet(self, query: str, *, budget: int = 2000,
                       project: str | None = None, candidates: list[str] | None = None,
                       strict_scope: bool = False) -> str:
        """Compact ring-memory CTX packet including verification labels."""
        fragments=[]
        used=0
        for m in self.search(query, limit=12, project=project, candidates=candidates,
                             strict_scope=strict_scope):
            body=(m['summary'] or m['text']).strip()
            confidence = 'verified' if m['verified'] else 'unverified'
            fragment=(f"[R{m['ring']} project={m['project']} {m['channel']} "
                      f"{confidence} {m['id']}] {body}")
            if used + len(fragment) + 1 > budget:
                continue
            fragments.append(fragment)
            used += len(fragment) + 1
        return '\n'.join(fragments)

    def pending_index(self, limit:int=50)->list[dict]:
        with self.db() as db:
            return [dict(r) for r in db.execute('SELECT * FROM prism_outbox ORDER BY id LIMIT ?', (limit,))]

    def index_ack(self, event_id:int):
        with self.lock,self.db() as db:
            db.execute('DELETE FROM prism_outbox WHERE id=?',(event_id,))

    def index_error(self, event_id:int, message:str):
        with self.lock,self.db() as db:
            db.execute('UPDATE prism_outbox SET attempts=attempts+1,last_error=? WHERE id=?',
                       (message[:500],event_id))

    def create_task(self, title:str, instructions:str, due_at:str, *, recurrence:str='none',
                    kind:str='reminder', project:str='global',priority:int=5,
                    dependencies:list[str]|None=None, approval_required:bool=True,
                    max_runs:int=0)->dict:
        if not title.strip(): raise ValueError('Task needs a title')
        if kind not in {'reminder','agent_prompt','memory_review'}: raise ValueError('Unknown task kind')
        if not 0 <= priority <= 10: raise ValueError('Priority 0..10 required')
        if max_runs<0: raise ValueError('max_runs must be >=0')
        self._validate_recurrence(recurrence)
        due=parse_time(due_at).isoformat(timespec='seconds')
        if kind == 'agent_prompt': approval_required=True  # no unattended agent actions
        task_id,now=str(uuid4()),utcnow()
        with self.lock,self.db() as db:
            for dep in set(dependencies or []):
                if not db.execute('SELECT 1 FROM orbit_tasks WHERE id=?',(dep,)).fetchone():
                    raise ValueError('Missing dependency: '+dep)
            db.execute('''INSERT INTO orbit_tasks(id,title,instructions,project,due_at,recurrence,kind,
                priority,approval_required,created_at,updated_at,max_runs)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)''',
                (task_id,title[:500],instructions[:10000],project[:200],due,recurrence,kind,
                 priority,int(approval_required),now,now,max_runs))
            for dep in set(dependencies or []):
                db.execute('INSERT INTO task_edges VALUES (?,?)',(task_id,dep))
            self._event(db,task_id,'created','Task scheduled')
        return self.get_task(task_id)

    def _validate_recurrence(self,recurrence:str):
        if recurrence in ('none','daily','weekly','monthly','weekdays'): return
        m=re.fullmatch(r'interval:(\d+)',recurrence)
        if not m or not 1<=int(m.group(1))<=525600:
            raise ValueError('Recurrence: none, daily, weekly, monthly, weekdays, or interval:MINUTES (1..525600)')

    def _event(self,db,task_id:str,event:str,detail:str):
        db.execute('INSERT INTO task_events(task_id,event,detail,happened_at) VALUES(?,?,?,?)',
                   (task_id,event,detail[:2000],utcnow()))

    def get_task(self, task_id:str)->dict|None:
        with self.db() as db:
            row=db.execute('SELECT * FROM orbit_tasks WHERE id=?',(task_id,)).fetchone()
            if row is None:return None
            task=dict(row)
            task['dependencies']=[r['depends_on'] for r in db.execute('SELECT depends_on FROM task_edges WHERE task_id=?',(task_id,))]
            return task

    def list_tasks(self, *, limit:int=100)->list[dict]:
        with self.db() as db:
            return [self.get_task(row['id']) for row in db.execute('''SELECT id FROM orbit_tasks
                  ORDER BY CASE status WHEN 'scheduled' THEN 0 WHEN 'awaiting_approval' THEN 1 ELSE 2 END,
                  due_at,priority DESC LIMIT ?''',(min(500,max(1,limit)),))]

    def _parents_done(self,db,task_id:str)->bool:
        blockers=db.execute('''SELECT count(*) FROM task_edges e JOIN orbit_tasks p ON e.depends_on=p.id
                     WHERE e.task_id=? AND p.status!='completed' ''',(task_id,)).fetchone()[0]
        return blockers==0

    def priority_score(self, task: dict, now: datetime | None = None) -> float:
        """ChronoOrbit urgency + wait fairness + unblocking value; descriptive."""
        moment=now or datetime.now(timezone.utc)
        deadline=parse_time(task['due_at'])
        hours=(deadline-moment).total_seconds()/3600
        urgency=1/(1+math.exp(max(-40,min(40,hours/12))))
        try:
            age=max(0,(moment-parse_time(task['created_at'])).total_seconds()/86400)
        except Exception: age=0
        with self.db() as db:
            unblock=db.execute('SELECT count(*) FROM task_edges WHERE depends_on=?',
                               (task['id'],)).fetchone()[0]
        return round(.6*task['priority']+2*urgency+.2*min(age,30)+.4*unblock,3)

    def due_tasks(self, now:str|None=None)->list[dict]:
        current=parse_time(now).isoformat(timespec='seconds') if now else utcnow()
        with self.db() as db:
            rows=db.execute('''SELECT id FROM orbit_tasks WHERE status='scheduled' AND due_at<=?
                                ORDER BY priority DESC,due_at LIMIT 50''',(current,)).fetchall()
            ids=[r['id'] for r in rows if self._parents_done(db,r['id'])]
        tasks=[self.get_task(i) for i in ids]
        for task in tasks: task['dynamic_priority']=self.priority_score(task)
        return sorted(tasks,key=lambda t:(-t['dynamic_priority'],t['due_at']))

    def start_task(self,task_id:str, *, explicit_approval:bool=False, now:str|None=None)->dict:
        time=parse_time(now).isoformat(timespec='seconds') if now else utcnow()
        with self.lock,self.db() as db:
            row=db.execute('SELECT * FROM orbit_tasks WHERE id=?',(task_id,)).fetchone()
            if row is None: raise KeyError(task_id)
            if row['status'] not in ('scheduled','awaiting_approval'):
                raise ValueError('Task is not eligible to start')
            if row['due_at']>time: raise ValueError('Task is not due yet')
            if not self._parents_done(db,task_id): raise ValueError('Task dependencies incomplete')
            if row['approval_required'] and not explicit_approval:
                db.execute("UPDATE orbit_tasks SET status='awaiting_approval',updated_at=? WHERE id=?",(time,task_id))
                self._event(db,task_id,'approval_required','Waiting for user to approve this run')
            else:
                db.execute("UPDATE orbit_tasks SET status='running',approved=?,lease_until=?,updated_at=? WHERE id=?",
                           (int(explicit_approval),(parse_time(time)+timedelta(minutes=30)).isoformat(timespec='seconds'),time,task_id))
                self._event(db,task_id,'started','Explicit user approval' if explicit_approval else 'Non-agent notification')
        return self.get_task(task_id)

    def finish_task(self,task_id:str, result:str, *, success:bool=True, now:str|None=None)->dict:
        moment=parse_time(now) if now else datetime.now(timezone.utc)
        with self.lock,self.db() as db:
            row=db.execute('SELECT * FROM orbit_tasks WHERE id=?',(task_id,)).fetchone()
            if row is None or row['status']!='running':raise ValueError('Task not running')
            runs=row['run_count']+1
            recur=row['recurrence']
            repeat=success and recur!='none' and (row['max_runs']==0 or runs<row['max_runs'])
            status='scheduled' if repeat else ('completed' if success else 'failed')
            if repeat:
                due=parse_time(row['due_at'])
                # Skip old occurrences after downtime; monthly recurrence
                # uses calendar dates and weekdays avoid Saturday/Sunday.
                for _ in range(100000):
                    if due > moment: break
                    if recur=='daily': due+=timedelta(days=1)
                    elif recur=='weekly': due+=timedelta(weeks=1)
                    elif recur=='weekdays':
                        due+=timedelta(days=1)
                        while due.weekday()>=5: due+=timedelta(days=1)
                    elif recur=='monthly':
                        from calendar import monthrange
                        year=due.year+(due.month//12)
                        month=due.month%12+1
                        day=min(due.day,monthrange(year,month)[1])
                        due=due.replace(year=year,month=month,day=day)
                    else: due+=timedelta(minutes=int(recur.split(':')[1]))
                else:
                    raise ValueError('Too many missed task recurrences')
                next_due=due.isoformat(timespec='seconds')
            else: next_due=row['due_at']
            db.execute('''UPDATE orbit_tasks SET status=?,result=?,run_count=?,due_at=?,
                     last_run_at=?,lease_until=NULL,approved=0,updated_at=? WHERE id=?''',
                     (status,result[:4000],runs,next_due,moment.isoformat(timespec='seconds'),utcnow(),task_id))
            self._event(db,task_id,status,result)
        return self.get_task(task_id)

    def cancel_task(self,task_id:str)->None:
        with self.lock,self.db() as db:
            row=db.execute('SELECT status FROM orbit_tasks WHERE id=?',(task_id,)).fetchone()
            if row is None:raise KeyError(task_id)
            if row['status']=='running':raise ValueError('Cannot cancel running task')
            db.execute("UPDATE orbit_tasks SET status='cancelled',updated_at=? WHERE id=?",(utcnow(),task_id))
            self._event(db,task_id,'cancelled','Cancelled by user')

    def restore_interrupted(self)->int:
        """After unexpected exit, never rerun ambiguous prior agent actions."""
        with self.lock,self.db() as db:
            rows=db.execute("SELECT id FROM orbit_tasks WHERE status='running'").fetchall()
            for row in rows:
                db.execute("UPDATE orbit_tasks SET status='awaiting_approval',approved=0,lease_until=NULL WHERE id=?",(row['id'],))
                self._event(db,row['id'],'interrupted','Prior run interrupted; requires manual review')
            return len(rows)

    def events(self,task_id:str,limit:int=40)->list[dict]:
        with self.db() as db:
            return [dict(row) for row in db.execute('SELECT * FROM task_events WHERE task_id=? ORDER BY event_id DESC LIMIT ?',
                      (task_id,min(500,max(1,limit))))]

    def overview(self)->dict:
        with self.db() as db:
            rings={int(r['ring']):r['n'] for r in db.execute('SELECT ring,count(*) n FROM ring_memories WHERE deleted=0 GROUP BY ring')}
            tasks={r['status']:r['n'] for r in db.execute('SELECT status,count(*) n FROM orbit_tasks GROUP BY status')}
            outbox=db.execute('SELECT count(*) FROM prism_outbox').fetchone()[0]
            return {'rings':{f'R{i} {name}':rings.get(i,0) for i,name in enumerate(RINGS)},
                    'tasks':tasks,'pending_weaviate_events':outbox,'collection':self.collection}
