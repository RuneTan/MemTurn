"""SQLite commits are short; task snapshots never hold a database transaction."""
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
import json
import hashlib
import sqlite3
import threading
import uuid

from ..contracts import Episode, Memory, Snapshot, dumps


class BudgetExceeded(RuntimeError):
    pass


class StaleSnapshot(RuntimeError):
    pass


class Store:
    def __init__(self, path, config, counter):
        self.path, self.config, self.counter = Path(path), config, counter
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.views = {}
        self.work_reservations = {}
        self.db = sqlite3.connect(str(self.path), check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA wal_autocheckpoint=8")
        self.db.execute('PRAGMA cache_spill=OFF')
        self.database_peak_bytes = 0
        pages = config.database_byte_cap // self.db.execute("PRAGMA page_size").fetchone()[0]
        if pages < 16:
            self.db.close()
            raise ValueError("database_byte_cap is too small for the runtime schema")
        self.db.execute(f"PRAGMA max_page_count={pages}")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS memories(id TEXT PRIMARY KEY,payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS episodes(id TEXT PRIMARY KEY,payload TEXT NOT NULL,pending INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS results(id TEXT PRIMARY KEY,payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS stream_slots(sequence INTEGER PRIMARY KEY,
                task_id TEXT NOT NULL, spec_hash TEXT NOT NULL, state TEXT NOT NULL);
        """)
        defaults = {"version": 0, "completed": 0, "utility": 0.0, "refused_writes": 0,
                    "model_version": "base", "artifact": None, "publication": None,
                    "regular_batch": -1, "capacity_batch": 0, "access_seq": 0, "stream": None, "compute_ledger": {},
                    "growth_initial_tokens":sum(counter.count(m.payload) for m in self._memories()),
                    "growth_admitted_tokens":0, "growth_regular_released_tokens":0,
                    "growth_internalized_tokens":0, "growth_capacity_released_tokens":0}
        for key, value in defaults.items():
            self.db.execute("INSERT OR IGNORE INTO meta VALUES (?,?)", (key, dumps(value)))
        self.db.commit()

    def close(self):
        with self.lock:
            self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self.db.close()

    def _get(self, key):
        return json.loads(self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()[0])

    def _set(self, key, value):
        self.db.execute("UPDATE meta SET value=? WHERE key=?", (dumps(value), key))

    def _memories(self):
        return tuple(Memory.from_dict(json.loads(row[0]))
                     for row in self.db.execute("SELECT payload FROM memories ORDER BY id"))

    def memories(self):
        with self.lock:
            return self._memories()

    @property
    def batch(self):
        with self.lock:
            return self._get("completed") // self.config.batch_size

    @property
    def model_version(self):
        with self.lock:
            return self._get("model_version")

    @property
    def artifact(self):
        with self.lock:
            return self._get("artifact")

    def stats(self):
        with self.lock:
            memories = self._memories()
            source_bytes = sum(len(row[0].encode("utf-8")) for row in self.db.execute("SELECT payload FROM episodes"))
            queued_bytes = sum(len(row[0].encode("utf-8")) for row in self.db.execute(
                "SELECT payload FROM episodes WHERE pending>0"))
            training_bytes = sum(len(dumps([asdict(e) for e in m.examples]).encode("utf-8")) for m in memories)
            metadata_bytes = sum(len(dumps({k: v for k, v in m.to_dict().items()
                                          if k not in ("content", "applicability", "examples")}).encode("utf-8"))
                                 for m in memories)
            metadata_bytes += sum(len(row[0].encode("utf-8")) for table in ("meta", "jobs", "results")
                                  for row in self.db.execute(f"SELECT {'value' if table == 'meta' else 'payload'} FROM {table}"))
            metadata_bytes += sum(len(dumps(row).encode('utf-8')) for row in self.db.execute('SELECT * FROM stream_slots'))
            return {"version": self._get("version"), "model_version": self._get("model_version"),
                    "completed": self._get("completed"), "utility": self._get("utility"),
                    "refused_writes": self._get("refused_writes"), "batch": self.batch,
                    "memories": len(memories), "active_tokens": sum(self.counter.count(m.payload) for m in memories),
                    "source_training_bytes": source_bytes + training_bytes + queued_bytes,
                    "queued_source_bytes": queued_bytes, "metadata_bytes": metadata_bytes,
                    "snapshot_bytes": sum(size for _, size in self.views.values()) + sum(self.work_reservations.values()),
                    "work_reserved_bytes": sum(self.work_reservations.values()),
                    "active_sessions": len(self.views), "token_counter": self.counter.name,
                    "stream": self._get('stream'),
                    "compute_ledger": self._get('compute_ledger'),
                    "stream_slots": self.db.execute('SELECT count(*) FROM stream_slots').fetchone()[0],
                    "growth":{key:self._get('growth_'+key) for key in ('initial_tokens','admitted_tokens',
                        'regular_released_tokens','internalized_tokens','capacity_released_tokens')},
                    "database_bytes": self.path.stat().st_size if self.path.exists() else 0,
                    "database_peak_bytes": self.database_peak_bytes,
                    "database_reserved_bytes": self.config.database_total_byte_cap,
                    "database_auxiliary_bytes": sum(p.stat().st_size for p in
                        (Path(str(self.path)+'-wal'),Path(str(self.path)+'-shm')) if p.exists())}

    def _check(self):
        stats = self.stats()
        for actual, cap in (("active_tokens", "memory_token_cap"),
                            ("source_training_bytes", "source_training_byte_cap"),
                            ("metadata_bytes", "metadata_byte_cap"),
                            ("snapshot_bytes", "snapshot_byte_cap")):
            if stats[actual] > getattr(self.config, cap):
                raise BudgetExceeded(f"{actual} exceeds {cap}")

    @contextmanager
    def transaction(self):
        with self.lock:
            busy, _, _ = self.db.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()
            if busy:
                raise BudgetExceeded('external SQLite reader prevents bounded WAL checkpoint')
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield
                self._check()
                self.db.commit()
            except BaseException:
                self.db.rollback()
                raise
            self.database_peak_bytes = max(self.database_peak_bytes, sum(p.stat().st_size for p in
                (self.path, Path(str(self.path)+'-wal'), Path(str(self.path)+'-shm')) if p.exists()))
            self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def episode(self, task_id):
        with self.lock:
            row = self.db.execute("SELECT payload FROM episodes WHERE id=?", (task_id,)).fetchone()
            return Episode.from_dict(json.loads(row[0])) if row else None

    def pending_episodes(self):
        with self.lock:
            return tuple(Episode.from_dict(json.loads(row[0])) for row in self.db.execute(
                "SELECT payload FROM episodes WHERE pending>0 ORDER BY rowid LIMIT ?",
                (self.config.max_pending_jobs,)))

    def writer_processed(self, task_id):
        with self.lock:
            row = self.db.execute("SELECT pending FROM episodes WHERE id=?", (task_id,)).fetchone()
            return bool(row and row[0] == 2)

    def recover_jobs(self):
        with self.transaction():
            version, artifact = self._get('model_version'), self._get('artifact')
            if version != 'base':
                publication = self._get('publication')
                if publication is None:
                    # Older databases need an actual committed record, never a
                    # READY candidate promoted by inference during migration.
                    records = [json.loads(row[0]) for row in self.db.execute('SELECT payload FROM jobs')]
                    publication = next((r for r in records if r.get('state') == 'COMMITTED'
                                        and r.get('artifact') == artifact), None)
                if (not artifact or artifact.get('version') != version or not publication
                        or publication.get('state') != 'COMMITTED' or publication.get('artifact') != artifact):
                    raise ValueError('current adapter has no matching committed publication record')
                self._set('publication', publication)
            for job_id, payload in self.db.execute("SELECT id,payload FROM jobs").fetchall():
                data = json.loads(payload)
                if data.get("state") not in {"COMMITTED", "MEMORY_COMMITTED", "SKIPPED", "FAILED", "CANCELLED"}:
                    data.update(state="CANCELLED", reason="restart_before_model_commit")
                    self.db.execute("UPDATE jobs SET payload=? WHERE id=?", (dumps(data), job_id))

    def configure_stream(self, stream_id, manifest_sha256):
        if not stream_id or len(stream_id) > 128 or len(manifest_sha256) != 64:
            raise ValueError('invalid stream identity')
        with self.transaction():
            current = self._get('stream')
            if current:
                if current['id'] != stream_id or current['manifest_sha256'] != manifest_sha256:
                    raise ValueError('resume requires the identical task stream and manifest')
            elif self._get('completed'):
                raise ValueError('cannot infer a stream cursor from legacy task counts')
            else:
                self._set('stream', {'id':stream_id, 'manifest_sha256':manifest_sha256, 'high_watermark':-1})

    def recover_stream(self):
        # Only the exclusive run owner may reset interrupted simulator tasks.
        with self.transaction():
            self.db.execute("UPDATE stream_slots SET state='RETRY' WHERE state='RUNNING'")

    def claim_task(self, task, sequence):
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise ValueError('stream sequence must be a nonnegative integer')
        with self.transaction():
            stream = self._get('stream')
            if not stream:
                raise ValueError('configure an immutable manifest before claiming tasks')
            if sequence <= stream['high_watermark']:
                return False
            if sequence > stream['high_watermark'] + self.config.stream_window:
                raise BudgetExceeded('task is beyond the bounded out-of-order stream window')
            fingerprint = hashlib.sha256(dumps(asdict(task)).encode('utf-8')).hexdigest()
            row = self.db.execute('SELECT task_id,spec_hash,state FROM stream_slots WHERE sequence=?', (sequence,)).fetchone()
            if row and row[:2] != (task.id, fingerprint):
                raise ValueError('stream position was reused for a different task')
            if row and row[2] in ('DONE', 'RUNNING'):
                return False
            self.db.execute('INSERT OR REPLACE INTO stream_slots VALUES (?,?,?,?)',
                            (sequence, task.id, fingerprint, 'RUNNING'))
            return True

    def _complete_stream(self, task_id, sequence):
        stream = self._get('stream')
        if not stream:
            raise ValueError('stream is not configured')
        if sequence <= stream['high_watermark']:
            return False
        row = self.db.execute('SELECT task_id,state FROM stream_slots WHERE sequence=?', (sequence,)).fetchone()
        if not row or row[0] != task_id:
            raise ValueError('task result has no matching stream claim')
        if row[1] == 'DONE':
            return False
        self.db.execute("UPDATE stream_slots SET state='DONE' WHERE sequence=?", (sequence,))
        high = stream['high_watermark']
        while self.db.execute("SELECT 1 FROM stream_slots WHERE sequence=? AND state='DONE'", (high + 1,)).fetchone():
            high += 1
        stream['high_watermark'] = high
        self._set('stream', stream)
        self.db.execute('DELETE FROM stream_slots WHERE sequence<=?', (high,))
        return True

    def reserve_compute(self, role, units, limit):
        self.reserve_computes([(role, units, limit)])

    def reserve_computes(self, reservations):
        with self.transaction():
            ledger = self._get('compute_ledger')
            for role, units, limit in reservations:
                if units < 0 or limit <= 0 or len(role) > 64:
                    raise ValueError('invalid compute reservation')
                if role not in ledger and len(ledger) >= 8:
                    raise BudgetExceeded('too many compute ledger roles')
                row = ledger.setdefault(role, {'charged':0, 'confirmed':0, 'calls':0})
                if row['charged'] + units > limit:
                    raise BudgetExceeded('run compute budget exhausted: ' + role)
                # Unknown, failed and interrupted calls retain the full reservation.
                row['charged'] += units
                row['calls'] += 1
            self._set('compute_ledger', ledger)

    def settle_compute(self, role, reserved, actual):
        if not 0 <= actual <= reserved:
            raise ValueError('actual compute exceeded the conservative reservation')
        with self.transaction():
            ledger = self._get('compute_ledger')
            ledger[role]['charged'] -= reserved - actual
            ledger[role]['confirmed'] += actual
            self._set('compute_ledger', ledger)

    def record_result(self, episode, keep_source=True, sequence=None, utility=None):
        """Count completed tasks even when there is no room to save experience."""
        accepted = False
        with self.transaction():
            if sequence is not None and not self._complete_stream(episode.task_id, sequence):
                return False
            if self.db.execute("SELECT 1 FROM results WHERE id=?", (episode.task_id,)).fetchone():
                return False
            self._set("completed", self._get("completed") + 1)
            self._set("utility", self._get("utility") + (episode.reward if utility is None else utility))
            summary = {"task_id": episode.task_id, "reward": episode.reward,
                       "status": episode.status, "model_version": episode.model_version}
            self.db.execute("INSERT INTO results VALUES (?,?)", (episode.task_id, dumps(summary)))
            self.db.execute("DELETE FROM results WHERE rowid NOT IN (SELECT rowid FROM results ORDER BY rowid DESC LIMIT ?)",
                            (self.config.max_job_records,))
            # Completion and its pending Writer source share a crash boundary.
            self.db.execute("SAVEPOINT source_admission")
            try:
                pending = self.db.execute("SELECT count(*) FROM episodes WHERE pending>0").fetchone()[0]
                if not keep_source or pending >= self.config.max_pending_jobs:
                    raise BudgetExceeded("pending writer queue is full")
                self.db.execute("INSERT OR IGNORE INTO episodes VALUES (?,?,1)",
                                (episode.task_id, dumps(episode.to_dict())))
                self._check()
                accepted = True
            except BudgetExceeded:
                self.db.execute("ROLLBACK TO source_admission")
                self._set("refused_writes", self._get("refused_writes") + 1)
            finally:
                self.db.execute("RELEASE source_admission")
        return accepted

    def refuse_write(self):
        with self.transaction():
            self._set("refused_writes", self._get("refused_writes") + 1)

    def requeue_source(self, episode):
        """Retry a known task's deferred Writer without counting another task."""
        with self.transaction():
            if not self.db.execute("SELECT 1 FROM results WHERE id=?", (episode.task_id,)).fetchone():
                raise ValueError("only a recently recorded task can be requeued")
            existing = self.db.execute("SELECT pending FROM episodes WHERE id=?", (episode.task_id,)).fetchone()
            if existing and existing[0] > 0:
                return
            pending = self.db.execute("SELECT count(*) FROM episodes WHERE pending>0").fetchone()[0]
            if pending >= self.config.max_pending_jobs:
                raise BudgetExceeded("pending writer queue is full")
            self.db.execute("INSERT OR REPLACE INTO episodes VALUES (?,?,1)",
                            (episode.task_id, dumps(episode.to_dict())))

    def finish_source(self, task_id):
        with self.transaction():
            self.db.execute("UPDATE episodes SET pending=0 WHERE id=?", (task_id,))
            self._gc_sources()

    def _gc_sources(self):
        needed = set().union(*(m.task_ids for m in self._memories())) if self._memories() else set()
        for view, _ in self.views.values():
            for memory in view.memories:
                needed.update(memory.task_ids)
        for task_id, pending in self.db.execute("SELECT id,pending FROM episodes").fetchall():
            if not pending and task_id not in needed:
                self.db.execute("DELETE FROM episodes WHERE id=?", (task_id,))

    def append(self, memory, source_task_id=None):
        with self.transaction():
            existing = self.db.execute("SELECT payload FROM memories WHERE id=?", (memory.id,)).fetchone()
            if existing:
                if existing[0] != dumps(memory.to_dict()):
                    raise StaleSnapshot("writer ID already has different content")
            else:
                self.db.execute("INSERT INTO memories VALUES (?,?)", (memory.id, dumps(memory.to_dict())))
                self._set('growth_admitted_tokens', self._get('growth_admitted_tokens') + self.counter.count(memory.payload))
                self._set("version", self._get("version") + 1)
            if source_task_id is not None:
                self.db.execute("UPDATE episodes SET pending=2 WHERE id=?", (source_task_id,))

    def replace_region(self, region, outputs, mode, job_id=None, **job_data):
        if not region or not outputs:
            raise ValueError("Do not delete a region without a valid replacement")
        if mode not in ("regular", "capacity") or len({m.ref for m in region}) != len(region):
            raise ValueError("invalid replacement mode or duplicate region")
        with self.transaction():
            current = {m.ref: m for m in self._memories()}
            for old in region:
                if old.ref not in current or (mode == "capacity" and (
                        current[old.ref].recency_anchor != old.recency_anchor or
                        current[old.ref].last_access_seq != old.last_access_seq)):
                    raise StaleSnapshot("region changed or was accessed after its snapshot")
            released = sum(self.counter.count(current[m.ref].payload) for m in region) - sum(
                self.counter.count(m.payload) for m in outputs)
            for item in outputs:
                if not item.parents or not set(item.parents).issubset({m.ref for m in region}):
                    raise ValueError("replacement has invalid contributing parents")
                if item.ref in current:
                    raise ValueError("replacement must have a new ID or revision")
            if mode == "capacity":
                before = sum(self.counter.count(m.payload) for m in region)
                after = sum(self.counter.count(m.payload) for m in outputs)
                if after >= before or after > int(before * self.config.return_token_ratio) or after > self.config.output_token_cap:
                    raise BudgetExceeded("capacity replacement exceeds half budget")
            for old in region:
                self.db.execute("DELETE FROM memories WHERE id=?", (old.id,))
            for item in outputs:
                anchor = max(current[parent].recency_anchor for parent in item.parents)
                access_seq = max(current[parent].last_access_seq for parent in item.parents)
                item = replace(item, recency_anchor=anchor, last_access_seq=access_seq)
                self.db.execute("INSERT INTO memories VALUES (?,?)", (item.id, dumps(item.to_dict())))
            self._set("version", self._get("version") + 1)
            growth_key = 'growth_capacity_released_tokens' if mode == 'capacity' else 'growth_regular_released_tokens'
            self._set(growth_key, self._get(growth_key) + released)
            self._gc_sources()
            if job_id:
                self._job(job_id, mode=mode, state="MEMORY_COMMITTED", memory_committed=True,
                          memory_version=self._get('version'), **job_data)

    @contextmanager
    def reserve_work(self, byte_count):
        """Bound temporary operator/training state alongside leased task views."""
        if byte_count < 0:
            raise ValueError('negative work reservation')
        lease = uuid.uuid4().hex
        with self.lock:
            if self.stats()['snapshot_bytes'] + byte_count > self.config.snapshot_byte_cap:
                raise BudgetExceeded('background working-state budget exceeded')
            self.work_reservations[lease] = byte_count
        try:
            yield
        finally:
            with self.lock:
                self.work_reservations.pop(lease, None)

    @contextmanager
    def session(self, read_only=False):
        with self.lock:
            if len(self.views) >= self.config.max_sessions:
                raise BudgetExceeded("foreground session limit reached")
            memories = self._memories()
            view = Snapshot(uuid.uuid4().hex, self._get("version"), self.batch,
                            self._get("model_version"), memories)
            size = len(dumps([m.to_dict() for m in memories]).encode("utf-8"))
            if size + sum(n for _, n in self.views.values()) + sum(self.work_reservations.values()) > self.config.snapshot_byte_cap:
                raise BudgetExceeded("task snapshot budget reached")
            self.views[view.lease_id] = (view, size)
        try:
            yield view
        finally:
            with self.lock:
                self.views.pop(view.lease_id, None)
            if not read_only:
                with self.transaction():
                    self._gc_sources()

    def touch(self, memories):
        with self.transaction():
            sequence = self._get("access_seq") + 1
            if memories:
                self._set("access_seq", sequence)
            for memory in memories:
                row = self.db.execute("SELECT payload FROM memories WHERE id=?", (memory.id,)).fetchone()
                if row:
                    active = Memory.from_dict(json.loads(row[0]))
                    if active.revision == memory.revision:
                        self.db.execute("UPDATE memories SET payload=? WHERE id=?",
                                        (dumps(replace(active, recency_anchor=self.batch,
                                                       last_access_seq=sequence).to_dict()), active.id))

    def evict_region(self, region, job_id):
        with self.transaction():
            current = {m.ref:m for m in self._memories()}
            for memory in region:
                if (memory.ref not in current or current[memory.ref].last_access_seq != memory.last_access_seq
                        or current[memory.ref].recency_anchor != memory.recency_anchor):
                    raise StaleSnapshot('cold eviction snapshot changed')
            released = sum(self.counter.count(m.payload) for m in region)
            for memory in region:
                self.db.execute('DELETE FROM memories WHERE id=?', (memory.id,))
            self._set('version', self._get('version') + 1)
            self._set('growth_capacity_released_tokens', self._get('growth_capacity_released_tokens') + released)
            self._gc_sources()
            self._job(job_id, mode='capacity', state='MEMORY_COMMITTED', policy='evict', released_tokens=released)
            return released

    def publish(self, selected, artifact, job_id=None, retire=True):
        with self.transaction():
            row = self.db.execute("SELECT payload FROM memories WHERE id=?", (selected.id,)).fetchone()
            if not row or Memory.from_dict(json.loads(row[0])).ref != selected.ref:
                raise StaleSnapshot("selected training version no longer exists")
            if artifact.parent_version != self._get("model_version"):
                raise StaleSnapshot("candidate was trained from an obsolete adapter")
            versions = {v.model_version for v, _ in self.views.values()}
            versions.update((artifact.version, self._get("model_version")))
            if len(versions) > self.config.max_adapter_versions:
                raise BudgetExceeded("too many live adapter versions")
            self._set("model_version", artifact.version)
            self._set("artifact", artifact.__dict__)
            self._set('publication', {'state':'COMMITTED', 'selected':selected.ref,
                                     'artifact':artifact.__dict__, 'job_id':job_id,
                                     'retired':retire,
                                     'memory_version':self._get('version') + 1})
            if retire:
                self._set('growth_internalized_tokens', self._get('growth_internalized_tokens') +
                          self.counter.count(Memory.from_dict(json.loads(row[0])).payload))
                self.db.execute("DELETE FROM memories WHERE id=?", (selected.id,))
            self._set("version", self._get("version") + 1)
            self._gc_sources()
            # The model pointer, single-memory exit and durable publication record
            # are one commit; a crash cannot leave a published READY job.
            self._job(job_id or "publish_" + artifact.version, mode="regular", state="COMMITTED",
                      selected=selected.ref, artifact=artifact.__dict__, retired=retire)

    def _job(self, job_id, **data):
        previous = self.db.execute("SELECT payload FROM jobs WHERE id=?", (job_id,)).fetchone()
        merged = dict(json.loads(previous[0]) if previous else {}, **data)
        self.db.execute("INSERT OR REPLACE INTO jobs VALUES (?,?)", (job_id, dumps(merged)))
        self.db.execute("DELETE FROM jobs WHERE rowid NOT IN (SELECT rowid FROM jobs ORDER BY rowid DESC LIMIT ?)",
                        (self.config.max_job_records,))

    def job(self, job_id, **data):
        with self.transaction():
            self._job(job_id, **data)

    def regular_due(self):
        with self.lock:
            return self.batch > self._get("regular_batch") and self.batch > 0

    def mark_regular(self, batch=None):
        with self.transaction():
            self._set("regular_batch", self.batch if batch is None else batch)

    def capacity_due(self, batch=None):
        with self.lock:
            batch = self.batch if batch is None else batch
            interval = self.config.capacity_interval_batches
            return batch // interval > self._get("capacity_batch") // interval

    def mark_capacity(self, batch):
        with self.transaction():
            self._set("capacity_batch", batch)
