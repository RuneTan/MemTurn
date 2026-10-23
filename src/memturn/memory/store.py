"""SQLite commits are short; task snapshots never hold a database transaction."""
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import json
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
        self.db = sqlite3.connect(str(self.path), check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA wal_autocheckpoint=8")
        pages = max(64, config.database_byte_cap // self.db.execute("PRAGMA page_size").fetchone()[0])
        self.db.execute(f"PRAGMA max_page_count={pages}")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS memories(id TEXT PRIMARY KEY,payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS episodes(id TEXT PRIMARY KEY,payload TEXT NOT NULL,pending INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS results(id TEXT PRIMARY KEY,payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,payload TEXT NOT NULL);
        """)
        defaults = {"version": 0, "completed": 0, "utility": 0.0, "refused_writes": 0,
                    "model_version": "base", "artifact": None, "regular_batch": -1}
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
            training_bytes = sum(len(dumps([e.__dict__ for e in m.examples]).encode("utf-8")) for m in memories)
            metadata_bytes = sum(len(dumps({k: v for k, v in m.to_dict().items()
                                          if k not in ("content", "applicability", "examples")}).encode("utf-8"))
                                 for m in memories)
            metadata_bytes += sum(len(row[0].encode("utf-8")) for table in ("meta", "jobs", "results")
                                  for row in self.db.execute(f"SELECT {'value' if table == 'meta' else 'payload'} FROM {table}"))
            return {"version": self._get("version"), "model_version": self._get("model_version"),
                    "completed": self._get("completed"), "utility": self._get("utility"),
                    "refused_writes": self._get("refused_writes"), "batch": self.batch,
                    "memories": len(memories), "active_tokens": sum(self.counter.count(m.payload) for m in memories),
                    "source_training_bytes": source_bytes + training_bytes, "metadata_bytes": metadata_bytes,
                    "snapshot_bytes": sum(size for _, size in self.views.values()),
                    "active_sessions": len(self.views), "token_counter": self.counter.name,
                    "database_bytes": self.path.stat().st_size if self.path.exists() else 0}

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
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield
                self._check()
                self.db.commit()
            except BaseException:
                self.db.rollback()
                raise
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
            for job_id, payload in self.db.execute("SELECT id,payload FROM jobs").fetchall():
                data = json.loads(payload)
                if data.get("state") not in {"COMMITTED", "MEMORY_COMMITTED", "SKIPPED", "FAILED", "CANCELLED"}:
                    data.update(state="CANCELLED", reason="restart_before_model_commit")
                    self.db.execute("UPDATE jobs SET payload=? WHERE id=?", (dumps(data), job_id))

    def record_result(self, episode, keep_source=True):
        """Count completed tasks even when there is no room to save experience."""
        with self.transaction():
            if self.db.execute("SELECT 1 FROM results WHERE id=?", (episode.task_id,)).fetchone():
                return False
            self._set("completed", self._get("completed") + 1)
            self._set("utility", self._get("utility") + episode.reward)
            summary = {"task_id": episode.task_id, "reward": episode.reward,
                       "status": episode.status, "model_version": episode.model_version}
            self.db.execute("INSERT INTO results VALUES (?,?)", (episode.task_id, dumps(summary)))
            self.db.execute("DELETE FROM results WHERE rowid NOT IN (SELECT rowid FROM results ORDER BY rowid DESC LIMIT ?)",
                            (self.config.max_job_records,))
        if not keep_source:
            self.refuse_write()
            return False
        try:
            with self.transaction():
                pending = self.db.execute("SELECT count(*) FROM episodes WHERE pending>0").fetchone()[0]
                if pending >= self.config.max_pending_jobs:
                    raise BudgetExceeded("pending writer queue is full")
                self.db.execute("INSERT OR IGNORE INTO episodes VALUES (?,?,1)",
                                (episode.task_id, dumps(episode.to_dict())))
            return True
        except (BudgetExceeded, sqlite3.DatabaseError):
            self.refuse_write()
            return False

    def refuse_write(self):
        with self.transaction():
            self._set("refused_writes", self._get("refused_writes") + 1)

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
                self._set("version", self._get("version") + 1)
            if source_task_id is not None:
                self.db.execute("UPDATE episodes SET pending=2 WHERE id=?", (source_task_id,))

    def replace_region(self, region, outputs, mode):
        if not region or not outputs:
            raise ValueError("Do not delete a region without a valid replacement")
        with self.transaction():
            current = {m.ref: m for m in self._memories()}
            for old in region:
                if old.ref not in current or (mode == "capacity" and current[old.ref].recency_anchor != old.recency_anchor):
                    raise StaleSnapshot("region changed or was accessed after its snapshot")
            for item in outputs:
                if not item.parents or not set(item.parents).issubset({m.ref for m in region}):
                    raise ValueError("replacement has invalid contributing parents")
            if mode == "capacity":
                before = sum(self.counter.count(m.payload) for m in region)
                after = sum(self.counter.count(m.payload) for m in outputs)
                if after > int(before * self.config.return_token_ratio) or after > self.config.output_token_cap:
                    raise BudgetExceeded("capacity replacement exceeds half budget")
            for old in region:
                self.db.execute("DELETE FROM memories WHERE id=?", (old.id,))
            for item in outputs:
                anchor = max(current[parent].recency_anchor for parent in item.parents)
                item = replace(item, recency_anchor=anchor)
                self.db.execute("INSERT INTO memories VALUES (?,?)", (item.id, dumps(item.to_dict())))
            self._set("version", self._get("version") + 1)
            self._gc_sources()

    @contextmanager
    def session(self):
        with self.lock:
            if len(self.views) >= self.config.max_sessions:
                raise BudgetExceeded("foreground session limit reached")
            memories = self._memories()
            view = Snapshot(uuid.uuid4().hex, self._get("version"), self.batch,
                            self._get("model_version"), memories)
            size = len(dumps([m.to_dict() for m in memories]).encode("utf-8"))
            if size + sum(n for _, n in self.views.values()) > self.config.snapshot_byte_cap:
                raise BudgetExceeded("task snapshot budget reached")
            self.views[view.lease_id] = (view, size)
        try:
            yield view
        finally:
            with self.lock:
                self.views.pop(view.lease_id, None)
            with self.transaction():
                self._gc_sources()

    def touch(self, memories):
        with self.transaction():
            for memory in memories:
                row = self.db.execute("SELECT payload FROM memories WHERE id=?", (memory.id,)).fetchone()
                if row:
                    active = Memory.from_dict(json.loads(row[0]))
                    if active.revision == memory.revision:
                        self.db.execute("UPDATE memories SET payload=? WHERE id=?",
                                        (dumps(replace(active, recency_anchor=self.batch).to_dict()), active.id))

    def publish(self, selected, artifact):
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
            self.db.execute("DELETE FROM memories WHERE id=?", (selected.id,))
            self._set("version", self._get("version") + 1)
            self._gc_sources()

    def job(self, job_id, **data):
        with self.transaction():
            self.db.execute("INSERT OR REPLACE INTO jobs VALUES (?,?)", (job_id, dumps(data)))
            self.db.execute("DELETE FROM jobs WHERE rowid NOT IN (SELECT rowid FROM jobs ORDER BY rowid DESC LIMIT ?)",
                            (self.config.max_job_records,))

    def regular_due(self):
        with self.lock:
            return self.batch > self._get("regular_batch") and self.batch > 0

    def mark_regular(self):
        with self.transaction():
            self._set("regular_batch", self.batch)
