"""Model-independent orchestration with separate memory and model commits."""
from dataclasses import asdict, replace
import queue
import threading
import time
import uuid

from ..contracts import Artifact, Episode, ExposedMemory, RetrievalEvent, RetrievalResult, Step
from ..memory.policies import cold_region, pack_half, select_one
from ..memory.retrieval import Retriever
from ..memory.store import BudgetExceeded, StaleSnapshot


class Runtime:
    def __init__(self, store, writer, consolidator, validator, learner=None, audit=None):
        self.store, self.writer, self.consolidator, self.validator = store, writer, consolidator, validator
        self.learner, self.audit = learner, audit
        self.config, self.counter = store.config, store.counter
        self.retriever = Retriever(self.counter, self.config)
        self.queue = queue.Queue(maxsize=self.config.max_pending_jobs)
        self.errors = []
        self.maintenance_lock = threading.RLock()
        self.access_lock = threading.RLock()
        self.candidate_versions = set()
        self.recent_refs = set()
        if learner and store.artifact:
            learner.prepare(Artifact(**store.artifact))
        store.recover_jobs()
        for episode in store.pending_episodes():
            self.queue.put_nowait(episode)
        self.worker = threading.Thread(target=self._worker, name="memturn-maintenance", daemon=True)
        self.worker.start()

    def event(self, kind, **values):
        if self.audit:
            self.audit({"event": kind, "time": time.time(), **values})

    def close(self):
        self.drain()
        self.queue.put(None)
        self.worker.join()
        self.collect_adapters()

    def drain(self):
        self.queue.join()

    def collect_adapters(self):
        if self.learner:
            with self.store.lock:
                live = {self.store.model_version} | {v.model_version for v, _ in self.store.views.values()}
                with self.access_lock:
                    live.update(self.candidate_versions)
                try:
                    self.learner.retain(live)
                except Exception as error:
                    self.event("adapter_retirement_failed", error=type(error).__name__)

    def run_task(self, task, environment, policy):
        started = time.monotonic()
        steps, retrievals, reward, success = [], [], 0.0, False
        def remember(query, memories, remaining, error=None):
            retrievals.append(RetrievalEvent(len(steps), query,
                tuple(ExposedMemory(m.ref, m.payload, tuple(sorted(m.task_ids))) for m in memories),
                remaining, error))
        evaluation = bool(task.metadata.get("evaluation_only", False))
        version, status = self.store.model_version, "failed_or_step_limit"
        try:
            with self.store.session() as view:
                version = view.model_version
                observation = environment.reset(task)
                memory_enabled = task.metadata.get("memory_enabled", True)
                scope_fn = getattr(environment, "retrieval_scope", lambda observation: None)
                scope = scope_fn(observation)
                memories = self.retriever.search(view.memories, observation, applicability=scope) if memory_enabled else ()
                if not evaluation:
                    self.store.touch(memories)
                    with self.access_lock:
                        self.recent_refs.update(m.ref for m in memories)
                searches = 0
                search_limit = self.config.max_extra_searches if memory_enabled else 0
                policy_history = [RetrievalResult(None, memories, search_limit)]
                remember(None, memories, search_limit)
                previous_queries = {}
                for _ in range(task.max_steps):
                    if time.monotonic() - started >= task.metadata.get("timeout_seconds", float("inf")):
                        status = "deadline"
                        break
                    action = policy.act(observation, memories, view.model_version, tuple(policy_history))
                    if isinstance(action, dict) and action.get("kind") == "memory_search":
                        query = action["query"]
                        if searches >= search_limit:
                            error = "search_budget_exhausted"
                            status = "tool_protocol_error"
                        else:
                            searches += 1
                            key = (scope, query.strip().casefold())
                            error = "duplicate_query" if key in previous_queries else None
                            if error:
                                memories = previous_queries[key]
                            else:
                                memories = self.retriever.search(view.memories, query, applicability=scope)
                                previous_queries[key] = memories
                            if not evaluation:
                                self.store.touch(memories)
                                with self.access_lock:
                                    self.recent_refs.update(m.ref for m in memories)
                        policy_history.append(RetrievalResult(query, memories, search_limit - searches, error))
                        remember(query, memories, search_limit - searches, error)
                        self.event("memory_search", task_id=task.id, query=query, error=error,
                                   remaining=search_limit - searches, refs=[m.ref for m in memories])
                        continue
                    transition = environment.step(action)
                    step = Step(observation, action, transition.observation, transition.accepted, transition.evidence)
                    steps.append(step)
                    policy_history.append(step)
                    reward += transition.reward
                    observation = transition.observation
                    scope = scope_fn(observation)
                    if transition.done:
                        success, status = transition.success, "completed"
                        break
        except BudgetExceeded as error:
            status = "rejected"
            self.event("task_rejected", task_id=task.id, reason=str(error))
        except Exception as error:
            status = "execution_error"
            self.event("task_error", task_id=task.id, error=type(error).__name__)
        finally:
            try:
                environment.close()
            except Exception as error:
                self.event("environment_cleanup_failed", task_id=task.id, error=type(error).__name__)
        episode = Episode(task.id, task.family, f"{task.family}:{task.version}", tuple(steps), reward,
                          success, version, time.monotonic() - started, status, tuple(retrievals), asdict(task))
        if evaluation:
            self.event("evaluation_completed", **episode.to_dict())
            return episode
        if self.store.record_result(episode):
            try:
                self.queue.put_nowait(episode)
            except queue.Full:
                self.store.refuse_write()
                self.store.finish_source(episode.task_id)
        elif self.store.regular_due():
            try:
                self.queue.put_nowait("regular")
            except queue.Full:
                self.event("maintenance_request_coalesced")
        self.event("task_completed", **episode.to_dict())
        return episode

    def _worker(self):
        while True:
            episode = self.queue.get()
            try:
                if episode is None:
                    return
                if episode == "regular":
                    if self.store.regular_due():
                        self.regular_maintenance()
                        self.store.mark_regular()
                else:
                    self._write(episode)
            except Exception as error:
                self.errors.append(f"{type(error).__name__}: {error}")
                self.errors = self.errors[-self.config.max_job_records:]
                self.event("background_error", error=type(error).__name__, message=str(error))
            finally:
                try:
                    if isinstance(episode, Episode):
                        self.store.finish_source(episode.task_id)
                except Exception as error:
                    self.event("source_cleanup_failed", error=type(error).__name__)
                finally:
                    self.queue.task_done()

    def _write(self, episode):
        memory = None if self.store.writer_processed(episode.task_id) else self.writer.write(episode, self.store.batch)
        if memory is not None:
            self.event("writer_candidate", task_id=episode.task_id, memory=memory.to_dict())
        if memory is not None and self.validator.valid_memory(memory):
            projected = self.store.stats()["active_tokens"] + self.counter.count(memory.payload)
            if projected >= self.config.memory_token_cap * self.config.capacity_trigger_ratio:
                self.capacity_maintenance(incoming_tokens=self.counter.count(memory.payload))
            try:
                self.store.append(memory, source_task_id=episode.task_id)
                self.event("memory_written", memory_ref=memory.ref)
            except BudgetExceeded:
                self.store.refuse_write()
                self.event("memory_write_refused", task_id=episode.task_id)
        elif memory is not None:
            self.event("memory_candidate_rejected", task_id=episode.task_id,
                       reason="independent_evidence_verification_failed")
        if self.store.regular_due():
            self.regular_maintenance()
            self.store.mark_regular()

    def _propose(self, region, mode):
        outputs = self.consolidator.consolidate(region, mode, self.store.batch, self.config.output_token_cap)
        if not self.validator.validate_replacement(region, outputs):
            raise ValueError("invalid or empty consolidation proposal")
        outputs = tuple(replace(m, examples=()) if m.examples and not self.validator.trainable(m) else m
                        for m in outputs)
        return outputs

    def regular_maintenance(self):
        with self.maintenance_lock:
            job_id = f"regular_{uuid.uuid4().hex}"
            batch = self.store.batch
            memories = self.store.memories()
            with self.access_lock:
                recent = set(self.recent_refs)
                self.recent_refs.clear()
            candidates = [m for m in memories if not m.consolidated or m.ref in recent]
            region, used = [], 0
            for memory in candidates:
                size = self.counter.count(memory.payload)
                if used + size <= self.config.region_token_cap:
                    region.append(memory)
                    used += size
            if region:
                try:
                    outputs = self._propose(tuple(region), "regular")
                    self.store.replace_region(tuple(region), outputs, "regular")
                    self.store.job(job_id, mode="regular", state="MEMORY_COMMITTED", batch=batch)
                    self.event("memory_committed", job_id=job_id, mode="regular", entries=len(outputs))
                except (ValueError, BudgetExceeded, StaleSnapshot) as error:
                    self.store.job(job_id, mode="regular", state="FAILED", reason=str(error))
                    self.event("consolidation_failed", job_id=job_id, reason=str(error))
                    return
            selected = select_one(self.store.memories(), self.validator, self.counter, self.config,
                                  seed=self.config.seed + batch)
            if not selected or not self.learner:
                self.store.job(job_id, mode="regular", state="SKIPPED",
                               reason="no_eligible_candidate" if not selected else "training_backend_disabled")
                return
            self._train(job_id, selected)

    def training_memories(self, selected):
        """Training reads do not refresh real-task recency."""
        return self.retriever.search(self.store.memories(), selected.applicability,
                                     excluded_refs=(selected.ref, *selected.parents),
                                     excluded_tasks=selected.task_ids)

    def _train(self, job_id, selected):
        with self.store.lock:
            live = {self.store.model_version} | {v.model_version for v, _ in self.store.views.values()}
        if len(live) >= self.config.max_adapter_versions:
            self.store.job(job_id, mode="regular", state="SKIPPED", reason="retired_versions_still_in_use")
            return
        artifact = None
        try:
            self.store.job(job_id, mode="regular", state="DATA_VALIDATED", selected=selected.ref,
                           support_tasks=self.validator.support_count(selected))
            examples = self.validator.training_examples(selected)
            if getattr(self.learner, "accepts_memory_context", False):
                allowed = self.training_memories(selected)
                # Fresh static inputs: do not inherit historical retrieval bodies or queries.
                histories = {(e.task_id, e.step): self.store.episode(e.task_id).steps[:e.step]
                             for e in examples if e.example_type == 'decision'}
                self.event("student_context", excluded=selected.ref, memory_refs=[m.ref for m in allowed],
                           example_types=[e.example_type for e in examples])
                artifact = self.learner.train(examples, self.store.model_version,
                                              memory_context=allowed, histories=histories)
            else:
                artifact = self.learner.train(examples, self.store.model_version)
            with self.access_lock:
                self.candidate_versions.add(artifact.version)
            self.learner.prepare(artifact)
            self.store.job(job_id, mode="regular", state="READY", selected=selected.ref,
                           artifact=artifact.__dict__)
            self.store.publish(selected, artifact)
            self.store.job(job_id, mode="regular", state="COMMITTED", selected=selected.ref,
                           artifact=artifact.__dict__)
            self.event("adapter_published", selected=selected.ref, **artifact.__dict__)
        except Exception as error:
            self.store.job(job_id, mode="regular", state="FAILED", selected=selected.ref, reason=str(error))
            self.event("training_failed", selected=selected.ref, reason=str(error))
        finally:
            if artifact:
                with self.access_lock:
                    self.candidate_versions.discard(artifact.version)
            self.collect_adapters()

    def capacity_maintenance(self, incoming_tokens=0):
        with self.maintenance_lock:
            processed = set()
            for _ in range(self.config.max_capacity_regions):
                stats = self.store.stats()
                if stats["active_tokens"] + incoming_tokens <= self.config.memory_token_cap * self.config.capacity_target_ratio:
                    return
                region = cold_region(self.store.memories(), self.store.batch, self.counter,
                                     self.config, excluded=processed)
                if not region:
                    break
                processed.update(m.ref for m in region)
                job_id = f"capacity_{uuid.uuid4().hex}"
                try:
                    candidates = self._propose(region, "capacity")
                    outputs, budget = pack_half(region, candidates, self.validator, self.counter,
                                                self.config, seed=self.config.seed + self.store.batch)
                    if not outputs:
                        raise ValueError("no whole validated memory fits half budget")
                    self.store.replace_region(region, outputs, "capacity")
                    processed.update(m.ref for m in outputs)
                    released = sum(self.counter.count(m.payload) for m in region) - sum(self.counter.count(m.payload) for m in outputs)
                    self.store.job(job_id, mode="capacity", state="MEMORY_COMMITTED",
                                   budget=budget, k=len(outputs), released_tokens=released)
                    self.event("capacity_committed", budget=budget, k=len(outputs), released_tokens=released)
                except (ValueError, BudgetExceeded, StaleSnapshot) as error:
                    self.store.job(job_id, mode="capacity", state="FAILED", reason=str(error))
                    self.event("capacity_failed", reason=str(error))
            if self.store.stats()["active_tokens"] + incoming_tokens > self.config.memory_token_cap * self.config.capacity_target_ratio:
                self.event("capacity_target_unmet")
