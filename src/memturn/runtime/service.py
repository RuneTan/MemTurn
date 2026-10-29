"""Model-independent orchestration with separate memory and model commits."""
from dataclasses import asdict, replace
import queue
import threading
import time
import uuid

from ..contracts import Artifact, Episode, ExposedMemory, RetrievalEvent, RetrievalResult, Step, dumps
from ..memory.policies import cold_region, pack_half, select_one
from ..memory.retrieval import Retriever
from ..memory.store import BudgetExceeded, StaleSnapshot
from ..models.protocol import bounded_operation, call_deadline


class Runtime:
    def __init__(self, store, writer, consolidator, validator, learner=None, audit=None, utility=None, learning_budget=None):
        self.store, self.writer, self.consolidator, self.validator = store, writer, consolidator, validator
        self.learner, self.audit = learner, audit
        self.utility, self.learning_budget = utility, learning_budget
        self.config, self.counter = store.config, store.counter
        self.retriever = Retriever(self.counter, self.config)
        self.queue = queue.Queue(maxsize=self.config.max_pending_jobs)
        self.errors = []
        self.maintenance_lock = threading.RLock()
        self.access_lock = threading.RLock()
        self.lifecycle = threading.Condition()
        self.closing, self.closed, self.active_tasks = False, False, 0
        self.candidate_versions = set()
        self.recent_refs = set()
        self.read_only = writer is None and consolidator is None and validator is None
        if not self.read_only:
            store.recover_jobs()
        if learner and store.artifact:
            learner.prepare(Artifact(**store.artifact))
        if not self.read_only and learner and hasattr(learner, 'recover_orphans'):
            with store.lock:
                live = {store.model_version} | {view.model_version for view, _ in store.views.values()}
            for recovery in learner.recover_orphans(live):
                self.event('adapter_recovery', **recovery)
        for episode in (() if self.read_only else store.pending_episodes()):
            self.queue.put_nowait(episode)
        if not self.read_only and self.queue.empty() and store.regular_due():
            self.queue.put_nowait("regular")
        self.worker = threading.Thread(target=self._worker, name="memturn-maintenance", daemon=True)
        self.worker.start()

    def event(self, kind, **values):
        if self.audit:
            try:
                self.audit({"event": kind, "time": time.time(), **values})
            except Exception as error:
                self.errors.append(f"audit_failed: {type(error).__name__}: {error}")
                self.errors = self.errors[-self.config.max_job_records:]

    def close(self):
        with self.lifecycle:
            if self.closing:
                self.lifecycle.wait_for(lambda: self.closed)
                return
            self.closing = True
            self.lifecycle.wait_for(lambda: self.active_tasks == 0)
        try:
            self.drain()
            self.queue.put(None)
            self.worker.join()
            self.collect_adapters()
        finally:
            with self.lifecycle:
                self.closed = True
                self.lifecycle.notify_all()

    def drain(self):
        self.queue.join()

    def collect_adapters(self):
        if self.learner:
            # Serialize collectors with training/publication, not with foreground
            # database access. A new session can only acquire the current version.
            with self.maintenance_lock:
                with self.store.lock:
                    live = {self.store.model_version} | {v.model_version for v, _ in self.store.views.values()}
                    with self.access_lock:
                        live.update(self.candidate_versions)
                try:
                    self.learner.retain(live)
                    if hasattr(self.learner, 'recover_orphans') and not self.read_only:
                        for recovery in self.learner.recover_orphans(live):
                            self.event('adapter_recovery', **recovery)
                except Exception as error:
                    self.event("adapter_retirement_failed", error=type(error).__name__)

    def run_task(self, task, environment, policy, sequence=None, deadline=None):
        with self.lifecycle:
            if self.closing:
                raise RuntimeError('runtime is closing or closed')
            self.active_tasks += 1
        try:
            if sequence is not None and not self.store.claim_task(task, sequence):
                environment.close()
                self.event('task_replay_skipped', task_id=task.id, sequence=sequence)
                return None
            return self._run_task(task, environment, policy, sequence, deadline)
        finally:
            with self.lifecycle:
                self.active_tasks -= 1
                self.lifecycle.notify_all()

    def _run_task(self, task, environment, policy, sequence=None, deadline=None):
        started = time.monotonic()
        steps, retrievals, reward, success = [], [], 0.0, False
        def remember(query, memories, remaining, error=None):
            retrievals.append(RetrievalEvent(len(steps), query,
                tuple(ExposedMemory(m.ref, m.payload, tuple(sorted(m.task_ids))) for m in memories),
                remaining, error))
        evaluation = bool(task.metadata.get("evaluation_only", False))
        if self.read_only and not evaluation:
            raise ValueError('an evaluation-only runtime cannot run learning tasks')
        version, status = self.store.model_version, "failed_or_step_limit"
        deadline = min(deadline if deadline is not None else float('inf'),
                       started + task.metadata.get('timeout_seconds', float('inf')))
        if hasattr(environment, 'set_deadline'):
            environment.set_deadline(deadline)
        try:
            with self.store.session(read_only=evaluation) as view:
                version = view.model_version
                observation = environment.reset(task)
                memory_enabled = self.config.memory_enabled and task.metadata.get("memory_enabled", True)
                scope_fn = getattr(environment, "retrieval_scope", lambda observation: None)
                scope = scope_fn(observation)
                memories = self.retriever.search(view.memories, observation, applicability=scope) if memory_enabled else ()
                if not evaluation:
                    self.store.touch(memories)
                    with self.access_lock:
                        self.recent_refs.update(m.ref for m in memories)
                searches, protocol_errors = 0, 0
                search_limit = self.config.max_extra_searches if memory_enabled else 0
                policy_history = [RetrievalResult(None, memories, search_limit)]
                remember(None, memories, search_limit)
                previous_queries = {}
                # Tool queries consume model calls, but never environment steps.
                for _ in range(task.max_steps + search_limit + 1):
                    if len(steps) >= task.max_steps:
                        break
                    if time.monotonic() - started >= task.metadata.get("timeout_seconds", float("inf")):
                        status = "deadline"
                        break
                    with call_deadline(deadline - time.monotonic()):
                        action = policy.act(observation, memories, view.model_version, tuple(policy_history))
                    if time.monotonic() - started >= task.metadata.get("timeout_seconds", float("inf")):
                        status = "deadline"
                        break
                    if isinstance(action, dict) and action.get("kind") == "memory_search":
                        query = action["query"]
                        if searches >= search_limit:
                            error = "search_budget_exhausted"
                            status = "tool_protocol_error"
                            memories = ()
                            protocol_errors += 1
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
                        if protocol_errors > 1:
                            break
                        continue
                    transition = environment.step(action)
                    step = Step(observation, action, transition.observation, transition.accepted, transition.evidence)
                    steps.append(step)
                    policy_history.append(step)
                    reward += transition.reward
                    observation = transition.observation
                    scope = scope_fn(observation)
                    if time.monotonic() > deadline:
                        status = 'deadline'
                        break
                    if transition.done:
                        success, status = transition.success, "completed"
                        break
        except TimeoutError:
            status = 'deadline'
            self.event('task_deadline', task_id=task.id)
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
        if self.store.record_result(episode, sequence=sequence,
                                    utility=self.utility(episode) if self.utility else None):
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

    def reject_task(self, task, sequence, status='rejected'):
        if not self.store.claim_task(task, sequence):
            return None
        episode = Episode(task.id, task.family, f'{task.family}:{task.version}', (), 0.0, False,
                          self.store.model_version, status=status, task_spec=asdict(task))
        self.store.record_result(episode, keep_source=False, sequence=sequence,
                                 utility=self.utility(episode) if self.utility else None)
        if self.store.regular_due():
            try: self.queue.put_nowait('regular')
            except queue.Full: pass
        self.event('task_completed', **episode.to_dict())
        return episode

    def _worker(self):
        while True:
            episode = self.queue.get()
            try:
                if episode is None:
                    return
                if episode == "regular":
                    self._run_regular_if_due()
                else:
                    self._write(episode)
            except Exception as error:
                if isinstance(episode, Episode):
                    self.store.refuse_write()
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
                    # Failed Writers also advance maintenance. Claim only the
                    # batch visible before this job, never later task arrivals.
                    if isinstance(episode, Episode):
                        try:
                            self._run_regular_if_due()
                        except Exception as error:
                            self.errors.append(f"{type(error).__name__}: {error}")
                            self.errors = self.errors[-self.config.max_job_records:]
                            self.event("background_error", error=type(error).__name__, message=str(error))
                    self.queue.task_done()

    def _run_regular_if_due(self):
        with self.maintenance_lock:
            if self.store.regular_due():
                self.regular_maintenance(scheduled=True)

    @bounded_operation('writer_timeout_seconds')
    def _write(self, episode):
        with call_deadline(self.config.writer_timeout_seconds):
            memory = None if not self.config.writer_enabled or self.store.writer_processed(episode.task_id) else self.writer.write(episode, self.store.batch)
        if memory is not None:
            self.event("writer_candidate", task_id=episode.task_id, memory=memory.to_dict())
        if (memory is not None and memory.task_ids == {episode.task_id}
                and not memory.parents and not memory.consolidated and self.validator.valid_memory(memory)):
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
            self.store.refuse_write()
            self.event("memory_candidate_rejected", task_id=episode.task_id,
                       reason="independent_evidence_verification_failed")

    @bounded_operation('consolidation_timeout_seconds')
    def _propose(self, region, mode):
        sources = [self.store.episode(task_id) for task_id in set().union(*(m.task_ids for m in region))]
        input_bytes = len(dumps([m.to_dict() for m in region]).encode('utf-8')) + sum(
            len(dumps(ep.to_dict()).encode('utf-8')) for ep in sources if ep)
        # A conservative full candidate-data allowance; no hidden unbounded
        # source copies or training output outside the configured work budget.
        with self.store.reserve_work(input_bytes + self.config.source_training_byte_cap):
            with call_deadline(self.config.consolidation_timeout_seconds):
                outputs = self.consolidator.consolidate(region, mode, self.store.batch, self.config.output_token_cap)
            if len(dumps([m.to_dict() for m in outputs]).encode('utf-8')) > self.config.source_training_byte_cap:
                raise BudgetExceeded('consolidation candidate data budget exceeded')
        if not self.validator.validate_replacement(region, outputs):
            raise ValueError("invalid or empty consolidation proposal")
        if sum(self.counter.count(m.payload) for m in outputs) > self.config.output_token_cap:
            raise BudgetExceeded("consolidation body output budget exceeded")
        if sum(len(m.examples) for m in outputs) > self.config.max_examples_per_job:
            # Invalid training material does not invalidate grounded external text.
            outputs = tuple(replace(m, examples=()) for m in outputs)
        outputs = tuple(replace(m, examples=()) if m.examples and not self.validator.trainable(m) else m
                        for m in outputs)
        return outputs

    def regular_maintenance(self, scheduled=False):
        with self.maintenance_lock:
            job_id = f"regular_{uuid.uuid4().hex}"
            with self.store.lock:
                batch = self.store.batch
                memories = self.store.memories()
                if scheduled:
                    self.store.mark_regular(batch)
            with self.access_lock:
                recent = set(self.recent_refs)
                self.recent_refs.intersection_update(m.ref for m in memories)
            candidates = [m for m in memories if not m.consolidated or m.ref in recent]
            region, used = [], 0
            for memory in candidates:
                size = self.counter.count(memory.payload)
                if used + size <= self.config.region_token_cap:
                    region.append(memory)
                    used += size
            if region and self.config.consolidation_enabled:
                try:
                    self.store.job(job_id, mode="regular", state="SNAPSHOT_READY", batch=batch,
                                   region=[m.ref for m in region])
                    outputs = self._propose(tuple(region), "regular")
                    self.store.replace_region(tuple(region), outputs, "regular", job_id=job_id, batch=batch)
                    with self.access_lock:
                        self.recent_refs.difference_update(m.ref for m in region)
                    self.event("memory_committed", job_id=job_id, mode="regular", entries=len(outputs))
                except (ValueError, BudgetExceeded, StaleSnapshot, TimeoutError, RuntimeError) as error:
                    self.store.job(job_id, mode="regular", state="FAILED", reason=str(error))
                    self.event("consolidation_failed", job_id=job_id, reason=str(error))
                    # Previously committed candidates remain eligible even if
                    # this particular new region could not be consolidated.
            if not self.config.internalization_enabled:
                self.store.job(job_id, mode='regular', state='SKIPPED', reason='internalization_disabled')
                return
            pool = self.store.memories()
            if self.config.training_source == 'raw':
                from ..learning.materials import bind_example
                pool = tuple(replace(m, consolidated=True, examples=tuple(
                    bind_example(e, m) for e in m.examples if e.example_type == 'decision'))
                    for m in pool if not m.consolidated)
            with call_deadline(self.config.consolidation_timeout_seconds):
                selected = select_one(pool, self.validator, self.counter, self.config,
                                      seed=self.config.seed + batch)
            if not selected or not self.learner:
                self.store.job(job_id, mode="regular", state="SKIPPED",
                               reason="no_eligible_candidate" if not selected else "training_backend_disabled")
                return
            self._train(job_id, selected)

    def training_memories(self, selected):
        """Training reads do not refresh real-task recency."""
        compact = lambda text: ' '.join(text.casefold().split())
        allowed = tuple(m for m in self.store.memories()
                        if compact(m.content) != compact(selected.content))
        return self.retriever.search(allowed, selected.applicability,
                                     excluded_refs=(selected.ref, *selected.parents),
                                     excluded_tasks=selected.task_ids)

    def training_histories(self, selected, examples):
        """Keep real states intact; reject contaminated inputs instead of inventing states."""
        compact = lambda text: ' '.join(text.casefold().split())
        forbidden = {compact(selected.content)}
        for memory in self.store.memories():
            if memory.ref in selected.parents or memory.task_ids & selected.task_ids:
                forbidden.add(compact(memory.content))
        for task_id in selected.task_ids:
            episode = self.store.episode(task_id)
            if episode:
                for event in episode.retrievals:
                    for exposed in event.memories:
                        if (exposed.ref in (selected.ref, *selected.parents)
                                or set(exposed.task_ids) & selected.task_ids):
                            forbidden.add(compact(exposed.payload))
                            forbidden.add(compact(exposed.payload.partition('\n')[2]))
        forbidden.discard('')
        histories = {}
        for example in examples:
            history = ()
            if example.example_type == 'decision':
                episode = self.store.episode(example.task_id)
                if episode is None:
                    raise ValueError('training source is no longer available')
                history = episode.steps[:example.step]
            inputs = [example.prompt, *(s.observation for s in history), *(s.action for s in history)]
            if any(fragment in compact(text) for text in inputs for fragment in forbidden):
                raise ValueError('training input repeats excluded memory; retain external memory')
            if example.example_type == 'decision':
                histories[(example.task_id, example.step)] = history
        return histories

    def _train(self, job_id, selected):
        with self.store.lock:
            live = {self.store.model_version} | {v.model_version for v, _ in self.store.views.values()}
        if live - {self.store.model_version}:
            self.store.job(job_id, mode="regular", state="SKIPPED", reason="retired_versions_still_in_use")
            return
        if hasattr(self.learner, 'has_artifact_capacity') and not self.learner.has_artifact_capacity(self.config.max_adapter_versions):
            self.store.job(job_id, mode='regular', state='SKIPPED', reason='artifact_directory_limit')
            self.event('training_skipped', reason='artifact_directory_limit; existing checkpoints preserved')
            return
        artifact = None
        try:
            self.store.job(job_id, mode="regular", state="DATA_VALIDATED", selected=selected.ref,
                           support_tasks=self.validator.support_count(selected))
            examples = self.validator.training_examples(selected)
            histories = self.training_histories(selected, examples)
            if self.learning_budget:
                reserved_tokens = self.learner.config.max_training_tokens
                reserved_updates = len(examples) * self.learner.config.epochs
                self.store.reserve_computes([('sft_tokens', reserved_tokens, self.learning_budget['tokens']),
                                             ('sft_updates', reserved_updates, self.learning_budget['updates'])])
            if getattr(self.learner, "accepts_memory_context", False):
                allowed = self.training_memories(selected)
                # Fresh static inputs: do not inherit historical retrieval bodies or queries.
                self.event("student_context", excluded=selected.ref, memory_refs=[m.ref for m in allowed],
                           example_types=[e.example_type for e in examples])
                payload_bytes = len(dumps({'examples':[asdict(e) for e in examples],
                    'memories':[m.to_dict() for m in allowed],
                    'histories':[[asdict(s) for s in history] for history in histories.values()]}).encode('utf-8'))
                token_limit = getattr(getattr(self.learner, 'config', None), 'max_training_tokens', 16384)
                with self.store.reserve_work(3 * payload_bytes + 16 * token_limit):
                    artifact = self.learner.train(examples, self.store.model_version,
                                                  memory_context=allowed, histories=histories)
            else:
                artifact = self.learner.train(examples, self.store.model_version)
            if self.learning_budget:
                self.store.settle_compute('sft_tokens', reserved_tokens, artifact.metrics['processed_training_tokens'])
                self.store.settle_compute('sft_updates', reserved_updates, artifact.metrics['steps'])
            with self.access_lock:
                self.candidate_versions.add(artifact.version)
            self.learner.prepare(artifact)
            self.store.job(job_id, mode="regular", state="READY", selected=selected.ref,
                           artifact=artifact.__dict__)
            self.store.publish(selected, artifact, job_id=job_id, retire=self.config.exit_after_training)
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
                    if self.config.capacity_policy == 'evict':
                        released = self.store.evict_region(region, job_id)
                        self.event('capacity_committed', policy='evict', k=0, released_tokens=released)
                        continue
                    self.store.job(job_id, mode="capacity", state="SNAPSHOT_READY", batch=self.store.batch,
                                   region=[m.ref for m in region],
                                   recency=[(m.ref, m.recency_anchor, m.last_access_seq) for m in region])
                    candidates = self._propose(region, "capacity")
                    outputs, budget = pack_half(region, candidates, self.validator, self.counter,
                                                self.config, seed=self.config.seed + self.store.batch)
                    if not outputs:
                        raise ValueError("no whole validated memory fits half budget")
                    released = sum(self.counter.count(m.payload) for m in region) - sum(self.counter.count(m.payload) for m in outputs)
                    self.store.replace_region(region, outputs, "capacity", job_id=job_id,
                                              budget=budget, k=len(outputs), released_tokens=released)
                    processed.update(m.ref for m in outputs)
                    self.event("capacity_committed", budget=budget, k=len(outputs), released_tokens=released)
                except (ValueError, BudgetExceeded, StaleSnapshot, TimeoutError, RuntimeError) as error:
                    self.store.job(job_id, mode="capacity", state="FAILED", reason=str(error))
                    self.event("capacity_failed", reason=str(error))
            if self.store.stats()["active_tokens"] + incoming_tokens > self.config.memory_token_cap * self.config.capacity_target_ratio:
                self.event("capacity_target_unmet")
