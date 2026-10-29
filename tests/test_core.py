from dataclasses import replace
from pathlib import Path
import tempfile
import threading
import time
import unittest

from memturn.config import Config
from memturn.contracts import Artifact, Episode, Memory, Step, Support, Task
from tests.fixtures.operators import ToyConsolidator, ToyWriter
from memturn.memory.policies import cold_region, pack_half, select_one
from memturn.memory.store import BudgetExceeded, StaleSnapshot, Store
from memturn.memory.validation import Validator
from memturn.runtime.service import Runtime
from tests.fixtures.environment import FINISH, PREPARE, ToyEnvironment, ToyPolicy, ToyVerifier
from memturn.tokens import OfflineCounter


def episode(task_id, condition="vault:v1"):
    family, version = condition.split(":")
    env = ToyEnvironment()
    before = env.reset(Task(task_id, family, version))
    prepared = env.step(PREPARE[condition])
    finished = env.step(FINISH[family])
    return Episode(task_id, family, condition,
                   (Step(before, PREPARE[condition], prepared.observation, True),
                    Step(prepared.observation, FINISH[family], finished.observation, True)), 1.0, True)


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config, self.counter = Config(), OfflineCounter()
        self.store = Store(self.root / "state.sqlite", self.config, self.counter)
        self.validator = Validator(self.store, ToyVerifier())
        self.runtime = None

    def tearDown(self):
        if self.runtime:
            self.runtime.close()
        self.store.close()
        self.temp.cleanup()

    def add(self, name, condition="vault:v1"):
        ep = episode(name, condition)
        self.assertTrue(self.store.record_result(ep))
        memory = ToyWriter().write(ep, self.store.batch)
        self.store.append(memory)
        self.store.finish_source(name)
        return memory

    def summaries(self, mode="regular"):
        return ToyConsolidator(self.store).consolidate(self.store.memories(), mode, self.store.batch, 2048)

    def make_runtime(self, learner=None):
        self.runtime = Runtime(self.store, ToyWriter(), ToyConsolidator(self.store), self.validator, learner)
        return self.runtime

    def test_distinct_task_support_not_number_of_records(self):
        raw = self.add("a")
        duplicated = replace(raw, supports=(raw.supports[0], raw.supports[0]))
        self.assertEqual(self.validator.support_count(duplicated), 1)
        self.assertIsNone(select_one((replace(duplicated, consolidated=True),), self.validator,
                                     self.counter, self.config))

    def test_fabricated_provenance_and_target_are_rejected(self):
        raw = self.add("a")
        self.assertFalse(self.validator.valid_memory(replace(raw, supports=(Support("missing", 0),))))
        wrong = replace(raw.examples[0], target="open")
        self.assertFalse(self.validator.trainable(replace(raw, examples=(wrong,))))

    def test_supports_must_come_from_declared_contributing_parents(self):
        first = self.add("a")
        unrelated = self.add("b", "api:v1")
        wrong_parent = replace(first, id="summary", parents=(unrelated.ref,), consolidated=True)
        self.assertTrue(self.validator.valid_memory(wrong_parent))
        self.assertFalse(self.validator.validate_replacement((first, unrelated), (wrong_parent,)))

    def test_selector_prefers_more_verified_tasks_and_returns_one(self):
        for i in range(3): self.add(f"v{i}")
        for i in range(2): self.add(f"a{i}", "api:v1")
        region, summaries = self.store.memories(), self.summaries()
        self.store.replace_region(region, summaries, "regular")
        selected = select_one(self.store.memories(), self.validator, self.counter, self.config)
        self.assertEqual(selected.applicability, "vault:v1")
        self.assertEqual(self.validator.support_count(selected), 3)

    def test_rewriting_does_not_inflate_support_or_reset_age(self):
        self.add("a"); self.add("b")
        outputs = self.summaries()
        self.store.replace_region(self.store.memories(), outputs, "regular")
        original = self.store.memories()[0]
        self.store.replace_region((original,), self.summaries(), "regular")
        rewritten = self.store.memories()[0]
        self.assertEqual(self.validator.support_count(rewritten), 2)
        self.assertEqual(rewritten.recency_anchor, original.recency_anchor)

    def test_budget_failure_rolls_back_memory_write(self):
        raw = self.add("a")
        before = self.store.stats()
        with self.assertRaises(BudgetExceeded):
            self.store.append(replace(raw, id="too_large", content="large " * 13000))
        self.assertEqual(self.store.stats()["active_tokens"], before["active_tokens"])
        self.assertEqual(len(self.store.memories()), 1)

    def test_rejected_write_does_not_freeze_batch_clock(self):
        for i in range(8): self.store.record_result(episode(f"clock{i}"), keep_source=False)
        self.assertEqual(self.store.batch, 2)
        self.assertEqual(self.store.stats()["completed"], 8)

    def test_stale_cold_region_does_not_delete_newly_accessed_memory(self):
        raw = self.add("a")
        outputs = self.summaries("capacity")
        for i in range(8): self.store.record_result(episode(f"clock{i}"), keep_source=False)
        self.store.touch((raw,))
        with self.assertRaises(StaleSnapshot):
            self.store.replace_region((raw,), outputs, "capacity")
        self.assertEqual(self.store.memories()[0].id, raw.id)

    def test_capacity_is_half_budget_and_never_trains(self):
        self.add("a"); self.add("b")
        for i in range(8): self.store.record_result(episode(f"clock{i}"), keep_source=False)
        runtime = self.make_runtime()
        before = self.store.stats()["active_tokens"]
        runtime.capacity_maintenance(incoming_tokens=self.config.memory_token_cap)
        after = self.store.stats()["active_tokens"]
        self.assertGreater(before, after)
        self.assertLessEqual(after, before // 2)
        self.assertEqual(self.store.model_version, "base")
        self.assertTrue(all(m.recency_anchor == 0 for m in self.store.memories()))

    def test_capacity_skips_nonfitting_whole_entry_and_dynamic_k(self):
        class V:
            def valid_memory(self, m): return True
            def support_count(self, m): return len(m.task_ids)
        raw = Memory("raw", "x " * 100, "c", (Support("a", 0),))
        large = Memory("large", "x " * 60, "c", (Support("a", 0), Support("b", 0)))
        small1 = Memory("small1", "x " * 15, "c", (Support("a", 0),))
        small2 = replace(small1, id="small2")
        kept, budget = pack_half((raw,), (large, small1, small2), V(), self.counter, self.config)
        self.assertEqual(len(kept), 2)
        self.assertNotIn(large, kept)
        self.assertLessEqual(sum(self.counter.count(m.payload) for m in kept), budget)

    def test_new_snapshot_excludes_retired_memory_old_task_keeps_its_view(self):
        raw = self.add("a")
        with self.store.session() as view:
            self.store.replace_region((raw,), self.summaries(), "regular")
            self.assertIn(raw, view.memories)
            with self.store.session() as newer:
                self.assertNotIn(raw.ref, {m.ref for m in newer.memories})
        self.assertEqual(self.store.stats()["snapshot_bytes"], 0)

    def test_training_exclusion_blocks_source_aliases(self):
        self.add("a"); self.add("b")
        self.store.replace_region(self.store.memories(), self.summaries(), "regular")
        chosen = self.store.memories()[0]
        self.store.append(replace(chosen, id="same_source_alias"))
        self.assertEqual(self.make_runtime().training_memories(chosen), ())

    def test_publication_only_removes_selected_version(self):
        raw = self.add("a")
        sibling = replace(raw, id="sibling")
        self.store.append(sibling)
        self.store.publish(raw, Artifact("v1", "fixture", "test_only", "base", {}))
        self.assertEqual([m.id for m in self.store.memories()], ["sibling"])
        self.assertIsNotNone(self.store.episode("a"))
        with self.assertRaises(StaleSnapshot):
            self.store.publish(raw, Artifact("v2", "fixture", "test_only", "v1", {}))

    def test_failed_training_retains_consolidation_and_all_memories(self):
        class Failing:
            def train(self, *args): raise RuntimeError("injected training failure")
            def retain(self, versions): pass
        self.add("a"); self.add("b")
        self.add("c", "api:v1"); self.add("d", "api:v1")
        self.make_runtime(Failing()).regular_maintenance()
        self.assertEqual(len(self.store.memories()), 2)
        self.assertTrue(all(m.consolidated for m in self.store.memories()))
        self.assertEqual(self.store.model_version, "base")

    def test_foreground_does_not_wait_for_training_loop(self):
        entered, release = threading.Event(), threading.Event()
        class Slow:
            def train(self, examples, parent):
                entered.set()
                if not release.wait(3): raise RuntimeError("test release timed out")
                raise RuntimeError("end fixture without publication")
            def retain(self, versions): pass
        self.add("a"); self.add("b")
        runtime = self.make_runtime(Slow())
        thread = threading.Thread(target=runtime.regular_maintenance)
        thread.start()
        try:
            self.assertTrue(entered.wait(2))
            finished = threading.Event()
            result = []
            def request():
                result.append(runtime.run_task(Task("live", "api", "v1"), ToyEnvironment(), ToyPolicy()))
                finished.set()
            request_thread = threading.Thread(target=request)
            request_thread.start()
            self.assertTrue(finished.wait(1), "foreground was blocked by background training")
            self.assertEqual(result[0].model_version, "base")
        finally:
            release.set()
            thread.join(4)
            runtime.drain()

    def test_environment_error_is_counted_and_source_queue_drains(self):
        class Broken:
            def reset(self, task): raise RuntimeError("injected simulator failure")
            def close(self): pass
        runtime = self.make_runtime()
        result = runtime.run_task(Task("broken", "vault", "v1"), Broken(), ToyPolicy())
        runtime.drain()
        self.assertEqual(result.status, "execution_error")
        self.assertEqual(self.store.stats()["completed"], 1)

    def test_evaluation_probe_does_not_change_learning_state(self):
        self.add("a")
        before = self.store.stats()
        runtime = self.make_runtime()
        task = Task("probe", "vault", "v1", metadata={"evaluation_only": True})
        result = runtime.run_task(task, ToyEnvironment(), ToyPolicy())
        runtime.drain()
        self.assertTrue(result.success)
        self.assertEqual(self.store.stats(), before)

    def test_reopen_recovers_committed_memory_and_model_pointer(self):
        raw = self.add("a")
        self.store.publish(raw, Artifact("committed", "fixture", "test_only", "base", {}))
        self.store.close()
        self.store = Store(self.root / "state.sqlite", self.config, self.counter)
        self.assertEqual(self.store.model_version, "committed")
        self.assertEqual(self.store.memories(), ())

    def test_deferred_writer_requeue_does_not_count_task_twice(self):
        ep = episode("deferred")
        self.store.record_result(ep)
        self.store.finish_source(ep.task_id)
        self.assertIsNone(self.store.episode(ep.task_id))
        self.store.requeue_source(ep)
        self.assertEqual(self.store.stats()["completed"], 1)
        self.make_runtime().drain()
        self.assertEqual(len(self.store.memories()), 1)
        self.assertEqual(self.store.stats()["completed"], 1)


    def test_restart_requeues_pending_writer_and_cancels_uncommitted_training(self):
        ep = episode("pending")
        self.assertTrue(self.store.record_result(ep))
        self.store.job("unfinished", state="READY", selected="not_published")
        self.store.close()
        self.store = Store(self.root / "state.sqlite", self.config, self.counter)
        self.validator = Validator(self.store, ToyVerifier())
        runtime = self.make_runtime()
        runtime.drain()
        self.assertEqual(len(self.store.memories()), 1)
        self.assertEqual(self.store.pending_episodes(), ())
        self.assertEqual(self.store.stats()["completed"], 1)
        self.assertEqual(self.store.model_version, "base")
        state = self.store.db.execute("SELECT payload FROM jobs WHERE id='unfinished'").fetchone()[0]
        self.assertIn('"CANCELLED"', state)

    def test_restart_after_writer_commit_does_not_rewrite_or_duplicate(self):
        ep = episode("written")
        self.store.record_result(ep)
        original = ToyWriter().write(ep, self.store.batch)
        self.store.append(original, source_task_id=ep.task_id)
        for i in range(4):
            self.store.record_result(episode(f"clock{i}"), keep_source=False)
        class MustNotWrite:
            def write(self, *args): raise AssertionError("committed Writer was replayed")
        self.runtime = Runtime(self.store, MustNotWrite(), ToyConsolidator(self.store), self.validator)
        self.runtime.drain()
        self.assertEqual(self.runtime.errors, [])
        self.assertEqual(self.store.pending_episodes(), ())
        self.assertEqual(len(self.store.memories()), 1)


if __name__ == "__main__": unittest.main()
