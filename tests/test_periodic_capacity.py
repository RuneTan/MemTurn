import tempfile
from pathlib import Path
import unittest
from dataclasses import replace

from memturn.config import Config
from memturn.contracts import Task
from memturn.memory.store import Store
from memturn.memory.validation import Validator
from memturn.runtime.service import Runtime
from memturn.tokens import OfflineCounter
from tests.fixtures.environment import ToyEnvironment, ToyPolicy, ToyVerifier
from tests.fixtures.operators import ToyWriter, ToyConsolidator
from tests.test_core import episode


class ScheduledRuntime(Runtime):
    def regular_maintenance(self, scheduled=False):
        self.audit(('regular', self.store.batch))
        if scheduled:
            self.store.mark_regular()

    def capacity_maintenance(self):
        self.audit(('capacity', self.store.batch))


class PeriodicCapacityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.config = replace(Config(), batch_size=1, capacity_interval_batches=3)
        self.store = Store(Path(self.temp.name)/'state.sqlite', self.config, OfflineCounter())
        self.runtime = None

    def tearDown(self):
        if self.runtime:
            self.runtime.close()
        self.store.close()
        self.temp.cleanup()

    def make_runtime(self, writer=None):
        self.events = []
        self.runtime = ScheduledRuntime(self.store, writer or ToyWriter(), ToyConsolidator(self.store),
            Validator(self.store, ToyVerifier()), audit=self.events.append)
        return self.runtime

    def test_independent_clock_and_order_below_memory_limit(self):
        runtime = self.make_runtime()
        for i in range(6):
            runtime.run_task(Task(str(i), "vault", "v1"), ToyEnvironment(), ToyPolicy())
            runtime.drain()
        calls = [e for e in self.events if isinstance(e, tuple)]
        self.assertEqual(calls, [('regular', 1), ('regular', 2), ('regular', 3), ('capacity', 3),
                                 ('regular', 4), ('regular', 5), ('regular', 6), ('capacity', 6)])
        self.assertLess(self.store.stats()['active_tokens'], self.config.memory_token_cap)

    def test_recovery_and_overdue_windows_coalesce_once(self):
        for i in range(7):
            self.store.record_result(episode(str(i)), keep_source=False)
        self.store.mark_regular()
        self.store.close()
        self.store = Store(Path(self.temp.name)/'state.sqlite', self.config, OfflineCounter())
        runtime = self.make_runtime()
        runtime.drain()
        self.assertEqual([e for e in self.events if isinstance(e, tuple)], [('capacity', 7)])
        runtime._run_regular_if_due()
        self.assertFalse(self.store.capacity_due())
        runtime.close()
        self.runtime = None
        runtime = self.make_runtime()
        runtime.drain()
        self.assertEqual(self.events, [])

    def test_failed_writer_still_advances_clock(self):
        class BrokenWriter:
            def write(self, *args):
                raise RuntimeError('writer unavailable')
        runtime = self.make_runtime(BrokenWriter())
        for i in range(3):
            runtime.run_task(Task(str(i), "vault", "v1"), ToyEnvironment(), ToyPolicy())
            runtime.drain()
        self.assertIn(('capacity', 3), self.events)

    def test_full_bank_does_not_trigger_unscheduled_maintenance(self):
        self.store.close()
        self.config = replace(self.config, memory_token_cap=8)
        self.store = Store(Path(self.temp.name)/'state.sqlite', self.config, OfflineCounter())
        runtime = self.make_runtime()
        runtime.run_task(Task("overflow", "vault", "v1"), ToyEnvironment(), ToyPolicy())
        runtime.drain()
        self.assertEqual(self.store.stats()['refused_writes'], 1)
        self.assertFalse(any(e[0] == 'capacity' for e in self.events if isinstance(e, tuple)))

    def test_interval_validation(self):
        for value in (0, 1, -1, True, 2.5, float('nan')):
            with self.assertRaises(ValueError):
                replace(self.config, capacity_interval_batches=value)

    def test_read_only_runtime_does_not_recover_scheduled_work(self):
        for i in range(3):
            self.store.record_result(episode(str(i)), keep_source=False)
        runtime = Runtime(self.store, None, None, None)
        try:
            runtime.drain()
            self.assertTrue(self.store.capacity_due())
        finally:
            runtime.close()
