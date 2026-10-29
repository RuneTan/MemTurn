"""Backend-free checks of the public policy and deadline contract."""
import unittest
from memturn.models.protocol import ChatPolicy, call_deadline, remaining_timeout


class ProtocolTests(unittest.TestCase):
    def test_policy_passes_pinned_version_to_injected_client(self):
        class Client:
            def complete(self, messages, model_override=None):
                self.version = model_override
                self.messages = messages
                return '{"kind":"environment","action":"inspect"}'
        client = Client()
        self.assertEqual(ChatPolicy(client).act('room', (), 'adapter_7'), 'inspect')
        self.assertEqual(client.version, 'adapter_7')
        self.assertIn('room', client.messages[-1]['content'])

    def test_lookup_is_not_an_environment_action(self):
        class Client:
            def complete(self, *args):
                return '{"kind":"memory_search","query":"door"}'
        self.assertEqual(ChatPolicy(Client()).act('room', ()),
                         {'kind': 'memory_search', 'query': 'door'})

    def test_invalid_action_fails_closed(self):
        class Client:
            def complete(self, *args):
                return '{"kind":"environment","action":null}'
        with self.assertRaises(ValueError):
            ChatPolicy(Client()).act('room', ())

    def test_nested_deadline_cannot_extend_outer_and_restores_on_failure(self):
        with self.assertRaises(RuntimeError):
            with call_deadline(1):
                with call_deadline(100):
                    self.assertLessEqual(remaining_timeout(100), 1)
                    raise RuntimeError('failed backend')
        self.assertEqual(remaining_timeout(100), 100)
