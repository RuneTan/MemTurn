import io
import json
import unittest
from urllib.error import HTTPError
from unittest.mock import patch

from memturn.chain_support import BudgetedChat, ExperimentPaused, grouped_tasks
from memturn.models.http import ChatConfig, HttpChat, ModelHTTPError


class ChainTests(unittest.TestCase):
    def test_grouped_plan_has_unique_tasks_and_revisits_conditions(self):
        tasks = list(grouped_tasks(500, 20))
        self.assertEqual(len({t.id for t in tasks}), 500)
        self.assertEqual([(tasks[i].family, tasks[i].version) for i in (0, 20, 40, 60, 80)],
                         [("vault", "v1"), ("api", "v1"), ("vault", "v2"), ("api", "v2"), ("vault", "v1")])

    def test_daily_limit_is_classified_without_auditing_provider_body(self):
        audit = []
        client = HttpChat(ChatConfig("http://unused.invalid/v1", "fixture"), audit.append)
        error = HTTPError("http://unused.invalid", 429, "limited", {"Retry-After": "12"},
                          io.BytesIO(b'{"error":{"message":"tokens per day (TPD) exceeded; secret text"}}'))
        with patch("memturn.models.http.urlopen", side_effect=error):
            with self.assertRaises(ModelHTTPError) as caught:
                client.complete([])
        self.assertTrue(caught.exception.daily)
        self.assertEqual(caught.exception.retry_after, 12)
        self.assertNotIn("secret text", json.dumps(audit))
        self.assertEqual(client.usage["failures"], 1)

    def test_successful_raw_response_is_saved_for_review(self):
        audit = []
        client = HttpChat(ChatConfig("http://unused.invalid/v1", "fixture"), audit.append)
        reply = {"choices": [{"message": {"content": '{"memory":null}'}}],
                 "usage": {"prompt_tokens": 100, "completion_tokens": 20,
                           "prompt_tokens_details": {"cached_tokens": 40}}}
        with patch("memturn.models.http.urlopen", return_value=io.BytesIO(json.dumps(reply).encode())):
            client.complete([])
        self.assertEqual(audit[0]["text"], '{"memory":null}')
        self.assertEqual(client.usage["cached_prompt_tokens"], 40)

    def test_writer_budget_pauses_before_another_network_call(self):
        class Client:
            def __init__(self):
                self.usage = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0}
            def complete(self, *args):
                self.usage.update(requests=1, prompt_tokens=1000, completion_tokens=200)
                return "{}"
        client = BudgetedChat(Client(), lambda event: None, token_budget=5000)
        self.assertEqual(client.complete([]), "{}")
        with self.assertRaises(ExperimentPaused): client.complete([])
        self.assertEqual(client.usage["requests"], 1)

    def test_explicit_resume_gets_new_allowance_and_keeps_cumulative_usage(self):
        class Client:
            usage = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0}
        resumed = BudgetedChat(Client(), lambda event: None, token_budget=5000,
            prior_usage={"requests": 12, "prompt_tokens": 8000, "completion_tokens": 2000})
        self.assertEqual(resumed.charged_tokens, 10000)
        self.assertEqual(resumed.budget_used, 0)
        self.assertTrue(resumed.can_start())
