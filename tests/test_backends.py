import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from memturn.contracts import Episode, Example, Step, Task
from memturn.experiments import inspect_profile, run_profile
from memturn.learning.huggingface import supervised_tokens
from memturn.models.http import ChatConfig, ChatPolicy, HttpChat, environment_action, parse_json, task_messages
from memturn.models.agents import ChatWriter, decode_memory
from memturn.tasks.alfworld import ALFWorldEnvironment
from memturn.tasks.scienceworld import ScienceWorldEnvironment


class BackendTests(unittest.TestCase):
    def test_writer_prompt_explicitly_requests_json_for_provider_json_mode(self):
        class Client:
            def complete(self, messages):
                self.messages = messages
                return '{"memory":null}'
        client = Client()
        self.assertIsNone(ChatWriter(client).write(Episode("a", "vault", "vault:v1", (), 0, False), 0))
        self.assertIn("json", client.messages[0]["content"].lower())

    def test_model_sample_references_copy_exact_accepted_source(self):
        source = Episode("a", "vault", "vault:v1",
                         (Step("verbatim observation", "inspect_lock", "accepted", True),
                          Step("another observation", "wrong", "rejected", False)), 1, True)
        value = {"content": "inspect_lock", "applicability": "vault:v1",
                 "supports": [{"task_id": "a", "step": 0}],
                 "examples": [{"task_id": "a", "step": 0}]}
        memory = decode_memory(value, 0, source_lookup=lambda _: source)
        self.assertEqual(memory.examples, (Example("a", 0, "verbatim observation", "inspect_lock"),))
        value["examples"][0]["step"] = 1
        with self.assertRaisesRegex(ValueError, "unaccepted"):
            decode_memory(value, 0, source_lookup=lambda _: source)

    def test_remote_template_reports_missing_setup_without_network(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "scienceworld.example.json"
        profile = json.loads(path.read_text(encoding="utf-8"))
        with patch.dict("os.environ", {}, clear=True):
            missing = inspect_profile(profile)
        self.assertIn("training.target_modules", missing)
        self.assertIn("verification_factory", missing)
        self.assertTrue(any("MEMTURN_TASK_BASE_URL" in item for item in missing))
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "must_not_launch"
            with self.assertRaisesRegex(ValueError, "Complete remote setup first"):
                run_profile(path, output)
            self.assertFalse(output.exists())

    def test_http_schema_role_model_and_usage_without_network(self):
        client = HttpChat(ChatConfig("http://unused.invalid/v1", "base-model"))
        reply = {"choices": [{"message": {"content": '{"kind":"environment","action":"look"}'}}],
                 "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        with patch("memturn.models.http.urlopen", return_value=io.BytesIO(json.dumps(reply).encode())) as request:
            action = ChatPolicy(client).act("room", (), "adapter_v2")
        self.assertEqual(action, "look")
        payload = json.loads(request.call_args.args[0].data)
        self.assertEqual(request.call_args.args[0].get_header("User-agent"), "MemTurn/0.1")
        self.assertEqual(payload["model"], "adapter_v2")
        self.assertNotIn("chat_template_kwargs", payload)
        self.assertEqual(client.usage["prompt_tokens"], 10)

    def test_qwen_options_and_json_mode_are_explicit_provider_settings(self):
        client = HttpChat(ChatConfig("http://unused.invalid/v1", "qwen",
            chat_template_kwargs={"enable_thinking": False}, response_format={"type": "json_object"}))
        reply = {"choices": [{"message": {"content": "{}"}}]}
        with patch("memturn.models.http.urlopen", return_value=io.BytesIO(json.dumps(reply).encode())) as request:
            client.complete([{"role": "user", "content": "JSON please"}])
        payload = json.loads(request.call_args.args[0].data)
        self.assertFalse(payload["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(payload["response_format"], {"type": "json_object"})

    def test_json_parser_rejects_malformed_output(self):
        self.assertEqual(parse_json('<think>hidden</think>```json\n{"a":1}\n```'), {"a": 1})
        with self.assertRaises((ValueError, json.JSONDecodeError)):
            parse_json('```json\n{"a":1}')

    def test_sft_masks_prompt_and_keeps_assistant_target(self):
        class Tokenizer:
            def apply_chat_template(self, messages, add_generation_prompt, **kwargs):
                return "".join(f"<{m['role']}>{m['content']}" for m in messages) + ("<assistant>" if add_generation_prompt else "!")
            def encode(self, text, **kwargs): return list(text.encode())
        ids, labels = supervised_tokens(Tokenizer(), "public observation", "inspect_lock", 2000)
        self.assertEqual(len(ids), len(labels))
        first_target = next(i for i, x in enumerate(labels) if x != -100)
        self.assertTrue(all(x == -100 for x in labels[:first_target]))
        self.assertEqual(bytes(labels[first_target:]).decode(), environment_action("inspect_lock") + "!")
        expected = Tokenizer().apply_chat_template(task_messages("public observation"), add_generation_prompt=True)
        self.assertEqual(bytes(ids[:first_target]).decode(), expected)
        with self.assertRaises(ValueError):
            supervised_tokens(Tokenizer(), "x", "long target", 5)

    def test_unstable_template_is_not_silently_mislabeled(self):
        class Bad:
            def apply_chat_template(self, messages, add_generation_prompt, **kwargs):
                return "A" if add_generation_prompt else "B_target"
            def encode(self, text, **kwargs): return list(text.encode())
        with self.assertRaises(ValueError): supervised_tokens(Bad(), "x", "y", 200)

    def test_scienceworld_contract_uses_cumulative_score_differences(self):
        class Env:
            def load(self, *args): self.loaded = args; self.score = 0
            def reset(self): return "lab", {}
            def get_task_description(self): return "boil water"
            def step(self, action):
                self.score += 50
                return "progress", 0, self.score == 100, {"score": self.score}
            def close(self): pass
        underlying = Env()
        env = ScienceWorldEnvironment(factory=lambda: underlying)
        task = Task("s", "scienceworld", "boil", metadata={"task_name": "boil", "variation": 2})
        self.assertIn("boil water", env.reset(task))
        self.assertEqual(underlying.loaded, ("boil", 2, ""))
        self.assertEqual(env.step("heat").reward, 0.5)
        final = env.step("wait")
        self.assertEqual(final.reward, 0.5)
        self.assertTrue(final.success)
        self.assertFalse(final.accepted, "completion must not auto-certify arbitrary intermediate actions")

    def test_alfworld_contract_checks_manifest_game(self):
        class Env:
            def seed(self, seed): self.seed_value = seed
            def reset(self): return ["room"], {"extra.gamefile": ["game-a"]}
            def step(self, action): return ["done"], [1], [True], {"won": [True]}
            def close(self): pass
        env = ALFWorldEnvironment(factory=Env)
        with self.assertRaises(ValueError):
            env.reset(Task("a", "alfworld", "v1", metadata={"gamefile": "wrong-game"}))
        env.reset(Task("b", "alfworld", "v1", metadata={"gamefile": "game-a"}))
        self.assertTrue(env.step("open").success)


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch not installed")
class TinyLoraTests(unittest.TestCase):
    def test_real_gradient_update_continues_same_rank_and_reloads(self):
        from memturn.learning.tiny_lora import TinyLora
        with tempfile.TemporaryDirectory() as folder:
            learner = TinyLora(folder)
            examples = (Example("a", 0, '{"family":"vault","version":"v1"}', "inspect_lock"),)
            first = learner.train(examples, "base")
            self.assertTrue(first.metrics["base_frozen"])
            self.assertGreater(first.metrics["gradient_norm"], 0)
            self.assertLess(first.metrics["loss_after"], first.metrics["loss_before"])
            learner.prepare(first)
            second = learner.train(examples, first.version)
            self.assertEqual(second.parent_version, first.version)
            self.assertAlmostEqual(second.metrics["loss_before"], first.metrics["loss_after"], places=5)
            learner.prepare(second)
            self.assertEqual(learner.act(examples[0].prompt, second.version), "inspect_lock")
            self.assertEqual(learner.models[first.version][1].shape, learner.models[second.version][1].shape)
            restored = TinyLora(folder)
            restored.prepare(second)
            self.assertEqual(restored.act(examples[0].prompt, second.version), "inspect_lock")
            learner.retain({second.version})
            self.assertFalse(Path(first.path).exists())
            self.assertTrue(Path(second.path).exists())
