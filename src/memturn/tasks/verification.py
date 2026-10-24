"""Fail-closed boundary for benchmark-specific evidence verification."""
from collections import OrderedDict
import hashlib

from ..contracts import Task, dumps


class NativeReplayVerifier:
    """Replay source facts; a separate domain predicate must approve the conclusion.

    Replay consistency/admissibility is NOT a proof of a free-form generalization.
    Without summary_check, memory support fails closed, even after successful replay.
    """
    def __init__(self, environment_factory, summary_check=None, max_cached=32):
        if max_cached < 1:
            raise ValueError('max_cached must be positive')
        self.environment_factory, self.summary_check = environment_factory, summary_check
        self.max_cached, self.cache = max_cached, OrderedDict()

    def replay_matches(self, episode):
        key = hashlib.sha256(dumps(episode.to_dict()).encode()).hexdigest()
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        valid, environment = False, None
        try:
            if episode.family not in ('scienceworld','alfworld') or not episode.task_spec or not episode.steps:
                return False
            task = Task(**episode.task_spec)
            if task.id != episode.task_id or task.family != episode.family:
                return False
            environment = self.environment_factory()
            observation = environment.reset(task)
            reward = 0.0
            for index, step in enumerate(episode.steps):
                if observation != step.observation: return False
                transition = environment.step(step.action)
                if transition.observation != step.feedback: return False
                if transition.evidence != step.evidence or transition.accepted != step.accepted: return False
                if transition.done and index != len(episode.steps)-1: return False
                observation = transition.observation
                reward += transition.reward
            valid = abs(reward-episode.reward) < 1e-8 and transition.success == episode.success
            return valid
        except (ValueError, RuntimeError, OSError):
            return False
        finally:
            if environment:
                try:
                    environment.close()
                except (RuntimeError, OSError):
                    pass
            self.cache[key] = valid
            while len(self.cache) > self.max_cached: self.cache.popitem(last=False)

    def verify_support(self, memory, episode, support):
        try:
            return bool(support.task_id == episode.task_id and 0 <= support.step < len(episode.steps)
                        and self.summary_check and self.replay_matches(episode)
                        and self.summary_check(memory,episode,support))
        except Exception:
            return False

    def verify_example(self, example, episode):
        if example.task_id != episode.task_id or not 0 <= example.step < len(episode.steps): return False
        step = episode.steps[example.step]
        return bool(example.input_step in (None,example.step) and step.observation == example.prompt
                    and step.action == example.target and step.evidence.get('action_admissible') is True
                    and self.replay_matches(episode))


class BenchmarkVerifierRequired:
    def verify_support(self, memory, episode, support):
        return False

    def verify_example(self, example, episode):
        return False


# Simulator completion rewards do not automatically certify intermediate
# actions or arbitrary free-form summaries. Register a replay/domain verifier
# instead of marking every action in a successful trajectory as correct.
