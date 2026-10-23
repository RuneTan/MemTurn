"""Two checkable tool families. A development fixture, not a paper benchmark."""
import json
from ..contracts import Episode, Example, Memory, Support, Task, Transition, dumps


PREPARE = {"vault:v1": "inspect_lock", "vault:v2": "check_permission",
           "api:v1": "refresh_token", "api:v2": "renew_session"}
FINISH = {"vault": "open", "api": "retry"}


def task_stream(count, seed=42):
    for index in range(count):
        family = "vault" if index % 2 == 0 else "api"
        version = "v1" if (index // 8) % 2 == 0 else "v2"
        yield Task(f"task_{seed}_{index:04}", family, version, seed + index)


class ToyEnvironment:
    def reset(self, task):
        if f"{task.family}:{task.version}" not in PREPARE:
            raise ValueError("unknown toy condition")
        self.task, self.phase, self.feedback = task, "prepare", ""
        self.mistakes = 0
        return self._observation()

    def _observation(self):
        return dumps({"family": self.task.family, "version": self.task.version,
                      "object": self.task.seed, "phase": self.phase, "feedback": self.feedback,
                      "goal": "complete the operation using accepted preparation then " + FINISH[self.task.family],
                      "available_actions": sorted(set(PREPARE.values()) | set(FINISH.values()))})

    def step(self, action):
        expected = PREPARE[f"{self.task.family}:{self.task.version}"] if self.phase == "prepare" else FINISH[self.task.family]
        if action != expected:
            self.mistakes += 1
            self.feedback = f"Rejected. Required next action: {expected}"
            return Transition(self._observation(), -0.1, False)
        if self.phase == "prepare":
            self.phase, self.feedback = "finish", "Preparation accepted. Complete the requested operation."
            return Transition(self._observation(), 0.0, False, True)
        self.phase, self.feedback = "done", "Task complete."
        return Transition(self._observation(), 1.0, True, True, True)

    def close(self):
        pass


class ToyVerifier:
    def verify_support(self, memory, episode, support):
        if support.task_id != episode.task_id or not 0 <= support.step < len(episode.steps):
            return False
        step = episode.steps[support.step]
        expected = PREPARE.get(episode.condition)
        return bool(expected and step.accepted and step.action == expected
                    and memory.applicability == episode.condition
                    and expected in memory.content)

    def verify_example(self, example, episode):
        if example.task_id != episode.task_id or not 0 <= example.step < len(episode.steps):
            return False
        step = episode.steps[example.step]
        return (step.accepted and step.observation == example.prompt and step.action == example.target
                and step.action == PREPARE.get(episode.condition))


class ToyPolicy:
    """Uses visible feedback or retrieved rules, never the verifier's answer table."""
    def __init__(self, learned_policy=None):
        self.learned_policy = learned_policy

    def act(self, observation, memories, model_version="base", history=()):
        state = json.loads(observation)
        if state["phase"] == "finish":
            return FINISH[state["family"]]
        marker = "Required next action: "
        if marker in state["feedback"]:
            return state["feedback"].split(marker, 1)[1]
        condition = f"{state['family']}:{state['version']}"
        for memory in memories:
            if memory.applicability == condition and memory.examples:
                return memory.examples[0].target
            if memory.applicability == condition and "first call " in memory.content:
                return memory.content.split("first call ", 1)[1].split(".", 1)[0]
        if self.learned_policy:
            return self.learned_policy.act(observation, model_version)
        return FINISH[state["family"]]
