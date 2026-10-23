"""ScienceWorld adapter. Optional Java/environment assets are installed separately."""
from ..contracts import Transition


class ScienceWorldEnvironment:
    def __init__(self, factory=None, jar_path=None, simplifications=""):
        self.factory, self.jar_path, self.simplifications = factory, jar_path, simplifications
        self.env = None

    def reset(self, task):
        if self.factory is None:
            from scienceworld import ScienceWorldEnv
            self.env = ScienceWorldEnv("", self.jar_path, envStepLimit=task.max_steps)
        else:
            self.env = self.factory()
        self.env.load(task.metadata["task_name"], int(task.metadata["variation"]), self.simplifications)
        observation, _ = self.env.reset()
        self.previous_score = 0.0
        self.description = self.env.get_task_description()
        return f"Task: {self.description}\n{observation}"

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if "score" not in info:
            raise ValueError("ScienceWorld response lacks cumulative score; check the pinned version")
        score = max(0.0, float(info["score"]) / 100.0)
        delta, self.previous_score = score - self.previous_score, score
        return Transition(f"Task: {self.description}\n{observation}", delta, bool(done),
                          accepted=False, success=bool(done and score >= 1.0))

    def close(self):
        if self.env is not None:
            self.env.close()
