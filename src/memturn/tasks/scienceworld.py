"""ScienceWorld adapter. Optional Java/environment assets are installed separately."""
from ..contracts import Transition


class ScienceWorldEnvironment:
    def __init__(self, factory=None, jar_path=None, simplifications=""):
        self.factory, self.jar_path, self.simplifications = factory, jar_path, simplifications
        self.env = None

    def reset(self, task):
        self.close()
        if self.factory is None:
            from scienceworld import ScienceWorldEnv
            self.env = ScienceWorldEnv("", self.jar_path, envStepLimit=task.max_steps)
        else:
            self.env = self.factory()
        self.env.load(task.metadata["task_name"], int(task.metadata["variation"]), self.simplifications)
        observation, info = self.env.reset()
        self.previous_score = 0.0
        self.valid = info.get('valid') if isinstance(info,dict) else None
        describe = getattr(self.env,'taskdescription',None) or self.env.get_task_description
        self.description = describe()
        return f"Task: {self.description}\n{observation}"

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if "score" not in info:
            raise ValueError("ScienceWorld response lacks cumulative score; check the pinned version")
        score = float(info["score"]) / 100.0
        admissible = action in self.valid if isinstance(self.valid,list) and all(isinstance(x,str) for x in self.valid) else None
        self.valid = info.get('valid')
        delta, self.previous_score = score - self.previous_score, score
        return Transition(f"Task: {self.description}\n{observation}", delta, bool(done),
                          accepted=False, success=bool(done and score >= 1.0),
                          evidence={'native_environment':'scienceworld','native_score':info['score'],
                                    'native_reward':reward,'action_admissible':admissible})

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
