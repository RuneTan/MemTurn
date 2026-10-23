"""One text environment per adapter; no globally shared simulator state."""
from ..contracts import Transition


class ALFWorldEnvironment:
    def __init__(self, config_path=None, split="eval_out_of_distribution", factory=None):
        self.config_path, self.split, self.factory, self.env = config_path, split, factory, None

    def reset(self, task):
        if self.factory:
            self.env = self.factory()
        else:
            import yaml
            from alfworld.agents.environment import get_environment
            with open(self.config_path, encoding="utf-8") as handle:
                config = yaml.safe_load(handle)
            self.env = get_environment(config["env"]["type"])(config, train_eval=self.split).init_env(batch_size=1)
        if hasattr(self.env, "seed"):
            self.env.seed(task.seed)
        observations, info = self.env.reset()
        expected = task.metadata.get("gamefile")
        actual = info.get("extra.gamefile", [None])[0]
        if expected and actual != expected:
            raise ValueError("ALFWorld reset selected a different game than the manifest")
        self.previous_score = 0.0
        return str(observations[0])

    def step(self, action):
        observations, scores, dones, info = self.env.step([action])
        score = float(scores[0])
        delta, self.previous_score = score - self.previous_score, score
        won = bool(info.get("won", [score >= 1.0])[0])
        return Transition(str(observations[0]), delta, bool(dones[0]), accepted=False,
                          success=bool(dones[0] and won))

    def close(self):
        if self.env is not None:
            self.env.close()
