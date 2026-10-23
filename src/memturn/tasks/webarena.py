"""Reserved integration boundary; postponed by the current experiment scope."""
class WebArenaEnvironment:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "WebArena is deferred. Choose a trainable task model, site deployment, "
            "task manifest and official evaluator before enabling this adapter."
        )
