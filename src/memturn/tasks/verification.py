"""Fail-closed boundary for benchmark-specific evidence verification."""
class BenchmarkVerifierRequired:
    def verify_support(self, memory, episode, support):
        return False

    def verify_example(self, example, episode):
        return False


# Simulator completion rewards do not automatically certify intermediate
# actions or arbitrary free-form summaries. Register a replay/domain verifier
# instead of marking every action in a successful trajectory as correct.
