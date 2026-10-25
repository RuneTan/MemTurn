"""A narrow native verifier for recorded transitions, not general language rules."""
import json

from .verification import NativeReplayVerifier


GUIDANCE = (
    'For the observed_transition verifier, content must be a JSON-encoded object with exactly '
    'kind="observed_transition", observation_excerpt:string, action:string, feedback_excerpt:string. '
    'All excerpts must be literal substrings of the cited native source step; action must match exactly. '
    'This states only an observed transition, never a universal rule or an optimal action. '
    'applicability must equal "Observed in FAMILY; public observation contains: EXCERPT", '
    'using the episode family and observation_excerpt. Cite only admissible actions with a positive '
    'native score change. Preserve this format when consolidating. If no shared transition is '
    'supported, emit no candidate. Training examples must refer to the same transition.'
)


def positive_progress(episode, index):
    step = episode.steps[index]
    before = step.evidence.get('native_score_before')
    if before is None and index > 0:
        before = episode.steps[index-1].evidence.get('native_score')
    after = step.evidence.get('native_score')
    return (step.evidence.get('action_admissible') is True and
            isinstance(before, (int, float)) and isinstance(after, (int, float)) and after > before)


def check_transition(memory, episode, support):
    try:
        claim = json.loads(memory.content)
        if set(claim) != {'kind','observation_excerpt','action','feedback_excerpt'}:
            return False
        if claim['kind'] != 'observed_transition' or not all(isinstance(v, str) and v.strip() for v in claim.values()):
            return False
        step = episode.steps[support.step]
        expected = f"Observed in {episode.family}; public observation contains: {claim['observation_excerpt']}"
        return (memory.applicability == expected and claim['observation_excerpt'] in step.observation
                and claim['action'] == step.action and claim['feedback_excerpt'] in step.feedback
                and positive_progress(episode, support.step))
    except (KeyError, TypeError, ValueError, IndexError):
        return False


class ObservedTransitionVerifier(NativeReplayVerifier):
    def verify_example(self, example, episode):
        return super().verify_example(example, episode) and positive_progress(episode, example.step)


def make_verifier(environment):
    from ..experiments import environment_for
    return ObservedTransitionVerifier(lambda: environment_for(environment), check_transition)
