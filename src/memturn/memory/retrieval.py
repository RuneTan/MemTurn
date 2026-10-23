"""A replaceable, fixed lexical retriever used for offline integration checks."""
import re


def terms(text):
    return set(re.findall(r"\w+", text.lower()))


class Retriever:
    def __init__(self, counter, config):
        self.counter, self.config = counter, config

    def search(self, memories, query, excluded_refs=(), excluded_tasks=(), applicability=None):
        blocked, source_blocked, query_terms = set(excluded_refs), set(excluded_tasks), terms(query)
        candidates = []
        for memory in memories:
            if memory.ref in blocked or source_blocked.intersection(memory.task_ids):
                continue
            if applicability is not None and memory.applicability != applicability:
                continue
            score = len(query_terms.intersection(terms(memory.payload)))
            if score:
                candidates.append((score, memory))
        candidates.sort(key=lambda item: (-item[0], item[1].ref))
        results, used = [], 0
        for _, memory in candidates:
            size = self.counter.count(memory.payload)
            if used + size <= self.config.retrieval_token_cap:
                results.append(memory)
                used += size
            if len(results) >= self.config.retrieval_top_k:
                break
        return tuple(results)
