"""Deterministic eligibility and seeded ranking; no learned scheduler."""
import math
import random
from ..contracts import Memory


def ranked_by_support(memories, valid_support_count, seed):
    items = sorted(memories, key=lambda m: m.ref)
    random.Random(seed).shuffle(items)
    return sorted(items, key=lambda m: valid_support_count(m), reverse=True)


def select_one(memories, validator, counter, config, seed=None):
    eligible = [m for m in memories if m.consolidated and m.examples
                and counter.count(m.payload) <= config.entry_token_cap
                and validator.support_count(m) >= config.min_support_tasks
                and validator.trainable(m)]
    seed = config.seed if seed is None else seed
    if config.selection_policy == 'random':
        ranked = sorted(eligible, key=lambda m: m.ref)
        random.Random(seed).shuffle(ranked)
    else:
        ranked = ranked_by_support(eligible, validator.support_count, seed)
    return ranked[0] if ranked else None


def cold_region(memories, batch, counter, config, excluded=()):
    blocked = set(excluded)
    cold = [m for m in memories if m.ref not in blocked and batch - m.recency_anchor >= 1]
    cold.sort(key=lambda m: (m.recency_anchor, m.id))
    selected, tokens = [], 0
    limit = max(1, math.ceil(len(cold) * config.cold_fraction))
    # Oversized cold entries must not permanently starve later usable regions.
    for index, memory in enumerate(cold):
        if index >= limit and selected:
            break
        size = counter.count(memory.payload)
        if tokens + size <= config.region_token_cap:
            selected.append(memory)
            tokens += size
    return tuple(selected)


def pack_half(region, candidates, validator, counter, config, seed=None):
    budget = min(math.floor(sum(counter.count(m.payload) for m in region)
                            * config.return_token_ratio), config.output_token_cap)
    kept, used = [], 0
    for memory in ranked_by_support(candidates, validator.support_count,
                                     config.seed if seed is None else seed):
        if not validator.valid_memory(memory):
            continue
        size = counter.count(memory.payload)
        if size and used + size <= budget:
            kept.append(memory)
            used += size
    return tuple(kept), budget
