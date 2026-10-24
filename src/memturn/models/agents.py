"""Fixed-prompt writer and bounded, read-only tool-using consolidator prototype."""
import json
import uuid
from dataclasses import replace
from ..contracts import Example, Memory, Support
from ..learning.materials import covered_text
from .http import parse_json


def decode_memory(value, batch, memory_id=None, source_lookup=None):
    memory_id = memory_id or f"model_{uuid.uuid4().hex}"
    memory = Memory(memory_id, value['content'], value['applicability'],
                    tuple(Support(**s) for s in value['supports']),
                    parents=tuple(value.get('parents', ())), created_batch=batch,
                    recency_anchor=batch, consolidated=bool(value.get('parents')))
    try:
        return replace(memory, examples=_decode_examples(value,memory,source_lookup))
    except (KeyError,TypeError,ValueError,AttributeError):
        if not memory.consolidated:
            raise
        # Training-material failure alone cannot discard a valid external summary.
        # Independent memory/provenance validation still runs before replacement.
        return memory


def _decode_examples(value, memory, source_lookup):
    local_id = value.get('local_id', memory.id)
    examples = []
    for original in value.get('training_examples', value.get('examples', [])):
        item = dict(original)
        kind = item.get('example_type', 'decision')
        if item.get('memory_local_id', local_id) != local_id:
            raise ValueError('example belongs to another emitted memory')
        if item.get('memory_ref', memory.ref) not in (memory.ref, local_id):
            raise ValueError('example belongs to another memory version')
        refs = item.get('covered_content_refs', ['content'] if memory.consolidated else [])
        normalized = []
        for ref in refs:
            if ref == 'content' or ref.startswith('span:'):
                normalized.append(memory.ref + '#' + ref)
            elif ref.startswith(str(local_id) + '#'):
                normalized.append(memory.ref + ref[len(str(local_id)):])
            elif ref.startswith(memory.ref + '#'):
                normalized.append(ref)
            else:
                raise ValueError('foreign covered-content reference')
        if kind == 'rule_qa':
            refs = tuple(normalized)
            target = item.get('target', covered_text(memory, refs))
            supports = tuple(Support(**s) for s in item.get('support_refs', value['supports']))
            examples.append(Example('', -1, item.get('question', item.get('prompt', '')), target,
                example_type='rule_qa', output_kind=item.get('output_kind', 'text'), memory_ref=memory.ref,
                covered_content_refs=refs, support_refs=supports))
            continue
        if kind != 'decision':
            raise ValueError('unknown example type')
        if "prompt" not in item or "target" not in item:
            episode = source_lookup(item["task_id"]) if source_lookup else None
            index = item["step"]
            if not episode or not isinstance(index, int) or not 0 <= index < len(episode.steps):
                raise ValueError("training reference has no available source step")
            source = episode.steps[index]
            if not source.accepted and not source.evidence.get('native_environment'):
                raise ValueError("training reference points to an unaccepted source action")
            item.update(prompt=source.observation, target=source.action)
        examples.append(Example(item['task_id'], item['step'], item['prompt'], item['target'],
            input_step=item.get('input_step'), example_type='decision',
            output_kind=item.get('output_kind', 'environment_action'),
            memory_ref=memory.ref if memory.consolidated else '',
            covered_content_refs=tuple(normalized),
            support_refs=(Support(item['task_id'],item['step']),) if memory.consolidated else ()))
    return tuple(examples)


SCHEMA = ('Memory fields: content:string, applicability:string, '
          'supports:[{task_id:string,step:integer}], '
          'examples:[{task_id:string,step:integer}], '
          'parents:[input memory refs]. A support must identify evidence for this '
          'specific conclusion. Step indices are ZERO-BASED. Do not attach unrelated '
          'source IDs. Examples cite accepted source steps; the program copies their '
          'exact observation/action. Never invent prompt, target or feedback text. '
          'Preserve version conditions and unresolved uncertainty.')

MIXED_SCHEMA = (
    'Memory fields: local_id:string, content:string, applicability:string, '
    'supports:[{task_id:string,step:integer}], parents:[contributing input refs], examples:[...]. '
    'In regular mode, a trainable memory needs one rule_qa plus source-linked decision examples. '
    'rule_qa fields: example_type="rule_qa", output_kind="text", question:string, '
    'covered_content_refs:["content"]. Its target is copied from content by the program; '
    'ask about applicable public circumstances without quoting the answer. '
    'decision fields: example_type="decision", output_kind="environment_action", task_id:string, '
    'step:zero-based integer, covered_content_refs:["content"]. The program copies the original '
    'observation/action; do not invent observations, move actions to earlier states, or add hidden labels. '
    'Choose decisions that actually evidence this content and its conditions. '
    'Use supports only for those specific conclusions. If insufficient evidence/material, emit an '
    'external memory with examples:[]; capacity mode emits examples:[]. No self-certified verification. '
    'Question generation does not add source task support. Preserve uncertainty and native environment rules.')


class ChatWriter:
    def __init__(self, client, task_guidance=""):
        self.client, self.task_guidance = client, task_guidance

    def write(self, episode, batch):
        messages = [{"role": "system", "content":
                     'Extract one grounded experience from this completed task. Return only JSON. '
                     'Return {"memory":null} if no acceptable evidence, otherwise '
                     '{"memory":{...}}. ' + SCHEMA + " " + self.task_guidance},
                    {"role": "user", "content": json.dumps(dict(episode.public_dict(),
                        steps=[dict(step.__dict__, step_index=i) for i, step in enumerate(episode.steps)]))}]
        result = parse_json(self.client.complete(messages))["memory"]
        if result is None:
            return None
        result["parents"] = []
        return decode_memory(result, batch, f"writer_{episode.task_id}",
                             lambda task_id: episode if task_id == episode.task_id else None)


class ChatConsolidator:
    """A prompted operator, explicitly not the GRPO-trained Auto-Dreamer artifact."""
    def __init__(self, client, store, counter, max_turns=12, transcript_token_cap=16384, task_guidance=""):
        self.client, self.store, self.counter = client, store, counter
        self.max_turns, self.transcript_token_cap = max_turns, transcript_token_cap
        self.task_guidance = task_guidance

    def consolidate(self, region, mode, batch, output_budget):
        refs = {m.ref: m for m in region}
        tasks = set().union(*(m.task_ids for m in region))
        messages = [{"role": "system", "content":
                     'Consolidate the selected read-only region. Return one JSON operation per turn: '
                     '{"op":"inspect_entry","ref":"..."}, '
                     '{"op":"inspect_source","task_id":"...","step":0}, '
                     '{"op":"emit","memory":{...}}, or {"op":"finish"}. '
                     'Only emitted entries replace the region; preserve useful conditions. '
                     'Never access sources outside the region. Capacity mode does not train. ' + MIXED_SCHEMA + " " + self.task_guidance},
                    {"role": "user", "content": json.dumps({"mode": mode, "body_token_budget": output_budget,
                     "max_examples_total": self.store.config.max_examples_per_job,
                     "region": [{"ref": m.ref, "text": m.payload} for m in region]})}]
        outputs = []
        for _ in range(self.max_turns):
            if self.counter.count(json.dumps(messages)) > self.transcript_token_cap:
                break
            text = self.client.complete(messages)
            operation = parse_json(text)
            messages.append({"role": "assistant", "content": text})
            kind = operation.get("op")
            if kind == "finish":
                break
            if kind == "inspect_entry":
                memory = refs.get(operation.get("ref"))
                result = memory.to_dict() if memory else {"error": "outside selected region"}
            elif kind == "inspect_source":
                task_id, step = operation.get("task_id"), operation.get("step")
                episode = self.store.episode(task_id) if task_id in tasks else None
                if episode and isinstance(step, int) and 0 <= step < len(episode.steps):
                    result = {"task_id": task_id, "step": step, "evidence": episode.steps[step].__dict__}
                    if episode.family not in ('scienceworld', 'alfworld'):
                        result['condition'] = episode.condition
                else:
                    result = {"error": "unavailable source or step"}
            elif kind == "emit":
                memory = decode_memory(operation["memory"], batch,
                    source_lookup=lambda task_id: self.store.episode(task_id) if task_id in tasks else None)
                if not memory.parents or not set(memory.parents) <= set(refs):
                    raise ValueError("emitted memory has invalid parents")
                outputs.append(memory)
                if sum(len(m.examples) for m in outputs) > self.store.config.max_examples_per_job:
                    raise ValueError('consolidator exceeded its training-example budget')
                if mode == 'capacity' and memory.examples:
                    raise ValueError('capacity consolidation must not emit training material')
                if sum(self.counter.count(m.payload) for m in outputs) > output_budget:
                    raise ValueError("consolidator exceeded its body output budget")
                result = {"accepted_as_candidate": memory.ref}
            else:
                raise ValueError("unknown consolidator operation")
            response = json.dumps(result)
            if self.counter.count(response) > self.transcript_token_cap // 2:
                response = json.dumps({"error": "requested evidence exceeds tool response budget"})
            messages.append({"role": "user", "content": response})
        return tuple(outputs)
