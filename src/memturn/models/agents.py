"""Fixed-prompt writer and bounded, read-only tool-using consolidator prototype."""
import json
import uuid
from ..contracts import Example, Memory, Support
from .http import parse_json


def decode_memory(value, batch, memory_id=None, source_lookup=None):
    examples = []
    for item in value.get("examples", []):
        if "prompt" not in item or "target" not in item:
            episode = source_lookup(item["task_id"]) if source_lookup else None
            index = item["step"]
            if not episode or not isinstance(index, int) or not 0 <= index < len(episode.steps):
                raise ValueError("training reference has no available source step")
            source = episode.steps[index]
            if not source.accepted:
                raise ValueError("training reference points to an unaccepted source action")
            item = {"task_id": item["task_id"], "step": index,
                    "prompt": source.observation, "target": source.action}
        examples.append(Example(**item))
    return Memory(memory_id or f"model_{uuid.uuid4().hex}", value["content"], value["applicability"],
                  tuple(Support(**s) for s in value["supports"]),
                  tuple(examples),
                  tuple(value.get("parents", [])), created_batch=batch,
                  recency_anchor=batch, consolidated=bool(value.get("parents")))


SCHEMA = ('Memory fields: content:string, applicability:string, '
          'supports:[{task_id:string,step:integer}], '
          'examples:[{task_id:string,step:integer}], '
          'parents:[input memory refs]. A support must identify evidence for this '
          'specific conclusion. Step indices are ZERO-BASED. Do not attach unrelated '
          'source IDs. Examples cite accepted source steps; the program copies their '
          'exact observation/action. Never invent prompt, target or feedback text. '
          'Preserve version conditions and unresolved uncertainty.')


class ChatWriter:
    def __init__(self, client, task_guidance=""):
        self.client, self.task_guidance = client, task_guidance

    def write(self, episode, batch):
        messages = [{"role": "system", "content":
                     'Extract one grounded experience from this completed task. Return only JSON. '
                     'Return {"memory":null} if no acceptable evidence, otherwise '
                     '{"memory":{...}}. ' + SCHEMA + " " + self.task_guidance},
                    {"role": "user", "content": json.dumps(dict(episode.to_dict(),
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
                     'Never access sources outside the region. Capacity mode does not train. ' + SCHEMA + " " + self.task_guidance},
                    {"role": "user", "content": json.dumps({"mode": mode, "body_token_budget": output_budget,
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
                    result = {"task_id": task_id, "condition": episode.condition,
                              "step": step, "evidence": episode.steps[step].__dict__}
                else:
                    result = {"error": "unavailable source or step"}
            elif kind == "emit":
                memory = decode_memory(operation["memory"], batch,
                    source_lookup=lambda task_id: self.store.episode(task_id) if task_id in tasks else None)
                if not memory.parents or not set(memory.parents) <= set(refs):
                    raise ValueError("emitted memory has invalid parents")
                outputs.append(memory)
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
