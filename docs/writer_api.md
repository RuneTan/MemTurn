# Optional external Writer

[English](writer_api.md) | [简体中文](writer_api.zh-CN.md)

Checked against official provider pages on 2026-10-23. A stronger fixed Writer
can run remotely while the task model and LoRA training remain local. It does
not need trainable weights. Provider output still passes the same evidence and
sample checks; JSON validity alone does not establish factual support.

## Free-tier candidates

| Provider | Candidate | Published free limits and suitability |
| --- | --- | --- |
| Groq | `openai/gpt-oss-120b` | 30 requests/minute, 1,000/day, 8,000 tokens/minute and 200,000/day; suitable for a small Writer pilot |
| Google AI Studio | `gemini-3.8-flash` | Free input/output tier listed; exact active model limits are shown per project in AI Studio |
| OpenRouter | A currently available free model | Free plan lists 50 requests/day; useful for spot checks, limited for repeated experiment runs |

Groq supports the chat-compatible API and strict structured output on GPT-OSS
120B. The template initially requests JSON object mode; adding a full strict
schema is a separate improvement and cannot replace provenance validation.
Google also offers a compatible chat endpoint. Free availability and limits can
change; verify the account's actual plan and selected model before running.
Google's pricing page states that free-tier content may be used to improve its
products, which matters when choosing what task data to submit.

Official sources:

- [Groq free-plan limits](https://console.groq.com/docs/rate-limits)
- [Groq structured outputs](https://console.groq.com/docs/structured-outputs)
- [Groq compatible endpoint](https://console.groq.com/docs/openai)
- [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing)
- [Gemini active rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)
- [Gemini compatible endpoint](https://ai.google.dev/gemini-api/docs/openai)
- [OpenRouter pricing](https://openrouter.ai/pricing)

## Configuration

Create a key through the provider's own console. Configure it in the launching
process as `MEMTURN_WRITER_API_KEY`; the program reads that named environment
variable and does not print it. `.env` files are not automatically loaded.
No provider key is included in either example JSON.

With that variable set, run a two-task pilot:

```text
.venv-local/Scripts/python.exe run_memturn.py local-smoke --tasks 2 --operators fixture --writer-profile configs/writer.groq.example.json
```

This combines a local task model, the remote Writer and a fixture Consolidator.
Switch to `configs/writer.gemini.example.json` to select Google. Increase the task
count and add `--train` only after checking provider access, actual usage and
Writer admission. Two tasks across different families do not guarantee enough
supports for an eligible SFT entry.

The HTTP client omits Qwen-specific `chat_template_kwargs` unless explicitly
configured, preventing those options from leaking into other providers. Existing
remote Qwen profiles explicitly disable thinking. It sends an application
User-Agent, and the Writer explicitly requests JSON as required by Groq's JSON
object mode. Both compatibility requirements were checked against the live API.

## Checked Groq pilot

Run `local_20261023T151237196728Z` combined local Qwen3-1.7B task execution, Groq
GPT-OSS 120B writing, fixture consolidation and local language-model SFT. Four
tasks completed; all four Writer outputs were admitted; one consolidation and
one LoRA publication completed, with no background errors. The Writer used four
successful requests, 3,561 input tokens and 1,248 completion tokens. Peak CUDA
allocation was 9.09 GiB. The adapter was published at the stream end, so subsequent
performance was not evaluated. Gemini remains untested.

The application reads credentials from environment variables and does not
automatically load credential files.
Rate limits and provider errors are recorded through the background failure path;
they are not silently converted into admitted memories.
