<!-- ryeos:signed:2026-10-28T23:15:20Z:ca03eb9b2bf32147a72fd30198205a26ca7d4dbe9cc77798e7c1665f7f54c55e:67YVOw7cf/Icyi008nWkfvFnRQ9vzUa+ZSe08wo9EOW05PjGmui/0dRyGUsPuEedoplkvvQ3J2P45sxl9ZzvAg==:82970d668dcc39d38a33264f59b1bc26760be487512a086be87bfd51e2d88b19 -->
---
description: "F0 gate of the harness-engineering study: one bounded Gemma turn calls the baseline context collector H1 (tool:gemma-agent/collect) on a declared fixture, then acts on what came back by naming the file and line to edit from the tool's output alone."
version: "0.1.0"
category: gemma-agent
model:
  provider: local-openai
  name: gemma-4-31b-it
  context_window: 32768
requires:
  capabilities:
    declared:
      - ryeos.execute.tool.gemma-agent/collect
return_nudge: "The collector budget is spent. Call directive_return now with query, target (path:line from the excerpts you already have, or none), evidence and notes."
limits:
  turns: 4
  tool_calls: 1
  tokens: 16000
  spend_usd: "0.01"
  duration_seconds: 900
inputs:
  - name: fixture
    type: string
    required: true
  - name: issue
    type: string
    required: true
outputs:
  - name: query
    description: "the query string you passed to the collector"
    type: string
  - name: target
    description: "path:line of the one place most likely to need the change, copied exactly from the collector's output"
    type: string
  - name: evidence
    description: "the one source line at that path:line, copied verbatim from the collector's excerpt"
    type: string
  - name: notes
    description: "one or two sentences on why that location, citing only what the collector returned"
    type: string
---

# Locate the change for one issue, using the collector only

You are working on the repository in fixture `${inputs.fixture}`. The issue to fix is:

${inputs.issue}

You cannot read files. Your only source of repository content is the collector tool.

## procedure
Exactly two actions: one collector call, then one return. Do not call the collector a second time, even if the
excerpts look incomplete; choose among what came back.
1. Call `tool:gemma-agent/collect` once with parameters
   {"fixture": "${inputs.fixture}", "query": "<three to six words naming the symbols or behaviour from the issue>", "max_excerpts": 6, "context_lines": 6}
2. Read the returned `excerpts`. Each has `path`, `line`, `score`, `definition` and `text` (numbered source lines).
   Choose the ONE path:line most likely to need the change for this issue.
3. Finish by calling `directive_return` with the four outputs: `query` as you sent it; `target` as `path:line` copied from that excerpt; `evidence` as the
   source line at that number copied verbatim from the excerpt's `text`; `notes` in one or two sentences.
   If no excerpt fits, set `target` to `none` and say so in `notes`. Never invent a path or a line that the
   collector did not return.
