# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### OpenAI models ([#2](https://github.com/KATOI-T/SATSUMAESTRO/pull/2))

#### Added

- `provider.openai.hosts` (default `api.openai.com`) names the hosts that get the adjustments for
  OpenAI's own API below, and `provider.openai.negotiate: false` turns off re-sending after a 400.
  Ollama and the other OpenAI-compatible servers get the same requests as before.
- `chat-latest` is recognised as an OpenAI model.

#### Changed

- `/effort` and `provider.sampling.reasoning_effort` accept `none`, `minimal`, `low`, `medium`,
  `high`, `xhigh` and `max`. Each model accepts a different subset.

#### Fixed

- Reasoning models such as `gpt-5-nano` no longer fail on `max_tokens`; the output cap is sent as
  `max_completion_tokens`. After a 400 the request is re-sent in another shape (a lower or no output
  cap, without a rejected sampling value, with effort `none` for tools, then without tools), and the
  shape that worked is remembered per model. A change to the tools is shown once.
- A model that needs the Responses API (the `pro` models) stops at once with a message saying so.

#### Checked models

There is no fixed table of models; requests adapt to each model at run time. Each of these 23 models
replied through `satsumaestro prompt` on `api.openai.com` with a tool available.

- Work as they are: `gpt-5.5`, `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano`, `gpt-5.2`, `gpt-5.1`,
  `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `chat-latest`, `o3`, `gpt-4.1`, `gpt-4.1-mini`
- Output cap lowered to 16384 after the first request: `gpt-4o`, `gpt-4o-mini`
- Call tools only with effort `none`, which is sent for them: `gpt-6-luna`, `gpt-6-sol`,
  `gpt-6.1-terra`, `gpt-5.6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra`
- Cannot call tools over Chat Completions, so they reply without tools: `gpt-6-astra`, `gpt-6.1-sol`

Deprecated by OpenAI: `gpt-5.4-nano`, `gpt-5.1`, `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `o3`, the
`gpt-4.1` models and the `gpt-4o` models. Not supported: `gpt-5.5-pro`, `gpt-5.4-pro` and
`gpt-5.2-pro` need the Responses API.

### Anthropic models ([#3](https://github.com/KATOI-T/SATSUMAESTRO/pull/3))

#### Changed

- The effort from `/effort` and `provider.sampling.reasoning_effort` now applies to Claude models
  (`none` and `minimal` are sent as `low`). A level the model does not accept is replaced by the
  nearest one it does, with a one-time notice. If the Models API cannot describe the model, the
  request is sent as before, with effort `high`.

#### Fixed

- Claude Opus 4.5, Sonnet 4.5 and Haiku 4.5 no longer fail with "adaptive thinking is not
  supported". Requests follow each model's capabilities from the Models API: adaptive or budgeted
  thinking, the effort levels it accepts and its output limit.
- When the conversation history has been edited (for example by compaction), thinking blocks that
  no longer match it are dropped instead of failing the request with a 400.

#### Checked models

There is no fixed table of models; requests follow each model's capabilities from the Models API.
Each of these 14 models replied through `satsumaestro prompt` with a tool available. The 4.5 models,
Haiku 5.5, Sonnet 5.5, Opus 5.5 and Fable 5.1 also completed a three-turn session that calls a tool.

- Fable: `claude-fable-5-1`, `claude-fable-5`
- Opus: `claude-opus-5-5`, `claude-opus-5`, `claude-opus-4-8`, `claude-opus-4-7`, `claude-opus-4-6`,
  `claude-opus-4-5-20251101`
- Sonnet: `claude-sonnet-5-5`, `claude-sonnet-5`, `claude-sonnet-4-6`, `claude-sonnet-4-5-20250929`
  (deprecated by Anthropic)
- Haiku: `claude-haiku-5-5`, `claude-haiku-4-5-20251001`

### xAI models ([#4](https://github.com/KATOI-T/SATSUMAESTRO/pull/4))

#### Fixed

- A reasoning effort that a Grok model does not accept no longer fails the request. xAI names the
  rejected parameter only in the error text, which is now read, so the request is re-sent without
  that value and the shape that worked is remembered per model, as for OpenAI.
  `provider.openai.negotiate: false` turns this off for xAI as well.
- `grok-4.20-multi-agent-0309` stops at once with a message saying it needs the Responses API.
- A "model at capacity" error that arrives in the middle of a streamed reply is retried.

#### Checked models

There is no fixed table of models; requests adapt to each model at run time. Each of these 7 models
replied through `satsumaestro prompt` on `api.x.ai`, read a file with a tool, and replied with
reasoning effort `none`, `low` and `max`.

- Accept effort, except `none` and `max`: `grok-4.7`, `grok-4.6`, `grok-4.5`
- Accept effort, except `max`: `grok-4.3`
- Do not accept effort, so it is not sent: `grok-4.20-0309-reasoning`,
  `grok-4.20-0309-non-reasoning`, `grok-build-0.1`

Not supported: `grok-4.20-multi-agent-0309` needs the Responses API. The `grok-imagine` models
generate images and video.

### Dependencies ([#11](https://github.com/KATOI-T/SATSUMAESTRO/pull/11))

#### Security

- `uv.lock` moves to versions that fix the advisories Dependabot reported: mcp 1.30.0, PyJWT
  2.15.1, cryptography 50.0.2 and certifi 2026.10.23, and for development only, virtualenv 21.15.5
  and filelock 4.2.8. A new install from PyPI already gets fixed versions. The Homebrew formula
  follows with the next release.

### LM Studio ([#14](https://github.com/KATOI-T/SATSUMAESTRO/pull/14))

#### Added

- LM Studio is recognised at startup. A server on a local-looking host (`localhost`, a private
  address or `*.local`) is asked once through `/api/v1/models` (`/api/v0/models` on older
  versions). `provider.local.server` (`auto`, `ollama`, `lmstudio` or `generic`) sets the kind
  without asking; a value that conflicts with `provider.ollama.detect` stops when the settings load.
- The window of the loaded model is read and compaction is fitted to it, as for Ollama. Before the
  model is loaded, the model's largest window is assumed and checked again after the first reply.
  When the window is small, the REPL suggests `lms load <model> --context-length <size>`.
- When the REPL closes, it unloads the models it caused LM Studio to load; models you loaded yourself
  stay. `provider.local.unload_on_exit: false` keeps them, and `provider.local.ttl` sets how long a
  model loaded this way stays loaded while idle.
- The loading notice, the missing-model notice (`lms get`) and `satsumaestro doctor --probe` (rows
  `lmstudio`, `lmstudio ctx`, `lmstudio loaded` and `lmstudio role`) cover LM Studio.
- `LM_API_TOKEN` is sent to LM Studio when it requires authentication. Like the API keys, it is
  removed from the environment of child processes and redacted in session records.
  `OPENAI_API_KEY` is not sent to LM Studio when it looks like a cloud key.

#### Changed

- `satsumaestro doctor` shows a `local server` row for an OpenAI-compatible server that is neither
  Ollama nor LM Studio.

#### Fixed

- When LM Studio rejects a request as longer than the model's window, the window is read again, the
  history is compacted and the request is resent once. A second rejection stops the turn as before.

### llama.cpp ([#17](https://github.com/KATOI-T/SATSUMAESTRO/pull/17))

#### Added

- llama.cpp's `llama-server` is recognised at startup through `/props`, asked after LM Studio on a
  local-looking host. `provider.local.server: llamacpp` sets the kind without asking.
- The window of one slot (`n_ctx` in `/props`) is read and compaction is fitted to it. When the
  window is small, the REPL suggests restarting with `-c <size>`, and with more than one slot also
  the size for `--parallel` without `--kv-unified`.
- A server started with one model answers under any name, so the model is not checked and `/model`
  lists that one model. In router mode the model must be one the router lists; one that is not
  loaded yet gets its trained window, checked again after the first reply. When the REPL closes, it
  unloads only the models it caused the router to load (`provider.local.unload_on_exit`).
- The number of slots (`total_slots`) is the server's parallelism for decomposition when
  `decomposition.provider_parallelism` is `auto`.
- `LLAMA_API_KEY` (the key given to `--api-key`) is sent when set. Like the API keys, it is removed
  from the environment of child processes and redacted in session records.
- The loading notice, the missing-model notice and `satsumaestro doctor --probe` (rows `llamacpp`
  and `llamacpp ctx`) cover llama.cpp.
- When the model's chat template does not accept tools (`chat_template_caps` in `/props`), the
  startup and `satsumaestro doctor --probe` (row `llamacpp tools`) say so and suggest
  `--chat-template-file`.

#### Fixed

- When llama-server rejects a request as longer than the window (400 `exceed_context_size_error`),
  the window and the token count it reports are used: the history is compacted, or tool results are
  shortened by the counted amount, and the request is resent once. Before, the turn failed. Such a
  rejection is no longer re-sent first in another shape (for example without strict tools).
- A request that the counted size shows cannot fit, with nothing left to shorten, is not sent
  again. If no reply came in that turn, the request is removed from the conversation, as when it
  does not fit before it is first sent.
- When a local-looking endpoint cannot be reached at startup (for example, llama-server is not
  running yet), SATSUMAESTRO stops at once and says so, instead of retrying for about 100 seconds.
  `satsumaestro doctor --probe` shows it in the `local server` row.
- A reply from llama-server that counts fewer tokens than were sent no longer stops the turn as a
  truncated request. llama-server rejects a request longer than the window instead, and a smaller
  count means the chat template left part of it out (for example the tools).

#### Checked servers

- llama-server b12172 on macOS with gpt-oss-20b: one model (`-c 65536`, and `-c 32768 -np 4` for
  a window of 8192 per slot), `--sleep-idle-seconds`, and router mode (`--models-dir`).

### Switching the provider

#### Added

- `provider.kind` chooses the provider: `auto` (the default, chosen as before), `ollama`,
  `lmstudio`, `llamacpp`, `openai`, `anthropic` or `xai`. With a kind set, the model ID and
  `OLLAMA_HOST` do not change it, and a local server is not probed.
- `/provider` in the REPL lists the providers with their endpoint or key, and
  `/provider <name|number> [URL] [model]` switches during the session. The switch is saved to
  `~/.satsuma/settings.json`: `provider.kind`, `provider.default_model`, `provider.base_url` for a
  local server, and `provider.profiles.<kind>` so that switching back needs no URL or model. When an
  endpoint, a key or the model is missing, it says what to set and keeps the current provider.
- `LMSTUDIO_HOST` and `LLAMACPP_HOST` give the endpoints of LM Studio and llama.cpp. They are read
  like `OLLAMA_HOST`, with the default ports 1234 and 9931.
- `openai` sends to `https://api.openai.com/v1` with `OPENAI_API_KEY`; `OPENAI_BASE_URL` is not
  needed.
- Switching away from a local server unloads the models used there, as closing the REPL does.

#### Changed

- `/model` saves the model to `~/.satsuma/settings.json` as well.
- The next start uses the provider and model chosen last: `provider.kind`, `provider.base_url` and
  `provider.default_model` in `~/.satsuma/settings.json` win over the project's settings.

#### Fixed

- The `api key` row of `satsumaestro doctor` shows the endpoint actually used
  (`provider.base_url` before `OLLAMA_HOST`), and the `provider` row shows the chosen kind or
  `auto (<kind>)`.
- A history with tool calls made on an OpenAI-compatible server is accepted by Anthropic. Tool ids
  that repeat (such as `call_synth_0`) or that Anthropic does not accept are renumbered when sent.
- Decomposition no longer asks a local server for its parallelism when the model goes to a cloud
  API.

## [0.5.0] - 2026-11-01

First public release.

### Added

- Agentic coding CLI: `satsumaestro repl` (interactive) and `satsumaestro prompt` (one-shot, with
  text or JSON output). The same command is also available as `satsuma` and `stro`.
- Plan mode: the model writes a plan file and asks for approval before changing anything; approved
  plans run step by step. Switch modes with Shift+Tab or `/mode`.
- Permissions: permission modes, a Markdown policy file (`.satsuma/policy.md`), confirmation cards,
  workspace limits for file tools, secret-file protection, and a guard against starting the agent
  from inside itself.
- Directory trust: settings, hooks, MCP servers, LSP servers, `.env` and the policy of a directory are
  loaded only after you trust it.
- Providers: Ollama, OpenAI-compatible servers, OpenAI, Anthropic and xAI. Ollama-specific support for
  context window reporting, keep-alive and unloading models on exit.
- Japanese and English: replies follow the language you write in, and the screen language can be
  chosen with `/language`.
- Web search (SearXNG or DuckDuckGo) and web fetch, both disabled by default.
- Optional features: workspace index (RAG) with local embeddings, language servers (LSP) and MCP
  servers.
- `satsumaestro doctor`, `config`, `session`, `trust`, `models`, `mcp`, `rag` and `eval` commands.
- On native Windows, the CLI stops at startup and points to WSL2.

[0.5.0]: https://github.com/KATOI-T/SATSUMAESTRO/releases/tag/v0.5.0
