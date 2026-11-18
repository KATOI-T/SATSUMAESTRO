# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- On `openai`, `anthropic` and `xai`, `/model` alone lists the models tested with SATSUMAESTRO and
  marks the current one, so you can pick one. While you type, they are suggested after `/model `
  and after `/provider <name> `.
  `satsumaestro models` shows them after the shorthands. Other model IDs work as before; the
  list only decides what is offered. Code that embeds SATSUMAESTRO gets the same list from
  `list_models()` on a cloud endpoint, from `ModelNeededError.models`, and from
  `recommended_models()` in `satsumaestro.providers`. The models are those listed under "Checked
  models" in 0.5.1, without the ones the providers deprecated, `gpt-6-astra` and `gpt-6.1-sol`,
  which cannot call tools over Chat Completions, and `chat-latest`, whose name does not fix the
  model it uses.
  ([#26](https://github.com/KATOI-T/SATSUMAESTRO/issues/26))
- The REPL checks the API keys for `openai`, `anthropic` and `xai` in the background when it
  starts, with a request that spends no tokens (`GET /v1/models`, or `GET /v1/api-key` on xAI).
  `/provider` and its suggestions show rejected keys in red. `satsumaestro doctor --probe` checks
  the key on its `api key` line.
  `satsumaestro prompt` and switching with `/provider` do not check.
  ([#56](https://github.com/KATOI-T/SATSUMAESTRO/issues/56))

### Changed

- `/provider <name>` without a model lists the models there after switching and marks the one in
  use, as `/model` alone does. When it cannot tell which model to use, it switches without one
  instead of stopping; choose one with `/model`. Code that calls `switch_provider()` gets this
  with `model_required=False`; by default it still raises `ModelNeededError`.
  ([#26](https://github.com/KATOI-T/SATSUMAESTRO/issues/26))
- While you type `/provider ollama `, `/provider lmstudio ` or `/provider llamacpp `, the models on
  that server are suggested, as they are after `/model `. `models_for()` in
  `satsumaestro.engine.provider_switch` gives the models of a provider without switching to it.
  ([#26](https://github.com/KATOI-T/SATSUMAESTRO/issues/26))
- A request the provider answers with 429 is not sent again. Before, it was retried up to 9 times
  over about 2 minutes, which does not help when the account has no credits left. With
  `provider.fallback_model` set, a 429 does not switch to that model either. Other passing errors
  (408, 409, 5xx and dropped connections) are retried as before.
  ([#56](https://github.com/KATOI-T/SATSUMAESTRO/issues/56))
- The line shown while waiting to send a request again now says why it failed: a server error with
  its status, could not connect, the connection dropped, no response, or a busy server.
  ([#56](https://github.com/KATOI-T/SATSUMAESTRO/issues/56))

### Fixed

- With no model set (no `--model`, `OLLAMA_MODEL` or `provider.default_model`), nothing is sent.
  `satsumaestro prompt` stops, says where to set the model and lists the models to choose from:
  those on the local server, or the recommended ones on a cloud. The REPL starts, and an
  instruction gets a note to choose a model with `/model` instead of being sent. Before, the first
  request failed with `model is required`. Resuming a session without a model set uses the
  session's model. Code that embeds SATSUMAESTRO gets `ModelNotSetError` from `connect()`, with the
  list in `models`, unless it passes `model_required=False`.
  ([#26](https://github.com/KATOI-T/SATSUMAESTRO/issues/26))
- A request refused for reasons of the account now says why, with the provider's message, instead
  of only the raw error: no credits left, a rate limit, or a rejected key.
  ([#56](https://github.com/KATOI-T/SATSUMAESTRO/issues/56))

## [0.5.2] - 2026-11-17

### Added

- The REPL lists slash commands under the prompt as you type `/`, with their descriptions, and then
  their arguments: the models of the current server for `/model`, and the values of `/provider`,
  `/mode`, `/sub-mode`, `/plan`, `/effort` and `/language`, each with a one-line note (`/provider`
  shows where each provider connects or whether its key is set). The first suggestion is chosen;
  ↑/↓ move the choice, Tab fills it in, Enter runs it and Esc closes the list. Right after a
  command and a space nothing is chosen, so Enter runs the command as typed. Lines that do not
  start with `/`, and paths, get no suggestions.
  ([#25](https://github.com/KATOI-T/SATSUMAESTRO/issues/25))
- `sandbox.env_mode: "allow-list"` passes only basic names (`PATH`, `HOME`, `LANG`, `LC_*`,
  `TMPDIR` and a few more) and the names in `sandbox.env_keep` to bash, hooks and LSP servers, so
  secrets whose names do not look like keys stay out too. The default, `strip`, works as before.
  Code that embeds SATSUMAESTRO can give `ConversationRuntime` and `RuntimeFactory` (bash, also in
  sub-agents), `HookRunner` and `LanguageServers` the environment to start from instead of
  `os.environ`.
  ([#33](https://github.com/KATOI-T/SATSUMAESTRO/issues/33))
- `dotenv.into_environ: false` reads `.env` without writing it into the process environment. The
  values are used for SATSUMAESTRO's own settings and keys only, so bash, hooks and LSP servers do
  not get them, and orchestration workers get only the names SATSUMAESTRO reads. The default,
  `true`, works as before.
  ([#33](https://github.com/KATOI-T/SATSUMAESTRO/issues/33))
- `redaction.live: true` replaces secrets with `[redacted]` in tool results before they reach the
  model and the screen, in the tool input shown on the screen and in approval cards, and in change
  previews, the same way as in session records. Expansions and shell syntax that runs in a bash
  command stay visible. What runs is unchanged. The default, `false`, works as
  before.
  ([#33](https://github.com/KATOI-T/SATSUMAESTRO/issues/33))
- `tools.web_fetch.allowed_hosts` limits where `web_fetch` may go to the listed hosts and their
  subdomains, also through redirects. Other hosts are refused before their names are looked up.
  The default, an empty list, works as before.
  ([#33](https://github.com/KATOI-T/SATSUMAESTRO/issues/33))
- `permission.confirm_after_external: true` asks again for the rest of a turn once `web_fetch`,
  `web_search` or an MCP tool has returned content in it (also in a sub-agent): file writes, bash
  commands, MCP tools, sub-agents and further web requests ask even when a rule, the mode or an
  earlier "always" allows them. Provably read-only bash commands and `web_fetch` to a host in
  `tools.web_fetch.allowed_hosts` do not. A sub-agent started after such content is refused them,
  as it cannot ask. The default, `false`, works as before.
  ([#33](https://github.com/KATOI-T/SATSUMAESTRO/issues/33))

### Changed

- The `web_search` and `web_fetch` settings now go under `tools` (`tools.web_search` and
  `tools.web_fetch`), next to the other tool settings. Settings at the top level are still read for
  now, with a note at startup naming the file. The environment variables are unchanged.
  ([#33](https://github.com/KATOI-T/SATSUMAESTRO/issues/33))
- When `web_fetch` refuses a private or local address, the model is told to leave
  `tools.web_fetch.allow_private_hosts` to the user and to show where to set it, with an example and
  the host that redirected there. Without `allowed_hosts`, it also says that the setting opens every
  private and local address. ([#33](https://github.com/KATOI-T/SATSUMAESTRO/issues/33))
- A short message in English, such as `HELLO`, gets a reply in English. Before, a message needed 8
  letters to change the reply language from the one before. Two letters or fewer (`ok`) still keep
  it. The check of the reply still uses `language.min_chars`.
  ([#27](https://github.com/KATOI-T/SATSUMAESTRO/issues/27))
- `satsumaestro`, `satsuma`, `stro` and `python -m satsumaestro` without a command start the REPL,
  and options without a command go to it (`stro -m qwen3:8b`). Commands, `--help` and `--version`
  work as before, and so does `satsumaestro repl`. When stdin or stdout is not a terminal, the help
  is shown as before. ([#22](https://github.com/KATOI-T/SATSUMAESTRO/issues/22))
- `openai` is kept below 3. openai 3 moves to httpx2 and has not been tried against real servers
  yet, and a project that installs SATSUMAESTRO as a dependency could get it.
  ([#35](https://github.com/KATOI-T/SATSUMAESTRO/issues/35))

### Removed

- `/lang`, which was easy to mistake for `/language`. In the REPL the reply language now always
  follows each message. To fix it, start with `--lang ja|en` or set `language.mode` to `fixed`.
  ([#27](https://github.com/KATOI-T/SATSUMAESTRO/issues/27))

### Fixed

- A quoted value given to a secret-like name, such as `DB_PASSWORD='correct horse battery staple'`,
  is now replaced whole in session records. Before, the part after the first space was kept.
  ([#33](https://github.com/KATOI-T/SATSUMAESTRO/issues/33))

## [0.5.1] - 2026-11-09

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
  and filelock 4.2.8. The Homebrew formula moves to the same versions.

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

### Switching the provider ([#19](https://github.com/KATOI-T/SATSUMAESTRO/pull/19))

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

[0.5.2]: https://github.com/KATOI-T/SATSUMAESTRO/releases/tag/v0.5.2
[0.5.1]: https://github.com/KATOI-T/SATSUMAESTRO/releases/tag/v0.5.1
[0.5.0]: https://github.com/KATOI-T/SATSUMAESTRO/releases/tag/v0.5.0
