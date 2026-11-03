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
