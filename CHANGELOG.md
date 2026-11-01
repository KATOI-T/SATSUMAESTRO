# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
