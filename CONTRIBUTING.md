# Contributing to SATSUMAESTRO

Thank you for your interest in SATSUMAESTRO. Bug reports, ideas, documentation fixes and code are all
welcome. By taking part you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

- **Security issues**: do not open a public issue. Follow [SECURITY.md](SECURITY.md) instead.
- **Bugs and feature ideas**: open an issue with one of the templates. For a larger change, please open
  an issue first so we can agree on the direction before you spend time on it.
- **Questions**: open an issue with the Question template. Japanese and English are both OK.

## Development setup

You need Python 3.12 or later, [uv](https://docs.astral.sh/uv/), git and make. macOS and Linux are
supported; on Windows, work inside WSL2.

```bash
git clone https://github.com/<you>/SATSUMAESTRO.git   # your fork
cd SATSUMAESTRO
git remote add upstream https://github.com/KATOI-T/SATSUMAESTRO.git
make setup                        # sync dependencies (including dev) and install the pre-commit hooks
uv run satsumaestro --version
```

Running against a real model needs [Ollama](https://ollama.com) (or another provider) and a model that
supports tool calling; see the [README](README.md#quick-start-with-ollama). The checks below do not
need a model or a GPU.

### Make targets

| Command | What it does |
|---|---|
| `make setup` | Sync dependencies (including dev) and install the pre-commit hooks |
| `make lint` | Lint with ruff |
| `make format` | Format with ruff (rewrites files) |
| `make type` | Type-check with mypy (strict) |
| `make test` | Run pytest |
| `make check` | All of the above in one go (lint, format check, type check, tests) |
| `make smoke` | REPL smoke test: a real REPL in a pty against a fake provider (plan, approval, steps, completion, deletion). No GPU, about 30 seconds |
| `make run ARGS="--version"` | Run the CLI with `ARGS` |
| `make repl MODEL=...` | Start the interactive REPL |
| `make prompt TEXT="..." MODEL=...` | One-shot prompt |
| `make eval MODEL=...` | Run the default evaluation suite (`evals/code_analysis`) |
| `make doctor` | Diagnose settings, provider, keys and policy |
| `make searxng-up` / `make searxng-down` | Start / stop a self-hosted SearXNG for `web_search` |
| `make clean` / `make distclean` | Remove caches / also remove `.venv` |

`make help` lists every target. The targets are thin wrappers around `uv run <tool>`, so you can also
run the tools directly.

### Pre-commit hooks

`make setup` (or `uv run pre-commit install`) installs git hooks that run **ruff, ruff-format, mypy
and pytest** on every `git commit`. If any of them fails, the commit is stopped. Use
`git commit --no-verify` only in an emergency; the same checks must pass before a pull request is
merged.

## Branches and pull requests

- `main` holds released code. `develop` is where work lands.
- Fork the repository and create a branch from the latest `develop` of this repository (for example
  `git fetch upstream && git switch -c fix/short-name upstream/develop`). Open the pull request
  against `develop`.
- Keep a pull request focused on one change. Describe what changed and why, and how you checked it.
- `make check` and `make smoke` must pass. Add or update tests for any change in behavior.
- Every commit must be signed off (see below).

## Developer Certificate of Origin (DCO)

SATSUMAESTRO uses the [Developer Certificate of Origin](https://developercertificate.org/) instead of a
contributor license agreement. By signing off a commit, you certify that you wrote the change or
otherwise have the right to submit it under the project's license.

Sign off each commit with `-s`:

```bash
git commit -s -m "Fix the context window hint for small models"
```

This adds a line like the following to the commit message. It must match the commit's author.

```
Signed-off-by: Your Name <you@example.com>
```

If you forgot, add the sign-off to existing commits and force-push your branch:

```bash
git commit --amend -s --no-edit        # the last commit
git rebase --signoff upstream/develop  # every commit on your branch
git push --force-with-lease
```

A DCO check runs on every pull request, and pull requests without a sign-off on every commit cannot be
merged.

## License of contributions

SATSUMAESTRO is licensed under the [Apache License 2.0](LICENSE). Unless you state otherwise, any
contribution you submit is licensed under the same terms (section 5 of the license). Do not submit
code or data you cannot license this way. If you add third-party material (for example test fixtures
or evaluation tasks), keep its license and attribution and add it to [NOTICE](NOTICE).

## Code style

- **Formatting and lint**: ruff with a line length of 100. ruff counts full-width (CJK) characters as
  width 2, so a Japanese comment line fills up sooner than it looks.
- **Types**: mypy in strict mode over `satsumaestro/`.
- **Comments**: most existing comments and docstrings are in Japanese. Keep them as they are; please
  do not translate them in unrelated pull requests. New comments may be in English or Japanese.
- **No history in comments**: do not write issue or pull request numbers, dates or "changed in ..."
  notes in comments and docstrings. Explain why the code is the way it is; the history belongs in git
  and in the pull request.
- **Text the model reads** (system prompt, tool descriptions, tool results and errors returned to the
  model) is written in English.
- **Text shown on screen** goes through the message catalogs. Add each new key to both
  `satsumaestro/i18n/messages/en.toml` and `satsumaestro/i18n/messages/ja.toml` and look it up with
  `ui("section.key")`. A test fails if a key is missing from either file. If you cannot write one of
  the two languages, say so in the pull request and we will help.

## Evaluations

Unit tests cannot tell whether a model still does the right thing. If your change affects what the
model sees or how the agent loop behaves (the system prompt, tool definitions or descriptions, tool
results, planning, context handling), run the evaluations before and after the change with the same
model and put both numbers in the pull request.

The evaluations call a real model, so you need Ollama (or another provider) and a model that supports
tool calling. Pass `--trust` when you run them in your clone: without it, an untrusted directory does
not read its `.env` and `.satsuma/settings*.json`, and the before and after runs may use different
settings. `--trust` trusts the directory for that run only and does not record anything.

```bash
# Code analysis (8 tasks)
uv run satsumaestro eval evals/code_analysis --model qwen3:8b --trust

# Canary (17 tasks), in Japanese (the default) and in English
uv run satsumaestro eval evals/canary/canary.toml --model qwen3:8b --trust
uv run satsumaestro eval evals/canary/canary.toml --model qwen3:8b --trust --prompt-lang en

# Plan generation: plan mode, approval, then the steps (writes a dashboard of 200+ lines)
uv run satsumaestro eval evals/file_generation/dashboard-html.toml --agent "satsuma:qwen3:8b+plan" --trust
```

- `--runs N` repeats the same evaluation; results vary between runs, so compare more than one run
  when the difference is small.
- Reports are written under `.satsuma/evals/` (or to `--out`). Do not commit them.
- `uv run satsumaestro eval --help` lists the other options (`--only`, `--baseline`, `--agent` for
  comparing other CLIs, and so on).

### Aider Polyglot

The comparison in the README uses all 225 tasks of
[Aider Polyglot](https://github.com/Aider-AI/polyglot-benchmark) (C++, Go, Java, JavaScript, Python and
Rust), graded by the unit tests that come with each task. The task list is in
`evals/official/aider-polyglot/ai.toml`; the exercises themselves are fetched at a pinned revision.

```bash
# Fetch the exercises into .satsuma/benchmarks/ (not tracked by git) and rebuild the task list
uv run python scripts/official/catalog.py --source aider-polyglot

# Check the graders without a model: reference solutions must all pass, empty ones must all fail
uv run python scripts/official/run.py aider-polyglot --smoke --dry-run reference
uv run python scripts/official/run.py aider-polyglot --smoke --dry-run empty

# A small run, then the full run used in the README
uv run python scripts/official/run.py aider-polyglot --smoke --model gpt-oss:120b
uv run python scripts/official/run.py aider-polyglot --priority all --agent satsuma \
  --model gpt-oss:120b --reasoning-effort medium
```

- `--priority high` (the default) covers Python, JavaScript and Rust, which need only Python,
  Node.js and cargo. `low` covers C++ (cmake), Go and Java 21 (the tasks' Gradle does not run on
  Java 22 or later). `all` runs both.
- `--agent aider` or `--agent opencode` runs the same tasks with another harness and the same model.
  Install them first: `uv tool install aider-chat` for aider, and [OpenCode](https://opencode.ai)
  with its installer.
- Results go to `.satsuma/evals/official/`. Running the same command again skips tasks that already
  have a result, so an interrupted run continues where it stopped.
- The exercises come from Exercism (MIT). Do not commit results, and do not use the tasks or their
  outputs as training data.

## Trademark

Contributing does not grant any right to the SATSUMAESTRO name or logo. If you distribute a modified
version, please give it a different name. See [TRADEMARK.md](TRADEMARK.md).
