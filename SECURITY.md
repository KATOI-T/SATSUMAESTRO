# Security policy

## Supported versions

Security fixes are made for the latest release only.

| Version | Supported |
|---|---|
| 0.5.x (latest release) | Yes |
| Older releases | No |

## Reporting a vulnerability

**Please do not report security issues in public issues, pull requests or discussions.**

Report them privately through GitHub's private vulnerability reporting:
<https://github.com/KATOI-T/SATSUMAESTRO/security/advisories/new> (the **Report a vulnerability**
button on the [Security tab](https://github.com/KATOI-T/SATSUMAESTRO/security)). Only the maintainers
can read the report. English and Japanese are both fine.

Please include as much of the following as you can:

- The version (`satsumaestro --version`), your OS, and how you installed it
- The provider and model, if the issue depends on the model's behavior
- The settings involved (permission mode, `.satsuma/policy.md`, trust state); remove any secrets
- Steps to reproduce, and what an attacker can achieve
- A proof of concept, if you have one

## What to expect

- We try to acknowledge a report within a week. SATSUMAESTRO is maintained by a small team, so a fix
  may take longer; we will keep you updated in the advisory.
- We will confirm the issue, prepare a fix and release it, then publish a GitHub security advisory.
  Please keep the details private until the advisory is published.
- We are happy to credit you in the advisory. Tell us the name you would like to use, or that you
  prefer not to be named.

## Scope

SATSUMAESTRO runs a language model that can read files, edit files and run commands on your machine.
These are the protections we treat as security boundaries. A way around any of them is a
vulnerability:

- **Permission checks**: a tool runs without the confirmation or denial that the permission mode,
  `.satsuma/policy.md` (Deny / Ask / Allow) or the secret-file rules require
- **Workspace limits**: the file tools read or write outside the workspace, or a redirect in bash
  writes outside it
- **Secret files**: `.env`, keys and similar files are read, searched or indexed without the
  confirmation described in the README
- **Trust**: settings, hooks, MCP servers, LSP servers, `.env` or the policy of an untrusted directory
  are loaded or run
- **Keys and records**: provider API keys reach child processes (bash, hooks, LSP servers), or keys
  and tokens are written unredacted to session records
- **Web fetch**: `web_fetch` reaches a private, loopback or cloud metadata address while
  `allow_private_hosts` is off (including through redirects), or runs without being enabled
- **Self-invocation**: the agent starts another copy of itself from bash despite the guard

## Out of scope

These are documented limits or intended behavior, not vulnerabilities:

- **No OS-level sandbox for bash**: bash commands run with your user's privileges and are not
  isolated with namespaces or seccomp. Command validation is a warning layer, and network blocking
  for bash is best effort. The workspace limits and the permission policy are the last line of
  defense (see "Permissions and safety" in the README).
- **Actions you approved or allowed**: anything that runs after you answer `y` or `a` on a
  confirmation card, or because you chose `danger-full-access`, `allow`, an Allow rule, or
  `web_fetch` with `confirm: never` or `allow_private_hosts: true`.
- **Trusted directories**: hooks, MCP servers and LSP servers configured in a directory you trusted
  run with your privileges by design. Trust only directories whose contents you trust.
- **Model behavior**: wrong, unsafe or misleading output, or a prompt injection that leads the model
  to *ask* for a harmful action. It is in scope only if it gets an action past a check listed above.
- **Services you expose**: Ollama and SearXNG have no authentication. Exposing them on a network
  is your choice (see "Using Ollama on another machine" in the README).
- **Third-party dependencies**: report these to their projects. Tell us as well if SATSUMAESTRO's
  use of a dependency makes the issue exploitable.
