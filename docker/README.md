# docker/ — optional services

[日本語](README.ja.md)

SATSUMAESTRO itself needs no container. This directory has compose files for optional services.
Today that is only [SearXNG](https://github.com/searxng/searxng), a self-hosted search backend for
the `web_search` tool (off by default). DuckDuckGo's unofficial HTML page tends to go silent under
bot detection after many requests, so SearXNG is recommended for agent use.

## Requirements

- A running Docker daemon (Docker Desktop, or [colima](https://github.com/abiosoft/colima) on macOS).
- `docker compose` (v2) or `docker-compose`. The Makefile picks whichever is available.

## Start, stop and check

```bash
make searxng-up                    # start in the background (container satsuma-searxng)
uv run satsumaestro doctor --probe # the "web search" line should say reachable
make searxng-down                  # stop
```

Without make:

```bash
docker compose -f docker/searxng/compose.yaml up -d
docker compose -f docker/searxng/compose.yaml down
docker logs satsuma-searxng
```

Then point SATSUMAESTRO at it, in `.env` or in the settings (the settings win):

```bash
# .env
SATSUMA_WEB_SEARCH_PROVIDER=searxng
SATSUMA_WEB_SEARCH_BASE_URL=http://localhost:8080
```

```json
{"web_search": {"provider": "searxng", "base_url": "http://localhost:8080"}}
```

## Change the port

The default is `127.0.0.1:8080`. If 8080 is taken, set `SATSUMA_SEARXNG_PORT` and use the same port
in SATSUMAESTRO:

```bash
make searxng-up SATSUMA_SEARXNG_PORT=8081
# .env: SATSUMA_WEB_SEARCH_BASE_URL=http://localhost:8081
```

You can also use another SearXNG you already run. SATSUMAESTRO uses its JSON API, so `json` must be
in `search.formats` of that server's `settings.yml` (otherwise it returns 403, and
`doctor --probe` tells you so).

## Update the image

The image is pinned to a tested version in [searxng/compose.yaml](searxng/compose.yaml). To update,
change the tag to a newer one from [Docker Hub](https://hub.docker.com/r/searxng/searxng/tags), then
run `make searxng-up` and check with `doctor --probe`.

## What settings.yml changes

[searxng/settings.yml](searxng/settings.yml) keeps SearXNG's defaults (`use_default_settings: true`)
and changes only:

- `search.formats: [html, json]` turns on the JSON API that SATSUMAESTRO uses (off by default).
- `server.limiter: false` turns off the rate limiter, which only gets in the way of one local user.
- `server.secret_key` is a placeholder for local use only.

## Using it from another machine

Read [Using Ollama on another machine](../README.md#using-ollama-on-another-machine) first. The
compose file binds to `127.0.0.1` only. To accept other machines, change `ports` in
[searxng/compose.yaml](searxng/compose.yaml):

```yaml
    ports:
      - "0.0.0.0:${SATSUMA_SEARXNG_PORT:-8080}:8080"
```

When you do:

- Set `secret_key` in `settings.yml` to a random value (for example `openssl rand -hex 32`).
- Consider turning the limiter back on.
- Keep it inside a trusted LAN. SearXNG has no authentication; never expose it directly to the
  internet.
