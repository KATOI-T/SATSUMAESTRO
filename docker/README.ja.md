# docker/ — 任意のサービス

[English](README.md)

SATSUMAESTRO 本体はコンテナなしで動きます。このディレクトリには、任意で立てるサービスの compose の
定義を置いています。いまは [SearXNG](https://github.com/searxng/searxng)（`web_search` ツールの検索先。
`web_search` は既定で無効）だけです。DuckDuckGo（非公式の HTML 版）は連続して使うと bot 判定で沈黙
しやすいので、エージェントで使うなら SearXNG をおすすめします。

## 前提

- Docker のデーモンが動いていること（Docker Desktop か、macOS なら [colima](https://github.com/abiosoft/colima)）
- `docker compose`（v2）か `docker-compose`。Makefile が使える方を選びます

## 起動・停止・確認

```bash
make searxng-up                    # 裏で起動（コンテナ名 satsuma-searxng）
uv run satsumaestro doctor --probe # web search の行が「疎通 OK」になれば OK
make searxng-down                  # 停止
```

`make` を使わないとき:

```bash
docker compose -f docker/searxng/compose.yaml up -d
docker compose -f docker/searxng/compose.yaml down
docker logs satsuma-searxng
```

SATSUMAESTRO 側は `.env` か設定に書きます（設定の値が優先）。

```bash
# .env
SATSUMA_WEB_SEARCH_PROVIDER=searxng
SATSUMA_WEB_SEARCH_BASE_URL=http://localhost:8080
```

```json
{"web_search": {"provider": "searxng", "base_url": "http://localhost:8080"}}
```

## ポートを変える

既定は `127.0.0.1:8080` です。8080 が使われていれば `SATSUMA_SEARXNG_PORT` で変え、SATSUMAESTRO 側も
同じポートにします。

```bash
make searxng-up SATSUMA_SEARXNG_PORT=8081
# .env: SATSUMA_WEB_SEARCH_BASE_URL=http://localhost:8081
```

すでに動かしている別の SearXNG を使うこともできます。SATSUMAESTRO は JSON API を使うので、その
サーバの `settings.yml` の `search.formats` に `json` が要ります（無いと 403 になり、`doctor --probe` が
知らせます）。

## イメージの更新

イメージは [searxng/compose.yaml](searxng/compose.yaml) で、確かめた版に固定しています。更新するときは、
[Docker Hub](https://hub.docker.com/r/searxng/searxng/tags) の新しいタグに書き換えてから
`make searxng-up` し、`doctor --probe` で確かめます。

## settings.yml で変えていること

[searxng/settings.yml](searxng/settings.yml) は SearXNG の既定（`use_default_settings: true`）を引き継ぎ、
次だけを変えています。

- `search.formats: [html, json]`: SATSUMAESTRO が使う JSON API を有効にする（既定は無効）
- `server.limiter: false`: 連続した検索を絞る仕組みを切る（手元で 1 人で使うなら邪魔になるだけ）
- `server.secret_key`: 手元だけで使うための仮の値

## 別のマシンから使う

先に README の[別のマシンの Ollama を使う](../README.ja.md#別のマシンの-ollama-を使う)を読んでください。
compose は `127.0.0.1` にしか bind しません。ほかのマシンから使うには
[searxng/compose.yaml](searxng/compose.yaml) の `ports` を変えます。

```yaml
    ports:
      - "0.0.0.0:${SATSUMA_SEARXNG_PORT:-8080}:8080"
```

そのときは必ず:

- `settings.yml` の `secret_key` をランダムな値にする（例: `openssl rand -hex 32`）
- limiter を有効に戻すことを考える
- 信頼できる LAN の中に限る。SearXNG に認証はありません。インターネットへ直接は公開しないでください
