# Aider Polyglot

練習問題を、題に付いた単体テストが通るまで直せるかを測る。

- 出典: [Aider-AI/polyglot-benchmark](https://github.com/Aider-AI/polyglot-benchmark) @ `7e0611e7`。225 題（C++・Go・Java・JavaScript・Python・Rust）。題の中身はすべて Exercism のもの（MIT・Copyright (c) 2021 Exercism）。リポジトリ自体にライセンスのファイルは無い
- 採点: 題に付いた単体テスト。1 回目に落ちたら、テストの出力を見せてもう 1 回直させる（2 回まで）
- 優先度: 高は道具を足さずに回せる言語（Python・JavaScript・Rust）。低は道具が要る言語（C++ の cmake・Go・Java 21。Java の題の Gradle 8.7 は Java 22 以降で動かない）。公式の値と比べるときは `--priority all` で全題を回す
- 参考の値（全題）: gpt-oss-120b 44.4%・gpt-oss-20b 34.2%（OpenAI の公表値）。gpt-oss-120b 41.8%（[aider の順位表](https://aider.chat/docs/leaderboards/)・diff・reasoning high）
- 回し方: `uv run python scripts/official/run.py aider-polyglot --smoke`。手順と文は公式の [aider の benchmark](https://github.com/Aider-AI/aider/tree/5dc9490bb35f9729ef2c95d00a19ccd30c26339c/benchmark) に合わせた。`javascript/ledger` は書き直しの題で、手を加えなくても通る
- ハーネスの比較: `--agent aider|opencode`。aider は公式の benchmark と同じ作りで Coder を作り（`scripts/official/aider_runner.py`）、窓はサーバの既定に固定する
