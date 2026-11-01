# 公式の評価から持ってきた題

広く使われている公開の評価の題を、評価ごとのフォルダに置く。自前の題（`evals/<スイート>/` と `evals/manual/`）とは分ける。

| フォルダ | 題 | 高・低 |
|---|---|---|
| [aider-polyglot](aider-polyglot/README.md) | 225 | 113・112 |

- 1 つの評価に置くのは、`README.md` と `ai.toml`（AI が採点する題）。
- `ai.toml` は 1 題が `[[tasks]]` の 1 つで、`prompt`（指示文）・`expected`（期待結果）・`priority`（`high`・`low`）と、回すのに要る値を持つ。
  `scripts/official/catalog.py` が元のデータ（`.satsuma/benchmarks/`）から作るので、手で直さない。
- 回し方: `uv run python scripts/official/run.py <フォルダ> --smoke`（選び方は `--priority`・`--ids`・`--limit`。同じコマンドで続きから回る）。
- 公式の題は、公式の英語の文のまま回す。採点は公式の仕組みを使い、部分集合の値を公式の全体の値と並べない。
- 公式の題とその出力は、モデルの学習データに入れない。結果は git に入れない。
