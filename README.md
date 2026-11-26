# AgenTag アンサンブル学習

先行研究（AgenTag）のCode onlyに対し、出力側にアンサンブル学習機を加えたものです
https://arxiv.org/html/2608.00966v1
コードに対してのアンサンブル評価のスクリプトと、その実行に必要な共通ソースコードが含まれてます。

##作成した目的

先行研究のRQ1では、code onlyの識別性能がPR本文やコミットメッセージを用いた場合より低い。そこで本実験ではMiniLM・CodeBERT・GraphCodeBERTの3も出rの予測を組み合わせて、各モデル単体で生じる誤分類を補い、識別性能を改善できるかを検証した。
実行した結果としては先行研究の重み付きF1スコアは0.558に対して、アンサンブル学習機をかけたものでは0.645のスコアまで上げることが確認できた。


## スクリプト

- `scripts/code_ensemble.py`: AIDevデータセットを用いて、MiniLM、CodeBERT、GraphCodeBERTのアンサンブルを評価
- `shared/common.py`: データパス、埋め込み生成用の補助関数、対照学習ヘッドの学習コードが含まれています。
- `requirements.txt`: pythonの依存ライブラリの一覧

## 入力データ

- `data/pull_request.parquet`
- `data/pr_commit_details.parquet`
- `results/extracted_features.csv`

データセットはファイルサージが大きいため、リポジトリに登録しませんでした。
ローカル環境に入力データを用意するか、また元の研究のReplication Package：https://figshare.com/s/6c4fb60ba3f5abf1ffdf?
から習得してください。

<!--

## Run

The environment used for the study is Python 3.13.

```bash
pip install -r requirements.txt
python scripts/code_ensemble.py
```

Useful options:

```bash
python scripts/code_ensemble.py --help
python scripts/code_ensemble.py --device cpu --folds 5 --batch-size 32
python scripts/code_ensemble.py --output-dir results/code_ensemble/run
```

The models are downloaded from Hugging Face on first use. Results are written
to `results/code_ensemble/<timestamp>/` by default.

-->
