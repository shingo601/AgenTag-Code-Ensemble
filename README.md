# AgenTag Code Ensemble

This repository contains the code-only ensemble evaluation script and the
shared source required to run it.

## Contents

- `scripts/code_ensemble.py`: evaluates MiniLM, CodeBERT, and GraphCodeBERT
  ensembles on repository-disjoint AIDev folds.
- `shared/common.py`: shared agents, data paths, embedding helpers, and
  contrastive-head training code.
- `requirements.txt`: pinned Python dependencies.

## Inputs

Create the following paths before running the script:

- `data/pull_request.parquet`
- `data/pr_commit_details.parquet`
- `results/extracted_features.csv`

These datasets are not included because they are large. Supply them as local
inputs or obtain them from the original replication package.

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
