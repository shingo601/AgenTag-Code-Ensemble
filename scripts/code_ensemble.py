"""Evaluate code-only encoder ensembles on repository-disjoint AIDev folds.

Run from the replication package with `python scripts/code_ensemble.py`.
MiniLM, CodeBERT, and GraphCodeBERT each predict from the same added lines.
The ensemble averages their five cosine similarity scores for each PR.
"""

import argparse
from datetime import datetime
from itertools import permutations
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
import torch
from transformers import AutoModel, AutoTokenizer, __version__ as transformers_version

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.common import AGENTS, DATA, RESULTS, SEED, added_code, cents, emb, train_head


ENCODERS = {
    "MiniLM": ("sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"),
    "CodeBERT": ("microsoft/codebert-base", "3b0952feddeffad0063f274080e3c23d75e7eb39"),
    "GraphCodeBERT": ("microsoft/graphcodebert-base", "2b0488a7bb0eefc7041f1bb2cad1ab26b0da269d"),
}
ENSEMBLE = "Ensemble"
CLASS_IDS = np.arange(len(AGENTS))
MAX_LENGTH = 256


def load_code_prs(max_per_agent):
    eligible = pd.read_csv(Path(RESULTS) / "extracted_features.csv", usecols=["pr_id"])
    meta = pd.read_parquet(
        Path(DATA) / "pull_request.parquet", columns=["id", "agent", "repo_id"]
    ).rename(columns={"id": "pr_id"})
    meta = meta.loc[meta["pr_id"].isin(eligible["pr_id"]) & meta["agent"].isin(AGENTS)]
    if meta["pr_id"].duplicated().any() or meta["repo_id"].isna().any():
        raise ValueError("PR metadata contains duplicate PR IDs or missing repository IDs.")
    sampled = pd.concat([
        rows.sample(min(len(rows), max_per_agent), random_state=SEED)
        for _, rows in meta.groupby("agent", sort=True)
    ], ignore_index=True)
    details = pd.read_parquet(
        Path(DATA) / "pr_commit_details.parquet",
        columns=["pr_id", "patch"],
        filters=[("pr_id", "in", sampled["pr_id"].tolist())],
    )
    code = details.groupby("pr_id", sort=False)["patch"].apply(added_code)
    sampled["code"] = sampled["pr_id"].map(code).fillna("")
    usable = sampled.loc[sampled["code"].str.strip().ne("")].reset_index(drop=True)
    coverage = pd.DataFrame({
        "sampled_prs": sampled.groupby("agent").size(),
        "evaluated_prs": usable.groupby("agent").size(),
    }).reindex(AGENTS).fillna(0).astype(int)
    return usable, coverage


def repository_folds(prs, y, n_splits):
    n_repos = prs.groupby("agent")["repo_id"].nunique().reindex(AGENTS, fill_value=0)
    if (n_repos < n_splits).any():
        raise ValueError(
            f"Each agent needs at least {n_splits} usable repositories. "
            "Increase --max-prs-per-agent or reduce --folds."
        )
    groups = prs["repo_id"].to_numpy()
    for split_seed in range(SEED, SEED + 32):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
        splits = list(cv.split(np.zeros(len(y)), y, groups))
        if all(np.unique(y[test]).size == len(AGENTS) for _, test in splits):
            return splits, split_seed
    raise ValueError("Could not construct repository folds containing all five agents.")


@torch.no_grad()
def encode(name, texts, device, batch_size):
    model_id, revision = ENCODERS[name]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModel.from_pretrained(
        model_id, revision=revision, use_safetensors=(name == "MiniLM")
    ).to(device).eval()
    parts = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        if name == "MiniLM":
            # Match SentenceTransformer's whitespace handling in the original RQ1.
            batch_texts = [text.strip() for text in batch_texts]
        tokens = tokenizer(
            batch_texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        ).to(device)
        hidden = model(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        parts.append(torch.nn.functional.normalize(pooled, dim=1).cpu().numpy())
        done = min(start + batch_size, len(texts))
        if (start // batch_size + 1) % 50 == 0 or done == len(texts):
            print(f"  Encoded {done:,}/{len(texts):,} PRs", flush=True)
    return np.concatenate(parts)


def score_folds(features, y, splits, head_steps):
    scores = np.empty((len(y), len(AGENTS)), dtype=np.float32)
    for fold, (train, test) in enumerate(splits, start=1):
        scaler = StandardScaler().fit(features[train])
        train_features = scaler.transform(features[train])
        test_features = scaler.transform(features[test])
        head = train_head(train_features, y[train], steps=head_steps)
        prototypes = cents(emb(head, train_features), y[train], CLASS_IDS)
        scores[test] = emb(head, test_features) @ prototypes.T
        print(f"  Scored fold {fold}/{len(splits)} ({len(test):,} test PRs)", flush=True)
    return scores


def f1_metrics(y, prediction):
    per_agent = f1_score(y, prediction, labels=CLASS_IDS, average=None, zero_division=0)
    counts = np.bincount(y, minlength=len(AGENTS))
    return {
        "macro_f1": per_agent.mean(),
        "weighted_f1": np.average(per_agent, weights=counts),
        **{f"f1_{agent}": score for agent, score in zip(AGENTS, per_agent)},
    }


def error_overlap(y, predictions):
    pairs = list(permutations(predictions, 2))
    populations = {"all": np.ones(len(y), dtype=bool)}
    populations.update({agent: y == label for label, agent in enumerate(AGENTS)})
    rows = []
    for scope, select in populations.items():
        for reference, other in pairs:
            ref_wrong = select & (predictions[reference] != y)
            recovered = ref_wrong & (predictions[other] == y)
            n_wrong = int(ref_wrong.sum())
            n_recovered = int(recovered.sum())
            rows.append({
                "true_agent": scope, "test_prs": int(select.sum()),
                "reference": reference, "other": other,
                "reference_wrong": n_wrong, "other_correct_when_reference_wrong": n_recovered,
                "both_wrong": n_wrong - n_recovered,
                "complement_rate": n_recovered / n_wrong if n_wrong else np.nan,
            })
    return pd.DataFrame(rows)


def macro_differences(y, repositories, predictions, repeats):
    # Precompute confusion matrices so each draw resamples whole repositories
    # identically for all methods. Models and held-out predictions stay fixed.
    repo_indices, repo_ids = pd.factorize(repositories)
    n_repos = len(repo_ids)
    n_classes = len(AGENTS)
    methods = [*ENCODERS, ENSEMBLE]
    matrices = np.zeros((n_repos, len(methods), n_classes, n_classes), dtype=np.int64)
    for index, name in enumerate(methods):
        np.add.at(matrices[:, index], (repo_indices, y, predictions[name]), 1)
    flat_matrices = matrices.reshape(n_repos, -1)

    rng = np.random.default_rng(SEED)
    deltas = np.empty((repeats, len(ENCODERS)))
    for repeat in range(repeats):
        counts = np.bincount(rng.integers(n_repos, size=n_repos), minlength=n_repos)
        totals = (counts @ flat_matrices).reshape(matrices.shape[1:])
        twice_tp = 2 * np.diagonal(totals, axis1=1, axis2=2)
        support = totals.sum(axis=1) + totals.sum(axis=2)
        f1s = np.divide(twice_tp, support, out=np.zeros_like(support, dtype=float), where=support > 0)
        macro = f1s.mean(axis=1)
        deltas[repeat] = macro[-1] - macro[:-1]

    lower, upper = np.percentile(deltas, [2.5, 97.5], axis=0)
    combined = predictions[ENSEMBLE] == y
    ensemble_f1 = f1_metrics(y, predictions[ENSEMBLE])["macro_f1"]
    return pd.DataFrame([
        {
            "baseline": name,
            "delta_macro_f1": ensemble_f1 - f1_metrics(y, predictions[name])["macro_f1"],
            "ci95_lower": lower[i], "ci95_upper": upper[i],
            "ensemble_recovers": int((combined & (predictions[name] != y)).sum()),
            "ensemble_loses": int((~combined & (predictions[name] == y)).sum()),
        }
        for i, name in enumerate(ENCODERS)
    ])


def save_results(output_dir, prs, y, folds, scores, coverage, settings, repeats):
    predictions = {name: values.argmax(axis=1) for name, values in scores.items()}
    classes = np.asarray(AGENTS)
    results = prs[["pr_id", "repo_id", "agent"]].assign(fold=folds)
    for name, prediction in predictions.items():
        results[f"{name}_prediction"] = classes[prediction]
        for label, agent in enumerate(AGENTS):
            results[f"{name}_score_{agent}"] = scores[name][:, label]
    results["n_single_models_correct"] = np.sum([predictions[name] == y for name in ENCODERS], axis=0)

    summary = pd.DataFrame([
        {"method": name, "test_prs": len(y), **f1_metrics(y, prediction)}
        for name, prediction in predictions.items()
    ])
    fold_rows = []
    for fold in np.unique(folds):
        select = folds == fold
        for name, prediction in predictions.items():
            fold_rows.append({
                "fold": fold, "method": name, "test_prs": int(select.sum()),
                **f1_metrics(y[select], prediction[select]),
            })
    deltas = macro_differences(y, prs["repo_id"], predictions, repeats)
    overlap = error_overlap(y, predictions)

    output_dir.mkdir(parents=True, exist_ok=False)
    for filename, frame in {
        "sample_coverage.csv": coverage.reset_index(names="agent"),
        "f1_summary.csv": summary,
        "f1_by_fold.csv": pd.DataFrame(fold_rows),
        "macro_f1_differences.csv": deltas,
        "error_overlap.csv": overlap,
        "pr_predictions.csv": results,
    }.items():
        frame.to_csv(output_dir / filename, index=False)
    (output_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    print("\nPooled held-out results", flush=True)
    print(summary.to_string(index=False, float_format="{:.4f}".format))
    print("\nEnsemble versus each individual model")
    print(deltas.to_string(index=False, float_format="{:.4f}".format))
    print("\nPR counts by number of individual models correct")
    print(results["n_single_models_correct"].value_counts().reindex(range(4), fill_value=0).to_string())
    print(f"\nSaved to {output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-prs-per-agent", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-steps", type=int, default=300)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, help="A new directory for this run's results")
    args = parser.parse_args()
    if args.folds < 2 or min(args.max_prs_per_agent, args.batch_size, args.head_steps, args.bootstrap_repeats) < 1:
        parser.error("Use at least two folds and positive values for the sample, batch, training and bootstrap sizes.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Use --device cpu or a GPU runtime.")
    use_cuda = args.device != "cpu" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    output_dir = args.output_dir or Path(RESULTS) / "code_ensemble" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if output_dir.exists():
        parser.error(f"Choose a new output directory: {output_dir}")

    prs, coverage = load_code_prs(args.max_prs_per_agent)
    print("PR coverage (empty added-line inputs are excluded)\n" + coverage.to_string(), flush=True)
    y = pd.Categorical(prs["agent"], categories=AGENTS).codes.astype(np.int64)
    splits, split_seed = repository_folds(prs, y, args.folds)
    fold_ids = np.empty(len(y), dtype=int)
    for fold, (_, test) in enumerate(splits, start=1):
        fold_ids[test] = fold
    print(f"Using {len(splits)} repository folds (seed={split_seed}), encoder device={device}", flush=True)

    texts = prs["code"].tolist()
    scores = {}
    for name in ENCODERS:
        print(f"\n{name}", flush=True)
        features = encode(name, texts, device, args.batch_size)
        scores[name] = score_folds(features, y, splits, args.head_steps)
        del features
    scores[ENSEMBLE] = np.mean(list(scores.values()), axis=0)

    settings = {
        "encoders": ENCODERS,
        "ensemble": "equal-weight mean of class cosine similarity scores",
        "device": str(device), "seed": SEED, "split_seed": split_seed, "folds": args.folds,
        "max_prs_per_agent": args.max_prs_per_agent, "max_tokens": MAX_LENGTH,
        "batch_size": args.batch_size, "head_steps": args.head_steps,
        "repository_bootstrap_repeats": args.bootstrap_repeats,
        "versions": {"sklearn": sklearn.__version__, "torch": torch.__version__, "transformers": transformers_version},
    }
    save_results(output_dir, prs, y, fold_ids, scores, coverage, settings, args.bootstrap_repeats)


if __name__ == "__main__":
    main()
