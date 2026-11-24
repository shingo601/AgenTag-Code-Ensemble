"""Evaluate code-only ensembles with repository or AgenTag reproduction folds.

Run from the replication package with `python scripts/code_ensemble.py`.
MiniLM, CodeBERT, and GraphCodeBERT each predict from the same added lines.
The ensemble averages their five cosine similarity scores for each PR.

--mode repository_disjoint (default) preserves the original experiment.
--mode legacy_reproduction uses cached bidx/code_bal, 5x5 stratified CV and
    standardization fitted on all PRs, matching shared.common.closed_ci.
--mode leakage_free uses the same cached PRs and CV but fits each scaler
    on the training fold only.
"""

import argparse
from datetime import datetime
import hashlib
from itertools import permutations
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
import torch
from transformers import AutoModel, AutoTokenizer, __version__ as transformers_version

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.common import AGENTS, DATA, N_REP, RESULTS, SEED, added_code, cents, ci, emb, train_head


ENCODERS = {
    "MiniLM": ("sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"),
    "CodeBERT": ("microsoft/codebert-base", "3b0952feddeffad0063f274080e3c23d75e7eb39"),
    "GraphCodeBERT": ("microsoft/graphcodebert-base", "2b0488a7bb0eefc7041f1bb2cad1ab26b0da269d"),
}
ENSEMBLE = "Ensemble"
CLASS_IDS = np.arange(len(AGENTS))
MAX_LENGTH = 256
DIMENSIONS = {"MiniLM": 384, "CodeBERT": 768, "GraphCodeBERT": 768}


def require_files(paths):
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Required inputs are missing:\n" + "\n".join(missing))


def load_code_prs(max_per_agent):
    require_files([
        Path(RESULTS) / "extracted_features.csv",
        Path(DATA) / "pull_request.parquet",
        Path(DATA) / "pr_commit_details.parquet",
    ])
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


def load_reproduction_prs(cache_path):
    """Reconstruct run_code's PR order and reuse build_cache's MiniLM vectors."""
    if not Path(cache_path).is_file():
        raise FileNotFoundError(
            f"Reproduction cache not found: {cache_path}. "
            "Pass --cache with the original repro_emb.npz containing bidx, agent and code_bal."
        )
    require_files([
        cache_path, Path(RESULTS) / "extracted_features.csv",
        Path(DATA) / "pull_request.parquet", Path(DATA) / "pr_commit_details.parquet",
    ])
    with np.load(cache_path, allow_pickle=True) as cache:
        missing = {"bidx", "agent", "code_bal"} - set(cache.files)
        if missing:
            raise ValueError(f"{cache_path} is missing cache keys: {sorted(missing)}")
        bidx = cache["bidx"]
        cached_agents = cache["agent"]
        minilm = cache["code_bal"].copy()
        cached_ids = cache["pr_id"] if "pr_id" in cache.files else None

    eligible_ids = pd.read_csv(Path(RESULTS) / "extracted_features.csv", usecols=["pr_id"])
    meta = pd.read_parquet(
        Path(DATA) / "pull_request.parquet", columns=["id", "agent", "repo_id"]
    )
    eligible = meta[meta["id"].isin(eligible_ids["pr_id"])].reset_index(drop=True)
    if eligible["id"].duplicated().any() or set(eligible["agent"]) != set(AGENTS):
        raise ValueError("Eligible metadata must have unique PR IDs and exactly the five AgenTag agents.")
    if not np.array_equal(eligible["agent"].to_numpy(), cached_agents):
        raise ValueError("Cache agent order does not match eligible metadata; check the data/cache snapshot.")
    if bidx.ndim != 1 or bidx.dtype.kind not in "iu":
        raise ValueError("Cache bidx must be a one-dimensional integer array.")
    if not len(bidx) or len(np.unique(bidx)) != len(bidx) or bidx.min() < 0 or bidx.max() >= len(eligible):
        raise ValueError("Cache bidx contains duplicate or out-of-range indices.")
    # Equivalent to build_cache's sorted groupby(...).apply(sample).index.
    expected_bidx = np.concatenate([
        group.sample(min(len(group), 2500), random_state=SEED).index.to_numpy()
        for _, group in eligible.groupby("agent", sort=True)
    ])
    if not np.array_equal(bidx, expected_bidx):
        raise ValueError("Cache bidx does not match build_cache's seed-42, cap-2500 selection and order.")
    selected = eligible.iloc[bidx].rename(columns={"id": "pr_id"}).reset_index(drop=True)
    if not np.array_equal(selected["agent"].to_numpy(), cached_agents[bidx]):
        raise ValueError("Selected PR labels do not match the cache.")
    if minilm.shape != (len(selected), DIMENSIONS["MiniLM"]) or not np.isfinite(minilm).all():
        raise ValueError(f"Cache code_bal must be finite with shape ({len(selected)}, 384).")
    if not np.allclose(np.linalg.norm(minilm, axis=1), 1, atol=1e-3):
        raise ValueError("Cache code_bal must contain the original L2-normalized MiniLM embeddings.")

    id_check = None
    if cached_ids is not None:
        if cached_ids.shape == (len(eligible),):
            cached_ids = cached_ids[bidx]
        if cached_ids.shape != (len(selected),) or not np.array_equal(cached_ids, selected["pr_id"].to_numpy()):
            raise ValueError("Cache pr_id does not match the selected PR IDs in order.")
        id_check = True

    details = pd.read_parquet(
        Path(DATA) / "pr_commit_details.parquet", columns=["pr_id", "patch"],
        filters=[("pr_id", "in", selected["pr_id"].tolist())],
    )
    code = details.groupby("pr_id")["patch"].apply(added_code).to_dict()
    selected["code"] = [code.get(pr_id, "") for pr_id in selected["pr_id"]]
    selected["eligible_index"] = bidx
    counts = selected.groupby("agent").size().reindex(AGENTS)
    coverage = pd.DataFrame({"sampled_prs": counts, "evaluated_prs": counts})
    validation = {
        "cache": str(Path(cache_path).resolve()),
        "n_eligible": len(eligible), "n_selected": len(selected),
        "expected_pr_count": 9498, "matches_expected_pr_count": len(selected) == 9498,
        "agent_order_matches_cache": True,
        "bidx_matches_build_cache_selection_and_order": True,
        "selection_matches_run_code": True,
        "cache_contains_pr_ids": cached_ids is not None,
        "historical_pr_ids_independently_verified": id_check,
        "pr_id_sha256": hashlib.sha256(selected["pr_id"].to_numpy(dtype="<i8").tobytes()).hexdigest(),
        "code_bal_sha256": hashlib.sha256(minilm.tobytes()).hexdigest(),
        "empty_added_code_prs": int(selected["code"].str.strip().eq("").sum()),
        "verification_scope": (
            "PR IDs also checked against the cache."
            if id_check else
            "Original build_cache omits PR IDs. Selection matches run_code for the supplied "
            "metadata, and cached labels/bidx match build_cache. Historical ID order within "
            "an agent cannot be independently proven without the original PR ID manifest."
        ),
    }
    return selected, coverage, minilm, validation


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
    ).to(device).requires_grad_(False).eval()
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


def score_repeated_folds(features, y, splits, mode):
    """Keep a separate score matrix per split; each PR is tested five times."""
    global_features = None
    if mode == "legacy_reproduction":
        global_features = StandardScaler().fit_transform(features)
    scores = []
    for index, (train, test) in enumerate(splits, start=1):
        if global_features is None:
            scaler = StandardScaler().fit(features[train])
            train_features = scaler.transform(features[train])
            test_features = scaler.transform(features[test])
        else:
            train_features, test_features = global_features[train], global_features[test]
        head = train_head(train_features, pd.factorize(y[train])[0])
        prototypes = cents(emb(head, train_features), y[train], CLASS_IDS)
        scores.append(emb(head, test_features) @ prototypes.T)
        print(f"  Scored split {index}/{len(splits)} ({len(test):,} test PRs)", flush=True)
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


def summarize_folds(frame, group_column, metrics):
    rows = []
    for group, values in frame.groupby(group_column, sort=False):
        row = {group_column: group, "n_splits": len(values)}
        for metric in metrics:
            row[f"{metric}_mean"], row[f"{metric}_ci95_halfwidth"] = ci(values[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def reproduction_tables(prs, y, splits, all_scores, mode):
    fold_rows, difference_rows, prediction_rows, overlaps = [], [], [], []
    for index, (train, test) in enumerate(splits):
        split_info = {"mode": mode, "split": index + 1, "repeat": index // 5 + 1, "fold": index % 5 + 1}
        scores = {name: all_scores[name][index] for name in ENCODERS}
        if any(values.shape != (len(test), len(AGENTS)) for values in scores.values()):
            raise ValueError(f"Encoder scores have inconsistent test rows in split {index + 1}.")
        # Average only these three matrices, all indexed by this split's test PRs.
        scores[ENSEMBLE] = np.mean(list(scores.values()), axis=0)
        predictions = {name: values.argmax(axis=1) for name, values in scores.items()}
        fold_metrics = {}
        rows = prs.iloc[test][["pr_id", "repo_id", "agent"]].copy()
        for key, value in split_info.items():
            rows[key] = value
        rows["pr_index"] = test
        for name, prediction in predictions.items():
            fold_metrics[name] = f1_metrics(y[test], prediction)
            fold_rows.append({
                **split_info, "method": name, "train_prs": len(train), "test_prs": len(test),
                **fold_metrics[name],
            })
            rows[f"{name}_prediction"] = np.asarray(AGENTS)[prediction]
            for label, agent in enumerate(AGENTS):
                rows[f"{name}_score_{agent}"] = scores[name][:, label]
        rows["n_single_models_correct"] = np.sum([predictions[name] == y[test] for name in ENCODERS], axis=0)
        prediction_rows.append(rows)
        ensemble_correct = predictions[ENSEMBLE] == y[test]
        for name in ENCODERS:
            difference_rows.append({
                **split_info, "baseline": name,
                "delta_weighted_f1": fold_metrics[ENSEMBLE]["weighted_f1"] - fold_metrics[name]["weighted_f1"],
                "delta_macro_f1": fold_metrics[ENSEMBLE]["macro_f1"] - fold_metrics[name]["macro_f1"],
                "ensemble_recovers": int((ensemble_correct & (predictions[name] != y[test])).sum()),
                "ensemble_loses": int((~ensemble_correct & (predictions[name] == y[test])).sum()),
            })
        overlaps.append(error_overlap(y[test], predictions).assign(**split_info))

    fold_table = pd.DataFrame(fold_rows)
    differences = pd.DataFrame(difference_rows)
    metrics = ["weighted_f1", "macro_f1", *[f"f1_{agent}" for agent in AGENTS]]
    summary = summarize_folds(fold_table, "method", metrics)
    summary.insert(0, "mode", mode)
    summary["unique_prs"] = len(prs)
    difference_summary = summarize_folds(differences, "baseline", ["delta_weighted_f1", "delta_macro_f1"])
    difference_summary.insert(0, "mode", mode)
    return {
        "f1_summary.csv": summary, "f1_by_fold.csv": fold_table,
        "f1_differences.csv": difference_summary, "f1_differences_by_fold.csv": differences,
        "pr_predictions.csv": pd.concat(prediction_rows, ignore_index=True),
        "error_overlap.csv": pd.concat(overlaps, ignore_index=True),
    }


def read_reference(reference_path):
    require_files([reference_path])
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    reference = reference.get("code", reference)
    missing = set(ENCODERS) - set(reference)
    if missing:
        raise ValueError(f"Reference JSON must contain numeric Weighted F1 for {sorted(ENCODERS)}; missing {sorted(missing)}")
    for name in ENCODERS:
        expected = float(reference[name])
        if not np.isfinite(expected) or not 0 <= expected <= 1:
            raise ValueError(f"Invalid reference Weighted F1 for {name}: {expected}")
    return reference


def check_reference(args, fold_table, output_dir):
    reference_path = args.reference_results or Path(RESULTS) / "rq1.json"
    if not reference_path.is_file():
        return {"status": "not_checked", "reason": "No reference-results JSON was supplied or found."}
    reference = read_reference(reference_path)
    rows = []
    for name in ENCODERS:
        expected = float(reference[name])
        observed = float(fold_table.loc[fold_table["method"].eq(name), "weighted_f1"].mean())
        rows.append({
            "method": name, "reference_weighted_f1": expected,
            "observed_weighted_f1": observed, "difference": observed - expected,
            "tolerance": args.reference_tolerance,
            "within_tolerance": abs(observed - expected) <= args.reference_tolerance,
        })
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "reference_comparison.csv", index=False)
    print("\nLegacy single-encoder reference comparison")
    print(comparison.to_string(index=False, float_format="{:.6f}".format))
    return {
        "status": "matched" if comparison["within_tolerance"].all() else "mismatch",
        "reference_results": str(reference_path.resolve()),
        "tolerance": args.reference_tolerance,
        "follow_up_if_mismatched": [
            "Check pr_id_validation.json and the historical cache/data snapshot.",
            "Check that MiniLM used the original code_bal; its model revision is not recorded by build_cache.",
            "Compare encoder revisions with the original _hf_encode, which did not pin revisions.",
            "Compare CPU/CUDA numerical behavior, embedding batch size, package versions and CPU thread count.",
        ],
    }


def run_reproduction(args, output_dir, device, settings):
    prs, coverage, minilm, validation = load_reproduction_prs(args.cache)
    y = pd.Categorical(prs["agent"], categories=AGENTS).codes.astype(np.int64)
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=N_REP, random_state=SEED)
    splits = list(splitter.split(np.zeros(len(prs)), y))
    output_dir.mkdir(parents=True, exist_ok=False)
    selection = prs[["eligible_index", "pr_id", "repo_id", "agent"]].copy()
    selection.insert(0, "pr_index", np.arange(len(prs)))
    selection["empty_added_code"] = prs["code"].str.strip().eq("")
    selection.to_csv(output_dir / "pr_selection.csv", index=False)
    coverage.reset_index(names="agent").to_csv(output_dir / "sample_coverage.csv", index=False)
    (output_dir / "pr_id_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    fold_indices = {
        "mode": args.mode, "index_base": 0, "index_reference": "pr_selection.csv:pr_index",
        "splits": [
            {"split": i + 1, "repeat": i // 5 + 1, "fold": i % 5 + 1,
             "train": train.tolist(), "test": test.tolist()}
            for i, (train, test) in enumerate(splits)
        ],
    }
    (output_dir / "fold_indices.json").write_text(json.dumps(fold_indices), encoding="utf-8")
    settings.update({
        "cache": str(args.cache.resolve()), "pr_count": len(prs),
        "max_prs_per_agent": None, "repository_bootstrap_repeats": None,
        "agent_counts": prs["agent"].value_counts().reindex(AGENTS).to_dict(),
        "max_added_code_characters": 4000,
        "cpu_threads": torch.get_num_threads(),
        "source_sha256": {
            name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for name, path in {
                "code_ensemble.py": __file__,
                "shared/common.py": Path(__file__).resolve().parents[1] / "shared/common.py",
            }.items()
        },
        "cv": {"class": "RepeatedStratifiedKFold", "n_splits": 5, "n_repeats": N_REP, "random_state": SEED},
        "standardization": "all PRs before CV" if args.mode == "legacy_reproduction" else "training PRs within each split",
        "aggregation": "mean and 1.96*sample_std/sqrt(25), rounded to 3 decimals by shared.common.ci",
        "confidence_interval_scope": "Same fold-based interval as closed_ci; repeated folds overlap.",
        "contrastive_head": {
            "hidden_dim": 256, "output_dim": 128, "steps": 300, "batch_size": 512,
            "optimizer": "Adam", "learning_rate": 0.001, "temperature": 0.1, "seed": SEED,
            "device": "cpu", "implementation": "shared.common.train_head",
        },
        "encoders": {
            name: {
                "model": model_id, "revision": None if name == "MiniLM" else revision,
                "dimensions": DIMENSIONS[name], "frozen": True,
                "source": "cache code_bal" if name == "MiniLM" else "mean-pooled last_hidden_state, L2-normalized",
            }
            for name, (model_id, revision) in ENCODERS.items()
        },
        "reference_encoder_differences": [
            "MiniLM is copied exactly from code_bal; build_cache does not record its model revision.",
            "CodeBERT/GraphCodeBERT use the fixed revisions listed here; _hf_encode used unpinned main.",
            "Tokenization/pooling/normalization match _hf_encode; --device may use CUDA instead of reference CPU.",
            "MiniLM tokenization is inherited from the cache; its original tokenizer settings cannot be verified from NPZ.",
        ],
    })
    (output_dir / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print(f"\nMode: {args.mode}; {len(prs):,} PRs; {len(splits)} shared splits", flush=True)
    print(coverage.to_string(), flush=True)
    print(f"Empty added code retained: {validation['empty_added_code_prs']}", flush=True)
    print(validation["verification_scope"], flush=True)
    all_scores = {}
    for name in ENCODERS:
        print(f"\n{name}", flush=True)
        features = minilm if name == "MiniLM" else encode(name, prs["code"].tolist(), device, args.batch_size)
        if features.shape != (len(prs), DIMENSIONS[name]) or not np.isfinite(features).all():
            raise ValueError(f"Invalid embedding matrix for {name}: {features.shape}")
        all_scores[name] = score_repeated_folds(features, y, splits, args.mode)
        del features
    tables = reproduction_tables(prs, y, splits, all_scores, args.mode)
    for filename, table in tables.items():
        table.to_csv(output_dir / filename, index=False)
    if args.mode == "legacy_reproduction":
        diagnosis = check_reference(args, tables["f1_by_fold.csv"], output_dir)
        (output_dir / "reproduction_check.json").write_text(json.dumps(diagnosis, indent=2), encoding="utf-8")
    print("\nFold-mean F1 and 95% confidence interval half-width")
    print(tables["f1_summary.csv"].to_string(index=False))
    print(f"\nSaved to {output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["repository_disjoint", "legacy_reproduction", "leakage_free"],
        default="repository_disjoint",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-prs-per-agent", type=int, default=2500, help="Repository mode only; reproduction modes use cache bidx.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-steps", type=int, default=300)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, help="A new directory for this run's results")
    parser.add_argument("--cache", type=Path, default=Path(RESULTS) / "repro_emb.npz")
    parser.add_argument("--reference-results", type=Path, help="Legacy Weighted F1 JSON, either {encoder: F1} or rq1.json with a code section.")
    parser.add_argument("--reference-tolerance", type=float, default=0.005)
    args = parser.parse_args()
    if args.folds < 2 or min(args.max_prs_per_agent, args.batch_size, args.head_steps, args.bootstrap_repeats) < 1:
        parser.error("Use at least two folds and positive values for the sample, batch, training and bootstrap sizes.")
    if not np.isfinite(args.reference_tolerance) or args.reference_tolerance < 0:
        parser.error("--reference-tolerance must be finite and nonnegative.")
    if args.mode != "repository_disjoint" and (args.folds != 5 or args.head_steps != 300):
        parser.error("Reproduction modes use exactly 5x5 CV and 300 head steps.")
    if args.reference_results:
        read_reference(args.reference_results)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Use --device cpu or a GPU runtime.")
    use_cuda = args.device != "cpu" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    output_root = Path(RESULTS) / "code_ensemble"
    if args.mode != "repository_disjoint":
        output_root = output_root / args.mode
    output_dir = args.output_dir or output_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if output_dir.exists():
        parser.error(f"Choose a new output directory: {output_dir}")

    settings = {
        "mode": args.mode, "encoders": ENCODERS,
        "ensemble": "equal-weight mean of class cosine similarity scores",
        "device": str(device), "seed": SEED,
        "max_prs_per_agent": args.max_prs_per_agent, "max_tokens": MAX_LENGTH,
        "batch_size": args.batch_size, "head_steps": args.head_steps,
        "repository_bootstrap_repeats": args.bootstrap_repeats,
        "versions": {
            "sklearn": sklearn.__version__, "torch": torch.__version__,
            "transformers": transformers_version, "numpy": np.__version__, "pandas": pd.__version__,
        },
    }
    if args.mode != "repository_disjoint":
        run_reproduction(args, output_dir, device, settings)
        return

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

    settings.update({"split_seed": split_seed, "folds": args.folds})
    save_results(output_dir, prs, y, fold_ids, scores, coverage, settings, args.bootstrap_repeats)


if __name__ == "__main__":
    main()
