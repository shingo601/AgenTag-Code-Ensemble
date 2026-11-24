"""
Shared machinery for every RQ: marker stripping, the embedding cache, the
contrastive metric space, and the cross-validation protocol.

Everything here has exactly one definition. That is deliberate. An earlier
version of this pipeline kept a private copy of the marker regex in each script,
and the copies drifted: the "canonical" one narrowed to commit trailers while
the others still stripped tool URLs and bare tool names. The result was that
98% of Copilot PRs, 98% of Devin PRs, and 83% of OpenAI Codex PRs still named
their own author in text reported as "latent", and a regex doing nothing but
matching those names scored a weighted F1 of 0.941. If you add a script, import
from here rather than copying.
"""
import os
import re
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

SEED = 42
N_REP = 5  # 5 splits x 5 repeats = 25 train/test estimates per configuration

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
RESULTS = os.path.join(BASE, "results")
CACHE = os.path.join(RESULTS, "repro_emb.npz")
REDUN = os.path.join(RESULTS, "features_after_redun.csv")

# The eight features RQ3 treats as explicit markers rather than latent behavior.
EXPLICIT = ["has_checklist", "num_bullets", "num_links", "num_code_blocks",
            "conventional_title", "conventional_commit_ratio",
            "multiline_commit_ratio", "capitalized_commit_ratio"]
TEXT_STYLE = ["title_length", "body_length", "avg_commit_length", "std_commit_length"]

AGENTS = ["Claude_Code", "Copilot", "Cursor", "Devin", "OpenAI_Codex"]

# --------------------------------------------------------------------------
# Self-disclosed identity, stripped in three cumulative tiers.
# Tiers 1-2 delete the whole line (the line exists only to disclose the tool).
# Tier 3 deletes the name token but keeps the sentence, since the writing style
# around it is the latent signal we want to measure.
# --------------------------------------------------------------------------
TIER1 = re.compile(
    r"(co-?authored-?by|signed-?off-?by|generated with|no-?reply@anthropic|\U0001F916)", re.I)
TIER2 = re.compile(
    r"(claude\.ai|claude\.com|anthropic\.com|"
    r"app\.devin\.ai|devin\.ai|"
    r"gh\.io/copilot|github\.com/[^\s]*copilot|"
    r"cursor\.com|cursor\.sh|cursoragent|devin-ai-integration|"
    r"chatgpt\.com|openai\.com/codex|"
    r"copilot coding agent|link to devin run|devin session|copilot chat)", re.I)
# Deliberately aggressive: also removes legitimate uses (a database "cursor"),
# which biases the latent estimate downward and keeps it a conservative bound.
TIER3 = re.compile(r"\b(devin|copilot|cursor|claude|codex|anthropic|openai)\b", re.I)

TIERS = {1: TIER1, 2: TIER2}


def strip_markers(text, tiers=(1, 2, 3)):
    """Remove self-disclosed tool identity. tiers=() returns the raw text."""
    if not isinstance(text, str) or not text:
        return ""
    lines = text.splitlines()
    for t in (1, 2):
        if t in tiers:
            lines = [l for l in lines if not TIERS[t].search(l)]
    out = "\n".join(lines)
    if 3 in tiers:
        out = TIER3.sub(" ", out)
    return re.sub(r"[ \t]{2,}", " ", out).strip()


def added_code(patches, cap=4000):
    """Added lines of a PR's patches, truncated to `cap` characters."""
    out = []
    for p in patches:
        if isinstance(p, str):
            out += [l[1:] for l in p.split("\n")
                    if l.startswith("+") and not l.startswith("+++")]
    return "\n".join(out)[:cap]


# --------------------------------------------------------------------------
# Contrastive metric space
# --------------------------------------------------------------------------
class Head(nn.Module):
    def __init__(s, d_in, d=128):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(d_in, 256), nn.ReLU(), nn.Linear(256, d))

    def forward(s, x):
        return nn.functional.normalize(s.net(x), dim=1)


def supcon(z, y, t=0.1):
    sim = z @ z.t() / t
    sim = sim - sim.max(1, keepdim=True).values.detach()
    B = z.size(0)
    eye = torch.eye(B, device=z.device).bool()
    pos = (y[:, None] == y[None, :]) & ~eye
    exp = torch.exp(sim) * (~eye)
    logp = sim - torch.log(exp.sum(1, keepdim=True) + 1e-12)
    pc = pos.sum(1)
    v = pc > 0
    return (-(logp * pos).sum(1)[v] / pc[v]).mean()


def train_head(X, y, steps=300, bs=512):
    torch.manual_seed(SEED)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y)
    h = Head(X.shape[1])
    opt = torch.optim.Adam(h.parameters(), lr=1e-3)
    n = len(X)
    for _ in range(steps):
        i = torch.randint(0, n, (min(bs, n),))
        loss = supcon(h(Xt[i]), yt[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return h


def emb(h, X):
    with torch.no_grad():
        return h(torch.tensor(X, dtype=torch.float32)).numpy()


def cents(Z, y, labs):
    C = np.stack([Z[y == a].mean(0) for a in labs])
    return C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)


def ci(v):
    """Mean and 95% half-width. The mean is what the paired test is defined on;
    see the paper's Conclusion Validity for why mean rather than median."""
    a = np.array(v)
    return round(float(a.mean()), 3), round(float(1.96 * a.std(ddof=1) / np.sqrt(len(a))), 3)


def closed_ci(X, y, contrastive=True, probe=True):
    """Repeated-CV weighted/macro F1 for the probe and the contrastive space."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    Xs = StandardScaler().fit_transform(X)
    labs = sorted(set(y))
    pw, pm, cw, cm = [], [], [], []
    perclass = {a: [] for a in labs}
    skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=N_REP, random_state=SEED)
    for tr, te in skf.split(Xs, y):
        if probe:
            p = LogisticRegression(max_iter=2000, n_jobs=-1).fit(Xs[tr], y[tr])
            pp = p.predict(Xs[te])
            pw.append(f1_score(y[te], pp, average="weighted"))
            pm.append(f1_score(y[te], pp, average="macro"))
        if contrastive:
            h = train_head(Xs[tr], pd.factorize(y[tr])[0])
            pred = np.array(labs)[(emb(h, Xs[te]) @ cents(emb(h, Xs[tr]), y[tr], labs).T).argmax(1)]
            cw.append(f1_score(y[te], pred, average="weighted"))
            cm.append(f1_score(y[te], pred, average="macro"))
            for a in labs:
                m = y[te] == a
                if m.sum():
                    perclass[a].append(f1_score(m, pred == a, zero_division=0))
    r = {}
    if probe:
        r.update({"probe_wF1": ci(pw), "probe_mF1": ci(pm)})
    if contrastive:
        r.update({"contr_wF1": ci(cw), "contr_mF1": ci(cm),
                  "contr_perclass": {a: round(float(np.mean(v)), 3) for a, v in perclass.items()}})
    return r


# --------------------------------------------------------------------------
# Embedding cache: four marker conditions per text stream
# --------------------------------------------------------------------------
CONDS = {"": (1, 2, 3), "_t12": (1, 2), "_t1": (1,), "_raw": ()}
"""Cumulative marker conditions. "" is the study default (latent: trailers,
boilerplate, and name tokens all removed). _t12 restores name mentions, _t1 also
restores boilerplate, _raw restores the trailers too. RQ3 walks this ladder."""


def build_cache(verbose=True):
    """Load or build the embedding cache. Building encodes 33,580 PRs x 4 marker
    conditions x 3 text streams and takes roughly 15 minutes on CPU."""
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        return {k: z[k] for k in z.files}

    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    enc = lambda s: m.encode(list(s), batch_size=128, normalize_embeddings=True,
                             show_progress_bar=False)

    rc = [c for c in pd.read_csv(REDUN, nrows=1).columns if c not in ("pr_id", "agent")]
    struct = pd.read_csv(os.path.join(RESULTS, "extracted_features.csv"))
    repo = pd.read_parquet(f"{DATA}/repository.parquet")[["url", "stars"]]
    pr = pd.read_parquet(f"{DATA}/pull_request.parquet")[["id", "agent", "title", "body", "repo_url"]]
    com = pd.read_parquet(f"{DATA}/pr_commits.parquet")[["pr_id", "message"]].dropna()
    com["message"] = com["message"].astype(str)

    ag = pr[pr["id"].isin(struct["pr_id"])].reset_index(drop=True)
    ag["title"] = ag["title"].fillna("")
    ag["body"] = ag["body"].fillna("")
    for suf, tiers in CONDS.items():
        msg = com["message"].apply(lambda t, _t=tiers: strip_markers(t, tiers=_t))
        cmsg = (com.assign(_m=msg).groupby("pr_id")["_m"]
                   .apply(lambda s: "\n".join(s)).rename("cmsg" + suf).reset_index())
        ag = ag.merge(cmsg, left_on="id", right_on="pr_id", how="left").drop(columns=["pr_id"])
        ag["cmsg" + suf] = ag["cmsg" + suf].fillna("")
        if suf:
            ag["title" + suf] = ag["title"].apply(lambda t, _t=tiers: strip_markers(t, tiers=_t))
            ag["body" + suf] = ag["body"].apply(lambda t, _t=tiers: strip_markers(t, tiers=_t))
        else:
            mlen = com.assign(L=msg.str.len()).groupby("pr_id")["L"]
            avgmap, stdmap = mlen.mean(), mlen.std(ddof=0)
    ag["title"] = ag["title"].apply(strip_markers)
    ag["body"] = ag["body"].apply(strip_markers)
    ag = ag.merge(struct[["pr_id"] + rc], left_on="id", right_on="pr_id", how="left").fillna(0)
    for c, mp in [("title_length", None), ("body_length", None),
                  ("avg_commit_length", avgmap), ("std_commit_length", stdmap)]:
        if c not in rc:
            continue
        ag[c] = ag["title"].str.len() if c == "title_length" else (
                ag["body"].str.len() if c == "body_length" else ag["id"].map(mp).fillna(0))
    ag = ag.merge(repo, left_on="repo_url", right_on="url", how="left")

    if verbose:
        print(f"[cache] encoding {len(ag)} PRs x {len(CONDS)} marker conditions...", flush=True)
    t0 = time.time()
    d = {"rc": np.array(rc, dtype=object), "agent": ag["agent"].values,
         "stars": ag["stars"].fillna(0).values, "struct": ag[rc].to_numpy(float)}
    for suf in CONDS:
        for stream in ("title", "body", "cmsg"):
            d[stream + suf] = enc(ag[stream + suf])
        if verbose:
            print(f"[cache]   '{suf or 'latent'}' done {time.time()-t0:.0f}s", flush=True)

    bidx = (ag.groupby("agent", group_keys=False)
              .apply(lambda g: g.sample(min(len(g), 2500), random_state=SEED)).index.to_numpy())
    d["bidx"] = bidx
    det = pd.read_parquet(f"{DATA}/pr_commit_details.parquet", columns=["pr_id", "patch"],
                          filters=[("pr_id", "in", ag.iloc[bidx]["id"].tolist())])
    cmap = det.groupby("pr_id")["patch"].apply(added_code).to_dict()
    d["code_bal"] = enc([cmap.get(i, "") for i in ag.iloc[bidx]["id"]])

    hu = pd.read_parquet(f"{DATA}/human_pull_request.parquet")[["title", "body"]].fillna("")
    d["h_title"] = enc(hu["title"].apply(strip_markers))
    d["h_body"] = enc(hu["body"].apply(strip_markers))

    os.makedirs(RESULTS, exist_ok=True)
    np.savez(CACHE, **d)
    if verbose:
        print("[cache] saved", flush=True)
    return d


def load_results(name):
    import json
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def save_results(name, obj):
    import json
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, name), "w") as f:
        json.dump(obj, f, indent=1, default=float)
    print(f"saved results/{name}", flush=True)
