"""Deduplicate semantic row measurements; retain code measurements separately."""

import re

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .features import validate_features
from .models import embed

FIELDS = ("name", "dtype", "source_columns", "definition", "kind", "spec")


def cluster(vectors, hypothesis, dtype, sources, threshold):
    """Complete linkage; incompatible owners, dtypes and sources never merge."""
    if not np.isfinite(threshold) or not 0 <= threshold <= 2:
        raise ValueError("Cosine distance threshold must be between 0 and 2")
    if len(vectors) == 0:
        return np.empty(0, int)
    if len(vectors) == 1:
        return np.ones(1, int)
    distance = (1 - vectors @ vectors.T).clip(0, 2)
    incompatible = (
        (hypothesis[:, None] == hypothesis)
        | (dtype[:, None] != dtype)
        | (sources[:, None] != sources)
    )
    distance[incompatible] = threshold + 1
    np.fill_diagonal(distance, 0)
    return fcluster(linkage(squareform(distance, checks=False), "complete"), threshold, "distance")


def rename(text, mapping):
    return re.sub(r"\w+", lambda m: mapping.get(m[0], m[0]), text)


def dedup(decomposed, env, cfg, vectors=None):
    """Cluster prompt features; optional normalized vectors follow prompt-only raw order."""
    forbidden = {env["target"], env.get("group"), "post_id", "comment_id"} - {None}
    for decomposition in decomposed:
        validate_features(decomposition["features"])
        # Outcomes and row identifiers are never predictor inputs.
        if any(forbidden.intersection(feature["source_columns"]) for feature in decomposition["features"]):
            raise ValueError("Outcome and identifier columns cannot be feature sources")
    raw = [(d["id"], f) for d in decomposed for f in d["features"]]
    original = forbidden | {source for _, f in raw for source in f["source_columns"]}
    if any(f["name"] in original for _, f in raw):
        raise ValueError("Feature names cannot collide with original input columns")
    pairs = []
    threshold = cfg["dedup"]["threshold"]
    semantic_indices = [i for i, (_, f) in enumerate(raw) if f["kind"] == "prompt"]
    semantic = [raw[i] for i in semantic_indices]
    owners = np.array([owner for owner, _ in semantic])
    dtypes = np.array([f["dtype"] for _, f in semantic])
    sources = np.array(["\0".join(sorted(f["source_columns"])) for _, f in semantic])
    if vectors is None:
        vectors = (embed([f"{f['name']}: {f['definition']}" for _, f in semantic], cfg["embedding"])
                   if len(semantic) > 1 else np.ones((len(semantic), 1)))
    if len(vectors) != len(semantic):
        raise ValueError("Embedding vectors must follow prompt-only feature order")
    semantic_labels = cluster(vectors, owners, dtypes, sources, threshold)
    labels = -np.arange(1, len(raw) + 1)
    labels[semantic_indices] = semantic_labels
    distances = (1 - vectors @ vectors.T).clip(0, 2)
    for i, (left_owner, left) in enumerate(semantic):
        for j in range(i + 1, len(semantic)):
            right_owner, right = semantic[j]
            pairs.append({
                "left": f"h{left_owner:05d}:{left['name']}",
                "right": f"h{right_owner:05d}:{right['name']}",
                "left_hypothesis": left_owner, "right_hypothesis": right_owner,
                "left_feature": left, "right_feature": right,
                "cosine_distance": float(distances[i, j]),
                "within_threshold": bool(distances[i, j] <= threshold),
                "merged": bool(semantic_labels[i] == semantic_labels[j]),
            })

    canon, taken, mappings = {}, set(original), {d["id"]: {} for d in decomposed}
    for (i, f), label in zip(raw, labels):
        if label not in canon:
            name = f["name"]
            suffix = 2
            while name in taken:
                name = f"{f['name']}_{suffix}"
                suffix += 1
            taken.add(name)
            canon[label] = {**{k: f[k] for k in FIELDS}, "name": name, "members": []}
        canon[label]["members"].append({"hypothesis": i, **{k: f[k] for k in FIELDS}})
        mappings[i][f["name"]] = canon[label]["name"]
    for feature in canon.values():
        if feature["kind"] == "code":
            feature["spec"] = rename(feature["spec"], mappings[feature["members"][0]["hypothesis"]])

    hypotheses = [
        {
            "id": d["id"],
            "hypothesis": d["hypothesis"],
            "h0": rename(d["h0"], m),
            "h1": rename(d["h1"], m),
            "formula": f"{env['target']} ~ {rename(d['rhs'], m)}",
            "term": d["term"] and rename(d["term"], m),
            "sign": d["sign"],
            "note": d["note"],
        }
        for d, m in zip(decomposed, mappings.values())
    ]
    return {
        "env": {k: env[k] for k in ("name", "target", "group")},
        "features": list(canon.values()),
        "hypotheses": hypotheses,
        "feature_pairs": pairs,
    }


def summary(plan):
    feats = plan["features"]
    raw = [m for f in feats for m in f["members"]]
    return {
        "hypotheses": len(plan["hypotheses"]),
        "features_raw": len(raw),
        "features_unique": len(feats),
        "prompt_features_raw": sum(m["kind"] == "prompt" for m in raw),
        "prompt_features_unique": sum(f["kind"] == "prompt" for f in feats),
    }
