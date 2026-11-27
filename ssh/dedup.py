"""Merge equivalent row measurements across owners with matching dtypes and sources."""

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
    return fcluster(linkage(squareform(distance.clip(0), checks=False), "complete"), threshold, "distance")


def rename(text, mapping):
    return re.sub(r"\w+", lambda m: mapping.get(m[0], m[0]), text)


def dedup(decomposed, env, cfg, vectors=None):
    """Cluster all features; optional normalized vectors follow raw feature order."""
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
    owners = np.array([owner for owner, _ in raw])
    dtypes = np.array([f["dtype"] for _, f in raw])
    sources = np.array(["\0".join(sorted(f["source_columns"])) for _, f in raw])
    if vectors is None:
        vectors = (embed([f"{f['name']}: {f['definition']}" for _, f in raw], cfg["embedding"])
                   if len(raw) > 1 else np.ones((len(raw), 1)))
    labels = cluster(vectors, owners, dtypes, sources, threshold)
    distances = (1 - vectors @ vectors.T).clip(0, 2)
    for i, (left_owner, left) in enumerate(raw):
        for j in range(i + 1, len(raw)):
            right_owner, right = raw[j]
            eligible = bool(owners[i] != owners[j] and dtypes[i] == dtypes[j]
                            and sources[i] == sources[j])
            pairs.append({
                "left": f"h{left_owner:05d}:{left['name']}",
                "right": f"h{right_owner:05d}:{right['name']}",
                "left_hypothesis": left_owner, "right_hypothesis": right_owner,
                "left_feature": left, "right_feature": right,
                "cosine_distance": float(distances[i, j]),
                "eligible": eligible,
                "within_threshold": bool(distances[i, j] <= threshold),
                "merged": bool(labels[i] == labels[j]),
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
