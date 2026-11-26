"""Merge equivalent features across hypotheses so each measurement is annotated once."""

import re
from collections import Counter

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .models import embed

FIELDS = ("name", "dtype", "definition", "kind", "spec")


def cluster(vectors, hypothesis, dtype, threshold):
    """Complete linkage on cosine distance; features of one hypothesis or of different dtypes never merge."""
    if len(vectors) == 1:
        return np.ones(1, int)
    distance = 1 - vectors @ vectors.T
    distance[(hypothesis[:, None] == hypothesis) | (dtype[:, None] != dtype)] = 2
    np.fill_diagonal(distance, 0)
    return fcluster(linkage(squareform(distance.clip(0), checks=False), "complete"), threshold, "distance")


def rename(text, mapping):
    return re.sub(r"\w+", lambda m: mapping.get(m[0], m[0]), text)


def dedup(decomposed, env, cfg):
    raw = [(d["id"], f) for d in decomposed for f in d["features"]]
    vectors = embed([f"{f['name']}: {f['definition']}" for _, f in raw], cfg["embedding"])
    owners, dtypes = np.array([i for i, _ in raw]), np.array([f["dtype"] for _, f in raw])
    labels = cluster(vectors, owners, dtypes, cfg["dedup"]["threshold"])

    canon, taken, mappings = {}, Counter(), {d["id"]: {} for d in decomposed}
    for (i, f), label in zip(raw, labels):
        if label not in canon:
            taken[f["name"]] += 1
            name = f["name"] if taken[f["name"]] == 1 else f"{f['name']}_{taken[f['name']]}"
            canon[label] = {**{k: f[k] for k in FIELDS}, "name": name, "members": []}
        canon[label]["members"].append({"hypothesis": i, **{k: f[k] for k in FIELDS}})
        mappings[i][f["name"]] = canon[label]["name"]
    for feature in canon.values():
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
    return {"env": {k: env[k] for k in ("name", "target", "group")}, "features": list(canon.values()), "hypotheses": hypotheses}


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
