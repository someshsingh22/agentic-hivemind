"""Deduplicate semantic measurements with matching sources; keep every code feature."""

import re

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .models import embed

FIELDS = ("name", "dtype", "source_columns", "definition", "kind", "spec")


def cluster(vectors, hypothesis, dtype, sources, threshold):
    """Complete linkage within matching source sets/dtypes, never within one hypothesis."""
    if len(vectors) == 1:
        return np.ones(1, int)
    distance = 1 - vectors @ vectors.T
    incompatible = (
        (hypothesis[:, None] == hypothesis)
        | (dtype[:, None] != dtype)
        | (sources[:, None] != sources)
    )
    distance[incompatible] = 2
    np.fill_diagonal(distance, 0)
    return fcluster(linkage(squareform(distance.clip(0), checks=False), "complete"), threshold, "distance")


def rename(text, mapping):
    return re.sub(r"\w+", lambda m: mapping.get(m[0], m[0]), text)


def dedup(decomposed, env, cfg):
    raw = [(d["id"], f) for d in decomposed for f in d["features"]]
    for _, feature in raw:
        sources = feature.get("source_columns")
        if (not isinstance(sources, list) or not sources
                or any(not isinstance(source, str) or not source for source in sources)
                or len(set(sources)) != len(sources)):
            raise ValueError(f"{feature['name']} needs explicit source_columns; regenerate old decompositions")
    labels = np.arange(len(raw))
    semantic = [index for index, (_, feature) in enumerate(raw) if feature["kind"] == "prompt"]
    if len(semantic) > 1:
        selected = [raw[index] for index in semantic]
        vectors = embed([f"{f['name']}: {f['definition']}" for _, f in selected], cfg["embedding"])
        owners = np.array([owner for owner, _ in selected])
        dtypes = np.array([f["dtype"] for _, f in selected])
        sources = np.array(["\0".join(sorted(f["source_columns"])) for _, f in selected])
        labels[semantic] = len(raw) + cluster(vectors, owners, dtypes, sources, cfg["dedup"]["threshold"])

    canon, taken, mappings = {}, set(), {d["id"]: {} for d in decomposed}
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
