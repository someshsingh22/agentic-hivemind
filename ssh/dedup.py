"""Deduplicate semantic measurements with matching sources; keep every code feature."""

import re

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .equivalence import validate_comparisons
from .features import validate_features
from .models import embed

FIELDS = ("name", "dtype", "source_columns", "definition", "kind", "spec")


def cluster(vectors, hypothesis, dtype, sources, threshold, approved=None):
    """Complete linkage within matching sources/dtypes and approved pairs."""
    if not np.isfinite(threshold) or not 0 <= threshold <= 2:
        raise ValueError("Cosine distance threshold must be between 0 and 2")
    if len(vectors) == 1:
        return np.ones(1, int)
    distance = (1 - vectors @ vectors.T).clip(0, 2)
    incompatible = (
        (hypothesis[:, None] == hypothesis)
        | (dtype[:, None] != dtype)
        | (sources[:, None] != sources)
    )
    if approved is not None:
        incompatible |= ~approved
    distance[incompatible] = threshold + 1
    np.fill_diagonal(distance, 0)
    return fcluster(linkage(squareform(distance.clip(0), checks=False), "complete"), threshold, "distance")


def rename(text, mapping):
    return re.sub(r"\w+", lambda m: mapping.get(m[0], m[0]), text)


def dedup(decomposed, env, cfg, review_cluster, vectors=None):
    """Review every eligible nearby pair, then merge only all-pair approvals.

    Optional normalized vectors follow prompt features in decomposition/raw order.
    Candidate complete-linkage labels are exported only as diagnostics.
    """
    for decomposition in decomposed:
        validate_features(decomposition["features"])
        forbidden = {env["target"], env.get("group"), "post_id", "comment_id"}
        if any(forbidden.intersection(feature["source_columns"]) for feature in decomposition["features"]):
            raise ValueError("Outcome and identifier columns cannot be feature sources")
    raw = [(d["id"], f) for d in decomposed for f in d["features"]]
    labels = np.arange(len(raw))
    reviews = []
    pairs = []
    semantic = [index for index, (_, feature) in enumerate(raw) if feature["kind"] == "prompt"]
    if len(semantic) > 1:
        selected = [raw[index] for index in semantic]
        if vectors is None:
            vectors = embed([f"{f['name']}: {f['definition']}" for _, f in selected], cfg["embedding"])
        owners = np.array([owner for owner, _ in selected])
        dtypes = np.array([f["dtype"] for _, f in selected])
        sources = np.array(["\0".join(sorted(f["source_columns"])) for _, f in selected])
        candidates = cluster(vectors, owners, dtypes, sources, cfg["dedup"]["threshold"])
        approved = np.eye(len(selected), dtype=bool)
        distances = (1 - vectors @ vectors.T).clip(0, 2)
        decisions = {}
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                eligible = (owners[i] != owners[j] and dtypes[i] == dtypes[j]
                            and sources[i] == sources[j])
                if not eligible or distances[i, j] > cfg["dedup"]["threshold"]:
                    continue
                features = [{"hypothesis_id": selected[index][0], "feature": selected[index][1]}
                            for index in (i, j)]
                comparisons = validate_comparisons(review_cluster(features), 2)
                comparison = comparisons[0]
                decisions[(i, j)] = comparison
                if comparison["verdict"] == "equivalent":
                    approved[i, j] = approved[j, i] = True
                reviews.append({
                    "features": features,
                    "comparisons": comparisons,
                    "distances": [{"left": 0, "right": 1,
                                   "cosine_distance": float(distances[i, j])}],
                })
        final_labels = cluster(
            vectors, owners, dtypes, sources, cfg["dedup"]["threshold"], approved=approved
        )
        labels[semantic] = len(raw) + final_labels
        for i, (left_owner, left) in enumerate(selected):
            for j in range(i + 1, len(selected)):
                right_owner, right = selected[j]
                decision = decisions.get((i, j), {})
                eligible = bool(owners[i] != owners[j] and dtypes[i] == dtypes[j]
                                and sources[i] == sources[j])
                pairs.append({
                    "left": f"h{left_owner:05d}:{left['name']}",
                    "right": f"h{right_owner:05d}:{right['name']}",
                    "left_hypothesis": left_owner, "right_hypothesis": right_owner,
                    "left_feature": left, "right_feature": right,
                    "cosine_distance": float(distances[i, j]),
                    "eligible": eligible,
                    "within_threshold": bool(distances[i, j] <= cfg["dedup"]["threshold"]),
                    "same_candidate_cluster": bool(candidates[i] == candidates[j]),
                    "review_verdict": decision.get("verdict", ""),
                    "review_reason": decision.get("reason", ""),
                    "merged": bool(final_labels[i] == final_labels[j]),
                })

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
    return {
        "env": {k: env[k] for k in ("name", "target", "group")},
        "features": list(canon.values()),
        "hypotheses": hypotheses,
        "dedup_reviews": reviews,
        "feature_pairs": pairs,
    }


def summary(plan):
    feats = plan["features"]
    raw = [m for f in feats for m in f["members"]]
    verdicts = [c["verdict"] for review in plan["dedup_reviews"] for c in review["comparisons"]]
    return {
        "hypotheses": len(plan["hypotheses"]),
        "features_raw": len(raw),
        "features_unique": len(feats),
        "prompt_features_raw": sum(m["kind"] == "prompt" for m in raw),
        "prompt_features_unique": sum(f["kind"] == "prompt" for f in feats),
        "reviewed_pairs": len(verdicts),
        "equivalent_pairs": verdicts.count("equivalent"),
        "distinct_pairs": verdicts.count("distinct"),
    }
