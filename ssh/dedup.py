"""Deduplicate semantic measurements with matching sources; keep every code feature."""

import re

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .equivalence import validate_comparisons
from .features import validate_features
from .models import embed

FIELDS = ("name", "dtype", "source_columns", "definition", "kind", "spec",
          "class", "annotation_unit", "depends_on")


def cluster(vectors, hypothesis, dtype, sources, threshold, approved=None, units=None):
    """Complete linkage within matching sources/dtypes/units and approved pairs."""
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
    if units is not None:
        incompatible |= units[:, None] != units
    if approved is not None:
        incompatible |= ~approved
    distance[incompatible] = threshold + 1
    np.fill_diagonal(distance, 0)
    return fcluster(linkage(squareform(distance.clip(0), checks=False), "complete"), threshold, "distance")


def rename(text, mapping):
    return re.sub(r"\w+", lambda m: mapping.get(m[0], m[0]), text)


def dedup(decomposed, env, cfg, review_cluster):
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
        vectors = embed([f"{f['name']}: {f['definition']}" for _, f in selected], cfg["embedding"])
        owners = np.array([owner for owner, _ in selected])
        dtypes = np.array([f["dtype"] for _, f in selected])
        sources = np.array(["\0".join(sorted(f["source_columns"])) for _, f in selected])
        units = np.array([f["annotation_unit"] for _, f in selected])
        candidates = cluster(vectors, owners, dtypes, sources, cfg["dedup"]["threshold"], units=units)
        approved = np.eye(len(selected), dtype=bool)
        distances = (1 - vectors @ vectors.T).clip(0, 2)
        decisions = {}
        for label in dict.fromkeys(candidates):
            indices = np.flatnonzero(candidates == label).tolist()
            if len(indices) < 2:
                continue
            features = [{"hypothesis_id": selected[index][0], "feature": selected[index][1]} for index in indices]
            comparisons = validate_comparisons(review_cluster(features), len(features))
            pair_distances = []
            for comparison in comparisons:
                left, right = comparison["left"], comparison["right"]
                i, j = indices[left], indices[right]
                pair_distances.append({"left": left, "right": right, "cosine_distance": float(distances[i, j])})
                decisions[(i, j)] = comparison
                if comparison["verdict"] == "equivalent":
                    approved[i, j] = approved[j, i] = True
            reviews.append({"features": features, "comparisons": comparisons, "distances": pair_distances})
        final_labels = cluster(
            vectors, owners, dtypes, sources, cfg["dedup"]["threshold"], approved=approved, units=units
        )
        labels[semantic] = len(raw) + final_labels
        for i, (left_owner, left) in enumerate(selected):
            for j in range(i + 1, len(selected)):
                right_owner, right = selected[j]
                decision = decisions.get((i, j), {})
                eligible = bool(owners[i] != owners[j] and dtypes[i] == dtypes[j]
                                and sources[i] == sources[j] and units[i] == units[j])
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
        feature["depends_on"] = [mappings[feature["members"][0]["hypothesis"]][name]
                                 for name in feature["depends_on"]]

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
    return {
        "hypotheses": len(plan["hypotheses"]),
        "features_raw": len(raw),
        "features_unique": len(feats),
        "prompt_features_raw": sum(m["kind"] == "prompt" for m in raw),
        "prompt_features_unique": sum(f["kind"] == "prompt" for f in feats),
        "comment_predicates_unique": sum(f["kind"] == "prompt" and f["annotation_unit"] == "comment" for f in feats),
        "row_semantic_tasks_unique": sum(f["kind"] == "prompt" and f["annotation_unit"] == "row" for f in feats),
        "secondary_features": sum(f["class"] == "secondary" for f in feats),
        "pairs_needing_alignment": sum(c["verdict"] == "needs_alignment"
                                      for review in plan["dedup_reviews"] for c in review["comparisons"]),
    }
