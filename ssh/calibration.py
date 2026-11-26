"""Immutable pair exports and descriptive counts of automated binary decisions.

Sensitivity reports reviewed and unreviewed candidates, not prediction accuracy.
No threshold is recommended or tuned.
"""

import csv
import json
import math
import random
from pathlib import Path


DISTANCE_BINS = (0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 2.000001)
THRESHOLDS = (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4)
FEATURE_FIELDS = (
    "name", "source_columns", "dtype", "definition", "kind", "spec",
)
PAIR_FIELDS = (
    "pair_id", "left", "right", "left_hypothesis", "right_hypothesis",
    *(f"{side}_{field}" for side in ("left", "right") for field in FEATURE_FIELDS),
    "left_feature", "right_feature", "cosine_distance", "eligible",
    "within_threshold", "same_candidate_cluster", "review_verdict", "review_reason",
    "merged", "configured_threshold",
)
SENSITIVITY_FIELDS = (
    "threshold", "total_pairs", "eligible_pairs", "ineligible_pairs",
    "candidate_total", "reviewed_candidate_total", "equivalent_candidate_total",
    "distinct_candidate_total", "unreviewed_candidate_total",
)


def _distance(value):
    try:
        distance = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid cosine distance: {value!r}") from exc
    if isinstance(value, bool) or not math.isfinite(distance) or not 0 <= distance <= 2:
        raise ValueError(f"Cosine distance must be finite and between 0 and 2: {value!r}")
    return distance


def _boolean(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1"):
            return True
        if normalized in ("false", "0"):
            return False
    raise ValueError(f"Eligibility/boolean must be true, false, 1, or 0: {value!r}")


def _pair_key(row):
    left, right = row.get("left"), row.get("right")
    if not isinstance(left, str) or not left.strip() or not isinstance(right, str) or not right.strip():
        raise ValueError("Each pair requires nonempty left and right feature keys")
    if left == right:
        raise ValueError("A feature cannot be paired with itself")
    return tuple(sorted((left, right)))


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _validate_rows(rows):
    """Validate before writing anything; detect reverse pairs as duplicates too."""
    validated, keys, ids = [], set(), set()
    for row in rows:
        key = _pair_key(row)
        pair_id = row.get("pair_id") or _json(key)
        if key in keys or pair_id in ids:
            raise ValueError(f"Duplicate pair ID: {pair_id}")
        keys.add(key)
        ids.add(pair_id)
        verdict = row.get("review_verdict", "")
        if verdict not in ("", "equivalent", "distinct"):
            raise ValueError(f"Invalid review_verdict: {verdict!r}")
        if "eligible" not in row:
            raise ValueError("Every pair requires explicit eligibility")
        eligible = _boolean(row["eligible"])
        if verdict and not eligible:
            raise ValueError("Ineligible pairs cannot have review verdicts")
        validated.append({**row, "cosine_distance": _distance(row.get("cosine_distance")),
                          "eligible": eligible, "review_verdict": verdict})
    return sorted(validated, key=_pair_key)


def _write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sensitivity(rows):
    eligible = [row for row in rows if row["eligible"]]
    for threshold in THRESHOLDS:
        candidates = [row for row in eligible if row["cosine_distance"] <= threshold]
        equivalent = sum(row["review_verdict"] == "equivalent" for row in candidates)
        distinct = sum(row["review_verdict"] == "distinct" for row in candidates)
        reviewed = equivalent + distinct
        yield {
            "threshold": threshold,
            "total_pairs": len(rows),
            "eligible_pairs": len(eligible),
            "ineligible_pairs": len(rows) - len(eligible),
            "candidate_total": len(candidates),
            "reviewed_candidate_total": reviewed,
            "equivalent_candidate_total": equivalent,
            "distinct_candidate_total": distinct,
            "unreviewed_candidate_total": len(candidates) - reviewed,
        }


def write_calibration(plan, out, threshold):
    """Export all supplied original semantic pairs and a reproducible audit sample.

    The producer supplies stable raw feature keys and original feature objects.
    Sampling is without replacement, at most ten eligible pairs per half-open bin
    (the last bin includes distance 2), seeded with 42 after sorting pair keys.
    Samples retain automated verdicts; unreviewed pairs remain explicitly unreviewed.
    """
    threshold = _distance(threshold)
    rows = []
    for pair in plan["feature_pairs"]:
        row = {field: pair[field] for field in (
            "left", "right", "left_hypothesis", "right_hypothesis", "cosine_distance",
            "eligible", "within_threshold", "same_candidate_cluster", "review_verdict",
            "review_reason", "merged",
        )}
        row["pair_id"] = _json(_pair_key(row))
        for side in ("left", "right"):
            feature = pair[f"{side}_feature"]
            row[f"{side}_feature"] = _json(feature)
            for field in FEATURE_FIELDS:
                value = feature.get(field, "")
                row[f"{side}_{field}"] = _json(value) if isinstance(value, (list, dict)) else value
        for field in ("within_threshold", "same_candidate_cluster", "merged"):
            row[field] = _boolean(row[field])
        row["configured_threshold"] = threshold
        rows.append(row)
    rows = _validate_rows(rows)
    randomizer = random.Random(42)
    sample = []
    for lower, upper in zip(DISTANCE_BINS, DISTANCE_BINS[1:]):
        bucket = [row for row in rows if row["eligible"] and lower <= row["cosine_distance"] < upper]
        sample.extend(randomizer.sample(bucket, min(10, len(bucket))))
    sample.sort(key=_pair_key)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "feature_pairs.csv", PAIR_FIELDS, rows)
    _write_csv(out / "calibration_sample.csv", PAIR_FIELDS, sample)
    _write_csv(out / "threshold_sensitivity.csv", SENSITIVITY_FIELDS, _sensitivity(rows))

