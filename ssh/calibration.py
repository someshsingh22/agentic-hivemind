"""Deterministic semantic-feature pair exports and threshold candidate counts."""

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
    "left_feature", "right_feature", "cosine_distance",
    "within_threshold", "merged", "configured_threshold",
)
SENSITIVITY_FIELDS = ("threshold", "total_pairs", "candidate_total")


def _distance(value):
    try:
        distance = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid cosine distance: {value!r}") from exc
    if isinstance(value, bool) or not math.isfinite(distance) or not 0 <= distance <= 2:
        raise ValueError(f"Cosine distance must be finite and between 0 and 2: {value!r}")
    return distance


def _boolean(value):
    if not isinstance(value, bool):
        raise ValueError(f"Pair flags must be boolean: {value!r}")
    return value


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
    validated, keys = [], set()
    for row in rows:
        key = _pair_key(row)
        if key in keys:
            raise ValueError(f"Duplicate feature pair: {key}")
        keys.add(key)
        validated.append({**row, "cosine_distance": _distance(row.get("cosine_distance"))})
    return sorted(validated, key=_pair_key)


def _write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sensitivity(rows, comparisons):
    for threshold in THRESHOLDS:
        yield {
            "threshold": threshold,
            "total_pairs": len(rows),
            "candidate_total": sum(row["cosine_distance"] <= threshold for row in comparisons),
        }


def write_calibration(plan, out, threshold):
    """Export prompt pairs and sample pairs with matching sources/dtypes across hypotheses.

    Sampling is without replacement, at most ten pairs per half-open bin
    (the last bin includes distance 2), seeded with 42 after sorting pair keys.
    """
    threshold = _distance(threshold)
    rows = []
    for pair in plan["feature_pairs"]:
        row = {field: pair[field] for field in (
            "left", "right", "left_hypothesis", "right_hypothesis", "cosine_distance",
            "within_threshold", "merged",
        )}
        row["pair_id"] = _json(_pair_key(row))
        for side in ("left", "right"):
            feature = pair[f"{side}_feature"]
            row[f"{side}_feature"] = _json(feature)
            for field in FEATURE_FIELDS:
                value = feature.get(field, "")
                row[f"{side}_{field}"] = _json(value) if isinstance(value, (list, dict)) else value
        for field in ("within_threshold", "merged"):
            row[field] = _boolean(row[field])
        row["configured_threshold"] = threshold
        rows.append(row)
    rows = _validate_rows(rows)
    comparisons = [row for row in rows
                   if row["left_hypothesis"] != row["right_hypothesis"]
                   and row["left_dtype"] == row["right_dtype"]
                   and set(json.loads(row["left_source_columns"])) == set(json.loads(row["right_source_columns"]))]
    randomizer = random.Random(42)
    sample = []
    for lower, upper in zip(DISTANCE_BINS, DISTANCE_BINS[1:]):
        bucket = [row for row in comparisons if lower <= row["cosine_distance"] < upper]
        sample.extend(randomizer.sample(bucket, min(10, len(bucket))))
    sample.sort(key=_pair_key)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "feature_pairs.csv", PAIR_FIELDS, rows)
    _write_csv(out / "calibration_sample.csv", PAIR_FIELDS, sample)
    _write_csv(out / "threshold_sensitivity.csv", SENSITIVITY_FIELDS, _sensitivity(rows, comparisons))

