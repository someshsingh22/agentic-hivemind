"""Immutable pair exports and human-label-only, descriptive calibration summaries.

Sensitivity describes the reviewed eligible subset, not population accuracy. Model
review verdicts never become human labels, and no threshold is recommended or tuned.
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
    "merged", "review_label", "review_reason_human", "configured_threshold",
)
SENSITIVITY_FIELDS = (
    "threshold", "total_pairs", "eligible_pairs", "ineligible_pairs",
    "reviewed_total", "reviewed_eligible_total", "reviewed_ineligible_total",
    "unreviewed_eligible_total", "label_coverage", "reviewed_equivalent_total",
    "reviewed_distinct_total", "reviewed_uncertain_total", "candidate_total",
    "reviewed_candidate_total", "true_positive", "false_positive", "missed",
    "uncertain_included", "precision", "recall", "assessment_scope",
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
        label = row.get("review_label", "")
        if label not in ("", "equivalent", "distinct", "uncertain"):
            raise ValueError(f"Invalid human review_label: {label!r}")
        if "eligible" not in row:
            raise ValueError("Every pair, including labeled pairs, requires explicit eligibility")
        validated.append({**row, "cosine_distance": _distance(row.get("cosine_distance")),
                          "eligible": _boolean(row["eligible"]), "review_label": label})
    return sorted(validated, key=_pair_key)


def _write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sensitivity(rows):
    eligible = [row for row in rows if row["eligible"]]
    reviewed = [row for row in rows if row["review_label"]]
    labeled = [row for row in eligible if row["review_label"]]
    equivalents = sum(row["review_label"] == "equivalent" for row in labeled)
    distinct = sum(row["review_label"] == "distinct" for row in labeled)
    uncertain = sum(row["review_label"] == "uncertain" for row in labeled)
    for threshold in THRESHOLDS:
        included = [row for row in labeled if row["cosine_distance"] <= threshold]
        true_positive = sum(row["review_label"] == "equivalent" for row in included)
        false_positive = sum(row["review_label"] == "distinct" for row in included)
        yield {
            "threshold": threshold,
            "total_pairs": len(rows),
            "eligible_pairs": len(eligible),
            "ineligible_pairs": len(rows) - len(eligible),
            "reviewed_total": len(reviewed),
            "reviewed_eligible_total": len(labeled),
            "reviewed_ineligible_total": len(reviewed) - len(labeled),
            "unreviewed_eligible_total": len(eligible) - len(labeled),
            "label_coverage": len(labeled) / len(eligible) if eligible else None,
            "reviewed_equivalent_total": equivalents,
            "reviewed_distinct_total": distinct,
            "reviewed_uncertain_total": uncertain,
            "candidate_total": sum(row["cosine_distance"] <= threshold for row in eligible),
            "reviewed_candidate_total": len(included),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "missed": equivalents - true_positive,
            "uncertain_included": sum(row["review_label"] == "uncertain" for row in included),
            "precision": true_positive / (true_positive + false_positive)
            if true_positive + false_positive else None,
            "recall": true_positive / equivalents if equivalents else None,
            "assessment_scope": "human-reviewed eligible subset only; not population accuracy; no cutoff recommendation",
        }


def write_calibration(plan, out, threshold):
    """Export all supplied original semantic pairs and a reproducible review sample.

    The producer supplies stable raw feature keys and original feature objects.
    Sampling is without replacement, at most ten eligible pairs per half-open bin
    (the last bin includes distance 2), seeded with 42 after sorting pair keys.
    All outputs start with empty human labels, regardless of model review verdicts.
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
        row.update(review_label="", review_reason_human="", configured_threshold=threshold)
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


def score_labels(path, out):
    """Score an explicitly human-labeled CSV without modifying that input file.

    Ineligible labels are counted as coverage only, never as metric observations.
    Unknown labels, duplicate unordered pairs, missing eligibility, and invalid
    distances fail before output. Blank precision/recall mean no labeled denominator.
    """
    path = Path(path)
    out = Path(out)
    destination = out / "threshold_sensitivity.csv"
    if path.resolve() == destination.resolve():
        raise ValueError("The sensitivity output must not overwrite the input CSV")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"left", "right", "cosine_distance", "eligible", "review_label"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"Labeled CSV requires columns: {', '.join(sorted(required))}")
        rows = _validate_rows(reader)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(destination, SENSITIVITY_FIELDS, _sensitivity(rows))
