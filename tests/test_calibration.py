import copy
import csv
import json
import random
import tempfile
import unittest
from pathlib import Path

from ssh import calibration


def feature(name, source="OC_comments", dtype="binary"):
    return {
        "name": name, "source_columns": [source],
        "dtype": dtype, "definition": "Any comment in this row expresses uncertainty, excluding quotations.",
        "spec": "Return 1 if any comment expresses uncertainty, 0 if none, null if unavailable.",
        "kind": "prompt",
    }


def pair(index, distance, eligible=True, **changes):
    left_name, right_name = f"uncertainty_{index}", f"doubt_{index}"
    return {
        "left": f"h00000:{left_name}", "right": f"h00001:{right_name}",
        "left_hypothesis": 0, "right_hypothesis": 1,
        "left_feature": feature(left_name), "right_feature": feature(right_name),
        "cosine_distance": distance, "eligible": eligible,
        "within_threshold": eligible and distance <= 0.2,
        "same_candidate_cluster": False, "review_verdict": "equivalent",
        "review_reason": "Model review is not a human label.", "merged": False,
        **changes,
    }


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_labels(path, rows):
    fields = ("left", "right", "cosine_distance", "eligible", "review_label", "review_verdict")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def labeled(index, distance, label="", eligible="true"):
    return {
        "left": f"h00000:f{index}", "right": f"h00001:g{index}",
        "cosine_distance": distance, "eligible": eligible, "review_label": label,
        "review_verdict": "equivalent",
    }


class CalibrationExportTests(unittest.TestCase):
    def test_exports_originals_including_ineligible_pairs_and_never_model_labels(self):
        pairs = [pair(0, 0), pair(1, 0.1, False), pair(2, 0.2, False), pair(3, 2, False)]
        pairs[1]["right_feature"]["source_columns"] = ["title"]
        pairs[2]["right_feature"]["dtype"] = "numeric"
        pairs[3]["right_hypothesis"] = pairs[3]["left_hypothesis"]
        pairs[0]["review_label"] = "equivalent"
        pairs[0]["left_feature"]["spec"] = 'Line 1, "quoted"\nLine 2: café'
        original = copy.deepcopy(pairs)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            calibration.write_calibration({"feature_pairs": pairs}, out, 0.2)
            rows = read_csv(out / "feature_pairs.csv")
            self.assertEqual(len(rows), 4)
            by_key = {row["left"]: row for row in rows}
            for item in pairs:
                row = by_key[item["left"]]
                self.assertEqual(row["review_label"], "")
                self.assertEqual(row["review_reason_human"], "")
                self.assertEqual(row["review_verdict"], "equivalent")
                self.assertEqual(row["left_hypothesis"], "0")
                self.assertEqual(row["right_hypothesis"], str(item["right_hypothesis"]))
                for side in ("left", "right"):
                    self.assertEqual(json.loads(row[f"{side}_feature"]), item[f"{side}_feature"])
                    self.assertEqual(json.loads(row[f"{side}_source_columns"]),
                                     item[f"{side}_feature"]["source_columns"])
                    self.assertEqual(row[f"{side}_spec"], item[f"{side}_feature"]["spec"])
            sample = read_csv(out / "calibration_sample.csv")
            self.assertEqual([row["left"] for row in sample], [pairs[0]["left"]])
        self.assertEqual(pairs, original)

    def test_bin_boundaries_include_zero_and_two_exactly_once(self):
        boundaries = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 2]
        with tempfile.TemporaryDirectory() as folder:
            calibration.write_calibration({"feature_pairs": [pair(i, value) for i, value in enumerate(boundaries)]},
                                          folder, 0.2)
            rows = read_csv(Path(folder) / "calibration_sample.csv")
            self.assertEqual(len(rows), len(boundaries))
            self.assertEqual(sorted(float(row["cosine_distance"]) for row in rows), boundaries)

    def test_samples_ten_per_bin_deterministically_independent_of_input_order(self):
        # Distances on lower boundaries must belong to the next bin, not the preceding one.
        distances = (0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4)
        pairs = [pair(bin_index * 100 + index, distance)
                 for bin_index, distance in enumerate(distances) for index in range(24)]
        pairs.extend(pair(1000 + i, 0.1, False) for i in range(20))
        shuffled = copy.deepcopy(pairs)
        random.Random(91).shuffle(shuffled)
        original = copy.deepcopy(shuffled)
        with tempfile.TemporaryDirectory() as folder:
            first, second = Path(folder) / "first", Path(folder) / "second"
            calibration.write_calibration({"feature_pairs": pairs}, first, 0.2)
            calibration.write_calibration({"feature_pairs": shuffled}, second, 0.2)
            self.assertEqual((first / "calibration_sample.csv").read_bytes(),
                             (second / "calibration_sample.csv").read_bytes())
            self.assertEqual((first / "feature_pairs.csv").read_bytes(),
                             (second / "feature_pairs.csv").read_bytes())
            rows = read_csv(first / "calibration_sample.csv")
            self.assertEqual(len(rows), 70)
            for distance in distances:
                self.assertEqual(sum(float(row["cosine_distance"]) == distance for row in rows), 10)
            self.assertTrue(all(row["eligible"] == "True" for row in rows))
            keys = [tuple(sorted((row["left"], row["right"]))) for row in rows]
            self.assertEqual(keys, sorted(keys))
            self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(shuffled, original)

    def test_empty_and_no_eligible_inputs_have_headers_and_no_quality_claim(self):
        for pairs in ([], [pair(0, 0.1, False)]):
            with self.subTest(pairs=pairs), tempfile.TemporaryDirectory() as folder:
                out = Path(folder)
                calibration.write_calibration({"feature_pairs": pairs}, out, 0.2)
                self.assertEqual(read_csv(out / "calibration_sample.csv"), [])
                self.assertIn("review_label", (out / "calibration_sample.csv").read_text().splitlines()[0])
                rows = read_csv(out / "threshold_sensitivity.csv")
                self.assertEqual(len(rows), 7)
                for row in rows:
                    self.assertEqual(row["precision"], "")
                    self.assertEqual(row["recall"], "")
                    self.assertEqual(row["reviewed_total"], "0")
                    self.assertIn("not population accuracy", row["assessment_scope"])
                    self.assertIn("no cutoff recommendation", row["assessment_scope"])

    def test_duplicate_and_invalid_export_distances_fail_before_writes(self):
        original = pair(0, 0.1)
        reversed_pair = {**original, "left": original["right"], "right": original["left"]}
        for pairs in ([original, reversed_pair], *[[{**pair(0, 0.1), "cosine_distance": distance}]
                      for distance in (float("nan"), float("inf"), -0.01, 2.01, None, True)]):
            with self.subTest(pairs=pairs), tempfile.TemporaryDirectory() as folder:
                out = Path(folder) / "new"
                with self.assertRaises(ValueError):
                    calibration.write_calibration({"feature_pairs": pairs}, out, 0.2)
                self.assertFalse(out.exists())


class CalibrationLabelScoringTests(unittest.TestCase):
    def test_human_labels_only_and_inclusive_thresholds(self):
        rows = [
            labeled(0, 0.1, "equivalent", "TRUE"),
            labeled(1, 0.2, "distinct", "1"),
            labeled(2, 0.3, "equivalent"),
            labeled(3, 0.15, "uncertain"),
            labeled(4, 0.05),
            labeled(5, 0.05, "equivalent", "false"),
            labeled(6, 0.05, "distinct", "0"),
        ]
        with tempfile.TemporaryDirectory() as folder:
            path, out = Path(folder) / "human.csv", Path(folder) / "scores"
            write_labels(path, rows)
            before = path.read_bytes()
            calibration.score_labels(path, out)
            self.assertEqual(path.read_bytes(), before)
            scores = {float(row["threshold"]): row for row in read_csv(out / "threshold_sensitivity.csv")}
            self.assertEqual(set(scores), {0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4})
            first = scores[0.1]
            for field, expected in {
                "total_pairs": "7", "eligible_pairs": "5", "ineligible_pairs": "2",
                "reviewed_total": "6", "reviewed_eligible_total": "4", "reviewed_ineligible_total": "2",
                "unreviewed_eligible_total": "1", "reviewed_equivalent_total": "2",
                "reviewed_distinct_total": "1", "reviewed_uncertain_total": "1",
                "candidate_total": "2", "reviewed_candidate_total": "1",
                "true_positive": "1", "false_positive": "0", "missed": "1", "uncertain_included": "0",
            }.items():
                self.assertEqual(first[field], expected, field)
            self.assertEqual(float(first["label_coverage"]), 0.8)
            self.assertEqual(float(first["precision"]), 1)
            self.assertEqual(float(first["recall"]), 0.5)
            self.assertEqual(scores[0.15]["uncertain_included"], "1")
            self.assertEqual(scores[0.2]["false_positive"], "1")
            self.assertEqual(float(scores[0.2]["precision"]), 0.5)
            self.assertEqual(scores[0.3]["true_positive"], "2")
            self.assertEqual(scores[0.3]["missed"], "0")
            self.assertEqual(float(scores[0.3]["recall"]), 1)

    def test_round_trip_labels_are_explicit_and_do_not_overwrite_pair_exports(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            calibration.write_calibration({"feature_pairs": [pair(0, 0.1), pair(1, 0.2)]}, out, 0.2)
            raw_before = (out / "feature_pairs.csv").read_bytes()
            sample = read_csv(out / "calibration_sample.csv")
            sample[0]["review_label"] = "distinct"
            human = out / "reviewed.csv"
            with human.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=calibration.PAIR_FIELDS)
                writer.writeheader()
                writer.writerows(sample)
            calibration.score_labels(human, out)
            row = read_csv(out / "threshold_sensitivity.csv")[0]
            self.assertEqual(row["false_positive"], "1")
            self.assertEqual(row["true_positive"], "0")
            self.assertEqual(float(row["precision"]), 0)
            self.assertEqual(row["recall"], "")
            self.assertEqual((out / "feature_pairs.csv").read_bytes(), raw_before)

    def test_no_labels_or_only_uncertain_labels_keep_precision_recall_null(self):
        for rows in ([], [labeled(0, 0.1)], [labeled(0, 0.1, "uncertain")],
                     [labeled(0, 0.1, "equivalent", "false")]):
            with self.subTest(rows=rows), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "labels.csv"
                write_labels(path, rows)
                calibration.score_labels(path, Path(folder) / "out")
                for score in read_csv(Path(folder) / "out" / "threshold_sensitivity.csv"):
                    self.assertEqual(score["precision"], "")
                    self.assertEqual(score["recall"], "")
                    self.assertEqual(score["true_positive"], "0")
                    self.assertEqual(score["false_positive"], "0")

    def test_rejects_invalid_labels_eligibility_distances_and_duplicate_pairs(self):
        duplicate = labeled(0, 0.1)
        reversed_duplicate = {**duplicate, "left": duplicate["right"], "right": duplicate["left"]}
        cases = [
            [labeled(0, 0.1, "needs_alignment")], [labeled(0, 0.1, "Equivalent")],
            [labeled(0, 0.1, "equivalent", "")], [labeled(0, 0.1, "", "maybe")],
            [duplicate, reversed_duplicate], [duplicate, duplicate],
            *[[labeled(0, distance)] for distance in ("nan", "inf", "-inf", -0.01, 2.01, "")],
        ]
        for rows in cases:
            with self.subTest(rows=rows), tempfile.TemporaryDirectory() as folder:
                path, out = Path(folder) / "labels.csv", Path(folder) / "out"
                write_labels(path, rows)
                before = path.read_bytes()
                with self.assertRaises(ValueError):
                    calibration.score_labels(path, out)
                self.assertEqual(path.read_bytes(), before)
                self.assertFalse(out.exists())

    def test_requires_eligibility_column_and_protects_input_destination(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "labels.csv"
            path.write_text("left,right,cosine_distance,review_label\na,b,0.1,equivalent\n")
            with self.assertRaises(ValueError):
                calibration.score_labels(path, Path(folder) / "out")
            path = Path(folder) / "threshold_sensitivity.csv"
            write_labels(path, [labeled(0, 0.1, "equivalent")])
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                calibration.score_labels(path, folder)
            self.assertEqual(path.read_bytes(), before)

    def test_rejects_reused_explicit_pair_id_even_for_different_features(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "labels.csv"
            rows = [{**labeled(i, 0.1), "pair_id": "same-id"} for i in range(2)]
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaises(ValueError):
                calibration.score_labels(path, Path(folder) / "out")


if __name__ == "__main__":
    unittest.main()
