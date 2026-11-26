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
        "same_candidate_cluster": False,
        "review_verdict": "equivalent" if eligible and distance <= 0.2 else "",
        "review_reason": "Both measure the same row-level uncertainty indicator."
        if eligible and distance <= 0.2 else "",
        "merged": False,
        **changes,
    }


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))




class CalibrationExportTests(unittest.TestCase):
    def test_exports_originals_binary_decisions_gates_and_merge_state(self):
        pairs = [
            pair(0, 0, merged=True, same_candidate_cluster=True),
            pair(1, 0.1, False), pair(2, 0.2, False), pair(3, 2, False),
            pair(4, 0.2, review_verdict="distinct",
                 review_reason="The quote exclusion changes the measured evidence."),
            pair(5, 0.3),
        ]
        pairs[1]["right_feature"]["source_columns"] = ["title"]
        pairs[2]["right_feature"]["dtype"] = "numeric"
        pairs[3]["right_hypothesis"] = pairs[3]["left_hypothesis"]
        pairs[0]["left_feature"]["spec"] = 'Line 1, "quoted"\nLine 2: café'
        original = copy.deepcopy(pairs)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            calibration.write_calibration({"feature_pairs": pairs}, out, 0.2)
            rows = read_csv(out / "feature_pairs.csv")
            self.assertEqual(len(rows), len(pairs))
            by_key = {row["left"]: row for row in rows}
            for item in pairs:
                row = by_key[item["left"]]
                self.assertNotIn("review_label", row)
                self.assertNotIn("review_reason_human", row)
                self.assertEqual(row["review_verdict"], item["review_verdict"])
                self.assertEqual(row["review_reason"], item["review_reason"])
                self.assertEqual(float(row["cosine_distance"]), item["cosine_distance"])
                self.assertEqual(float(row["configured_threshold"]), 0.2)
                for field in ("eligible", "within_threshold", "same_candidate_cluster", "merged"):
                    self.assertEqual(row[field], str(item[field]))
                self.assertEqual(row["left_hypothesis"], "0")
                self.assertEqual(row["right_hypothesis"], str(item["right_hypothesis"]))
                for side in ("left", "right"):
                    self.assertEqual(json.loads(row[f"{side}_feature"]), item[f"{side}_feature"])
                    self.assertEqual(json.loads(row[f"{side}_source_columns"]),
                                     item[f"{side}_feature"]["source_columns"])
                    self.assertEqual(row[f"{side}_spec"], item[f"{side}_feature"]["spec"])
            sample = read_csv(out / "calibration_sample.csv")
            self.assertEqual([row["left"] for row in sample],
                             [item["left"] for item in pairs if item["eligible"]])
            self.assertEqual({row["review_verdict"] for row in sample},
                             {"equivalent", "distinct", ""})
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

    def test_empty_and_no_eligible_inputs_have_headers_and_zero_candidate_counts(self):
        for pairs in ([], [pair(0, 0.1, False)]):
            with self.subTest(pairs=pairs), tempfile.TemporaryDirectory() as folder:
                out = Path(folder)
                calibration.write_calibration({"feature_pairs": pairs}, out, 0.2)
                self.assertEqual(read_csv(out / "calibration_sample.csv"), [])
                self.assertIn("review_verdict", (out / "calibration_sample.csv").read_text().splitlines()[0])
                rows = read_csv(out / "threshold_sensitivity.csv")
                self.assertEqual(len(rows), 7)
                for row in rows:
                    self.assertEqual(set(row), {
                        "threshold", "total_pairs", "eligible_pairs", "ineligible_pairs",
                        "candidate_total", "reviewed_candidate_total", "equivalent_candidate_total",
                        "distinct_candidate_total", "unreviewed_candidate_total",
                    })
                    self.assertEqual(row["total_pairs"], str(len(pairs)))
                    self.assertEqual(row["ineligible_pairs"], str(len(pairs)))
                    for field in ("eligible_pairs", "candidate_total", "reviewed_candidate_total",
                                  "equivalent_candidate_total", "distinct_candidate_total",
                                  "unreviewed_candidate_total"):
                        self.assertEqual(row[field], "0")

    def test_duplicate_and_invalid_export_distances_fail_before_writes(self):
        original = pair(0, 0.1)
        reversed_pair = {**original, "left": original["right"], "right": original["left"]}
        for pairs in ([original, original], [original, reversed_pair],
                      *[[{**pair(0, 0.1), "cosine_distance": distance}]
                        for distance in (float("nan"), float("inf"), float("-inf"),
                                         -0.01, 2.01, None, True, "")]):
            with self.subTest(pairs=pairs), tempfile.TemporaryDirectory() as folder:
                out = Path(folder) / "new"
                with self.assertRaises(ValueError):
                    calibration.write_calibration({"feature_pairs": pairs}, out, 0.2)
                self.assertFalse(out.exists())


class CalibrationDecisionTests(unittest.TestCase):
    def test_sensitivity_counts_binary_decisions_and_unreviewed_cutoffs(self):
        pairs = [
            pair(0, 0.1),
            pair(1, 0.15, review_verdict="distinct",
                 review_reason="The exclusion rule changes which comments count."),
            pair(2, 0.2),
            pair(3, 0.25), pair(4, 0.3), pair(5, 2),
            pair(6, 0.1, False),
        ]
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            calibration.write_calibration({"feature_pairs": pairs}, out, 0.2)
            scores = {float(row["threshold"]): row
                      for row in read_csv(out / "threshold_sensitivity.csv")}
            expected = {
                0.1: (1, 1, 1, 0, 0),
                0.15: (2, 2, 1, 1, 0),
                0.2: (3, 3, 2, 1, 0),
                0.25: (4, 3, 2, 1, 1),
                0.3: (5, 3, 2, 1, 2),
                0.35: (5, 3, 2, 1, 2),
                0.4: (5, 3, 2, 1, 2),
            }
            self.assertEqual(set(scores), set(expected))
            for threshold, counts in expected.items():
                row = scores[threshold]
                self.assertEqual(row["total_pairs"], "7")
                self.assertEqual(row["eligible_pairs"], "6")
                self.assertEqual(row["ineligible_pairs"], "1")
                for field, count in zip((
                    "candidate_total", "reviewed_candidate_total", "equivalent_candidate_total",
                    "distinct_candidate_total", "unreviewed_candidate_total",
                ), counts):
                    self.assertEqual(row[field], str(count), (threshold, field))

    def test_rejects_nonbinary_verdicts_and_reviews_on_ineligible_pairs(self):
        cases = [
            pair(0, 0.1, review_verdict=verdict)
            for verdict in ("uncertain", "needs_alignment", "Equivalent", " equivalent ", None, True)
        ]
        cases.extend(pair(0, 0.1, eligible=False, review_verdict=verdict)
                     for verdict in ("equivalent", "distinct"))
        cases.extend(pair(0, 0.1, eligible=value, review_verdict="")
                     for value in ("", "maybe", None))
        for item in cases:
            with self.subTest(item=item), tempfile.TemporaryDirectory() as folder:
                out = Path(folder) / "new"
                with self.assertRaises(ValueError):
                    calibration.write_calibration({"feature_pairs": [item]}, out, 0.2)
                self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
