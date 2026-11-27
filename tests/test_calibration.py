import copy
import csv
import json
import random
import tempfile
import unittest
from pathlib import Path

from ssh import calibration


def feature(name, kind="prompt"):
    return {"name": name, "source_columns": ["argument"], "dtype": "binary",
            "definition": "Explicit uncertainty", "kind": kind,
            "spec": "Return uncertainty indicator"}


def pair(index, distance, eligible=True):
    return {"left": f"h00000:a{index}", "right": f"h00001:b{index}",
            "left_hypothesis": 0, "right_hypothesis": 1,
            "left_feature": feature(f"a{index}"), "right_feature": feature(f"b{index}", "code"),
            "cosine_distance": distance, "eligible": eligible,
            "within_threshold": eligible and distance <= .2, "merged": eligible and distance <= .2}


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


class CalibrationTests(unittest.TestCase):
    def test_all_feature_types_and_source_gates_are_exported_without_mutation(self):
        pairs = [pair(0, 0), pair(1, .1, False), pair(2, .2, False), pair(3, 2, False)]
        pairs[1]["right_feature"]["source_columns"] = ["title"]
        pairs[2]["right_feature"]["dtype"] = "numeric"
        pairs[3]["right_hypothesis"] = 0
        pairs[0]["left_feature"]["spec"] = 'Line 1, "quoted"\nLine 2: café'
        original = copy.deepcopy(pairs)
        with tempfile.TemporaryDirectory() as folder:
            calibration.write_calibration({"feature_pairs": pairs}, folder, .2)
            rows = read_csv(Path(folder) / "feature_pairs.csv")
            self.assertEqual(len(rows), 4)
            for item, row in zip(pairs, rows):
                self.assertEqual(set(row), set(calibration.PAIR_FIELDS))
                self.assertFalse(any("review" in key or "candidate_cluster" in key for key in row))
                for side in ("left", "right"):
                    self.assertEqual(json.loads(row[f"{side}_feature"]), item[f"{side}_feature"])
                    self.assertEqual(json.loads(row[f"{side}_source_columns"]), item[f"{side}_feature"]["source_columns"])
                for field in ("eligible", "within_threshold", "merged"):
                    self.assertEqual(row[field], str(item[field]))
                self.assertEqual(float(row["cosine_distance"]), item["cosine_distance"])
            self.assertEqual(len(read_csv(Path(folder) / "calibration_sample.csv")), 1)
        self.assertEqual(pairs, original)

    def test_bin_boundaries_include_zero_and_two_once(self):
        distances = [0, .1, .15, .2, .25, .3, .4, 2]
        with tempfile.TemporaryDirectory() as folder:
            calibration.write_calibration({"feature_pairs": [pair(i, d) for i, d in enumerate(distances)]}, folder, .2)
            rows = read_csv(Path(folder) / "calibration_sample.csv")
            self.assertEqual(sorted(float(row["cosine_distance"]) for row in rows), distances)

    def test_seed_42_sample_and_csv_ignore_input_order(self):
        distances = [0, .1, .15, .2, .25, .3, .4]
        pairs = [pair(b * 100 + i, d) for b, d in enumerate(distances) for i in range(24)]
        pairs.extend(pair(1000 + i, .1, False) for i in range(20))
        shuffled = copy.deepcopy(pairs)
        random.Random(91).shuffle(shuffled)
        with tempfile.TemporaryDirectory() as folder:
            first, second = Path(folder) / "first", Path(folder) / "second"
            calibration.write_calibration({"feature_pairs": pairs}, first, .2)
            calibration.write_calibration({"feature_pairs": shuffled}, second, .2)
            for filename in ("feature_pairs.csv", "calibration_sample.csv", "threshold_sensitivity.csv"):
                self.assertEqual((first / filename).read_bytes(), (second / filename).read_bytes())
            expected = []
            rng = random.Random(42)
            ordered = sorted(pairs, key=lambda row: tuple(sorted((row["left"], row["right"]))))
            for distance in distances:
                expected.extend(rng.sample([p for p in ordered if p["eligible"] and p["cosine_distance"] == distance], 10))
            rows = read_csv(first / "calibration_sample.csv")
            self.assertEqual({row["left"] for row in rows}, {row["left"] for row in expected})
            self.assertEqual(len(rows), 70)

    def test_sensitivity_counts_candidates_at_inclusive_boundaries(self):
        pairs = [pair(i, d) for i, d in enumerate([.1, .15, .2, .25, .3, 2])]
        pairs.append(pair(6, .1, False))
        for items in (pairs, [], [pair(0, .1, False)]):
            with self.subTest(items=items), tempfile.TemporaryDirectory() as folder:
                calibration.write_calibration({"feature_pairs": items}, folder, .2)
                rows = read_csv(Path(folder) / "threshold_sensitivity.csv")
                self.assertEqual(len(rows), len(calibration.THRESHOLDS))
                for row in rows:
                    self.assertEqual(set(row), {"threshold", "total_pairs", "eligible_pairs", "ineligible_pairs", "candidate_total"})
                    eligible = [p for p in items if p["eligible"]]
                    self.assertEqual(int(row["total_pairs"]), len(items))
                    self.assertEqual(int(row["eligible_pairs"]), len(eligible))
                    self.assertEqual(int(row["ineligible_pairs"]), len(items) - len(eligible))
                    self.assertEqual(int(row["candidate_total"]), sum(p["cosine_distance"] <= float(row["threshold"]) for p in eligible))

    def test_invalid_and_duplicate_pairs_fail_before_writes(self):
        original = pair(0, .1)
        reverse = {**original, "left": original["right"], "right": original["left"]}
        cases = [[original, original], [original, reverse]]
        cases.extend([{**original, "cosine_distance": value}] for value in (float("nan"), float("inf"), -.01, 2.01, None, True, ""))
        cases.extend([{**original, "eligible": value}] for value in ("", "maybe", None))
        cases.append([{**original, "right": original["left"]}])
        for items in cases:
            with self.subTest(items=items), tempfile.TemporaryDirectory() as folder:
                out = Path(folder) / "new"
                with self.assertRaises(ValueError):
                    calibration.write_calibration({"feature_pairs": items}, out, .2)
                self.assertFalse(out.exists())
