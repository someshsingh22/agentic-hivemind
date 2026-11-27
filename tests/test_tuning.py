import csv
import json
import math
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml


class TuningTests(unittest.TestCase):
    def test_all_features_share_vectors_and_resume_uses_existing_decomposition(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "data.csv"
            source.write_text("argument,title,delta,post_id\nhello,title,0,a\nworld,other,1,b\n")
            hypotheses = root / "claims.json"
            hypotheses.write_text(json.dumps({"hypotheses": ["Claim A", "Claim B", "Claim C", "Claim D"]}))
            env = root / "env.yaml"
            env.write_text(yaml.safe_dump({"name": "smoke", "csv": str(source),
                "hypotheses": str(hypotheses), "target": "delta", "group": "post_id"}))
            cfg = yaml.safe_load(Path("config.yaml").read_text())
            cfg["out"] = str(root / "runs")
            cfg["n_hypotheses"] = 0
            config = root / "config.yaml"
            config.write_text(yaml.safe_dump(cfg))
            names = ["uncertainty_a", "uncertainty_b", "uncertainty_c", "title_uncertainty"]

            def submit_feature(prompt, system, home, agent_cfg, schema, out):
                index = int(Path(out).stem)
                feature = {"name": names[index], "dtype": "binary",
                    "source_columns": ["title" if index == 3 else "argument"],
                    "definition": "Explicit uncertainty", "kind": "code" if index == 1 else "prompt",
                    "spec": "int('uncertain' in str(row['argument']))" if index == 1 else
                            "Return 1 for explicit uncertainty, 0 for absence, null if missing."}
                Path(out).write_text(json.dumps({"features": [feature], "h0": "No association",
                    "h1": "Positive association", "rhs": names[index], "term": names[index], "sign": "+", "note": ""}))

            angles = [0, math.acos(.78), math.acos(.78) + math.acos(.72), 0]
            vectors = np.array([[math.cos(angle), math.sin(angle)] for angle in angles])
            out = root / "runs/smoke"
            argv = ["ssh", "tune", str(env), "--config", str(config)]
            with patch("ssh.decompose.agent", side_effect=submit_feature) as decomposer, \
                 patch("ssh.models.agent", side_effect=AssertionError("No judge calls permitted")) as judge, \
                 patch("ssh.models.embed", return_value=vectors) as embedder:
                for attempt in range(2):
                    if attempt:
                        hypotheses.write_text(json.dumps({"hypotheses": ["Changed A", "Changed B", "Changed C", "Changed D"]}))
                        source.write_text("argument,title,delta,post_id\nchanged,title,1,z\n")
                        cfg["agent"]["model"] = "changed-model-with-cached-decompositions"
                        config.write_text(yaml.safe_dump(cfg))
                    with patch.object(sys, "argv", argv):
                        runpy.run_module("ssh", run_name="__main__")
                    self.assertEqual(decomposer.call_count, 4)
                    judge.assert_not_called()
                    self.assertEqual(embedder.call_count, attempt + 1)
                    texts = embedder.call_args.args[0]
                    self.assertEqual(len(texts), 4)
                    self.assertTrue(all(any(name in text for text in texts) for name in names))
                    plans = {threshold: json.loads((out / f"threshold_{threshold}/plan.json").read_text())
                             for threshold in ("0.20", "0.25", "0.30")}
                    self.assertEqual([len(plan["features"]) for plan in plans.values()], [4, 3, 3])
                    for plan in plans.values():
                        self.assertNotIn("dedup_reviews", plan)
                        self.assertEqual(len(plan["feature_pairs"]), 6)
                        self.assertLessEqual(max(len(f["members"]) for f in plan["features"]), 2)
                        title_pairs = [p for p in plan["feature_pairs"] if "title" in
                                       p["left_feature"]["source_columns"] + p["right_feature"]["source_columns"]]
                        self.assertEqual(len(title_pairs), 3)
                        self.assertTrue(all(not p["eligible"] and not p["merged"] for p in title_pairs))
                        expected_claims = [f"{'Changed' if attempt else 'Claim'} {letter}" for letter in "ABCD"]
                        self.assertEqual([h["hypothesis"] for h in plan["hypotheses"]], expected_claims)
                    distances = [[p["cosine_distance"] for p in plan["feature_pairs"]] for plan in plans.values()]
                    self.assertEqual(distances[0], distances[1])
                    self.assertEqual(distances[1], distances[2])
                    summary = json.loads((out / "tuning_summary.json").read_text())
                    self.assertEqual(set(summary), {"0.20", "0.25", "0.30"})
                    self.assertTrue(all(len(counts) == 5 for counts in summary.values()))
                    self.assertFalse(any("review" in key or "equivalent" in key for counts in summary.values() for key in counts))
                    with (out / "threshold_0.30/calibration_sample.csv").open() as stream:
                        sample = list(csv.DictReader(stream))
                    self.assertEqual(len(sample), 3)
                    self.assertFalse(any("review" in key for row in sample for key in row))
                    self.assertFalse((out / "automated").exists())
                    self.assertFalse((out / "decomposition_inputs.json").exists())
                    self.assertFalse((out / "results.csv").exists())

    def test_description_comparison_keeps_saved_bank_and_separates_vector_blocks(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            claims = []
            for owner, name, source, spec in (
                (8, "exactly_one_comment", "OC_comments", "Return 1 for exactly one comment, otherwise 0."),
                (2, "at_least_two_comments", "OC_comments", "Return 1 for at least two comments, otherwise 0."),
                (30, "title_mentions_comments", "title", "Return 1 if the title mentions comments, otherwise 0."),
            ):
                claims.append({"id": owner, "hypothesis": f"Claim {owner}", "features": [{
                    "name": name, "dtype": "binary", "source_columns": [source],
                    "definition": "Comment-related indicator", "kind": "prompt", "spec": spec,
                }], "h0": "No association", "h1": "Positive association", "rhs": name,
                    "term": name, "sign": "+", "note": ""})
            source = root / "decomposed.json"
            source.write_text(json.dumps({"env": {"name": "smoke", "target": "delta", "group": "post_id"},
                                         "decomposed": claims}))
            original = source.read_bytes()
            baseline = root / "threshold_0.25/plan.json"
            baseline.parent.mkdir()
            baseline.write_text('{"existing_baseline": true}')
            cfg = yaml.safe_load(Path("config.yaml").read_text())
            cfg["out"] = str(root / "unused")
            cfg["n_hypotheses"] = 1
            config = root / "config.yaml"
            config.write_text(yaml.safe_dump(cfg))
            vectors = np.array([[1, 0], [.9, math.sqrt(1 - .9 ** 2)], [1, 0],
                                [1, 0], [0, 1], [1, 0]])
            argv = ["ssh", "compare_descriptions", str(source), "--config", str(config)]
            with patch("ssh.models.embed", return_value=vectors), \
                 patch("ssh.models.agent", side_effect=AssertionError("No generation or judge permitted")), \
                 patch("ssh.decompose.decompose_all", side_effect=AssertionError("Use the saved bank")), \
                 patch.object(sys, "argv", argv):
                runpy.run_module("ssh", run_name="__main__")
            out = root / "description_comparison"
            for representation, expected_groups in (
                ("short_definition", {frozenset({8, 2}), frozenset({30})}),
                ("full_description", {frozenset({8}), frozenset({2}), frozenset({30})}),
            ):
                for threshold in ("0.20", "0.25", "0.30"):
                    plan = json.loads((out / representation / f"threshold_{threshold}/plan.json").read_text())
                    groups = {frozenset(m["hypothesis"] for m in f["members"]) for f in plan["features"]}
                    self.assertEqual(groups, expected_groups)
                    self.assertEqual({h["id"] for h in plan["hypotheses"]}, {8, 2, 30})
                    title_pairs = [p for p in plan["feature_pairs"] if p["right_hypothesis"] == 30]
                    self.assertTrue(all(not p["eligible"] and not p["merged"] for p in title_pairs))
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(baseline.read_text(), '{"existing_baseline": true}')
            self.assertFalse((root / "unused").exists())
