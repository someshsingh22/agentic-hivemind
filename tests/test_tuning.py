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
