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
    def test_shared_measurements_and_reviews_preserve_threshold_boundaries_on_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "data.csv"
            source.write_text("argument,delta,post_id\nhello,0,a\nworld,1,b\n")
            hypotheses = root / "claims.json"
            hypotheses.write_text(json.dumps({"hypotheses": ["Claim A", "Claim B", "Claim C"]}))
            env = root / "env.yaml"
            env.write_text(yaml.safe_dump({"name": "smoke", "csv": str(source),
                "hypotheses": str(hypotheses), "target": "delta", "group": "post_id"}))
            cfg = yaml.safe_load(Path("config.yaml").read_text())
            cfg["out"] = str(root / "runs")
            config = root / "config.yaml"
            config.write_text(yaml.safe_dump(cfg))
            names = ["uncertainty_a", "uncertainty_b", "uncertainty_c"]

            def submit_feature(prompt, system, home, agent_cfg, schema, out):
                index = int(Path(out).stem)
                feature = {"name": names[index], "dtype": "binary", "source_columns": ["argument"],
                    "definition": "OP expresses uncertainty", "kind": "prompt",
                    "spec": "Read argument; return 1 for explicit OP uncertainty, 0 for assessable absence, null if missing."}
                Path(out).write_text(json.dumps({"features": [feature], "h0": "No association",
                    "h1": "Positive association", "rhs": names[index], "term": names[index], "sign": "+", "note": ""}))

            def submit_review(prompt, system, home, agent_cfg, schema, out):
                Path(out).write_text(json.dumps({"comparisons": [{"left": 0, "right": 1,
                    "verdict": "equivalent", "reason": "Fixture: same full measurement."}]}))

            # A/B distance .22 enters only .25/.30; B/C is .28, only .30.
            angles = [0, math.acos(.78), math.acos(.78) + math.acos(.72)]
            vectors = np.array([[math.cos(a), math.sin(a)] for a in angles])
            out = root / "runs/smoke/automated"
            legacy_plan = out.parent / "threshold_0.25" / "plan.json"
            legacy_plan.parent.mkdir(parents=True)
            legacy_plan.write_text('{"protocol": "previous pending-review run"}')
            argv = ["ssh", "tune", str(env), "--config", str(config)]
            with patch("ssh.decompose.agent", side_effect=submit_feature) as decomposer, \
                 patch("ssh.equivalence.agent", side_effect=submit_review) as reviewer, \
                 patch("ssh.models.embed", return_value=vectors) as embedder:
                for attempt in range(2):
                    with patch.object(sys, "argv", argv):
                        runpy.run_module("ssh", run_name="__main__")
                    self.assertEqual(decomposer.call_count, 3)
                    self.assertEqual(reviewer.call_count, 2, "Shared review cache must avoid repeated paid reviews")
                    self.assertEqual(embedder.call_count, attempt + 1)
                    plans = {threshold: json.loads((out / f"threshold_{threshold}" / "plan.json").read_text())
                             for threshold in ("0.20", "0.25", "0.30")}
                    self.assertEqual([len(plans[t]["dedup_reviews"]) for t in plans], [0, 1, 2])
                    self.assertEqual([len(plans[t]["features"]) for t in plans], [3, 2, 2])
                    for plan in plans.values():
                        self.assertEqual([h["hypothesis"] for h in plan["hypotheses"]], ["Claim A", "Claim B", "Claim C"])
                        self.assertLessEqual(max(len(f["members"]) for f in plan["features"]), 2)
                    summary = json.loads((out / "tuning_summary.json").read_text())
                    self.assertEqual(set(summary), {"0.20", "0.25", "0.30"})
                    self.assertEqual(legacy_plan.read_text(), '{"protocol": "previous pending-review run"}')
                    self.assertEqual(summary["0.30"]["reviewed_pairs"], 2)
                    self.assertEqual(summary["0.30"]["equivalent_pairs"], 2)
                    self.assertEqual(summary["0.30"]["distinct_pairs"], 0)
                    with (out / "threshold_0.30/calibration_sample.csv").open() as stream:
                        sample = list(csv.DictReader(stream))
                    self.assertEqual({row["review_verdict"] for row in sample}, {"", "equivalent"})
                    self.assertFalse((out / "results.csv").exists())
