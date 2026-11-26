import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from ssh import equivalence, prompts


def comparison(left=0, right=1, verdict="equivalent", reason="Fixture: identical row-level measurements."):
    return {"left": left, "right": right, "verdict": verdict, "reason": reason}


def candidate_features():
    return [
        {"hypothesis_id": index, "feature": {
            "name": name, "dtype": "binary", "source_columns": ["argument"],
            "definition": "OP argument: explicit uncertainty about the OP's own claim",
            "kind": "prompt",
            "spec": "Output 1 for explicit self-doubt; 0 if absent; null if argument is empty.",
        }}
        for index, name in enumerate(("uncertainty", "self_doubt"))
    ]


def full_config():
    return {"agent": {
        "model": "claude-sonnet-5-5", "max_tokens": 100, "compat": {},
        "base_url": "https://unused.invalid", "api": "anthropic-messages",
        "api_key_env": "UNUSED_KEY", "provider": "unused", "workers": 1,
        "thinking": "off", "tools": "read,bash,submit",
    }}


def submit_review(prompt, system, home, cfg, schema, out):
    # Emulate the external submit wrapper, never the API or model's judgment.
    Path(out).write_text(json.dumps({"comparisons": [comparison()]}))


class ComparisonValidationTests(unittest.TestCase):
    def test_requires_every_pair_once_and_normalizes_order(self):
        items = [comparison(2, 1, "uncertain"), comparison(2, 0, "distinct"), comparison(1, 0)]
        result = equivalence.validate_comparisons(items, 3)
        self.assertEqual(result, [comparison(0, 1), comparison(0, 2, "distinct"),
                                  comparison(1, 2, "uncertain")])
        self.assertEqual(items[0]["left"], 2)
        self.assertEqual(equivalence.validate_comparisons([], 0), [])
        self.assertEqual(equivalence.validate_comparisons([], 1), [])

    def test_needs_alignment_is_a_supported_nonapproval_verdict(self):
        item = comparison(verdict="needs_alignment",
                          reason="Thresholds differ; align row measurements before sharing labels.")
        self.assertEqual(equivalence.validate_comparisons([item], 2), [item])

    def test_invalid_or_incomplete_decisions_fail_closed(self):
        invalid = {
            "missing_all": [],
            "not_list": {"comparisons": [comparison()]},
            "not_object": [None],
            "missing_reason": [{key: value for key, value in comparison().items() if key != "reason"}],
            "extra_field": [{**comparison(), "confidence": 1}],
            "self_pair": [comparison(0, 0)],
            "negative_index": [comparison(-1, 1)],
            "unavailable_index": [comparison(0, 2)],
            "string_index": [comparison("0", 1)],
            "float_index": [comparison(0.0, 1)],
            "boolean_index": [comparison(False, 1)],
            "unknown_verdict": [comparison(verdict="similar")],
            "missing_verdict": [comparison(verdict=None)],
            "blank_reason": [comparison(reason=" \n")],
            "nonstring_reason": [comparison(reason=7)],
            "duplicate_pair": [comparison(), comparison()],
            "reverse_duplicate": [comparison(), comparison(1, 0)],
        }
        for label, items in invalid.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                equivalence.validate_comparisons(items, 2)
        with self.assertRaises(ValueError):
            equivalence.validate_comparisons([comparison(0, 1), comparison(1, 2)], 3)


class EquivalenceReviewerTests(unittest.TestCase):
    def test_identical_inputs_reuse_review_across_instances(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cfg = full_config()
            features = candidate_features()
            with patch.object(equivalence, "agent", side_effect=submit_review) as agent:
                first = equivalence.EquivalenceReviewer(cfg, root)(features)
                second = equivalence.EquivalenceReviewer(copy.deepcopy(cfg), root)(copy.deepcopy(features))
            self.assertEqual(first, [comparison()])
            self.assertEqual(second, first)
            self.assertEqual(agent.call_count, 1)
            self.assertEqual(cfg["agent"]["tools"], "read,bash,submit")
            review_path = agent.call_args.args[-1]
            self.assertEqual(review_path.parent, root / "dedup_reviews")
            self.assertRegex(review_path.name, r"^[0-9a-f]{64}\.json$")
            self.assertEqual(json.loads(review_path.read_text()), {"comparisons": first})
            self.assertEqual(agent.call_args.args[3]["tools"], "submit")

    def test_pair_cache_reuses_verdict_across_threshold_trials(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(equivalence, "agent", side_effect=submit_review) as agent:
                results = []
                for threshold in (0.20, 0.25, 0.30):
                    cfg = {**full_config(), "dedup": {"threshold": threshold}}
                    results.append(equivalence.EquivalenceReviewer(cfg, root)(candidate_features()))
            self.assertEqual(results, [[comparison()]] * 3)
            self.assertEqual(agent.call_count, 1)

    def test_changed_full_rubric_or_hypothesis_identity_needs_new_review(self):
        changes = {
            "name": "uncertainty_revised",
            "definition": "OP argument: explicit uncertainty about another person's claim",
            "spec": "Output 1 for uncertainty anywhere, including quotations; otherwise 0.",
            "source_columns": ["title", "argument"],
            "dtype": "numeric",
            "kind": "code",
        }
        for field, value in changes.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as folder:
                reviewer = equivalence.EquivalenceReviewer(full_config(), Path(folder))
                original = candidate_features()
                changed = copy.deepcopy(original)
                changed[0]["feature"][field] = value
                with patch.object(equivalence, "agent", side_effect=submit_review) as agent:
                    reviewer(original)
                    reviewer(changed)
                    reviewer(changed)
                self.assertEqual(agent.call_count, 2)
                self.assertNotEqual(agent.call_args_list[0].args[-1], agent.call_args_list[1].args[-1])
        with tempfile.TemporaryDirectory() as folder:
            reviewer = equivalence.EquivalenceReviewer(full_config(), Path(folder))
            changed = candidate_features()
            changed[0]["hypothesis_id"] = 50
            with patch.object(equivalence, "agent", side_effect=submit_review) as agent:
                reviewer(candidate_features())
                reviewer(changed)
            self.assertEqual(agent.call_count, 2)

    def test_changed_prompt_schema_or_agent_config_needs_new_review(self):
        for change in ("system_message", "user_template", "schema", "model", "thinking"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                prompt_path = root / "prompts.yaml"
                prompt_path.write_text(prompts.PROMPTS.read_text())
                schema_path = root / "schema.json"
                schema_path.write_text(equivalence.SCHEMA.read_text())
                cfg = full_config()
                with patch.object(prompts, "PROMPTS", prompt_path), \
                        patch.object(equivalence, "SCHEMA", schema_path), \
                        patch.object(equivalence, "agent", side_effect=submit_review) as agent:
                    equivalence.EquivalenceReviewer(cfg, root)(candidate_features())
                    if change in ("system_message", "user_template"):
                        document = yaml.safe_load(prompt_path.read_text())
                        document["agents"]["feature_equivalence"]["prompt_config"][change] += "\nRevised review instruction."
                        prompt_path.write_text(yaml.safe_dump(document))
                    elif change == "schema":
                        document = json.loads(schema_path.read_text())
                        document["description"] = "Revised review schema."
                        schema_path.write_text(json.dumps(document))
                    else:
                        cfg["agent"][change] = "replacement-model" if change == "model" else "high"
                    result = equivalence.EquivalenceReviewer(cfg, root)(candidate_features())
                    equivalence.EquivalenceReviewer(cfg, root)(candidate_features())
                self.assertEqual(result, [comparison()])
                self.assertEqual(agent.call_count, 2)
                self.assertNotEqual(agent.call_args_list[0].args[-1], agent.call_args_list[1].args[-1])

    def test_malformed_external_review_fails_closed_on_creation_and_resume(self):
        malformed = {
            "invalid_json": "not json",
            "wrong_root": json.dumps([comparison()]),
            "extra_root_field": json.dumps({"comparisons": [comparison()], "approved": True}),
            "missing_root_field": "{}",
            "incomplete_pairs": json.dumps({"comparisons": []}),
            "unknown_verdict": json.dumps({"comparisons": [comparison(verdict="similar")]}),
            "duplicate_pairs": json.dumps({"comparisons": [comparison(), comparison(1, 0)]}),
        }
        for label, contents in malformed.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)

                def submit_malformed(prompt, system, home, cfg, schema, out):
                    Path(out).write_text(contents)

                with patch.object(equivalence, "agent", side_effect=submit_malformed) as agent:
                    with self.assertRaises(ValueError):
                        equivalence.EquivalenceReviewer(full_config(), root)(candidate_features())
                    with self.assertRaises(ValueError):
                        equivalence.EquivalenceReviewer(full_config(), root)(candidate_features())
                self.assertEqual(agent.call_count, 1)

    def test_nonapproval_external_decisions_are_preserved(self):
        for verdict in ("distinct", "uncertain", "needs_alignment"):
            with self.subTest(verdict=verdict), tempfile.TemporaryDirectory() as folder:
                def submit_decision(prompt, system, home, cfg, schema, out):
                    Path(out).write_text(json.dumps({"comparisons": [comparison(verdict=verdict)]}))

                with patch.object(equivalence, "agent", side_effect=submit_decision):
                    result = equivalence.EquivalenceReviewer(full_config(), Path(folder))(candidate_features())
                self.assertEqual(result, [comparison(verdict=verdict)])

    def test_cached_review_requires_matching_input_stamp(self):
        for change in ("missing", "changed", "malformed"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                reviewer = equivalence.EquivalenceReviewer(full_config(), root)
                with patch.object(equivalence, "agent", side_effect=submit_review) as agent:
                    reviewer(candidate_features())
                    review_path = agent.call_args.args[-1]
                    stamp = review_path.with_suffix(".inputs.json")
                    if change == "missing":
                        stamp.unlink()
                    elif change == "changed":
                        document = json.loads(stamp.read_text())
                        document["features"][0]["feature"]["spec"] = "An unrelated cached rubric."
                        stamp.write_text(json.dumps(document))
                    else:
                        stamp.write_text("not json")
                    with self.assertRaises(ValueError):
                        reviewer(candidate_features())
                self.assertEqual(agent.call_count, 1)


if __name__ == "__main__":
    unittest.main()
