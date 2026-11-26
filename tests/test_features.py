import copy
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import yaml

from ssh import decompose as decomposition
from ssh import dedup as deduplication
from ssh import prompts
from ssh.features import validate_features


ENV = {"name": "cmv", "target": "delta", "group": "post_id"}
CFG = {"embedding": {}, "dedup": {"threshold": 0.25}}


def feature(name, sources=("argument",), kind="prompt", dtype="binary", spec=None):
    return {
        "name": name,
        "dtype": dtype,
        "source_columns": list(sources),
        "definition": "OP argument: explicit uncertainty about the OP's own claim",
        "kind": kind,
        "spec": spec or "Read this row. Output 1 for explicit uncertainty, 0 if absent, null if unassessable.",
    }


def hypothesis(index, features, rhs=None):
    term = features[0]["name"] if features else None
    return {
        "id": index,
        "hypothesis": f"Claim {index}",
        "features": features,
        "h0": f"{term} has no association",
        "h1": f"{term} has a positive association",
        "rhs": rhs or term or "1",
        "term": term,
        "sign": "+",
        "note": "",
    }


def identical_vectors(texts, cfg):
    return np.tile([1.0, 0.0], (len(texts), 1))


def equivalent_review(cluster):
    return [
        {"left": left, "right": right, "verdict": "equivalent",
         "reason": "Deterministic fixture: these measurements have identical row-level outputs."}
        for left in range(len(cluster))
        for right in range(left + 1, len(cluster))
    ]


class DeduplicationTests(unittest.TestCase):
    def test_matching_source_sets_merge_and_rewrite_claims(self):
        claims = [
            hypothesis(0, [feature("uncertainty", ("argument", "title"))]),
            hypothesis(1, [feature("self_doubt", ("title", "argument"))]),
        ]
        claims[1]["sign"] = "-"
        with patch.object(deduplication, "embed", identical_vectors):
            plan = deduplication.dedup(claims, ENV, CFG, equivalent_review)
        self.assertEqual([f["name"] for f in plan["features"]], ["uncertainty"])
        self.assertEqual([m["hypothesis"] for m in plan["features"][0]["members"]], [0, 1])
        self.assertEqual(plan["hypotheses"][1]["formula"], "delta ~ uncertainty")
        self.assertEqual(plan["hypotheses"][1]["term"], "uncertainty")
        self.assertEqual(plan["hypotheses"][1]["sign"], "-")
        self.assertEqual([h["hypothesis"] for h in plan["hypotheses"]], ["Claim 0", "Claim 1"])

    def test_different_source_sets_cannot_merge_even_at_zero_distance(self):
        claims = [hypothesis(i, [feature(f"f{i}", sources)]) for i, sources in enumerate([
            ("argument",), ("title",), ("argument", "title"),
        ])]
        with patch.object(deduplication, "embed", identical_vectors):
            plan = deduplication.dedup(claims, ENV, CFG, equivalent_review)
        self.assertEqual([f["source_columns"] for f in plan["features"]], [
            ["argument"], ["title"], ["argument", "title"],
        ])
        self.assertEqual([len(f["members"]) for f in plan["features"]], [1, 1, 1])

    def test_code_features_stay_distinct_and_use_no_embedding_service(self):
        claims = [hypothesis(i, [feature(name, kind="code", spec=f"df['{name}'] = 1")])
                  for i, name in enumerate(["count", "count", "count_2"])]
        with patch.object(deduplication, "embed", side_effect=AssertionError("Unexpected embedding request")):
            plan = deduplication.dedup(claims, ENV, CFG, equivalent_review)
        names = [f["name"] for f in plan["features"]]
        self.assertEqual(names, ["count", "count_2", "count_2_2"])
        self.assertEqual([h["term"] for h in plan["hypotheses"]], names)
        self.assertEqual([f["spec"] for f in plan["features"]], [f"df['{name}'] = 1" for name in names])

    def test_code_never_merges_with_semantic_feature(self):
        claims = [hypothesis(0, [feature("uncertainty")]),
                  hypothesis(1, [feature("uncertainty", kind="code")]),
                  hypothesis(2, [feature("self_doubt")])]
        with patch.object(deduplication, "embed", identical_vectors):
            plan = deduplication.dedup(claims, ENV, CFG, equivalent_review)
        self.assertEqual([(f["kind"], len(f["members"])) for f in plan["features"]],
                         [("prompt", 2), ("code", 1)])
        self.assertEqual([h["term"] for h in plan["hypotheses"]],
                         ["uncertainty", "uncertainty_2", "uncertainty"])

    def test_derived_code_uses_canonical_semantic_measurement(self):
        claims = [hypothesis(0, [feature("intensity", dtype="numeric")]),
                  hypothesis(1, [feature("emotion", dtype="numeric"),
                                 feature("high_emotion", kind="code",
                                         spec="df['high_emotion'] = df['emotion'] >= 7")],
                             rhs="high_emotion + emotion")]
        with patch.object(deduplication, "embed", identical_vectors):
            plan = deduplication.dedup(claims, ENV, CFG, equivalent_review)
        derived = next(f for f in plan["features"] if f["name"] == "high_emotion")
        self.assertEqual(derived["spec"], "df['high_emotion'] = df['intensity'] >= 7")
        self.assertEqual(plan["hypotheses"][1]["formula"], "delta ~ high_emotion + intensity")
        self.assertEqual(plan["hypotheses"][1]["term"], "intensity")
        self.assertEqual(plan["hypotheses"][1]["h0"], "intensity has no association")
        self.assertEqual(plan["hypotheses"][1]["h1"], "intensity has a positive association")

    def test_same_hypothesis_and_different_dtypes_remain_distinct(self):
        claims = [hypothesis(0, [feature("uncertain"), feature("certain")]),
                  hypothesis(1, [feature("intensity", dtype="numeric")])]
        with patch.object(deduplication, "embed", identical_vectors):
            plan = deduplication.dedup(claims, ENV, CFG, equivalent_review)
        self.assertEqual([f["name"] for f in plan["features"]], ["uncertain", "certain", "intensity"])

    def test_complete_linkage_does_not_chain_distant_endpoints(self):
        vectors = np.array([[math.cos(a), math.sin(a)] for a in [0, 0.5, 1.0]])
        claims = [hypothesis(i, [feature(f"f{i}")]) for i in range(3)]
        with patch.object(deduplication, "embed", return_value=vectors):
            plan = deduplication.dedup(claims, ENV, CFG, equivalent_review)
        groups = [{m["hypothesis"] for m in f["members"]} for f in plan["features"]]
        self.assertEqual(sorted(map(len, groups)), [1, 2])
        self.assertFalse(any({0, 2} <= group for group in groups))

    def test_empty_features_and_one_semantic_feature_need_no_embeddings(self):
        with patch.object(deduplication, "embed", side_effect=AssertionError("Unexpected embedding request")):
            empty = deduplication.dedup([hypothesis(0, [])], ENV, CFG, equivalent_review)
            single = deduplication.dedup([hypothesis(0, [feature("tone")])], ENV, CFG, equivalent_review)
        self.assertEqual(empty["features"], [])
        self.assertIsNone(empty["hypotheses"][0]["term"])
        self.assertEqual(single["features"][0]["name"], "tone")

    def test_missing_source_metadata_is_not_silently_merged(self):
        item = feature("tone")
        del item["source_columns"]
        with self.assertRaises(ValueError):
            deduplication.dedup([hypothesis(0, [item])], ENV, CFG, equivalent_review)

    def test_seeded_simulation_preserves_merge_invariants(self):
        rng = np.random.default_rng(42)
        claims, semantic_names = [], []
        for index in range(12):
            items = []
            for number in range(3):
                name = f"f{index}_{number}"
                kind = "code" if number == 2 else "prompt"
                sources = [("argument",), ("title",), ("argument", "title")][rng.integers(3)]
                items.append(feature(name, sources, kind=kind, dtype=["binary", "numeric"][rng.integers(2)]))
                if kind == "prompt":
                    semantic_names.append(name)
            claims.append(hypothesis(index, items))
        vectors = rng.normal(size=(len(semantic_names), 4))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        by_name = dict(zip(semantic_names, vectors))
        with patch.object(deduplication, "embed", return_value=vectors):
            plan = deduplication.dedup(claims, ENV, CFG, equivalent_review)
        self.assertEqual(sum(len(f["members"]) for f in plan["features"]), 36)
        for canonical in plan["features"]:
            members = canonical["members"]
            if canonical["kind"] == "code":
                self.assertEqual(len(members), 1)
                continue
            self.assertEqual(len({m["hypothesis"] for m in members}), len(members))
            self.assertEqual(len({tuple(sorted(m["source_columns"])) for m in members}), 1)
            self.assertEqual(len({m["dtype"] for m in members}), 1)
            for left in members:
                for right in members:
                    distance = 1 - by_name[left["name"]] @ by_name[right["name"]]
                    self.assertLessEqual(distance, 0.25 + 1e-12)

    def test_calibrated_bad_clusters_stay_separate_when_reviewer_rejects(self):
        # These are the completed calibration's exact rubrics. Stubbed verdicts test
        # deduplication behavior, not an external model's semantic judgment.
        calibration = json.loads(
            (Path(__file__).resolve().parents[1] / "runs/cmv/plan.json").read_text()
        )
        pairs = [
            ("any_all", (2, "oc_relies_on_abstract_normative_principle"),
             (16, "oc_only_abstract_normative")),
            ("quote_reference", (54, "any_comment_quotes_op"),
             (87, "any_comment_quotes_or_references_op_text")),
            ("complexity_quality", (48, "oc_cognitive_complexity"),
             (98, "oc_well_reasoned_score")),
            ("evidence_numeric", (59, "oc_specific_empirical_evidence"),
             (99, "any_oc_comment_cites_numerical_data")),
        ]
        members = {
            (member["hypothesis"], member["name"]): member
            for canonical in calibration["features"] for member in canonical["members"]
        }
        for label, left, right in pairs:
            claims = [
                hypothesis(owner, [{key: members[(owner, name)][key]
                                    for key in feature(name)}])
                for owner, name in (left, right)
            ]
            for verdict in ("distinct", "uncertain", "needs_alignment"):
                with self.subTest(pair=label, verdict=verdict):
                    decisions = [{"left": 0, "right": 1, "verdict": verdict,
                                  "reason": f"Fixture decision for {label}: do not share annotations."}]
                    reviewer = Mock(return_value=decisions)
                    with patch.object(deduplication, "embed", identical_vectors):
                        plan = deduplication.dedup(claims, ENV, CFG, reviewer)
                    self.assertEqual(len(plan["features"]), 2)
                    self.assertEqual([len(item["members"]) for item in plan["features"]], [1, 1])
                    self.assertEqual(plan["hypotheses"][0]["term"], left[1])
                    self.assertEqual(plan["hypotheses"][1]["term"], right[1])
                    self.assertEqual(plan["dedup_reviews"][0]["comparisons"], decisions)
                    reviewer.assert_called_once_with([
                        {"hypothesis_id": claim["id"], "feature": claim["features"][0]}
                        for claim in claims
                    ])

    def test_all_candidate_pairs_and_distances_are_saved(self):
        vectors = np.array([[math.cos(a), math.sin(a)] for a in [0, 0.1, 0.2, 0.3]])
        claims = [hypothesis(i, [feature(f"f{i}")]) for i in range(4)]
        reviewer = Mock(side_effect=equivalent_review)
        with patch.object(deduplication, "embed", return_value=vectors):
            plan = deduplication.dedup(claims, ENV, CFG, reviewer)
        self.assertEqual(len(plan["features"]), 1)
        reviewer.assert_called_once()
        review, = plan["dedup_reviews"]
        self.assertEqual(review["features"], [
            {"hypothesis_id": claim["id"], "feature": claim["features"][0]} for claim in claims
        ])
        expected_pairs = {(left, right) for left in range(4) for right in range(left + 1, 4)}
        self.assertEqual({(item["left"], item["right"]) for item in review["comparisons"]}, expected_pairs)
        self.assertEqual(len(review["comparisons"]), 6)
        self.assertEqual({(item["left"], item["right"]) for item in review["distances"]}, expected_pairs)
        self.assertEqual(len(review["distances"]), 6)
        for item in review["distances"]:
            self.assertAlmostEqual(item["cosine_distance"],
                                   1 - vectors[item["left"]] @ vectors[item["right"]])
        self.assertEqual(len(plan["feature_pairs"]), 6)
        exported = {(item["left_hypothesis"], item["right_hypothesis"]): item
                    for item in plan["feature_pairs"]}
        self.assertEqual(set(exported), expected_pairs)
        for indices, item in exported.items():
            self.assertAlmostEqual(item["cosine_distance"], 1 - vectors[indices[0]] @ vectors[indices[1]])
            self.assertTrue(item["eligible"])
            self.assertTrue(item["within_threshold"])
            self.assertTrue(item["same_candidate_cluster"])
            self.assertEqual(item["review_verdict"], "equivalent")
            self.assertTrue(item["review_reason"])
            self.assertTrue(item["merged"])

    def test_nontransitive_semantic_approvals_cannot_merge_all_three(self):
        claims = [hypothesis(i, [feature(name)]) for i, name in enumerate(["a", "b", "c"])]
        decisions = [
            {"left": 0, "right": 1, "verdict": "equivalent", "reason": "Fixture approves A/B."},
            {"left": 1, "right": 2, "verdict": "equivalent", "reason": "Fixture approves B/C."},
            {"left": 0, "right": 2, "verdict": "distinct", "reason": "Fixture rejects A/C."},
        ]
        with patch.object(deduplication, "embed", identical_vectors):
            plan = deduplication.dedup(claims, ENV, CFG, lambda cluster: decisions)
        groups = [{member["hypothesis"] for member in item["members"]} for item in plan["features"]]
        self.assertEqual(sorted(map(len, groups)), [1, 2])
        self.assertFalse(any({0, 2} <= group for group in groups))
        approved = {frozenset((0, 1)), frozenset((1, 2))}
        self.assertTrue(all(len(group) == 1 or frozenset(group) in approved for group in groups))

    def test_direct_any_and_share_are_distinct_row_measurements(self):
        claims = [
            hypothesis(0, [feature("any_uncertain", ("OC_comments",),
                                  spec="Output 1 if any comment expresses uncertainty; 0 if none; null if unavailable.")]),
            hypothesis(1, [feature("uncertain_share", ("OC_comments",), dtype="numeric",
                                  spec="Output the share of comments expressing uncertainty; null for an empty list.")]),
        ]
        reviewer = Mock(side_effect=AssertionError("Different dtypes cannot share row labels"))
        with patch.object(deduplication, "embed", identical_vectors):
            plan = deduplication.dedup(claims, ENV, CFG, reviewer)
        self.assertEqual([item["name"] for item in plan["features"]],
                         ["any_uncertain", "uncertain_share"])
        self.assertEqual([item["term"] for item in plan["hypotheses"]],
                         ["any_uncertain", "uncertain_share"])
        reviewer.assert_not_called()

    def test_same_dtype_direct_aggregations_require_explicit_equivalence(self):
        claims = [
            hypothesis(0, [feature("uncertain_count", ("OC_comments",), dtype="numeric",
                                  spec="Count comments expressing uncertainty in this row.")]),
            hypothesis(1, [feature("uncertain_share", ("OC_comments",), dtype="numeric",
                                  spec="Return the share of comments expressing uncertainty in this row.")]),
        ]
        for verdict, expected in (("distinct", 2), ("needs_alignment", 2), ("equivalent", 1)):
            with self.subTest(verdict=verdict):
                decisions = [{"left": 0, "right": 1, "verdict": verdict,
                              "reason": "Explicit deterministic reviewer decision."}]
                with patch.object(deduplication, "embed", identical_vectors):
                    plan = deduplication.dedup(claims, ENV, CFG, lambda cluster: decisions)
                self.assertEqual(len(plan["features"]), expected)
                self.assertEqual(plan["feature_pairs"][0]["merged"], verdict == "equivalent")

    def test_needs_alignment_preserves_features_and_review_reason(self):
        claims = [hypothesis(owner, [feature(name)])
                  for owner, name in enumerate(("weak_evidence", "strong_evidence"))]
        reason = "Same evidence concept but different cutoffs; sharing labels requires alignment."
        decisions = [{"left": 0, "right": 1, "verdict": "needs_alignment", "reason": reason}]
        with patch.object(deduplication, "embed", identical_vectors):
            plan = deduplication.dedup(claims, ENV, CFG, lambda cluster: decisions)
        self.assertEqual(len(plan["features"]), 2)
        self.assertEqual(plan["dedup_reviews"][0]["comparisons"], decisions)
        pair, = plan["feature_pairs"]
        self.assertEqual(pair["review_verdict"], "needs_alignment")
        self.assertEqual(pair["review_reason"], reason)
        self.assertFalse(pair["merged"])

    def test_row_paraphrases_keep_opposite_direction_hypotheses(self):
        claims = [hypothesis(owner, [feature(name, ("OC_comments",))])
                  for owner, name in enumerate(("uncertainty", "self_doubt"))]
        claims[1].update({"sign": "-", "h1": "self_doubt has a negative association"})
        with patch.object(deduplication, "embed", identical_vectors):
            plan = deduplication.dedup(claims, ENV, CFG, equivalent_review)
        self.assertEqual(len(plan["features"]), 1)
        self.assertEqual(len(plan["hypotheses"]), 2)
        self.assertEqual([item["sign"] for item in plan["hypotheses"]], ["+", "-"])
        self.assertEqual([item["term"] for item in plan["hypotheses"]], ["uncertainty", "uncertainty"])
        self.assertEqual(plan["hypotheses"][0]["h1"], "uncertainty has a positive association")
        self.assertEqual(plan["hypotheses"][1]["h1"], "uncertainty has a negative association")

    def test_far_pairs_are_exported_without_requesting_review(self):
        vectors = np.array([[1.0, 0.0], [-1.0, 0.0]])
        claims = [hypothesis(owner, [feature(f"f{owner}")]) for owner in range(2)]
        reviewer = Mock(side_effect=AssertionError("Far pairs should not be reviewed"))
        with patch.object(deduplication, "embed", return_value=vectors):
            plan = deduplication.dedup(claims, ENV, CFG, reviewer)
        reviewer.assert_not_called()
        self.assertEqual(plan["dedup_reviews"], [])
        pair, = plan["feature_pairs"]
        self.assertEqual((pair["left"], pair["right"]), ("h00000:f0", "h00001:f1"))
        self.assertEqual(pair["left_feature"], claims[0]["features"][0])
        self.assertEqual(pair["right_feature"], claims[1]["features"][0])
        self.assertEqual(pair["cosine_distance"], 2.0)
        self.assertTrue(pair["eligible"])
        self.assertFalse(pair["within_threshold"])
        self.assertFalse(pair["same_candidate_cluster"])
        self.assertEqual(pair["review_verdict"], "")
        self.assertEqual(pair["review_reason"], "")
        self.assertFalse(pair["merged"])

    def test_missing_pair_decision_fails_closed_instead_of_merging(self):
        claims = [hypothesis(i, [feature(f"f{i}")]) for i in range(3)]
        incomplete = equivalent_review([None] * 3)[:-1]
        with patch.object(deduplication, "embed", identical_vectors):
            with self.assertRaises(ValueError):
                deduplication.dedup(claims, ENV, CFG, lambda cluster: incomplete)

    def test_hard_gates_do_not_send_incompatible_pairs_for_review(self):
        cases = [
            [hypothesis(0, [feature("a")]), hypothesis(1, [feature("b", ("title",))])],
            [hypothesis(0, [feature("a")]), hypothesis(1, [feature("b", dtype="numeric")])],
            [hypothesis(0, [feature("a"), feature("b")])],
            [hypothesis(0, [feature("a", kind="code")]), hypothesis(1, [feature("b")])],
        ]
        for threshold in (0.25, 2):
            for claims in cases:
                with self.subTest(claims=claims, threshold=threshold):
                    reviewer = Mock(side_effect=AssertionError("Incompatible review request"))
                    cfg = {**CFG, "dedup": {"threshold": threshold}}
                    with patch.object(deduplication, "embed", identical_vectors):
                        plan = deduplication.dedup(claims, ENV, cfg, reviewer)
                    self.assertEqual(len(plan["features"]), 2)
                    self.assertEqual(plan["dedup_reviews"], [])
                    reviewer.assert_not_called()
                    for pair in plan["feature_pairs"]:
                        self.assertFalse(pair["eligible"])
                        self.assertFalse(pair["merged"])
                        self.assertEqual(pair["review_verdict"], "")
                    semantic_count = sum(item["kind"] == "prompt"
                                         for claim in claims for item in claim["features"])
                    self.assertEqual(len(plan["feature_pairs"]),
                                     semantic_count * (semantic_count - 1) // 2)


class FeatureValidationTests(unittest.TestCase):
    def test_valid_row_semantic_and_code_measurements(self):
        items = [
            feature("score", ("OC_comments", "argument"), dtype="numeric",
                    spec="Return the mean uncertainty score across this row's comments, or null if empty."),
            feature("raw_count", ("OC_comments",), kind="code", dtype="numeric",
                    spec="df['raw_count'] = df['OC_comments'].map(len)"),
            feature("thresholded", ("argument", "OC_comments"), kind="code",
                    spec="df['thresholded'] = df['score'] >= 7"),
            feature("normalized", ("OC_comments", "argument"), kind="code", dtype="numeric",
                    spec="df['normalized'] = df['score'] / df['raw_count']"),
        ]
        original = copy.deepcopy(items)
        self.assertIs(validate_features(items, ["argument", "OC_comments"]), items)
        self.assertEqual(items, original)
        self.assertEqual(validate_features([]), [])

    def test_all_feature_fields_are_required_and_unknown_fields_are_rejected(self):
        complete = feature("base")
        for field in complete:
            with self.subTest(missing=field):
                item = {key: value for key, value in complete.items() if key != field}
                with self.assertRaises(ValueError):
                    validate_features([item])
        with self.assertRaises(ValueError):
            validate_features([{**complete, "aggregation": "any"}])

    def test_invalid_feature_contracts_fail_closed(self):
        changes = [
            {"name": "Bad Name"}, {"name": ""}, {"name": "trailing_"},
            {"kind": "llm"},
            {"dtype": "vector"}, {"definition": " \n"}, {"spec": None},
            {"source_columns": []}, {"source_columns": ["argument", "argument"]},
            {"source_columns": [" argument"]}, {"source_columns": ["post_id"]},
            {"source_columns": ["comment_id"]}, {"source_columns": ["missing"]},
        ]
        for changed in changes:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_features([{**feature("base"), **changed}],
                                  ["argument", "OC_comments", "other_comments"])
        for invalid in (None, {}, "features", [None]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_features(invalid)

    def test_duplicate_names_and_generated_source_names_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_features([feature("base"), feature("base")])
        derived = feature("derived", ("base",), kind="code",
                          spec="df['derived'] = df['base'] >= 7")
        with self.assertRaises(ValueError):
            validate_features([feature("base"), derived], ["argument"])

    def test_code_references_do_not_require_extra_metadata_or_declaration_order(self):
        derived = feature("derived", kind="code", spec="df['derived'] = df['base'] >= 7")
        items = [derived, feature("base", dtype="numeric")]
        self.assertIs(validate_features(items, ["argument"]), items)


class SourceContractTests(unittest.TestCase):
    def test_cached_decomposition_rejects_missing_unknown_or_outcome_sources(self):
        for sources in [None, [], ["missing"], ["delta"], ["post_id"], ["argument", "argument"]]:
            with self.subTest(sources=sources), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                _, env, cfg = self.make_run(root)
                with patch.object(decomposition, "agent", side_effect=self.submit_decomposition) as agent:
                    decomposition.decompose_all({0: "Claim 0"}, env, cfg, root)
                    path = root / "decomposed/0.json"
                    cached = json.loads(path.read_text())
                    cached["features"][0]["source_columns"] = sources
                    path.write_text(json.dumps(cached))
                    with self.assertRaises(ValueError):
                        decomposition.decompose_all({0: "Claim 0"}, env, cfg, root)
                    self.assertEqual(agent.call_count, 1)

    def make_run(self, root):
        source = root / "data.csv"
        with source.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["post_id", "comment_id", "delta", "argument", "title"])
            writer.writerow(["p1", "c1", 1, "Example argument", "Example title"])
        env = {**ENV, "csv": str(source), "description": "One row is one discussion."}
        cfg = {"model": "claude-sonnet-5-5", "max_tokens": 100, "compat": {},
               "base_url": "https://unused.invalid", "api": "anthropic-messages",
               "api_key_env": "UNUSED_KEY", "provider": "unused", "workers": 1,
               "thinking": "off", "tools": "read,bash,submit"}
        return source, env, cfg

    @staticmethod
    def submit_decomposition(prompt, system, home, cfg, schema, out):
        out.write_text(json.dumps(hypothesis(int(out.stem), [feature("tone")])))

    def test_cached_run_rejects_a_source_removed_from_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, env, cfg = self.make_run(root)
            with patch.object(decomposition, "agent", side_effect=self.submit_decomposition) as agent:
                original = decomposition.decompose_all({0: "Claim 0"}, env, cfg, root)
                self.assertEqual(agent.call_count, 1)
                self.assertTrue((root / "decomposition_inputs.json").is_file())
                self.assertEqual(decomposition.decompose_all({0: "Claim 0"}, env, cfg, root), original)
                self.assertEqual(agent.call_count, 1)
                source.write_text("post_id,comment_id,delta,title\np1,c1,1,Example title\n")
                with self.assertRaises(ValueError):
                    decomposition.decompose_all({0: "Claim 0"}, env, cfg, root)
                self.assertEqual(agent.call_count, 1)

    def test_legacy_cache_without_input_identity_is_rejected(self):
        for cached in ["decomposed/0.json", "decomposed.json"]:
            with self.subTest(cached=cached), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                _, env, cfg = self.make_run(root)
                path = root / cached
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(hypothesis(0, [feature("tone")])))
                with patch.object(decomposition, "agent", side_effect=AssertionError("Unexpected model request")):
                    with self.assertRaises(ValueError):
                        decomposition.decompose_all({0: "Claim 0"}, env, cfg, root)

    def test_cached_run_rejects_changed_data_claim_environment_and_model(self):
        for changed in ["csv_values", "hypothesis", "environment", "model"]:
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                source, env, cfg = self.make_run(root)
                claims = {0: "Claim 0"}
                with patch.object(decomposition, "agent", side_effect=self.submit_decomposition) as agent:
                    decomposition.decompose_all(claims, env, cfg, root)
                    if changed == "csv_values":
                        source.write_text(source.read_text().replace("Example argument", "Different argument"))
                    elif changed == "hypothesis":
                        claims[0] = "A different exact claim"
                    elif changed == "environment":
                        env["description"] = "Different speaker attribution."
                    else:
                        cfg["model"] = "claude-sonnet-different"
                    with self.assertRaises(ValueError):
                        decomposition.decompose_all(claims, env, cfg, root)
                    self.assertEqual(agent.call_count, 1)

    def test_cached_run_rejects_changed_prompt_and_schema(self):
        for changed in ["prompt", "schema"]:
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                _, env, cfg = self.make_run(root)
                prompt_path = root / "prompts.yaml"
                prompt_path.write_text(prompts.PROMPTS.read_text())
                schema_path = root / "schema.json"
                schema_path.write_text(decomposition.SCHEMA.read_text())
                with patch.object(prompts, "PROMPTS", prompt_path), patch.object(decomposition, "SCHEMA", schema_path):
                    with patch.object(decomposition, "agent", side_effect=self.submit_decomposition) as agent:
                        decomposition.decompose_all({0: "Claim 0"}, env, cfg, root)
                        if changed == "prompt":
                            document = yaml.safe_load(prompt_path.read_text())
                            config = document["agents"]["hypothesis_decomposer"]["prompt_config"]
                            config["system_message"] += "\nAdditional exact-claim constraint."
                            prompt_path.write_text(yaml.safe_dump(document))
                        else:
                            schema = json.loads(schema_path.read_text())
                            schema["description"] = "Changed extraction schema."
                            schema_path.write_text(json.dumps(schema))
                        with self.assertRaises(ValueError):
                            decomposition.decompose_all({0: "Claim 0"}, env, cfg, root)
                        self.assertEqual(agent.call_count, 1)

    def test_failed_decomposition_resumes_only_unfinished_claims_with_matching_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, env, cfg = self.make_run(root)
            claims = {0: "Claim 0", 1: "Claim 1"}

            def interrupted_agent(prompt, system, home, cfg, schema, out):
                if out.stem == "1":
                    raise RuntimeError("Synthetic external agent interruption")
                self.submit_decomposition(prompt, system, home, cfg, schema, out)

            with patch.object(decomposition, "agent", side_effect=interrupted_agent):
                with self.assertRaises(RuntimeError):
                    decomposition.decompose_all(claims, env, cfg, root)
            self.assertTrue((root / "decomposition_inputs.json").is_file())
            self.assertTrue((root / "decomposed/0.json").is_file())
            self.assertFalse((root / "decomposed/1.json").exists())
            with patch.object(decomposition, "agent", side_effect=self.submit_decomposition) as agent:
                result = decomposition.decompose_all(claims, env, cfg, root)
            self.assertEqual([item["id"] for item in result], [0, 1])
            self.assertEqual(agent.call_count, 1)
            self.assertEqual(agent.call_args.args[-1].stem, "1")


if __name__ == "__main__":
    unittest.main()
