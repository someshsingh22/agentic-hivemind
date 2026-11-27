import copy
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ssh import decompose as decomposition
from ssh import dedup as deduplication
from ssh.features import validate_features


ENV = {"name": "cmv", "target": "delta", "group": "post_id"}
CFG = {"embedding": {}, "dedup": {"threshold": 0.25}}


def feature(name, sources=("argument",), kind="prompt", dtype="binary", spec=None):
    return {"name": name, "dtype": dtype, "source_columns": list(sources),
            "definition": "Explicit uncertainty about the OP's claim", "kind": kind,
            "spec": spec or "Return 1 for uncertainty, 0 if absent, null if unavailable."}


def hypothesis(index, features, rhs=None):
    term = features[0]["name"] if features else None
    return {"id": index, "hypothesis": f"Claim {index}", "features": features,
            "h0": f"{term} has no association", "h1": f"{term} has a positive association",
            "rhs": rhs or term or "1", "term": term, "sign": "+", "note": ""}


def identical_vectors(texts, cfg):
    return np.tile([1.0, 0.0], (len(texts), 1))


class DeduplicationTests(unittest.TestCase):
    def dedup(self, claims, cfg=CFG):
        with patch.object(deduplication, "embed", side_effect=identical_vectors):
            return deduplication.dedup(claims, ENV, cfg)

    def test_all_kinds_merge_and_embed_name_definition_in_raw_order(self):
        claims = [hypothesis(i, [feature(name, kind=kind)]) for i, (name, kind) in enumerate(
            [("uncertainty", "code"), ("self_doubt", "prompt"), ("hesitation", "code")])]
        original = copy.deepcopy(claims)
        with patch.object(deduplication, "embed", side_effect=identical_vectors) as embed:
            plan = deduplication.dedup(claims, ENV, CFG)
        embed.assert_called_once_with([f"{f['name']}: {f['definition']}" for d in claims
                                       for f in d["features"]], CFG["embedding"])
        self.assertEqual(len(plan["features"]), 1)
        self.assertEqual([h["term"] for h in plan["hypotheses"]], ["uncertainty"] * 3)
        self.assertEqual(plan["hypotheses"][1]["formula"], "delta ~ uncertainty")
        self.assertEqual(plan["hypotheses"][1]["h0"], "uncertainty has no association")
        self.assertEqual(claims, original)
        self.assertEqual(deduplication.summary(plan), {
            "hypotheses": 3, "features_raw": 3, "features_unique": 1,
            "prompt_features_raw": 1, "prompt_features_unique": 0})
        self.assertNotIn("dedup_reviews", plan)

    def test_matching_sources_are_order_independent(self):
        plan = self.dedup([hypothesis(0, [feature("a", ("argument", "title"))]),
                           hypothesis(1, [feature("b", ("title", "argument"))])])
        self.assertEqual(len(plan["features"]), 1)

    def test_owner_dtype_source_gates_hold_at_threshold_two(self):
        claims = [hypothesis(0, [feature("a"), feature("b")]),
                  hypothesis(1, [feature("c", dtype="numeric")]),
                  hypothesis(2, [feature("d", ("title",))])]
        for threshold in (0.25, 2):
            with self.subTest(threshold=threshold):
                plan = self.dedup(claims, {**CFG, "dedup": {"threshold": threshold}})
                self.assertEqual(len(plan["features"]), 4)
                self.assertTrue(all(not p["eligible"] and not p["merged"] for p in plan["feature_pairs"]))

    def test_complete_linkage_does_not_chain_distant_endpoints(self):
        vectors = np.array([[math.cos(a), math.sin(a)] for a in (0, 0.5, 1.0)])
        claims = [hypothesis(i, [feature(f"f{i}")]) for i in range(3)]
        with patch.object(deduplication, "embed", side_effect=AssertionError("Unexpected embedding")):
            plan = deduplication.dedup(claims, ENV, CFG, vectors=vectors)
        self.assertEqual(sorted(len(f["members"]) for f in plan["features"]), [1, 2])
        self.assertFalse(plan["feature_pairs"][1]["merged"])
        self.assertFalse(plan["feature_pairs"][1]["within_threshold"])
        self.assertEqual(len(plan["feature_pairs"]), 3)
        for pair in plan["feature_pairs"]:
            i, j = pair["left_hypothesis"], pair["right_hypothesis"]
            self.assertEqual(set(pair), {"left", "right", "left_hypothesis", "right_hypothesis",
                                        "left_feature", "right_feature", "cosine_distance",
                                        "eligible", "within_threshold", "merged"})
            self.assertEqual(pair["left"], f"h{i:05d}:f{i}")
            self.assertAlmostEqual(pair["cosine_distance"], 1 - vectors[i] @ vectors[j])

    def test_empty_and_single_need_no_embeddings(self):
        with patch.object(deduplication, "embed", side_effect=AssertionError("Unexpected embedding")):
            for claims in ([], [hypothesis(0, [])], [hypothesis(0, [feature("tone")])]):
                plan = deduplication.dedup(claims, ENV, CFG)
                self.assertEqual(len(plan["features"]), sum(len(d["features"]) for d in claims))
                self.assertEqual(plan["feature_pairs"], [])

    def test_collision_names_and_code_assignment_are_canonical(self):
        claims = [hypothesis(0, [feature("count", kind="code", spec="count = 1"),
                                  feature("count_2", kind="code", spec="df['count_2'] = count + 1"),
                                  feature("count_3", kind="code", spec="count_3 = count_2 + 1")]),
                  hypothesis(1, [feature("count", ("count_4",), kind="code", spec="df['count'] = 4")])]
        plan = self.dedup(claims)
        self.assertEqual([f["name"] for f in plan["features"]], ["count", "count_2", "count_3", "count_5"])
        self.assertEqual(plan["features"][-1]["spec"], "df['count_5'] = 4")
        self.assertEqual(plan["hypotheses"][-1]["term"], "count_5")

    def test_code_dependency_executes_against_canonical_measurement(self):
        claims = [hypothesis(0, [feature("intensity", dtype="numeric")]),
                  hypothesis(1, [feature("emotion", dtype="numeric"), feature("derived", kind="code",
                      spec="derived = emotion >= 7")], rhs="derived + emotion")]
        plan = self.dedup(claims)
        namespace = {"intensity": np.array([0, 6, 7, 10])}
        exec(plan["features"][1]["spec"], namespace)
        np.testing.assert_array_equal(namespace["derived"], [False, False, True, True])
        self.assertEqual(plan["hypotheses"][1]["formula"], "delta ~ derived + intensity")

    def test_prompt_rubrics_remain_byte_unchanged(self):
        spec = "Read argument. Return 1 if any comment expresses doubt; null if unavailable."
        claims = [hypothesis(0, [feature("any", kind="code")]),
                  hypothesis(1, [feature("any", ("title",), spec=spec)]),
                  hypothesis(2, [feature("doubt", ("title",), spec=spec)])]
        plan = self.dedup(claims)
        self.assertEqual(plan["features"][1]["name"], "any_2")
        self.assertEqual(plan["features"][1]["spec"], spec)
        self.assertEqual(plan["hypotheses"][2]["term"], "any_2")

    def test_invalid_sources_names_and_thresholds_fail(self):
        for name in ("title", "delta", "post_id", "comment_id", "thread_id"):
            claims = [hypothesis(0, [feature(name)]), hypothesis(1, [feature("other", ("title",))])]
            with self.subTest(name=name), self.assertRaises(ValueError):
                deduplication.dedup(claims, {**ENV, "group": "thread_id"}, CFG)
        for source in ("delta", "post_id", "comment_id"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                self.dedup([hypothesis(0, [feature("base", (source,))])])
        for threshold in (-1, 3, float("nan")):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                self.dedup([], {**CFG, "dedup": {"threshold": threshold}})


class FeatureValidationTests(unittest.TestCase):
    def test_contract_is_structural_and_allows_bare_dependencies(self):
        items = [feature("derived", kind="code", spec="derived = base >= 3"), feature("base")]
        original = copy.deepcopy(items)
        self.assertIs(validate_features(items, ["argument"]), items)
        self.assertEqual(items, original)
        self.assertEqual(validate_features([]), [])

    def test_exact_six_fields_are_required(self):
        complete = feature("base")
        for field in complete:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_features([{k: v for k, v in complete.items() if k != field}])
        with self.assertRaises(ValueError):
            validate_features([{**complete, "aggregation": "any"}])

    def test_invalid_contracts_sources_and_collisions_fail(self):
        for changed in ({"name": "Bad Name"}, {"name": "trailing_"}, {"kind": "llm"},
                        {"dtype": "vector"}, {"definition": " "}, {"spec": None},
                        {"source_columns": []}, {"source_columns": ["argument", "argument"]},
                        {"source_columns": [" argument"]}, {"source_columns": ["post_id"]},
                        {"source_columns": ["missing"]}, {"name": "argument"}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_features([{**feature("base"), **changed}], ["argument"])
        for invalid in (None, {}, "features", [None], [feature("base"), feature("base")],
                        [feature("base"), feature("derived", ("base",))]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_features(invalid, ["argument"])
        with self.assertRaises(ValueError):
            validate_features([feature("title"), feature("other", ("title",))])


class SourceContractTests(unittest.TestCase):
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

    def test_existing_file_reused_without_manifest_or_model_request(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, env, cfg = self.make_run(root)
            path = root / "decomposed/0.json"
            path.parent.mkdir()
            cached = hypothesis(0, [feature("tone")])
            path.write_text(json.dumps(cached))
            with patch.object(decomposition, "agent", side_effect=AssertionError("Unexpected model request")):
                self.assertEqual(decomposition.decompose_all({0: "Claim 0"}, env, cfg, root), [cached])
            self.assertFalse((root / "decomposition_inputs.json").exists())

    def test_changed_inputs_reuse_existing_file_like_main(self):
        for changed in ("csv_values", "hypothesis", "environment", "model"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                source, env, cfg = self.make_run(root)
                claims = {0: "Claim 0"}
                with patch.object(decomposition, "agent", side_effect=self.submit_decomposition) as agent:
                    original = decomposition.decompose_all(claims, env, cfg, root)
                    if changed == "csv_values":
                        source.write_text(source.read_text().replace("Example argument", "Different argument"))
                    elif changed == "hypothesis":
                        claims[0] = "Changed claim"
                    elif changed == "environment":
                        env["description"] = "Changed environment"
                    else:
                        cfg["model"] = "different-model"
                    self.assertEqual(decomposition.decompose_all(claims, env, cfg, root), original)
                    self.assertEqual(agent.call_count, 1)

    def test_cached_sources_still_require_structural_and_domain_validation(self):
        for sources in (None, [], ["missing"], ["delta"], ["post_id"], ["argument", "argument"]):
            with self.subTest(sources=sources), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                _, env, cfg = self.make_run(root)
                path = root / "decomposed/0.json"
                path.parent.mkdir()
                cached = hypothesis(0, [feature("tone")])
                cached["features"][0]["source_columns"] = sources
                path.write_text(json.dumps(cached))
                with patch.object(decomposition, "agent", side_effect=AssertionError("Unexpected model request")):
                    with self.assertRaises(ValueError):
                        decomposition.decompose_all({0: "Claim 0"}, env, cfg, root)

    def test_removed_csv_source_rejected_even_with_cached_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, env, cfg = self.make_run(root)
            with patch.object(decomposition, "agent", side_effect=self.submit_decomposition) as agent:
                decomposition.decompose_all({0: "Claim 0"}, env, cfg, root)
                source.write_text("post_id,comment_id,delta,title\np1,c1,1,Example title\n")
                with self.assertRaises(ValueError):
                    decomposition.decompose_all({0: "Claim 0"}, env, cfg, root)
                self.assertEqual(agent.call_count, 1)

    def test_interrupted_run_resumes_only_unfinished_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, env, cfg = self.make_run(root)
            claims = {0: "Claim 0", 1: "Claim 1"}

            def interrupted(prompt, system, home, cfg, schema, out):
                if out.stem == "1":
                    raise RuntimeError("Synthetic interruption")
                self.submit_decomposition(prompt, system, home, cfg, schema, out)

            with patch.object(decomposition, "agent", side_effect=interrupted):
                with self.assertRaises(RuntimeError):
                    decomposition.decompose_all(claims, env, cfg, root)
            self.assertTrue((root / "decomposed/0.json").is_file())
            self.assertFalse((root / "decomposed/1.json").exists())
            with patch.object(decomposition, "agent", side_effect=self.submit_decomposition) as agent:
                result = decomposition.decompose_all(claims, env, cfg, root)
            self.assertEqual([d["id"] for d in result], [0, 1])
            self.assertEqual(agent.call_count, 1)
            self.assertEqual(agent.call_args.args[-1].stem, "1")


if __name__ == "__main__":
    unittest.main()
