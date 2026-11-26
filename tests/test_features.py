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


ENV = {"name": "cmv", "target": "delta", "group": "post_id"}
CFG = {"embedding": {}, "dedup": {"threshold": 0.25}}


def feature(name, sources=("argument",), kind="prompt", dtype="binary", spec=None):
    return {
        "name": name,
        "dtype": dtype,
        "source_columns": list(sources),
        "definition": "OP argument: explicit uncertainty about the OP's own claim",
        "kind": kind,
        "spec": spec or "Read argument for this row. Output 1 for explicit uncertainty, 0 if absent, null if unassessable.",
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


class DeduplicationTests(unittest.TestCase):
    def test_matching_source_sets_merge_and_rewrite_claims(self):
        claims = [
            hypothesis(0, [feature("uncertainty", ("argument", "title"))]),
            hypothesis(1, [feature("self_doubt", ("title", "argument"))]),
        ]
        claims[1]["sign"] = "-"
        with patch.object(deduplication, "embed", identical_vectors):
            plan = deduplication.dedup(claims, ENV, CFG)
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
            plan = deduplication.dedup(claims, ENV, CFG)
        self.assertEqual([f["source_columns"] for f in plan["features"]], [
            ["argument"], ["title"], ["argument", "title"],
        ])
        self.assertEqual([len(f["members"]) for f in plan["features"]], [1, 1, 1])

    def test_code_features_stay_distinct_and_use_no_embedding_service(self):
        claims = [hypothesis(i, [feature(name, kind="code", spec=f"df['{name}'] = 1")])
                  for i, name in enumerate(["count", "count", "count_2"])]
        with patch.object(deduplication, "embed", side_effect=AssertionError("Unexpected embedding request")):
            plan = deduplication.dedup(claims, ENV, CFG)
        names = [f["name"] for f in plan["features"]]
        self.assertEqual(names, ["count", "count_2", "count_2_2"])
        self.assertEqual([h["term"] for h in plan["hypotheses"]], names)
        self.assertEqual([f["spec"] for f in plan["features"]], [f"df['{name}'] = 1" for name in names])

    def test_code_never_merges_with_semantic_feature(self):
        claims = [hypothesis(0, [feature("uncertainty")]),
                  hypothesis(1, [feature("uncertainty", kind="code")]),
                  hypothesis(2, [feature("self_doubt")])]
        with patch.object(deduplication, "embed", identical_vectors):
            plan = deduplication.dedup(claims, ENV, CFG)
        self.assertEqual([(f["kind"], len(f["members"])) for f in plan["features"]],
                         [("prompt", 2), ("code", 1)])
        self.assertEqual([h["term"] for h in plan["hypotheses"]],
                         ["uncertainty", "uncertainty_2", "uncertainty"])

    def test_derived_code_uses_canonical_semantic_measurement(self):
        claims = [hypothesis(0, [feature("intensity", dtype="numeric")]),
                  hypothesis(1, [feature("emotion", dtype="numeric"),
                                 feature("high_emotion", kind="code", spec="df['high_emotion'] = df['emotion'] >= 7")],
                             rhs="high_emotion")]
        with patch.object(deduplication, "embed", identical_vectors):
            plan = deduplication.dedup(claims, ENV, CFG)
        derived = next(f for f in plan["features"] if f["name"] == "high_emotion")
        self.assertEqual(derived["spec"], "df['high_emotion'] = df['intensity'] >= 7")
        self.assertEqual(plan["hypotheses"][1]["formula"], "delta ~ high_emotion")

    def test_same_hypothesis_and_different_dtypes_remain_distinct(self):
        claims = [hypothesis(0, [feature("uncertain"), feature("certain")]),
                  hypothesis(1, [feature("intensity", dtype="numeric")])]
        with patch.object(deduplication, "embed", identical_vectors):
            plan = deduplication.dedup(claims, ENV, CFG)
        self.assertEqual([f["name"] for f in plan["features"]], ["uncertain", "certain", "intensity"])

    def test_complete_linkage_does_not_chain_distant_endpoints(self):
        vectors = np.array([[math.cos(a), math.sin(a)] for a in [0, 0.5, 1.0]])
        claims = [hypothesis(i, [feature(f"f{i}")]) for i in range(3)]
        with patch.object(deduplication, "embed", return_value=vectors):
            plan = deduplication.dedup(claims, ENV, CFG)
        groups = [{m["hypothesis"] for m in f["members"]} for f in plan["features"]]
        self.assertEqual(sorted(map(len, groups)), [1, 2])
        self.assertFalse(any({0, 2} <= group for group in groups))

    def test_empty_features_and_one_semantic_feature_need_no_embeddings(self):
        with patch.object(deduplication, "embed", side_effect=AssertionError("Unexpected embedding request")):
            empty = deduplication.dedup([hypothesis(0, [])], ENV, CFG)
            single = deduplication.dedup([hypothesis(0, [feature("tone")])], ENV, CFG)
        self.assertEqual(empty["features"], [])
        self.assertIsNone(empty["hypotheses"][0]["term"])
        self.assertEqual(single["features"][0]["name"], "tone")

    def test_missing_source_metadata_is_not_silently_merged(self):
        item = feature("tone")
        del item["source_columns"]
        with self.assertRaises(ValueError):
            deduplication.dedup([hypothesis(0, [item])], ENV, CFG)

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
            plan = deduplication.dedup(claims, ENV, CFG)
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


class SourceContractTests(unittest.TestCase):
    def test_cached_decomposition_rejects_missing_unknown_or_outcome_sources(self):
        for sources in [None, [], ["missing"], ["delta"], ["post_id"], ["argument", "argument"]]:
            with self.subTest(sources=sources), tempfile.TemporaryDirectory() as root:
                folder = Path(root) / "decomposed"
                folder.mkdir()
                item = feature("tone")
                item["source_columns"] = sources
                (folder / "0.json").write_text(json.dumps(hypothesis(0, [item])))
                with self.assertRaises(ValueError):
                    decomposition.decompose(0, "Claim 0", ENV, {}, root, Path(root), ["argument", "title"])

    def test_cached_run_rejects_a_source_removed_from_csv(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root / "data.csv"
            with source.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["post_id", "comment_id", "delta", "argument", "title"])
                writer.writerow(["p1", "c1", 1, "Example argument", "Example title"])
            (root / "decomposed").mkdir()
            (root / "decomposed/0.json").write_text(json.dumps(hypothesis(0, [feature("tone")])))
            cfg = {"model": "unused", "max_tokens": 100, "compat": {}, "base_url": "https://unused.invalid",
                   "api": "anthropic-messages", "api_key_env": "UNUSED_KEY", "provider": "unused", "workers": 1}
            with patch.object(decomposition, "agent", side_effect=AssertionError("Unexpected model request")):
                decomposition.decompose_all({0: "Claim 0"}, {**ENV, "csv": str(source)}, cfg, root)
                source.write_text("post_id,comment_id,delta,title\np1,c1,1,Example title\n")
                with self.assertRaises(ValueError):
                    decomposition.decompose_all({0: "Claim 0"}, {**ENV, "csv": str(source)}, cfg, root)


if __name__ == "__main__":
    unittest.main()
