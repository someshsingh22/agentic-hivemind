import copy
import unittest

from sonnet_dedup.core import build_plan, prepare_groups, validate_partition


ENV = {"name": "cmv", "target": "delta", "group": "post_id"}


def feature(name, sources=("argument",), dtype="binary", kind="prompt", spec=None):
    return {
        "name": name, "dtype": dtype, "source_columns": list(sources),
        "definition": "The author's explicit uncertainty about their own claim",
        "kind": kind,
        "spec": spec or "Return 1 for explicit self-doubt, 0 for assessable absence, null for missing input.",
    }


def hypothesis(owner, features, rhs=None):
    term = features[0]["name"] if features else None
    return {
        "id": owner, "hypothesis": f"Frozen claim {owner}", "features": features,
        "h0": f"{term} has no association", "h1": f"{term} has a positive association",
        "rhs": rhs or term or "1", "term": term, "sign": "+", "note": "Frozen limitation",
    }


class SonnetDedupTests(unittest.TestCase):
    def test_batches_are_prompt_only_source_order_independent_and_deterministic(self):
        claims = [
            hypothesis(9, [feature("numeric", dtype="numeric"),
                           feature("both", ("title", "argument")),
                           feature("code", kind="code", spec="code = 1")]),
            hypothesis(2, [feature("binary"), feature("reverse", ("argument", "title")),
                           feature("title_only", ("title",))]),
            hypothesis(7, [feature("another")]),
        ]
        before = copy.deepcopy(claims)
        batches = prepare_groups(claims, ENV)
        self.assertEqual([[entry["key"] for entry in batch] for batch in batches], [
            ["h00002:binary", "h00007:another"],
            ["h00009:numeric"],
            ["h00009:both", "h00002:reverse"],
            ["h00002:title_only"],
        ])
        self.assertEqual(batches, prepare_groups(claims, ENV))
        self.assertEqual(claims, before)

    def test_equivalents_merge_across_hypotheses_in_raw_order_without_rewriting(self):
        spec = "Read argument. Return 1 for explicit self-doubt, 0 if absent; null if unavailable."
        claims = [hypothesis(11, [feature("uncertainty", spec=spec)]),
                  hypothesis(3, [feature("self_doubt", spec=spec)]),
                  hypothesis(5, [])]
        before = copy.deepcopy(claims)
        partitions = [{"members": ["h00003:self_doubt", "h00011:uncertainty"],
                       "reason": "Same author, text scope, self-doubt indicator and null rules."}]
        plan = build_plan(claims, ENV, partitions)
        self.assertEqual(len(plan["features"]), 1)
        canonical = plan["features"][0]
        self.assertEqual(canonical["name"], "uncertainty")
        self.assertEqual(canonical["spec"], spec)
        self.assertEqual(canonical["members"], [
            {"hypothesis": 11, **claims[0]["features"][0]},
            {"hypothesis": 3, **claims[1]["features"][0]},
        ])
        for item in plan["hypotheses"][:2]:
            self.assertEqual(item["term"], "uncertainty")
            self.assertEqual(item["formula"], "delta ~ uncertainty")
            self.assertEqual(item["h0"], "uncertainty has no association")
            self.assertEqual(item["h1"], "uncertainty has a positive association")
        self.assertIsNone(plan["hypotheses"][2]["term"])
        self.assertEqual(plan["hypotheses"][2]["formula"], "delta ~ 1")
        for original, frozen in zip(claims, plan["hypotheses"]):
            for field in ("id", "hypothesis", "sign", "note"):
                self.assertEqual(frozen[field], original[field])
        self.assertEqual(claims, before)

    def test_code_stays_unique_and_computes_the_same_values(self):
        spec = "indicator = [value >= 2 for value in comment_count]"
        claims = [hypothesis(owner, [feature("indicator", ("comment_count",),
                                            kind="code", spec=spec)]) for owner in (0, 1)]
        before = copy.deepcopy(claims)
        plan = build_plan(claims, ENV, [])
        self.assertEqual([item["name"] for item in plan["features"]], ["indicator", "indicator_2"])
        self.assertEqual(plan["features"][0]["spec"], spec)
        self.assertEqual(plan["features"][1]["spec"],
                         "indicator_2 = [value >= 2 for value in comment_count]")
        self.assertTrue(all(len(item["members"]) == 1 for item in plan["features"]))
        namespace = {"comment_count": [0, 1, 2, 3]}
        for item in plan["features"]:
            exec(item["spec"], namespace)
            self.assertEqual(namespace[item["name"]], [False, False, True, True])
            self.assertEqual(item["members"][0]["spec"], spec)
        self.assertEqual([item["formula"] for item in plan["hypotheses"]],
                         ["delta ~ indicator", "delta ~ indicator_2"])
        self.assertEqual(claims, before)

    def test_merged_prompt_dependencies_and_formulas_use_canonical_names(self):
        claims = [hypothesis(0, [feature("intensity", dtype="numeric")]),
                  hypothesis(1, [feature("emotion", dtype="numeric"),
                                 feature("derived", kind="code",
                                         spec="derived = [value >= 7 for value in emotion]")],
                             rhs="derived + emotion")]
        before = copy.deepcopy(claims)
        plan = build_plan(claims, ENV, [{"members": ["h00000:intensity", "h00001:emotion"],
                                         "reason": "Identical frozen scalar rubric."}])
        self.assertEqual([item["name"] for item in plan["features"]], ["intensity", "derived"])
        self.assertEqual(plan["features"][1]["spec"],
                         "derived = [value >= 7 for value in intensity]")
        namespace = {"intensity": [0, 6, 7, 10]}
        exec(plan["features"][1]["spec"], namespace)
        self.assertEqual(namespace["derived"], [False, False, True, True])
        self.assertEqual(plan["hypotheses"][1]["formula"], "delta ~ derived + intensity")
        self.assertEqual(plan["hypotheses"][1]["term"], "intensity")
        self.assertEqual(plan["hypotheses"][1]["h0"], "intensity has no association")
        self.assertEqual(claims, before)

    def test_canonical_names_skip_sources_and_remap_code_assignments(self):
        claims = [hypothesis(0, [feature("count", kind="code", spec="count = 1"),
                                  feature("count_2", kind="code", spec="df['count_2'] = count + 1"),
                                  feature("count_3", kind="code", spec="count_3 = count_2 + 1")]),
                  hypothesis(1, [feature("count", ("count_4",), kind="code", spec="df['count'] = 4")])]
        plan = build_plan(claims, ENV, [])
        self.assertEqual([item["name"] for item in plan["features"]],
                         ["count", "count_2", "count_3", "count_5"])
        self.assertEqual(plan["features"][-1]["spec"], "df['count_5'] = 4")
        self.assertEqual(plan["hypotheses"][-1]["term"], "count_5")
        self.assertEqual(plan["features"][-1]["members"][0]["spec"], "df['count'] = 4")

    def test_name_collision_does_not_rewrite_prompt_specifications(self):
        spec = "Read title; return 1 if any comment expresses doubt; null if missing."
        claims = [hypothesis(0, [feature("any", kind="code", spec="any = 1")]),
                  hypothesis(1, [feature("any", ("title",), spec=spec)]),
                  hypothesis(2, [feature("doubt", ("title",), spec=spec)])]
        plan = build_plan(claims, ENV, [{"members": ["h00001:any", "h00002:doubt"],
                                         "reason": "Same any-doubt measurement."}])
        self.assertEqual([item["name"] for item in plan["features"]], ["any", "any_2"])
        self.assertEqual(plan["features"][1]["spec"], spec)
        self.assertEqual(plan["hypotheses"][2]["term"], "any_2")

    def test_partition_requires_exact_coverage_and_valid_shapes(self):
        entries = prepare_groups([hypothesis(0, [feature("a")]),
                                  hypothesis(1, [feature("b")])], ENV)[0]
        a, b = [entry["key"] for entry in entries]
        invalid = [
            None, {}, {"groups": {}}, {"groups": [], "extra": True}, {"groups": []},
            {"groups": [None]}, {"groups": [{"members": [], "reason": "Empty"}]},
            {"groups": [{"members": [a], "reason": "Missing b"}]},
            {"groups": [{"members": [a, a, b], "reason": "Repeated a"}]},
            {"groups": [{"members": [a, b], "reason": "Both"},
                        {"members": [a], "reason": "Repeated group"}]},
            {"groups": [{"members": [a, b, "h99999:unknown"], "reason": "Unknown"}]},
            {"groups": [{"members": [a, b, {}], "reason": "Non-string key"}]},
            {"groups": [{"members": a, "reason": "Not a list"}]},
            {"groups": [{"members": [a, b]}]},
            {"groups": [{"members": [a, b], "reason": "  "}]},
            {"groups": [{"members": [a, b], "reason": None}]},
            {"groups": [{"members": [a, b], "reason": ["Not a string"]}]},
            {"groups": [{"members": [a, b], "reason": "Fine", "name": "Rewrite"}]},
        ]
        for response in invalid:
            with self.subTest(response=response), self.assertRaises(ValueError):
                validate_partition(entries, response)
        with self.assertRaises(ValueError):
            build_plan([hypothesis(0, [feature("a")])], ENV, [])

    def test_partitions_reject_same_owner_and_source_or_output_mismatches(self):
        cases = [
            [hypothesis(0, [feature("a"), feature("b")])],
            [hypothesis(0, [feature("a")]), hypothesis(1, [feature("b", ("title",))])],
            [hypothesis(0, [feature("a")]), hypothesis(1, [feature("b", dtype="numeric")])],
        ]
        for claims in cases:
            entries = [entry for batch in prepare_groups(claims, ENV) for entry in batch]
            partitions = [{"members": [entry["key"] for entry in entries], "reason": "Invalid merge"}]
            with self.subTest(claims=claims):
                with self.assertRaises(ValueError):
                    validate_partition(entries, {"groups": partitions})
                with self.assertRaises(ValueError):
                    build_plan(claims, ENV, partitions)

    def test_source_sets_can_match_despite_original_column_order(self):
        claims = [hypothesis(0, [feature("a", ("argument", "title"))]),
                  hypothesis(1, [feature("b", ("title", "argument"))])]
        plan = build_plan(claims, ENV, [{"members": ["h00000:a", "h00001:b"],
                                         "reason": "Same joint-input measurement."}])
        self.assertEqual(len(plan["features"]), 1)
        self.assertEqual(plan["features"][0]["members"][1]["source_columns"], ["title", "argument"])

    def test_all_features_are_validated_before_prompt_selection(self):
        for source in ("delta", "post_id", "comment_id", "thread_id"):
            for kind in ("prompt", "code"):
                claims = [hypothesis(0, [feature("base", (source,), kind=kind)])]
                with self.subTest(source=source, kind=kind), self.assertRaises(ValueError):
                    prepare_groups(claims, {**ENV, "group": "thread_id"})
        invalid_code = feature("base", kind="code")
        del invalid_code["spec"]
        with self.assertRaises(ValueError):
            prepare_groups([hypothesis(0, [invalid_code])], ENV)
        for name in ("delta", "post_id", "comment_id", "thread_id", "title"):
            claims = [hypothesis(0, [feature(name, kind="code")]),
                      hypothesis(1, [feature("other", ("title",))])]
            with self.subTest(name=name), self.assertRaises(ValueError):
                prepare_groups(claims, {**ENV, "group": "thread_id"})
        with self.assertRaises(ValueError):
            prepare_groups([hypothesis(0, [feature("title")]),
                            hypothesis(1, [feature("code", ("title",), kind="code")])], ENV)
        for owners in ((0, 0), ("0", 1), (True, 2)):
            with self.subTest(owners=owners), self.assertRaises(ValueError):
                prepare_groups([hypothesis(owner, [feature("a")]) for owner in owners], ENV)
        with self.assertRaises(ValueError):
            prepare_groups([hypothesis(0, [feature("a"), feature("a")])], ENV)

    def test_empty_inputs_and_singletons_need_only_their_complete_partition(self):
        self.assertEqual(prepare_groups([], ENV), [])
        self.assertEqual(build_plan([], ENV, []), {
            "env": ENV, "features": [], "hypotheses": [], "partitions": [],
        })
        claims = [hypothesis(7, [feature("only")])]
        partitions = [{"members": ["h00007:only"], "reason": "Only measurement in batch."}]
        self.assertEqual(len(build_plan(claims, ENV, partitions)["features"]), 1)


if __name__ == "__main__":
    unittest.main()
