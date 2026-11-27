import unittest

from sonnet_dedup.hybrid import decisions_for, partitions_for


class HybridTests(unittest.TestCase):
    def test_approved_chain_does_not_merge_unapproved_endpoints(self):
        entries = [{"key": key} for key in ("a", "b", "c")]
        groups = partitions_for(entries, {("a", "b"), ("b", "c")})
        self.assertEqual([group["members"] for group in groups], [["a", "b"], ["c"]])
        complete = partitions_for(entries, {("a", "b"), ("a", "c"), ("b", "c")})
        self.assertEqual([group["members"] for group in complete], [["a", "b", "c"]])

    def test_decisions_must_cover_pairs_once_with_real_booleans(self):
        batch = [{"id": 0}, {"id": 1}]
        a = {"id": 0, "equivalent": False, "reason": "A quote need not refute."}
        b = {"id": 1, "equivalent": True, "reason": "Same measurement and boundaries."}
        self.assertFalse(decisions_for(batch, {"decisions": [b, a]})[0]["equivalent"])
        for decisions in ([a], [a, a], [a, {**b, "id": 2}],
                          [a, {**b, "equivalent": "false"}], [a, {**b, "reason": " "}]):
            with self.subTest(decisions=decisions), self.assertRaises(ValueError):
                decisions_for(batch, {"decisions": decisions})
