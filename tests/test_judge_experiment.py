import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import judge_experiment as je
from find_categories import SetupError
from tests.fakes import FAKE_KEY, FakeJudgeOpenAI

BRIEF = "I study citation checking in RAG. I am not interested in chatbot benchmarks."


def paper(i, title):
    return {"arxiv_id": i, "title": title,
            "abstract": f"This abstract about {title.lower()} describes the work in enough words to quote."}


class PoolTest(unittest.TestCase):
    def test_pool_is_matches_plus_each_top_n_deduplicated(self):
        orders = {"embedding": ["a", "b", "c"], "bm25": ["b", "d", "a"], "hybrid": ["b", "a", "d"]}
        pool = je.build_pool({"a", "z"}, orders, top_n=2)
        self.assertEqual(list(pool), ["a", "z", "b", "d"])
        self.assertEqual(pool["a"], ["llm_match", "embedding_top2", "hybrid_top2"])
        self.assertEqual(pool["b"], ["embedding_top2", "bm25_top2", "hybrid_top2"])
        self.assertEqual(pool["z"], ["llm_match"])

    def test_orders_are_read_from_saved_rankings(self):
        with tempfile.TemporaryDirectory() as d:
            emb, hyb = Path(d) / "e.csv", Path(d) / "h.csv"
            emb.write_text("rank,arxiv_id\n2,x\n1,y\n", encoding="utf-8")
            hyb.write_text("rank,arxiv_id,bm25_rank\n1,x,2\n2,y,1\n", encoding="utf-8")
            self.assertEqual(je.load_orders(emb, hyb), {"embedding": ["y", "x"], "bm25": ["y", "x"],
                                                        "hybrid": ["x", "y"]})
            hyb.write_text("rank,arxiv_id,bm25_rank\n1,x,1\n2,q,2\n", encoding="utf-8")
            with self.assertRaises(SetupError):
                je.load_orders(emb, hyb)

    def test_search_run_must_match_brief_and_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            inputs, info = Path(d) / "inputs", Path(d) / "run_info.json"
            inputs.mkdir()
            (inputs / "p.csv").write_text("x", encoding="utf-8")
            info.write_text(json.dumps({"query": BRIEF, "inputs_sha256": {"p.csv": je.sha256(inputs / "p.csv")}}),
                            encoding="utf-8")
            je.check_search_run(info, inputs, BRIEF)
            with self.assertRaises(SetupError):
                je.check_search_run(info, inputs, "another brief")
            (inputs / "p.csv").write_text("changed", encoding="utf-8")
            with self.assertRaises(SetupError):
                je.check_search_run(info, inputs, BRIEF)


class QuoteTest(unittest.TestCase):
    ABSTRACT = "We propose a method.  It checks\ncitations against retrieved sources carefully."

    def test_exact_quote_with_whitespace_differences_is_ok(self):
        self.assertEqual(je.check_quote("It checks citations against retrieved sources", self.ABSTRACT), "ok")
        self.assertEqual(je.check_quote('"It checks citations against retrieved"', self.ABSTRACT), "ok")

    def test_paraphrase_ellipsis_or_case_change_is_not_found(self):
        for q in ("It verifies citations against retrieved sources", "It checks ... retrieved sources",
                  "it checks citations against retrieved sources"):
            self.assertEqual(je.check_quote(q, self.ABSTRACT), "not_found", q)

    def test_trivial_quote_is_too_short(self):
        self.assertEqual(je.check_quote("a method", self.ABSTRACT), "too_short")
        self.assertEqual(je.check_quote("", self.ABSTRACT), "too_short")


@mock.patch("openai.OpenAI", FakeJudgeOpenAI)
class JudgePoolTest(unittest.TestCase):
    def setUp(self):
        FakeJudgeOpenAI.reset()
        self.papers = {"2609.00001": paper("2609.00001", "Citation checking for RAG"),
                       "2609.00002": paper("2609.00002", "A RAG cache"),
                       "2609.00003": paper("2609.00003", "Image models")}
        self.tmp = tempfile.TemporaryDirectory()
        self.raw = Path(self.tmp.name) / "judgments.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def run_pool(self, ids):
        with contextlib.redirect_stdout(io.StringIO()):
            return je.judge_pool(ids, self.papers, BRIEF, FAKE_KEY, raw=self.raw)

    def test_judge_sees_only_brief_title_and_abstract(self):
        self.run_pool(["2609.00001"])
        sent = json.loads(FakeJudgeOpenAI.inputs[0])
        self.assertEqual(sent, {"research_brief": BRIEF, "paper": {
            "title": "Citation checking for RAG", "abstract": self.papers["2609.00001"]["abstract"]}})
        self.assertNotIn("2609.00001", FakeJudgeOpenAI.inputs[0])

    def test_one_request_per_paper_and_results_map_to_their_paper(self):
        records, error = self.run_pool(list(self.papers))
        self.assertIsNone(error)
        self.assertEqual(len(FakeJudgeOpenAI.inputs), 3)
        self.assertEqual({i: r["verdict"] for i, r in records.items()},
                         {"2609.00001": "relevant", "2609.00002": "partly_relevant", "2609.00003": "irrelevant"})
        self.assertTrue(all(r["quote_check"] == "ok" for r in records.values()))

    def test_saved_judgments_are_reused_only_with_the_same_settings(self):
        self.run_pool(list(self.papers))
        self.assertEqual(set(je.load_raw(self.raw, je.fingerprint(BRIEF))), set(self.papers))
        self.assertEqual(je.load_raw(self.raw, je.fingerprint("a different brief")), {})

    def test_failure_keeps_finished_judgments_on_disk(self):
        FakeJudgeOpenAI.reset(fail_on_call=2)
        records, error = self.run_pool(list(self.papers))
        self.assertIn("Could not reach the OpenAI API", str(error))
        self.assertEqual(len(records), len(FakeJudgeOpenAI.inputs) - 1)
        self.assertEqual(set(je.load_raw(self.raw, je.fingerprint(BRIEF))), set(records))


class CompareTest(unittest.TestCase):
    def setUp(self):
        v = lambda verdict: {"verdict": verdict, "reason": "r", "quote": "q", "quote_check": "ok"}
        self.orders = {"embedding": ["a", "b", "c", "d"], "bm25": ["c", "a", "d", "b"], "hybrid": ["a", "c", "b", "d"]}
        self.judged = {"a": v("relevant"), "b": v("irrelevant"), "c": v("partly_relevant"), "d": v("relevant")}

    def test_breakdown_counts_unjudged_separately(self):
        self.assertEqual(je.breakdown(["a", "c", "x"], self.judged),
                         {"relevant": 1, "partly_relevant": 1, "irrelevant": 0, "unjudged": 1})

    def test_compare_counts_methods_and_finds_misses(self):
        r = je.compare({"b", "d"}, self.orders, self.judged, top_n=2, cutoffs=(1, 2))
        self.assertEqual(r["methods"]["bm25"][2], {"relevant": 1, "partly_relevant": 1, "irrelevant": 0, "unjudged": 0})
        self.assertEqual(r["methods"]["embedding"][1]["relevant"], 1)
        self.assertEqual(r["llm_matches"], {"relevant": 1, "partly_relevant": 0, "irrelevant": 1, "unjudged": 0})
        self.assertEqual(r["found_by_search_missed_by_llm"], ["a", "c"])  # relevant first, then partly
        self.assertEqual(r["llm_matches_below_search_top"], ["d"])  # b is in embedding's top 2

    def test_disagreements(self):
        dropped, added = je.disagreements({"b", "d"}, self.judged)
        self.assertEqual((dropped, added), (["b"], ["a"]))


if __name__ == "__main__":
    unittest.main()
