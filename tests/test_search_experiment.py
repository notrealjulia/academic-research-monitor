import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import search_experiment as se
from find_categories import SetupError
from tests.fakes import FAKE_KEY, FakeEmbeddingsOpenAI


def paper(arxiv_id, title, abstract="An abstract."):
    return {"arxiv_id": arxiv_id, "title": title, "abstract": abstract}


class RankingTest(unittest.TestCase):
    def test_rank_orders_by_score_then_id(self):
        self.assertEqual(se.rank({"b": 1.0, "a": 1.0, "c": 2.0}), ["c", "a", "b"])
        self.assertEqual(se.positions(["c", "a", "b"]), {"c": 1, "a": 2, "b": 3})

    def test_cosine(self):
        self.assertAlmostEqual(se.cosine([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(se.cosine([1, 0], [0, 2]), 0.0)
        self.assertAlmostEqual(se.cosine([1, 1], [2, 2]), 1.0)
        self.assertEqual(se.cosine([0, 0], [1, 1]), 0.0)

    def test_tokenize_folds_plurals_the_same_way_on_both_sides(self):
        self.assertEqual(se.tokenize("Hallucinations, queries & Retrieval-Augmented QA"),
                         ["hallucination", "query", "retrieval", "augmented", "qa"])
        self.assertEqual(se.tokenize("class analysis focus"), ["class", "analysis", "focus"])

    def test_bm25_prefers_rarer_and_repeated_terms(self):
        docs = {"a": "citation checking for rag", "b": "rag rag benchmarks", "c": "image models",
                "d": "citation citation citation checking"}
        s = se.bm25_scores("citation checking", docs)
        self.assertEqual(s["c"], 0.0)
        self.assertEqual(s["b"], 0.0)
        self.assertGreater(s["d"], s["a"])
        self.assertGreater(s["a"], 0.0)

    def test_bm25_common_words_never_score_negative(self):
        docs = {str(i): "the model" for i in range(5)} | {"x": "the citation"}
        s = se.bm25_scores("the citation", docs)
        self.assertTrue(all(v >= 0 for v in s.values()))
        self.assertEqual(se.rank(s)[0], "x")

    def test_bm25_ignores_query_terms_absent_from_all_papers(self):
        self.assertEqual(se.bm25_scores("zebra", {"a": "rag"}), {"a": 0.0})

    def test_rrf_combines_ranks(self):
        s = se.rrf(["a", "b", "c"], ["c", "a", "b"], k=60)
        self.assertAlmostEqual(s["a"], 1 / 61 + 1 / 62)
        self.assertAlmostEqual(s["c"], 1 / 63 + 1 / 61)
        self.assertEqual(se.rank(s), ["a", "c", "b"])

    def test_rows_map_ids_to_titles_ranks_and_previous_membership(self):
        titles = {"x": "X paper", "y": "Y paper"}
        emb = se.embedding_rows(["y", "x"], {"x": 0.1, "y": 0.9}, titles, {"x"})
        self.assertEqual(emb[0], {"rank": 1, "arxiv_id": "y", "title": "Y paper", "similarity": "0.900000",
                                  "previous_match": "no"})
        self.assertEqual(emb[1]["previous_match"], "yes")
        hyb = se.hybrid_rows(["x", "y"], {"x": 0.03, "y": 0.02}, {"x": 2, "y": 1}, {"x": 1, "y": 2},
                             {"x": 3.0, "y": 0.0}, titles, {"x"})
        self.assertEqual((hyb[0]["arxiv_id"], hyb[0]["embedding_rank"], hyb[0]["bm25_rank"], hyb[0]["title"]),
                         ("x", 2, 1, "X paper"))

    def test_overlap_counts_previous_matches_at_cutoffs(self):
        order = [str(i) for i in range(100)]
        self.assertEqual(se.overlap(order, {"0", "15", "40", "99"}), {10: 1, 20: 2, 50: 3})


class EmbeddingCacheTest(unittest.TestCase):
    def setUp(self):
        FakeEmbeddingsOpenAI.inputs = []
        self.papers = [paper("2609.00001", "Citation checking"), paper("2609.00002", "Image models")]

    def fill(self, cache, query):
        todo = se.missing_texts(cache, self.papers, query)
        with mock.patch("openai.OpenAI", FakeEmbeddingsOpenAI), contextlib.redirect_stdout(io.StringIO()):
            se.fill_cache(cache, todo, FAKE_KEY, save=lambda c: None)
        return todo

    def test_vectors_follow_input_order_even_if_returned_shuffled(self):
        cache = {"model": se.EMBED_MODEL, "papers": {}, "queries": {}}
        self.fill(cache, "citation")
        got = se.decode(cache["papers"]["2609.00002"]["embedding_b64"])
        self.assertEqual(got, FakeEmbeddingsOpenAI.vector(se.paper_text(self.papers[1])))

    def test_only_missing_or_changed_texts_are_embedded_again(self):
        cache = {"model": se.EMBED_MODEL, "papers": {}, "queries": {}}
        self.assertEqual(len(self.fill(cache, "citation")), 3)  # 2 papers + query
        self.assertEqual(se.missing_texts(cache, self.papers, "citation"), [])
        todo = se.missing_texts(cache, self.papers, "a new query")
        self.assertEqual([kind for _, kind, _ in todo], ["queries"])
        self.papers[0]["abstract"] = "A revised abstract."
        todo = se.missing_texts(cache, self.papers, "citation")
        self.assertEqual([key for key, _, _ in todo], ["2609.00001"])

    def test_scores_use_each_papers_own_vector(self):
        cache = {"model": se.EMBED_MODEL, "papers": {}, "queries": {}}
        self.fill(cache, "citation checking")
        scores = se.embedding_scores(cache, self.papers, "citation checking")
        self.assertEqual(se.rank(scores)[0], "2609.00001")

    def test_cache_from_another_model_is_not_reused(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "emb.json"
            se.save_cache({"model": "old-model", "papers": {"x": {}}, "queries": {}}, path)
            with contextlib.redirect_stdout(io.StringIO()):
                cache = se.load_cache(path)
            self.assertEqual(cache, {"model": se.EMBED_MODEL, "papers": {}, "queries": {}})

    def test_wrong_vector_count_is_rejected(self):
        client = SimpleNamespace(embeddings=SimpleNamespace(create=lambda **kw: SimpleNamespace(
            data=[SimpleNamespace(index=0, embedding=[1.0])], usage=SimpleNamespace(total_tokens=1))))
        with self.assertRaises(SetupError):
            se.embed_texts(client, ["one", "two"])

    def test_base64_round_trip(self):
        self.assertEqual(se.decode(se.encode([0.5, -1.25, 2.0])), [0.5, -1.25, 2.0])


class SnapshotTest(unittest.TestCase):
    def test_snapshot_never_overwrites(self):
        with tempfile.TemporaryDirectory() as d:
            src, dest = Path(d) / "papers.csv", Path(d) / "inputs"
            src.write_text("v1", encoding="utf-8")
            se.snapshot([src, Path(d) / "absent.csv"], dest)
            src.write_text("v2", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()) as out:
                se.snapshot([src], dest)
            self.assertEqual((dest / "papers.csv").read_text(encoding="utf-8"), "v1")
            self.assertIn("changed since the snapshot", out.getvalue())
            self.assertFalse((dest / "absent.csv").exists())

    def test_brief_must_be_a_single_stored_interest(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "sel.csv"
            path.write_text("code,interest\ncs.CL,RAG hallucination\ncs.IR,RAG hallucination\n", encoding="utf-8")
            self.assertEqual(se.stored_brief(path), "RAG hallucination")
            path.write_text("code,interest\ncs.CL,one\ncs.IR,two\n", encoding="utf-8")
            with self.assertRaises(SetupError):
                se.stored_brief(path)


if __name__ == "__main__":
    unittest.main()
