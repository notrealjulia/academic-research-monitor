import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import find_paper_matches as fm
import rerank_papers as rr
import search_experiment as se
from find_categories import SetupError
from tests.fakes import FAKE_KEY, FakeEmbeddingsOpenAI, FakeRerankOpenAI, rate_limit_error


def paper(i, title, abstract=None):
    return {"arxiv_id": f"2609.{i:05d}", "title": title, "abstract": abstract or f"About {title.lower()}.",
            "authors": "A", "submitted": "2026-09-21T09:00:00Z", "primary_category": "cs.CL",
            "categories": "cs.CL", "url": f"https://arxiv.org/abs/2609.{i:05d}", "pdf_url": ""}


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


class ShortlistTest(unittest.TestCase):
    def test_top_ten_percent_rounded_up(self):
        self.assertEqual(rr.SHORTLIST_PERCENT, 10)
        for retrieved, shortlisted in [(2197, 220), (845, 85), (411, 42), (100, 10), (101, 11), (9, 1), (1, 1),
                                       (0, 0)]:
            self.assertEqual(rr.shortlist_size(retrieved), shortlisted, retrieved)

    def test_percentage_is_configurable(self):
        self.assertEqual(rr.shortlist_size(845, percent=25), 212)  # 211.25 rounds up
        self.assertEqual(rr.shortlist_size(845, percent=100), 845)
        with mock.patch.object(rr, "SHORTLIST_PERCENT", 7):
            self.assertEqual(rr.shortlist_size(845), 60)  # 59.15 rounds up

    def test_retrieval_query_keeps_details_separate(self):
        self.assertEqual(rr.query_text("RAG"), "RAG")
        self.assertEqual(rr.query_text("RAG", "No surveys."), "RAG\n\nNo surveys.")


class CleanKeywordsTest(unittest.TestCase):
    def test_trims_drops_unusable_and_deduplicates_case_insensitively(self):
        got = rr.clean_keywords(["  citation   checking ", "", "!!!", "RAG", "rag", "x" * (rr.MAX_KEYWORD_CHARS + 1),
                                 "Citation checking"])
        self.assertEqual(got, ["citation checking", "RAG"])

    def test_caps_the_number_of_terms(self):
        self.assertEqual(rr.clean_keywords([f"term{i}" for i in range(50)]),
                         [f"term{i}" for i in range(rr.MAX_KEYWORDS)])


class CollectTest(unittest.TestCase):
    def setUp(self):
        self.batch = [paper(i, f"Paper {i}") for i in range(3)]

    def test_labels_map_to_this_batchs_papers(self):
        got = rr.collect([("P01", "accept", 80, " a "), ("p03", "reject", 5, "c")], self.batch)
        self.assertEqual(got, {"2609.00000": ("accept", 80, "a"), "2609.00002": ("reject", 5, "c")})

    def test_unknown_labels_bad_scores_and_decisions_are_dropped(self):
        got = rr.collect([("P04", "accept", 80, "x"), ("2609.00000", "accept", 80, "x"),
                          ("P01", "accept", 101, "x"), ("P02", "accept", -1, "x"), ("P03", "maybe", 50, "x")],
                         self.batch)
        self.assertEqual(got, {})

    def test_duplicate_label_keeps_first_decision(self):
        got = rr.collect([("P02", "reject", 10, "first"), ("P02", "accept", 100, "second")], self.batch)
        self.assertEqual(got, {"2609.00001": ("reject", 10, "first")})

    def test_same_label_in_different_batches(self):
        other = [paper(i, f"Paper {i}") for i in range(20, 23)]
        self.assertEqual(list(rr.collect([("P01", "accept", 50, "r")], self.batch)), ["2609.00000"])
        self.assertEqual(list(rr.collect([("P01", "accept", 50, "r")], other)), ["2609.00020"])

    def test_accepted_sorted_by_score_then_id_and_rejects_dropped(self):
        by_id = {p["arxiv_id"]: p for p in self.batch}
        decisions = {"2609.00002": ("accept", 60, "c"), "2609.00000": ("accept", 60, "a"),
                     "2609.00001": ("reject", 99, "b")}
        got = rr.accepted(decisions, by_id)
        self.assertEqual([(p["arxiv_id"], p["relevance_score"]) for p in got], [("2609.00000", 60), ("2609.00002", 60)])
        self.assertEqual(rr.accepted({}, by_id), [])


class RetryTest(unittest.TestCase):
    def setUp(self):
        self.sleeps = []

    def run_calls(self, *outcomes, on_retry=None):
        """Each outcome is an exception to raise or a value to return, in order."""
        outcomes, calls = list(outcomes), []

        def call():
            calls.append(1)
            o = outcomes.pop(0)
            if isinstance(o, Exception):
                raise o
            return o

        result = quiet(rr.with_retries, call, sleep=self.sleeps.append, on_retry=on_retry)
        return result, len(calls)

    def test_rate_limit_waits_retry_after_then_succeeds(self):
        result, calls = self.run_calls(rate_limit_error(retry_after=3), "ok")
        self.assertEqual((result, calls, self.sleeps), ("ok", 2, [3.0]))

    def test_each_retry_is_reported_with_its_wait(self):
        seen = []
        self.run_calls(rate_limit_error(retry_after=3), rate_limit_error(), "ok", on_retry=seen.append)
        self.assertEqual(seen, [3.0, 2.0])
        self.assertEqual(seen, self.sleeps)

    def test_rate_limit_without_header_backs_off_exponentially(self):
        result, calls = self.run_calls(rate_limit_error(), rate_limit_error(), rate_limit_error(), "ok")
        self.assertEqual((result, self.sleeps), ("ok", [1.0, 2.0, 4.0]))

    def test_quota_errors_are_never_retried(self):
        for code in ("insufficient_quota", "credit_balance_exhausted", "project_spend_limit_exceeded"):
            with self.assertRaises(SetupError) as ctx:
                self.run_calls(rate_limit_error(code=code), "never reached")
            self.assertIn("not retried", str(ctx.exception))
        self.assertEqual(self.sleeps, [])

    def test_persistent_rate_limit_fails_after_bounded_retries(self):
        with self.assertRaises(SetupError) as ctx:
            self.run_calls(*[rate_limit_error() for _ in range(rr.RATE_LIMIT_RETRIES + 1)])
        self.assertIn("still reached", str(ctx.exception))
        self.assertEqual(len(self.sleeps), rr.RATE_LIMIT_RETRIES)
        self.assertTrue(all(s <= rr.MAX_WAIT for s in self.sleeps))

    def test_excessive_retry_after_fails_instead_of_stalling(self):
        with self.assertRaises(SetupError):
            self.run_calls(rate_limit_error(retry_after=rr.MAX_WAIT + 1), "never reached")
        self.assertEqual(self.sleeps, [])

    def test_connection_errors_are_retried(self):
        import httpx2
        import openai
        error = openai.APIConnectionError(request=httpx2.Request("POST", "https://api.openai.com/v1/responses"))
        result, calls = self.run_calls(error, "ok")
        self.assertEqual((result, calls), ("ok", 2))

    def test_other_errors_are_not_retried(self):
        import httpx2
        import openai
        response = httpx2.Response(401, request=httpx2.Request("POST", "https://api.openai.com/v1/responses"))
        with self.assertRaises(openai.AuthenticationError):
            self.run_calls(openai.AuthenticationError("bad key", response=response, body=None), "never reached")
        self.assertEqual(self.sleeps, [])


@mock.patch("openai.OpenAI", FakeRerankOpenAI)
class RunTest(unittest.TestCase):
    def setUp(self):
        FakeRerankOpenAI.reset()
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self.tmp.name) / "embeddings.json"
        self.papers = [paper(1, "Machine translation evaluation"), paper(2, "Graph neural networks"),
                       paper(3, "Translation quality estimation"), paper(4, "Protein folding")]
        # These tests use a handful of papers, where 10% would shortlist just one. Shortlist them all unless a
        # test is about the shortlist size itself.
        patcher = mock.patch.object(rr, "SHORTLIST_PERCENT", 100)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def test_default_shortlist_is_ten_percent_rounded_up(self):
        papers = [paper(i, "Translation study" if i % 3 == 0 else f"Topic {i}") for i in range(45)]
        with mock.patch.object(rr, "SHORTLIST_PERCENT", 10):
            matches, stats, error = self.run_path(papers)
        self.assertEqual((stats["shortlist_percent"], stats["shortlisted"], stats["assessed"]), (10, 5, 5))
        [sent] = FakeRerankOpenAI.inputs
        self.assertEqual(sent.count("Title: "), 5)  # only the shortlist reaches the model
        self.assertEqual(len(stats["shortlist_ids"]), 5)

    def test_plan_shows_shortlist_count_and_percentage(self):
        papers = [paper(i, f"Topic {i}") for i in range(411)]
        with mock.patch.object(rr, "SHORTLIST_PERCENT", 10):
            text = rr.describe_plan(papers, "translation", cache_path=self.cache)
        self.assertIn("shortlist the top 42 (10% of 411, rounded up)", text)
        self.assertIn("Screening will send those 42 titles and abstracts", text)
        self.assertIn("in 3 requests", text)  # 42 papers in batches of 20

    def run_path(self, papers=None, details=""):
        return quiet(rr.run, "translation", papers or self.papers, FAKE_KEY, details, self.cache)

    def test_embeddings_are_saved_and_reused_only_for_same_text_and_model(self):
        _, stats, _ = self.run_path()
        self.assertEqual(sum(map(len, FakeEmbeddingsOpenAI.inputs)), 5)  # 4 papers + the query
        self.assertEqual((stats["papers_embedded"], stats["papers_cached"]), (4, 0))
        _, stats, _ = self.run_path()
        self.assertEqual((stats["texts_embedded"], stats["papers_embedded"], stats["papers_cached"]), (0, 0, 4))
        self.papers[1]["abstract"] = "A revised abstract."
        _, stats, _ = self.run_path()
        self.assertEqual((stats["texts_embedded"], stats["papers_embedded"], stats["papers_cached"]), (1, 1, 3))
        _, stats, _ = self.run_path(details="A new query text.")  # only the query is new
        self.assertEqual((stats["texts_embedded"], stats["papers_embedded"]), (1, 0))
        saved = json.loads(self.cache.read_text(encoding="utf-8"))
        self.cache.write_text(json.dumps({**saved, "model": "an-older-embedding-model"}), encoding="utf-8")
        _, stats, _ = self.run_path()
        self.assertEqual(stats["texts_embedded"], 5)  # vectors from another model are never mixed in

    def test_hybrid_order_ranks_every_paper_and_fuses_both_rankings(self):
        self.run_path()
        cache = se.load_cache(self.cache)
        order = rr.hybrid_order(self.papers, cache, "translation")
        self.assertEqual(sorted(order), sorted(p["arxiv_id"] for p in self.papers))
        emb = se.rank(se.embedding_scores(cache, self.papers, "translation"))
        bm25 = se.rank(se.bm25_scores("translation", {p["arxiv_id"]: se.paper_text(p) for p in self.papers}))
        self.assertEqual(order, se.rank(se.rrf(emb, bm25)))
        self.assertIn(order[0], {"2609.00001", "2609.00003"})  # the two papers that say "translation"

    def test_only_the_shortlist_is_screened_in_bounded_labelled_batches(self):
        papers = [paper(i, "Translation study" if i % 7 == 0 else f"Topic {i}") for i in range(50)]
        with mock.patch.object(rr, "SHORTLIST_PERCENT", 60), mock.patch.object(rr, "RERANK_BATCH", 12):
            matches, stats, error = self.run_path(papers)
        self.assertIsNone(error)
        self.assertEqual((stats["shortlisted"], stats["batches"], len(FakeRerankOpenAI.inputs)), (30, 3, 3))
        self.assertEqual(sorted(i.count("Title: ") for i in FakeRerankOpenAI.inputs), [6, 12, 12])
        self.assertTrue(all("2609." not in i for i in FakeRerankOpenAI.inputs))  # labels only, no arXiv IDs
        shortlisted = set(stats["shortlist_ids"])
        self.assertTrue({m["arxiv_id"] for m in matches} <= shortlisted)
        self.assertEqual(FakeRerankOpenAI.max_retries[-1], 0)  # with_retries is the only retry loop

    def test_search_ranks_and_scores_are_not_sent_to_the_model(self):
        self.run_path(details="Only evaluation papers.")
        [sent] = FakeRerankOpenAI.inputs
        self.assertTrue(sent.startswith("Researcher's interests:\ntranslation\n\n"
                                        "Additional details about which papers the researcher wants:\n"
                                        "Only evaluation papers.\n\nPapers:\nPaper: P01\n"))
        for leaked in ("rank", "score", "similarity", "bm25", "rrf"):
            self.assertNotIn(leaked, sent.lower())

    def test_accepted_papers_come_back_in_score_order_without_invented_ones(self):
        matches, stats, error = self.run_path()
        self.assertEqual([(m["arxiv_id"], m["relevance_score"]) for m in matches],
                         [("2609.00003", 90), ("2609.00001", 70)])
        self.assertEqual((stats["assessed"], stats["not_assessed"], stats["accepted"]), (4, 0, 2))
        self.assertEqual(matches[0]["title"], "Translation quality estimation")  # from our CSV, not the model

    # --- keyword extraction (BM25 query only)

    BRIEF = "I study citation checking. I am not interested in chatbot benchmarks."

    def keyword_papers(self):
        return [paper(1, "Citation verification for RAG"), paper(2, "Benchmarks of chatbot benchmarks"),
                paper(3, "Protein folding"), paper(4, "Graph neural networks")]

    def test_keywords_are_the_bm25_query_and_embeddings_keep_the_description(self):
        FakeRerankOpenAI.reset(keyword="citation", keywords=["citation verification", "RAG"])
        papers = self.keyword_papers()
        with mock.patch.object(se, "bm25_scores", wraps=se.bm25_scores) as bm25_spy:
            matches, stats, error = quiet(rr.run, self.BRIEF, papers, FAKE_KEY, "", self.cache)
        [call] = bm25_spy.call_args_list
        self.assertEqual(call.args[0], "citation verification RAG")  # BM25 got the keywords, not the description
        self.assertEqual(call.args[1], {p["arxiv_id"]: se.paper_text(p) for p in papers})  # paper text unchanged
        self.assertEqual(stats["bm25_keywords"], ["citation verification", "RAG"])
        embedded = [t for batch in FakeEmbeddingsOpenAI.inputs for t in batch]
        self.assertIn(self.BRIEF, embedded)  # the description is embedded unchanged
        self.assertNotIn("citation verification RAG", embedded)  # the keywords are not
        cache = se.load_cache(self.cache)
        emb = se.rank(se.embedding_scores(cache, papers, self.BRIEF))
        bm25 = se.rank(se.bm25_scores("citation verification RAG",
                                      {p["arxiv_id"]: se.paper_text(p) for p in papers}))
        self.assertEqual(stats["shortlist_ids"], se.rank(se.rrf(emb, bm25)))  # fusion and shortlist unchanged
        # What the change is for: BM25 on the description ranks the excluded topic first, the keywords do not.
        old_bm25 = se.rank(se.bm25_scores(self.BRIEF, {p["arxiv_id"]: se.paper_text(p) for p in papers}))
        self.assertEqual((old_bm25[0], bm25[0]), ("2609.00002", "2609.00001"))

    def test_keyword_request_sees_only_description_and_details(self):
        quiet(rr.run, self.BRIEF, self.keyword_papers(), FAKE_KEY, "Only evaluation papers.", self.cache)
        [sent] = FakeRerankOpenAI.keyword_inputs
        self.assertEqual(json.loads(sent), {"research_description": self.BRIEF,
                                            "extra_details": "Only evaluation papers."})
        self.assertEqual(len(FakeRerankOpenAI.inputs), 1)  # screening is unchanged: one batch

    def test_no_usable_keywords_stops_before_embedding_or_screening(self):
        for unusable in ([], ["", "!!!", "   "]):
            FakeRerankOpenAI.reset(keywords=unusable)
            with mock.patch.object(se, "bm25_scores", wraps=se.bm25_scores) as bm25_spy:
                with self.assertRaises(SetupError) as ctx:
                    quiet(rr.run, self.BRIEF, self.keyword_papers(), FAKE_KEY, "", self.cache)
            self.assertIn("no usable keyword search terms", str(ctx.exception))
            self.assertIn("Nothing was embedded or screened", str(ctx.exception))
            self.assertEqual((FakeEmbeddingsOpenAI.inputs, FakeRerankOpenAI.inputs), ([], []))
            self.assertEqual(bm25_spy.call_count, 0)  # no ranking with the description either
            self.assertFalse(self.cache.exists())

    def test_keyword_rate_limit_is_retried_and_counted(self):
        FakeRerankOpenAI.reset(keyword_errors=[rate_limit_error(retry_after=2)])
        with mock.patch("rerank_papers.time.sleep", lambda s: None):
            _, stats, error = quiet(rr.run, self.BRIEF, self.keyword_papers(), FAKE_KEY, "", self.cache)
        self.assertIsNone(error)
        self.assertEqual((stats["keyword_extraction"]["retries"], stats["keyword_extraction"]["retry_wait_seconds"]),
                         (1, 2.0))
        self.assertEqual(stats["screening"]["retries"], 0)

    def test_keyword_quota_error_stops_before_anything_else_is_sent(self):
        FakeRerankOpenAI.reset(keyword_errors=[rate_limit_error(code="insufficient_quota")])
        with self.assertRaises(SetupError):
            quiet(rr.run, self.BRIEF, self.keyword_papers(), FAKE_KEY, "", self.cache)
        self.assertEqual((FakeEmbeddingsOpenAI.inputs, FakeRerankOpenAI.inputs), ([], []))

    def test_stage_times_are_recorded_without_retries(self):
        _, stats, _ = self.run_path()
        self.assertEqual(set(stats["seconds"]), {"keywords", "embedding", "ranking", "screening", "total"})
        self.assertTrue(all(v >= 0 for v in stats["seconds"].values()))
        self.assertEqual((stats["screening"]["retries"], stats["screening"]["retry_wait_seconds"]), (0, 0.0))

    def test_empty_result_is_valid(self):
        FakeRerankOpenAI.reset(keyword="nothing matches this")
        matches, stats, error = self.run_path()
        self.assertEqual((matches, error, stats["accepted"]), ([], None, 0))

    def test_rate_limited_batch_waits_and_retries(self):
        FakeRerankOpenAI.reset(errors=[rate_limit_error(retry_after=3)])
        sleeps = []
        with mock.patch("rerank_papers.time.sleep", sleeps.append):
            matches, stats, error = self.run_path()
        self.assertIsNone(error)
        self.assertEqual(sleeps, [3.0])
        self.assertEqual((stats["screening"]["retries"], stats["screening"]["retry_wait_seconds"]), (1, 3.0))
        self.assertEqual(stats["screening"]["requests"], 1)  # a retried request still counts once
        self.assertEqual(len(FakeRerankOpenAI.inputs), 2)
        self.assertEqual(len(matches), 2)

    def test_quota_error_stops_and_keeps_finished_batches(self):
        papers = [paper(i, "Translation study") for i in range(45)]
        FakeRerankOpenAI.reset(errors=[rate_limit_error(code="insufficient_quota")])
        with mock.patch.object(rr, "WORKERS", 1):
            matches, stats, error = self.run_path(papers)
        self.assertIn("not retried", str(error))
        self.assertEqual(stats["batches_done"], 0)
        self.assertEqual(stats["batches"], 3)
        self.assertEqual(len(FakeRerankOpenAI.inputs), 1)  # the rest were cancelled, never sent


class SaveAndReportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.out, self.papers_csv = d / "paper_matches.csv", d / "retrieved.csv"
        self.papers_csv.write_text("x", encoding="utf-8")
        self.stats = {"retrieved": 4, "shortlisted": 4, "assessed": 4, "not_assessed": 0, "accepted": 1,
                      "batches_done": 1, "batches": 1}

    def tearDown(self):
        self.tmp.cleanup()

    def test_csv_has_scores_and_settings_record_the_run(self):
        match = {**paper(3, "Translation quality estimation"), "relevance_score": 90, "match_reason": "Fits."}
        rr.save([match], self.stats, "translation", "No surveys.", None, self.papers_csv, self.out)
        rows = fm.load_papers(self.out)
        self.assertEqual((rows[0]["arxiv_id"], rows[0]["relevance_score"]), ("2609.00003", "90"))
        settings = json.loads(rr.settings_path(self.out).read_text(encoding="utf-8"))
        self.assertEqual(settings["screening_model"], rr.MODEL)
        self.assertEqual(settings["embedding_model"], se.EMBED_MODEL)
        self.assertEqual((settings["shortlist_percent"], settings["shortlisted"], settings["details"],
                          settings["complete"]), (rr.SHORTLIST_PERCENT, 4, "No surveys.", True))
        self.assertIn("not a probability", settings["relevance_score"])

    def test_llm_only_save_removes_stale_rerank_settings(self):
        rr.save([], self.stats, "translation", "", None, self.papers_csv, self.out)
        self.assertTrue(rr.settings_path(self.out).exists())
        fm.save([], self.out)
        self.assertFalse(rr.settings_path(self.out).exists())

    def test_printed_results_explain_scores(self):
        match = {**paper(3, "Translation quality estimation"), "relevance_score": 90, "match_reason": "Fits."}
        with contextlib.redirect_stdout(io.StringIO()) as out:
            rr.print_results([match], self.stats, None)
        self.assertIn("1. [90] Translation quality estimation", out.getvalue())
        self.assertIn("they are not probabilities", out.getvalue())
        self.assertIn("1 accepted of 4 assessed (shortlist of 4 from 4 retrieved papers)", out.getvalue())


if __name__ == "__main__":
    unittest.main()
