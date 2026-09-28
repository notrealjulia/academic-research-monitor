import contextlib
import csv
import io
import os
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

import research_monitor as rm
from tests.fakes import FAKE_KEY, FakeArxiv, FakeOpenAI, ScriptedInput, entry

TODAY = date(2026, 9, 28)


class ParseDateTest(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(rm.parse_date("21-09-2026", TODAY), date(2026, 9, 21))
        self.assertEqual(rm.parse_date("28-09-2026", TODAY), TODAY)  # today is allowed

    def test_rejects_other_formats_impossible_and_future_dates(self):
        for text in ["2026-09-21", "21/09/2026", "31-02-2026", "", "yesterday"]:
            with self.assertRaisesRegex(ValueError, "not a valid dd-mm-yyyy date"):
                rm.parse_date(text, TODAY)
        with self.assertRaisesRegex(ValueError, "in the future"):
            rm.parse_date("29-09-2026", TODAY)

    def test_window_is_inclusive_utc_calendar_days(self):
        start, end = rm.utc_window(date(2026, 9, 21), date(2026, 9, 22))
        self.assertEqual((start.isoformat(), end.isoformat()),
                         ("2026-09-21T00:00:00+00:00", "2026-09-22T23:59:00+00:00"))
        self.assertIn("(2 days, inclusive)", rm.describe_range(date(2026, 9, 21), date(2026, 9, 22)))
        self.assertIn("(1 day, inclusive)", rm.describe_range(TODAY, TODAY))


class ChooseDatesTest(unittest.TestCase):
    def run_dates(self, *answers):
        with mock.patch("builtins.input", ScriptedInput(*answers)), contextlib.redirect_stdout(io.StringIO()) as out:
            return rm.choose_dates(TODAY), out.getvalue()

    def test_blank_end_means_today(self):
        (start, end), _ = self.run_dates("21-09-2026", "")
        self.assertEqual((start, end), (date(2026, 9, 21), TODAY))

    def test_reasks_after_bad_input(self):
        (start, end), out = self.run_dates("2026-09-21", "30-09-2026", "21-09-2026", "20-09-2026", "31-09-2026",
                                           "22-09-2026")
        self.assertEqual((start, end), (date(2026, 9, 21), date(2026, 9, 22)))
        self.assertIn("not a valid dd-mm-yyyy date", out)
        self.assertIn("in the future", out)
        self.assertIn("must not be before the start date", out)

    def test_same_start_and_end_day(self):
        (start, end), _ = self.run_dates("25-09-2026", "25-09-2026")
        self.assertEqual(start, end)

    def test_x_exits(self):
        with self.assertRaises(SystemExit) as cm:
            self.run_dates("21-09-2026", "x")
        self.assertEqual(cm.exception.code, 0)


def picks_for(interest, taxonomy, api_key):
    """Stands in for find_categories.ask_model: different picks per description, plus an invented code."""
    assert api_key == FAKE_KEY
    if "revised" in interest:
        return [("cs.CL", "Revised: computational linguistics."), ("cs.NOPE", "invented")]
    return [("cs.CV", "First try: vision.")]


PAPERS = [
    entry("2609.00001", "2026-09-21T09:00:00Z", title="Low-resource machine translation evaluation"),
    entry("2609.00002", "2026-09-21T10:00:00Z", ("cs.LG", "cs.CL"), title="Graph neural networks"),
    entry("2609.00003", "2026-09-22T11:00:00Z", title="Translation quality estimation for Yoruba"),
]


@mock.patch.dict(os.environ, {"OPENAI_API_KEY": FAKE_KEY})
@mock.patch("fetch_papers.time.sleep", lambda s: None)
@mock.patch("find_categories.ask_model", picks_for)
@mock.patch("openai.OpenAI", FakeOpenAI)
class FlowTest(unittest.TestCase):
    def setUp(self):
        FakeOpenAI.reset()
        tmp = Path(tempfile.mkdtemp())
        self.selected, self.retrieved, self.matches = tmp / "selected.csv", tmp / "retrieved.csv", tmp / "matches.csv"
        self.selected.write_text("code,field,subfield,subject,reason,interest\nastro-ph.CO,,,,old,old interest\n",
                                 encoding="utf-8")
        for name, path in [("SELECTED", self.selected), ("RETRIEVED", self.retrieved), ("MATCHES", self.matches)]:
            patcher = mock.patch.object(rm, name, path)
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_flow(self, *answers, papers=PAPERS):
        self.arxiv = FakeArxiv(list(papers))
        self.input = ScriptedInput(*answers)
        out = io.StringIO()
        code = None
        with mock.patch("builtins.input", self.input), mock.patch("fetch_papers.fetch_page", self.arxiv), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            try:
                rm.main()
            except SystemExit as e:
                code = e.code
        return out.getvalue(), code

    def rows(self, path):
        with path.open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    # --- stage 1: category choices

    def test_exit_at_categories_keeps_previous_selection(self):
        out, code = self.run_flow("first try", "x")
        self.assertEqual(code, 0)
        self.assertIn("No categories were saved", out)
        self.assertEqual(self.rows(self.selected)[0]["interest"], "old interest")  # untouched
        self.assertEqual(self.arxiv.queries, [])

    def test_revise_then_accept_saves_only_accepted_suggestion(self):
        out, code = self.run_flow("first try", "?", "r", "", "revised: MT", "a", "x")
        self.assertIn("Please type a, r or x.", out)
        self.assertIn("Please enter a description.", out)
        self.assertIn("First try: vision.", out)
        self.assertIn("Ignored codes not in the taxonomy: cs.NOPE", out)
        rows = self.rows(self.selected)
        self.assertEqual([(r["code"], r["interest"]) for r in rows], [("cs.CL", "revised: MT")])
        self.assertEqual(rows[0]["subject"], "Computation and Language")  # from the taxonomy, not the model
        self.assertEqual(self.arxiv.queries, [])  # exited at the date prompt: nothing fetched

    def test_no_valid_categories_only_offers_revise_or_exit(self):
        with mock.patch("find_categories.ask_model", lambda *a: [("cs.NOPE", "invented")]):
            out, code = self.run_flow("anything", "a", "x")
        self.assertIn("No valid categories were suggested.", out)
        self.assertIn("Please type r or x.", out)
        self.assertEqual(code, 0)

    # --- stages 2-3: dates and fetching

    def test_fetches_accepted_categories_for_inclusive_range(self):
        out, code = self.run_flow("revised: MT", "a", "21-09-2026", "22-09-2026", "", "n")
        query, start = self.arxiv.queries[0]
        self.assertEqual(query, "(cat:cs.CL) AND submittedDate:[202609210000 TO 202609222359]")
        self.assertIn("from 21-09-2026 00:00 UTC through 22-09-2026 23:59 UTC (2 days, inclusive)", out)
        self.assertIn("Retrieved 3 unique papers", out)
        self.assertNotIn("Sample of retrieved papers", out)
        self.assertIn(f"Saved to {self.retrieved}\n\nStep 4 of 5", out)
        self.assertEqual(len(self.rows(self.retrieved)), 3)

    def test_fetch_finishes_and_saves_before_step_4(self):
        answers = ScriptedInput("revised: MT", "a", "21-09-2026", "22-09-2026", "", "n")
        seen = {}

        def watching_input(prompt=""):
            if prompt.startswith("Optional: add details"):
                seen["rows_on_disk"] = len(self.rows(self.retrieved))
                seen["fetch_requests"] = len(self.arxiv.queries)
            return answers(prompt)

        self.arxiv = FakeArxiv(list(PAPERS))
        with mock.patch("builtins.input", watching_input), mock.patch("fetch_papers.fetch_page", self.arxiv), \
                contextlib.redirect_stdout(io.StringIO()):
            rm.main()
        self.assertEqual(seen, {"rows_on_disk": 3, "fetch_requests": 1})

    def test_no_papers_stops_before_matching(self):
        out, code = self.run_flow("revised: MT", "a", "21-09-2026", "21-09-2026", papers=[])
        self.assertEqual(code, 0)
        self.assertIn("Retrieved 0 unique papers", out)
        self.assertIn("nothing to match", out)
        self.assertEqual(self.rows(self.retrieved), [])  # an empty file, not a stale one
        self.assertNotIn("Step 4", out)
        self.assertEqual(FakeOpenAI.inputs, [])

    # --- stages 4-5: details and matching

    def test_declining_matching_keeps_retrieved_papers(self):
        out, code = self.run_flow("revised: MT", "a", "21-09-2026", "22-09-2026", "Only evaluation.", "n")
        self.assertIn("Your research description:\n  revised: MT", out)
        self.assertIn("3 papers to screen", out)
        self.assertIn("in 1 requests.", out)
        self.assertNotIn("$", out)
        self.assertNotIn("tokens", out)
        self.assertIn("Run matching? [y/N]", self.input.prompts[-1])
        self.assertIn("No matching was run", out)
        self.assertEqual(FakeOpenAI.inputs, [])
        self.assertFalse(self.matches.exists())
        self.assertEqual(len(self.rows(self.retrieved)), 3)

    def test_eof_at_matching_prompt_counts_as_no(self):
        out, code = self.run_flow("revised: MT", "a", "21-09-2026", "22-09-2026", "")
        self.assertIn("No matching was run", out)
        self.assertEqual(FakeOpenAI.inputs, [])

    def test_full_flow_passes_both_descriptions_and_saves_matches(self):
        out, code = self.run_flow("revised: MT", "a", "21-09-2026", "22-09-2026", "Only evaluation.", "y")
        self.assertIsNone(code)
        [sent] = FakeOpenAI.inputs
        self.assertTrue(sent.startswith("Researcher's interests:\nrevised: MT\n\n"
                                        "Additional details about which papers the researcher wants:\n"
                                        "Only evaluation.\n\nPapers:\n"))
        rows = self.rows(self.matches)
        self.assertEqual([r["arxiv_id"] for r in rows], ["2609.00003", "2609.00001"])  # invented ID dropped
        self.assertIn("title and abstract only", out)
        self.assertIn("2 of the 3 retrieved papers look relevant", out)

    def test_eof_at_first_prompt_exits_cleanly(self):
        out, code = self.run_flow()
        self.assertEqual(code, 0)
        self.assertIn("Exited.", out)


if __name__ == "__main__":
    unittest.main()
