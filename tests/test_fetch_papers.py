import contextlib
import io
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import fetch_papers as fp
from tests.fakes import FakeArxiv, entry, feed

FIXTURE = Path(__file__).parent / "fixtures" / "arxiv_page.xml"  # a real arXiv API response


def utc(*args):
    return datetime(*args, tzinfo=timezone.utc)


class ParseFeedTest(unittest.TestCase):
    def test_real_arxiv_page(self):
        total, papers = fp.parse_feed(FIXTURE.read_bytes())
        self.assertEqual((total, len(papers)), (1069, 5))
        p = papers[0]
        self.assertEqual(p["arxiv_id"], "2609.31619")  # version suffix stripped
        self.assertEqual(p["primary_category"], "cs.AI")
        self.assertEqual(p["categories"], "cs.AI cs.CL cs.LG")  # primary first
        self.assertEqual(p["url"], "https://arxiv.org/abs/2609.31619")
        self.assertEqual(p["pdf_url"], "https://arxiv.org/pdf/2609.31619v1")
        self.assertTrue(p["title"] and p["abstract"] and p["authors"])

    def test_arxiv_error_entry_raises(self):
        err = feed(['<entry><id>http://arxiv.org/api/errors#bad</id><summary>bad query</summary></entry>'])
        with self.assertRaisesRegex(fp.SetupError, "bad query"):
            fp.parse_feed(err)

    def test_strip_version(self):
        self.assertEqual(fp.strip_version("2609.00001v12"), "2609.00001")
        self.assertEqual(fp.strip_version("hep-th/9901001v2"), "hep-th/9901001")
        self.assertEqual(fp.strip_version("solv-int/9901001"), "solv-int/9901001")


@mock.patch("fetch_papers.time.sleep", lambda s: None)
@mock.patch("builtins.print", lambda *a, **k: None)
class FetchPapersTest(unittest.TestCase):
    def test_query_covers_whole_minutes_of_the_range(self):
        q = fp.build_query(["cs.CL", "cs.LG"], utc(2026, 9, 21, 0, 0), utc(2026, 9, 22, 23, 59))
        self.assertEqual(q, "(cat:cs.CL OR cat:cs.LG) AND submittedDate:[202609210000 TO 202609222359]")

    def test_date_boundaries_are_inclusive_calendar_days(self):
        arxiv = FakeArxiv([
            entry("2609.00001", "2026-09-20T23:59:59Z"),  # day before: out
            entry("2609.00002", "2026-09-21T00:00:00Z"),  # first second: in
            entry("2609.00003", "2026-09-22T23:59:59Z"),  # last second: in
            entry("2609.00004", "2026-09-23T00:00:00Z"),  # day after: out
        ])
        with mock.patch("fetch_papers.fetch_page", arxiv):
            papers, _ = fp.fetch_papers(["cs.CL"], utc(2026, 9, 21, 0, 0), utc(2026, 9, 22, 23, 59))
        self.assertEqual([p["arxiv_id"] for p in papers], ["2609.00002", "2609.00003"])

    def test_pages_through_everything_retries_empty_page_and_deduplicates(self):
        pages = {
            0: [feed([entry("2609.00001", "2026-09-21T01:00:00Z"), entry("2609.00002", "2026-09-21T02:00:00Z")], 4)],
            2: [feed([], 4), feed([entry("2609.00002", "2026-09-21T02:00:00Z"),
                                   entry("2609.00003", "2026-09-21T03:00:00Z")], 4)],
        }
        calls = []

        def fake(query, start):
            calls.append(start)
            return pages[start].pop(0)

        with mock.patch("fetch_papers.fetch_page", fake):
            papers, total = fp.fetch_papers(["cs.CL"], utc(2026, 9, 21, 0, 0), utc(2026, 9, 21, 23, 59))
        self.assertEqual(calls, [0, 2, 2])  # empty page retried
        self.assertEqual([p["arxiv_id"] for p in papers], ["2609.00001", "2609.00002", "2609.00003"])

    def test_persistent_empty_page_fails_loudly(self):
        pages = {0: [feed([entry("2609.00001", "2026-09-21T01:00:00Z")], 3)], 1: [feed([], 3)] * 3}
        with mock.patch("fetch_papers.fetch_page", lambda q, s: pages[s].pop(0)):
            with self.assertRaisesRegex(fp.SetupError, "after 3 tries: empty page"):
                fp.fetch_papers(["cs.CL"], utc(2026, 9, 21, 0, 0), utc(2026, 9, 21, 23, 59))


class FakeTerminal(io.StringIO):
    def isatty(self):
        return True


def screen(raw):
    """What a terminal would show: each carriage return redraws the current line."""
    return "\n".join(line.rsplit("\r", 1)[-1] for line in raw.split("\n"))


WINDOW = (utc(2026, 9, 21, 0, 0), utc(2026, 9, 21, 23, 59))
MANY = [entry(f"2609.{i:05d}", "2026-09-21T01:00:00Z") for i in range(1069)]


@mock.patch("fetch_papers.time.sleep", lambda s: None)
class ProgressTest(unittest.TestCase):
    def fetch(self, fake_page, terminal=True):
        out = FakeTerminal() if terminal else io.StringIO()
        error = None
        with mock.patch("fetch_papers.fetch_page", fake_page), contextlib.redirect_stdout(out):
            try:
                papers, total = fp.fetch_papers(["cs.CL"], *WINDOW)
            except fp.SetupError as e:
                papers, error = None, e
        return out.getvalue(), papers, error

    def test_bar_redraws_in_place_and_ends_with_newline(self):
        raw, papers, _ = self.fetch(FakeArxiv(MANY))
        self.assertEqual(len(papers), 1069)
        self.assertEqual(raw.count("\r"), 3)  # one redraw per page
        self.assertTrue(raw.endswith("\n"))
        self.assertEqual(screen(raw).splitlines(), [
            "arXiv reports 1,069 papers. Fetching 500 per request, 3 s apart...",
            "  [##############################] 1,069/1,069",
        ])
        self.assertIn("\r  [##############................]   500/1,069", raw)
        self.assertIn("\r  [############################..] 1,000/1,069", raw)

    def test_retry_notice_gets_its_own_line(self):
        arxiv, failed = FakeArxiv(MANY), []

        def flaky(query, start):  # the page at 500 comes back empty once, then succeeds
            if start == 500 and not failed:
                failed.append(start)
                return feed([], total=1069)
            return arxiv(query, start)

        raw, papers, _ = self.fetch(flaky)
        self.assertEqual(len(papers), 1069)
        self.assertEqual(screen(raw).splitlines(), [
            "arXiv reports 1,069 papers. Fetching 500 per request, 3 s apart...",
            "  [##############................]   500/1,069",
            "  arXiv request at result 500 failed (empty page); retrying (2/3)...",
            "  [##############################] 1,069/1,069",
        ])

    def test_failure_closes_the_bar_line_before_the_error(self):
        arxiv = FakeArxiv(MANY)
        raw, papers, error = self.fetch(lambda q, s: feed([], total=1069) if s == 500 else arxiv(q, s))
        self.assertIn("after 3 tries: empty page", str(error))
        self.assertTrue(raw.endswith("\n"))
        self.assertEqual(screen(raw).splitlines()[-2:], [
            "  arXiv request at result 500 failed (empty page); retrying (3/3)...",
            "  [##############................]   500/1,069",
        ])

    def test_redirected_output_prints_one_plain_line_per_page(self):
        raw, papers, _ = self.fetch(FakeArxiv(MANY), terminal=False)
        self.assertNotIn("\r", raw)
        self.assertEqual(raw.splitlines()[1:], [
            "  [##############................]   500/1,069",
            "  [############################..] 1,000/1,069",
            "  [##############################] 1,069/1,069",
        ])

    def test_no_papers_draws_no_bar(self):
        raw, papers, _ = self.fetch(FakeArxiv([]))
        self.assertEqual(papers, [])
        self.assertEqual(raw, "arXiv reports 0 papers. Fetching 500 per request, 3 s apart...\n")


@mock.patch("fetch_papers.time.sleep", lambda s: None)
class MainOutputTest(unittest.TestCase):
    def test_prints_count_and_path_without_samples(self):
        tmp = Path(tempfile.mkdtemp()) / "retrieved.csv"
        arxiv = FakeArxiv([entry("2609.00001", "2026-09-21T01:00:00Z"), entry("2609.00002", "2026-09-21T02:00:00Z")])
        save = fp.save_papers
        with mock.patch("fetch_papers.fetch_page", arxiv), mock.patch("fetch_papers.load_codes", lambda: ["cs.CL"]), \
                mock.patch("fetch_papers.save_papers", lambda papers: save(papers, tmp)), \
                mock.patch("sys.argv", ["fetch_papers", "2026-09-21"]), contextlib.redirect_stdout(io.StringIO()) as out:
            fp.main()
        text = out.getvalue()
        self.assertIn("Retrieved 2 unique papers submitted on 2026-09-21 (arXiv reported 2).", text)
        self.assertIn(f"Saved to {fp.RETRIEVED}", text)
        self.assertNotIn("Sample of retrieved papers", text)
        self.assertEqual(len(tmp.read_text(encoding="utf-8").splitlines()), 3)  # header + 2 papers


if __name__ == "__main__":
    unittest.main()
