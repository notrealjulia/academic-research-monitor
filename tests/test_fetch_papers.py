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


if __name__ == "__main__":
    unittest.main()
