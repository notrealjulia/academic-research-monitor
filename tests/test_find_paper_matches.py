import unittest
from unittest import mock

import find_paper_matches as fm
from tests.fakes import FAKE_KEY, FakeOpenAI


def paper(i, title):
    return {"arxiv_id": f"2609.{i:05d}", "title": title, "abstract": f"Abstract {i}.", "authors": "A",
            "submitted": f"2026-09-21T{i % 24:02d}:00:00Z", "primary_category": "cs.CL", "categories": "cs.CL",
            "url": "", "pdf_url": ""}


@mock.patch("openai.OpenAI", FakeOpenAI)
class MatchPapersTest(unittest.TestCase):
    def setUp(self):
        FakeOpenAI.reset()
        self.papers = [paper(i, "Machine translation study" if i % 10 == 0 else "Something else")
                       for i in range(100)]  # 3 batches of up to 40

    def test_prompt_without_details_is_unchanged(self):
        self.assertEqual(fm.format_interest("MT for Yoruba"), "Researcher's interests:\nMT for Yoruba")

    def test_details_are_a_separate_section(self):
        text = fm.format_interest("MT for Yoruba", "Only evaluation papers.")
        self.assertEqual(text, "Researcher's interests:\nMT for Yoruba\n\n"
                               "Additional details about which papers the researcher wants:\nOnly evaluation papers.")

    def test_matches_only_labels_from_the_batch(self):
        with mock.patch("builtins.print"):
            matches, done, error = fm.match_papers("MT", self.papers, FAKE_KEY, "Only evaluation.")
        self.assertIsNone(error)
        self.assertEqual(done, 3)
        self.assertEqual(len(FakeOpenAI.inputs), 3)
        self.assertEqual(sorted(m["arxiv_id"] for m in matches), [f"2609.{i:05d}" for i in range(0, 100, 10)])
        self.assertTrue(all("Additional details about which papers the researcher wants:\nOnly evaluation."
                            in i for i in FakeOpenAI.inputs))

    def test_failure_keeps_finished_batches_and_is_reported(self):
        FakeOpenAI.reset(fail_on_call=2)
        with mock.patch("builtins.print"):
            matches, done, error = fm.match_papers("MT", self.papers, FAKE_KEY)
        self.assertIn("Could not reach the OpenAI API", str(error))
        self.assertEqual(done, len(FakeOpenAI.inputs) - 1)
        self.assertGreaterEqual(done, 1)

    def test_prompt_uses_labels_not_arxiv_ids(self):
        text = fm.format_batch(self.papers[:2])
        self.assertEqual(text, "Paper: P01\nTitle: Machine translation study\nAbstract: Abstract 0.\n\n"
                               "Paper: P02\nTitle: Something else\nAbstract: Abstract 1.")
        self.assertNotIn("2609.", text)

    def test_collect_maps_labels_to_papers(self):
        got = fm.collect([("P01", " a "), ("p03", "c")], self.papers[:3])
        self.assertEqual(got, {"2609.00000": "a", "2609.00002": "c"})

    def test_collect_rejects_unknown_labels(self):
        batch = self.papers[:2]
        got = fm.collect([("P03", "x"), ("P00", "x"), ("2609.00000", "x"), ("", "x"), ("P1", "x")], batch)
        self.assertEqual(got, {})

    def test_collect_keeps_first_reason_for_duplicate_labels(self):
        got = fm.collect([("P02", "first"), ("P02", "second"), (" p02", "third")], self.papers[:2])
        self.assertEqual(got, {"2609.00001": "first"})

    def test_same_label_in_different_batches_maps_to_each_batchs_paper(self):
        first, second = self.papers[:40], self.papers[40:80]
        self.assertEqual(fm.collect([("P01", "a")], first), {"2609.00000": "a"})
        self.assertEqual(fm.collect([("P01", "b")], second), {"2609.00040": "b"})
        # Through the full run: every batch reuses P01, P11, ... and each maps to its own paper.
        with mock.patch("builtins.print"):
            matches, _, _ = fm.match_papers("MT", self.papers, FAKE_KEY)
        self.assertTrue(all("Paper: P01\n" in i for i in FakeOpenAI.inputs))
        self.assertEqual({m["arxiv_id"] for m in matches}, {f"2609.{i:05d}" for i in range(0, 100, 10)})
        self.assertTrue(all(m["title"] == "Machine translation study" for m in matches))


if __name__ == "__main__":
    unittest.main()
