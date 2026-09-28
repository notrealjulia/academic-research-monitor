"""Test doubles: scripted terminal input, fake arXiv pages and a fake OpenAI client. No network."""

import re
import threading
from types import SimpleNamespace

FAKE_KEY = "sk-test-not-a-real-key"


class ScriptedInput:
    """Replaces input(): returns the given answers in order, then raises EOFError."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.prompts = []

    def __call__(self, prompt=""):
        self.prompts.append(prompt)
        if not self.answers:
            raise EOFError
        return self.answers.pop(0)


def entry(arxiv_id, published, cats=("cs.CL",), title=None):
    title = title or f"Paper {arxiv_id}"
    terms = "".join(f'<category term="{c}"/>' for c in cats)
    return (f"<entry><id>http://arxiv.org/abs/{arxiv_id}v1</id><published>{published}</published>"
            f"<title>{title}</title><summary>Abstract of {arxiv_id}.</summary><author><name>A. Author</name></author>"
            f'<arxiv:primary_category term="{cats[0]}"/>{terms}'
            f'<link href="https://arxiv.org/pdf/{arxiv_id}v1" rel="related" title="pdf"/></entry>')


def feed(entries, total=None):
    total = len(entries) if total is None else total
    return (f'<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom" '
            f'xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">'
            f"<opensearch:totalResults>{total}</opensearch:totalResults>{''.join(entries)}</feed>").encode()


class FakeArxiv:
    """Replaces fetch_papers.fetch_page: serves one list of entries, paged by offset."""

    def __init__(self, entries):
        self.entries = entries
        self.queries = []

    def __call__(self, query, start):
        self.queries.append((query, start))
        return feed(self.entries[start:start + 500], total=len(self.entries))


class FakeOpenAI:
    """Replaces openai.OpenAI. Matches papers whose title contains `keyword`; records every request."""

    keyword = "translation"
    fail_on_call = None  # 1-based request number that raises a connection error
    inputs = []
    lock = threading.Lock()

    def __init__(self, api_key):
        assert api_key == FAKE_KEY, "tests must never use the real key"
        self.responses = SimpleNamespace(parse=self.parse)

    @classmethod
    def reset(cls, keyword="translation", fail_on_call=None):
        cls.keyword, cls.fail_on_call, cls.inputs = keyword, fail_on_call, []

    def parse(self, model, reasoning, instructions, input, text_format):
        with self.lock:
            FakeOpenAI.inputs.append(input)
            n = len(FakeOpenAI.inputs)
        if n == self.fail_on_call:
            import httpx2
            import openai
            raise openai.APIConnectionError(request=httpx2.Request("POST", "https://api.openai.com/v1/responses"))
        ids = re.findall(r"^ID: (\S+)$", input, re.M)
        titles = re.findall(r"^Title: (.*)$", input, re.M)
        picks = [{"arxiv_id": i, "reason": f"Mentions {self.keyword}."}
                 for i, t in zip(ids, titles) if self.keyword in t.lower()]
        picks.append({"arxiv_id": "9999.99999", "reason": "An ID that was not in the batch."})
        return SimpleNamespace(output_parsed=text_format(matches=picks))
