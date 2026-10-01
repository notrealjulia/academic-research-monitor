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


class FakeEmbeddingsOpenAI:
    """Replaces openai.OpenAI for embeddings. A text's vector counts the letters a-z; data come back reversed."""

    inputs = []  # one list of texts per request

    def __init__(self, api_key):
        assert api_key == FAKE_KEY, "tests must never use the real key"
        self.embeddings = SimpleNamespace(create=self.create)

    @staticmethod
    def vector(text):
        return [float(text.lower().count(c)) for c in "abcdefghijklmnopqrstuvwxyz"]

    def create(self, model, input, encoding_format):
        FakeEmbeddingsOpenAI.inputs.append(list(input))
        data = [SimpleNamespace(index=n, embedding=self.vector(t)) for n, t in enumerate(input)]
        return SimpleNamespace(data=data[::-1], usage=SimpleNamespace(total_tokens=sum(len(t) for t in input)))


class FakeJudgeOpenAI:
    """Replaces openai.OpenAI for the judge. Verdict from keywords in the title; quotes the abstract's first 40 chars."""

    inputs = []
    fail_on_call = None  # 1-based request number that raises a connection error
    lock = threading.Lock()

    def __init__(self, api_key):
        assert api_key == FAKE_KEY, "tests must never use the real key"
        self.responses = SimpleNamespace(parse=self.parse)

    @classmethod
    def reset(cls, fail_on_call=None):
        cls.inputs, cls.fail_on_call = [], fail_on_call

    def parse(self, model, reasoning, instructions, input, text_format):
        import json

        with self.lock:
            FakeJudgeOpenAI.inputs.append(input)
            n = len(FakeJudgeOpenAI.inputs)
        if n == self.fail_on_call:
            import httpx2
            import openai
            raise openai.APIConnectionError(request=httpx2.Request("POST", "https://api.openai.com/v1/responses"))
        paper = json.loads(input)["paper"]
        title = paper["title"].lower()
        verdict = "relevant" if "citation" in title else "partly_relevant" if "rag" in title else "irrelevant"
        parsed = text_format(verdict=verdict, reason=f"Judged {verdict}.", quote=paper["abstract"][:40])
        return SimpleNamespace(output_parsed=parsed, usage=SimpleNamespace(input_tokens=100, output_tokens=50))


def rate_limit_error(code=None, retry_after=None):
    """An openai.RateLimitError shaped like the API's: `code` in the error body, optional Retry-After header."""
    import httpx2
    import openai

    headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
    response = httpx2.Response(429, headers=headers,
                               request=httpx2.Request("POST", "https://api.openai.com/v1/responses"))
    return openai.RateLimitError("Rate limited", response=response, body={"code": code, "message": "Rate limited"})


class FakeRerankOpenAI:
    """Replaces openai.OpenAI for rerank_papers: letter-count embeddings plus scripted screening decisions.

    Accepts papers whose title contains `keyword` (score 90 if it also says "quality", else 70), rejects the rest,
    and adds an unknown label and a conflicting duplicate. `errors` are raised by the next screening requests.
    Keyword extraction returns `keywords`, or the description's words when None; `keyword_errors` are raised first.
    """

    keyword = "translation"
    keywords = None
    errors = []
    keyword_errors = []
    inputs = []  # screening requests only
    keyword_inputs = []
    max_retries = []  # max_retries of every client created
    lock = threading.Lock()

    def __init__(self, api_key, max_retries=None):
        assert api_key == FAKE_KEY, "tests must never use the real key"
        FakeRerankOpenAI.max_retries.append(max_retries)
        self.embeddings = FakeEmbeddingsOpenAI(api_key).embeddings
        self.responses = SimpleNamespace(parse=self.parse)

    @classmethod
    def reset(cls, keyword="translation", errors=(), keywords=None, keyword_errors=()):
        cls.keyword, cls.errors, cls.inputs, cls.max_retries = keyword, list(errors), [], []
        cls.keywords, cls.keyword_errors, cls.keyword_inputs = keywords, list(keyword_errors), []
        FakeEmbeddingsOpenAI.inputs = []

    def parse(self, model, reasoning, instructions, input, text_format):
        if "terms" in text_format.model_fields:  # keyword extraction
            import json

            FakeRerankOpenAI.keyword_inputs.append(input)
            if FakeRerankOpenAI.keyword_errors:
                raise FakeRerankOpenAI.keyword_errors.pop(0)
            terms = self.keywords if self.keywords is not None else json.loads(input)["research_description"].split()
            return SimpleNamespace(output_parsed=text_format(terms=terms),
                                   usage=SimpleNamespace(input_tokens=50, output_tokens=10))
        with self.lock:
            FakeRerankOpenAI.inputs.append(input)
            error = FakeRerankOpenAI.errors.pop(0) if FakeRerankOpenAI.errors else None
        if error:
            raise error
        labels = re.findall(r"^Paper: (\S+)$", input, re.M)
        titles = re.findall(r"^Title: (.*)$", input, re.M)
        papers = []
        for lab, t in zip(labels, titles):
            accept = self.keyword in t.lower()
            papers.append({"label": lab, "decision": "accept" if accept else "reject",
                           "score": (90 if "quality" in t.lower() else 70) if accept else 10,
                           "reason": f"Mentions {self.keyword}." if accept else "Unrelated."})
        papers.append({"label": "P99", "decision": "accept", "score": 99, "reason": "Not in the batch."})
        if labels:
            papers.append({"label": labels[0], "decision": "accept", "score": 100, "reason": "Conflicting duplicate."})
        usage = SimpleNamespace(input_tokens=len(input) // 4, output_tokens=20 * len(labels))
        return SimpleNamespace(output_parsed=text_format(papers=papers), usage=usage)


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
        labels = re.findall(r"^Paper: (\S+)$", input, re.M)
        titles = re.findall(r"^Title: (.*)$", input, re.M)
        picks = [{"label": lab, "reason": f"Mentions {self.keyword}."}
                 for lab, t in zip(labels, titles) if self.keyword in t.lower()]
        picks.append({"label": "P99", "reason": "A label that was not in the batch."})
        return SimpleNamespace(output_parsed=text_format(matches=picks))
